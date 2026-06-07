"""Top-level grafast execution over a lightweight, ExecutionContext-free run context.

``grafast_execute`` / ``grafast_subscribe`` are the function-seam entry points: they own
the GraphQL frontend (validate schema, parse, validate, coerce variables, select the
operation) using graphql-core's version-stable helpers, then run the engine's
plan-then-execute pipeline over a :class:`_PlanRunContext` — a minimal object that
supplies exactly the attributes/methods the pipeline reaches into, WITHOUT subclassing
graphql-core's ``ExecutionContext`` (whose shape differs between 3.2 and 3.3). The result
is packaged via the OWNED, version-independent :func:`grafast_py._compat.make_result`, so
the returned value is a plain :class:`graphql.ExecutionResult` on every version.

The same pipeline body lives in :func:`run_planned_operation`; the thin
``GrafastExecutionContext`` shim (see :mod:`grafast_py.context`) calls it too, so both the
host pattern (``execution_context_class=GrafastExecutionContext``) and the function entry
share one core.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Union

from graphql.error import GraphQLError, located_error
from graphql.execution import ExecutionResult, create_source_event_stream
from graphql.execution.middleware import MiddlewareManager
from graphql.execution.values import get_variable_values
from graphql.language import (
    DocumentNode,
    FragmentDefinitionNode,
    OperationDefinitionNode,
    OperationType,
    Source,
    parse,
)
from graphql.pyutils import AwaitableOrValue, is_awaitable as default_is_awaitable
from graphql.type import GraphQLSchema, validate_schema
from graphql.validation import validate

from . import _compat
from .bubble import Bubble
from .config import DEFAULT_CONFIG, GrafastConfig, GrafastTimeoutError, log
from .execute import execute_object_plan, execute_object_plan_serially
from .plan import plan_operation


class _PlanRunContext:
    """A minimal request context the plan-then-execute pipeline runs against.

    Deliberately NOT a subclass of graphql-core's ``ExecutionContext`` (whose method
    signatures and ``errors`` initialisation differ between 3.2 and 3.3). It supplies the
    exact surface the pipeline reads — the data attributes (``schema``, ``fragments``,
    ``variable_values``, ``root_value``, ``context_value``, ``operation``,
    ``field_resolver``, ``type_resolver``, ``middleware_manager``, ``errors``,
    ``is_awaitable``), the per-call ``grafast_config``, and the three inherited-method
    equivalents (``build_resolve_info``, ``collect_subfields``,
    ``ensure_valid_runtime_type``) — and otherwise allows the planner / seams to stash their
    ``_grafast_*`` attributes freely (plain instance, no ``__slots__``; the
    subfield-collection cache lives under one such attribute, lazily created by the seam).
    """

    def __init__(
        self,
        schema: GraphQLSchema,
        fragments: Dict[str, FragmentDefinitionNode],
        root_value: Any,
        context_value: Any,
        operation: OperationDefinitionNode,
        variable_values: Dict[str, Any],
        field_resolver: Callable,
        type_resolver: Callable,
        errors: Optional[List[GraphQLError]],
        middleware_manager: Optional[MiddlewareManager],
        is_awaitable: Optional[Callable[[Any], bool]] = None,
        grafast_config: GrafastConfig = DEFAULT_CONFIG,
    ) -> None:
        self.schema = schema
        self.fragments = fragments
        self.root_value = root_value
        self.context_value = context_value
        self.operation = operation
        self.variable_values = variable_values
        self.field_resolver = field_resolver
        self.type_resolver = type_resolver
        self.errors = errors if errors is not None else []
        self.middleware_manager = middleware_manager
        self.is_awaitable = is_awaitable or default_is_awaitable
        self.grafast_config = grafast_config

    def build_resolve_info(self, field_def, field_nodes, parent_type, path):
        """Build a ``GraphQLResolveInfo`` (delegates to the version-stable seam)."""
        return _compat.build_resolve_info(
            self, field_def, field_nodes, parent_type, path
        )

    def collect_subfields(self, object_type, field_nodes):
        """Collect subfields as ``{response_name: [FieldNode]}`` (version-stable seam)."""
        return _compat.collect_subfields(self, object_type, field_nodes)

    def ensure_valid_runtime_type(
        self, runtime_type_name, return_type, field_nodes, info, result
    ):
        """Validate an abstract type's resolved runtime type (version-stable seam)."""
        return _compat.ensure_valid_runtime_type(
            self, runtime_type_name, return_type, field_nodes, info, result
        )


def run_planned_operation(
    context, operation: OperationDefinitionNode, root_value: Any
) -> AwaitableOrValue[Any]:
    """Plan ``operation`` and execute the plan over buckets, returning the root data dict.

    This is the ONE pipeline entry shared by :func:`grafast_execute` and the
    ``GrafastExecutionContext`` shim. Queries and subscription events run their root
    fields in parallel; mutations run them serially. The execution timeout and bounded
    concurrency apply to the async path. Returns the root output dict (or an awaitable of
    it); raises ``GraphQLError`` via :func:`root_output` when a top-level non-null field
    nulled the whole root.

    ``context`` is any object exposing the pipeline surface — a :class:`_PlanRunContext`
    or a ``GrafastExecutionContext`` shim — including a ``grafast_config`` attribute.
    """
    config = context.grafast_config
    # expose the opt-in step-batch tracing hook + concurrency gate to the executor without
    # changing the stock context constructor signature.
    context._grafast_on_step_batch = config.on_step_batch
    context._grafast_concurrency = (
        asyncio.Semaphore(config.max_step_concurrency)
        if config.max_step_concurrency is not None
        else None
    )

    root_type = context.schema.get_root_type(operation.operation)
    if root_type is None:
        raise GraphQLError(
            "Schema is not configured to execute"
            f" {operation.operation.value} operation.",
            operation,
        )

    root_fields = _compat.collect_root_fields(context, root_type, operation)

    op_span = _enter_span(config.on_operation, context, operation)
    plan_span = _enter_span(config.on_plan, context, operation)
    plan = plan_operation(context, operation, root_type, root_fields)
    _exit_span(plan_span)

    op_name = operation.name.value if operation.name else None
    log.debug(
        "planned operation",
        op=op_name,
        root=root_type.name,
        steps=len(context._grafast_plan.steps),
    )

    # the root is a single "bucket" of one parent: the root value. mutations take the
    # serial path; queries / subscription events the parallel one.
    if operation.operation == OperationType.MUTATION:
        results = execute_object_plan_serially(context, plan, [root_value], [None])
    else:
        results = execute_object_plan(context, plan, [root_value], [None])

    if context.is_awaitable(results):

        async def await_root():
            try:
                awaited = await _with_timeout(
                    results, config.execution_timeout_s, op_name
                )
                return root_output(awaited[0])
            finally:
                _exit_span(op_span)

        return await_root()

    _exit_span(op_span)
    return root_output(results[0])


def _enter_span(hook, *args) -> Optional[Any]:
    """Call a tracing hook and enter its span (no-op when the hook returns None)."""
    span = hook(*args)
    if span is None:
        return None
    span.__enter__()
    return span


def _exit_span(span: Optional[Any]) -> None:
    """Exit a previously entered span, if any."""
    if span is not None:
        span.__exit__(None, None, None)


async def _with_timeout(awaitable, timeout_s: Optional[float], op_name):
    """Await with an optional wall-clock budget; raise GrafastTimeoutError on overrun.

    Only the async path can be interrupted — there is no event loop on the sync path to
    enforce a timeout, so a sync operation runs unbounded (documented).
    """
    if timeout_s is None:
        return await awaitable
    try:
        return await asyncio.wait_for(asyncio.ensure_future(awaitable), timeout_s)
    except asyncio.TimeoutError as raw_error:
        log.error("operation timed out", limit_s=timeout_s, op=op_name)
        raise located_error(
            GrafastTimeoutError(f"operation exceeded {timeout_s}s execution timeout"),
            [],
            [],
        ) from raw_error


def root_output(result):
    """Surface the root output dict, or raise if the whole root was nulled.

    A top-level non-null field that nulled the root surfaces as a `Bubble` (its located
    error has not been appended yet). Raising that error lets the caller append it once and
    build a `{"data": None}` response, matching the spec.
    """
    if isinstance(result, Bubble):
        raise result.error
    return result


def _select_operation(
    document: DocumentNode, operation_name: Optional[str]
) -> Union[OperationDefinitionNode, List[GraphQLError]]:
    """Pick the operation to execute and gather fragments — mirrors ExecutionContext.build.

    Returns the selected ``OperationDefinitionNode``, or a list of ``GraphQLError`` when no
    valid single operation can be chosen (multiple operations without a name, unknown name,
    or none present).
    """
    operation: Optional[OperationDefinitionNode] = None
    for definition in document.definitions:
        if isinstance(definition, OperationDefinitionNode):
            if operation_name is None:
                if operation:
                    return [
                        GraphQLError(
                            "Must provide operation name"
                            " if query contains multiple operations."
                        )
                    ]
                operation = definition
            elif definition.name and definition.name.value == operation_name:
                operation = definition
    if operation is None:
        if operation_name is not None:
            return [GraphQLError(f"Unknown operation named '{operation_name}'.")]
        return [GraphQLError("Must provide an operation.")]
    return operation


def _gather_fragments(document: DocumentNode) -> Dict[str, FragmentDefinitionNode]:
    """Collect the document's fragment definitions by name."""
    return {
        definition.name.value: definition
        for definition in document.definitions
        if isinstance(definition, FragmentDefinitionNode)
    }


def _prepare_request(
    schema,
    document,
    root_value,
    context_value,
    variable_values,
    operation_name,
    field_resolver,
    type_resolver,
    middleware,
    is_awaitable,
    config,
    validate_document=True,
):
    """Run the GraphQL frontend, returning a ready ``(_PlanRunContext, operation)`` or an
    early ``ExecutionResult`` of errors.

    The shared front-end for :func:`grafast_execute` and
    :func:`experimental_execute_incrementally`: schema validation, parse, validation, operation
    selection, variable coercion, middleware-manager construction, and context build — all via
    graphql-core's version-stable helpers. Returning the early ``ExecutionResult`` (instead of
    raising) matches graphql-core's "response with only errors" for an invalid request.

    ``validate_document`` mirrors the graphql() vs execute() split: :func:`grafast_execute` is
    the graphql()-equivalent full pipeline and validates (True), while
    :func:`experimental_execute_incrementally` is the execute()-level entry the conformance suite
    drives with already-validated (sometimes intentionally schema-invalid-but-executable)
    documents, so it skips validation (False) — matching graphql-core's own
    ``experimental_execute_incrementally``, which never validates.
    """
    from graphql.execution.execute import default_field_resolver, default_type_resolver

    schema_validation_errors = validate_schema(schema)
    if schema_validation_errors:
        return ExecutionResult(data=None, errors=schema_validation_errors)

    if isinstance(document, DocumentNode):
        parsed = document
    else:
        try:
            parsed = parse(document)
        except GraphQLError as error:
            return ExecutionResult(data=None, errors=[error])

    # validate BOTH a freshly-parsed document AND a caller-supplied DocumentNode: grafast_execute
    # is the graphql()-equivalent full pipeline, so it must report the same validation errors
    # whether or not the caller pre-parsed (matching upstream grafast() / graphql-core's
    # graphql()). A pre-parsed INVALID document previously slipped through here — e.g. an unknown
    # field was silently dropped by the planner instead of surfacing a validation error.
    # The execute()-level experimental entry opts out (validate_document=False): like graphql-core's
    # own experimental_execute_incrementally it assumes a pre-validated document and must EXECUTE
    # the intentionally-invalid-but-executable docs the conformance suite passes it.
    if validate_document:
        validation_errors = validate(schema, parsed)
        if validation_errors:
            return ExecutionResult(data=None, errors=validation_errors)

    selected = _select_operation(parsed, operation_name)
    if isinstance(selected, list):
        return ExecutionResult(data=None, errors=selected)
    operation = selected

    coerced_variable_values = get_variable_values(
        schema,
        operation.variable_definitions or (),
        variable_values or {},
        max_errors=50,
    )
    if isinstance(coerced_variable_values, list):
        return ExecutionResult(data=None, errors=coerced_variable_values)

    middleware_manager: Optional[MiddlewareManager] = None
    if middleware is not None:
        if isinstance(middleware, (list, tuple)):
            middleware_manager = MiddlewareManager(*middleware)
        elif isinstance(middleware, MiddlewareManager):
            middleware_manager = middleware
        else:
            raise TypeError(
                "Middleware must be passed as a list or tuple of functions"
                " or objects, or as a single MiddlewareManager object."
                f" Got {middleware!r} instead."
            )

    context = _PlanRunContext(
        schema=schema,
        fragments=_gather_fragments(parsed),
        root_value=root_value,
        context_value=context_value,
        operation=operation,
        variable_values=coerced_variable_values,
        field_resolver=field_resolver or default_field_resolver,
        type_resolver=type_resolver or default_type_resolver,
        errors=[],
        middleware_manager=middleware_manager,
        is_awaitable=is_awaitable,
        grafast_config=config,
    )
    return context, operation


def grafast_execute(
    schema: GraphQLSchema,
    document: Union[str, Source, DocumentNode],
    root_value: Any = None,
    context_value: Any = None,
    variable_values: Optional[Dict[str, Any]] = None,
    operation_name: Optional[str] = None,
    field_resolver: Optional[Callable] = None,
    type_resolver: Optional[Callable] = None,
    middleware=None,
    is_awaitable: Optional[Callable[[Any], bool]] = None,
    config: GrafastConfig = DEFAULT_CONFIG,
) -> AwaitableOrValue[ExecutionResult]:
    """Plan-then-execute a GraphQL operation, returning a plain ``ExecutionResult``.

    The function-seam entry point: owns the GraphQL frontend (schema validation, parse,
    validation, variable coercion, operation selection) with graphql-core's version-stable
    helpers, runs the engine pipeline over a :class:`_PlanRunContext`, and packages the
    result via the OWNED :func:`grafast_py._compat.make_result` (deterministic error sort
    on every version). Returns an awaitable of the result only when execution is async.

    ``document`` may be a query string, a ``Source``, or an already-parsed ``DocumentNode``
    (mirroring graphql-core's ``graphql_impl``/``execute`` accepting both).
    """
    front = _prepare_request(
        schema,
        document,
        root_value,
        context_value,
        variable_values,
        operation_name,
        field_resolver,
        type_resolver,
        middleware,
        is_awaitable,
        config,
    )
    if isinstance(front, ExecutionResult):
        return front
    context, operation = front

    try:
        result = run_planned_operation(context, operation, root_value)
    except GraphQLError as error:
        context.errors.append(error)
        return _compat.make_result(None, context.errors)

    if context.is_awaitable(result):

        async def await_result() -> ExecutionResult:
            try:
                data = await result
            except GraphQLError as error:
                context.errors.append(error)
                return _compat.make_result(None, context.errors)
            return _compat.make_result(data, context.errors)

        return await_result()

    return _compat.make_result(result, context.errors)


UNEXPECTED_MULTIPLE_PAYLOADS = (
    "Executing this GraphQL operation would unexpectedly produce multiple payloads"
    " (due to @defer or @stream directive)"
)

UNEXPECTED_EXPERIMENTAL_DIRECTIVES = (
    "The provided schema unexpectedly contains experimental directives"
    " (@defer or @stream). These directives may only be utilized"
    " if experimental execution features are explicitly enabled."
)


def grafast_execute_plain(
    schema: GraphQLSchema,
    document: Union[str, Source, DocumentNode],
    root_value: Any = None,
    context_value: Any = None,
    variable_values: Optional[Dict[str, Any]] = None,
    operation_name: Optional[str] = None,
    field_resolver: Optional[Callable] = None,
    type_resolver: Optional[Callable] = None,
    subscribe_field_resolver: Optional[Callable] = None,
    enable_early_execution: bool = False,
    middleware=None,
    execution_context_class=None,
    is_awaitable: Optional[Callable[[Any], bool]] = None,
    is_async_iterable=None,
    config: GrafastConfig = DEFAULT_CONFIG,
    **custom_context_args,
):
    """The plain ``execute`` entry: a single ``ExecutionResult``, RAISING on incremental work.

    Mirrors graphql-core 3.3's ``execute``: it delegates to
    :func:`experimental_execute_incrementally` and, if that would produce multiple payloads
    (an ``ExperimentalIncrementalExecutionResults``), raises the
    ``UNEXPECTED_MULTIPLE_PAYLOADS`` error instead — the behaviour the conformance
    ``original_execute_function_throws_error_if_deferred*`` cases assert. Used to patch the
    module-level ``execute`` on 3.3 so those tests route through grafast.
    """
    from graphql.execution import ExperimentalIncrementalExecutionResults

    # the plain execute() refuses a schema that even DECLARES @defer/@stream (the experimental
    # directives are only legal under experimental_execute_incrementally), matching upstream.
    if schema.get_directive("defer") or schema.get_directive("stream"):
        raise GraphQLError(UNEXPECTED_EXPERIMENTAL_DIRECTIVES)

    result = experimental_execute_incrementally(
        schema,
        document,
        root_value,
        context_value,
        variable_values,
        operation_name,
        field_resolver,
        type_resolver,
        subscribe_field_resolver,
        enable_early_execution,
        middleware,
        execution_context_class,
        is_awaitable,
        is_async_iterable,
        config=config,
    )
    if isinstance(result, ExecutionResult):
        return result
    if isinstance(result, ExperimentalIncrementalExecutionResults):
        raise GraphQLError(UNEXPECTED_MULTIPLE_PAYLOADS)

    async def await_result():
        awaited = await result
        if isinstance(awaited, ExecutionResult):
            return awaited
        raise GraphQLError(UNEXPECTED_MULTIPLE_PAYLOADS)

    return await_result()


def run_planned_operation_incremental(context, operation, root_value):
    """Plan + run the INITIAL incremental tree, returning (root_data, incremental_records).

    Like :func:`run_planned_operation` but plans with ``incremental=True`` (so @defer'd groups
    are partitioned out and @stream markers read), and installs defer/stream sinks so the
    executor captures the root-level deferred fragments' execution groups + stream records. The
    captured records are the initial-result children the publisher promotes to root nodes.
    Returns the root output dict (deferred keys omitted) plus the captured records; either may
    arrive via an awaitable when execution is async.

    ``enable_early_execution`` (threaded from the entry) is recorded on the context so the
    @stream / @defer machinery can run groups eagerly when requested.
    """
    from .incremental import make_defer_sink

    config = context.grafast_config
    context._grafast_on_step_batch = config.on_step_batch
    context._grafast_concurrency = (
        asyncio.Semaphore(config.max_step_concurrency)
        if config.max_step_concurrency is not None
        else None
    )
    # mark the request incremental so abstract concrete-type subtrees partition @defer too.
    context._grafast_incremental = True

    root_type = context.schema.get_root_type(operation.operation)
    if root_type is None:
        raise GraphQLError(
            "Schema is not configured to execute"
            f" {operation.operation.value} operation.",
            operation,
        )

    root_collected = _compat.collect_root_fields_raw(context, root_type, operation)
    plan = plan_operation(
        context, operation, root_type, root_collected, incremental=True
    )

    captured: List[Any] = []
    context._grafast_defer_map = None
    context._grafast_defer_sink = make_defer_sink(context, captured)
    context._grafast_stream_sink = lambda rec: captured.append(rec)

    if operation.operation == OperationType.MUTATION:
        results = execute_object_plan_serially(context, plan, [root_value], [None])
    else:
        results = execute_object_plan(context, plan, [root_value], [None])

    if context.is_awaitable(results):

        async def await_root():
            awaited = await _with_timeout(
                results, config.execution_timeout_s, None
            )
            return root_output(awaited[0]), captured

        return await_root()
    return root_output(results[0]), captured


def experimental_execute_incrementally(
    schema: GraphQLSchema,
    document: Union[str, Source, DocumentNode],
    root_value: Any = None,
    context_value: Any = None,
    variable_values: Optional[Dict[str, Any]] = None,
    operation_name: Optional[str] = None,
    field_resolver: Optional[Callable] = None,
    type_resolver: Optional[Callable] = None,
    subscribe_field_resolver: Optional[Callable] = None,
    enable_early_execution: bool = False,
    middleware=None,
    execution_context_class=None,
    is_awaitable: Optional[Callable[[Any], bool]] = None,
    is_async_iterable=None,
    config: GrafastConfig = DEFAULT_CONFIG,
    **custom_context_args,
):
    """Execute a GraphQL operation with @defer / @stream incremental delivery (3.3 only).

    Returns a plain ``ExecutionResult`` when the operation has NO incremental work (no @defer
    / @stream survives — including ``@defer(if:false)`` and no-directive), else an
    ``ExperimentalIncrementalExecutionResults`` (initial result + an async generator of
    subsequent payloads). The non-incremental entry is :func:`grafast_execute`; this one shares
    its frontend (validate / parse / coerce / select) and only differs in routing the planned
    run through the incremental driver.

    The signature mirrors graphql-core 3.3's ``experimental_execute_incrementally`` (incl.
    ``enable_early_execution``) so the conformance harness can call it positionally/by-keyword;
    ``enable_early_execution`` is accepted for parity but the driver runs deferred groups when
    their parent completes (the default-path grouping the test suite asserts).
    """
    from .incremental import run_incremental

    front = _prepare_request(
        schema,
        document,
        root_value,
        context_value,
        variable_values,
        operation_name,
        field_resolver,
        type_resolver,
        middleware,
        is_awaitable,
        config,
        # execute()-level entry: assume a pre-validated document (matching graphql-core's own
        # experimental_execute_incrementally), so the conformance suite's intentionally-invalid-
        # but-executable documents execute rather than short-circuiting to a validation error.
        validate_document=False,
    )
    if isinstance(front, ExecutionResult):
        return front
    context, operation = front
    # record the early-execution flag so the @defer/@stream machinery can run groups eagerly.
    context._grafast_enable_early_execution = enable_early_execution

    class _InitialRun:
        __slots__ = ("data", "errors")

    def package(root_data, incremental_records):
        if not incremental_records:
            # no incremental work survived (e.g. @defer(if:false), or every defer collapsed
            # into the initial payload): a plain ExecutionResult, byte-identical to grafast_execute.
            return _compat.make_result(root_data, context.errors)
        run = _InitialRun()
        run.data = root_data
        run.errors = list(context.errors)
        return run_incremental(context, run, incremental_records)

    try:
        produced = run_planned_operation_incremental(context, operation, root_value)
    except GraphQLError as error:
        context.errors.append(error)
        return _compat.make_result(None, context.errors)

    if context.is_awaitable(produced):

        async def await_incremental():
            try:
                root_data, incremental_records = await produced
            except GraphQLError as error:
                context.errors.append(error)
                return _compat.make_result(None, context.errors)
            return package(root_data, incremental_records)

        return await_incremental()

    root_data, incremental_records = produced
    return package(root_data, incremental_records)


def grafast_subscribe(
    schema: GraphQLSchema,
    document: Union[str, Source, DocumentNode],
    root_value: Any = None,
    context_value: Any = None,
    variable_values: Optional[Dict[str, Any]] = None,
    operation_name: Optional[str] = None,
    field_resolver: Optional[Callable] = None,
    type_resolver: Optional[Callable] = None,
    subscribe_field_resolver: Optional[Callable] = None,
    enable_early_execution: bool = False,
    middleware=None,
    execution_context_class=None,
    is_awaitable: Optional[Callable[[Any], bool]] = None,
    is_async_iterable=None,
    config: GrafastConfig = DEFAULT_CONFIG,
    **custom_context_args,
):
    """Subscribe: build the source event stream (graphql-core), execute each event.

    Delegates source-stream creation to graphql-core's version-stable
    ``create_source_event_stream``, then maps each payload through the per-event execution
    (root_value=payload). Per-event execution routes through the incremental entry so a
    @defer/@stream on a subscription field surfaces the upstream "not supported on subscription
    operations" error (raised by collect_fields / the stream completer) as a field error —
    matching the conformance ``subscribe_function_returns_errors_with_defer/stream`` cases —
    while ordinary events are plain ``ExecutionResult``s (upstream executes subscription events
    non-incrementally).

    MAYBE-AWAITABLE shape (3.3): when ``create_source_event_stream`` returns the stream
    synchronously, this returns the mapped ``AsyncIterator`` DIRECTLY (so ``isinstance(result,
    AsyncIterator)`` holds without an await); only when the source creation is async does it
    return a coroutine. The map-async-iterable wrapper moved between versions, so this probes.

    The signature mirrors graphql-core 3.3's ``subscribe`` so the install hook can swap it in.
    """
    # graphql-core's `subscribe` takes an already-parsed DocumentNode and passes it straight to
    # `create_source_event_stream` (a non-document errors naturally — `should_pass_through_
    # unexpected_errors_thrown_in_subscribe` asserts an AttributeError on a dict document). Only
    # parse a string/Source for convenience; pass anything else through untouched.
    if isinstance(document, (str, Source)):
        try:
            parsed = parse(document)
        except GraphQLError as error:
            return _error_aiter(ExecutionResult(data=None, errors=[error]))
    else:
        parsed = document

    async def map_payload(payload: Any):
        # per-event execution: incremental on 3.3 (so defer/stream-on-subscription errors
        # surface), plain on 3.2. A subscription event never legitimately produces multiple
        # payloads (defer/stream are disallowed), so the result is always an ExecutionResult.
        runner = (
            experimental_execute_incrementally
            if _compat.supports_incremental()
            else grafast_execute
        )
        mapped = runner(
            schema,
            parsed,
            root_value=payload,
            context_value=context_value,
            variable_values=variable_values,
            operation_name=operation_name,
            field_resolver=field_resolver,
            type_resolver=type_resolver,
            middleware=middleware,
            config=config,
        )
        if default_is_awaitable(mapped):
            return await mapped
        return mapped

    # create_source_event_stream is an ``async def`` on 3.2 (always awaitable) but a
    # maybe-awaitable plain function on 3.3 (returns the stream directly when the subscribe
    # resolver is sync); only wrap in a coroutine when the source creation is actually async.
    result_or_stream = create_source_event_stream(
        schema,
        parsed,
        root_value,
        context_value,
        variable_values,
        operation_name,
        subscribe_field_resolver,
    )

    if default_is_awaitable(result_or_stream):

        async def subscribe_async():
            stream = await result_or_stream
            if isinstance(stream, ExecutionResult):
                return stream
            return _map_async_iterable(stream, map_payload)

        return subscribe_async()

    if isinstance(result_or_stream, ExecutionResult):
        return result_or_stream
    return _map_async_iterable(result_or_stream, map_payload)


def _map_async_iterable(source, mapper):
    """Map an async iterable through ``mapper``, using the version-appropriate wrapper."""
    if _compat.IS_32:
        from graphql.execution.map_async_iterator import MapAsyncIterator

        return MapAsyncIterator(source, mapper)

    from graphql.execution.async_iterables import map_async_iterable

    return map_async_iterable(source, mapper)


async def _error_aiter(result: ExecutionResult):
    """Wrap a single ``ExecutionResult`` so a parse failure surfaces as an awaitable."""
    return result
