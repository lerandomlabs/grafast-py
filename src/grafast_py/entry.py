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


def grafast_subscribe(
    schema: GraphQLSchema,
    document: Union[str, Source, DocumentNode],
    root_value: Any = None,
    context_value: Any = None,
    variable_values: Optional[Dict[str, Any]] = None,
    operation_name: Optional[str] = None,
    field_resolver: Optional[Callable] = None,
    subscribe_field_resolver: Optional[Callable] = None,
    config: GrafastConfig = DEFAULT_CONFIG,
):
    """Subscribe: build the source event stream, then plan-then-execute each event.

    Uses graphql-core's version-stable ``create_source_event_stream`` to produce the source
    stream, then maps each payload through :func:`grafast_execute` (root_value=payload). The
    map-async-iterable wrapper moved between versions (3.2 ``MapAsyncIterator`` class, 3.3
    ``map_async_iterable`` function), so this probes for the right one. Non-incremental
    delivery only — full @defer/@stream subscription delivery is a later phase (P7).

    Returns an async iterator of ``ExecutionResult`` (or a single ``ExecutionResult`` when
    the source-stream creation itself errored), as an awaitable.
    """
    if isinstance(document, DocumentNode):
        parsed = document
    else:
        try:
            parsed = parse(document)
        except GraphQLError as error:
            return _error_aiter(ExecutionResult(data=None, errors=[error]))

    async def subscribe_impl():
        # create_source_event_stream is an ``async def`` on 3.2 (always awaitable) but a
        # maybe-awaitable plain function on 3.3 (it returns the stream directly when the
        # subscribe resolver is sync); await only when needed.
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
            result_or_stream = await result_or_stream
        if isinstance(result_or_stream, ExecutionResult):
            return result_or_stream

        async def map_payload(payload: Any) -> ExecutionResult:
            mapped = grafast_execute(
                schema,
                parsed,
                root_value=payload,
                context_value=context_value,
                variable_values=variable_values,
                operation_name=operation_name,
                field_resolver=field_resolver,
                config=config,
            )
            if default_is_awaitable(mapped):
                return await mapped
            return mapped

        return _map_async_iterable(result_or_stream, map_payload)

    return subscribe_impl()


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
