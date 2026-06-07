"""The GrafastExecutionContext.

A genuine plan-then-execute GraphQL execution context and an unconditional drop-in
for graphql-core's ``ExecutionContext``. ``execute_operation`` collects the root
fields, builds a static plan (an OutputPlan tree + a step DAG), then executes it by
running each step ONCE over a *batch* (a bucket) — a `loadMany` over N parents sees
every key in a single call — rather than re-entering a tree walk per (field, parent)
pair.

The engine is FULLY NATIVE: every GraphQL output-type kind is handled by a
completer (leaf, object, abstract interface/union, list, arbitrarily nested
non-null), queries and subscription events run their root fields in parallel,
mutations run them serially, and both sync and async resolvers are supported with
native non-null null-bubbling and correct error ``path`` accumulation. Every field
carries a step in the operation plan — a plan step, or a resolver-adapter
(``ResolveStep``) for a plain-resolver field — so plain-resolver schemas, and the
graphql-core conformance suite, run unchanged through the one unified path.

Production hardening (execution timeout, bounded concurrency, structured logging,
tracing hooks, configurable pg pool) is OPT-IN via a :class:`GrafastConfig` attached
to the context class; with the defaults (every hardening knob off) the engine behaves
as a plain unhardened executor. See :mod:`grafast_py.config`. (Query cost/depth limiting
is a validation-layer concern — use your server's validation rules, not this executor.)
"""

import asyncio
from typing import Any, Optional

from graphql.error import GraphQLError, located_error
from graphql.execution.collect_fields import collect_fields
from graphql.execution.execute import ExecutionContext
from graphql.language import OperationDefinitionNode, OperationType
from graphql.pyutils import AwaitableOrValue

from .bubble import Bubble
from .config import DEFAULT_CONFIG, GrafastConfig, GrafastTimeoutError, log
from .execute import execute_object_plan, execute_object_plan_serially
from .plan import plan_operation


class GrafastExecutionContext(ExecutionContext):
    """Plan-then-execute GraphQL execution context (unconditional drop-in)."""

    # opt-in hardening config; the default leaves every hardening knob off. A
    # host overrides it by subclassing or assigning
    # ``GrafastExecutionContext.grafast_config = GrafastConfig(...)``. Read per
    # operation via ``type(self).grafast_config``.
    grafast_config: GrafastConfig = DEFAULT_CONFIG

    def execute_operation(
        self, operation: OperationDefinitionNode, root_value: Any
    ) -> Optional[AwaitableOrValue[Any]]:
        """Plan the operation, then execute the plan over buckets.

        Queries and subscription events run the root fields in parallel; mutations
        run them serially (each field's resolver and completion finish before the
        next field starts). The execution timeout and bounded concurrency apply to
        the async path.
        """
        config = type(self).grafast_config
        # expose the opt-in step-batch tracing hook + concurrency gate to the
        # executor without changing the stock context constructor signature.
        self._grafast_on_step_batch = config.on_step_batch
        self._grafast_concurrency = (
            asyncio.Semaphore(config.max_step_concurrency)
            if config.max_step_concurrency is not None
            else None
        )

        root_type = self.schema.get_root_type(operation.operation)
        if root_type is None:
            raise GraphQLError(
                "Schema is not configured to execute"
                f" {operation.operation.value} operation.",
                operation,
            )

        root_fields = collect_fields(
            self.schema,
            self.fragments,
            self.variable_values,
            root_type,
            operation.selection_set,
        )

        op_span = _enter_span(config.on_operation, self, operation)
        plan_span = _enter_span(config.on_plan, self, operation)
        plan = plan_operation(self, operation, root_type, root_fields)
        _exit_span(plan_span)

        op_name = operation.name.value if operation.name else None
        log.debug(
            "planned operation",
            op=op_name,
            root=root_type.name,
            steps=len(self._grafast_plan.steps),
        )

        # the root is a single "bucket" of one parent: the root value. mutations
        # take the serial path; queries / subscription events the parallel one.
        if operation.operation == OperationType.MUTATION:
            results = execute_object_plan_serially(self, plan, [root_value], [None])
        else:
            results = execute_object_plan(self, plan, [root_value], [None])

        if self.is_awaitable(results):

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

    Only the async path can be interrupted — there is no event loop on the sync path
    to enforce a timeout, so a sync operation runs unbounded (documented).
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

    A top-level non-null field that nulled the root surfaces as a `Bubble` (its
    located error has not been appended yet). Raising that error out of
    `execute_operation` lets the stock `execute()` append it once and build a
    `{"data": None}` response, matching the spec.
    """
    if isinstance(result, Bubble):
        raise result.error
    return result
