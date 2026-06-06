"""The GrafastExecutionContext — a thin delegating shim over the function-seam core.

A drop-in for graphql-core's ``ExecutionContext`` for hosts that thread
``execution_context_class=GrafastExecutionContext`` into ``graphql()``/``execute()``
(or their Ariadne/FastAPI app). It carries no engine logic of its own: ``execute_operation``
delegates straight into :func:`grafast_py.entry.run_planned_operation`, the same pipeline
:func:`grafast_py.grafast_execute` runs, so the two share one core.

The shim still SUBCLASSES graphql-core's ``ExecutionContext`` so the host's call sequence
(``build()`` → ``execute_operation`` → ``build_response``, and the subscription per-event
path) finds the methods it expects. But it no longer DEPENDS on the 3.2-vs-3.3 shape of
that base: the version differences are confined to the seams in :mod:`grafast_py._compat`,
and ``execute_operation`` itself is dual-signature —

* 3.2: ``execute_operation(operation, root_value)`` returns the *data dict* (graphql-core's
  ``execute()`` then calls ``build_response`` itself).
* 3.3: ``execute_operation()`` is nullary and must return a *built* ``ExecutionResult``;
  it also re-initialises ``self.errors = []`` because 3.3's ``ExecutionContext.__init__``
  (and ``build_per_event_execution_context``) set ``self.errors = None`` while the pipeline
  does ``context.errors.append(...)``.

The engine is FULLY NATIVE: every GraphQL output-type kind is handled by a completer
(leaf, object, abstract interface/union, list, arbitrarily nested non-null), queries and
subscription events run their root fields in parallel, mutations run them serially, and
both sync and async resolvers are supported with native non-null null-bubbling and correct
error ``path`` accumulation. Every field carries a step in the operation plan — a plan
step, or a resolver-adapter (``ResolveStep``) for a plain-resolver field — so
plain-resolver schemas, and the graphql-core conformance suite, run unchanged through the
one unified path.

Production hardening (execution timeout, bounded concurrency, structured logging, tracing
hooks, configurable pg pool) is OPT-IN via a :class:`GrafastConfig` attached to the context
class; the defaults reproduce the engine's pre-hardening behaviour exactly. See
:mod:`grafast_py.config`. (Query cost/depth limiting is a validation-layer concern — use
your server's validation rules, not this executor.)
"""

from typing import Any, Optional

from graphql.error import GraphQLError
from graphql.execution.execute import ExecutionContext
from graphql.pyutils import AwaitableOrValue

from . import _compat
from .config import DEFAULT_CONFIG, GrafastConfig
from .entry import run_planned_operation

# sentinel so the dual-signature execute_operation can tell "called with the 3.2
# (operation, root_value) args" from "called nullary on 3.3".
_UNSET: Any = object()


class GrafastExecutionContext(ExecutionContext):
    """Plan-then-execute GraphQL execution context (unconditional drop-in)."""

    # opt-in hardening config; the default reproduces pre-hardening behaviour. A host
    # overrides it by subclassing or assigning
    # ``GrafastExecutionContext.grafast_config = GrafastConfig(...)``. The pipeline reads
    # it as ``context.grafast_config`` — which resolves this class attribute identically
    # to the old ``type(context).grafast_config`` (hosts set the CLASS attr, never an
    # instance attr), so the read is behaviour-preserving.
    grafast_config: GrafastConfig = DEFAULT_CONFIG

    def execute_operation(
        self, operation: Any = _UNSET, root_value: Any = _UNSET
    ) -> Optional[AwaitableOrValue[Any]]:
        """Plan the operation, then execute the plan over buckets.

        Dual-signature so the SAME method binds both host call shapes:

        * On 3.2 graphql-core calls ``execute_operation(operation, root_value)`` and then
          wraps the returned *data dict* via ``build_response`` itself — so we return the
          dict (or an awaitable of it), exactly as before.
        * On 3.3 graphql-core calls ``execute_operation()`` nullary and expects a *built*
          ``ExecutionResult`` back; it also leaves ``self.errors`` as ``None`` (both in
          ``__init__`` and in the subscription per-event ``build_per_event_execution_context``),
          so we re-initialise ``self.errors = []`` before the pipeline appends to it.

        Queries and subscription events run the root fields in parallel; mutations run
        them serially. The execution timeout and bounded concurrency apply to the async
        path.
        """
        if _compat.IS_32:
            # 3.2: args are passed positionally; return the data dict for build_response.
            return run_planned_operation(self, operation, root_value)

        # 3.3: nullary. 3.3's ExecutionContext.__init__ (and the per-event context copy in
        # build_per_event_execution_context) set self.errors = None, but the pipeline does
        # context.errors.append(...); re-init to a list so the append never hits None. This
        # one reinit also covers the subscription per-event path, which routes through this
        # same nullary execute_operation on the copied context.
        self.errors = []
        try:
            data = run_planned_operation(self, self.operation, self.root_value)
        except GraphQLError as error:
            self.errors.append(error)
            return _compat.make_result(None, self.errors)

        if self.is_awaitable(data):

            async def await_result():
                try:
                    awaited = await data
                except GraphQLError as error:
                    self.errors.append(error)
                    return _compat.make_result(None, self.errors)
                return _compat.make_result(awaited, self.errors)

            return await_result()

        return _compat.make_result(data, self.errors)

    def collect_subfields(self, return_type, field_nodes):
        """Collect subfields as ``{response_name: [FieldNode]}`` (version-stable seam).

        Overrides the inherited method so the planner gets the node-map shape on BOTH
        versions: 3.3's inherited ``collect_subfields`` returns a ``CollectedFields`` of
        ``FieldDetails`` and expects ``FieldDetails`` input, which the planner does not have.
        """
        return _compat.collect_subfields(self, return_type, field_nodes)

    def ensure_valid_runtime_type(
        self, runtime_type_name, return_type, field_nodes, info, result
    ):
        """Validate an abstract type's resolved runtime type (version-stable seam).

        Overrides the inherited method because 3.3's expects ``FieldDetails`` (it calls
        ``to_nodes`` on them) while the planner only ever has raw ``FieldNode``s.
        """
        return _compat.ensure_valid_runtime_type(
            self, runtime_type_name, return_type, field_nodes, info, result
        )
