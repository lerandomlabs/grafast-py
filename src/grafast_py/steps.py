"""The resolver-adapter step and its per-bucket-invocation extra.

A *step* is executed once over a whole *bucket* (a batch of parent objects),
producing one output value per parent. This is the core Grafast idea: a field
resolver runs as a single batched pass over every parent in the current layer
rather than being re-entered per (field, parent) pair by a tree walk.

This module ships the legacy-resolver adapter `ResolveStep`. It wraps an ordinary
field resolver and, per the Step contract, runs once over the bucket — but
internally it still loops `resolve_fn(parent, info, **args)` once per parent, which
is correct (if un-batched) for arbitrary user resolvers and is what keeps the
graphql-core conformance suite green. Genuine batch steps (loadOne/loadMany) that
issue ONE call for the whole bucket arrive in the build-stage steps; the batching
payoff is theirs, not this adapter's.

Every field now carries a `FieldPlan.step` — a plan step or a `ResolveStep` — so
the executor reads each field's value column uniformly from the bucket store; there
is no longer a separate per-parent mini-DAG (`ParentStep -> ResolveStep`) built at
completion time. The `ResolveStep` lives in `plan.steps`, built once at plan time
depending on the bucket `parent_step`, and is reused across every bucket of its
layer and across requests under the plan cache. It therefore MUST hold only
schema-stable objects and take the per-invocation request context + per-parent
paths as an argument, never as stored state — see `BucketExtra`.

Deliberate divergence from upstream: upstream's `graphqlResolver.ts` threads
`variableValues`/`rootValue` as unary DEPENDENCIES and degrades `info.path`
(`prev: undefined`). Our per-parent paths are NOT unary, so we thread a
per-invocation `BucketExtra` argument instead, and we build the FULL `info.path`
chain per parent because the graphql-core conformance oracle asserts the complete
path.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

from graphql.pyutils import Path

from .plan import FieldPlan
from .step_model import Step


@dataclass(frozen=True)
class BucketExtra:
    """The per-bucket-invocation extra threaded into a `wants_extra` step's `execute`.

    Holds the request `context` (for field_resolver / middleware / build_resolve_info /
    is_awaitable) and the per-parent `parent_paths` (one Path per bucket position, used to
    build each resolver `info.path`). It is a FRESH argument per `run_steps` call — never
    stored on the shared step — so it is concurrency-safe under the async await loop and
    survives the plan cache deepcopy untouched (it is never part of the cached plan).
    """

    context: Any
    parent_paths: List[Any]


class ResolveStep(Step):
    """Legacy-resolver adapter: runs a field's resolver once per parent in a bucket.

    Depends on a single bucket `parent_step` (dep 0), so `values[0]` is the parent
    column. `execute` builds the effective resolver exactly as graphql-core's
    execute_field does (field resolver or context default, wrapped through
    middleware), then invokes it once per parent, returning the raw per-parent value
    column. Argument coercion that failed at plan time is replayed as a per-parent
    raised value located at the field; a resolver that itself raises is carried as a
    value (exceptions-as-values) so the completer locates it per parent. An async
    resolver leaves a coroutine in the column, which `complete_values`/`is_awaitable`
    detect and await — exactly as the plan-field path already does.

    The step holds ONLY schema-stable objects (it is built once at plan time and
    reused across buckets/requests, and is deep-copied by the plan cache): the
    request context and per-parent paths arrive per invocation via `BucketExtra`
    (`wants_extra`). Per-parent resolve `info` and field `path` are rebuilt by the
    completer from the same paths, so nothing resolver-specific lives in completion.

    Looping over parents internally is acceptable for arbitrary user resolvers; the
    batching payoff comes from the batch steps (loadOne/loadMany), not this adapter.
    """

    # a resolver is arbitrary/possibly side-effecting, so it must NEVER merge with
    # another field's resolver step.
    dedupable = False
    # it needs the per-invocation request context + per-parent paths via BucketExtra.
    wants_extra = True

    def __init__(self, field_def, parent_type, field_nodes, response_name, args, args_error):
        super().__init__()
        self.field_def = field_def
        self.parent_type = parent_type
        self.field_nodes = field_nodes
        self.response_name = response_name
        self.args = args
        self.args_error = args_error

    @property
    def peer_key(self) -> str:
        # a resolver step is unique; never fold it into a peer.
        return f"resolve|{id(self)}"

    def dedup_params(self) -> tuple:
        return (id(self),)

    def execute(self, count, values, extra):
        context = extra.context
        parents = values[0]

        resolve_fn = self.field_def.resolve or context.field_resolver
        if context.middleware_manager:
            resolve_fn = context.middleware_manager.get_field_resolver(resolve_fn)

        args: Dict[str, Any] = self.args or {}
        out: List[Any] = []

        for parent, parent_path in zip(parents, extra.parent_paths):
            if self.args_error is not None:
                out.append(self.args_error)
                continue
            field_path = Path(parent_path, self.response_name, self.parent_type.name)
            info = context.build_resolve_info(
                self.field_def, self.field_nodes, self.parent_type, field_path
            )
            try:
                value = resolve_fn(parent, info, **args)
            except Exception as raw_error:  # resolver raised → carry as a value
                out.append(raw_error)
                continue
            out.append(value)
        return out
