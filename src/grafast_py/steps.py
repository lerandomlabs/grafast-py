"""Step types for the plan-then-execute engine.

A *step* is executed once over a whole *bucket* (a batch of parent objects),
producing one output value per parent. This is the core Grafast idea: a field
resolver runs as a single batched pass over every parent in the current layer
rather than being re-entered per (field, parent) pair by a tree walk.

This module defines the executor's bucket-entry point steps:

- `ParentStep`: a 0-dependency source step seeded with the bucket's live parent
  objects; its `execute` simply returns that column. It is the root that other
  steps in a bucket read from.
- `ResolveStep`: the legacy-resolver adapter. It wraps an ordinary field resolver
  and, per the Step contract, runs once over the bucket — but internally it still
  loops `resolve_fn(parent, info, **args)` once per parent, which is correct (if
  un-batched) for arbitrary user resolvers and is what keeps the graphql-core
  conformance suite green. Genuine batch steps (loadOne/loadMany) that issue ONE
  call for the whole bucket arrive in the next build stage.

`run_resolve_step` keeps its existing signature and `ResolveOutcome` return
contract, but now produces the field's raw value list by building and running a
tiny `ParentStep -> ResolveStep` DAG through the shared `run_steps` executor, so
even the legacy path flows through the real step model.
"""

from typing import Any, Dict, List, NamedTuple, Optional

from graphql.execution.execute import GraphQLResolveInfo
from graphql.pyutils import Path

from .plan import FieldPlan
from .step_model import Step, run_steps


class ResolveOutcome(NamedTuple):
    """The batched result of running one field's resolver over a bucket.

    `values[i]` is the raw resolver return for parent `i` (or the raised
    exception, kept as a value so the executor applies located-error handling per
    parent). `infos[i]`/`paths[i]` are that parent's resolve info and field path.
    `awaitable` flags that at least one value is a coroutine, which forces the
    executor to defer this field to the stock async-capable path.
    """

    values: List[Any]
    infos: List[GraphQLResolveInfo]
    paths: List[Path]
    awaitable: bool


class ParentStep(Step):
    """A 0-dependency source step seeded with a bucket's parent objects.

    `execute` returns the parent column verbatim; it is the DAG root that a
    `ResolveStep` (and, later, access/load steps) reads from. The parents are bound
    per bucket invocation, so this step is constructed fresh for each bucket rather
    than shared across the plan.
    """

    def __init__(self, parents: List[Any]) -> None:
        super().__init__()
        self.parents = parents

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        return self.parents


class ResolveStep(Step):
    """Legacy-resolver adapter: runs a field's resolver once per parent in a bucket.

    Depends on a single `ParentStep` (dep 0). `execute` builds the effective
    resolver exactly as graphql-core's execute_field does (field resolver or
    context default, wrapped through middleware), then invokes it once per parent,
    returning the raw per-parent value column. Per-parent resolve `info` and field
    `path` are recorded as side state (`self.infos` / `self.paths`) for the
    completer, and an awaitable result sets `self.awaitable`. Argument coercion that
    failed at plan time is replayed as a per-parent raised value located at the
    field.

    Looping over parents internally is acceptable for arbitrary user resolvers; the
    batching payoff comes from the batch steps (loadOne/loadMany), not this adapter.
    """

    def __init__(self, context, field_plan: FieldPlan, parent_paths: List[Path]):
        super().__init__()
        self.context = context
        self.field_plan = field_plan
        self.parent_paths = parent_paths
        self.infos: List[GraphQLResolveInfo] = []
        self.paths: List[Path] = []
        self.awaitable = False

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        context = self.context
        field_plan = self.field_plan
        parents = values[0]

        field_def = field_plan.field_def
        resolve_fn = field_def.resolve or context.field_resolver
        if context.middleware_manager:
            resolve_fn = context.middleware_manager.get_field_resolver(resolve_fn)

        is_awaitable = context.is_awaitable
        out: List[Any] = []
        infos: List[GraphQLResolveInfo] = []
        paths: List[Path] = []

        args: Dict[str, Any] = field_plan.args or {}

        for parent, parent_path in zip(parents, self.parent_paths):
            field_path = Path(
                parent_path, field_plan.response_name, field_plan.parent_type.name
            )
            info = context.build_resolve_info(
                field_def, field_plan.field_nodes, field_plan.parent_type, field_path
            )
            infos.append(info)
            paths.append(field_path)

            if field_plan.args_error is not None:
                out.append(field_plan.args_error)
                continue
            try:
                value = resolve_fn(parent, info, **args)
            except Exception as raw_error:  # resolver raised → carry as a value
                out.append(raw_error)
                continue
            if is_awaitable(value):
                self.awaitable = True
            out.append(value)

        self.infos = infos
        self.paths = paths
        return out


def run_resolve_step(
    context,
    field_plan: FieldPlan,
    parents: List[Any],
    parent_paths: List[Path],
) -> ResolveOutcome:
    """Run a field's resolver batched across every parent in the bucket.

    Builds a tiny `ParentStep -> ResolveStep` DAG and runs it once through the
    shared step executor (`run_steps`), so the legacy path uses the same per-bucket
    step model the batch steps will. The resolver-step's recorded per-parent infos
    and paths, plus its raw value column, are assembled back into a `ResolveOutcome`
    with the unchanged downstream contract.
    """
    parent_step = ParentStep(parents)
    resolve_step = ResolveStep(context, field_plan, parent_paths)
    resolve_step.add_dependency(parent_step)

    parent_step.id = 0
    resolve_step.id = 1

    results = run_steps(len(parents), [parent_step, resolve_step], context.is_awaitable)
    values = results[resolve_step.id]

    return ResolveOutcome(
        values=list(values),
        infos=resolve_step.infos,
        paths=resolve_step.paths,
        awaitable=resolve_step.awaitable,
    )
