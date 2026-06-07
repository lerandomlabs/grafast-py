"""The bucket/batch executor.

Executes an ObjectPlan over a *bucket* of parent objects layer by layer. For each
field plan we run one batched ResolveStep over all parents, then drive the field's
**completer** (the wrapping-type descent: NonNull/List/leaf/object) over that batch
of resolved values via `complete_values`. Object completion recurses into a child
bucket (the Grafast batching point); lists fan their items into a fresh batch.

Null-bubbling is reproduced natively *per parent*: completion returns a `Bubble`
for a parent whose non-null sub-structure resolved to None. At the field's write
boundary an uncaught bubble nulls the whole parent dict and propagates the bubble
upward via `state.bubble[i]`; at a nullable enclosing field the bubble's error is
appended and the value becomes None; at the operation root it surfaces as a raised
GraphQLError (data -> None).

Async resolvers, async list items, and async iterables are supported: when any
completion is awaitable the whole object-plan execution becomes a coroutine, but
the resulting `data`/`errors` are identical to the synchronous path.
"""

from typing import Any, Dict, List, Optional

from graphql.pyutils import Path

from .bubble import Bubble
from .completion import NonNullCompleter, complete_values
from .dag import order_steps_within
from .plan import FieldPlan, LayerPlan, ObjectPlan
from .step_model import run_steps
from .steps import run_resolve_step


class FieldCompletion:
    """Per-bucket completion state: the running output dicts plus their bubbles.

    `outputs[i]` is parent i's dict, or None once a non-null child nulled it.
    `bubble[i]` is a Bubble when parent i is dead due to a non-null violation that
    has not yet reached a nullable boundary; the enclosing object field inspects it.
    """

    def __init__(self, n: int):
        self.outputs: List[Optional[dict]] = [{} for _ in range(n)]
        self.bubble: List[Optional[Bubble]] = [None] * n


def is_live(state: "FieldCompletion", i: int) -> bool:
    """A parent is live while it has no bubble and its dict is not nulled."""
    return state.bubble[i] is None and state.outputs[i] is not None


def run_layer(context, layer: LayerPlan, parents: List[Any]):
    """Run a LayerPlan's steps ONCE over a bucket of parents -> the bucket store.

    Pure batch-boundary concern: reads ONLY the LayerPlan — its `parent_step` boundary and
    its finalize-materialised `ordered_steps` — never the output shape. We seed the
    `parent_step` id with `parents` (the operation root, or the enclosing plan field's
    column) and run the layer's steps in dependency order via `run_steps`. The decisive
    effect: a `loadMany` in this layer sees EVERY key across all `parents` in one
    `execute`, so its batch callback fires exactly once for the whole bucket.

    `run_steps` already includes any `effect_steps` an optimizer orphaned (a mutation whose
    return value was inlined): no field consumes them, so they RUN FOR EFFECT here, their
    output column discarded but the write executed.

    Returns a dict `step.id -> output column` (or a coroutine resolving to it when an async
    load is in the sub-DAG), or `None` for a bucket with no steps to run (the pure
    legacy-resolver path, or a boundary-less bucket) — unchanged from before the de-fusion.
    """
    if layer.parent_step is None or not layer.run_steps:
        return None
    seed = {layer.parent_step.id: parents}
    return run_steps(
        len(parents),
        layer.ordered_steps,
        context.is_awaitable,
        seed=seed,
        on_step_batch=getattr(context, "_grafast_on_step_batch", None),
    )


def execute_object_plan(
    context,
    plan: ObjectPlan,
    parents: List[Any],
    parent_paths: List[Path],
):
    """Execute an ObjectPlan over a bucket: run its LayerPlan, then walk its output.

    Returns a list (one entry per parent) of either an output dict or a `Bubble`
    (when a non-null subfield nulled that parent). Returns a coroutine resolving
    to that list when any field involves an awaitable.

    The two halves are DE-FUSED: `run_layer` produces the bucket store (step.id -> column)
    from the LayerPlan alone, and `walk_output` serialises THIS object plan against that
    store — each plan field reads its already-batched value column instead of re-entering a
    resolver per parent; legacy fields keep the per-parent adapter.
    """
    store = run_layer(context, plan.layer, parents)
    if context.is_awaitable(store):

        async def after_steps():
            cols = await store
            return walk_output(context, plan, parents, parent_paths, cols)

        return _await_bucket(context, after_steps())
    return walk_output(context, plan, parents, parent_paths, store)


def _await_bucket(context, awaitable):
    """Await a bucket completion that may itself resolve to a coroutine."""

    async def run():
        result = await awaitable
        if context.is_awaitable(result):
            return await result
        return result

    return run()


def walk_output(
    context,
    plan: ObjectPlan,
    parents: List[Any],
    parent_paths: List[Path],
    store: Optional[Dict[int, List[Any]]],
):
    """Serialise one object plan against an ALREADY-PRODUCED bucket store.

    Walks `plan.fields`, completing each against `store` (the `run_layer` result). Does NOT
    run this bucket's layer — the caller did — so it can be driven with any conforming
    store, which is what de-fuses serialization from execution.
    """
    state = FieldCompletion(len(parents))

    pending = []
    for field_plan in plan.fields:
        maybe = complete_field(
            context, field_plan, parents, parent_paths, state, store
        )
        if maybe is not None:
            pending.append(maybe)

    # A synchronous non-null violation short-circuits: graphql-core raises out of
    # the field loop before awaiting any sibling. If every parent in this bucket
    # was already nulled (bubbled) by the synchronous pass, the pending async
    # fields can only write into dead parents, so abandon them and return now.
    if pending and all(b is not None for b in state.bubble):
        return assemble(state)

    if not pending:
        return assemble(state)

    async def finish():
        # sibling fields run concurrently: their resolver coroutines were already
        # created during the synchronous pass above, so gathering the per-field
        # completion coroutines lets all in-flight resolvers progress in parallel.
        # When `max_step_concurrency` is set, a semaphore caps how many run at once.
        from asyncio import gather

        semaphore = getattr(context, "_grafast_concurrency", None)
        if semaphore is None:
            await gather(*pending)
        else:
            await gather(*(_gated(semaphore, p) for p in pending))
        return assemble(state)

    return finish()


async def _gated(semaphore, awaitable):
    """Await `awaitable` while holding `semaphore` (bounded concurrency)."""
    async with semaphore:
        return await awaitable


def execute_object_plan_serially(
    context,
    plan: ObjectPlan,
    parents: List[Any],
    parent_paths: List[Path],
):
    """Execute an ObjectPlan over a bucket with each field fully resolved in turn.

    This is the mutation path: the spec requires the top-level fields of a mutation
    to run serially, so a field's resolver — and its entire completion subtree — is
    awaited to completion before the next field's resolver is invoked. Unlike
    `execute_object_plan`, sibling resolvers are never created concurrently; the
    side effects of `first` are observed by `second`.

    Returns the same per-parent dict / `Bubble` list (or a coroutine) as the
    parallel executor.

    `plan.layer.effect_steps` (side-effecting steps an optimizer orphaned by inlining their
    return value) are run FOR EFFECT up front: a mutation whose result is not selected
    still must write. If that run is async the whole serial pass becomes a coroutine that
    awaits the effects before completing any field. With the default identity optimize
    `effect_steps` is empty, so this step is skipped entirely.
    """
    state = FieldCompletion(len(parents))

    if plan.layer.effect_steps:
        effects = run_effect_steps(context, plan, parents)
        if context.is_awaitable(effects):

            async def after_effects():
                await effects
                later = execute_object_plan_serially_fields(
                    context, plan, parents, parent_paths, state
                )
                if context.is_awaitable(later):
                    return await later
                return later

            return after_effects()

    return execute_object_plan_serially_fields(
        context, plan, parents, parent_paths, state
    )


def execute_object_plan_serially_fields(
    context,
    plan: ObjectPlan,
    parents: List[Any],
    parent_paths: List[Path],
    state: "FieldCompletion",
):
    """Drive a mutation bucket's fields serially (the field loop of the serial path)."""
    # if any field is async the whole pass must become a coroutine so each field
    # awaits before the next begins; detect by completing the first field and
    # seeing whether it returned a coroutine, then continue accordingly
    for index, field_plan in enumerate(plan.fields):
        maybe = complete_field(
            context,
            field_plan,
            parents,
            parent_paths,
            state,
            None,
            plan.layer.parent_step,
        )
        if maybe is not None:
            # this field is async: finish it, then drive the rest one-by-one
            async def finish(start: int, first_pending):
                await first_pending
                for later_plan in plan.fields[start + 1 :]:
                    later = complete_field(
                        context,
                        later_plan,
                        parents,
                        parent_paths,
                        state,
                        None,
                        plan.layer.parent_step,
                    )
                    if later is not None:
                        await later
                return assemble(state)

            return finish(index, maybe)

    return assemble(state)


def assemble(state: FieldCompletion) -> List[Any]:
    """Collapse per-parent state into dicts / Bubbles for the caller."""
    return [b if b is not None else o for o, b in zip(state.outputs, state.bubble)]


def complete_field(
    context,
    field_plan: FieldPlan,
    parents,
    parent_paths,
    state,
    step_columns=None,
    parent_step=None,
):
    """Resolve and complete one field across the bucket (sync → None, async → coro).

    Resolves only the parents still live, drives the field completer over their
    values, then writes each completed value (or Bubble) back into its parent.

    A field WITH a plan resolver reads its value column from the bucket-level
    `step_columns` (the batched step DAG already ran once over the whole bucket —
    `field_plan.step` is the field's value step) rather than re-entering a resolver
    per parent; that is where automatic batching is realised. A field WITHOUT a plan
    resolver takes the legacy per-parent `ResolveStep` adapter.
    """
    live_idx = [i for i in range(len(parents)) if is_live(state, i)]
    if not live_idx:
        return None

    live_parents = [parents[i] for i in live_idx]
    live_paths = [parent_paths[i] for i in live_idx]
    if field_plan.step is not None and step_columns is not None:
        outcome = plan_field_outcome(
            context, field_plan, live_idx, live_paths, step_columns
        )
    elif field_plan.step is not None:
        # serial (mutation) path: no bucket-level columns were precomputed, so run
        # just this field's step sub-DAG over the live parents — keeping each
        # mutation field's effects ordered rather than batched across siblings.
        outcome = run_serial_plan_field(
            context, field_plan, live_parents, live_paths, parent_step
        )
        if context.is_awaitable(outcome):

            async def finish_serial():
                resolved = await outcome
                completed_serial = complete_values(
                    context,
                    field_plan.completer,
                    list(resolved.values),
                    resolved.paths,
                    resolved.infos,
                    field_plan.field_nodes,
                    field_plan.field_label,
                )
                if context.is_awaitable(completed_serial):
                    completed_serial = await completed_serial
                scatter(
                    context, field_plan, completed_serial, live_idx, resolved.paths, state
                )

            return finish_serial()
    else:
        outcome = run_resolve_step(context, field_plan, live_parents, live_paths)

    completed = complete_values(
        context,
        field_plan.completer,
        list(outcome.values),
        outcome.paths,
        outcome.infos,
        field_plan.field_nodes,
        field_plan.field_label,
    )

    if not context.is_awaitable(completed):
        scatter(context, field_plan, completed, live_idx, outcome.paths, state)
        return None

    async def finish():
        scatter(context, field_plan, await completed, live_idx, outcome.paths, state)

    return finish()


def run_effect_steps(context, plan: ObjectPlan, parents: List[Any]):
    """Run a bucket's orphaned side-effecting steps FOR EFFECT, discarding outputs.

    A mutation whose return value an optimizer inlined is orphaned (no `FieldPlan.step`
    consumes it) yet must still write; `plan.layer.effect_steps` holds those steps. We seed the
    bucket's `parent_step` boundary with `parents` and run the effect steps' reachable
    sub-DAG once — the `execute` is the write. The result columns are intentionally
    dropped (no field reads them). Returns a coroutine when an effect step is async (a pg
    mutation), else None. Only called when `effect_steps` is non-empty.
    """
    boundary = {plan.layer.parent_step.id}
    ordered = order_steps_within(plan.layer.effect_steps, boundary)
    seed = {plan.layer.parent_step.id: parents}
    columns = run_steps(
        len(parents),
        ordered,
        context.is_awaitable,
        seed=seed,
        on_step_batch=getattr(context, "_grafast_on_step_batch", None),
    )
    if context.is_awaitable(columns):

        async def finish():
            await columns

        return finish()
    return None


def run_serial_plan_field(context, field_plan, live_parents, live_paths, parent_step):
    """Run ONE plan field's step sub-DAG over live parents (mutation serial path).

    Seeds the bucket's `parent_step` boundary with `live_parents` and runs only this
    field's reachable steps, so a mutation field's effects are not batched across its
    siblings. Returns a `ResolveOutcome` (or a coroutine resolving to one when the
    sub-DAG has an async load).
    """
    from .steps import ResolveOutcome

    boundary = {parent_step.id}
    ordered = order_steps_within([field_plan.step], boundary)
    seed = {parent_step.id: live_parents}
    columns = run_steps(
        len(live_parents),
        ordered,
        context.is_awaitable,
        seed=seed,
        on_step_batch=getattr(context, "_grafast_on_step_batch", None),
    )

    field_def = field_plan.field_def
    infos = []
    paths = []
    for parent_path in live_paths:
        field_path = Path(
            parent_path, field_plan.response_name, field_plan.parent_type.name
        )
        infos.append(
            context.build_resolve_info(
                field_def, field_plan.field_nodes, field_plan.parent_type, field_path
            )
        )
        paths.append(field_path)

    step_id = field_plan.step.id

    if context.is_awaitable(columns):

        async def finish():
            cols = await columns
            return ResolveOutcome(
                values=list(cols[step_id]), infos=infos, paths=paths, awaitable=False
            )

        return finish()

    return ResolveOutcome(
        values=list(columns[step_id]), infos=infos, paths=paths, awaitable=False
    )


def plan_field_outcome(context, field_plan, live_idx, live_paths, step_columns):
    """Assemble a `ResolveOutcome` for a plan-resolver field from its step column.

    The bucket DAG produced `field_plan.step`'s output for every parent position;
    we select the live entries and build the field path / resolve info per live
    parent (completion needs them for abstract-type resolution and error locating).
    The values came from ONE batched step run, not a per-parent resolver loop.
    """
    from .steps import ResolveOutcome

    field_def = field_plan.field_def
    column = step_columns[field_plan.step.id]
    values = [column[i] for i in live_idx]

    infos = []
    paths = []
    for parent_path in live_paths:
        field_path = Path(
            parent_path, field_plan.response_name, field_plan.parent_type.name
        )
        infos.append(
            context.build_resolve_info(
                field_def, field_plan.field_nodes, field_plan.parent_type, field_path
            )
        )
        paths.append(field_path)

    return ResolveOutcome(values=values, infos=infos, paths=paths, awaitable=False)


def scatter(context, field_plan, completed, live_idx, paths, state) -> None:
    """Write each completed value back into its parent's dict, handling bubbles."""
    for k, value in enumerate(completed):
        i = live_idx[k]
        write_value(context, field_plan, state, i, value, paths[k])


def write_value(context, field_plan, state, i, completed, path) -> None:
    """Write a completed field value into parent i's dict, handling bubbles.

    The field write is a *catching* boundary keyed on the field's return-type
    nullability. A `Bubble` arriving at a **nullable** field is caught here: its
    error is appended once and the field becomes None. At a **non-null** field the
    bubble is not caught — it nulls the whole parent dict and propagates upward via
    `state.bubble[i]`, matching graphql-core's re-raise out of `execute_field`.
    """
    if isinstance(completed, Bubble):
        if is_non_null_field(field_plan):
            state.outputs[i] = None
            state.bubble[i] = completed
            return
        context.errors.append(completed.error)
        completed = None
    if state.outputs[i] is None:
        return
    state.outputs[i][field_plan.response_name] = completed


def is_non_null_field(field_plan: FieldPlan) -> bool:
    """True when the field's outermost return type is NonNull (no catching here)."""
    return isinstance(field_plan.completer, NonNullCompleter)
