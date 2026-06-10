"""The bucket/batch executor.

Executes an ObjectPlan over a *bucket* of parent objects layer by layer. The
bucket's steps run ONCE over all parents (a plan step or a resolver-adapter
ResolveStep — every field carries one), producing each field's value column in the
bucket store; we then drive the field's **completer** (the wrapping-type descent:
NonNull/List/leaf/object) over that batch of resolved values via `complete_values`.
Object completion recurses into a child bucket (the Grafast batching point); lists
fan their items into a fresh batch.

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

from typing import Any, AsyncIterable, Dict, List, NamedTuple, Optional

from graphql.error import GraphQLError, located_error
from graphql.pyutils import Path, Undefined, is_iterable

from . import _compat
from .bubble import Bubble
from .completion import (
    HoistBridge,
    NonNullCompleter,
    complete_values,
    find_object_completer,
)
from .dag import order_steps_within
from .plan import FieldPlan, LayerPlan, ObjectPlan
from .step_model import run_steps
from .steps import BucketExtra


class _FieldOutcome(NamedTuple):
    """The per-field batched result completion consumes: value column + per-parent info/path.

    Execute-internal carrier (the old cross-module `ResolveOutcome`, minus the unused
    `awaitable` flag): `values[i]` is the field's raw value for live parent `i` (or a raised
    exception kept as a value), `infos[i]`/`paths[i]` its resolve info and field path. Async
    is driven by coroutines sitting in `values`, detected by `is_awaitable` in completion —
    no awaitable flag is needed.
    """

    values: List[Any]
    infos: List[Any]
    paths: List[Path]


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


def run_layer(
    context,
    layer: LayerPlan,
    parents: List[Any],
    parent_paths: List[Path],
    parent_store: Optional[Dict[int, List[Any]]] = None,
):
    """Run a LayerPlan's steps ONCE over a bucket of parents -> the bucket store.

    Pure batch-boundary concern: reads ONLY the LayerPlan — its `parent_step` boundary and
    its finalize-materialised `ordered_steps` — never the output shape. We seed the
    `parent_step` id with `parents` (the operation root, or the enclosing field's column),
    build the per-invocation `BucketExtra` (request context + per-parent `parent_paths`,
    needed by any `ResolveStep` to build each resolver `info.path`), and run the layer's
    steps in dependency order via `run_steps`. The decisive effect: a `loadMany` in this
    layer sees EVERY key across all `parents` in one `execute`, so its batch callback fires
    exactly once for the whole bucket.

    `run_steps` already includes any `effect_steps` an optimizer orphaned (a mutation whose
    return value was inlined): no field consumes them, so they RUN FOR EFFECT here, their
    output column discarded but the write executed.

    `parent_store` is the GENERIC request-state seam: an optional `step.id -> column` dict
    whose entries are seeded as ADDITIONAL boundary columns alongside `parent_step`. This is
    one channel three uses share rather than three parallel ones: (1) child-layer
    detachment seeds a child object layer directly from a parent value column + index
    map, decoupled from the parent walk (exercised by tests/test_lifecycle_detachment.py);
    (2) per-request placeholder-value maps; (3) deferred-job continuations re-reading
    a parent column. INVARIANT: every `parent_store` key MUST already be a boundary excluded
    from `layer.ordered_steps` — which was precomputed against `{parent_step.id}` at finalize
    (`populate_layers`). Seeding `parent_step.id` itself (the only key child-layer detachment
    uses) always satisfies this; seeding a NON-boundary column would have to recompute the
    order (`order_steps_within`) for that call. We do NOT recompute the order per call here —
    that would be a perf regression and is unnecessary for the boundary-only keys used today.

    Returns a dict `step.id -> output column` (or a coroutine resolving to it when an async
    load is in the sub-DAG), or `None` for a bucket with no steps to run (a boundary-less
    bucket) — unchanged from before the de-fusion.
    """
    if layer.parent_step is None or not layer.run_steps:
        return None
    seed = dict(parent_store) if parent_store else {}
    seed[layer.parent_step.id] = parents
    extra = BucketExtra(
        context, parent_paths, getattr(context, "_grafast_source_values", {})
    )
    return run_steps(
        len(parents),
        layer.ordered_steps,
        context.is_awaitable,
        seed=seed,
        on_step_batch=getattr(context, "_grafast_on_step_batch", None),
        extra=extra,
    )


def execute_object_plan(
    context,
    plan: ObjectPlan,
    parents: List[Any],
    parent_paths: List[Path],
    parent_store: Optional[Dict[int, List[Any]]] = None,
):
    """Execute an ObjectPlan over a bucket: run its LayerPlan, then walk its output.

    Returns a list (one entry per parent) of either an output dict or a `Bubble`
    (when a non-null subfield nulled that parent). Returns a coroutine resolving
    to that list when any field involves an awaitable.

    The two halves are DE-FUSED: `run_layer` produces the bucket store (step.id -> column)
    from the LayerPlan alone, and `walk_output` serialises THIS object plan against that
    store — every field (a plan step or a resolver-adapter ResolveStep) reads its
    already-batched value column from the store, read uniformly.

    `parent_store` is forwarded GENERICALLY to `run_layer` as the request-state seam (see
    its docstring): additional `step.id -> column` boundary seeds shared by child-layer
    detachment, placeholder-value maps and deferred continuations. Nothing here
    special-cases a consumer — the dict is the only channel.
    """
    store = run_layer(context, plan.layer, parents, parent_paths, parent_store)
    if context.is_awaitable(store):

        async def after_steps():
            cols = await store
            return walk_output(context, plan, parents, parent_paths, cols)

        return _await_bucket(context, after_steps())
    return walk_output(context, plan, parents, parent_paths, store)


def run_child_layer_from_store(context, child_plan, parent_store, index_map, child_paths):
    """Run a child object layer seeded from a parent store column + an index map.

    The decisive simplification here: a child object layer is runnable from
    its parent's value column ALONE, without re-entering the parent walk. `parent_store`
    holds `child_plan.layer.parent_step.id -> parent value column`; `index_map` (which IS the
    completion-time `keep_origin`/`origin` scatter list) projects each child-bucket position
    to its parent-bucket position. The seed list is the parent column projected through the
    index map, which is element-for-element equal to the transient `keep_objs`/`objs` the
    inline completion path materialises — so the IDENTICAL child step DAG runs over the
    IDENTICAL bucket.

    The projection reads an ALREADY-RESOLVED column (the caller projects after
    `resolve_awaitable_values`), so async ordering is unchanged. Returns the child store /
    completion list exactly as `execute_object_plan` would, leaving the per-parent scatter
    back through `index_map` to the caller.
    """
    parent_column = parent_store[child_plan.layer.parent_step.id]
    seed_objs = [parent_column[o] for o in index_map]
    return execute_object_plan(context, child_plan, seed_objs, child_paths)


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

    # mint this level's deferred fragment records BEFORE walking fields, so an INITIAL
    # object field's deeper @defer groups (captured mid-walk) can resolve a parent record minted
    # at THIS level (e.g. a root-fragment @defer whose only field merged into the initial data).
    # No-op off the incremental path (no defer sink). Group CREATION still happens after the walk
    # (it needs live parents), in `capture_deferred_jobs`.
    premint_defer_records(context, plan, parents, parent_paths, state)

    pending = []
    for field_plan in plan.fields:
        maybe = complete_field(
            context, field_plan, parents, parent_paths, state, store
        )
        if maybe is not None:
            pending.append(maybe)

    # early execution: capture this level's @defer groups right after the SYNC field pass
    # (before awaiting async siblings), so a fast deferred resolver runs before a slow initial
    # one — matching upstream's collect_execution_groups firing during the walk. Liveness uses
    # the post-sync-pass state (a parent already bubbled here is dropped; a later async bubble
    # is handled by the publisher's path filter). Only when early execution is on; otherwise the
    # capture happens after the gather, where group runs are lazy anyway (byte-identical).
    early = getattr(context, "_grafast_enable_early_execution", False)
    if early and pending:
        capture_deferred_jobs(context, plan, parents, parent_paths, state)

    # A synchronous non-null violation short-circuits: graphql-core raises out of
    # the field loop before awaiting any sibling. If every parent in this bucket
    # was already nulled (bubbled) by the synchronous pass, the pending async
    # fields can only write into dead parents, so abandon them and return now.
    if pending and all(b is not None for b in state.bubble):
        if not early:
            capture_deferred_jobs(context, plan, parents, parent_paths, state)
        return assemble(state)

    if not pending:
        capture_deferred_jobs(context, plan, parents, parent_paths, state)
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
        if not early:
            capture_deferred_jobs(context, plan, parents, parent_paths, state)
        return assemble(state)

    return finish()


def premint_defer_records(context, plan, parents, parent_paths, state) -> None:
    """Mint this level's deferred fragment records (per parent path) BEFORE the field walk.

    Records must exist before descending into INITIAL object fields, so a deeper @defer
    group captured mid-walk resolves a parent record minted at this level. Group CREATION still
    happens after the walk (it needs live parents). No defer sink (the default / non-incremental
    path) => no-op, byte-identical. Mints for every parent path (liveness is not yet known; a
    dead parent's records are simply never turned into groups).
    """
    if not plan.deferred.new_defer_usages:
        return
    if getattr(context, "_grafast_defer_sink", None) is None:
        return
    from .incremental import add_new_deferred_fragments, get_defer_registry

    registry = get_defer_registry(context)
    for path in parent_paths:
        add_new_deferred_fragments(registry, plan.deferred.new_defer_usages, path)


def capture_deferred_jobs(context, plan, parents, parent_paths, state) -> None:
    """Record this bucket's @defer'd execution groups for the incremental driver.

    When ``plan.deferred`` carries new grouped-field-sets AND the context carries a
    deferred-group sink (set only by the incremental entry on graphql-core 3.3), record — per
    LIVE parent — the deferred fragment records minted from this level's new defer usages and an
    execution group per new grouped-field-set. A parent is live when its output dict survived
    (not nulled by a non-null violation), so a defer hanging off a bubbled parent is dropped
    (the cancels-deferred-fields-on-null-bubbling behaviour). No sink (the default /
    non-incremental path) => this is a no-op and byte-identical.
    """
    if not plan.deferred.new_defer_usages and not plan.deferred.new_groups:
        return
    sink = getattr(context, "_grafast_defer_sink", None)
    if sink is None:
        return
    for i in range(len(parents)):
        if not is_live(state, i):
            continue
        sink(plan.deferred, parents[i], parent_paths[i], plan.parent_type)


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

    Every field carries a `FieldPlan.step` (a plan step or a resolver-adapter
    ResolveStep), so completion reads its value column uniformly: the batched path
    selects the column from the bucket store `step_columns` (the step DAG already ran
    once over the whole bucket — this is where automatic batching is realised), and
    the serial (mutation) path runs just this field's step sub-DAG over the live
    parents. There is no separate per-parent resolver path.
    """
    live_idx = [i for i in range(len(parents)) if is_live(state, i)]
    if not live_idx:
        return None

    live_parents = [parents[i] for i in live_idx]
    live_paths = [parent_paths[i] for i in live_idx]
    bridge = None
    if step_columns is not None:
        # batched path: this field's value column is already in the bucket store
        # (a plan step or a resolver-adapter ResolveStep — read uniformly).
        outcome = plan_field_outcome(
            context, field_plan, live_idx, live_paths, step_columns
        )
        # hoist channel: when this field descends into a child layer that had steps hoisted
        # OUT of it, build the bridge carrying THIS bucket's store (where the hoisted columns
        # now live) + the per-value parent-bucket owner (`live_idx`), so the child seeds those
        # columns instead of re-running the hoisted steps. None when nothing was hoisted (the
        # default — byte-identical: `complete_values` never threads it).
        bridge = hoist_bridge_for_field(
            field_plan, step_columns, live_idx, context.is_awaitable
        )
    else:
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

    # @stream: a @stream'd list field completes items[:initialCount] inline (into the
    # initial data) and hands items[initialCount:] to a stream producer the driver drains
    # item-by-item. Only when a stream sink is present (the incremental entry on 3.3); else
    # the field completes whole, byte-identical.
    if field_plan.stream is not None and getattr(context, "_grafast_stream_sink", None):
        return complete_stream_field(
            context, field_plan, outcome, live_idx, state, bridge
        )

    completed = complete_values(
        context,
        field_plan.completer,
        list(outcome.values),
        outcome.paths,
        outcome.infos,
        field_plan.field_nodes,
        field_plan.field_label,
        bridge,
    )

    if not context.is_awaitable(completed):
        scatter(context, field_plan, completed, live_idx, outcome.paths, state)
        return None

    async def finish():
        resolved = await completed
        scatter(context, field_plan, resolved, live_idx, outcome.paths, state)

    return finish()


def complete_stream_field(context, field_plan, outcome, live_idx, state, bridge):
    """Complete a @stream'd list field: head inline, tail to a per-item stream producer.

    The @stream path (only when a stream sink is present, 3.3): each list value's
    ``items[:initialCount]`` complete inline into the initial data, and ``items[initialCount:]``
    (RAW, uncompleted) are handed to a stream producer the driver drains item-by-item with
    upstream's batching (sync list, list of awaitables, and async iterator all uniform). A
    @stream arg coercion error or a negative initialCount surfaces as a located field error
    (the value bubbles). Returns None (sync) or a coroutine, like ``complete_field``.
    """
    from .incremental import build_stream_record

    list_completer = find_list_completer(field_plan.completer)
    sink = context._grafast_stream_sink
    pending = []
    for k, value in enumerate(outcome.values):
        i = live_idx[k]
        # hoist channel for the streamed items: every one of THIS parent's list items
        # (head completed inline + tail drained by the producer) descends into the child layer
        # carrying THIS parent's hoisted values (completion value k). Carry a per-parent seed — a
        # 1-element column per hoisted step the head expands ×len and each tail item uses as-is.
        # None when nothing was hoisted (byte-identical).
        item_bridge = (
            HoistBridge(columns={hid: [col[k]] for hid, col in bridge.columns.items()})
            if bridge is not None
            else None
        )
        result = complete_one_stream_value(
            context, field_plan, list_completer, value, outcome.paths[k],
            outcome.infos[k], sink, build_stream_record, i, state, item_bridge,
        )
        if context.is_awaitable(result):
            pending.append(result)
    if not pending:
        return None

    async def finish():
        from asyncio import gather

        await gather(*pending)

    return finish()


def complete_one_stream_value(
    context, field_plan, list_completer, value, path, info, sink,
    build_stream_record, parent_index, state, item_bridge=None,
):
    """Complete ONE parent's @stream'd list value (sync → None / async → coroutine).

    `item_bridge` is the per-parent hoist seed (1-element columns) threaded down to the head +
    tail item completion so streamed child objects seed their hoisted-out columns.
    """
    stream = field_plan.stream
    if isinstance(stream, _compat.StreamError):
        error = located_error(stream.error, field_plan.field_nodes, path.as_list())
        write_value(context, field_plan, state, parent_index, Bubble(error), path)
        return None
    initial_count, label = stream

    def emit(head_completed):
        write_value(context, field_plan, state, parent_index, head_completed, path)

    if context.is_awaitable(value):

        async def after_await():
            try:
                resolved = await value
            except (GraphQLError, TypeError, ValueError, RuntimeError) as exc:
                emit(Bubble(located_error(exc, field_plan.field_nodes, path.as_list())))
                return
            inner = drive_stream_value(
                context, field_plan, list_completer, resolved, path, info,
                initial_count, label, sink, build_stream_record, parent_index, state, item_bridge,
            )
            if context.is_awaitable(inner):
                await inner

        return after_await()

    return drive_stream_value(
        context, field_plan, list_completer, value, path, info,
        initial_count, label, sink, build_stream_record, parent_index, state, item_bridge,
    )


def drive_stream_value(
    context, field_plan, list_completer, value, path, info, initial_count, label,
    sink, build_stream_record, parent_index, state, item_bridge=None,
):
    """Complete the head + register the stream producer for one resolved list value.

    `item_bridge` (per-parent hoist seed) threads into BOTH the inline head completion and the
    producer that drains the tail, so streamed child objects seed their hoisted-out columns.
    """
    from .incremental import AsyncStreamProducer, SyncStreamProducer

    if initial_count < 0:
        error = located_error(
            GraphQLError("initialCount must be a positive integer"),
            field_plan.field_nodes,
            path.as_list(),
        )
        write_value(context, field_plan, state, parent_index, Bubble(error), path)
        return None

    item_completer = list_completer.item_completer

    if value is None:
        write_value(context, field_plan, state, parent_index, None, path)
        return None

    if not is_iterable(value) and isinstance(value, AsyncIterable):
        iterator = value.__aiter__()
        early_return = getattr(iterator, "aclose", None)
        producer = AsyncStreamProducer(
            context, field_plan, item_completer, path, info, iterator, initial_count, item_bridge
        )
        # pull + complete the head (items[:initial_count]) inline into the initial data.
        return drain_async_head(
            context, field_plan, producer, path, sink, build_stream_record,
            label, early_return, parent_index, state, item_bridge,
        )

    if not is_iterable(value):
        err = GraphQLError(
            "Expected Iterable, but did not find one for field"
            f" '{field_plan.field_label}'."
        )
        write_value(
            context, field_plan, state, parent_index,
            Bubble(located_error(err, field_plan.field_nodes, path.as_list())), path,
        )
        return None

    items = list(value)
    head_raw = items[:initial_count]
    tail_raw = items[initial_count:]
    producer = SyncStreamProducer(
        context, field_plan, item_completer, path, info, tail_raw, initial_count, item_bridge
    )
    stream_rec = build_stream_record(path, label, producer, None)
    completed_head = complete_stream_head(
        context, field_plan, item_completer, head_raw, path, info, item_bridge
    )
    if context.is_awaitable(completed_head):

        async def finish():
            head = await completed_head
            head = scatter_stream_head(context, field_plan, item_completer, head)
            if isinstance(head, Bubble):
                write_value(context, field_plan, state, parent_index, head, path)
                return
            write_value(context, field_plan, state, parent_index, head, path)
            sink(stream_rec)

        return finish()
    head = scatter_stream_head(context, field_plan, item_completer, completed_head)
    if isinstance(head, Bubble):
        write_value(context, field_plan, state, parent_index, head, path)
        return None
    write_value(context, field_plan, state, parent_index, head, path)
    sink(stream_rec)
    return None


def drain_async_head(
    context, field_plan, producer, path, sink, build_stream_record, label,
    early_return, parent_index, state, item_bridge=None,
):
    """Pull + complete the async iterator's head (items[:initialCount]) inline.

    Returns a coroutine (an async iterator always awaits ``anext``). The head items become the
    field's initial list value; the live iterator is then handed to the driver as a stream
    record. An error WHILE pulling the head surfaces as a field error (the field bubbles)."""

    from graphql.pyutils import Undefined as _Undefined

    async def run():
        # pull the raw head items (items[:initialCount]) from the iterator; a pull error before
        # initialCount surfaces as a FIELD error (the field bubbles, no stream). Exhaustion just
        # ends the head early. Each item is completed through the list item completer with the
        # MAIN context so a nullable head item error lands in the initial result's errors.
        raw_head = []
        bubbled = None
        for _ in range(producer.initial_count):
            try:
                item = await producer._iterator.__anext__()
            except StopAsyncIteration:
                break
            except (GraphQLError, TypeError, ValueError, RuntimeError) as exc:
                bubbled = located_error(exc, field_plan.field_nodes, path.as_list())
                break
            raw_head.append(item)
            producer._index += 1
        if bubbled is None:
            completed_head = complete_stream_head(
                context, field_plan, producer._item_completer, raw_head, path,
                info_of(producer), item_bridge,
            )
            if context.is_awaitable(completed_head):
                completed_head = await completed_head
            head = scatter_stream_head(
                context, field_plan, producer._item_completer, completed_head
            )
            if isinstance(head, Bubble):
                bubbled = head.error
        if bubbled is not None:
            write_value(context, field_plan, state, parent_index, Bubble(bubbled), path)
            if early_return is not None:
                from contextlib import suppress

                with suppress(Exception):
                    await early_return()
            return
        write_value(context, field_plan, state, parent_index, head, path)
        producer.start()
        stream_rec = build_stream_record(path, label, producer, early_return)
        sink(stream_rec)

    return run()


def info_of(producer):
    return producer._info


def complete_stream_head(context, field_plan, item_completer, head_raw, path, info, item_bridge=None):
    """Complete the head items (items[:initialCount]) through the list item completer.

    `item_bridge` is the per-parent hoist seed (1-element columns); since every head item shares
    the same parent's hoisted values, expand each column to one entry per head item so the child
    object completer projects the hoisted column for each.
    """
    if not head_raw:
        return []
    item_paths = [path.add_key(idx, None) for idx in range(len(head_raw))]
    item_infos = [info] * len(head_raw)
    head_bridge = (
        HoistBridge(columns={hid: col * len(head_raw) for hid, col in item_bridge.columns.items()})
        if item_bridge is not None
        else None
    )
    return complete_values(
        context, item_completer, list(head_raw), item_paths, item_infos,
        field_plan.field_nodes, field_plan.field_label, head_bridge,
    )


def scatter_stream_head(context, field_plan, item_completer, completed_items):
    """Re-collect completed head items into the list value, handling item bubbles.

    Mirrors ``collect_lists`` for the head: a non-null item bubble nulls the whole list
    (returned as a Bubble); a nullable item bubble appends its error + None."""
    item_is_non_null = isinstance(item_completer, NonNullCompleter)
    out = []
    for item in completed_items:
        if isinstance(item, Bubble):
            if item_is_non_null:
                return item
            context.errors.append(item.error)
            out.append(None)
            continue
        out.append(item)
    return out


def find_list_completer(completer):
    """Return the ListCompleter at the head of a (possibly NonNull-wrapped) @stream'd field."""
    from .completion import ListCompleter

    if isinstance(completer, ListCompleter):
        return completer
    if isinstance(completer, NonNullCompleter):
        return find_list_completer(completer.inner)
    return None


def hoist_bridge_for_field(field_plan, step_columns, live_idx, is_awaitable):
    """Build the :class:`HoistBridge` for a field that descends into a hoist-affected child.

    Projects each column the child's layer HOISTED OUT (produced once in THIS bucket's store,
    `step_columns`) from parent-bucket granularity to the field's COMPLETION-VALUE granularity via
    `live_idx` (the per-outcome-value parent-bucket position: outcome value k belongs to parent
    bucket row `live_idx[k]`), so the bridge's columns are parallel to the field's completed values
    and reshape in lockstep with them through completion. Built ONLY when the field's leaf object
    child plan actually has hoisted-out steps; otherwise (the default / hoist-off path) returns
    `None`, so completion never threads a bridge and is byte-identical.
    """
    child = find_object_completer(field_plan.completer)
    if child is None or child.child_plan is None:
        return None
    hoisted_out = child.child_plan.layer.hoisted_out_ids
    if not hoisted_out:
        return None
    columns = {hid: [step_columns[hid][p] for p in live_idx] for hid in hoisted_out}
    # SHARE-POINT GUARD: a hoisted column is FANNED to many child rows, so EVERY value must be
    # concrete — an awaitable is single-await and cannot be copied (the @stream path, completing
    # rows separately, would re-await it -> "cannot reuse already awaited coroutine"). Uses the
    # request's CONFIGURED ``is_awaitable`` (not ``inspect.isawaitable``) so a host's custom
    # promise-like is recognised exactly as completion would. A hoisted step is sync-eligible by
    # construction; an awaitable here means a step lied about sync-ness (the LambdaStep detection
    # blind spot). Checks the WHOLE column (cheap; does not rely on the request-constant uniformity
    # invariant). Fail loudly rather than alias it.
    for hid, col in columns.items():
        if any(is_awaitable(v) for v in col):
            raise AssertionError(
                f"hoisted step #{hid} produced an awaitable; a hoisted value is fanned across rows"
                f" and must be concrete (an async fn must not be hoistable — for async I/O use a"
                f" load step, which batches AND resolves its column before any fan-out)"
            )
    return HoistBridge(columns=columns)


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
    siblings. Threads a per-invocation `BucketExtra` (request context + per-parent paths)
    so a resolver-adapter ResolveStep on this field gets its context/paths. Returns a
    `_FieldOutcome` (or a coroutine resolving to one when the sub-DAG has an async load).
    """
    boundary = {parent_step.id}
    ordered = order_steps_within([field_plan.step], boundary)
    seed = {parent_step.id: live_parents}
    extra = BucketExtra(
        context, live_paths, getattr(context, "_grafast_source_values", {})
    )
    columns = run_steps(
        len(live_parents),
        ordered,
        context.is_awaitable,
        seed=seed,
        on_step_batch=getattr(context, "_grafast_on_step_batch", None),
        extra=extra,
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
            return _FieldOutcome(values=list(cols[step_id]), infos=infos, paths=paths)

        return finish()

    return _FieldOutcome(values=list(columns[step_id]), infos=infos, paths=paths)


def plan_field_outcome(context, field_plan, live_idx, live_paths, step_columns):
    """Assemble a `_FieldOutcome` for a field from its already-batched step column.

    The bucket DAG produced `field_plan.step`'s output for every parent position (a plan
    step or a resolver-adapter ResolveStep alike); we select the live entries and build
    the field path / resolve info per live parent (completion needs them for abstract-type
    resolution and error locating). The values came from ONE batched step run.
    """
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

    return _FieldOutcome(values=values, infos=infos, paths=paths)


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
