"""Value completers: a static, wrapping-type-driven completion tree.

graphql-core completes a field value with `complete_value(return_type, result)`,
a recursive descent over the *wrapping* type (NonNull -> inner; List -> per-item;
leaf -> serialize; object -> recurse into subfields). We pre-compute that descent
once, at plan time, into a `Completer` tree so execution does not re-inspect the
type structure per value.

The executor runs a field's resolver once over a whole bucket, then drives the
field's completer over that *batch* of raw values via `complete_values`. Object
completion stays batched (one child bucket -> one recursion) — the Grafast point —
while lists fan their items into a fresh batch, recurse, and re-collect per item
in positional order.

Null-bubbling is reproduced natively: a NonNull whose inner value completes to
None yields a `Bubble` carrying the located error; the bubble propagates upward
through enclosing NonNulls and is caught at the first nullable boundary (a
nullable field, nullable list item, or nullable wrapper), where its error is
appended once and the value becomes None. At the field's top boundary an
uncaught bubble nulls the parent object (handled by the object executor's
`write_value`).

The `non_null_error` / "Expected Iterable" messages key on the *field's* parent
type and name (the field whose return type is being completed), even for failures
that occur inside a list item — matching graphql-core, which reuses that field's
`info`. We thread that label as `field_label` (e.g. "Query.listField") through the
recursion rather than re-deriving it per item.
"""

from typing import Any, AsyncIterable, List, NamedTuple, Optional

from graphql.error import GraphQLError, located_error
from graphql.pyutils import Path, Undefined, inspect, is_iterable
from graphql.type import (
    GraphQLAbstractType,
    GraphQLLeafType,
    GraphQLObjectType,
    GraphQLOutputType,
    is_leaf_type,
    is_list_type,
    is_non_null_type,
    is_object_type,
)

from .bubble import Bubble


class LeafCompleter(NamedTuple):
    """Serialize a scalar/enum value."""

    leaf_type: GraphQLLeafType


class ObjectCompleter(NamedTuple):
    """Recurse into an object's sub-selection (a child bucket)."""

    object_type: GraphQLObjectType
    child_plan: Any  # ObjectPlan (avoids a circular import with plan.py)


class ListCompleter(NamedTuple):
    """Fan out into per-item completion with the inner (item) completer."""

    item_completer: "Completer"


class AbstractCompleter(NamedTuple):
    """Resolve an interface/union value's runtime object type, then recurse.

    Unlike `ObjectCompleter`, an abstract field can resolve to a *different*
    concrete object type per value, so no single child plan can be pre-computed.
    At completion time each value's runtime type is resolved (via the abstract
    type's `resolve_type` or the context `type_resolver`), validated, and values
    are grouped by concrete type; each group is collected + planned (cached) and
    executed as its own bucket. `field_nodes` is carried so subfields can be
    collected per concrete type at runtime.
    """

    abstract_type: GraphQLAbstractType
    field_nodes: Any
    # concrete-type -> ObjectPlan cache, filled lazily during completion
    plan_cache: dict


class NonNullCompleter(NamedTuple):
    """Forbid a null inner value; a None inner result becomes a Bubble."""

    inner: "Completer"


# a completer is one of the five above (leaf / object / abstract / list / non-null).
# Those five kinds exhaust GraphQLOutputType (scalar+enum = leaf, interface+union =
# abstract, object, list, non-null), so every output type has a native completer.
Completer = Any


class HoistBridge(NamedTuple):
    """The channel carrying a parent bucket's HOISTED columns down to a child bucket.

    When a step is hoisted to a shallower (parent) layer, its column is produced in the
    PARENT bucket but a child bucket still needs it (keyed by the hoisted step id). This
    bridge threads that down through the value-completion recursion so `run_object_children`
    can seed the child layer from `parent_store` (the parent bucket store, indexed by parent
    bucket position) projected by `value_owner` (which parent bucket position each completed
    `values[j]` belongs to).

    `value_owner[j]` is the parent BUCKET position of completion value `j` — it follows the
    list flatten / object scatter so the hoisted column (produced at parent bucket granularity)
    is expanded to each child position correctly. It is `None` (the default everywhere) when no
    field in this descent has any hoisted-out column, so every non-hoisting path is
    byte-identical (the bridge is never built and never read).
    """

    parent_store: dict
    value_owner: List[int]


def build_completer(
    context, return_type: GraphQLOutputType, field_nodes
) -> Completer:
    """Build the completer tree for a field's return type.

    Lists, non-nulls, leaves, objects and abstract (interface/union) types are all
    handled natively, nested to arbitrary depth — these five kinds exhaust
    `GraphQLOutputType`, so the function always returns a completer. Object
    completers carry their `object_type` only; the planner fills `child_plan`
    afterwards (it needs the field_nodes to collect subfields). Abstract completers
    carry `field_nodes` directly because their per-concrete-type plans are built
    lazily at runtime.
    """
    if is_non_null_type(return_type):
        return NonNullCompleter(
            inner=build_completer(context, return_type.of_type, field_nodes)
        )
    if is_list_type(return_type):
        return ListCompleter(
            item_completer=build_completer(context, return_type.of_type, field_nodes)
        )
    if is_leaf_type(return_type):
        return LeafCompleter(leaf_type=return_type)
    if is_object_type(return_type):
        return ObjectCompleter(object_type=return_type, child_plan=None)
    # the only remaining GraphQLOutputType kind: interface / union (abstract).
    return AbstractCompleter(
        abstract_type=return_type, field_nodes=field_nodes, plan_cache={}
    )


def find_object_completer(completer: Completer) -> Optional[ObjectCompleter]:
    """Return the ObjectCompleter at the leaf of a wrapper chain, if any.

    The planner uses this to attach the child plan to an object completer that may
    be wrapped in List/NonNull layers (e.g. `[DataType!]!`).
    """
    if isinstance(completer, ObjectCompleter):
        return completer
    if isinstance(completer, NonNullCompleter):
        return find_object_completer(completer.inner)
    if isinstance(completer, ListCompleter):
        return find_object_completer(completer.item_completer)
    return None


def complete_values(
    context,
    completer: Completer,
    values: List[Any],
    paths: List[Path],
    infos: List[Any],
    field_nodes,
    field_label: str,
    bridge: Optional["HoistBridge"] = None,
):
    """Complete a batch of raw values through `completer`.

    Returns a list (one per input) of completed values — serialized leaf / list /
    object dict, None, or a `Bubble` (an unresolved non-null violation to be caught
    by an enclosing nullable boundary or by the field's parent object). Returns a
    coroutine resolving to that list when any completion is awaitable.

    `values[i]` may be a raw resolver value, an Exception (resolver raised /
    coercion error — re-raised and located here), Undefined, or an awaitable.
    `infos[i]` is the field's `GraphQLResolveInfo` for value `i`; abstract-type
    resolution needs it (and graphql-core reuses the field's single info for every
    list item, so list expansion replicates the parent info per item).

    `bridge` is the :class:`HoistBridge` carrying a parent bucket's hoisted columns
    down to the object child bucket (it follows list flatten / object scatter so each
    child position seeds the right hoisted value). It is `None` by default and on every
    non-hoisting path, so completion is byte-identical when nothing was hoisted.
    """
    if isinstance(completer, LeafCompleter):
        return complete_leaf_values(context, completer, values, paths, field_nodes)
    if isinstance(completer, NonNullCompleter):
        return complete_non_null_values(
            context, completer, values, paths, infos, field_nodes, field_label, bridge
        )
    if isinstance(completer, ListCompleter):
        return complete_list_values(
            context, completer, values, paths, infos, field_nodes, field_label, bridge
        )
    if isinstance(completer, ObjectCompleter):
        return complete_object_values(
            context, completer, values, paths, infos, field_nodes, bridge
        )
    if isinstance(completer, AbstractCompleter):
        return complete_abstract_values(
            context, completer, values, paths, infos, field_nodes
        )
    raise TypeError(f"unknown completer: {completer!r}")


def resolve_awaitable_values(context, values: List[Any]):
    """Await awaitable raw values in-place, capturing rejections as exceptions.

    A rejected awaitable is stored as its exception value so downstream completion
    locates it as a field/item error, exactly as graphql-core's `await_completed`.
    Returns a coroutine to await, or None when nothing was awaitable.
    """
    from asyncio import gather

    idx = [i for i, v in enumerate(values) if context.is_awaitable(v)]
    if not idx:
        return None

    async def run():
        resolved = await gather(*(values[i] for i in idx), return_exceptions=True)
        for i, r in zip(idx, resolved):
            values[i] = r

    return run()


# ---------------------------------------------------------------------------
# leaf
# ---------------------------------------------------------------------------


def complete_leaf_values(context, completer, values, paths, field_nodes):
    pending = resolve_awaitable_values(context, values)
    if pending is not None:

        async def finish():
            await pending
            return [
                complete_single_leaf(context, completer.leaf_type, v, paths[i], field_nodes)
                for i, v in enumerate(values)
            ]

        return finish()
    return [
        complete_single_leaf(context, completer.leaf_type, v, paths[i], field_nodes)
        for i, v in enumerate(values)
    ]


def complete_single_leaf(context, leaf_type, value, path, field_nodes):
    """Serialize one leaf value, surfacing any error as a propagating Bubble.

    A leaf is not itself a catching boundary (graphql-core has no try-block around
    `complete_leaf_value`): an error here becomes a `Bubble` carrying the located
    error, which propagates up until the nearest *catching* boundary (a nullable
    list item or the field write) appends it. A null value passes through as plain
    None (an enclosing NonNull converts it; otherwise it is a legitimate null).
    """
    if isinstance(value, Exception):
        return Bubble(located_error(value, field_nodes, path.as_list()))
    if value is None or value is Undefined:
        return None
    try:
        return serialize_leaf(leaf_type, value)
    except (GraphQLError, TypeError, ValueError) as raw_error:
        return Bubble(located_error(raw_error, field_nodes, path.as_list()))


def serialize_leaf(leaf_type: GraphQLLeafType, value: Any):
    serialized = leaf_type.serialize(value)
    if serialized is Undefined or serialized is None:
        raise TypeError(
            f"Expected `{inspect(leaf_type)}.serialize({inspect(value)})`"
            f" to return non-nullable value, returned: {inspect(serialized)}"
        )
    return serialized


# ---------------------------------------------------------------------------
# non-null
# ---------------------------------------------------------------------------


def complete_non_null_values(
    context, completer, values, paths, infos, field_nodes, field_label, bridge=None
):
    """Complete the inner type, then forbid None: a None inner becomes a Bubble.

    The Bubble carries the located cannot-return-null error built at *this*
    boundary's path. An inner value that is already a Bubble propagates unchanged
    (its originating error is preserved). `bridge` (the hoist channel) passes through to
    the inner completer unchanged — a NonNull wrapper does not reshape the value batch.
    """
    inner = complete_values(
        context, completer.inner, values, paths, infos, field_nodes, field_label, bridge
    )

    if context.is_awaitable(inner):

        async def finish():
            return [
                enforce_non_null(r, paths[i], field_nodes, field_label)
                for i, r in enumerate(await inner)
            ]

        return finish()
    return [
        enforce_non_null(r, paths[i], field_nodes, field_label)
        for i, r in enumerate(inner)
    ]


def enforce_non_null(completed, path, field_nodes, field_label):
    if isinstance(completed, Bubble):
        return completed
    if completed is None:
        error = located_error(
            non_null_error(field_label), field_nodes, path.as_list()
        )
        return Bubble(error)
    return completed


def non_null_error(field_label: str) -> TypeError:
    """The synthetic cannot-return-null error, keyed by the field's own label.

    `field_label` is "ParentType.fieldName" of the field whose return type is
    non-null (its `info`), matching graphql-core even when the failing value is a
    list item.
    """
    return TypeError(f"Cannot return null for non-nullable field {field_label}.")


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def complete_list_values(
    context, completer, values, paths, infos, field_nodes, field_label, bridge=None
):
    """Complete a batch of list-typed values.

    Each value must be iterable (str/dict/Mapping rejected like graphql-core);
    its items are flattened into one batch, completed once through the item
    completer, then re-collected per source list in positional order. A per-item
    Bubble is caught when the item type is nullable (error appended, item -> None)
    or propagated (the whole list -> Bubble) when the item type is non-null.

    Async raw values and async iterables are drained/awaited first, then the
    synchronous flatten+complete path runs; awaitable items are handled by the
    item completer's own awaiting.

    `bridge` (the hoist channel) is threaded into the flatten so each item carries the
    parent BUCKET position of its source list — a list reshapes the value batch, so the
    hoisted column must be re-projected per flattened item.
    """
    pending = prepare_list_inputs(context, values)
    if pending is not None:

        async def finish_inputs():
            await pending
            built = build_list_results(
                context, completer, values, paths, infos, field_nodes, field_label, bridge
            )
            if context.is_awaitable(built):
                return await built
            return built

        return finish_inputs()
    return build_list_results(
        context, completer, values, paths, infos, field_nodes, field_label, bridge
    )


def prepare_list_inputs(context, values):
    """Await awaitable list values and drain async iterables into lists, in place."""
    from asyncio import gather

    work = []
    targets = []
    for i, v in enumerate(values):
        if context.is_awaitable(v):
            targets.append(i)
            work.append(("await", v))
        elif not is_iterable(v) and isinstance(v, AsyncIterable):
            targets.append(i)
            work.append(("drain", v))
    if not work:
        return None

    async def settle(kind, v):
        if kind == "await":
            try:
                awaited = await v
            except (GraphQLError, TypeError, ValueError, RuntimeError) as exc:
                return exc
            if not is_iterable(awaited) and isinstance(awaited, AsyncIterable):
                return [item async for item in awaited]
            return awaited
        return [item async for item in v]

    async def run():
        resolved = await gather(*(settle(k, v) for k, v in work))
        for i, r in zip(targets, resolved):
            values[i] = r

    return run()


def build_list_results(
    context, completer, values, paths, infos, field_nodes, field_label, bridge=None
):
    """Flatten items into one batch, complete, then re-collect per list."""
    item_values: List[Any] = []
    item_paths: List[Path] = []
    # parallel to item_values: each list item reuses its source list's field info
    # (graphql-core threads the field's single `info` through list completion)
    item_infos: List[Any] = []
    # hoist channel: parallel to item_values, the parent BUCKET position of each item's
    # source list (carried from `bridge.value_owner[i]`) so the item-level bridge can project
    # the hoisted column per flattened item. Built only when a bridge is present.
    item_owner: List[int] = [] if bridge is not None else None
    # spans[i] is (start, length) into item_values for source list i, the sentinel
    # ("null",) when the value is null, or None when the value is invalid (with the
    # located error stashed in `invalid[i]`)
    spans: List[Any] = []
    invalid: List[Optional[Any]] = [None] * len(values)

    for i, value in enumerate(values):
        if isinstance(value, Exception):
            invalid[i] = located_error(value, field_nodes, paths[i].as_list())
            spans.append(None)
            continue
        if value is None or value is Undefined:
            spans.append(("null",))
            continue
        if not is_iterable(value):
            err = GraphQLError(
                "Expected Iterable, but did not find one for field"
                f" '{field_label}'."
            )
            invalid[i] = located_error(err, field_nodes, paths[i].as_list())
            spans.append(None)
            continue
        start = len(item_values)
        for index, item in enumerate(value):
            item_values.append(item)
            item_paths.append(paths[i].add_key(index, None))
            item_infos.append(infos[i])
            if item_owner is not None:
                item_owner.append(bridge.value_owner[i])
        spans.append((start, len(item_values) - start))

    item_bridge = (
        bridge._replace(value_owner=item_owner) if bridge is not None else None
    )
    completed_items = complete_values(
        context,
        completer.item_completer,
        item_values,
        item_paths,
        item_infos,
        field_nodes,
        field_label,
        item_bridge,
    )

    if context.is_awaitable(completed_items):

        async def finish():
            return collect_lists(
                context, completer, await completed_items, spans, invalid
            )

        return finish()
    return collect_lists(context, completer, completed_items, spans, invalid)


def collect_lists(context, completer, completed_items, spans, invalid):
    """Re-collect completed items back into per-source lists, handling item bubbles."""
    item_is_non_null = isinstance(completer.item_completer, NonNullCompleter)
    results: List[Any] = []
    for i, span in enumerate(spans):
        if span is None:
            # invalid list (non-iterable / resolver error): the list level is not a
            # catching boundary, so the located error bubbles to the field/item
            results.append(Bubble(invalid[i]))
            continue
        if span == ("null",):
            results.append(None)
            continue
        start, length = span
        out_list: List[Any] = []
        bubbled = None
        for k in range(length):
            item = completed_items[start + k]
            if isinstance(item, Bubble):
                if item_is_non_null:
                    # item ! violation bubbles to the list: the whole list nulls;
                    # the error was located at the item path already
                    bubbled = item
                    break
                context.errors.append(item.error)
                out_list.append(None)
                continue
            out_list.append(item)
        results.append(bubbled if bubbled is not None else out_list)
    return results


# ---------------------------------------------------------------------------
# object
# ---------------------------------------------------------------------------


def complete_object_values(context, completer, values, paths, infos, field_nodes, bridge=None):
    """Complete a batch of object values by recursing into the child bucket.

    `bridge` (the hoist channel) carries the parent bucket's hoisted columns + the per-value
    parent-bucket-owner map down to `run_object_children`, which seeds the child layer with the
    hoisted columns (so a hoisted step is read, not re-run, in the child bucket).
    """
    pending = resolve_awaitable_values(context, values)
    if pending is not None:

        async def finish_inputs():
            await pending
            built = run_object_bucket(context, completer, values, paths, field_nodes, bridge)
            if context.is_awaitable(built):
                return await built
            return built

        return finish_inputs()
    return run_object_bucket(context, completer, values, paths, field_nodes, bridge)


def run_object_bucket(context, completer, values, paths, field_nodes, bridge=None):
    object_type = completer.object_type
    child_objs: List[Any] = []
    child_paths: List[Path] = []
    origin: List[int] = []
    results: List[Any] = [None] * len(values)
    # parallel to child_objs: the is_type_of result (bool or awaitable) per object,
    # only populated when the object type defines an is_type_of guard
    guard: List[Any] = []

    for i, value in enumerate(values):
        if isinstance(value, Exception):
            # not a catching boundary: the located error bubbles to the field/item
            results[i] = Bubble(located_error(value, field_nodes, paths[i].as_list()))
            continue
        if value is None or value is Undefined:
            results[i] = None
            continue
        child_objs.append(value)
        child_paths.append(paths[i])
        origin.append(i)
        if object_type.is_type_of is not None:
            info = context.build_resolve_info(
                # any field def of this object works for info; use the first field's
                info_field_def(object_type),
                field_nodes,
                object_type,
                paths[i],
            )
            guard.append(object_type.is_type_of(value, info))

    if not child_objs:
        return results

    if guard and needs_await(context, guard):

        async def finish_guarded():
            resolved = await gather_guard(context, guard)
            return run_object_children(
                context, completer, child_objs, child_paths, origin, results, field_nodes, resolved, bridge
            )

        return finish_guarded()

    return run_object_children(
        context, completer, child_objs, child_paths, origin, results, field_nodes, guard or None, bridge
    )


def info_field_def(object_type):
    """Any field def of an object type, for building a resolve info for is_type_of."""
    return next(iter(object_type.fields.values()))


def needs_await(context, items) -> bool:
    return any(context.is_awaitable(x) for x in items)


def gather_guard(context, guard):
    from asyncio import gather

    async def run():
        return await gather(
            *(g if context.is_awaitable(g) else _wrap(g) for g in guard)
        )

    return run()


async def _wrap(value):
    return value


def run_object_children(
    context, completer, child_objs, child_paths, origin, results, field_nodes, guard, bridge=None
):
    """Filter out is_type_of failures (as Bubbles), then recurse into the bucket.

    `bridge` (the hoist channel) seeds the child layer with any columns the planner HOISTED
    out of it: a hoisted step's column was produced once in a shallower (parent) bucket, so
    here we project it to each kept child position via the parent-bucket-owner map and pass it
    as `parent_store` to the child `execute_object_plan`. The child layer's `ordered_steps`
    already exclude those step ids (`populate_layers` added them to the boundary), so the step
    is READ from the seed, never re-executed in the child bucket — the fire-once guarantee. No
    bridge (the default / no hoisted-out step) => `parent_store` stays None and the call is
    byte-identical.
    """
    from .execute import execute_object_plan

    keep_objs: List[Any] = []
    keep_paths: List[Path] = []
    keep_origin: List[int] = []
    for k in range(len(child_objs)):
        if guard is not None and not guard[k]:
            error = located_error(
                invalid_return_type_error(completer.object_type, child_objs[k], field_nodes),
                field_nodes,
                child_paths[k].as_list(),
            )
            results[origin[k]] = Bubble(error)
            continue
        keep_objs.append(child_objs[k])
        keep_paths.append(child_paths[k])
        keep_origin.append(origin[k])

    if not keep_objs:
        return results

    # test-only seam (no-op in production): exposes this hop's child seed + index map so a
    # test can prove the child layer is runnable from (parent value column, keep_origin)
    # alone — keep_objs == [parent_column[o] for o in keep_origin] element-for-element.
    capture = getattr(context, "_grafast_capture_keep_origin", None)
    if capture is not None:
        capture(completer, keep_objs, keep_origin, keep_paths)

    child_parent_store = build_hoist_parent_store(
        completer.child_plan, bridge, keep_origin
    )

    child_results = execute_object_plan(
        context, completer.child_plan, keep_objs, keep_paths, child_parent_store
    )

    if context.is_awaitable(child_results):

        async def finish():
            for k, child in enumerate(await child_results):
                results[keep_origin[k]] = child
            return results

        return finish()
    for k, child in enumerate(child_results):
        results[keep_origin[k]] = child
    return results


def invalid_return_type_error(object_type, result, field_nodes) -> GraphQLError:
    return GraphQLError(
        f"Expected value of type '{object_type.name}' but got: {inspect(result)}.",
        field_nodes,
    )


def build_hoist_parent_store(child_plan, bridge, keep_origin):
    """Build the child bucket's `parent_store` for the steps hoisted OUT of its layer.

    For each step id in `child_plan.layer.hoisted_out_ids`, the planner relocated the step to
    a shallower (parent) layer; its column lives in the PARENT bucket store (`bridge.parent_store`,
    indexed by parent bucket position). We project it to the kept child positions: kept child k
    came from completion value `keep_origin[k]`, whose parent bucket position is
    `bridge.value_owner[keep_origin[k]]`. The resulting column is element-for-element aligned with
    `keep_objs`, so the child layer reads each hoisted step's value at its own bucket position
    without re-running the step. Returns `None` when nothing was hoisted out (the default path —
    `run_layer` then seeds only the `parent_step` boundary, byte-identical).
    """
    hoisted_out = child_plan.layer.hoisted_out_ids
    if not hoisted_out or bridge is None:
        return None
    value_owner = bridge.value_owner
    parent_store = bridge.parent_store
    child_store = {}
    for hid in hoisted_out:
        parent_column = parent_store[hid]
        child_store[hid] = [parent_column[value_owner[o]] for o in keep_origin]
    return child_store


# ---------------------------------------------------------------------------
# abstract (interface / union)
# ---------------------------------------------------------------------------


def complete_abstract_values(context, completer, values, paths, infos, field_nodes):
    """Complete a batch of interface/union values by resolving each runtime type.

    Raw awaitable values are awaited first (mirroring object completion). Then each
    value's runtime concrete type is resolved (the abstract type's `resolve_type`
    or the context `type_resolver`, sync or async), validated, and values are
    grouped by concrete type for batched recursion. A resolve_type/validation
    failure for a value becomes a `Bubble` located at that value's field path.
    """
    pending = resolve_awaitable_values(context, values)
    if pending is not None:

        async def finish_inputs():
            await pending
            built = resolve_abstract_bucket(
                context, completer, values, paths, infos, field_nodes
            )
            if context.is_awaitable(built):
                return await built
            return built

        return finish_inputs()
    return resolve_abstract_bucket(context, completer, values, paths, infos, field_nodes)


def resolve_abstract_bucket(context, completer, values, paths, infos, field_nodes):
    """Resolve each value's runtime type (sync or async), then dispatch by type.

    `runtimes[i]` holds the resolved type name, an Exception (resolve_type raised),
    or a sentinel for null/exception values that skip resolution.
    """
    abstract_type = completer.abstract_type
    resolve_type_fn = abstract_type.resolve_type or context.type_resolver

    runtimes: List[Any] = [None] * len(values)
    awaitable_idx: List[int] = []
    for i, value in enumerate(values):
        if isinstance(value, Exception) or value is None or value is Undefined:
            continue
        try:
            runtime = resolve_type_fn(value, infos[i], abstract_type)
        except Exception as raw_error:  # resolve_type / is_type_of raised
            runtimes[i] = raw_error
            continue
        if context.is_awaitable(runtime):
            awaitable_idx.append(i)
        runtimes[i] = runtime

    if awaitable_idx:

        async def finish():
            await settle_runtimes(context, runtimes, awaitable_idx)
            return dispatch_abstract(
                context, completer, values, paths, infos, field_nodes, runtimes
            )

        return finish()
    return dispatch_abstract(
        context, completer, values, paths, infos, field_nodes, runtimes
    )


def settle_runtimes(context, runtimes, awaitable_idx):
    """Await awaitable runtime types in place, capturing rejections as exceptions."""
    from asyncio import gather

    async def run():
        resolved = await gather(
            *(runtimes[i] for i in awaitable_idx), return_exceptions=True
        )
        for i, r in zip(awaitable_idx, resolved):
            runtimes[i] = r

    return run()


def dispatch_abstract(context, completer, values, paths, infos, field_nodes, runtimes):
    """Validate runtime types, group values by concrete type, recurse per group.

    Returns a results list (parallel to `values`) or a coroutine. Each value's
    concrete type is validated via the inherited `ensure_valid_runtime_type`; a
    failure (or a resolve_type exception) becomes a located `Bubble`. Values
    sharing a concrete type form one bucket so the child selection set is planned
    once and resolved batched.
    """
    abstract_type = completer.abstract_type
    results: List[Any] = [None] * len(values)
    # concrete-type-name -> (objs, paths, origin-indices) for batched recursion
    groups: dict = {}

    for i, value in enumerate(values):
        if isinstance(value, Exception):
            results[i] = Bubble(
                located_error(value, field_nodes, paths[i].as_list())
            )
            continue
        if value is None or value is Undefined:
            results[i] = None
            continue
        runtime = runtimes[i]
        if isinstance(runtime, Exception):
            results[i] = Bubble(
                located_error(runtime, field_nodes, paths[i].as_list())
            )
            continue
        try:
            object_type = context.ensure_valid_runtime_type(
                runtime, abstract_type, field_nodes, infos[i], value
            )
        except GraphQLError as raw_error:
            results[i] = Bubble(
                located_error(raw_error, field_nodes, paths[i].as_list())
            )
            continue
        bucket = groups.setdefault(object_type.name, ([], [], [], object_type))
        bucket[0].append(value)
        bucket[1].append(paths[i])
        bucket[2].append(i)

    if not groups:
        return results

    # test-only seam (no-op in production): exposes each concrete-type group's seed + index
    # map (the group `origin` indexing the abstract field's own value column) so a test can
    # prove the group layer is runnable from (abstract value column, origin) alone.
    capture = getattr(context, "_grafast_capture_group_origin", None)

    pending = []
    for objs, group_paths, origin, object_type in groups.values():
        child_plan = abstract_child_plan(context, completer, object_type)
        if capture is not None:
            capture(object_type, child_plan, objs, origin, group_paths, values)
        child_results = execute_object_plan_for_group(
            context, child_plan, objs, group_paths
        )
        if context.is_awaitable(child_results):
            pending.append((child_results, origin))
        else:
            for k, child in enumerate(child_results):
                results[origin[k]] = child

    if not pending:
        return results

    async def finish():
        from asyncio import gather

        gathered = await gather(*(cr for cr, _ in pending))
        for (_, origin), child_list in zip(pending, gathered):
            for k, child in enumerate(child_list):
                results[origin[k]] = child
        return results

    return finish()


def abstract_child_plan(context, completer, object_type):
    """Build (and cache) the ObjectPlan for one concrete type under this field.

    Subfields are collected with the concrete `object_type` (so type-conditioned
    fragments resolve), then planned as a SELF-CONTAINED subtree: a fresh `RootStep`
    seeds this concrete-type group's row objects and every plan-resolver field (incl.
    nested pg relations) hangs its step off it, exactly as `plan_operation` does for the
    operation root. The subtree's DAG is deduplicated + remapped so the executor seeds
    `child_plan.layer.parent_step` (the RootStep) with the group's objects and runs the steps
    once per group. Cached per concrete type on the completer so repeated buckets of the
    same type reuse the plan.

    Why a fresh RootStep per concrete type rather than threading the field's outer plan:
    an abstract value resolves to a DIFFERENT concrete type per row, so no single child
    step DAG spans them — each concrete-type group is its own bucket. A concrete type with
    no plan-resolver fields (the legacy/resolver path, e.g. the conformance suite) plans to
    an empty DAG with a RootStep parent, so the executor's bucket-step run is a no-op and
    the legacy per-parent resolver path is entirely unaffected.
    """
    from .core_steps import RootStep
    from .dag import Plan
    from .plan import LayerReason, finalize_plan, plan_object

    cached = completer.plan_cache.get(object_type.name)
    if cached is not None:
        return cached

    plan = Plan()
    # mirror `plan_operation`: an abstract concrete-type subtree is its own finalize
    # path (own RootStep + DAG), so it must carry the same plan-level inlining /
    # placeholder / caching decisions off the context's config — else a relation under a
    # polymorphic field would never be considered for folding even with inlining enabled,
    # and (the placeholder gap) a `$variable` arg on a field UNDER a concrete type would see
    # empty provenance, so `FieldArgs.is_variable` would be False and the host would inline it
    # as a literal even with placeholders enabled. Thread all three so a concrete-type subtree
    # plans EXACTLY as the operation root does.
    config = context.grafast_config
    plan.inline_relations = config.inline_relations
    plan.placeholders = config.placeholders
    plan.cache_plans = config.cache_plans
    # carry the hoist decision too so a concrete-type subtree finalizes EXACTLY as the
    # operation root does. A subtree's top layer reason is ROOT (its own RootStep), so the
    # hoist pass naturally finds no shallower layer to lift into and is inert here; threading
    # it keeps parity and future-proofs nested subtrees. `is_mutation` stays False — an
    # abstract subtree is only reached from a query/event walk, never a mutation root.
    plan.hoist = config.hoist
    # carry the incremental decision so a concrete-type subtree under an abstract field
    # partitions @defer / reads @stream EXACTLY as the operation root does. Off (3.2, or a
    # 3.3 op without incremental directives) => the legacy collection seam runs, byte-identical.
    plan.incremental = getattr(context, "_grafast_incremental", False)
    root_step = RootStep()
    plan.add_step(root_step)
    from .plan import collect_subfields_partitioned

    sub_fields, sub_details, child_deferred = collect_subfields_partitioned(
        context, plan, object_type, completer.field_nodes
    )
    child_plan = plan_object(
        context,
        object_type,
        sub_fields,
        parent_step=root_step,
        plan=plan,
        reason=LayerReason.ROOT,
        deferred=child_deferred,
        details_map=sub_details,
    )
    child_plan = finalize_plan(plan, child_plan)

    completer.plan_cache[object_type.name] = child_plan
    return child_plan


def execute_object_plan_for_group(context, child_plan, objs, group_paths):
    from .execute import execute_object_plan

    return execute_object_plan(context, child_plan, objs, group_paths)
