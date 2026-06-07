"""Version-probed seams over the parts of graphql-core that differ between 3.2 and 3.3.

The engine reuses graphql-core's version-STABLE frontend untouched (parse, validate,
the type system, the `values` coercion helpers, leaf serialize, `GraphQLResolveInfo`,
`located_error`, `ExecutionResult`). Only a handful of internals moved or changed shape
between the 3.2 line and the 3.3 alpha line; this module isolates exactly those so the
rest of the engine never imports a version-specific symbol directly.

The differences P6 has to bridge (all verified empirically against 3.2.8 and 3.3.0a12):

- ``get_field_def`` is a module function in ``graphql.execution.execute`` on 3.2 but was
  removed on 3.3, where the equivalent logic lives on ``GraphQLSchema.get_field``.
- ``collect_fields``/``collect_subfields`` return a plain ``{response_name: [FieldNode]}``
  map on 3.2 but a ``CollectedFields`` (whose ``grouped_field_set`` holds ``FieldDetails``
  for @defer bookkeeping) on 3.3, and take the operation rather than a selection set.
- ``build_response`` SORTS errors deterministically on 3.2; 3.3's ``build_data_response``
  does not. We own the sort on both versions so output is byte-identical across versions.
- ``ensure_valid_runtime_type`` takes raw ``FieldNode``s on 3.2 but ``FieldDetails`` on 3.3
  (it calls ``to_nodes`` on them), so the inherited method cannot be fed raw nodes on 3.3.

The probe is FEATURE-based, never a version-string parse: ``get_field_def`` is present on
3.2's execute module and absent on 3.3.
"""

import importlib
from typing import Any, Dict, List

from graphql.error import GraphQLError
from graphql.execution import ExecutionResult
from graphql.language import FieldNode
from graphql.pyutils import inspect as gql_inspect
from graphql.type import (
    GraphQLAbstractType,
    GraphQLObjectType,
    GraphQLResolveInfo,
    GraphQLSchema,
    is_object_type,
)


def _execute_module():
    """Return the real ``graphql.execution.execute`` submodule.

    ``graphql.execution`` re-exports a *function* named ``execute`` that shadows the
    sibling submodule, so ``import graphql.execution.execute as m`` binds the function,
    not the module. ``importlib.import_module`` resolves the true module past the shadow
    (the same dodge ``grafast_py.__init__._graphql_module`` uses for the install patch).
    """
    return importlib.import_module("graphql.execution.execute")


# Feature probe: ``get_field_def`` lives on the 3.2 execute module and was removed on 3.3.
# True on 3.2.x, False on the 3.3 alpha line. (verified against 3.2.8 / 3.3.0a12.)
IS_32: bool = hasattr(_execute_module(), "get_field_def")


if IS_32:
    from graphql.execution.execute import get_field_def as _get_field_def_32

    def get_field_def(
        schema: GraphQLSchema, parent_type: GraphQLObjectType, field_node: FieldNode
    ):
        """Look up a field definition, special-casing the three meta-fields.

        On 3.2 this delegates to graphql-core's own ``get_field_def``.
        """
        return _get_field_def_32(schema, parent_type, field_node)

else:

    def get_field_def(
        schema: GraphQLSchema, parent_type: GraphQLObjectType, field_node: FieldNode
    ):
        """Look up a field definition, special-casing the three meta-fields.

        On 3.3 ``get_field_def`` was removed from the execute module; the same logic
        (the ``__schema``/``__type``/``__typename`` introspection special-casing then a
        plain field lookup) lives on ``GraphQLSchema.get_field``.
        """
        return schema.get_field(parent_type, field_node.name.value)


def collect_root_fields(
    context, root_type: GraphQLObjectType, operation
) -> Dict[str, List[FieldNode]]:
    """Collect an operation's root fields as ``{response_name: [FieldNode]}``.

    This is the shape the planner consumes. On 3.3 ``collect_fields`` takes the
    operation (not its selection set) and returns a ``CollectedFields`` whose
    ``grouped_field_set`` maps to ``FieldDetails``; we unwrap each back to its raw node.
    """
    if IS_32:
        from graphql.execution.collect_fields import collect_fields

        return collect_fields(
            context.schema,
            context.fragments,
            context.variable_values,
            root_type,
            operation.selection_set,
        )

    from graphql.execution.collect_fields import collect_fields

    collected = collect_fields(
        context.schema,
        context.fragments,
        context.variable_values,
        root_type,
        operation,
    )
    return {
        response_name: [detail.node for detail in group]
        for response_name, group in collected.grouped_field_set.items()
    }


def collect_subfields(
    context, object_type: GraphQLObjectType, field_nodes: List[FieldNode]
) -> Dict[str, List[FieldNode]]:
    """Collect the subfields of ``field_nodes`` under ``object_type`` as a node map.

    Mirrors graphql-core's ``ExecutionContext.collect_subfields`` cache (an id-keyed map
    on the context) so repeated list elements don't re-collect, but returns the
    version-independent ``{response_name: [FieldNode]}`` map the planner wants. On 3.3 the
    field group is ``FieldDetails`` and the result is a ``CollectedFields``, so we wrap the
    raw nodes going in and unwrap the grouped set coming out.

    The cache lives under a grafast-owned attribute (``_grafast_subfields_cache``), lazily
    created, rather than graphql-core's own — the 3.2 base names it ``_subfields_cache`` but
    the 3.3 base names it ``_relevant_sub_fields`` and stores a different (CollectedFields)
    shape, so reusing either would be version-fragile; owning the attribute keeps both the
    ``_PlanRunContext`` and the shim subclass on one cache regardless of version.
    """
    cache = getattr(context, "_grafast_subfields_cache", None)
    if cache is None:
        cache = {}
        context._grafast_subfields_cache = cache
    key = (
        (object_type, id(field_nodes[0]))
        if len(field_nodes) == 1
        else (object_type, *map(id, field_nodes))
    )
    cached = cache.get(key)
    if cached is not None:
        return cached

    if IS_32:
        from graphql.execution.collect_fields import collect_sub_fields

        result = collect_sub_fields(
            context.schema,
            context.fragments,
            context.variable_values,
            object_type,
            field_nodes,
        )
    else:
        from graphql.execution.collect_fields import FieldDetails, collect_subfields

        collected = collect_subfields(
            context.schema,
            context.fragments,
            context.variable_values,
            context.operation,
            object_type,
            [FieldDetails(node, None) for node in field_nodes],
        )
        result = {
            response_name: [detail.node for detail in group]
            for response_name, group in collected.grouped_field_set.items()
        }
    cache[key] = result
    return result


def collect_subfields_raw(
    context, object_type: GraphQLObjectType, field_nodes: List[FieldNode]
):
    """Expose the UN-unwrapped subfield collection for a later @defer-aware phase (P7).

    On 3.2 this is the same ``{response_name: [FieldNode]}`` map (3.2 has no defer
    bookkeeping); on 3.3 it returns the raw ``CollectedFields`` (whose ``grouped_field_set``
    holds ``FieldDetails`` carrying ``defer_usage``). P6 does not consume this — it exists
    so P7 can opt into incremental delivery without re-deriving the collection.
    """
    if IS_32:
        from graphql.execution.collect_fields import collect_sub_fields

        return collect_sub_fields(
            context.schema,
            context.fragments,
            context.variable_values,
            object_type,
            field_nodes,
        )

    from graphql.execution.collect_fields import FieldDetails, collect_subfields

    return collect_subfields(
        context.schema,
        context.fragments,
        context.variable_values,
        context.operation,
        object_type,
        [FieldDetails(node, None) for node in field_nodes],
    )


def collect_root_fields_raw(context, root_type: GraphQLObjectType, operation):
    """Expose the UN-unwrapped root-field collection for the @defer-aware phase (P7).

    On 3.2 this is a plain ``{response_name: [FieldNode]}`` map (no defer bookkeeping);
    on 3.3 it returns the raw ``CollectedFields`` (whose ``grouped_field_set`` holds
    ``FieldDetails`` carrying ``defer_usage`` plus the operation's ``new_defer_usages``).
    The non-incremental path keeps using :func:`collect_root_fields`; this is the feed
    P7's incremental driver partitions into initial vs deferred groups.
    """
    if IS_32:
        from graphql.execution.collect_fields import collect_fields

        return collect_fields(
            context.schema,
            context.fragments,
            context.variable_values,
            root_type,
            operation.selection_set,
        )

    from graphql.execution.collect_fields import collect_fields

    return collect_fields(
        context.schema,
        context.fragments,
        context.variable_values,
        root_type,
        operation,
    )


def collect_subfields_details(context, object_type: GraphQLObjectType, field_group):
    """Collect subfields of a field group, PRESERVING each field's ``defer_usage`` (P7, 3.3).

    Unlike :func:`collect_subfields_raw` (which wraps raw nodes with a None defer_usage), this
    feeds the field group's actual ``FieldDetails`` to upstream ``collect_subfields`` so each
    field's ``defer_usage`` is threaded into its subfields as the parent — the nested-defer /
    parent-payload-dedup behaviour. ``field_group`` may be a list of ``FieldDetails`` (the
    incremental path) or, defensively, raw ``FieldNode``s (wrapped with None defer_usage).
    Returns the raw ``CollectedFields``. 3.3-only (the planner gates on ``plan.incremental``).
    """
    from graphql.execution.collect_fields import FieldDetails, collect_subfields

    details = [
        d if isinstance(d, FieldDetails) else FieldDetails(d, None) for d in field_group
    ]
    return collect_subfields(
        context.schema,
        context.fragments,
        context.variable_values,
        context.operation,
        object_type,
        details,
    )


def build_resolve_info(
    context,
    field_def,
    field_nodes: List[FieldNode],
    parent_type: GraphQLObjectType,
    path,
) -> GraphQLResolveInfo:
    """Construct a ``GraphQLResolveInfo`` from a context-like object.

    The constructor takes the identical 12 positional arguments on 3.2 and 3.3 (verified),
    so this is version-independent; it works for both the ``_PlanRunContext`` and the
    ``GrafastExecutionContext`` shim because it reads only their common attributes.
    """
    return GraphQLResolveInfo(
        field_nodes[0].name.value,
        field_nodes,
        field_def.type,
        parent_type,
        path,
        context.schema,
        context.fragments,
        context.root_value,
        context.operation,
        context.variable_values,
        context.context_value,
        context.is_awaitable,
    )


def ensure_valid_runtime_type(
    context,
    runtime_type_name: Any,
    return_type: GraphQLAbstractType,
    field_nodes: List[FieldNode],
    info: GraphQLResolveInfo,
    result: Any,
) -> GraphQLObjectType:
    """Validate that an abstract type's runtime type name names a possible object type.

    A version-independent reimplementation of graphql-core's
    ``ExecutionContext.ensure_valid_runtime_type``: 3.2 takes raw ``FieldNode``s while 3.3
    takes ``FieldDetails`` (it calls ``to_nodes`` on them), so the inherited method cannot
    be fed the planner's raw nodes on 3.3. Inlining it here keeps the call site identical
    across versions and works for both context shapes (it reads only ``context.schema``).
    """
    if runtime_type_name is None:
        raise GraphQLError(
            f"Abstract type '{return_type.name}' must resolve"
            " to an Object type at runtime"
            f" for field '{info.parent_type.name}.{info.field_name}'."
            f" Either the '{return_type.name}' type should provide"
            " a 'resolve_type' function or each possible type should provide"
            " an 'is_type_of' function.",
            field_nodes,
        )

    if is_object_type(runtime_type_name):  # pragma: no cover
        raise GraphQLError(
            "Support for returning GraphQLObjectType from resolve_type was"
            " removed in GraphQL-core 3.2, please return type name instead."
        )

    if not isinstance(runtime_type_name, str):
        raise GraphQLError(
            f"Abstract type '{return_type.name}' must resolve"
            " to an Object type at runtime"
            f" for field '{info.parent_type.name}.{info.field_name}' with value"
            f" {gql_inspect(result)}, received '{gql_inspect(runtime_type_name)}'.",
            field_nodes,
        )

    runtime_type = context.schema.get_type(runtime_type_name)

    if runtime_type is None:
        raise GraphQLError(
            f"Abstract type '{return_type.name}' was resolved to a type"
            f" '{runtime_type_name}' that does not exist inside the schema.",
            field_nodes,
        )

    if not is_object_type(runtime_type):
        raise GraphQLError(
            f"Abstract type '{return_type.name}' was resolved"
            f" to a non-object type '{runtime_type_name}'.",
            field_nodes,
        )

    if not context.schema.is_sub_type(return_type, runtime_type):
        raise GraphQLError(
            f"Runtime Object type '{runtime_type.name}' is not a possible"
            f" type for '{return_type.name}'.",
            field_nodes,
        )

    return runtime_type


def make_result(
    data, errors: List[GraphQLError]
) -> ExecutionResult:
    """Package (data, errors) into an ``ExecutionResult``, owning the deterministic sort.

    Inlines 3.2's ``build_response`` error-sort on BOTH versions (3.3's
    ``build_data_response`` does not sort), so two executions of the same operation on
    different graphql-core versions produce a byte-identical ``.formatted`` payload.
    """
    if not errors:
        return ExecutionResult(data, None)
    errors.sort(
        key=lambda error: (error.locations or [], error.path or [], error.message)
    )
    return ExecutionResult(data, errors)


def supports_incremental() -> bool:
    """Whether the underlying graphql-core supports @defer/@stream incremental delivery.

    True only on 3.3+. P7 consumes it to opt into ``experimental_execute_incrementally``
    for deferred/streamed payloads and to gate the entire incremental driver off on 3.2,
    where the @defer/@stream directives do not exist (so the 3.2 path is byte-identical).
    """
    return not IS_32


def partition_defer(collected, parent_defer_usages=None):
    """Split a raw ``CollectedFields`` into (initial node map, deferred groups).

    A port of upstream ``build_execution_plan`` (graphql-core 3.3
    ``build_execution_plan.py``) over the engine's node-map shape: for each response key
    we compute its FILTERED defer-usage-set (``get_filtered_defer_usage_set`` — a key with
    ANY non-deferred member is "initial"; otherwise its set is its members minus those that
    have a parent already in the set). A key whose filtered set EQUALS ``parent_defer_usages``
    belongs to the level being executed now (initial here); the rest are bucketed by their
    defer-usage-set into deferred groups.

    Returns ``(initial, deferred)`` where ``initial`` is ``{response_name: [FieldNode]}``
    and ``deferred`` is a list of ``(defer_usage_set, {response_name: [FieldNode]}, ...)``.
    Each group's ``defer_usage_set`` is a ``frozenset`` of the (identity-keyed) DeferUsage
    objects, paired with the per-key node map.

    On 3.2 ``collected`` is already a plain node map (no defer info) → returns
    ``(node_map, [])`` so the 3.2 path is byte-identical.
    """
    if IS_32:
        return collected, []

    if parent_defer_usages is None:
        parent_defer_usages = frozenset()

    grouped_field_set = collected.grouped_field_set
    # the initial / deferred maps carry the full FieldDetails (node + defer_usage) so a field's
    # subfields are later collected WITH its defer-usage context (the nested-defer threading).
    initial: Dict[str, list] = {}
    # preserve first-seen response-key order across groups, keyed by the frozenset of
    # DeferUsage identities (RefSet-equivalent: DeferUsage is a NamedTuple but upstream
    # compares by identity via RefSet, so we key by id()).
    deferred_order: List[frozenset] = []
    deferred_map: Dict[frozenset, Dict[str, list]] = {}
    deferred_usages: Dict[frozenset, frozenset] = {}

    parent_key = frozenset(id(du) for du in parent_defer_usages)
    for response_key, field_group in grouped_field_set.items():
        filtered = get_filtered_defer_usage_set(field_group)
        key = frozenset(id(du) for du in filtered)
        if key == parent_key:
            initial[response_key] = list(field_group)
            continue
        if key not in deferred_map:
            deferred_order.append(key)
            deferred_map[key] = {}
            deferred_usages[key] = filtered
        deferred_map[key][response_key] = list(field_group)

    deferred = [
        (deferred_usages[key], deferred_map[key]) for key in deferred_order
    ]
    return initial, deferred


def build_execution_plan_groups(collected, parent_defer_usages=None):
    """Port of upstream ``build_execution_plan`` over the engine's group shape (P7, 3.3).

    Returns ``(initial_details, initial_nodes, new_groups, new_defer_usages)``:

    * ``initial_details`` — ``{response_name: [FieldDetails]}`` for the keys whose filtered
      defer-usage-set equals ``parent_defer_usages`` (executed now at THIS level).
    * ``initial_nodes`` — the same map projected to raw nodes (what the planner consumes).
    * ``new_groups`` — a list of ``(defer_usage_set, {response_name: [FieldDetails]})`` in
      first-seen defer-usage-set order, exactly like upstream's ``new_grouped_field_sets``
      (a ``RefMap[DeferUsageSet, GroupedFieldSet]``). ``defer_usage_set`` is a tuple of the
      actual DeferUsage objects (identity-meaningful), preserving first-add order.
    * ``new_defer_usages`` — the DeferUsage objects NEW at this level (``collected.new_defer_usages``),
      used to mint deferred-fragment records.

    A direct transcription of ``build_execution_plan``: for each response key compute its
    filtered defer-usage-set; if it equals the parent set the key is initial, else it is
    grouped under that set. Multiple keys sharing a filtered set form ONE group (the shared
    grouped-field-set that lets two overlapping fragments deliver a field once). 3.3-only.
    """
    grouped_field_set = collected.grouped_field_set
    parent_ids = (
        frozenset(id(du) for du in parent_defer_usages)
        if parent_defer_usages
        else frozenset()
    )

    initial_details: Dict[str, list] = {}
    # new_grouped_field_sets, keyed by the (order-preserving) defer-usage-set. We compare sets
    # by membership (RefSet-equivalent), so dedupe by the frozenset of identities but keep the
    # usage objects + first-seen order for emission/record minting.
    group_order: List[frozenset] = []
    group_map: Dict[frozenset, Dict[str, list]] = {}
    group_usages: Dict[frozenset, tuple] = {}

    for response_key, field_group in grouped_field_set.items():
        filtered = get_filtered_defer_usage_set(field_group)
        key = frozenset(id(du) for du in filtered)
        if key == parent_ids:
            initial_details[response_key] = list(field_group)
            continue
        if key not in group_map:
            group_order.append(key)
            group_map[key] = {}
            # preserve the actual usage objects in a stable order for record minting / paths.
            group_usages[key] = tuple(filtered)
        group_map[key][response_key] = list(field_group)

    initial_nodes = {
        response_name: [detail.node for detail in details]
        for response_name, details in initial_details.items()
    }
    new_groups = [(group_usages[key], group_map[key]) for key in group_order]
    new_defer_usages = list(getattr(collected, "new_defer_usages", []) or [])
    return initial_details, initial_nodes, new_groups, new_defer_usages


def nest_deferred_groups(deferred, parent_defer_usages=None):
    """Arrange this object level's deferred groups into a parent/child forest (P7, 3.3).

    ``deferred`` is the flat ``[(usage_set, field_map), ...]`` from :func:`partition_defer`.
    A group is a ROOT at this level when none of the deferred usages of ANY group at this
    level is the ``parent_defer_usage`` of this group's usages; otherwise it nests under the
    group owning its parent usage. Mirrors upstream's record graph
    (``DeferredFragmentRecord.parent`` + ``_promote_non_empty_to_root``): a nested @defer's
    pending is emitted only when its parent fragment completes.

    Returns ``[(usage_set, field_map, [children...]), ...]`` of ROOT groups (each child is the
    same 3-tuple shape), preserving first-seen order at each level.
    """
    if IS_32:
        return deferred
    # map a DeferUsage identity -> the group whose usage_set CONTAINS it (its owner).
    owner_of: Dict[int, int] = {}
    for index, (usage_set, _field_map) in enumerate(deferred):
        for du in usage_set:
            owner_of[id(du)] = index

    children: Dict[int, list] = {i: [] for i in range(len(deferred))}
    roots: List[int] = []
    for index, (usage_set, _field_map) in enumerate(deferred):
        parent_index = None
        for du in usage_set:
            parent = du.parent_defer_usage
            while parent is not None:
                if id(parent) in owner_of and owner_of[id(parent)] != index:
                    parent_index = owner_of[id(parent)]
                    break
                parent = parent.parent_defer_usage
            if parent_index is not None:
                break
        if parent_index is None:
            roots.append(index)
        else:
            children[parent_index].append(index)

    def build(index):
        usage_set, field_map = deferred[index]
        return (usage_set, field_map, [build(c) for c in children[index]])

    return [build(i) for i in roots]


def get_filtered_defer_usage_set(field_group):
    """Port of upstream ``get_filtered_defer_usage_set`` (build_execution_plan.py).

    Returns an ORDERED tuple of DeferUsage objects (identity-meaningful, first-seen order — a
    RefSet-equivalent) that govern this response key's group: empty if any member is
    non-deferred (the key is "initial"), otherwise the members' defer-usages minus any whose
    parent defer-usage is also present (so a child-defer key collapses to its nearest in-set
    ancestor — the dedup that makes a field already in a parent defer not re-emitted in the
    child). Order is preserved (unlike a set) because pending / defer-record minting order
    follows it (the wire id assignment order the suite asserts).
    """
    filtered: list = []
    seen = set()
    for field_detail in field_group:
        defer_usage = field_detail.defer_usage
        if defer_usage is None:
            return ()
        if id(defer_usage) not in seen:
            seen.add(id(defer_usage))
            filtered.append(defer_usage)

    keep = []
    for defer_usage in filtered:
        parent = defer_usage.parent_defer_usage
        pruned = False
        while parent is not None:
            if id(parent) in seen:
                pruned = True
                break
            parent = parent.parent_defer_usage
        if not pruned:
            keep.append(defer_usage)
    return tuple(keep)


class StreamError:
    """A @stream argument coercion error captured at plan time (surfaced as a field error).

    ``get_directive_values`` raises on a non-integer ``initialCount`` / non-string ``label``;
    upstream surfaces that as a located field error nulling the field rather than aborting the
    whole operation. The planner wraps the raised error here and the stream completer locates it.
    """

    __slots__ = ("error",)

    def __init__(self, error):
        self.error = error


def get_stream_usage(field_node: FieldNode, variable_values):
    """Return ``(initial_count, label)`` for a @stream'd list field, else ``None``.

    Mirrors upstream's stream-directive read: a list field carrying ``@stream`` (and not
    disabled by ``if: false``) is streamed; ``initialCount`` defaults to 0 when omitted.
    On 3.2 the @stream directive does not exist, so this is always ``None`` and the list
    path is byte-identical.
    """
    if IS_32:
        return None
    from graphql.execution.values import get_directive_values
    from graphql.type import GraphQLStreamDirective

    stream = get_directive_values(GraphQLStreamDirective, field_node, variable_values)
    if not stream or stream.get("if") is False:
        return None
    initial_count = stream.get("initialCount")
    if initial_count is None:
        initial_count = 0
    return initial_count, stream.get("label")
