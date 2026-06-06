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

    True only on 3.3+. P6 does not branch delivery on this; a later phase (P7) consumes it
    to opt into ``experimental_execute_incrementally`` for deferred/streamed payloads.
    """
    return not IS_32
