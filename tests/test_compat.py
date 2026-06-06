"""The version-probe seam: shape assertions that hold on whichever graphql-core is installed.

These assert the contract the rest of the engine relies on — the feature probe resolves,
the field-def / field-collection shims return the version-INDEPENDENT
``{response_name: [FieldNode]}`` shape the planner consumes, and the raw-collection accessor
(the P7 @defer opt-in) exists. The probe value is asserted against the installed graphql-core
version so the test is meaningful under BOTH the 3.2 baseline (.venv) and the 3.3 leg (.venv33).
"""

import graphql
from graphql import (
    GraphQLField,
    GraphQLObjectType,
    GraphQLSchema,
    GraphQLString,
    parse,
)
from graphql.language import FieldNode

from grafast_py import _compat


def _is_32() -> bool:
    return graphql.__version__.startswith("3.2")


def make_schema_and_context():
    """A tiny schema plus a _PlanRunContext over a one-field query for the seam shims."""
    from grafast_py.entry import _PlanRunContext, _gather_fragments, _select_operation

    Child = GraphQLObjectType(
        "Child", {"name": GraphQLField(GraphQLString, resolve=lambda r, i: "x")}
    )
    Query = GraphQLObjectType(
        "Query",
        {
            "child": GraphQLField(Child, resolve=lambda r, i: {"name": "x"}),
            "hello": GraphQLField(GraphQLString, resolve=lambda r, i: "hi"),
        },
    )
    schema = GraphQLSchema(query=Query)
    document = parse("{ hello child { name } }")
    operation = _select_operation(document, None)
    from graphql.execution.execute import default_field_resolver, default_type_resolver

    context = _PlanRunContext(
        schema=schema,
        fragments=_gather_fragments(document),
        root_value=None,
        context_value=None,
        operation=operation,
        variable_values={},
        field_resolver=default_field_resolver,
        type_resolver=default_type_resolver,
        errors=[],
        middleware_manager=None,
    )
    return schema, context, operation, Query, Child


def test_probe_matches_installed_version():
    assert _compat.IS_32 is _is_32()
    assert _compat.supports_incremental() is (not _is_32())


def test_get_field_def_resolves_field_and_meta_fields():
    schema, context, operation, Query, _Child = make_schema_and_context()
    hello_node = FieldNode(name=graphql.language.NameNode(value="hello"))
    field_def = _compat.get_field_def(schema, Query, hello_node)
    assert field_def is Query.fields["hello"]
    from graphql.type import TypeNameMetaFieldDef

    typename_node = FieldNode(name=graphql.language.NameNode(value="__typename"))
    meta = _compat.get_field_def(schema, Query, typename_node)
    assert meta is TypeNameMetaFieldDef


def test_collect_root_fields_returns_node_map():
    _schema, context, operation, _Query, _Child = make_schema_and_context()
    root_fields = _compat.collect_root_fields(context, _Query, operation)
    assert set(root_fields) == {"hello", "child"}
    for nodes in root_fields.values():
        assert isinstance(nodes, list)
        assert all(isinstance(node, FieldNode) for node in nodes)


def test_collect_subfields_returns_node_map_and_caches():
    _schema, context, _operation, _Query, Child = make_schema_and_context()
    root_fields = _compat.collect_root_fields(context, _Query, _operation_of(context))
    child_nodes = root_fields["child"]
    sub = _compat.collect_subfields(context, Child, child_nodes)
    assert set(sub) == {"name"}
    assert all(isinstance(node, FieldNode) for node in sub["name"])
    # second call hits the id-keyed cache and returns the same object.
    again = _compat.collect_subfields(context, Child, child_nodes)
    assert again is sub


def test_collect_subfields_raw_exists():
    _schema, context, _operation, _Query, Child = make_schema_and_context()
    root_fields = _compat.collect_root_fields(context, _Query, _operation_of(context))
    raw = _compat.collect_subfields_raw(context, Child, root_fields["child"])
    if _compat.IS_32:
        # 3.2 has no defer bookkeeping: raw is the plain node map.
        assert "name" in raw
    else:
        # 3.3 returns the un-unwrapped CollectedFields with a grouped_field_set.
        assert hasattr(raw, "grouped_field_set")
        assert "name" in raw.grouped_field_set


def _operation_of(context):
    return context.operation
