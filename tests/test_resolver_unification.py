"""Phase B regression: the resolver path is unified into the operation step plan.

Every no-plan resolver field now carries a `FieldPlan.step` that is a `ResolveStep`
living in the operation plan (one uniform path; no hidden per-parent mini-DAG). These
tests pin the two properties the unification must preserve:

1. a plain-resolver field appears as a `ResolveStep` in its layer's `ordered_steps`,
   and EVERY field carries a step (`field_plan.step is not None`) — proving the
   `step is None` dual branch is gone.
2. a resolver reading `info.path` gets the correct FULL path chain per parent under
   batching across multiple parents — the deliberate divergence from upstream's
   degraded fake path (`prev: undefined`), which the graphql-core oracle asserts.
"""

from graphql import (
    GraphQLField,
    GraphQLList,
    GraphQLObjectType,
    GraphQLSchema,
    GraphQLString,
    execute,
    parse,
)
from graphql.execution.collect_fields import collect_fields

from grafast_py import GrafastExecutionContext
from grafast_py.plan import plan_operation
from grafast_py.steps import ResolveStep


def build_plan(schema, query: str):
    document = parse(query)
    operation = document.definitions[0]
    ctx = GrafastExecutionContext.build(schema, document)
    root_type = schema.query_type
    root_fields = collect_fields(
        ctx.schema, ctx.fragments, ctx.variable_values, root_type, operation.selection_set
    )
    object_plan = plan_operation(ctx, operation, root_type, root_fields)
    return ctx, object_plan


def test_no_plan_resolver_field_is_a_step_in_its_layer():
    """A plain-resolver field is a ResolveStep in its layer's ordered_steps."""
    schema = GraphQLSchema(
        query=GraphQLObjectType(
            "Query",
            {"greeting": GraphQLField(GraphQLString, resolve=lambda r, i: "hi")},
        )
    )

    _, object_plan = build_plan(schema, "{ greeting }")

    # every field carries a step — there is no `step is None` path anymore.
    for field_plan in object_plan.fields:
        assert field_plan.step is not None

    greeting = object_plan.fields[0]
    assert isinstance(greeting.step, ResolveStep)
    # one uniform path: the resolver step is in the layer's run set.
    assert greeting.step in object_plan.layer.ordered_steps
    assert greeting.step in object_plan.layer.run_steps
    # a resolver must never merge with another field's resolver, and it pulls the
    # per-invocation request context + paths via BucketExtra.
    assert greeting.step.dedupable is False
    assert greeting.step.wants_extra is True


def test_resolver_info_path_round_trips_under_batching():
    """A resolver reading info.path gets the full chain per parent across a bucket."""
    item_type = GraphQLObjectType(
        "Item",
        {
            "whereAmI": GraphQLField(
                GraphQLList(GraphQLString),
                resolve=lambda item, info: [str(p) for p in info.path.as_list()],
            )
        },
    )
    schema = GraphQLSchema(
        query=GraphQLObjectType(
            "Query",
            {
                "items": GraphQLField(
                    GraphQLList(item_type),
                    resolve=lambda root, info: [{}, {}, {}],
                )
            },
        )
    )

    result = execute(
        schema,
        parse("{ items { whereAmI } }"),
        execution_context_class=GrafastExecutionContext,
    )

    assert result.errors is None
    # three parents in one bucket; each gets its own FULL path chain.
    assert result.data == {
        "items": [
            {"whereAmI": ["items", "0", "whereAmI"]},
            {"whereAmI": ["items", "1", "whereAmI"]},
            {"whereAmI": ["items", "2", "whereAmI"]},
        ]
    }
