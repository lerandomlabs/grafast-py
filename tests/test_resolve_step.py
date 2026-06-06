"""Tests for the resolver path flowing through the operation step plan.

The resolver-adapter path no longer builds a bespoke per-parent mini-DAG: every
field carries a `FieldPlan.step`, and a plain-resolver field's step is a `ResolveStep`
that lives in the operation plan and runs once over the bucket through the shared
`run_steps` executor. This test confirms a full query through `GrafastExecutionContext`
produces output byte-for-byte identical to stock graphql-core — the regression
tripwire that conformance can't itself exercise via the gtests directory. The
step-in-layer and info.path round-trip assertions live in `test_resolver_unification.py`.
"""

from graphql import (
    GraphQLField,
    GraphQLList,
    GraphQLNonNull,
    GraphQLObjectType,
    GraphQLSchema,
    GraphQLString,
    execute,
    parse,
)
from graphql.execution.execute import ExecutionContext

from grafast_py import GrafastExecutionContext


PEOPLE = {
    "1": {"name": "Luke", "friend_ids": ["2", "3"]},
    "2": {"name": "Han", "friend_ids": ["1", "3"]},
    "3": {"name": "Leia", "friend_ids": ["1", "2"]},
}


def make_schema(counts):
    def resolve_name(person, info):
        counts["name"] += 1
        return person["name"]

    def resolve_friends(person, info):
        counts["friends"] += 1
        return [PEOPLE[fid] for fid in person["friend_ids"]]

    def resolve_people(root, info):
        return list(PEOPLE.values())

    person_type = GraphQLObjectType(
        "Person",
        lambda: {
            "name": GraphQLField(GraphQLNonNull(GraphQLString), resolve=resolve_name),
            "friends": GraphQLField(GraphQLList(person_type), resolve=resolve_friends),
        },
    )
    return GraphQLSchema(
        query=GraphQLObjectType(
            "Query",
            {"people": GraphQLField(GraphQLList(person_type), resolve=resolve_people)},
        )
    )


QUERY = "{ people { name friends { name } } }"


def test_grafast_output_matches_stock_graphql_core():
    grafast_schema = make_schema({"name": 0, "friends": 0})
    stock_schema = make_schema({"name": 0, "friends": 0})

    grafast = execute(
        grafast_schema,
        parse(QUERY),
        execution_context_class=GrafastExecutionContext,
    )
    stock = execute(stock_schema, parse(QUERY), execution_context_class=ExecutionContext)

    assert grafast.errors is None
    assert grafast.data == stock.data
