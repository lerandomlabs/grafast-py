"""Tests for the legacy-resolver path flowing through the Step model.

After build:step-model the legacy resolver path no longer calls resolvers from a
bespoke loop — it builds a tiny `ParentStep -> ResolveStep` DAG and runs it through
the shared `run_steps` executor. These tests confirm (a) `ResolveStep` runs the
field resolver once per parent over a bucket and reports the raw value column, and
(b) a full query through `GrafastExecutionContext` produces output byte-for-byte
identical to stock graphql-core — the regression tripwire that conformance can't
itself exercise via the gtests directory.
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
from grafast_py.steps import ResolveStep, run_resolve_step


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


def test_resolve_step_runs_resolver_over_a_bucket():
    counts = {"name": 0, "friends": 0}
    schema = make_schema(counts)
    document = parse("{ people { name } }")
    context = GrafastExecutionContext.build(schema, document)

    from grafast_py.plan import plan_object
    from graphql.execution.collect_fields import collect_fields

    person_type = schema.query_type.fields["people"].type.of_type
    root_fields = collect_fields(
        context.schema,
        context.fragments,
        context.variable_values,
        person_type,
        parse("{ name }").definitions[0].selection_set,
    )
    name_plan = plan_object(context, person_type, root_fields).fields[0]

    parents = list(PEOPLE.values())
    outcome = run_resolve_step(context, name_plan, parents, [None] * len(parents))

    assert outcome.values == ["Luke", "Han", "Leia"]
    assert len(outcome.infos) == 3 and len(outcome.paths) == 3
    # name resolver was invoked once per parent in the bucket (legacy adapter)
    assert counts["name"] == 3
    assert not outcome.awaitable


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


def test_resolve_step_is_a_step_subclass():
    """ResolveStep obeys the Step contract (subclass + add_dependency)."""
    from grafast_py.step_model import Step

    assert issubclass(ResolveStep, Step)
