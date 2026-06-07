"""The function seam: grafast_execute() and the GrafastExecutionContext shim share a core.

Both entry points run the SAME plan-then-execute pipeline (``run_planned_operation``), so
the same query/schema must yield identical ``{data, errors}`` whether driven through the
top-level ``grafast_execute()`` callable or through graphql-core's ``execute()`` with
``execution_context_class=GrafastExecutionContext`` (the exact host pattern the oogy
consumer uses). ``grafast_execute`` additionally returns a PLAIN ``graphql.ExecutionResult``
(not a subclass), and the OWNED ``make_result`` error-sort must match graphql-core's own
``build_response`` sort for sibling-field errors.
"""

import graphql
import pytest
from graphql import (
    GraphQLField,
    GraphQLInt,
    GraphQLList,
    GraphQLNonNull,
    GraphQLObjectType,
    GraphQLSchema,
    GraphQLString,
)

from grafast_py import (
    GrafastExecutionContext,
    grafast_execute,
    grafast_subscribe,
    set_field_plan,
)
from grafast_py.core_steps import get


def make_schema() -> GraphQLSchema:
    """A schema mixing plain-resolver fields, a plan-resolver field, errors and nesting."""

    def resolve_name(parent, info):
        return parent["name"]

    def resolve_boom(parent, info):
        raise RuntimeError("kaboom")

    # a plan-resolver field: its value comes from a step, not a resolver, so the step
    # path (not just the ResolveStep adapter) is exercised through both entries.
    doubled_field = GraphQLField(GraphQLInt)
    Child = GraphQLObjectType(
        "Child",
        {
            "label": GraphQLField(GraphQLString, resolve=resolve_name),
            "doubled": doubled_field,
        },
    )

    Query = GraphQLObjectType(
        "Query",
        {
            "greeting": GraphQLField(
                GraphQLString, resolve=lambda r, i: "hello world"
            ),
            # two sibling fields that error, to exercise the deterministic error sort.
            "firstError": GraphQLField(GraphQLString, resolve=resolve_boom),
            "secondError": GraphQLField(GraphQLString, resolve=resolve_boom),
            "children": GraphQLField(
                GraphQLList(Child),
                resolve=lambda r, i: [{"name": "a", "n": 1}, {"name": "b", "n": 2}],
            ),
            # a non-null field that resolves None → nulls the whole root (Bubble path).
            "required": GraphQLField(
                GraphQLNonNull(GraphQLString), resolve=lambda r, i: None
            ),
        },
    )

    set_field_plan(doubled_field, lambda parent, args, info: get(parent, "n"))
    return GraphQLSchema(query=Query)


def execute_via_shim(schema, query, **kwargs):
    """Run a query through the host pattern: execute(..., execution_context_class=...)."""
    return graphql.execute(
        schema,
        graphql.parse(query),
        execution_context_class=GrafastExecutionContext,
        **kwargs,
    )


def test_grafast_execute_returns_plain_execution_result():
    schema = make_schema()
    result = grafast_execute(schema, "{ greeting }")
    assert type(result) is graphql.ExecutionResult
    assert result.data == {"greeting": "hello world"}
    assert result.errors is None


def test_seam_matches_shim_on_mixed_query():
    schema = make_schema()
    query = """
    {
      g: greeting
      kids: children { label doubled }
    }
    """
    seam = grafast_execute(schema, query)
    shim = execute_via_shim(schema, query)
    assert seam.formatted == shim.formatted
    assert seam.data == {
        "g": "hello world",
        "kids": [{"label": "a", "doubled": 1}, {"label": "b", "doubled": 2}],
    }


def test_seam_matches_shim_on_sibling_errors_sort():
    """Two sibling fields raise: the OWNED make_result sort must match build_response.

    Equality of ``.formatted`` against graphql-core's own ``build_response`` output is the
    proof — both must order the two errors identically by the ``(locations, path, message)``
    key. We additionally assert the order is the deterministic one that key produces.
    """
    schema = make_schema()
    query = "{ secondError firstError }"
    seam = grafast_execute(schema, query)
    shim = execute_via_shim(schema, query)
    assert seam.formatted == shim.formatted
    assert seam.errors is not None
    assert len(seam.errors) == 2
    keys = [
        (e.locations or [], list(e.path or []), e.message) for e in seam.errors
    ]
    assert keys == sorted(keys)


def test_seam_matches_shim_on_root_null_bubble():
    """A top-level non-null nulled to None bubbles to {data: None} on both entries."""
    schema = make_schema()
    query = "{ required }"
    seam = grafast_execute(schema, query)
    shim = execute_via_shim(schema, query)
    assert seam.formatted == shim.formatted
    assert seam.data is None
    assert seam.errors is not None


def test_seam_validation_error_is_plain_result():
    """grafast_execute owns the frontend: an invalid query short-circuits to validation
    errors as a plain ExecutionResult (graphql.execute does NOT validate, so the seam's
    validation step is what the consumer would otherwise get from graphql.graphql)."""
    schema = make_schema()
    result = grafast_execute(schema, "{ nonExistentField }")
    assert type(result) is graphql.ExecutionResult
    assert result.data is None
    assert result.errors is not None
    assert "nonExistentField" in result.errors[0].message


def test_seam_accepts_parsed_document():
    """grafast_execute accepts an already-parsed DocumentNode, like graphql-core's execute."""
    schema = make_schema()
    document = graphql.parse("{ greeting }")
    result = grafast_execute(schema, document)
    assert result.data == {"greeting": "hello world"}


def test_seam_validates_a_pre_parsed_document():
    """A caller-supplied (invalid) DocumentNode is validated just like the string path —
    grafast_execute is the full graphql() pipeline, so parse('{ missing }') and the string
    '{ missing }' must yield the SAME validation error, not silently-dropped success."""
    schema = make_schema()
    from_string = grafast_execute(schema, "{ nonExistentField }")
    from_parsed = grafast_execute(schema, graphql.parse("{ nonExistentField }"))
    assert from_parsed.data is None
    assert from_parsed.errors is not None
    assert "nonExistentField" in from_parsed.errors[0].message
    # the two entry shapes agree (the bug was the parsed path skipping validation)
    assert [e.message for e in from_parsed.errors] == [e.message for e in from_string.errors]


# A shared cross-version fixture: the SAME query executed by grafast_execute must produce
# this EXACT ``.formatted`` payload on EVERY graphql-core version. Both CI legs (3.2 and
# 3.3) run this test against the same golden, so a pass on both legs proves
# result_32.formatted == result_33.formatted (the OWNED make_result error-sort is what makes
# the error order version-independent, despite 3.3 no longer sorting in build_data_response).
CROSS_VERSION_QUERY = "{ g: greeting errB: secondError errA: firstError kids: children { label doubled } }"
CROSS_VERSION_GOLDEN = {
    "data": {
        "g": "hello world",
        "errB": None,
        "errA": None,
        "kids": [
            {"label": "a", "doubled": 1},
            {"label": "b", "doubled": 2},
        ],
    },
    "errors": [
        {
            "message": "kaboom",
            "locations": [{"line": 1, "column": 15}],
            "path": ["errB"],
        },
        {
            "message": "kaboom",
            "locations": [{"line": 1, "column": 33}],
            "path": ["errA"],
        },
    ],
}


def test_cross_version_formatted_is_stable():
    """grafast_execute().formatted matches a fixed golden on whichever graphql-core is loaded.

    This is the cross-version equality gate: both the 3.2 and 3.3 CI legs assert against the
    SAME golden, so passing on both proves the two versions yield identical ``.formatted``.
    """
    schema = make_schema()
    result = grafast_execute(schema, CROSS_VERSION_QUERY)
    assert result.formatted == CROSS_VERSION_GOLDEN


@pytest.mark.asyncio
async def test_grafast_subscribe_round_trips_events():
    """grafast_subscribe exists and round-trips non-incremental subscription events.

    Covers the version-stable source-event-stream seam (3.2 ``create_source_event_stream``
    is a coroutine; 3.3 returns maybe-awaitable) and the map-async-iterable wrapper that
    moved between versions. Full incremental subscription delivery is a later phase (P7).
    """

    async def numbers(root, info):
        for i in range(3):
            yield {"value": i}

    Subscription = GraphQLObjectType(
        "Subscription",
        {
            "counter": GraphQLField(
                GraphQLString,
                resolve=lambda event, info: str(event["value"]),
                subscribe=numbers,
            )
        },
    )
    schema = GraphQLSchema(query=make_schema().query_type, subscription=Subscription)

    stream = await grafast_subscribe(schema, "subscription { counter }")
    events = [result.formatted async for result in stream]
    assert events == [
        {"data": {"counter": "0"}},
        {"data": {"counter": "1"}},
        {"data": {"counter": "2"}},
    ]
