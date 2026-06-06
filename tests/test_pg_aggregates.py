"""Connection domain aggregates: a SEPARATE batched sum/avg/min/max/count statement.

A connection field can ask for DOMAIN aggregates over each parent's set — ``sum``/``avg``/
``min``/``max``/``count`` over a column, optionally GROUPed by an extra column. These are a
THIRD optional batched statement built like ``totalCount``'s count aggregate
(:meth:`PgConnectionStep.build_aggregate_query`), mirroring ``build_count_query`` exactly:
the SAME key match (single ``= ANY`` / composite tuple-IN), the SAME host
``where_predicates`` (INCLUDING any compiled filter Condition) as the page query, but
WITHOUT the keyset cursor predicate — so the aggregate covers each parent's FULL set,
correct even on an empty terminal page. It is issued only when the selection set asks for an
aggregate (:func:`connection_aggregates`), so an unaggregated connection emits no extra
statement.

These tests assert:

- the aggregate SQL is ``match[, group_by], <aggs> GROUP BY match[, group_by]`` — one
  statement for the whole bucket, AND-ing the same WHERE predicates as the page;
- a connection aggregates each parent's set correctly (sum/avg/min/max/count), surfaces an
  ungrouped aggregate flat under ``aggregates`` and a grouped one under ``aggregateGroups``,
  and reports the SQL-faithful EMPTY-set defaults (count 0, others null) for a parent with no
  rows — even though Postgres elides the GROUP row;
- a ``where_tree`` filter folds onto the aggregate's WHERE (the aggregate is over the
  FILTERED set, one statement);
- the whole connection layer is at most THREE statements across ALL parents (page + count +
  aggregate), O(depth);
- the fail-loud guards fire — an unknown function / a column-less non-count / a missing alias
  is a declaration bug rejected at construction;
- DEDUP CORRECTNESS (no DB), BOTH directions: two connections differing only by their
  aggregate spec (function / column / alias) or their GROUP BY get DIFFERENT keys and do NOT
  merge through ``dag.Plan.deduplicate()``; structurally-identical specs DO merge; the
  aggregate spec rides ON TOP OF the existing needs_total + filter discriminators (an
  aggregated connection never merges with an unaggregated peer).

DB tests are marked ``pg`` (deselectable ``-m 'not pg'``); they use a dedicated
``line_items`` fixture and touch ONLY the ``grafast_demo`` schema, perturbing nothing else.
"""

from decimal import Decimal

import pytest
import pytest_asyncio
from sqlalchemy.dialects import postgresql

from grafast_py.core_steps import constant
from grafast_py.dag import Plan
from grafast_py.pg.conditions import Compare
from grafast_py.pg.connection import (
    AGGREGATE_FUNCTIONS,
    PgAggregate,
    PgConnectionStep,
)
from grafast_py.pg.engine import count_sql, dispose_engine, get_engine
from grafast_py.pg.executor import SQLAlchemyExecutor, pg_request_context
from grafast_py.pg.resource import PgCodec, PgColumn, PgRegistry, PgResource
from examples.seed import setup_demo_schema, setup_line_items_table


def make_line_items() -> PgResource:
    """A fresh ``line_items`` resource (its own registry) for the aggregate tests."""
    return PgResource(
        "line_items",
        "grafast_demo",
        "line_items",
        ["id", "order_id", "category", "status", "quantity", "price"],
        registry=PgRegistry(),
    )


def make_conn(**kwargs) -> PgConnectionStep:
    """An ``order -> line_items`` connection keyed on ``order_id`` (ordered by id)."""
    return PgConnectionStep(
        make_line_items(), constant(None), "order_id", order_by=["id"], **kwargs
    )


@pytest_asyncio.fixture
async def seeded():
    """(Re)seed ``grafast_demo`` + the ``line_items`` fixture (fresh engine per test)."""
    await dispose_engine()
    await setup_demo_schema()
    await setup_line_items_table()
    yield
    await dispose_engine()


# ----------------------------------------------------------------- aggregate spec


def test_aggregate_projection_compiles_per_function():
    """Each aggregate projects ``<function>(<column>) AS <alias>`` (count(*) for a bare count).

    A label renders its ``AS <alias>`` only inside a SELECT column list, so render the
    projection in a one-column ``select`` to observe the labelled form the real query emits.
    """
    from sqlalchemy import select

    def render(agg: PgAggregate) -> str:
        return str(select(agg.projection()).compile(dialect=postgresql.dialect()))

    assert render(PgAggregate("sum", "quantity", "qty")) == "SELECT sum(quantity) AS qty"
    assert render(PgAggregate("avg", "price", "p")) == "SELECT avg(price) AS p"
    assert render(PgAggregate("min", "price", "lo")) == "SELECT min(price) AS lo"
    assert render(PgAggregate("max", "price", "hi")) == "SELECT max(price) AS hi"
    assert render(PgAggregate("count", "status", "ok")) == "SELECT count(status) AS ok"
    assert render(PgAggregate("count", None, "n")) == "SELECT count(*) AS n"


def test_supported_functions_are_the_documented_five():
    """The aggregate function set is exactly sum/avg/min/max/count."""
    assert AGGREGATE_FUNCTIONS == {"sum", "avg", "min", "max", "count"}


def test_unsupported_function_fails_loud():
    """An unknown aggregate function is a declaration bug — rejected at construction."""
    with pytest.raises(ValueError, match="unsupported aggregate function"):
        PgAggregate("median", "price", "p")


def test_non_count_without_column_fails_loud():
    """A sum/avg/min/max needs a column; only count may omit one."""
    with pytest.raises(ValueError, match="needs a column"):
        PgAggregate("sum", None, "s")


def test_missing_alias_fails_loud():
    """An aggregate needs an output alias (it is the dict key its value lands under)."""
    with pytest.raises(ValueError, match="needs an output alias"):
        PgAggregate("sum", "price", "")


# ------------------------------------------------------ the separate aggregate SQL


def test_aggregate_query_mirrors_count_grouped_over_match():
    """The aggregate SQL groups the projections over the match key (mirroring count)."""
    step = make_conn(
        aggregates=[PgAggregate("sum", "quantity", "qty"), PgAggregate("count", None, "n")]
    )
    sql = str(step.build_aggregate_query().compile(dialect=postgresql.dialect()))
    assert "sum(quantity) AS qty" in sql
    assert "count(*) AS n" in sql
    assert "GROUP BY order_id" in sql
    # the key match is the SAME = ANY(:keys) skeleton as the page/count.
    assert "order_id = ANY" in sql


def test_grouped_aggregate_adds_group_by_column():
    """An ``aggregate_group_by`` column is added to BOTH the projection and the GROUP BY."""
    step = make_conn(
        aggregates=[PgAggregate("sum", "quantity", "qty")],
        aggregate_group_by=["category"],
    )
    sql = str(step.build_aggregate_query().compile(dialect=postgresql.dialect()))
    assert "order_id, category, sum(quantity) AS qty" in sql
    assert "GROUP BY order_id, category" in sql


def test_empty_aggregates_defaults_count_of_a_column_to_zero():
    """A ``count(column)`` over an empty set defaults to 0, not None (no DB).

    ``count`` is the only aggregate immune to the empty-set→NULL rule: ``count(col)`` counts
    non-NULL values and returns bigint 0 over zero rows. The empty-set synthesis must follow
    SQL, so a column ``count`` and a column-less ``count(*)`` both default to 0 while
    sum/min surface NULL.
    """
    step = make_conn(
        aggregates=[
            PgAggregate("count", "status", "c_col"),
            PgAggregate("count", None, "c_star"),
            PgAggregate("sum", "quantity", "s"),
            PgAggregate("min", "price", "lo"),
        ]
    )
    assert step.empty_aggregates() == {"c_col": 0, "c_star": 0, "s": None, "lo": None}


def test_no_aggregates_means_no_statement_and_no_aggregate_keys():
    """Without aggregates the connection has no aggregate statement and no aggregate output."""
    step = make_conn(first=2)
    assert step.has_aggregates is False
    assert step.aggregate_spec() == ()
    # an unaggregated connection dict carries neither `aggregates` nor `aggregateGroups`.
    out = step.build_connection([], 0)
    assert "aggregates" not in out
    assert "aggregateGroups" not in out


# ------------------------------------------- aggregate decode through resource codec (no DB)


def coded_line_items() -> PgResource:
    """A ``line_items`` resource whose ``category`` column carries a decoding codec.

    The codec uppercase-tags the value (``book`` -> ``DECODED:book``) so a decode is OBSERVABLE
    in the output; it lets the aggregate path prove that group-by keys and min/max values are
    decoded the same way page nodes are.
    """
    return PgResource(
        "line_items",
        "grafast_demo",
        "line_items",
        [
            "id",
            "order_id",
            PgColumn("category", codec=PgCodec(to_py=lambda v: "DECODED:" + v)),
            "status",
            "quantity",
            "price",
        ],
        registry=PgRegistry(),
    )


def coded_conn(**kwargs) -> PgConnectionStep:
    """A connection over :func:`coded_line_items` keyed on ``order_id``."""
    return PgConnectionStep(
        coded_line_items(), constant(None), "order_id", order_by=["id"], **kwargs
    )


def test_aggregate_group_by_key_is_decoded_through_the_resource_codec():
    """An ``aggregateGroups`` group-by key rides the SAME codec decode a page node gets.

    Grouping by a codec'd ``category`` column must surface the DECODED key (``DECODED:book``),
    not the raw asyncpg value — matching what the same column carries on a node.
    """
    step = coded_conn(
        aggregates=[PgAggregate("sum", "quantity", "qty")],
        aggregate_group_by=["category"],
    )
    conn = step.build_connection([], 0, aggregate_rows=[{"order_id": 1, "category": "book", "qty": 5}])
    assert conn["aggregateGroups"] == [{"category": "DECODED:book", "qty": 5}]


def test_min_max_over_codec_column_is_decoded_but_sum_count_are_not():
    """``min`` / ``max`` over a codec'd column decode; ``sum`` / ``count`` (numeric) do not.

    A ``max(category)`` returns a value OF the category column's type, so it rides the column
    codec; a ``sum``/``count`` produces a numeric not of that type and must pass through raw.
    """
    step = coded_conn(
        aggregates=[
            PgAggregate("max", "category", "hi"),
            PgAggregate("min", "category", "lo"),
            PgAggregate("count", "category", "n"),
        ]
    )
    conn = step.build_connection(
        [], 0, aggregate_rows=[{"order_id": 1, "hi": "media", "lo": "book", "n": 4}]
    )
    aggs = conn["aggregates"]
    assert aggs["hi"] == "DECODED:media"  # max decoded
    assert aggs["lo"] == "DECODED:book"  # min decoded
    assert aggs["n"] == 4  # count is numeric, not decoded


# --------------------------------------------------------- DB: aggregate execution


@pytest.mark.pg
@pytest.mark.asyncio
async def test_ungrouped_aggregate_computes_per_parent_set(seeded):
    """An ungrouped aggregate sums/avgs/min/maxes/counts each parent's FULL set."""
    aggs = [
        PgAggregate("sum", "quantity", "qty_sum"),
        PgAggregate("avg", "price", "price_avg"),
        PgAggregate("min", "price", "price_min"),
        PgAggregate("max", "price", "price_max"),
        PgAggregate("count", None, "n"),
    ]
    with pg_request_context(SQLAlchemyExecutor(get_engine())):
        step = make_conn(first=2, aggregates=aggs)
        out = await step.execute(2, [[1, 2]])

    # order 1: quantities 2,3,1,10 -> sum 16, count 4; prices 10,20,5,99 -> min 5, max 99.
    o1 = out[0]["aggregates"]
    assert o1["qty_sum"] == 16
    assert o1["n"] == 4
    assert o1["price_min"] == Decimal("5.00")
    assert o1["price_max"] == Decimal("99.00")
    assert o1["price_avg"] == Decimal("33.50")  # (10+20+5+99)/4
    # order 2: quantities 4,5 -> sum 9, count 2; prices 8,12.50 -> min 8, max 12.50.
    o2 = out[1]["aggregates"]
    assert o2["qty_sum"] == 9
    assert o2["n"] == 2
    assert o2["price_min"] == Decimal("8.00")
    assert o2["price_max"] == Decimal("12.50")


@pytest.mark.pg
@pytest.mark.asyncio
async def test_aggregate_is_over_full_set_not_the_page(seeded):
    """The aggregate covers the FULL per-parent set even when the page is sliced to first:1."""
    with pg_request_context(SQLAlchemyExecutor(get_engine())):
        step = make_conn(first=1, aggregates=[PgAggregate("count", None, "n")])
        out = await step.execute(1, [[1]])

    conn = out[0]
    # the page is one row, but the aggregate counts all 4 of order 1's items.
    assert len(conn["edges"]) == 1
    assert conn["aggregates"]["n"] == 4


@pytest.mark.pg
@pytest.mark.asyncio
async def test_empty_parent_reports_sql_faithful_aggregate_defaults(seeded):
    """A parent with NO rows reports count 0 (count(*) AND count(col)) and null sum/min (no GROUP row)."""
    aggs = [
        PgAggregate("count", None, "n"),
        PgAggregate("count", "status", "n_status"),
        PgAggregate("sum", "quantity", "qty"),
    ]
    with pg_request_context(SQLAlchemyExecutor(get_engine())):
        # order 3 owns no line items.
        step = make_conn(first=2, aggregates=aggs)
        out = await step.execute(1, [[3]])

    conn = out[0]
    assert conn["edges"] == []
    assert conn["aggregates"]["n"] == 0  # count(*) of empty set is 0
    assert conn["aggregates"]["n_status"] == 0  # count(column) of empty set is ALSO 0, not NULL
    assert conn["aggregates"]["qty"] is None  # sum of empty set is SQL NULL


@pytest.mark.pg
@pytest.mark.asyncio
async def test_grouped_aggregate_yields_one_bucket_per_group(seeded):
    """A grouped aggregate yields one ``aggregateGroups`` entry per (parent, group) bucket."""
    with pg_request_context(SQLAlchemyExecutor(get_engine())):
        step = make_conn(
            aggregates=[PgAggregate("sum", "quantity", "qty")],
            aggregate_group_by=["category"],
        )
        out = await step.execute(1, [[1]])

    groups = {g["category"]: g["qty"] for g in out[0]["aggregateGroups"]}
    # order 1: book quantities 2,3 -> 5; media quantities 1,10 -> 11.
    assert groups == {"book": 5, "media": 11}


@pytest.mark.pg
@pytest.mark.asyncio
async def test_filter_folds_onto_aggregate_where(seeded):
    """A ``where_tree`` filter folds onto the aggregate WHERE — the aggregate is over the FILTERED set."""
    with pg_request_context(SQLAlchemyExecutor(get_engine())):
        step = make_conn(
            aggregates=[PgAggregate("sum", "quantity", "qty"), PgAggregate("count", None, "n")]
        )
        step.builder().where_tree(Compare("status", "eq", "ok"))
        out = await step.execute(1, [[1]])

    conn = out[0]
    # status='ok' drops order 1's voided item (quantity 10): remaining 2,3,1 -> sum 6, count 3.
    assert conn["aggregates"]["qty"] == 6
    assert conn["aggregates"]["n"] == 3
    # the filter is in the aggregate WHERE alongside the key match (a bound :param).
    sql = str(step.build_aggregate_query())
    assert "status = :" in sql
    assert "ANY (:keys)" in sql


@pytest.mark.pg
@pytest.mark.asyncio
async def test_connection_layer_is_at_most_three_statements(seeded):
    """Page + count + aggregate = 3 statements for the WHOLE bucket (O(depth), all parents)."""
    aggs = [PgAggregate("sum", "quantity", "qty")]
    with pg_request_context(SQLAlchemyExecutor(get_engine())):
        step = make_conn(first=2, needs_total=True, aggregates=aggs)
        with count_sql(get_engine()) as counter:
            out = await step.execute(2, [[1, 2]])

    # one page + one count + one aggregate, regardless of the two parents in the bucket.
    assert counter.count == 3
    assert [o["totalCount"] for o in out] == [4, 2]
    assert [o["aggregates"]["qty"] for o in out] == [16, 9]


@pytest.mark.pg
@pytest.mark.asyncio
async def test_aggregate_not_selected_issues_no_extra_statement(seeded):
    """Without aggregates the connection is page-only (no aggregate statement issued)."""
    with pg_request_context(SQLAlchemyExecutor(get_engine())):
        step = make_conn(first=2)
        with count_sql(get_engine()) as counter:
            await step.execute(1, [[1]])

    # the page query ONLY — no count, no aggregate.
    assert counter.count == 1


# --------------------------------------------------------------- dedup correctness


def dedup_key(step):
    """The class + peer_key + dedup_params slice of the planner's structural key."""
    return (type(step), step.peer_key, step.dedup_params())


def test_aggregate_spec_differing_by_function_does_not_dedup():
    """REGRESSION GATE: two connections aggregating with DIFFERENT functions never merge.

    The aggregate spec folds into the dedup key, so a ``sum`` and an ``avg`` over the same
    column emit different aggregate SQL and stay distinct survivors through deduplicate().
    """
    a = make_conn(aggregates=[PgAggregate("sum", "quantity", "q")])
    b = make_conn(aggregates=[PgAggregate("avg", "quantity", "q")])
    assert a.peer_key != b.peer_key
    assert a.dedup_params() != b.dedup_params()
    assert dedup_key(a) != dedup_key(b)

    plan = Plan()
    plan.add_step(a)
    plan.add_step(b)
    remap = plan.deduplicate()
    assert remap[a.id] is a
    assert remap[b.id] is b


def test_aggregate_spec_differing_by_column_or_alias_does_not_dedup():
    """Aggregates differing only by COLUMN or by ALIAS are not peers (different SQL)."""
    base = make_conn(aggregates=[PgAggregate("sum", "quantity", "q")])
    other_col = make_conn(aggregates=[PgAggregate("sum", "price", "q")])
    other_alias = make_conn(aggregates=[PgAggregate("sum", "quantity", "total")])
    assert base.peer_key != other_col.peer_key
    assert base.peer_key != other_alias.peer_key
    assert base.dedup_params() != other_col.dedup_params()
    assert base.dedup_params() != other_alias.dedup_params()


def test_identical_aggregate_specs_merge():
    """Two connections with the STRUCTURALLY-identical aggregate spec DO merge."""
    a = make_conn(
        aggregates=[PgAggregate("sum", "quantity", "q"), PgAggregate("count", None, "n")]
    )
    b = make_conn(
        aggregates=[PgAggregate("sum", "quantity", "q"), PgAggregate("count", None, "n")]
    )
    assert a.peer_key == b.peer_key
    assert a.dedup_params() == b.dedup_params()
    assert dedup_key(a) == dedup_key(b)

    plan = Plan()
    plan.add_step(a)
    plan.add_step(b)
    remap = plan.deduplicate()
    # the lower-id step wins; both references resolve to the same survivor.
    assert remap[a.id] is remap[b.id]


def test_group_by_discriminates_dedup_key():
    """The same aggregate grouped by an extra column does NOT merge with the ungrouped one."""
    ungrouped = make_conn(aggregates=[PgAggregate("sum", "quantity", "q")])
    grouped = make_conn(
        aggregates=[PgAggregate("sum", "quantity", "q")], aggregate_group_by=["category"]
    )
    assert ungrouped.peer_key != grouped.peer_key
    assert ungrouped.dedup_params() != grouped.dedup_params()
    assert dedup_key(ungrouped) != dedup_key(grouped)


def test_aggregated_vs_unaggregated_do_not_dedup():
    """An aggregated connection never merges with an unaggregated one over the same skeleton."""
    plain = make_conn(first=2)
    aggregated = make_conn(first=2, aggregates=[PgAggregate("count", None, "n")])
    assert plain.peer_key != aggregated.peer_key
    assert dedup_key(plain) != dedup_key(aggregated)


def test_aggregate_spec_rides_on_top_of_needs_total_and_filter():
    """The aggregate discriminator is INDEPENDENT of needs_total and the filter signature.

    Two connections with the SAME aggregate spec but a different needs_total — or a different
    filter VALUE — must still differ (the aggregate fold does not collapse the existing
    discriminators), and two with the same needs_total + filter + aggregate spec must merge.
    """
    aggs = [PgAggregate("sum", "quantity", "q")]
    with_total = make_conn(aggregates=aggs, needs_total=True)
    without_total = make_conn(aggregates=aggs, needs_total=False)
    assert with_total.peer_key != without_total.peer_key

    a = make_conn(aggregates=aggs)
    a.builder().where_tree(Compare("status", "eq", "ok"))
    b = make_conn(aggregates=aggs)
    b.builder().where_tree(Compare("status", "eq", "void"))
    # same aggregate spec, different filter value -> still distinct.
    assert a.peer_key != b.peer_key
    assert a.dedup_params() != b.dedup_params()

    c = make_conn(aggregates=aggs)
    c.builder().where_tree(Compare("status", "eq", "ok"))
    # same aggregate spec AND same filter -> peers.
    assert a.peer_key == c.peer_key
    assert a.dedup_params() == c.dedup_params()
