"""Tests for composite (multi-column) match keys across the pg steps.

Two layers:

- NO-DB: the SQL skeleton and the dedup-correctness invariant. A single match column
  keeps the cheap ``= ANY(:keys)`` ($1::T[]) fast path; a COMPOSITE key (a tuple of
  columns) emits ``(c1, c2, ...) IN (:keys)`` and partitions/groups over the tuple. The
  ``match_columns`` tuple is folded into ``peer_key`` / ``dedup_params`` (it changes the
  emitted SQL skeleton AND the grouping key), so two steps differing only by their match
  columns are NOT peers, and identical ones ARE — proven both directions.
- DB (``pg`` marker): the end-to-end gate over ``grafast_demo.regions`` /
  ``grafast_demo.stores`` (a two-column FK). The tuple match batches every parent into ONE
  statement, scatters by the column TUPLE (so the cross-org/region cross-link trap is
  avoided), and the window slice + Relay connection page per-parent over the tuple. The
  ``(org_id, region_id)`` pairs reuse single-column values across orgs, so a match on a
  single column alone would mis-scatter — only the tuple match is correct.

Marked ``pg`` on the DB tests only (the module mixes no-DB and DB tests, so there is no
module-level ``pytestmark``).
"""

import pytest
from sqlalchemy import ForeignKeyConstraint, Integer, Text
from sqlalchemy.dialects.postgresql import asyncpg as asyncpg_dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from grafast_py.core_steps import constant, list_step
from grafast_py.dag import Plan
from grafast_py.pg.connection import PgConnectionStep
from grafast_py.pg.engine import count_sql, dispose_engine, get_engine
from grafast_py.pg.executor import RawExecutor, SQLAlchemyExecutor, pg_request_context
from grafast_py.pg.from_sqlalchemy import resource_from_model, wire_relations
from grafast_py.pg.resource import PgRegistry, PgResource
from grafast_py.pg.steps import (
    PgSelectSingleStep,
    PgSelectStep,
    as_match_columns,
    grouping_key,
    normalize_lookup_key,
)
from examples.seed import setup_composite_tables, setup_demo_schema


# ----------------------------------------------------------------- resource helpers


def make_stores() -> PgResource:
    """A ``stores`` resource (composite FK ``(org_id, region_id)`` to ``regions``)."""
    return PgResource(
        "stores",
        "grafast_demo",
        "stores",
        ["id", "org_id", "region_id", "name"],
    )


def make_regions() -> PgResource:
    """A ``regions`` resource with a composite primary key ``(org_id, region_id)``."""
    return PgResource(
        "regions",
        "grafast_demo",
        "regions",
        ["org_id", "region_id", "label"],
        primary_key="org_id",
    )


def composite_select(first=None, offset=0) -> PgSelectStep:
    """A stores select keyed on the ``(org_id, region_id)`` tuple."""
    key = list_step([constant(0), constant(0)])
    return PgSelectStep(
        make_stores(), key, ("org_id", "region_id"),
        order_by=["id"], first=first, offset=offset,
    )


# ------------------------------------------------------- no-DB coercion / grouping


def test_as_match_columns_collapses_single_and_tuple():
    """A bare column name becomes a 1-tuple; a tuple passes through; empty fails loud."""
    assert as_match_columns("author_id") == ("author_id",)
    assert as_match_columns(("org_id", "region_id")) == ("org_id", "region_id")
    assert as_match_columns(["a", "b"]) == ("a", "b")
    with pytest.raises(ValueError):
        as_match_columns(())


def test_grouping_key_is_scalar_for_single_and_tuple_for_composite():
    """A single column groups on the scalar; a composite on the column tuple."""
    row = {"org_id": 2, "region_id": 1, "name": "x"}
    assert grouping_key(row, ("org_id",)) == 2
    assert grouping_key(row, ("org_id", "region_id")) == (2, 1)


def test_normalize_lookup_key_tuple_ifies_and_nulls_partial():
    """Composite keys tuple-ify; any None component (or whole) normalises to None."""
    # single: scalar passes through
    assert normalize_lookup_key(5, composite=False) == 5
    assert normalize_lookup_key(None, composite=False) is None
    # composite: a list (from list_step) becomes a tuple
    assert normalize_lookup_key([2, 1], composite=True) == (2, 1)
    # a partial key (one component None) cannot match a full FK -> None
    assert normalize_lookup_key([2, None], composite=True) is None
    assert normalize_lookup_key(None, composite=True) is None


# ----------------------------------------------------------------- SQL skeleton


def test_single_column_keeps_the_any_array_fast_path():
    """A single match column still compiles to the cheap ``= ANY(:keys)`` ($1 array)."""
    step = PgSelectStep(make_stores(), constant(None), "org_id", order_by=["id"])
    assert step.is_composite is False
    sql = str(step.build_query()).upper()
    assert "ANY" in sql
    assert "IN (" not in sql  # not the tuple-IN branch


def test_composite_emits_tuple_in_not_any():
    """A composite key compiles to ``(c1, c2) IN (...)`` — never the single ANY array."""
    step = composite_select()
    assert step.is_composite is True
    # bake the keys at build time so the (asyncpg) postcompile renders the IN list.
    compiled = step.build_query([(1, 1), (2, 1)]).compile(
        dialect=asyncpg_dialect.dialect(),
        compile_kwargs={"render_postcompile": True},
    )
    sql = str(compiled)
    assert "(org_id, region_id) IN" in sql
    # two tuples -> $1..$4 expanded, not a single $1 array.
    assert "$4" in sql
    assert "ANY" not in sql.upper()


def test_composite_match_column_accessor_raises():
    """The single-column ``match_column`` accessor fails loud on a composite step."""
    step = composite_select()
    with pytest.raises(ValueError, match="composite"):
        _ = step.match_column


def test_composite_window_slice_partitions_over_the_tuple():
    """A limited composite select partitions ``row_number()`` over BOTH match columns."""
    step = composite_select(first=2)
    sql = str(step.build_query([(1, 1)])).lower()
    assert "partition by org_id, region_id" in sql
    assert "row_number()" in sql


def test_composite_connection_partitions_and_groups_over_the_tuple():
    """A composite connection partitions the page AND groups the count over the tuple."""
    key = list_step([constant(0), constant(0)])
    conn = PgConnectionStep(
        make_stores(), key, ("org_id", "region_id"),
        order_by=["id"], first=2, needs_total=True,
    )
    page = str(conn.build_page_query([(1, 1)])).lower()
    assert "partition by org_id, region_id" in page
    count = str(conn.build_count_query([(1, 1)])).lower()
    assert "group by org_id, region_id" in count


# --------------------------------------------------------------- dedup correctness


def dedup_key(step):
    """The class + peer_key + dedup_params slice of the planner's structural key."""
    return (type(step), step.peer_key, step.dedup_params())


def test_composite_vs_single_are_not_peers():
    """A composite-key select never merges with a single-key one over the same table.

    The match_columns tuple changes the emitted SQL (= ANY vs tuple-IN) and the grouping
    key, so it MUST discriminate the dedup key (different => not peers).
    """
    single = PgSelectStep(make_stores(), constant(None), "org_id", order_by=["id"])
    composite = composite_select()
    assert single.peer_key != composite.peer_key
    assert single.dedup_params() != composite.dedup_params()
    assert dedup_key(single) != dedup_key(composite)

    plan = Plan()
    plan.add_step(single)
    plan.add_step(composite)
    remap = plan.deduplicate()
    assert remap[single.id] is single
    assert remap[composite.id] is composite


def test_different_composite_columns_are_not_peers():
    """Two composite selects over DIFFERENT column tuples are not peers (different SQL)."""
    a = PgSelectStep(
        make_stores(), list_step([constant(0), constant(0)]),
        ("org_id", "region_id"), order_by=["id"],
    )
    b = PgSelectStep(
        make_stores(), list_step([constant(0), constant(0)]),
        ("region_id", "org_id"), order_by=["id"],
    )
    assert a.peer_key != b.peer_key
    assert a.dedup_params() != b.dedup_params()
    assert dedup_key(a) != dedup_key(b)


def test_identical_composite_keys_are_peers():
    """Two selects over the IDENTICAL composite key + skeleton ARE peers (same key)."""
    a = composite_select()
    b = composite_select()
    assert a.peer_key == b.peer_key
    assert a.dedup_params() == b.dedup_params()
    assert dedup_key(a) == dedup_key(b)


def test_composite_connection_dedup_both_directions():
    """A composite connection folds match_columns into its dedup key, both directions."""
    def conn(columns):
        return PgConnectionStep(
            make_stores(), list_step([constant(0), constant(0)]),
            columns, order_by=["id"], first=2, needs_total=True,
        )

    same_a, same_b = conn(("org_id", "region_id")), conn(("org_id", "region_id"))
    assert dedup_key(same_a) == dedup_key(same_b)

    diff = conn(("region_id", "org_id"))
    assert dedup_key(same_a) != dedup_key(diff)


def test_composite_single_step_dedup_both_directions():
    """A composite single-row step folds match_columns into its dedup key, both ways."""
    def single(columns):
        return PgSelectSingleStep(
            make_regions(), list_step([constant(0), constant(0)]), columns
        )

    a, b = single(("org_id", "region_id")), single(("org_id", "region_id"))
    assert dedup_key(a) == dedup_key(b)
    other = single(("region_id", "org_id"))
    assert dedup_key(a) != dedup_key(other)


# ------------------------------------------------------- from_sqlalchemy wiring (no-DB)


class CompBase(DeclarativeBase):
    pass


class CkRegion(CompBase):
    """A composite-PK parent (``(org_id, region_id)``) with a composite-FK child list."""

    __tablename__ = "ck_regions"

    org_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    region_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    label: Mapped[str] = mapped_column(Text, nullable=False)
    stores: Mapped[list["CkStore"]] = relationship(
        "CkStore", back_populates="region"
    )


class CkStore(CompBase):
    """A child whose two-column FK ``(org_id, region_id)`` references ``ck_regions``."""

    __tablename__ = "ck_stores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    org_id: Mapped[int] = mapped_column(Integer, nullable=False)
    region_id: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    region: Mapped["CkRegion"] = relationship("CkRegion", back_populates="stores")
    __table_args__ = (
        ForeignKeyConstraint(
            ["org_id", "region_id"], ["ck_regions.org_id", "ck_regions.region_id"]
        ),
    )


def test_composite_fk_relation_is_wired_not_skipped():
    """``resources_from_models`` (via wire_relations) wires a composite FK as a tuple relation.

    A composite FK used to be skipped; it is now derived as a relation over the local /
    remote column tuples (single-column orientation rules unchanged). ``strict=True`` must
    NOT raise on the composite FK any more.
    """
    registry = PgRegistry()
    # the parent has a composite PK PgResource can't auto-derive, so pass the override
    # explicitly; the child has a single PK. wire_relations then derives the FK tuples.
    model_to_resource = {
        CkRegion: resource_from_model(
            CkRegion, registry=registry, primary_key="org_id"
        ),
        CkStore: resource_from_model(CkStore, registry=registry),
    }
    wire_relations(CkRegion, model_to_resource, strict=True)
    wire_relations(CkStore, model_to_resource, strict=True)

    has_many = registry["ck_regions"].relations["stores"]
    assert has_many.kind == "has_many"
    assert has_many.is_composite is True
    assert has_many.local_columns == ("org_id", "region_id")
    assert has_many.remote_columns == ("org_id", "region_id")

    has_one = registry["ck_stores"].relations["region"]
    assert has_one.kind == "has_one"
    assert has_one.is_composite is True
    assert has_one.local_columns == ("org_id", "region_id")
    assert has_one.remote_columns == ("org_id", "region_id")


# ----------------------------------------------------------------- DB end-to-end


@pytest.mark.pg
@pytest.mark.asyncio
async def test_composite_hasmany_batches_into_one_statement():
    """A composite ``find`` over N parents issues exactly ONE tuple-IN statement."""
    await dispose_engine()
    await setup_demo_schema()
    await setup_composite_tables()
    stores = make_stores()
    key = list_step([constant(0), constant(0)])
    step = PgSelectStep(stores, key, ("org_id", "region_id"), order_by=["id"])
    engine = get_engine()
    with count_sql(engine) as counter:
        with pg_request_context(SQLAlchemyExecutor(engine)):
            # three parents: (1,1) owns 2, (1,2) owns 1, (2,1) owns 3
            out = await step.execute(3, [[[1, 1], [1, 2], [2, 1]]])
    assert counter.count == 1
    assert [sorted(r["name"] for r in out[i]) for i in range(3)] == [
        ["store-a", "store-b"],
        ["store-c"],
        ["store-d", "store-e", "store-f"],
    ]
    await dispose_engine()


@pytest.mark.pg
@pytest.mark.asyncio
async def test_composite_match_avoids_single_column_crosslink():
    """The tuple match scatters by the WHOLE pair, never by a single shared column.

    ``(1,1)`` and ``(2,1)`` share ``region_id == 1``; ``(2,1)`` and ``(2,2)`` share
    ``org_id == 2``. A single-column match would cross-link those rows. The tuple match
    keeps each parent's stores strictly to its own ``(org_id, region_id)``.
    """
    await dispose_engine()
    await setup_demo_schema()
    await setup_composite_tables()
    stores = make_stores()
    key = list_step([constant(0), constant(0)])
    step = PgSelectStep(stores, key, ("org_id", "region_id"), order_by=["id"])
    engine = get_engine()
    with pg_request_context(SQLAlchemyExecutor(engine)):
        out = await step.execute(2, [[[1, 1], [2, 1]]])
    # if region_id alone matched, both would carry all of a,b,d,e,f — they must not.
    assert sorted(r["name"] for r in out[0]) == ["store-a", "store-b"]
    assert sorted(r["name"] for r in out[1]) == ["store-d", "store-e", "store-f"]
    await dispose_engine()


@pytest.mark.pg
@pytest.mark.asyncio
async def test_composite_missing_and_empty_keys_scatter_to_empty():
    """A pair with no rows -> empty; a partial (None component) key -> empty, no match."""
    await dispose_engine()
    await setup_demo_schema()
    await setup_composite_tables()
    stores = make_stores()
    key = list_step([constant(0), constant(0)])
    step = PgSelectStep(stores, key, ("org_id", "region_id"), order_by=["id"])
    engine = get_engine()
    with pg_request_context(SQLAlchemyExecutor(engine)):
        # (2,2) owns no stores; [2, None] is a partial key (cannot match a full FK).
        out = await step.execute(3, [[[2, 2], [2, None], [1, 1]]])
    assert out[0] == []
    assert out[1] == []
    assert sorted(r["name"] for r in out[2]) == ["store-a", "store-b"]
    await dispose_engine()


@pytest.mark.pg
@pytest.mark.asyncio
async def test_composite_hasone_single_row_per_parent():
    """A composite single-row select returns each parent's one matching region row."""
    await dispose_engine()
    await setup_demo_schema()
    await setup_composite_tables()
    regions = make_regions()
    key = list_step([constant(0), constant(0)])
    step = PgSelectSingleStep(regions, key, ("org_id", "region_id"))
    engine = get_engine()
    with pg_request_context(SQLAlchemyExecutor(engine)):
        out = await step.execute(3, [[[2, 1], [99, 99], [1, 2]]])
    assert out[0]["label"] == "org2-north"
    assert out[1] is None  # missing pair -> None
    assert out[2]["label"] == "org1-south"
    await dispose_engine()


@pytest.mark.pg
@pytest.mark.asyncio
async def test_composite_window_slice_pages_per_parent_in_one_statement():
    """A composite ``first=2`` pages EACH parent's stores in one windowed statement."""
    await dispose_engine()
    await setup_demo_schema()
    await setup_composite_tables()
    stores = make_stores()
    key = list_step([constant(0), constant(0)])
    step = PgSelectStep(
        stores, key, ("org_id", "region_id"), order_by=["id"], first=2
    )
    engine = get_engine()
    with count_sql(engine) as counter:
        with pg_request_context(SQLAlchemyExecutor(engine)):
            out = await step.execute(2, [[[2, 1], [1, 1]]])
    assert counter.count == 1
    # (2,1) owns d,e,f -> first 2 = d,e; (1,1) owns a,b -> first 2 = a,b
    assert [r["name"] for r in out[0]] == ["store-d", "store-e"]
    assert [r["name"] for r in out[1]] == ["store-a", "store-b"]
    await dispose_engine()


@pytest.mark.pg
@pytest.mark.asyncio
async def test_composite_connection_pages_and_counts_per_parent():
    """A composite Relay connection pages + totals per parent over the tuple (2 statements)."""
    await dispose_engine()
    await setup_demo_schema()
    await setup_composite_tables()
    stores = make_stores()
    key = list_step([constant(0), constant(0)])
    conn = PgConnectionStep(
        stores, key, ("org_id", "region_id"),
        order_by=["id"], first=2, needs_total=True,
    )
    engine = get_engine()
    with count_sql(engine) as counter:
        with pg_request_context(SQLAlchemyExecutor(engine)):
            out = await conn.execute(2, [[[2, 1], [1, 1]]])
    # page query + the separate totalCount aggregate.
    assert counter.count == 2

    north2 = out[0]
    assert north2["totalCount"] == 3
    assert [n["name"] for n in north2["nodes"]] == ["store-d", "store-e"]
    assert north2["pageInfo"]["hasNextPage"] is True

    north1 = out[1]
    assert north1["totalCount"] == 2
    assert [n["name"] for n in north1["nodes"]] == ["store-a", "store-b"]
    assert north1["pageInfo"]["hasNextPage"] is False
    await dispose_engine()


@pytest.mark.pg
@pytest.mark.asyncio
async def test_composite_runs_on_raw_executor_as_expanded_tuple_in():
    """The composite statement runs through a host RawExecutor as expanded tuple-IN $N."""
    await dispose_engine()
    await setup_demo_schema()
    await setup_composite_tables()
    stores = make_stores()
    key = list_step([constant(0), constant(0)])
    step = PgSelectStep(stores, key, ("org_id", "region_id"), order_by=["id"])
    engine = get_engine()

    seen_sql: list[str] = []

    async def run_on_raw_pool(sql_text, positional_params, settings):
        seen_sql.append(sql_text)
        raw = await engine.raw_connection()
        try:
            asyncpg_conn = raw.driver_connection
            records = await asyncpg_conn.fetch(sql_text, *positional_params)
            return [dict(r) for r in records]
        finally:
            raw.close()

    with pg_request_context(RawExecutor(run_on_raw_pool)):
        out = await step.execute(2, [[[1, 1], [2, 1]]])

    assert "(org_id, region_id) IN" in seen_sql[0]
    # two pairs -> four positional binds expanded ($1..$4), never one $1 array.
    assert "$4" in seen_sql[0]
    assert sorted(r["name"] for r in out[0]) == ["store-a", "store-b"]
    assert sorted(r["name"] for r in out[1]) == ["store-d", "store-e", "store-f"]
    await dispose_engine()
