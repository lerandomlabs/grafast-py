"""Structured ORDER BY: direction, multi-column, PK tie-break, and NULLS placement.

This module replaces the ascending-column-names-only ordering with structured
:class:`~grafast_py.pg.ordering.OrderTerm` specs. These tests assert the EMITTED
ordering is correct (direction, multiple columns, NULLS placement, and the always-on
primary-key tie-break unless the order is declared unique) AND that ordering never
changes the batching profile — one statement per resource-layer, as before.

DB tests are marked ``pg`` (deselectable with ``-m 'not pg'``); they touch ONLY the
``grafast_demo`` schema of ``grafast_py_test``. The dedup-correctness tests need no DB
(they only build steps and inspect their dedup keys).
"""

import pytest
import pytest_asyncio
from graphql import graphql

from grafast_py.context import GrafastExecutionContext
from grafast_py.core_steps import access, constant
from grafast_py.pg.engine import count_sql, dispose_engine, get_engine
from grafast_py.pg.executor import SQLAlchemyExecutor, pg_request_context
from grafast_py.pg.ordering import OrderTerm, normalize_order
from grafast_py.pg.resource import PgRegistry, PgResource
from grafast_py.pg.steps import PgSelectAllStep, PgSelectStep
from grafast_py.schema import make_grafast_schema
from examples.seed import setup_demo_schema, setup_things_table


async def run(schema, query, variables=None):
    """Run a query through our engine over the convenience engine for the request."""
    with pg_request_context(SQLAlchemyExecutor(get_engine())):
        return await graphql(
            schema,
            query,
            variable_values=variables,
            execution_context_class=GrafastExecutionContext,
        )


def build_posts_schema(order_by, *, order_is_unique=False):
    """A one-resource schema exposing ``Query.posts`` ordered by ``order_by``."""
    registry = PgRegistry()
    posts = PgResource(
        "posts", "grafast_demo", "posts", ["id", "author_id", "title"], registry=registry
    )

    def plan_posts(parent_step, args, info):
        return PgSelectAllStep(
            posts, order_by=order_by, order_is_unique=order_is_unique
        ).for_parent(parent_step)

    sdl = """
    type Query { posts: [Post!]! }
    type Post { id: Int! author_id: Int! title: String! }
    """
    plans = {
        "Query": {"posts": plan_posts},
        "Post": {
            "id": lambda p, a, i: access(p, ("id",)),
            "author_id": lambda p, a, i: access(p, ("author_id",)),
            "title": lambda p, a, i: access(p, ("title",)),
        },
    }
    return make_grafast_schema(sdl, plans)


def build_things_schema(order_by):
    """A one-resource schema over ``grafast_demo.things`` (nullable ``rank`` column)."""
    registry = PgRegistry()
    things = PgResource(
        "things", "grafast_demo", "things", ["id", "rank"], registry=registry
    )

    def plan_things(parent_step, args, info):
        return PgSelectAllStep(things, order_by=order_by).for_parent(parent_step)

    sdl = """
    type Query { things: [Thing!]! }
    type Thing { id: Int! rank: Int }
    """
    plans = {
        "Query": {"things": plan_things},
        "Thing": {
            "id": lambda p, a, i: access(p, ("id",)),
            "rank": lambda p, a, i: access(p, ("rank",)),
        },
    }
    return make_grafast_schema(sdl, plans)


@pytest_asyncio.fixture
async def seeded():
    """Reseed ``grafast_demo`` (demo tables + the ``things`` fixture) for each test."""
    await dispose_engine()
    await setup_demo_schema()
    await setup_things_table()
    yield
    await dispose_engine()


# --------------------------------------------------------------------- direction


@pytest.mark.pg
@pytest.mark.asyncio
async def test_descending_direction_orders_desc(seeded):
    """``id DESC`` returns strictly descending ids in ONE statement."""
    schema = build_posts_schema([OrderTerm("id", descending=True)])
    with count_sql(get_engine()) as counter:
        result = await run(schema, "{ posts { id } }")
    assert result.errors is None
    assert counter.count == 1
    ids = [p["id"] for p in result.data["posts"]]
    assert ids == sorted(ids, reverse=True)
    assert ids[0] > ids[-1]


# ------------------------------------------------------------------- multi-column


@pytest.mark.pg
@pytest.mark.asyncio
async def test_multi_column_author_asc_id_desc(seeded):
    """``(author_id ASC, id DESC)`` orders by author then descending id within author."""
    schema = build_posts_schema(
        [OrderTerm("author_id"), OrderTerm("id", descending=True)]
    )
    with count_sql(get_engine()) as counter:
        result = await run(schema, "{ posts { id author_id } }")
    assert result.errors is None
    assert counter.count == 1

    rows = [(p["author_id"], p["id"]) for p in result.data["posts"]]
    # author_id ascending overall; within each author block id descending.
    assert [r[0] for r in rows] == sorted(r[0] for r in rows)
    by_author: dict[int, list[int]] = {}
    for author_id, post_id in rows:
        by_author.setdefault(author_id, []).append(post_id)
    for ids in by_author.values():
        assert ids == sorted(ids, reverse=True)


# ----------------------------------------------------------------- PK tie-break


@pytest.mark.pg
@pytest.mark.asyncio
async def test_non_unique_order_appends_pk_for_determinism(seeded):
    """Ordering by a non-unique column appends the PK, yielding deterministic output."""
    schema = build_posts_schema([OrderTerm("author_id")])
    posts = PgResource("posts", "grafast_demo", "posts", ["id", "author_id", "title"])
    step = PgSelectAllStep(posts, order_by=[OrderTerm("author_id")])
    # the normalized order carries the appended PK tie-break.
    assert step.order_by == (OrderTerm("author_id"), OrderTerm("id"))

    result = await run(schema, "{ posts { id author_id } }")
    assert result.errors is None
    rows = [(p["author_id"], p["id"]) for p in result.data["posts"]]
    # within each author the ids ascend (the PK tie-break), so the order is total.
    by_author: dict[int, list[int]] = {}
    for author_id, post_id in rows:
        by_author.setdefault(author_id, []).append(post_id)
    for ids in by_author.values():
        assert ids == sorted(ids)


def test_order_is_unique_does_not_append_pk():
    """An already-unique order does NOT append a duplicate PK term (inspect the SQL)."""
    posts = PgResource("posts", "grafast_demo", "posts", ["id", "author_id", "title"])
    step = PgSelectStep(
        posts, constant(None), "author_id",
        order_by=[OrderTerm("id", descending=True)], order_is_unique=True,
    )
    # the normalized order is exactly the one term — no appended PK tie-break.
    assert step.order_by == (OrderTerm("id", descending=True),)
    sql = str(step.build_query())
    assert sql.split("ORDER BY", 1)[1].strip() == "id DESC"


# ----------------------------------------------------------------------- NULLS


@pytest.mark.pg
@pytest.mark.asyncio
async def test_nulls_first_places_null_rows_first(seeded):
    """``rank NULLS FIRST`` puts the NULL-rank rows ahead of the ranked ones."""
    schema = build_things_schema([OrderTerm("rank", nulls="first")])
    result = await run(schema, "{ things { id rank } }")
    assert result.errors is None
    ranks = [t["rank"] for t in result.data["things"]]
    # the NULLs (rows 2 and 4) come first, then ascending ranks 10, 20, 30.
    assert ranks[:2] == [None, None]
    assert ranks[2:] == [10, 20, 30]


@pytest.mark.pg
@pytest.mark.asyncio
async def test_nulls_last_places_null_rows_last(seeded):
    """``rank NULLS LAST`` puts the NULL-rank rows after the ranked ones."""
    schema = build_things_schema([OrderTerm("rank", nulls="last")])
    result = await run(schema, "{ things { id rank } }")
    assert result.errors is None
    ranks = [t["rank"] for t in result.data["things"]]
    assert ranks[:3] == [10, 20, 30]
    assert ranks[3:] == [None, None]


# ------------------------------------------------------------- dedup correctness


def dedup_key(step):
    """The class + peer_key + dedup_params slice of the planner's structural key.

    The full key in :func:`grafast_py.dag._structural_key` also folds the dependency
    survivor ids; these steps share an identical ``constant(None)`` dep shape, so the
    ordering discriminator lives entirely in ``peer_key`` / ``dedup_params``.
    """
    return (type(step), step.peer_key, step.dedup_params())


def test_different_order_specs_do_not_dedup():
    """Two selects over the same resource/match but different order are NOT peers."""
    posts = PgResource("posts", "grafast_demo", "posts", ["id", "author_id", "title"])
    asc = PgSelectStep(posts, constant(None), "author_id", order_by=[OrderTerm("id")])
    desc = PgSelectStep(
        posts, constant(None), "author_id", order_by=[OrderTerm("id", descending=True)]
    )
    assert asc.peer_key != desc.peer_key
    assert asc.dedup_params() != desc.dedup_params()
    assert dedup_key(asc) != dedup_key(desc)


def test_identical_order_specs_dedup():
    """Two selects with the same order spec ARE peers (same dedup key)."""
    posts = PgResource("posts", "grafast_demo", "posts", ["id", "author_id", "title"])
    a = PgSelectStep(posts, constant(None), "author_id", order_by=[OrderTerm("id")])
    b = PgSelectStep(posts, constant(None), "author_id", order_by=[OrderTerm("id")])
    assert a.peer_key == b.peer_key
    assert a.dedup_params() == b.dedup_params()
    assert dedup_key(a) == dedup_key(b)


def test_string_and_orderterm_ascending_normalize_equal():
    """A string column and an ascending OrderTerm normalize to the same dedup key."""
    posts = PgResource("posts", "grafast_demo", "posts", ["id", "author_id", "title"])
    as_str = PgSelectStep(posts, constant(None), "author_id", order_by=["title"])
    as_term = PgSelectStep(
        posts, constant(None), "author_id", order_by=[OrderTerm("title")]
    )
    assert as_str.order_by == as_term.order_by
    assert as_str.peer_key == as_term.peer_key
    assert as_str.dedup_params() == as_term.dedup_params()
    assert dedup_key(as_str) == dedup_key(as_term)


def test_order_is_unique_with_empty_order_still_total():
    """``order_is_unique`` with no terms still yields a total order (PK floor)."""
    assert normalize_order([], primary_key="id", order_is_unique=True) == (
        OrderTerm("id"),
    )
    assert normalize_order(None, primary_key="id", order_is_unique=True) == (
        OrderTerm("id"),
    )


def test_same_effective_order_dedups_regardless_of_unique_flag():
    """When the effective order is identical, the order_is_unique flag must not block
    a merge — the normalized order_by tuple is the sole ordering discriminator."""
    posts = PgResource("posts", "grafast_demo", "posts", ["id", "author_id", "title"])
    # order_by already contains the PK, so both normalize to (OrderTerm("id"),).
    trusted = PgSelectStep(
        posts, constant(None), "author_id", order_by=["id"], order_is_unique=True
    )
    untrusted = PgSelectStep(
        posts, constant(None), "author_id", order_by=["id"], order_is_unique=False
    )
    assert trusted.order_by == untrusted.order_by
    assert dedup_key(trusted) == dedup_key(untrusted)
