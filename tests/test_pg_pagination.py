"""Per-parent FIRST/OFFSET pushed INTO the SQL via a window-function slice.

A limited relation/connection must fetch ONLY each parent's page rows in ONE batched
statement — never fetch-all-then-slice-in-Python, never a bucket-wide ``LIMIT`` (which
would limit the whole ``= ANY($1)`` result across ALL parents). These tests assert:

- a limited ``PgSelectStep`` returns at most ``first`` rows PER PARENT in one statement,
  and the page slice lives IN the compiled SQL (``row_number`` + a parameterised ``__rn``
  filter) — not Python;
- ``offset`` advances the per-parent page;
- the unlimited path is byte-for-byte the old plain SELECT (no ``row_number`` overhead);
- the Relay connection keeps correct edges/totalCount/hasNextPage, INCLUDING ``totalCount``
  on an empty terminal page (it fetches all rows + slices in Python so the per-parent
  ``count(*) OVER`` total survives an empty page — slicing the connection in SQL with a
  separate ``totalCount`` aggregate is a Phase-6 concern);
- dedup discriminates on ``first``/``offset`` (different bounds never merge; identical do).

DB tests are marked ``pg`` (deselectable ``-m 'not pg'``); they touch ONLY the
``grafast_demo`` schema of ``grafast_py_test`` and do not alter authors/posts/comments.
"""

import pytest
import pytest_asyncio
from graphql import graphql

from grafast_py.context import GrafastExecutionContext
from grafast_py.core_steps import constant
from grafast_py.pg.engine import count_sql, dispose_engine, get_engine
from grafast_py.pg.executor import SQLAlchemyExecutor, pg_request_context
from grafast_py.pg.ordering import OrderTerm
from grafast_py.pg.resource import PgRegistry, PgResource
from grafast_py.pg.steps import PgSelectStep
from examples.demo_schema import build_demo_schema
from examples.seed import setup_demo_schema

pytestmark = pytest.mark.pg


async def run(schema, query, variables=None):
    """Run a query through our engine over the convenience engine for the request."""
    with pg_request_context(SQLAlchemyExecutor(get_engine())):
        return await graphql(
            schema,
            query,
            variable_values=variables,
            execution_context_class=GrafastExecutionContext,
        )


def make_posts_resource():
    """A fresh ``posts`` resource (its own registry) for the plain-select tests."""
    registry = PgRegistry()
    return PgResource(
        "posts", "grafast_demo", "posts", ["id", "author_id", "title"], registry=registry
    )


@pytest_asyncio.fixture
async def demo_schema():
    """(Re)seed ``grafast_demo`` and build the demo schema (fresh engine per test)."""
    await dispose_engine()
    await setup_demo_schema()
    schema = build_demo_schema()
    yield schema
    await dispose_engine()


# ----------------------------------------------------------- plain select first


@pytest.mark.asyncio
async def test_plain_select_first_limits_per_parent_in_one_statement():
    """``first=2`` returns at most 2 posts PER author in ONE windowed statement."""
    await dispose_engine()
    await setup_demo_schema()
    posts = make_posts_resource()
    step = PgSelectStep(posts, constant(None), "author_id", order_by=["id"], first=2)

    engine = get_engine()
    with count_sql(engine) as counter:
        with pg_request_context(SQLAlchemyExecutor(engine)):
            out = await step.execute(3, [[1, 2, 3]])
    assert counter.count == 1

    # author i has i+1 posts (2/3/4), but first=2 caps EACH parent at 2.
    assert [len(out[0]), len(out[1]), len(out[2])] == [2, 2, 2]
    # author 1 owns posts 1..2; the first two are exactly those.
    assert [r["id"] for r in out[0]] == [1, 2]
    # author 3 owns posts 6..9; first two are 6, 7 (not the whole bucket's first two).
    assert [r["id"] for r in out[2]] == [6, 7]

    # the slice is IN SQL: row_number + a parameterised __rn filter, not Python.
    sql = str(step.build_query())
    assert "row_number" in sql.lower()
    assert "__rn > :offset" in sql
    assert step.is_limited is True
    await dispose_engine()


@pytest.mark.asyncio
async def test_plain_select_offset_advances_per_parent_page():
    """``first=2, offset=1`` returns the 2nd-3rd post per author (window OFFSET)."""
    await dispose_engine()
    await setup_demo_schema()
    posts = make_posts_resource()
    step = PgSelectStep(
        posts, constant(None), "author_id", order_by=["id"], first=2, offset=1
    )

    engine = get_engine()
    with count_sql(engine) as counter:
        with pg_request_context(SQLAlchemyExecutor(engine)):
            out = await step.execute(3, [[1, 2, 3]])
    assert counter.count == 1

    # author 1 owns posts 1,2 -> offset 1 skips post 1, leaving just post 2.
    assert [r["id"] for r in out[0]] == [2]
    # author 2 owns posts 3,4,5 -> 2nd..3rd are posts 4, 5.
    assert [r["id"] for r in out[1]] == [4, 5]
    # author 3 owns posts 6,7,8,9 -> 2nd..3rd are posts 7, 8.
    assert [r["id"] for r in out[2]] == [7, 8]
    await dispose_engine()


@pytest.mark.asyncio
async def test_no_limit_path_is_plain_select_no_window():
    """``first=None, offset=0`` keeps the plain SELECT — NO row_number window."""
    await dispose_engine()
    await setup_demo_schema()
    posts = make_posts_resource()
    step = PgSelectStep(posts, constant(None), "author_id", order_by=["id"])

    assert step.is_limited is False
    sql = str(step.build_query())
    assert "row_number" not in sql.lower()
    assert "__rn" not in sql

    engine = get_engine()
    with count_sql(engine) as counter:
        with pg_request_context(SQLAlchemyExecutor(engine)):
            out = await step.execute(3, [[1, 2, 3]])
    assert counter.count == 1
    # unlimited: every matching row comes back (2/3/4 posts).
    assert [len(out[0]), len(out[1]), len(out[2])] == [2, 3, 4]
    await dispose_engine()


# ----------------------------------------------------------------- connection paging


@pytest.mark.asyncio
async def test_connection_paging_correct(demo_schema):
    """The connection pages every parent in ONE statement with correct totals/flags."""
    query = """
    {
      authors {
        name
        postsConnection(first: 2) {
          totalCount
          edges { cursor node { id } }
          pageInfo { hasNextPage hasPreviousPage startCursor endCursor }
        }
      }
    }
    """
    with count_sql(get_engine()) as counter:
        result = await run(demo_schema, query)
    assert result.errors is None
    # authors (1) + connection page + separate totalCount aggregate = 3 (O(depth)).
    assert counter.count == 3

    grace = result.data["authors"][2]
    conn = grace["postsConnection"]
    assert conn["totalCount"] == 4  # the separate count is the FULL per-parent count
    assert len(conn["edges"]) == 2
    assert conn["pageInfo"]["hasNextPage"] is True
    assert conn["pageInfo"]["hasPreviousPage"] is False
    assert conn["pageInfo"]["endCursor"] == conn["edges"][-1]["cursor"]

    ada = result.data["authors"][0]
    assert ada["postsConnection"]["totalCount"] == 2
    assert ada["postsConnection"]["pageInfo"]["hasNextPage"] is False


@pytest.mark.asyncio
async def test_connection_after_cursor_advances_page(demo_schema):
    """An ``after`` (keyset) cursor advances the page to the parent's later rows."""
    first_page = await run(
        demo_schema,
        """
        { author(id: 3) { postsConnection(first: 2) {
            pageInfo { endCursor } } } }
        """,
    )
    end_cursor = first_page.data["author"]["postsConnection"]["pageInfo"]["endCursor"]

    query = """
    query Page($after: String!) {
      author(id: 3) {
        postsConnection(first: 2, after: $after) {
          edges { node { id } }
          pageInfo { hasNextPage hasPreviousPage }
        }
      }
    }
    """
    result = await run(demo_schema, query, {"after": end_cursor})
    assert result.errors is None
    conn = result.data["author"]["postsConnection"]
    assert [e["node"]["id"] for e in conn["edges"]] == [8, 9]
    assert conn["pageInfo"]["hasNextPage"] is False
    assert conn["pageInfo"]["hasPreviousPage"] is True


@pytest.mark.asyncio
async def test_connection_empty_terminal_page_keeps_totalcount(demo_schema):
    """An ``after`` cursor at/past the last row yields an EMPTY page but a correct total.

    totalCount comes from the SEPARATE batched aggregate (unaffected by first/after), so
    it survives even when no page rows remain — the Phase-3 regression (an empty SQL slice
    dropping the total) cannot recur. totalCount must stay the real count, hasPreviousPage
    true.
    """
    # author 1 owns exactly 2 posts; an `after` at the last edge yields an empty terminal
    # page whose totalCount must still be 2, not 0.
    walk = await run(
        demo_schema,
        """
        { author(id: 1) { postsConnection(first: 2) {
            edges { cursor } pageInfo { endCursor } } } }
        """,
    )
    end_cursor = walk.data["author"]["postsConnection"]["pageInfo"]["endCursor"]

    query = """
    query Page($after: String!) {
      author(id: 1) {
        postsConnection(first: 2, after: $after) {
          totalCount
          edges { node { id } }
          pageInfo { hasNextPage hasPreviousPage }
        }
      }
    }
    """
    result = await run(demo_schema, query, {"after": end_cursor})
    assert result.errors is None
    conn = result.data["author"]["postsConnection"]
    assert conn["edges"] == []
    assert conn["totalCount"] == 2
    assert conn["pageInfo"]["hasNextPage"] is False
    assert conn["pageInfo"]["hasPreviousPage"] is True


# --------------------------------------------------------------- dedup correctness


def dedup_key(step):
    """The class + peer_key + dedup_params slice of the planner's structural key."""
    return (type(step), step.peer_key, step.dedup_params())


def test_different_first_does_not_dedup():
    """Two selects differing only in ``first`` are NOT peers (different SQL slice)."""
    posts = make_posts_resource()
    a = PgSelectStep(posts, constant(None), "author_id", order_by=["id"], first=2)
    b = PgSelectStep(posts, constant(None), "author_id", order_by=["id"], first=3)
    assert a.peer_key != b.peer_key
    assert a.dedup_params() != b.dedup_params()
    assert dedup_key(a) != dedup_key(b)


def test_different_offset_does_not_dedup():
    """Two selects differing only in ``offset`` are NOT peers."""
    posts = make_posts_resource()
    a = PgSelectStep(posts, constant(None), "author_id", order_by=["id"], first=2)
    b = PgSelectStep(
        posts, constant(None), "author_id", order_by=["id"], first=2, offset=1
    )
    assert a.peer_key != b.peer_key
    assert a.dedup_params() != b.dedup_params()
    assert dedup_key(a) != dedup_key(b)


def test_identical_first_offset_dedup():
    """Two selects with the same first/offset (and order) ARE peers."""
    posts = make_posts_resource()
    a = PgSelectStep(
        posts, constant(None), "author_id",
        order_by=[OrderTerm("id")], first=2, offset=1,
    )
    b = PgSelectStep(
        posts, constant(None), "author_id",
        order_by=[OrderTerm("id")], first=2, offset=1,
    )
    assert a.peer_key == b.peer_key
    assert a.dedup_params() == b.dedup_params()
    assert dedup_key(a) == dedup_key(b)
