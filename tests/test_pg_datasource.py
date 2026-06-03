"""DB-backed tests for the Postgres data source (Phase B).

These create/drop ONLY within the ``grafast_demo`` schema of ``grafast_py_test``
(the single permitted database) via :func:`setup_demo_schema`, then run live GraphQL
operations through :class:`GrafastExecutionContext` and assert both correctness AND
the batching profile: a depth-D nested query issues ~D SQL statements (one per
resource-layer), proven by the :func:`count_sql` statement counter — NOT O(rows).

Marked ``pg`` so a no-DB run can deselect them (``-m 'not pg'``).
"""

import pytest
import pytest_asyncio
from graphql import graphql

from grafast_py.context import GrafastExecutionContext
from grafast_py.pg.connection import encode_cursor
from examples.demo_schema import build_demo_schema
from grafast_py.pg.engine import count_sql, dispose_engine
from grafast_py.pg.resource import PgRegistry, PgResource
from examples.seed import setup_demo_schema
from grafast_py.pg.steps import PgSelectSingleStep, PgSelectStep

pytestmark = pytest.mark.pg


async def run(schema, query, variables=None):
    """Run a query through our engine (not the stock executor)."""
    return await graphql(
        schema,
        query,
        variable_values=variables,
        execution_context_class=GrafastExecutionContext,
    )


@pytest_asyncio.fixture
async def demo_schema():
    """Build the demo schema after (re)seeding ``grafast_demo`` for each test.

    Function-scoped so each test runs on its own event loop with a fresh engine —
    the async engine's connection pool is bound to the loop that created it, so a
    module-scoped engine across per-test loops would error. Reseeding is cheap.
    """
    await dispose_engine()
    await setup_demo_schema()
    schema = build_demo_schema()
    yield schema
    await dispose_engine()


# ----------------------------------------------------------------- batch steps


@pytest.mark.asyncio
async def test_pg_select_batches_all_keys_into_one_statement():
    """PgSelectStep over N keys issues exactly ONE SQL statement (WHERE = ANY)."""
    await dispose_engine()
    await setup_demo_schema()
    registry = PgRegistry()
    posts = PgResource(
        "posts", "grafast_demo", "posts", ["id", "author_id", "title"], registry=registry
    )
    from grafast_py.core_steps import constant

    step = PgSelectStep(posts, constant(None), "author_id", order_by=["id"])
    with count_sql() as counter:
        out = step.execute(3, [[1, 2, 3]])
        out = await out
    assert counter.count == 1
    # author i has i+1 posts in the seed
    assert [len(out[0]), len(out[1]), len(out[2])] == [2, 3, 4]
    await dispose_engine()


@pytest.mark.asyncio
async def test_pg_select_single_missing_key_is_null():
    """A missing key scatters to None (single); present keys to their row."""
    await dispose_engine()
    await setup_demo_schema()
    registry = PgRegistry()
    authors = PgResource(
        "authors", "grafast_demo", "authors", ["id", "name"], registry=registry
    )
    from grafast_py.core_steps import constant

    step = PgSelectSingleStep(authors, constant(None), "id")
    out = await step.execute(3, [[1, 999, 2]])
    assert out[0]["name"] == "Ada Lovelace"
    assert out[1] is None
    assert out[2]["name"] == "Alan Turing"
    await dispose_engine()


# ----------------------------------------------------------- end-to-end O(depth)


@pytest.mark.asyncio
async def test_nested_query_is_o_depth_not_o_rows(demo_schema):
    """A 3-level nested query issues one batched statement per resource-layer.

    authors -> posts (hasMany) -> [author (hasOne), comments (hasMany) -> author]
    touches 5 resource-layers; with 9 posts / 18 comments a naive resolver would fire
    dozens of queries. We assert exactly 5 statements, proving O(depth).
    """
    query = """
    {
      authors {
        id
        name
        posts {
          id
          title
          author { id name }
          comments { id body author { name } }
        }
      }
    }
    """
    with count_sql() as counter:
        result = await run(demo_schema, query)

    assert result.errors is None
    assert counter.count == 5

    authors = result.data["authors"]
    assert [a["name"] for a in authors] == ["Ada Lovelace", "Alan Turing", "Grace Hopper"]
    # author i has i+1 posts
    assert [len(a["posts"]) for a in authors] == [2, 3, 4]
    first_post = authors[0]["posts"][0]
    assert first_post["author"]["name"] == "Ada Lovelace"
    assert len(first_post["comments"]) == 2
    assert first_post["comments"][0]["author"]["name"]  # resolved, non-empty


@pytest.mark.asyncio
async def test_deep_query_statement_count_independent_of_rows(demo_schema):
    """Adding more rows-per-layer does not add statements: count stays at depth."""
    shallow = "{ authors { id posts { id } } }"
    with count_sql() as counter_shallow:
        r1 = await run(demo_schema, shallow)
    assert r1.errors is None
    assert counter_shallow.count == 2  # authors + posts, regardless of 9 posts

    deep = "{ authors { id posts { id comments { id } } } }"
    with count_sql() as counter_deep:
        r2 = await run(demo_schema, deep)
    assert r2.errors is None
    assert counter_deep.count == 3  # authors + posts + comments


@pytest.mark.asyncio
async def test_hasone_relation_resolves_single_parent(demo_schema):
    """Post.author (hasOne) returns the single owning author row."""
    query = "{ posts { id author { id name } } }"
    with count_sql() as counter:
        result = await run(demo_schema, query)
    assert result.errors is None
    assert counter.count == 2  # posts + authors
    posts = result.data["posts"]
    assert posts[0]["author"]["name"] == "Ada Lovelace"
    # post 1..2 belong to author 1, post 3..5 to author 2, etc.
    assert posts[2]["author"]["name"] == "Alan Turing"


@pytest.mark.asyncio
async def test_root_argument_lookup(demo_schema):
    """Query.author(id) fetches one author by primary key."""
    result = await run(demo_schema, "{ author(id: 2) { id name } }")
    assert result.errors is None
    assert result.data["author"] == {"id": 2, "name": "Alan Turing"}

    missing = await run(demo_schema, "{ author(id: 999) { id name } }")
    assert missing.errors is None
    assert missing.data["author"] is None


# ----------------------------------------------------------------- connections


@pytest.mark.asyncio
async def test_connection_paging_batched_across_parents(demo_schema):
    """postsConnection(first) pages every author in ONE windowed statement."""
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
    with count_sql() as counter:
        result = await run(demo_schema, query)
    assert result.errors is None
    assert counter.count == 2  # authors + ONE window query for all authors' posts

    grace = result.data["authors"][2]
    conn = grace["postsConnection"]
    assert conn["totalCount"] == 4
    assert len(conn["edges"]) == 2
    assert conn["pageInfo"]["hasNextPage"] is True
    assert conn["pageInfo"]["hasPreviousPage"] is False
    assert conn["pageInfo"]["endCursor"] == conn["edges"][-1]["cursor"]

    ada = result.data["authors"][0]
    assert ada["postsConnection"]["totalCount"] == 2
    assert ada["postsConnection"]["pageInfo"]["hasNextPage"] is False


@pytest.mark.asyncio
async def test_connection_after_cursor(demo_schema):
    """An `after` cursor advances the page; hasNextPage flips at the end."""
    after = encode_cursor(2)
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
    result = await run(demo_schema, query, {"after": after})
    assert result.errors is None
    conn = result.data["author"]["postsConnection"]
    assert [e["node"]["id"] for e in conn["edges"]] == [8, 9]
    assert conn["pageInfo"]["hasNextPage"] is False
    assert conn["pageInfo"]["hasPreviousPage"] is True
