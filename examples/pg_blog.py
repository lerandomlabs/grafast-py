"""End-to-end Postgres-backed GraphQL demo for grafast-py (Phase B).

Builds the ``grafast_demo`` schema (authors -> posts -> comments) in the scratch
database ``grafast_py_test``, then runs nested GraphQL operations through
:class:`grafast_py.GrafastExecutionContext` and prints both the result and the
number of SQL statements each operation issued.

The whole point of the Postgres data source is the same as Grafast's
``@dataplan/pg``: relations are *plan resolvers* that build batched ``pg_select`` /
``pg_select_single`` steps, so a depth-D nested query issues ~D SQL statements
TOTAL (one parameterised ``WHERE col = ANY($1)`` per resource-layer) rather than
one query per row. We prove that here with :func:`count_sql`, an actual
``before_cursor_execute`` statement counter — the count tracks DEPTH, not ROWS.

SAFETY: the only database touched is ``grafast_py_test`` and the only schema is
``grafast_demo`` (dropped + recreated idempotently by ``setup_demo_schema``).

Run:  uv run python examples/pg_blog.py
"""

import asyncio
import sys
from pathlib import Path

# run as a script: put the repo root on sys.path so the sibling `examples` package
# (demo_schema/seed) is importable as `examples.*`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphql import graphql

from grafast_py import GrafastExecutionContext
from examples.demo_schema import build_demo_schema
from grafast_py.pg.engine import count_sql, dispose_engine, get_engine
from grafast_py.pg.executor import SQLAlchemyExecutor, pg_request_context
from examples.seed import setup_demo_schema


async def run(schema, query, variables=None):
    """Execute one operation through our plan-then-execute engine, counting SQL.

    Builds a :class:`SQLAlchemyExecutor` over the convenience engine and binds it for
    the request, so the pg steps run their statements via the request-scoped executor;
    ``count_sql`` instruments that same engine.
    """
    engine = get_engine()
    executor = SQLAlchemyExecutor(engine)
    with count_sql(engine) as counter:
        with pg_request_context(executor):
            result = await graphql(
                schema,
                query,
                variable_values=variables,
                execution_context_class=GrafastExecutionContext,
            )
    if result.errors:
        raise AssertionError(f"query errored: {result.errors}")
    return result.data, counter


def banner(text: str) -> None:
    print("=" * 72)
    print(text)
    print("=" * 72)


async def main() -> None:
    banner("0. SEED grafast_demo (drop + recreate authors/posts/comments)")
    await setup_demo_schema()
    print("   seeded 3 authors; author i has i+1 posts; 2 comments per post.")
    print("   => 3 authors, 9 posts, 18 comments.\n")

    schema = build_demo_schema()

    # ---------------------------------------------------------------- depth 1
    banner("1. ROOT LIST (depth 1) — one SELECT for all authors")
    data, counter = await run(schema, "{ authors { id name } }")
    print(f"   data:  {data}")
    print(f"   SQL statements issued: {counter.count}  (expected 1)\n")

    # ---------------------------------------------------------------- depth 2
    banner("2. authors -> posts (depth 2) — one SELECT per layer")
    data, counter = await run(schema, "{ authors { name posts { id title } } }")
    post_counts = [len(a["posts"]) for a in data["authors"]]
    print(f"   posts per author: {post_counts}  (9 posts across 3 authors)")
    print(f"   SQL statements issued: {counter.count}  (expected 2 — NOT 1+3)\n")

    # ---------------------------------------------------------------- depth 3
    banner("3. authors -> posts -> comments (depth 3)")
    data, counter = await run(
        schema, "{ authors { name posts { title comments { id body } } } }"
    )
    total_comments = sum(
        len(p["comments"]) for a in data["authors"] for p in a["posts"]
    )
    print(f"   total comments materialised: {total_comments}  (18)")
    print(f"   SQL statements issued: {counter.count}  (expected 3 — NOT one-per-row)\n")

    # ----------------------------------------------------- deepest nested query
    banner("4. DEEPEST nested query — 5 resource-layers, O(depth) SQL")
    deep_query = """
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
    data, counter = await run(schema, deep_query)
    n_authors = len(data["authors"])
    n_posts = sum(len(a["posts"]) for a in data["authors"])
    n_comments = sum(
        len(p["comments"]) for a in data["authors"] for p in a["posts"]
    )
    rows_touched = n_authors + n_posts + n_comments
    print("   layers: authors -> posts -> {author, comments -> author}")
    print(f"   rows touched: {rows_touched}  ({n_authors} authors / {n_posts} posts"
          f" / {n_comments} comments)")
    print(f"   SQL statements issued: {counter.count}")
    print(f"   => {counter.count} statements over {rows_touched} rows: the count tracks"
          " DEPTH, not ROWS. A naive resolver would fire dozens.\n")
    for i, stmt in enumerate(counter.statements, 1):
        print(f"     SQL {i}: {' '.join(stmt.split())[:90]}")
    print()

    # ----------------------------------------------------- Relay connection
    banner("5. RELAY CONNECTION — postsConnection(first) batched across authors")
    conn_query = """
    {
      authors {
        name
        postsConnection(first: 2) {
          totalCount
          edges { cursor node { id } }
          pageInfo { hasNextPage endCursor }
        }
      }
    }
    """
    data, counter = await run(schema, conn_query)
    for a in data["authors"]:
        c = a["postsConnection"]
        print(f"   {a['name']:<14} totalCount={c['totalCount']} "
              f"edges={len(c['edges'])} hasNextPage={c['pageInfo']['hasNextPage']}")
    print(f"   SQL statements issued: {counter.count}  (authors + ONE windowed"
          " posts query for ALL authors)\n")

    banner("DONE — every relation layer is ONE batched SQL statement (O(depth)).")

    await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
