"""N+1 benchmark: prove batched SQL count is O(depth), constant in N.

Two proofs, scaled across ``N in {10, 50, 200, 1000}``:

1. **Postgres data source** — scale-seed ``grafast_demo`` to N authors (fixed
   fan-out: ``POSTS_PER_AUTHOR`` posts/author, ``COMMENTS_PER_POST`` comments/post,
   so rows grow linearly in N), then run the canonical nested query
   ``{ authors { id name posts { id title comments { id body } } } }`` (depth 3 in
   resource layers) ONCE inside ``count_sql()`` and record the statement count. The
   PASS invariant: the batched count == 3 for EVERY N (root authors + posts ANY +
   comments ANY) — it tracks DEPTH, not ROWS. For contrast we also report what a
   naive per-row resolver would issue: ``1 + N + POSTS_PER_AUTHOR * N`` statements
   (one root + one per author + one per post), i.e. O(N).

2. **Generic in-memory loadMany** — the source-agnostic batch path
   (``core_steps.load_many``) over the same shape, with a call counter living INSIDE
   the user batch callback (the independent-observation pattern). PASS invariant:
   exactly 1 batch-callback invocation at every N (the loader receives all N keys in
   one call), vs N for a naive per-parent resolver.

Latency (median + p95, ms) is measured with ``time.perf_counter()`` over R repeats
(after a discarded warmup) and REPORTED, not hard-thresholded — it scales with rows
materialised, which is expected and not a defect.

SAFETY: the only database touched is ``grafast_py_test`` and the only schema is
``grafast_demo``. The seeder imports the shared engine (``get_engine``) and never
constructs its own connection URL; all DDL/DML is confined to ``grafast_demo``.

Run:  uv run python bench/bench_nplus1.py
"""

import asyncio
import statistics
import sys
import time
from pathlib import Path
from typing import Any, List

# run as a script: put the repo root on sys.path so `examples` (demo fixtures) imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphql import GraphQLSchema, graphql, graphql_sync, parse
from graphql import (
    GraphQLField,
    GraphQLInt,
    GraphQLList,
    GraphQLNonNull,
    GraphQLObjectType,
    GraphQLString,
)
from sqlalchemy import text

from grafast_py import (
    GrafastExecutionContext,
    access,
    constant,
    load_many,
    make_grafast_schema,
)
from examples.demo_schema import build_demo_schema
from grafast_py.pg.engine import count_sql, dispose_engine, get_engine
from grafast_py.pg.executor import SQLAlchemyExecutor, pg_request_context
from examples.seed import DEMO_SCHEMA

# fixed fan-out so rows scale LINEARLY in N (N authors -> dataset bounded even at
# N=1000: 1k authors + 5k posts + 20k comments = 26k rows).
POSTS_PER_AUTHOR = 5
COMMENTS_PER_POST = 4

N_VALUES = [10, 50, 200, 1000]

# the canonical depth-3 nested query (authors -> posts -> comments).
NESTED_QUERY = "{ authors { id name posts { id title comments { id body } } } }"

# repeats per N for the latency sample (after one discarded warmup run).
LATENCY_REPEATS = 30

RESULTS_MD = Path(__file__).parent / "results.md"


# --------------------------------------------------------------- scale-seeding


async def scale_seed(n: int) -> None:
    """Idempotently (re)seed ``grafast_demo`` with N authors + fixed fan-out.

    Drops and recreates the schema each call (re-runnable), then bulk-inserts
    deterministic rows. Confined strictly to ``grafast_demo`` in ``grafast_py_test``
    via the shared engine — no other database is reachable.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP SCHEMA IF EXISTS {DEMO_SCHEMA} CASCADE"))
        await conn.execute(text(f"CREATE SCHEMA {DEMO_SCHEMA}"))
        await conn.execute(
            text(
                f"""
                CREATE TABLE {DEMO_SCHEMA}.authors (
                    id   integer PRIMARY KEY,
                    name text NOT NULL
                )
                """
            )
        )
        await conn.execute(
            text(
                f"""
                CREATE TABLE {DEMO_SCHEMA}.posts (
                    id        integer PRIMARY KEY,
                    author_id integer NOT NULL REFERENCES {DEMO_SCHEMA}.authors (id),
                    title     text NOT NULL
                )
                """
            )
        )
        await conn.execute(
            text(
                f"""
                CREATE TABLE {DEMO_SCHEMA}.comments (
                    id        integer PRIMARY KEY,
                    post_id   integer NOT NULL REFERENCES {DEMO_SCHEMA}.posts (id),
                    author_id integer NOT NULL REFERENCES {DEMO_SCHEMA}.authors (id),
                    body      text NOT NULL
                )
                """
            )
        )

        authors = [{"id": a, "name": f"Author {a}"} for a in range(1, n + 1)]

        posts = []
        post_id = 1
        for author_id in range(1, n + 1):
            for _ in range(POSTS_PER_AUTHOR):
                posts.append(
                    {
                        "id": post_id,
                        "author_id": author_id,
                        "title": f"Post {post_id} by author {author_id}",
                    }
                )
                post_id += 1

        comments = []
        comment_id = 1
        for post in posts:
            for _ in range(COMMENTS_PER_POST):
                # commenter cycles over the N authors so Comment.author resolves.
                commenter = ((comment_id - 1) % n) + 1
                comments.append(
                    {
                        "id": comment_id,
                        "post_id": post["id"],
                        "author_id": commenter,
                        "body": f"Comment {comment_id} on post {post['id']}",
                    }
                )
                comment_id += 1

        await conn.execute(
            text(f"INSERT INTO {DEMO_SCHEMA}.authors (id, name) VALUES (:id, :name)"),
            authors,
        )
        await _insert_chunked(
            conn,
            f"INSERT INTO {DEMO_SCHEMA}.posts (id, author_id, title)"
            " VALUES (:id, :author_id, :title)",
            posts,
        )
        await _insert_chunked(
            conn,
            f"INSERT INTO {DEMO_SCHEMA}.comments (id, post_id, author_id, body)"
            " VALUES (:id, :post_id, :author_id, :body)",
            comments,
        )


async def _insert_chunked(conn, sql: str, rows: List[dict], chunk: int = 1000) -> None:
    """Insert rows in modest batches so a large N still loads quickly."""
    stmt = text(sql)
    for start in range(0, len(rows), chunk):
        await conn.execute(stmt, rows[start : start + chunk])


def expected_rows(n: int) -> int:
    """Total rows seeded for N authors under the fixed fan-out."""
    return n + POSTS_PER_AUTHOR * n + COMMENTS_PER_POST * POSTS_PER_AUTHOR * n


def naive_sql_count(n: int) -> int:
    """SQL a naive per-row resolver would issue for the nested query (O(N)).

    1 root authors SELECT + 1 posts SELECT per author + 1 comments SELECT per post
    (POSTS_PER_AUTHOR posts per author). Comments-leaf needs no further query.
    """
    return 1 + n + POSTS_PER_AUTHOR * n


# ---------------------------------------------------------------- pg benchmark


async def run_nested_once(schema: GraphQLSchema):
    """Execute the nested query once through our engine; return (data, errors).

    Binds a :class:`SQLAlchemyExecutor` over the shared engine for the request so the
    pg steps run their statements via the request-scoped executor.
    """
    with pg_request_context(SQLAlchemyExecutor(get_engine())):
        result = await graphql(
            schema,
            NESTED_QUERY,
            execution_context_class=GrafastExecutionContext,
        )
    return result.data, result.errors


def p95(samples: List[float]) -> float:
    """The 95th-percentile of a sample (nearest-rank, robust for small R)."""
    ordered = sorted(samples)
    idx = max(0, min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1)))))
    return ordered[idx]


async def bench_pg(n: int) -> dict:
    """Reseed to N, measure batched SQL count + latency for the nested query."""
    await scale_seed(n)
    schema = build_demo_schema()

    # SQL count: ONE run inside count_sql() at concurrency 1 (count_sql is
    # process-global, so the count must be measured with a single op in flight).
    with count_sql() as counter:
        data, errors = await run_nested_once(schema)
    if errors:
        raise AssertionError(f"nested query errored at N={n}: {errors}")
    sql_count = counter.count

    # spot-check the result shape so we know the rows were actually materialised.
    assert len(data["authors"]) == n, (len(data["authors"]), n)
    first = data["authors"][0]
    assert len(first["posts"]) == POSTS_PER_AUTHOR
    assert len(first["posts"][0]["comments"]) == COMMENTS_PER_POST

    # latency: discard one warmup, then R timed repeats.
    await run_nested_once(schema)
    latencies: List[float] = []
    for _ in range(LATENCY_REPEATS):
        t0 = time.perf_counter()
        await run_nested_once(schema)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return {
        "n": n,
        "sql_count": sql_count,
        "naive_sql_count": naive_sql_count(n),
        "p50_ms": statistics.median(latencies),
        "p95_ms": p95(latencies),
        "rows": expected_rows(n),
        "statements": counter.statements,
    }


# ---------------------------------------------------- in-memory loadMany bench


def build_inmemory_schema(n: int, counter: dict) -> GraphQLSchema:
    """A grafast schema where Author.posts is a loadMany (counter inside callback)."""
    authors = [{"id": i, "name": f"Author {i}"} for i in range(1, n + 1)]
    posts_by_author = {
        i: [
            {"id": i * 1000 + j, "title": f"p{i}-{j}"}
            for j in range(POSTS_PER_AUTHOR)
        ]
        for i in range(1, n + 1)
    }

    def load_posts(keys: List[Any]) -> List[Any]:
        # the independent observation: this counter is INSIDE the user callback, so
        # it directly counts how many times the engine invoked the batch loader.
        counter["calls"] += 1
        counter["max_keys"] = max(counter["max_keys"], len(keys))
        return [posts_by_author.get(k, []) for k in keys]

    def plan_authors(parent, args, info):
        from grafast_py import load_one

        return load_one(constant("all"), lambda _keys: [authors])

    def plan_posts(parent, args, info):
        return load_many(access(parent, ["id"]), load_posts)

    return make_grafast_schema(
        """
        type Query { authors: [Author!]! }
        type Author { id: Int!  name: String!  posts: [Post!]! }
        type Post { id: Int!  title: String! }
        """,
        {
            "Query": {"authors": plan_authors},
            "Author": {
                "id": lambda p, a, i: access(p, ["id"]),
                "name": lambda p, a, i: access(p, ["name"]),
                "posts": plan_posts,
            },
            "Post": {
                "id": lambda p, a, i: access(p, ["id"]),
                "title": lambda p, a, i: access(p, ["title"]),
            },
        },
    )


def build_naive_inmemory_schema(n: int, counter: dict) -> GraphQLSchema:
    """Stock graphql-core schema whose Author.posts resolver fires once per parent."""
    authors = [{"id": i} for i in range(1, n + 1)]
    posts_by_author = {
        i: [{"id": i * 1000, "title": "p"}] for i in range(1, n + 1)
    }

    def resolve_posts(author, info):
        counter["calls"] += 1  # once PER parent — the N+1
        return posts_by_author[author["id"]]

    post_t = GraphQLObjectType(
        "Post",
        {
            "id": GraphQLField(GraphQLNonNull(GraphQLInt)),
            "title": GraphQLField(GraphQLNonNull(GraphQLString)),
        },
    )
    author_t = GraphQLObjectType(
        "Author",
        {
            "id": GraphQLField(GraphQLNonNull(GraphQLInt)),
            "posts": GraphQLField(
                GraphQLNonNull(GraphQLList(GraphQLNonNull(post_t))),
                resolve=resolve_posts,
            ),
        },
    )
    query_t = GraphQLObjectType(
        "Query",
        {
            "authors": GraphQLField(
                GraphQLNonNull(GraphQLList(GraphQLNonNull(author_t))),
                resolve=lambda root, info: authors,
            )
        },
    )
    return GraphQLSchema(query=query_t)


def bench_inmemory(n: int) -> dict:
    """Measure loadMany batch-call count (==1) + latency for the in-memory path."""
    query = "{ authors { id posts { id title } } }"

    counter = {"calls": 0, "max_keys": 0}
    schema = build_inmemory_schema(n, counter)
    parsed = parse(query)

    # SQL/fetch-count analogue: batch-callback invocations for ONE operation.
    from graphql import execute

    result = execute(schema, parsed, execution_context_class=GrafastExecutionContext)
    assert not result.errors, result.errors
    batch_calls = counter["calls"]
    max_keys = counter["max_keys"]
    assert len(result.data["authors"]) == n

    # naive control: per-parent resolver call count.
    naive_counter = {"calls": 0}
    naive_schema = build_naive_inmemory_schema(n, naive_counter)
    naive_result = graphql_sync(naive_schema, query)
    assert naive_result.errors is None
    naive_calls = naive_counter["calls"]

    # latency for the batched in-memory path (warmup + R repeats).
    bench_counter = {"calls": 0, "max_keys": 0}
    bench_schema = build_inmemory_schema(n, bench_counter)
    execute(bench_schema, parsed, execution_context_class=GrafastExecutionContext)
    latencies: List[float] = []
    for _ in range(LATENCY_REPEATS):
        t0 = time.perf_counter()
        execute(bench_schema, parsed, execution_context_class=GrafastExecutionContext)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return {
        "n": n,
        "batch_calls": batch_calls,
        "max_keys": max_keys,
        "naive_calls": naive_calls,
        "p50_ms": statistics.median(latencies),
        "p95_ms": p95(latencies),
    }


# ---------------------------------------------------------------- orchestration


def render_results_md(pg_rows: List[dict], mem_rows: List[dict]) -> str:
    """Render the two markdown results tables + the O(depth) verdict."""
    lines: List[str] = []
    lines.append("# N+1 benchmark results")
    lines.append("")
    lines.append(
        "Generated by `bench/bench_nplus1.py`. Fixed fan-out: "
        f"{POSTS_PER_AUTHOR} posts/author, {COMMENTS_PER_POST} comments/post "
        "(rows grow linearly in N). Latency = median/p95 over "
        f"{LATENCY_REPEATS} repeats (one warmup discarded), full plan+execute+"
        "serialize."
    )
    lines.append("")

    lines.append("## 1. Postgres data source")
    lines.append("")
    lines.append(
        "Nested query `{ authors { id name posts { id title comments { id body } } } }`"
        " (depth 3 in resource layers) over `grafast_demo`."
    )
    lines.append("")
    lines.append(
        "| N | pg sql_count (batched) | naive sql_count (O(N)) | batched p50 (ms) |"
        " batched p95 (ms) | rows touched |"
    )
    lines.append("|---|---|---|---|---|---|")
    for r in pg_rows:
        lines.append(
            f"| {r['n']} | {r['sql_count']} | {r['naive_sql_count']} |"
            f" {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | {r['rows']} |"
        )
    lines.append("")
    pg_counts = sorted({r["sql_count"] for r in pg_rows})
    lines.append(
        f"**O(depth) verdict:** batched sql_count = {pg_counts} for "
        f"N in {[r['n'] for r in pg_rows]} — constant in N (== 3, one statement per "
        "resource layer: root authors + posts `= ANY` + comments `= ANY`). The naive "
        "column grows as `1 + N + 5N`, i.e. O(N)."
    )
    lines.append("")
    lines.append("The 3 statements issued (from the N=10 run):")
    lines.append("")
    if pg_rows:
        for i, stmt in enumerate(pg_rows[0]["statements"], 1):
            lines.append(f"{i}. `{' '.join(stmt.split())[:110]}`")
    lines.append("")

    lines.append("## 2. Generic in-memory loadMany")
    lines.append("")
    lines.append(
        "Same shape `{ authors { id posts { id title } } }` where `Author.posts` is a"
        " `load_many`; the batch-call counter lives INSIDE the user callback."
    )
    lines.append("")
    lines.append(
        "| N | loadMany batch calls | naive resolver calls | p50 (ms) | p95 (ms) |"
    )
    lines.append("|---|---|---|---|---|")
    for r in mem_rows:
        lines.append(
            f"| {r['n']} | {r['batch_calls']} | {r['naive_calls']} |"
            f" {r['p50_ms']:.3f} | {r['p95_ms']:.3f} |"
        )
    lines.append("")
    mem_calls = sorted({r["batch_calls"] for r in mem_rows})
    lines.append(
        f"**Batching verdict:** loadMany batch calls = {mem_calls} for all N "
        "(exactly 1 — the loader receives every key in one call), vs the naive "
        "per-parent path firing N times. Constant in N."
    )
    lines.append("")
    return "\n".join(lines)


async def main() -> None:
    print("N+1 benchmark — scaling N over", N_VALUES)
    print("=" * 72)

    pg_rows: List[dict] = []
    for n in N_VALUES:
        row = await bench_pg(n)
        pg_rows.append(row)
        print(
            f"[pg]  N={n:<5} sql_count={row['sql_count']} "
            f"(naive {row['naive_sql_count']})  p50={row['p50_ms']:.2f}ms "
            f"p95={row['p95_ms']:.2f}ms  rows={row['rows']}"
        )

    await dispose_engine()

    print("-" * 72)
    mem_rows: List[dict] = []
    for n in N_VALUES:
        row = bench_inmemory(n)
        mem_rows.append(row)
        print(
            f"[mem] N={n:<5} batch_calls={row['batch_calls']} "
            f"(naive {row['naive_calls']})  keys_in_one_call={row['max_keys']}  "
            f"p50={row['p50_ms']:.3f}ms p95={row['p95_ms']:.3f}ms"
        )

    print("=" * 72)

    # CHECKPOINT assertions: SQL count O(depth), constant in N == 3; loadMany == 1.
    pg_counts = {r["sql_count"] for r in pg_rows}
    assert pg_counts == {3}, f"batched pg sql_count not constant ==3: {pg_counts}"
    mem_calls = {r["batch_calls"] for r in mem_rows}
    assert mem_calls == {1}, f"loadMany batch calls not constant ==1: {mem_calls}"
    print("CHECKPOINT OK: pg sql_count == 3 for all N (O(depth)); "
          "loadMany batch calls == 1 for all N.")

    md = render_results_md(pg_rows, mem_rows)
    RESULTS_MD.write_text(md)
    print(f"wrote {RESULTS_MD}")


if __name__ == "__main__":
    asyncio.run(main())
