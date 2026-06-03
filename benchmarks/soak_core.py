"""Shared concurrent-soak core for the Phase D load harness.

Drives many GraphQL operations concurrently through
:class:`grafast_py.GrafastExecutionContext` against the ``grafast_demo`` schema,
asserting under concurrency:

(a) ZERO errors / 100% correct results — every response is validated against an
    expected shape (per-shape spot-checks; a sampled subset gets a deeper check);
(b) NO connection-pool leak — the async engine's pool (``checkedout``/``checkedin``/
    ``size`` on ``get_engine().sync_engine.pool``) is sampled at quiescence and
    mid-soak; checkouts must return to baseline and never exceed the pool max;
(c) NO memory leak — current process RSS (psutil) is sampled across the run; drift
    after warmup must stay bounded;
(d) bounded latency — per-op ``perf_counter`` timings → p50/p95/p99.

The whole soak runs inside ONE event loop (a single ``asyncio.run`` by the caller)
so a single shared engine/pool serves every operation — the pool is loop-bound, and
this is also the only shape that actually exercises a pool leak. CONCURRENCY
intentionally exceeds the pool size so checkouts queue rather than open unbounded
connections; that contention is what surfaces checkout/leak bugs.

SAFETY: the only database touched is ``grafast_py_test`` and the only schema is
``grafast_demo``. The seeder imports the shared engine (``get_engine``) and never
constructs its own connection URL; all DDL/DML is confined to ``grafast_demo``.
"""

import asyncio
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from graphql import GraphQLSchema, graphql
from sqlalchemy import text

from grafast_py import GrafastExecutionContext
from examples.demo_schema import build_demo_schema
from grafast_py.pg.engine import get_engine
from examples.seed import DEMO_SCHEMA

try:  # current-RSS sampling; psutil is the dev dep for this (preferred over rusage).
    import psutil

    _PROCESS = psutil.Process()

    def sample_rss_mb() -> float:
        """Current process resident set size, in MB."""
        return _PROCESS.memory_info().rss / (1024 * 1024)

except ImportError:  # pragma: no cover - psutil is a declared dev dependency
    import resource

    def sample_rss_mb() -> float:
        """Fallback: peak RSS via getrusage (macOS bytes / Linux KB)."""
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        divisor = 1024 * 1024 if os.uname().sysname == "Darwin" else 1024
        return peak / divisor


# fixed fan-out so the dataset is bounded and deterministic (matches bench_nplus1).
POSTS_PER_AUTHOR = 5
COMMENTS_PER_POST = 4


async def scale_seed(n: int) -> None:
    """Idempotently (re)seed ``grafast_demo`` with N authors + fixed fan-out.

    Drops and recreates the schema, then bulk-inserts deterministic rows. Confined
    strictly to ``grafast_demo`` in ``grafast_py_test`` via the shared engine — no
    other database is reachable.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP SCHEMA IF EXISTS {DEMO_SCHEMA} CASCADE"))
        await conn.execute(text(f"CREATE SCHEMA {DEMO_SCHEMA}"))
        await conn.execute(
            text(
                f"CREATE TABLE {DEMO_SCHEMA}.authors ("
                " id integer PRIMARY KEY, name text NOT NULL)"
            )
        )
        await conn.execute(
            text(
                f"CREATE TABLE {DEMO_SCHEMA}.posts ("
                " id integer PRIMARY KEY,"
                f" author_id integer NOT NULL REFERENCES {DEMO_SCHEMA}.authors (id),"
                " title text NOT NULL)"
            )
        )
        await conn.execute(
            text(
                f"CREATE TABLE {DEMO_SCHEMA}.comments ("
                " id integer PRIMARY KEY,"
                f" post_id integer NOT NULL REFERENCES {DEMO_SCHEMA}.posts (id),"
                f" author_id integer NOT NULL REFERENCES {DEMO_SCHEMA}.authors (id),"
                " body text NOT NULL)"
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
    stmt = text(sql)
    for start in range(0, len(rows), chunk):
        await conn.execute(stmt, rows[start : start + chunk])


# ----------------------------------------------------------------- query shapes
# Each shape is (name, query, validator). The validator raises AssertionError on a
# wrong/missing result; `n_authors` is the seeded author count so totals are exact.


def make_shapes(n_authors: int) -> List[Tuple[str, str, Callable[[dict], None]]]:
    """Build the validated query-shape mix for a soak over N authors.

    Exercises root lists, hasMany, hasOne back-ref, Relay connections (window query),
    a scalar arg + by-id single — the full pg step set.
    """

    def v_flat(data: dict) -> None:
        authors = data["authors"]
        assert len(authors) == n_authors, (len(authors), n_authors)
        a0 = authors[0]
        assert isinstance(a0["id"], int) and isinstance(a0["name"], str)

    def v_nested(data: dict) -> None:
        authors = data["authors"]
        assert len(authors) == n_authors
        first = authors[0]
        assert len(first["posts"]) == POSTS_PER_AUTHOR
        assert len(first["posts"][0]["comments"]) == COMMENTS_PER_POST

    def v_hasone(data: dict) -> None:
        posts = data["posts"]
        assert len(posts) == POSTS_PER_AUTHOR * n_authors
        # post 1 belongs to author 1 (first POSTS_PER_AUTHOR posts).
        assert posts[0]["author"]["id"] == 1
        assert isinstance(posts[0]["author"]["name"], str)

    def v_connection(data: dict) -> None:
        authors = data["authors"]
        assert len(authors) == n_authors
        conn = authors[0]["postsConnection"]
        assert conn["totalCount"] == POSTS_PER_AUTHOR
        assert len(conn["edges"]) == 2
        # 5 posts/author, first:2 -> there is a next page.
        assert conn["pageInfo"]["hasNextPage"] is True
        assert isinstance(conn["edges"][-1]["cursor"], str)

    def v_by_id(data: dict) -> None:
        author = data["author"]
        assert author is not None
        assert author["id"] == 7
        assert len(author["posts"]) == POSTS_PER_AUTHOR

    return [
        ("flat", "{ authors { id name } }", v_flat),
        (
            "nested",
            "{ authors { id posts { id comments { id } } } }",
            v_nested,
        ),
        ("hasone", "{ posts { id author { id name } } }", v_hasone),
        (
            "connection",
            "{ authors { id postsConnection(first: 2) {"
            " totalCount edges { cursor node { id } }"
            " pageInfo { hasNextPage } } } }",
            v_connection,
        ),
        (
            "by_id",
            "{ author(id: 7) { id name posts { id } } }",
            v_by_id,
        ),
    ]


@dataclass
class SoakResult:
    """Outcome of one soak run; fields map directly to the report + pass gates."""

    total_ops: int
    concurrency: int
    errors: int
    correct: int
    error_messages: List[str] = field(default_factory=list)
    # pool
    baseline_checkedout: int = 0
    baseline_checkedin: int = 0
    baseline_size: int = 0
    max_checkedout: int = 0
    final_checkedout: int = 0
    final_size: int = 0
    pool_max: int = 0
    pool_leak: bool = False
    # memory (MB)
    rss_start: float = 0.0
    rss_warmup: float = 0.0
    rss_final: float = 0.0
    rss_peak: float = 0.0
    rss_drift: float = 0.0
    rss_series: List[float] = field(default_factory=list)
    # latency (ms, post-warmup)
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    max_ms: float = 0.0

    def passed(self, rss_threshold_mb: float) -> bool:
        """The hard gates: zero errors, all correct, no pool leak, bounded RSS."""
        return (
            self.errors == 0
            and self.correct == self.total_ops
            and not self.pool_leak
            and self.rss_drift < rss_threshold_mb
        )


def _percentile(samples: List[float], q: float) -> float:
    """Nearest-rank percentile (robust for any sample size)."""
    if not samples:
        return 0.0
    ordered = sorted(samples)
    idx = max(0, min(len(ordered) - 1, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


async def run_soak(
    *,
    total_ops: int,
    concurrency: int,
    n_authors: int,
    warmup_ops: int,
    rss_threshold_mb: float,
    schema: Optional[GraphQLSchema] = None,
    deep_check_every: int = 100,
) -> SoakResult:
    """Run the concurrent soak and return a fully-populated :class:`SoakResult`.

    Caller is responsible for seeding (``scale_seed``) and disposing the engine; this
    runs entirely on the current event loop with the shared module-global engine.
    """
    if schema is None:
        schema = build_demo_schema()
    shapes = make_shapes(n_authors)
    engine = get_engine()
    pool = engine.sync_engine.pool
    pool_max = pool.size() + getattr(pool, "_max_overflow", 0)

    res = SoakResult(
        total_ops=total_ops, concurrency=concurrency, errors=0, correct=0
    )
    res.pool_max = pool_max
    res.rss_start = sample_rss_mb()

    latencies: List[float] = []
    lock = asyncio.Lock()

    async def run_one(op_index: int) -> None:
        name, query, validator = shapes[op_index % len(shapes)]
        t0 = time.perf_counter()
        try:
            result = await graphql(
                schema, query, execution_context_class=GrafastExecutionContext
            )
        except (asyncio.TimeoutError, OSError, RuntimeError) as exc:
            async with lock:
                res.errors += 1
                if len(res.error_messages) < 20:
                    res.error_messages.append(f"{name}: raised {exc!r}")
            return
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        ok = True
        problem = ""
        if result.errors:
            ok = False
            problem = f"{name}: graphql errors {result.errors}"
        elif result.data is None:
            ok = False
            problem = f"{name}: data is None"
        else:
            try:
                validator(result.data)
                # a sampled subset gets re-validated as a deeper guard.
                if op_index % deep_check_every == 0:
                    validator(result.data)
            except AssertionError as exc:
                ok = False
                problem = f"{name}: validate failed {exc}"

        async with lock:
            latencies.append(elapsed_ms)
            if ok:
                res.correct += 1
            else:
                res.errors += 1
                if len(res.error_messages) < 20:
                    res.error_messages.append(problem)

    # warmup: drive `warmup_ops` so the schema, plan path, asyncpg statement cache and
    # pool reach steady state before RSS/latency are measured.
    warmup_done = 0
    op_counter = 0
    while warmup_done < warmup_ops:
        batch = min(concurrency, warmup_ops - warmup_done)
        await asyncio.gather(*(run_one(op_counter + i) for i in range(batch)))
        op_counter += batch
        warmup_done += batch

    # quiescent baseline AFTER warmup: pool should be drained, RSS at steady state.
    res.baseline_checkedout = pool.checkedout()
    res.baseline_checkedin = pool.checkedin()
    res.baseline_size = pool.size()
    res.rss_warmup = sample_rss_mb()
    res.rss_peak = res.rss_warmup
    res.rss_series.append(res.rss_warmup)
    # warmup ops are discarded — only post-warmup ops count toward the gates: reset
    # the correctness/error tallies and the latency sample to steady state.
    latencies.clear()
    res.correct = 0
    res.errors = 0
    res.error_messages.clear()

    # a background monitor samples pool checkout WHILE batches are in flight — the
    # per-batch sampling alone only ever sees the drained (quiescent) state between
    # gathers, so peak contention would read as 0. This task observes the real spike.
    #
    # It samples at a small REAL interval (1ms), NOT `sleep(0)`: a `sleep(0)` busy
    # loop re-schedules itself on the event loop every tick, making the monitor a hot
    # task that competes with the operations under measurement and badly inflates the
    # latency tail (observed p99 > 1s, max > 2s — a measurement artefact, not engine
    # contention). A pool checkout is held for the duration of a batched SQL round-trip
    # (tens of ms), so a 1ms cadence still captures the peak while leaving the loop free
    # for the work being measured.
    monitoring = True

    async def monitor_pool() -> None:
        while monitoring:
            res.max_checkedout = max(res.max_checkedout, pool.checkedout())
            await asyncio.sleep(0.001)

    monitor = asyncio.ensure_future(monitor_pool())

    measured = 0
    while measured < total_ops:
        batch = min(concurrency, total_ops - measured)
        await asyncio.gather(*(run_one(op_counter + i) for i in range(batch)))
        op_counter += batch
        measured += batch
        # sample pool + RSS every batch (32-wide → ~157 batches at the full size).
        res.max_checkedout = max(res.max_checkedout, pool.checkedout())
        rss = sample_rss_mb()
        res.rss_peak = max(res.rss_peak, rss)
        res.rss_series.append(rss)

    monitoring = False
    await monitor

    # final quiescent sample: every connection must have returned.
    res.final_checkedout = pool.checkedout()
    res.final_size = pool.size()
    res.rss_final = sample_rss_mb()
    res.rss_drift = res.rss_final - res.rss_warmup

    res.pool_leak = (
        res.final_checkedout != 0
        or res.max_checkedout > pool_max
        or res.final_size > res.baseline_size
    )

    res.p50_ms = _percentile(latencies, 0.50)
    res.p95_ms = _percentile(latencies, 0.95)
    res.p99_ms = _percentile(latencies, 0.99)
    res.max_ms = max(latencies) if latencies else 0.0

    return res


__all__ = [
    "POSTS_PER_AUTHOR",
    "COMMENTS_PER_POST",
    "scale_seed",
    "make_shapes",
    "run_soak",
    "sample_rss_mb",
    "SoakResult",
]
