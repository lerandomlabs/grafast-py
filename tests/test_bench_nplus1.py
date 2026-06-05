"""CI guard for the N+1 benchmark invariants (build:benchmark checkpoint).

A scaled-down run of ``bench/bench_nplus1.py`` that asserts the two PASS invariants
so CI guards them without the multi-minute full sweep:

- the batched Postgres nested query issues a CONSTANT number of SQL statements (== 3,
  one per resource layer) for every N — O(depth), NOT O(rows);
- the generic in-memory ``load_many`` path fires its batch callback EXACTLY ONCE per
  operation for every N (vs N for a naive per-parent resolver).

Marked ``pg`` (touches only ``grafast_demo`` in ``grafast_py_test``). The full sweep
(N up to 1000, latency percentiles, results.md) lives in the bench script.
"""

import sys
from pathlib import Path

import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

import bench_nplus1 as bench  # noqa: E402
from grafast_py.pg.engine import dispose_engine  # noqa: E402

pytestmark = pytest.mark.pg

# scaled-down N values keep the CI guard fast while still proving constancy across
# an order-of-magnitude change in N (the full {10,50,200,1000} sweep is the script).
GUARD_N = [10, 50, 200]


@pytest_asyncio.fixture
async def fresh_engine():
    """Each test on its own loop with a fresh engine (pool is loop-bound)."""
    await dispose_engine()
    yield
    await dispose_engine()


@pytest.mark.inline_off
@pytest.mark.asyncio
async def test_pg_sql_count_is_o_depth_constant_in_n(fresh_engine):
    """Batched pg nested query == 3 SQL statements for every N (O(depth))."""
    counts = []
    for n in GUARD_N:
        row = await bench.bench_pg(n)
        counts.append(row["sql_count"])
        # the naive control is genuinely O(N), confirming the gap is real.
        assert row["naive_sql_count"] == 1 + n + bench.POSTS_PER_AUTHOR * n
    assert set(counts) == {3}, f"sql_count must be constant ==3, got {counts}"


def test_inmemory_loadmany_is_one_call_constant_in_n():
    """Generic load_many fires its batch callback exactly once for every N."""
    calls = []
    for n in GUARD_N:
        row = bench.bench_inmemory(n)
        calls.append(row["batch_calls"])
        # the loader saw the whole bucket in that one call.
        assert row["max_keys"] == n
        # naive per-parent control fires N times.
        assert row["naive_calls"] == n
    assert set(calls) == {1}, f"loadMany batch calls must be ==1, got {calls}"
