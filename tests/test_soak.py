"""CI guard for the concurrent-soak PASS gates (build:soak checkpoint).

A scaled-down run of ``bench/soak.py`` (fewer ops, modest concurrency, smaller N) so
CI guards the four PASS gates without a multi-minute full run:

(a) ZERO errors / 100% correct results under concurrency;
(b) NO connection-pool leak — checkouts return to baseline, never exceed pool max;
(c) NO memory leak — RSS drift after warmup stays under threshold;
(d) bounded latency — p50/p95/p99 recorded (asserted soft-bounded here).

The full 5000-op @ concurrency 32 run is the bench script. Marked ``pg`` — touches
only ``grafast_demo`` in ``grafast_py_test`` (the engine hard-codes the URL).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

from soak_core import run_soak, scale_seed, SoakResult  # noqa: E402
from grafast_py.pg.engine import dispose_engine, get_engine  # noqa: E402

pytestmark = pytest.mark.pg

# scaled-down: still > pool-max concurrency (16 > 15) so checkouts queue and the leak
# path is exercised, still a few thousand ops so a per-op leak would compound visibly.
CI_TOTAL_OPS = 1000
CI_CONCURRENCY = 16
CI_WARMUP_OPS = 100
CI_N_AUTHORS = 50
CI_RSS_THRESHOLD_MB = 50.0


@pytest.mark.asyncio
async def test_concurrent_soak_no_errors_no_leak_bounded_rss():
    """1000 mixed ops at concurrency 16: zero errors, no pool leak, bounded RSS."""
    await dispose_engine()  # fresh engine bound to THIS test's loop
    await scale_seed(CI_N_AUTHORS)
    try:
        res: SoakResult = await run_soak(
            total_ops=CI_TOTAL_OPS,
            concurrency=CI_CONCURRENCY,
            n_authors=CI_N_AUTHORS,
            warmup_ops=CI_WARMUP_OPS,
            rss_threshold_mb=CI_RSS_THRESHOLD_MB,
        )
    finally:
        await dispose_engine()

    # (a) zero errors / 100% correct
    assert res.errors == 0, res.error_messages[:5]
    assert res.correct == res.total_ops, (res.correct, res.total_ops)

    # (b) no pool leak — concurrency exceeded the pool, so checkouts spiked under
    # contention (a background monitor observed the peak), stayed within the bound,
    # and all returned at quiescence.
    assert res.max_checkedout > 0, "contention never observed — monitor not sampling"
    assert res.max_checkedout <= res.pool_max, (res.max_checkedout, res.pool_max)
    assert res.final_checkedout == 0, res.final_checkedout
    assert res.final_size <= res.baseline_size, (res.final_size, res.baseline_size)
    assert not res.pool_leak

    # (c) bounded RSS drift after warmup.
    assert res.rss_drift < CI_RSS_THRESHOLD_MB, res.rss_drift

    # (d) latency percentiles recorded and sane (non-zero, ordered).
    assert res.p50_ms > 0.0
    assert res.p50_ms <= res.p95_ms <= res.p99_ms <= res.max_ms

    assert res.passed(CI_RSS_THRESHOLD_MB)
