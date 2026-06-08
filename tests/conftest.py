"""Local conftest for grafast-py's own test suite.

These tests drive `GrafastExecutionContext` explicitly via
`execution_context_class=...`, so they are independent of the `GRAFAST` env var
and of the conformance-oracle root conftest. We put the repo root on `sys.path`
here so the suite is self-sufficient: a few tests import the `examples` package
(the demo schema/seed fixtures live there), and that must resolve without relying
on the (separate, conformance-only) root conftest.

The `GRAFAST_INLINE_RELATIONS=1` switch flips LATERAL inlining ON for this whole suite
via :func:`inline_relations_suite_toggle`, so the existing pg tests — every one of which
already asserts EXACT result data — become the broadest possible byte-identical oracle
for inlining. It is OFF by default, so a plain `uv run pytest tests` (and the conformance
run, which has its own conftest and never sets this) is unaffected.

The plan-caching + runtime-placeholder switches do the same via the sibling
:func:`cache_plans_suite_toggle`:

- `GRAFAST_CACHE_PLANS=1` forces `cache_plans=True` (and, because only a value-agnostic
  placeholder-bearing plan is cacheable across values, `placeholders=True` with it), so a
  cache hit changes only WHETHER planning re-runs — never the SQL or the data, hence the
  whole result-asserting suite stays byte-identical.
- `GRAFAST_PLACEHOLDERS=1` forces ONLY `placeholders=True`, so the per-argument variable
  provenance surface (and the placeholder dedup path) is exercised independently of caching:
  a host that does not call `pg_placeholder` still inlines literals exactly, so the suite
  again stays byte-identical (provenance is computed but unused). This lets the two knobs
  be A/B'd separately — placeholders can be exercised suite-wide without caching.

Both are OFF by default; a plain `uv run pytest tests` and the conformance run are
unaffected (caching + placeholders are off by default).

The hoisting switch does the same for cross-parent hoisting, via :func:`hoist_suite_toggle`:

- Hoisting is ON by DEFAULT now (like upstream Grafast), so the plain `uv run pytest tests`,
  the conformance run, and every other oracle leg already exercise hoist-ON. The byte-identity
  oracle therefore runs the OFF baseline: `GRAFAST_HOIST=0` forces `hoist=False` across the
  suite, and it must produce the SAME result assertions as the default-on run — proving hoisting
  only LIFTS a step to a shallower layer (changing WHERE a step runs, never WHETHER), so the data
  is BYTE-IDENTICAL whether on or off. The existing corpus has no count-asserted hoistable shape,
  so it is byte-identical INCLUDING fetchCounts; the dedicated `tests/test_hoist.py` constructs a
  hoistable shape and proves the ON path (relocation + fire-once + guards). `GRAFAST_HOIST=1`
  forces it ON (redundant with the default, kept for explicitness).
"""

import os
import sys
from pathlib import Path

import pytest


# repo root (parent of tests/) on sys.path so `import examples.*` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# DB-backed tests target ONLY the local scratch database (never another DB on the
# server). setdefault so an explicit GRAFAST_PG_URL still wins; the pg engine itself
# bakes in no database (see grafast_py.pg.engine).
os.environ.setdefault("GRAFAST_PG_URL", "postgresql+asyncpg:///grafast_py_test")

# the suite-wide inlining switch: when set, the autouse fixture below flips the BASE
# GrafastExecutionContext's config to inline_relations=True for the whole suite.
INLINE_ENV_VAR = "GRAFAST_INLINE_RELATIONS"

# the suite-wide caching / placeholder switches: when set, the autouse fixture below flips
# the BASE GrafastExecutionContext's config for the whole suite, so the EXISTING
# result-asserting suite becomes the broadest byte-identical oracle.
#   GRAFAST_CACHE_PLANS    -> cache_plans=True (a hit changes only WHETHER planning re-runs);
#                            caching is only safe across values for a value-agnostic plan, so
#                            placeholders are forced on with it.
#   GRAFAST_PLACEHOLDERS   -> placeholders=True only (the variable-provenance + placeholder
#                            dedup path, A/B'd WITHOUT caching). A host that does not opt into
#                            pg_placeholder still inlines literals, so this too is byte-identical.
CACHE_ENV_VAR = "GRAFAST_CACHE_PLANS"
PLACEHOLDERS_ENV_VAR = "GRAFAST_PLACEHOLDERS"

# the cross-parent hoisting switch: when set (CI job `hoist-on`), the autouse fixture below flips the BASE
# GrafastExecutionContext's config to hoist=True for the whole suite, so the EXISTING
# result-asserting suite becomes the broadest byte-identical oracle for hoisting.
HOIST_ENV_VAR = "GRAFAST_HOIST"


def _env_flag(name: str) -> bool:
    """Whether env var ``name`` is set to a truthy value (unset / 0 / false => off)."""
    return os.environ.get(name, "") not in ("", "0", "false", "False")


def inline_relations_enabled() -> bool:
    """Whether the suite-wide switch asked for inlining ON across the whole suite."""
    return _env_flag(INLINE_ENV_VAR)


def cache_plans_enabled() -> bool:
    """Whether the suite-wide switch asked for plan caching ON across the whole suite."""
    return _env_flag(CACHE_ENV_VAR)


def placeholders_enabled() -> bool:
    """Whether the suite-wide switch asked for the placeholder provenance surface ON.

    Caching implies placeholders (only a value-agnostic placeholder-bearing plan is cacheable
    across values), so ``GRAFAST_CACHE_PLANS`` turns this on too; ``GRAFAST_PLACEHOLDERS`` turns
    it on WITHOUT caching, so the two can be A/B'd independently.
    """
    return _env_flag(PLACEHOLDERS_ENV_VAR) or cache_plans_enabled()


def hoist_override():
    """Explicit hoist override from ``GRAFAST_HOIST``: True (force on), False (force off), or None.

    Hoisting now defaults ON, so the byte-identity oracle needs the OFF baseline too: ``=0`` /
    ``false`` forces it OFF across the suite, ``=1`` forces it ON (redundant with the default but
    explicit), and unset (None) leaves the default. Three-state, unlike the other on/off toggles.
    """
    val = os.environ.get(HOIST_ENV_VAR)
    if val is None or val == "":
        return None
    return val not in ("0", "false", "False")


def pytest_configure(config):
    # register the asyncio marker mode for this directory's async step tests
    config.option.asyncio_mode = "strict"
    # Pin the default async-fixture loop scope to "function" explicitly. pytest-asyncio
    # otherwise leaves it UNSET (warning, and version-dependent default), so an async fixture
    # and its test could run on different event loops — a hazard for the process-global pg
    # engine (created lazily on the running loop). Pinning it makes every async fixture share
    # its test's loop deterministically.
    if getattr(config.option, "asyncio_default_fixture_loop_scope", None) is None:
        config.option.asyncio_default_fixture_loop_scope = "function"
    # the `pg` marker tags DB-backed tests (they touch only the grafast_demo schema
    # of grafast_py_test) so a no-DB run can deselect them with `-m 'not pg'`.
    config.addinivalue_line("markers", "pg: database-backed test (grafast_demo schema)")
    # the `inline_off` marker pins a test to the batched baseline even under the
    # suite-wide inline-on switch (it asserts the EXACT batched statement count, which
    # inlining legitimately reduces — the data oracle still holds, but its count would not).
    config.addinivalue_line(
        "markers",
        "inline_off: keep inlining OFF for this test even under GRAFAST_INLINE_RELATIONS "
        "(it asserts the exact batched statement count, which a fold reduces)",
    )
    # the `cache_off` marker pins a test to per-request planning even under the suite-wide
    # cache-on switch: a test that asserts the EXACT number of plan BUILDS (which caching
    # legitimately reduces on a hit), or that mutates a context's grafast_config mid-test in a
    # way the shared base override would fight, opts out. The result/statement-count oracle
    # still holds for it; only the plan-build count would differ.
    config.addinivalue_line(
        "markers",
        "cache_off: keep plan caching OFF for this test even under GRAFAST_CACHE_PLANS "
        "(it asserts the exact plan-build count, which a cache hit reduces)",
    )
    # the `hoist_off` marker pins a test to the naive (per-child-bucket) layout even under the
    # CI hoist-on switch: a test that asserts the EXACT per-bucket fetchCount for a shape a
    # legal hoist would lift (firing FEWER times by design) opts out. The data oracle still
    # holds for it; only the count would differ. No corpus test needs it today (the corpus has
    # no hoistable shape), but it keeps the global byte-identical-fetchCounts invariant intact
    # if a future corpus addition introduces one.
    config.addinivalue_line(
        "markers",
        "hoist_off: keep hoisting OFF for this test even under GRAFAST_HOIST "
        "(it asserts the exact per-child-bucket fetchCount, which a hoist reduces)",
    )


@pytest.fixture(autouse=True)
def inline_relations_suite_toggle(request):
    """Flip LATERAL inlining ON for the whole suite under `GRAFAST_INLINE_RELATIONS=1`.

    The suite-wide "broadest oracle": run the EXISTING pg suite — which already asserts
    exact result data everywhere — with inlining forced on, proving the data is
    BYTE-IDENTICAL to the batched baseline across the board. We monkeypatch the BASE
    :class:`GrafastExecutionContext`'s class-level ``grafast_config`` (the one every pg
    test's ``execution_context_class=GrafastExecutionContext`` reads) to
    ``inline_relations=True`` and restore it after each test.

    Three things keep this surgical:

    - It is a NO-OP unless ``GRAFAST_INLINE_RELATIONS`` is set, so the default run (and
      the separate conformance run) is untouched.
    - A test that defines its OWN ``grafast_config`` on a context subclass (the inlining
      equivalence battery, the hardening tests) shadows this base attribute, so its
      explicit config wins — we never override an intentional config.
    - A test marked ``inline_off`` (it asserts the EXACT batched statement count, which a
      fold reduces) is left on the batched baseline; its data oracle still holds, only its
      count would differ, so pinning it OFF keeps the count assertion meaningful while the
      data-equivalence is covered by the dedicated equivalence module.
    """
    if not inline_relations_enabled() or request.node.get_closest_marker("inline_off"):
        yield
        return

    from grafast_py.config import GrafastConfig
    from grafast_py.context import GrafastExecutionContext

    previous = GrafastExecutionContext.__dict__.get("grafast_config")
    GrafastExecutionContext.grafast_config = GrafastConfig(inline_relations=True)
    try:
        yield
    finally:
        if previous is None:
            del GrafastExecutionContext.grafast_config
        else:
            GrafastExecutionContext.grafast_config = previous


@pytest.fixture(autouse=True)
def cache_plans_suite_toggle(request):
    """Flip plan caching / placeholders ON for the whole suite under the suite-wide switches.

    The "broadest oracle" — the sibling of :func:`inline_relations_suite_toggle`: run the
    EXISTING result-asserting suite with caching and/or the placeholder provenance surface
    forced on, proving they change only WHETHER planning re-runs (a cache hit) and WHICH
    provenance is computed (placeholders) — never the SQL or the data, so results AND
    statement counts stay BYTE-IDENTICAL to the per-request baseline. We monkeypatch the BASE
    :class:`GrafastExecutionContext`'s class-level ``grafast_config`` and restore it after each
    test; the process-global plan cache is CLEARED around each test so no entry leaks across
    tests (each test builds its own schema, but a fresh cache keeps the oracle hermetic).

    Two independent env switches drive ONE merged config so they cannot fight over the shared
    ``grafast_config`` attribute (a single fixture sets the union):

    - ``GRAFAST_CACHE_PLANS=1`` => ``cache_plans=True`` (and ``placeholders=True`` with it,
      since only a value-agnostic placeholder-bearing plan is cacheable across values).
    - ``GRAFAST_PLACEHOLDERS=1`` => ``placeholders=True`` only (the provenance + placeholder
      dedup path, A/B'd WITHOUT caching).

    Surgical, exactly like the inlining toggle:

    - A NO-OP unless at least one switch is set, so the default run and the conformance run are
      untouched.
    - A test with its OWN ``grafast_config`` on a context subclass shadows this base attribute,
      so its explicit config wins.
    - A test marked ``cache_off`` (it asserts the exact plan-BUILD count, which a hit reduces)
      is kept off the CACHE forcing — but placeholders, which never change the build count or
      the data, are still forced if their switch is set, so the placeholder oracle keeps its
      reach even over the cache-build-count tests.
    """
    force_cache = cache_plans_enabled() and not request.node.get_closest_marker("cache_off")
    force_placeholders = placeholders_enabled()
    if not force_cache and not force_placeholders:
        yield
        return

    from grafast_py.cache import default_cache
    from grafast_py.config import GrafastConfig
    from grafast_py.context import GrafastExecutionContext

    previous = GrafastExecutionContext.__dict__.get("grafast_config")
    GrafastExecutionContext.grafast_config = GrafastConfig(
        cache_plans=force_cache, placeholders=force_placeholders
    )
    default_cache().clear()
    try:
        yield
    finally:
        default_cache().clear()
        if previous is None:
            del GrafastExecutionContext.grafast_config
        else:
            GrafastExecutionContext.grafast_config = previous


@pytest.fixture(autouse=True)
def hoist_suite_toggle(request):
    """Force the suite's cross-parent hoisting setting for the byte-identity oracle — BOTH ways.

    Hoisting now defaults ON (config.py / upstream), so the default run, the conformance run, and
    every other oracle leg already exercise hoist-ON. The byte-identity oracle therefore needs the
    OFF baseline: ``GRAFAST_HOIST=0`` forces hoisting OFF across the whole result-asserting suite,
    and the default (no env) leaves it ON — both must produce the same result assertions, proving
    hoisting changes only WHERE a step runs (lifting a request-/parent-constant step to a shallower
    layer), never the data. ``GRAFAST_HOIST=1`` forces it ON (redundant with the default, kept for
    explicitness). We monkeypatch the BASE :class:`GrafastExecutionContext`'s class-level
    ``grafast_config`` and restore it after each test.

    Surgical, like the inlining/caching toggles:

    - A NO-OP unless ``GRAFAST_HOIST`` is set OR the test is ``hoist_off``-marked, so the default
      run is untouched (it uses the on-by-default config).
    - A test that defines its OWN ``grafast_config`` on a context subclass (e.g.
      ``tests/test_hoist.py``) shadows this base attribute, so its explicit config wins.
    - A test marked ``hoist_off`` (it asserts the EXACT per-child-bucket fetchCount, which a legal
      hoist reduces) is pinned OFF regardless of the default. No corpus test needs this today (the
      corpus has no count-asserted hoistable shape), so the whole suite is byte-identical INCLUDING
      counts whether hoist is on or off.
    """
    override = hoist_override()
    if request.node.get_closest_marker("hoist_off"):
        override = False
    if override is None:
        yield
        return

    from grafast_py.config import GrafastConfig
    from grafast_py.context import GrafastExecutionContext

    previous = GrafastExecutionContext.__dict__.get("grafast_config")
    GrafastExecutionContext.grafast_config = GrafastConfig(hoist=override)
    try:
        yield
    finally:
        if previous is None:
            del GrafastExecutionContext.grafast_config
        else:
            GrafastExecutionContext.grafast_config = previous
