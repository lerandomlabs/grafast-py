"""Local conftest for grafast-py's own test suite.

These tests drive `GrafastExecutionContext` explicitly via
`execution_context_class=...`, so they are independent of the `GRAFAST` env var
and of the conformance-oracle root conftest. We put the repo root on `sys.path`
here so the suite is self-sufficient: a few tests import the `examples` package
(the demo schema/seed fixtures live there), and that must resolve without relying
on the (separate, conformance-only) root conftest.

The `GRAFAST_INLINE_RELATIONS=1` switch (the Wave 3b step-9 CI job) flips LATERAL
inlining ON for this whole suite via :func:`inline_relations_suite_toggle`, so the
existing pg tests — every one of which already asserts EXACT result data — become the
broadest possible byte-identical oracle for inlining. It is OFF by default, so a plain
`uv run pytest tests` (and the conformance run, which has its own conftest and never
sets this) is unaffected.
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

# the step-9 CI switch: when set (CI job `inline-on`), the autouse fixture below flips
# the BASE GrafastExecutionContext's config to inline_relations=True for the whole suite.
INLINE_ENV_VAR = "GRAFAST_INLINE_RELATIONS"


def inline_relations_enabled() -> bool:
    """Whether the step-9 CI switch asked for inlining ON across the whole suite."""
    return os.environ.get(INLINE_ENV_VAR, "") not in ("", "0", "false", "False")


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
    # the `inline_off` marker pins a test to the batched baseline even under the CI
    # inline-on switch (it asserts the EXACT batched statement count, which inlining
    # legitimately reduces — the data oracle still holds, but its count would not).
    config.addinivalue_line(
        "markers",
        "inline_off: keep inlining OFF for this test even under GRAFAST_INLINE_RELATIONS "
        "(it asserts the exact batched statement count, which a fold reduces)",
    )


@pytest.fixture(autouse=True)
def inline_relations_suite_toggle(request):
    """Flip LATERAL inlining ON for the whole suite under `GRAFAST_INLINE_RELATIONS=1`.

    The step-9 CI job's "broadest oracle": run the EXISTING pg suite — which already
    asserts exact result data everywhere — with inlining forced on, proving the data is
    BYTE-IDENTICAL to the batched baseline across the board. We monkeypatch the BASE
    :class:`GrafastExecutionContext`'s class-level ``grafast_config`` (the one every pg
    test's ``execution_context_class=GrafastExecutionContext`` reads) to
    ``inline_relations=True`` and restore it after each test.

    Three things keep this surgical:

    - It is a NO-OP unless ``GRAFAST_INLINE_RELATIONS`` is set, so the default run (and
      the separate conformance run) is untouched — inlining ships dark.
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
