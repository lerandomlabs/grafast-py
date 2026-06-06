"""Local conftest for grafast-py's own test suite.

These tests drive `GrafastExecutionContext` explicitly via
`execution_context_class=...`, so they are independent of the `GRAFAST` env var
and of the conformance-oracle root conftest. We put the repo root on `sys.path`
here so the suite is self-sufficient: a few tests import the `examples` package
(the demo schema/seed fixtures live there), and that must resolve without relying
on the (separate, conformance-only) root conftest.
"""

import os
import sys
from pathlib import Path


# repo root (parent of tests/) on sys.path so `import examples.*` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# DB-backed tests target ONLY the local scratch database (never another DB on the
# server). setdefault so an explicit GRAFAST_PG_URL still wins; the pg engine itself
# bakes in no database (see grafast_py.pg.engine).
os.environ.setdefault("GRAFAST_PG_URL", "postgresql+asyncpg:///grafast_py_test")


def pytest_configure(config):
    # register the asyncio marker mode for this directory's async step tests
    config.option.asyncio_mode = "strict"
    # the `pg` marker tags DB-backed tests (they touch only the grafast_demo schema
    # of grafast_py_test) so a no-DB run can deselect them with `-m 'not pg'`.
    config.addinivalue_line("markers", "pg: database-backed test (grafast_demo schema)")
