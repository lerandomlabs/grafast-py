"""Conformance-suite conftest (committed; the fetched suite lives in ``_suite/``).

Scoped to ``conformance/`` only — pytest does not load this for ``pytest tests``,
so grafast-py's own suite is provably independent of the global executor patch.

Run modes (after ``uv run python scripts/fetch_conformance.py``):
    uv run pytest conformance              # baseline: stock graphql-core executor
    GRAFAST=1 uv run pytest conformance    # routed through grafast-py

``conformance/_suite/execution/test_customize.py`` is skipped under GRAFAST because
it tests graphql-core's *own* execution-context customization hook — graphql-core
internals, not GraphQL semantics, so an alternative engine cannot satisfy it.
"""

import os
import sys
from pathlib import Path

import pytest

# Repo root on sys.path so the fetched suite imports as ``conformance._suite.*``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

GRAFAST = bool(os.environ.get("GRAFAST"))


def pytest_configure(config):
    """Under GRAFAST=1, install grafast-py as graphql-core's executor.

    Importing grafast_py runs its ``install()`` (patches the ``ExecutionContext``
    that ``execute()`` / ``subscribe()`` fall back to). We import it ONLY when
    GRAFAST is set, so the baseline run never loads grafast-py and stays on the
    stock executor.
    """
    if not GRAFAST:
        return
    from grafast_py import install

    install()


def pytest_collection_modifyitems(config, items):
    if not GRAFAST:
        return
    skip_internal = pytest.mark.skip(
        reason="tests graphql-core's own execution-context customization hook; "
        "out of scope for an alternative engine"
    )
    for item in items:
        if "test_customize.py" in str(getattr(item, "fspath", "")):
            item.add_marker(skip_internal)
