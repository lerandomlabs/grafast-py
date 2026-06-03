"""Async SQLAlchemy engine plumbing for the Postgres data source.

A single module-global :class:`AsyncEngine` over the ONLY database this project may
touch — ``postgresql+asyncpg:///grafast_py_test`` — is created lazily and reused
across requests (one connection pool). All DDL/data the data source creates lives in
the ``grafast_demo`` schema inside that database.

``count_sql`` is the proof instrument for the O(depth) batching gate: it attaches a
``before_cursor_execute`` listener to the engine's underlying sync engine and counts
every statement executed while it is open, so a test can assert that a depth-D nested
GraphQL query issues ~D SQL statements (one per resource-layer) rather than O(rows).
"""

from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

# the ONLY database url the data source is permitted to open. Hard-coded here so a
# stray connection string can never point anywhere else (see HARD SAFETY RULES).
# The pool/connection KNOBS below are configurable; the URL (i.e. the database) is
# NOT — tuning never repoints the data source at another database.
DATABASE_URL = "postgresql+asyncpg:///grafast_py_test"

# the schema every demo object lives in; create/drop is confined to it.
DEMO_SCHEMA = "grafast_demo"

_engine: Optional[AsyncEngine] = None


def get_engine(
    *,
    pool_size: int = 5,
    max_overflow: int = 10,
    pool_timeout: float = 30.0,
    pool_pre_ping: bool = False,
    pool_recycle: int = -1,
    connect_args: Optional[Dict[str, Any]] = None,
    echo: bool = False,
) -> AsyncEngine:
    """Return the shared async engine, creating it on first use with the given pool.

    Only the FIRST call (or an explicit :func:`configure_engine`) decides the pool /
    connection settings — once the singleton exists, later calls reuse it and ignore
    the kwargs (call :func:`dispose_engine` first to rebuild). The keyword arguments
    are passed through to :func:`sqlalchemy.ext.asyncio.create_async_engine`.

    Concurrency / pool relationship: the effective ceiling on concurrent in-flight
    SQL statements is ``pool_size + max_overflow`` (SQLAlchemy default 5 + 10 = 15).
    If application-level concurrency exceeds that, the excess operations queue on
    checkout, which raises the latency tail (the Phase D soak saw 32 concurrent ops
    over the 15-connection default queue to a ~0.5s p99). Size the pool to your
    target concurrency, cap concurrency with ``GrafastConfig.max_step_concurrency``,
    or both. The database URL is fixed to ``grafast_py_test`` — these knobs tune the
    pool/connection only, never the target database (HARD SAFETY RULE).
    """
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            DATABASE_URL,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_pre_ping=pool_pre_ping,
            pool_recycle=pool_recycle,
            connect_args=connect_args or {},
            echo=echo,
        )
    return _engine


def configure_engine(**kwargs: Any) -> AsyncEngine:
    """Dispose any existing engine and (re)build the singleton with `kwargs`.

    Use to set the pool/connection settings explicitly before the first
    :func:`get_engine`, or to re-tune in a long-lived process. Async dispose of an
    already-created engine cannot run here (no event loop guarantee); this drops the
    reference so the next :func:`get_engine` builds fresh. Prefer awaiting
    :func:`dispose_engine` first in async code. See :func:`get_engine` for the knobs.
    """
    global _engine
    _engine = None
    return get_engine(**kwargs)


async def dispose_engine() -> None:
    """Dispose the shared engine and its pool (test teardown)."""
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None


class SqlCounter:
    """Counts SQL statements and records their text for an O(depth) assertion."""

    def __init__(self) -> None:
        self.count = 0
        self.statements: List[str] = []

    def record(self, statement: str) -> None:
        self.count += 1
        self.statements.append(statement)


@contextmanager
def count_sql() -> Iterator[SqlCounter]:
    """Count SQL statements executed on the shared engine while open.

    Attaches a ``before_cursor_execute`` listener to the engine's sync engine (the
    asyncpg dialect still drives the sync event hooks under the hood) and removes it
    on exit. Use to prove that a nested GraphQL operation issues one batched query
    per resource-layer.
    """
    counter = SqlCounter()
    sync_engine = get_engine().sync_engine

    def on_execute(conn, cursor, statement, parameters, context, executemany):
        counter.record(statement)

    event.listen(sync_engine, "before_cursor_execute", on_execute)
    try:
        yield counter
    finally:
        event.remove(sync_engine, "before_cursor_execute", on_execute)


__all__ = [
    "DATABASE_URL",
    "DEMO_SCHEMA",
    "get_engine",
    "configure_engine",
    "dispose_engine",
    "SqlCounter",
    "count_sql",
]
