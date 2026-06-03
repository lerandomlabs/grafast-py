"""Async SQLAlchemy engine plumbing for the Postgres data source.

The database URL is supplied by the CONSUMER (``configure_engine(url=...)`` or the
``GRAFAST_PG_URL`` environment variable) — the library bakes in no database. A single
process-global :class:`AsyncEngine` is created lazily and reused across requests (one
connection pool); see :func:`get_engine` for the URL resolution, the pool knobs, and
the single-engine caveat.

``count_sql`` is a diagnostic helper (also used by the O(depth) batching tests): it
attaches a ``before_cursor_execute`` listener to the engine's underlying sync engine
and counts every statement executed while it is open. It is process-global (it counts
all SQL on the shared engine during the block), so it is for single-flow diagnostics /
tests, not per-request metering under concurrency.
"""

import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

# The data source has NO baked-in database: the URL comes from configure_engine(url=)
# / get_engine(url=) or this environment variable. get_engine() raises if neither is
# set, so the engine can never silently open an unintended database.
URL_ENV_VAR = "GRAFAST_PG_URL"

_engine: Optional[AsyncEngine] = None
_engine_url: Optional[str] = None


def _resolve_url(url: Optional[str]) -> str:
    """Return the explicit URL or ``$GRAFAST_PG_URL``; raise loudly if neither set."""
    resolved = url or os.environ.get(URL_ENV_VAR)
    if not resolved:
        raise RuntimeError(
            "grafast_py.pg: no database URL configured — pass url=... to "
            "configure_engine() / get_engine(), or set the GRAFAST_PG_URL env var "
            "(e.g. 'postgresql+asyncpg://user:pass@host/dbname')."
        )
    return resolved


def get_engine(
    *,
    url: Optional[str] = None,
    pool_size: int = 5,
    max_overflow: int = 10,
    pool_timeout: float = 30.0,
    pool_pre_ping: bool = False,
    pool_recycle: int = -1,
    connect_args: Optional[Dict[str, Any]] = None,
    echo: bool = False,
) -> AsyncEngine:
    """Return the shared async engine, creating it on first use.

    The database URL comes from ``url=`` or ``$GRAFAST_PG_URL`` (no database is baked
    in; raises if neither is set). Only the FIRST call (or an explicit
    :func:`configure_engine`) decides the URL + pool settings — later calls reuse the
    singleton and ignore kwargs (call :func:`configure_engine` / :func:`dispose_engine`
    to rebuild). Pool kwargs pass through to ``create_async_engine``.

    Concurrency / pool relationship: the effective ceiling on concurrent in-flight DB
    connections is ``pool_size + max_overflow`` (default 5 + 10 = 15); excess
    operations queue on checkout, raising the latency tail. **This pool ceiling — not
    ``GrafastConfig.max_step_concurrency`` — is the real bound on concurrent DB
    round-trips;** size it to your target concurrency. For per-statement bounds set a
    server-side ``statement_timeout`` via ``connect_args`` (the GrafastConfig
    execution timeout does not itself cancel in-flight SQL).

    Single-engine caveat: one process-global engine serves one URL. To talk to
    multiple databases in one process, dispose + reconfigure between them; a
    per-database registry is a known future improvement.
    """
    global _engine, _engine_url
    if _engine is None:
        _engine_url = _resolve_url(url)
        _engine = create_async_engine(
            _engine_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_pre_ping=pool_pre_ping,
            pool_recycle=pool_recycle,
            connect_args=connect_args or {},
            echo=echo,
        )
    return _engine


def configure_engine(*, url: Optional[str] = None, **kwargs: Any) -> AsyncEngine:
    """Dispose any existing engine and (re)build the singleton for ``url``.

    Use to set the URL + pool/connection settings explicitly, or to repoint at another
    database in a long-lived process. The previous engine's pool is disposed
    synchronously first (closing its connections) so reconfiguring does not leak it.
    See :func:`get_engine` for URL resolution and the pool knobs.
    """
    global _engine, _engine_url
    if _engine is not None:
        _engine.sync_engine.dispose()  # close the old pool now (sync) — no leak
        _engine = None
        _engine_url = None
    return get_engine(url=url, **kwargs)


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
    "URL_ENV_VAR",
    "get_engine",
    "configure_engine",
    "dispose_engine",
    "SqlCounter",
    "count_sql",
]
