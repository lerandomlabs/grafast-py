"""Request-scoped execution seam: BUILD SQL here, EXECUTE it on the host's pool.

The pg steps BUILD a SQLAlchemy Core statement at execute time but no longer decide HOW
it runs: a :class:`PgExecutor`, supplied by the host per request, runs it and returns
column-keyed dicts (what ``AccessStep`` expects). It is read from a pg-managed
:class:`~contextvars.ContextVar` set around ``await graphql(...)`` via
:func:`pg_request_context`; asyncio copies that context into every child task, so the
batched ``asyncio.gather`` fan-out reads the SAME executor in each concurrent step.

Two executors ship: :class:`SQLAlchemyExecutor` runs the Core statement on an
:class:`AsyncEngine` the library owns (``count_sql`` instruments its ``.engine``);
:class:`RawExecutor` compiles to ``($1, $2, …)`` positional SQL and hands it to a host
callback running on THEIR pool — proving build and execute are decoupled
(``col = ANY(:keys)`` stays a SINGLE ``$1::T[]`` array param, preserving O(depth)).

``settings`` carries pgSettings (RLS): when present, ``SQLAlchemyExecutor`` opens a
transaction, applies them all via ``set_config(key, value, true)`` (``is_local=true`` →
the GUCs are scoped to that transaction and auto-clear at commit, never leaking onto the
pooled connection), and runs the query in the same transaction. ``settings=None`` takes
the plain no-transaction path, leaving the O(depth) batching counts untouched.
"""

import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Iterator, List, Mapping, Optional, Protocol

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import asyncpg as asyncpg_dialect
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.sql import ClauseElement

from ..config import log

# Apply ALL pgSettings in ONE round-trip: json_each_text unpacks the bound JSON param
# into (key, value) text pairs, and set_config(..., is_local=true) scopes each GUC to
# THIS transaction (it clears at commit, never leaking onto the pooled connection). The
# settings JSON is bound as a param (cast(:__pg_settings as json)) — never interpolated.
SET_CONFIG_SQL = (
    "select set_config(t.key, t.value, true) "
    "from json_each_text(cast(:__pg_settings as json)) as t(key, value)"
)


class PgExecutor(Protocol):
    """Runs a SQLAlchemy Core statement and returns rows as column-keyed dicts.

    ``statement`` is a SQLAlchemy Core ``ClauseElement`` (or its compiled
    ``(text, params)`` pair for executors that compile themselves). ``params`` are the
    bind values (e.g. ``{"keys": [...]}`` for ``= ANY(:keys)``). ``settings`` are
    pgSettings to apply before the query for the duration of its transaction (RLS);
    ``None`` means apply none. ``commit`` requests a COMMITTING transaction (the mutation
    path): a write must persist, so its DML runs in a transaction that commits rather than
    the plain no-transaction read path (which rolls back on close).
    """

    async def run(
        self,
        statement: Any,
        params: Optional[Mapping[str, Any]],
        *,
        settings: Optional[Mapping[str, str]] = None,
        commit: bool = False,
    ) -> List[dict]:
        ...


@dataclass
class PgRequestContext:
    """The per-request execution context resolved from the host.

    ``executor`` runs every pg statement for this request; ``settings`` are the pgSettings
    (RLS) the steps pass through to ``executor.run`` — host-supplied per request via
    :func:`pg_request_context` and applied with ``set_config(..., true)``; ``None`` means
    no RLS scoping. ``context`` is the per-request value a resource's ``select_customizer``
    is resolved against (the GraphQL context analogue: tenant id, viewer, visibility
    flags); ``None`` means no context.
    """

    executor: PgExecutor
    settings: Optional[Mapping[str, str]] = None
    context: Any = None


class SQLAlchemyExecutor:
    """A :class:`PgExecutor` that runs Core statements on an :class:`AsyncEngine`.

    The engine is exposed as ``.engine`` so ``count_sql`` can attach its statement
    counter to the SAME instance the steps run against (the O(depth) batching gate).
    """

    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    async def run(
        self,
        statement: Any,
        params: Optional[Mapping[str, Any]],
        *,
        settings: Optional[Mapping[str, str]] = None,
        commit: bool = False,
    ) -> List[dict]:
        """Execute ``statement`` on the engine; apply ``settings`` in a txn first.

        ``settings`` or ``commit`` each force a transaction: settings apply per-request
        GUCs (RLS) for the query's duration, and ``commit`` makes a mutation's DML persist
        (the plain no-transaction read path rolls back on close, so a write would vanish).
        """
        async with self.engine.connect() as conn:
            if settings or commit:
                # A transaction is needed for RLS settings AND/OR a committing write. RLS:
                # apply ALL pgSettings via ONE set_config(..., is_local=true) statement so
                # they live only for THIS transaction, then run the query in the same txn
                # (the GUCs auto-clear at commit, never leaking on the pool). The
                # `conn.begin()` block commits on exit — persisting a mutation's write.
                async with conn.begin():
                    if settings:
                        await conn.execute(
                            text(SET_CONFIG_SQL),
                            {"__pg_settings": json.dumps(dict(settings))},
                        )
                    result = await conn.execute(statement, params or {})
                    return [dict(row) for row in result.mappings().all()]
            # No settings and no commit: plain connect + execute (no transaction), so the
            # O(depth) batching counts stay untouched on the read path.
            result = await conn.execute(statement, params or {})
            return [dict(row) for row in result.mappings().all()]


class RawExecutor:
    """A :class:`PgExecutor` that compiles to ``$1`` SQL and runs it on a host callback.

    ``run_callable(text, positional_params, settings)`` is the host's "run raw SQL on
    my pool" function; it returns ``list[dict]``. The Core statement is compiled with
    the asyncpg dialect (paramstyle ``numeric_dollar`` → ``$1, $2, …``) and
    ``render_postcompile=True`` so ``col = ANY(:keys)`` stays a SINGLE ``$1`` array
    param rather than expanding to inlined per-element literals.

    ``commit`` is accepted on :meth:`run` for protocol uniformity but the commit
    decision rests with the host callback: a :class:`RawExecutor` owns no engine, so the
    host's pool function controls the connection/transaction lifecycle (and thus whether a
    mutation's RETURNING-bearing DML persists).
    """

    def __init__(
        self,
        run_callable: Callable[..., Any],
        *,
        dialect: Optional[Any] = None,
    ) -> None:
        self._run_callable = run_callable
        # asyncpg paramstyle is numeric_dollar ($1); compiling against it yields the
        # positional SQL a raw asyncpg pool expects.
        self._dialect = dialect or asyncpg_dialect.dialect()

    async def run(
        self,
        statement: Any,
        params: Optional[Mapping[str, Any]],
        *,
        settings: Optional[Mapping[str, str]] = None,
        commit: bool = False,
    ) -> List[dict]:
        """Compile ``statement`` to ``$1`` SQL + positional params, then run it.

        ``commit`` is forwarded to the host callback so it can run a mutation's DML in a
        committing transaction on its own pool (the host owns the connection lifecycle).
        """
        if isinstance(statement, ClauseElement):
            compiled = statement.compile(
                dialect=self._dialect,
                compile_kwargs={"render_postcompile": True},
            )
            sql_text = str(compiled)
            merged = dict(compiled.params)
            if params:
                merged.update(params)
            # positional_params follows the compiled param order ($1, $2, …); for the
            # `= ANY(:keys)` selects this is exactly one array param ($1::T[]).
            positional_params = [merged[name] for name in compiled.positiontup]
        else:
            # already a (text, positional_params) pair.
            sql_text, positional_params = statement

        log.debug("pg raw execute", params=len(positional_params))
        rows = await self._run_callable(sql_text, positional_params, settings)
        return [dict(row) for row in rows]


_pg_request: ContextVar[PgRequestContext] = ContextVar("grafast_pg_request")


@contextmanager
def pg_request_context(
    executor: PgExecutor,
    settings: Optional[Mapping[str, str]] = None,
    context: Any = None,
) -> Iterator[PgRequestContext]:
    """Bind the per-request :class:`PgRequestContext` for the duration of the block.

    Set this around ``await graphql(...)``: asyncio copies the current context into
    each child task at task-creation time, so the batched ``asyncio.gather`` fan-out in
    the bucket executor reads the SAME executor in every concurrent step. The token is
    reset on exit so the binding never leaks past the request. ``context`` is the value a
    resource ``select_customizer`` resolves against (thread it the same way as the
    executor); planning runs inside this block, so a step reads it at construction time.
    """
    ctx = PgRequestContext(executor=executor, settings=settings, context=context)
    token = _pg_request.set(ctx)
    try:
        yield ctx
    finally:
        _pg_request.reset(token)


def current_pg_request() -> PgRequestContext:
    """Return the request-scoped :class:`PgRequestContext`, raising if unset.

    Fails loud rather than silently falling back to a process global: a pg step that
    runs without ``pg_request_context`` around the operation is a wiring bug.
    """
    try:
        return _pg_request.get()
    except LookupError:
        raise RuntimeError(
            "grafast_py.pg: no request-scoped executor — wrap the GraphQL "
            "execution in pg_request_context(executor) (set it around "
            "`await graphql(...)`) so the pg steps can run their statements."
        ) from None


__all__ = [
    "PgExecutor",
    "PgRequestContext",
    "SQLAlchemyExecutor",
    "RawExecutor",
    "pg_request_context",
    "current_pg_request",
]
