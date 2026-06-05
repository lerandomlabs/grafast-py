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

``shared_txn`` is an OPT-IN per-request mode: when set on :func:`pg_request_context`, the
whole operation runs on ONE connection inside ONE ``REPEATABLE READ`` transaction. The
isolation level and the pgSettings are applied ONCE at transaction open (a single ``SET
LOCAL`` round-trip + one ``set_config(..., true)``), not per statement, so every batched
read in the operation observes a single consistent snapshot (no inter-statement skew
between the depth-D resource layers) and pays the settings round-trip only once. The held
connection/transaction lives on the :class:`PgRequestContext`; the reads roll the txn back
on close (a read request persists nothing), while a mutation runs with ``commit=True`` and
COMMITS the single request transaction. The default (``shared_txn=False``) path is
UNCHANGED — each statement still opens its own short-lived connection. Because the mode is
a REQUEST-level constant (one operation = one mode), it does NOT enter any per-step dedup
key: it never changes the SQL TEXT a step emits, and folding a request-constant would only
over-discriminate (see :class:`PgRequestContext`).

TRADE-OFF — shared_txn SERIALISES the request's read fan-out: with every statement running
on the ONE held :class:`AsyncConnection`, SQLAlchemy serialises overlapping ``executor.run``
calls (the sibling-field ``asyncio.gather`` fan-out) via its internal greenlet lock rather
than running them in parallel — so a depth-D request that would otherwise issue concurrent
reads runs them SEQUENTIALLY. This costs latency on read-heavy requests in exchange for the
single consistent snapshot; the default per-statement path keeps the intra-request
concurrency (each statement on its own connection). Enable shared_txn when snapshot
consistency across the depth-D layers matters more than the concurrency it gives up.
"""

import asyncio
import json
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
)

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import asyncpg as asyncpg_dialect
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine
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

# Pin the shared request transaction to REPEATABLE READ so every batched read in the
# operation reads the SAME snapshot (no inter-statement skew across the depth-D resource
# layers). Issued ONCE right after the transaction opens (before any query) — it is a
# transaction-property statement, not interpolated user data.
SET_TRANSACTION_REPEATABLE_READ = (
    "set transaction isolation level repeatable read"
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

    ``shared_txn`` opts the WHOLE operation into a single connection inside a single
    ``REPEATABLE READ`` transaction (see the module docstring). When set, the
    :class:`SQLAlchemyExecutor` holds the connection + transaction on this context
    (``_txn_conn`` / ``_txn``) and reuses them for every statement, applying the isolation
    level and pgSettings ONCE at txn open rather than per statement; the reads roll back on
    close, a mutation commits the held transaction. The held attributes are runtime state
    (not constructor args) and are torn down by :func:`pg_request_context` on block exit.
    Because every statement shares the one held connection, the request's read fan-out runs
    SERIALLY (SQLAlchemy serialises overlapping statements on a single connection) — the
    snapshot guarantee is bought at the cost of intra-request query concurrency (module
    docstring TRADE-OFF).

    DEDUP NOTE: ``shared_txn`` is deliberately NOT a step input and is NOT folded into any
    step's ``peer_key`` / ``dedup_params``. The mode is a REQUEST-level constant (one
    operation runs under exactly one mode) and it does not change the SQL TEXT any step
    emits — it changes only HOW the identical statements are run (one held connection/txn vs
    a fresh connection each). Folding a value that is constant across the whole dedup set
    could only OVER-discriminate (it can never make two genuinely different statements look
    alike), so it would merely defeat valid deduplication. The shared_txn dedup test asserts
    that toggling the mode leaves every step key unchanged within a request.
    """

    executor: PgExecutor
    settings: Optional[Mapping[str, str]] = None
    context: Any = None
    shared_txn: bool = False
    # Runtime state for the shared-transaction mode: the single held connection and its
    # open transaction, lazily acquired on the first statement and torn down on block exit.
    # Not constructor args (they are an execution detail, not host configuration).
    _txn_conn: Optional[AsyncConnection] = field(default=None, init=False, repr=False)
    _txn: Any = field(default=None, init=False, repr=False)
    # Serialises the LAZY open of the held connection: the open is a check-then-act over an
    # `await` (connect/begin yields the loop), so two concurrent first statements — the
    # sibling-field gather fan-out — would otherwise both pass the None-guard and each open a
    # separate connection+transaction, leaking one. The lock makes the first opener win and
    # every racer re-check under it, so exactly ONE connection/transaction is ever held.
    _open_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    async def aclose(self) -> None:
        """Release the held shared-txn connection: roll back the read txn and close it.

        Called by :func:`pg_request_context_async` on block exit. The reads never committed
        (a read request persists nothing), so the open ``REPEATABLE READ`` transaction is
        rolled back and the connection returned to the pool. A mutation already committed and
        cleared the held state (``_txn_conn`` is ``None``), so this is then a no-op. Idempotent
        and safe when ``shared_txn`` was never enabled (nothing was ever held).
        """
        if self._txn_conn is None:
            return
        if self._txn is not None:
            await self._txn.rollback()
            self._txn = None
        await self._txn_conn.close()
        self._txn_conn = None


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

        When the request opted into ``shared_txn`` (:func:`pg_request_context`), every
        statement instead runs on ONE held connection inside ONE ``REPEATABLE READ``
        transaction, opened lazily on the first statement with the isolation level + the
        pgSettings applied ONCE (see :meth:`run_in_shared_txn`). The ``shared_txn=False``
        default path below is UNCHANGED.
        """
        request = _current_request_or_none()
        if request is not None and request.shared_txn:
            return await self.run_in_shared_txn(
                request, statement, params, settings=settings, commit=commit
            )
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

    async def run_in_shared_txn(
        self,
        request: "PgRequestContext",
        statement: Any,
        params: Optional[Mapping[str, Any]],
        *,
        settings: Optional[Mapping[str, str]] = None,
        commit: bool = False,
    ) -> List[dict]:
        """Run ``statement`` on the request's single held ``REPEATABLE READ`` transaction.

        The first statement of the request opens the connection + transaction (held on
        ``request``), pins it to ``REPEATABLE READ`` and applies the pgSettings ONCE — a
        single ``SET LOCAL`` round-trip plus one ``set_config(..., true)`` for the whole
        operation, not per statement. Every later statement reuses that same connection, so
        all batched reads observe ONE consistent snapshot. The transaction is left open for
        the reads (rolled back on block exit — a read request persists nothing); a mutation
        passes ``commit=True``, which COMMITS the held transaction and clears the held state
        so any following statement opens a fresh one.
        """
        conn = await self.shared_txn_connection(request, settings)
        result = await conn.execute(statement, params or {})
        rows = [dict(row) for row in result.mappings().all()]
        if commit:
            # a mutation must persist: commit the single request transaction, then clear
            # the held state so a subsequent statement re-opens a fresh REPEATABLE READ txn.
            await request._txn.commit()
            request._txn = None
            await request._txn_conn.close()
            request._txn_conn = None
        return rows

    async def shared_txn_connection(
        self, request: "PgRequestContext", settings: Optional[Mapping[str, str]]
    ) -> AsyncConnection:
        """Return the request's held connection, opening the shared transaction on first use.

        On the first call it acquires a connection, begins a transaction, issues ``SET
        TRANSACTION ISOLATION LEVEL REPEATABLE READ`` (before any query, as Postgres
        requires) and — when ``settings`` are present — applies them ONCE via one
        ``set_config(..., true)`` so the GUCs are transaction-local. Later calls return the
        already-held connection untouched (the isolation level and GUCs are set once per
        transaction, never re-issued).

        The lazy open is SERIALISED under the request's ``_open_lock`` with a re-check inside:
        the open spans an ``await`` (connect/begin yields the loop), so two concurrent first
        statements (the sibling-field gather fan-out) would otherwise both pass the un-held
        None-guard and each open a connection+transaction — the last writer winning ``_txn_conn``
        and the other leaking. The fast path below returns the already-held connection WITHOUT
        taking the lock (the common steady state); only an un-opened request contends.
        """
        if request._txn_conn is not None:
            return request._txn_conn
        async with request._open_lock:
            # re-check under the lock: a racer that opened the connection while we waited wins,
            # and we return its connection rather than opening (and leaking) a second one.
            if request._txn_conn is not None:
                return request._txn_conn
            conn = await self.engine.connect()
            txn = await conn.begin()
            await conn.execute(text(SET_TRANSACTION_REPEATABLE_READ))
            if settings:
                await conn.execute(
                    text(SET_CONFIG_SQL),
                    {"__pg_settings": json.dumps(dict(settings))},
                )
            request._txn_conn = conn
            request._txn = txn
            return conn


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
    *,
    shared_txn: bool = False,
) -> Iterator[PgRequestContext]:
    """Bind the per-request :class:`PgRequestContext` for the duration of the block.

    Set this around ``await graphql(...)``: asyncio copies the current context into
    each child task at task-creation time, so the batched ``asyncio.gather`` fan-out in
    the bucket executor reads the SAME executor in every concurrent step. The token is
    reset on exit so the binding never leaks past the request. ``context`` is the value a
    resource ``select_customizer`` resolves against (thread it the same way as the
    executor); planning runs inside this block, so a step reads it at construction time.

    ``shared_txn`` is REJECTED here: its held connection is async and cannot be rolled
    back/closed from a SYNCHRONOUS contextmanager's ``finally`` (that would leak the
    transaction). Use :func:`pg_request_context_async` for ``shared_txn`` — it awaits
    :meth:`PgRequestContext.aclose` on exit. Rejecting at ENTRY (before any connection is
    opened) keeps the failure clean: nothing is ever held to leak.
    """
    if shared_txn:
        raise RuntimeError(
            "grafast_py.pg: shared_txn requires the async pg_request_context_async(...) "
            "(an `async with`) — the sync pg_request_context cannot await the held "
            "connection's rollback/close on exit, which would leak the transaction."
        )
    ctx = PgRequestContext(
        executor=executor, settings=settings, context=context, shared_txn=shared_txn
    )
    token = _pg_request.set(ctx)
    try:
        yield ctx
    finally:
        _pg_request.reset(token)


@asynccontextmanager
async def pg_request_context_async(
    executor: PgExecutor,
    settings: Optional[Mapping[str, str]] = None,
    context: Any = None,
    *,
    shared_txn: bool = False,
) -> "AsyncIterator[PgRequestContext]":
    """Async sibling of :func:`pg_request_context` that awaits the shared-txn teardown.

    Identical binding to :func:`pg_request_context` but as an ``async with``, so its exit
    can ``await`` :meth:`PgRequestContext.aclose` — rolling back and closing the held
    ``REPEATABLE READ`` connection a ``shared_txn`` request opened (a read persists nothing;
    a mutation already committed and cleared the held state). With ``shared_txn=False`` this
    holds nothing and is equivalent to the sync form.
    """
    ctx = PgRequestContext(
        executor=executor, settings=settings, context=context, shared_txn=shared_txn
    )
    token = _pg_request.set(ctx)
    try:
        yield ctx
    finally:
        _pg_request.reset(token)
        await ctx.aclose()


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


def _current_request_or_none() -> Optional[PgRequestContext]:
    """The request-scoped context, or ``None`` when unbound.

    The executor reads this to detect ``shared_txn`` WITHOUT requiring a bound context: a
    :class:`SQLAlchemyExecutor` can be driven directly (no ``pg_request_context``), in which
    case there is no shared-txn mode to honour and the default per-statement path runs.
    """
    return _pg_request.get(None)


__all__ = [
    "PgExecutor",
    "PgRequestContext",
    "SQLAlchemyExecutor",
    "RawExecutor",
    "pg_request_context",
    "pg_request_context_async",
    "current_pg_request",
]
