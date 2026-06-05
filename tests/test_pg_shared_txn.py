"""Opt-in per-request shared transaction (``shared_txn``): single connection, REPEATABLE READ.

These prove the executor-level mode added on :class:`PgRequestContext`: when a request
opts in, EVERY pg statement runs on ONE held connection inside ONE ``REPEATABLE READ``
transaction, with the isolation level + pgSettings applied ONCE at txn open (a single
``SET LOCAL`` round-trip), not per statement. The default (``shared_txn=False``) path is
unchanged — each statement opens its own short-lived connection.

Two kinds of test live here:

* a NO-DB dedup-correctness test — the txn mode is a REQUEST-level constant that does NOT
  change the SQL a step emits, so it is NOT folded into any step's dedup key. Identical
  steps built under ``shared_txn=True`` vs ``False`` MUST keep identical ``peer_key`` /
  ``dedup_params`` (folding a request-constant would only over-discriminate).

* DB-backed tests (marked ``pg``) — driving :class:`SQLAlchemyExecutor` inside the shared
  transaction and observing the BACKEND PID (one connection), the transaction id (one
  transaction), ``transaction_isolation`` (``repeatable read``), the ``SET TRANSACTION``
  count (issued once for the whole request), and that a mutation COMMITS the held txn. They
  touch only ``grafast_demo``.
"""

import asyncio
from contextlib import contextmanager

import pytest
import pytest_asyncio
from sqlalchemy import column, func, select

from grafast_py.core_steps import constant
from grafast_py.pg.connection import PgConnectionStep
from grafast_py.pg.engine import count_sql, dispose_engine, get_engine
from grafast_py.pg.executor import (
    SET_TRANSACTION_REPEATABLE_READ,
    PgRequestContext,
    SQLAlchemyExecutor,
    _pg_request,
    pg_request_context,
    pg_request_context_async,
)
from grafast_py.pg.mutations import PgInsertSingleStep
from grafast_py.pg.steps import PgSelectAllStep, PgSelectSingleStep, PgSelectStep
from examples.demo_schema import build_registry
from examples.seed import setup_demo_schema


# --------------------------------------------------------------- dedup correctness (no DB)


def dedup_key(step):
    """The class + peer_key + dedup_params slice of the planner's structural key."""
    return (type(step), step.peer_key, step.dedup_params())


@contextmanager
def request_in_mode(shared_txn):
    """Bind a :class:`PgRequestContext` in the given mode for step construction only.

    Constructs the context directly (no engine, no connection opened) and binds it on the
    pg request contextvar so a step built inside reads the same per-request context it would
    in a real request — letting the dedup tests prove the key is identical across modes
    without standing up a database. (The sync ``pg_request_context`` rejects shared_txn at
    entry because it cannot await the held connection's teardown; here nothing is opened.)
    """
    ctx = PgRequestContext(
        executor=SQLAlchemyExecutor.__new__(SQLAlchemyExecutor), shared_txn=shared_txn
    )
    token = _pg_request.set(ctx)
    try:
        yield ctx
    finally:
        _pg_request.reset(token)


def make_select():
    """A hasMany select over the demo ``posts`` resource (one per call, fresh registry)."""
    _registry, _authors, posts, _comments = build_registry()
    return PgSelectStep(posts, constant(None), "author_id", order_by=["id"])


def make_connection():
    """A Relay connection over the demo ``posts`` resource."""
    _registry, _authors, posts, _comments = build_registry()
    return PgConnectionStep(
        posts, constant(None), "author_id", order_by=["id"], first=5, needs_total=True
    )


def test_txn_mode_does_not_change_step_keys():
    """``shared_txn`` is a request constant: building a step under either mode is identical.

    The mode lives on :class:`PgRequestContext`, never on the step, and it does not change
    the SQL TEXT a step emits — so two byte-identical steps must produce the SAME dedup key
    regardless of which mode the surrounding request is in. The step keys are computed
    purely from step state (resource / match / order / page / customization), so the mode
    cannot leak in.
    """
    # build the SAME step under a shared_txn=False request and a shared_txn=True request;
    # the key is computed from step state, so both contexts must yield the identical key.
    with request_in_mode(shared_txn=False):
        plain = make_select()
        plain_key = dedup_key(plain)
    with request_in_mode(shared_txn=True):
        shared = make_select()
        shared_key = dedup_key(shared)

    assert plain_key == shared_key
    assert plain.peer_key == shared.peer_key
    assert plain.dedup_params() == shared.dedup_params()


def test_txn_mode_constant_within_a_request_keeps_steps_peers():
    """Two identical steps in ONE shared_txn request remain peers (the mode adds no skew).

    Within a single request the mode is the SAME constant for every step, so it can never
    discriminate one step from another — two identical selects stay peers, exactly as they
    would under the default mode.
    """
    with request_in_mode(shared_txn=True):
        a = make_select()
        b = make_select()
    assert dedup_key(a) == dedup_key(b)


def test_txn_mode_does_not_change_connection_keys():
    """The request-constant invariant holds for the connection step's richer key too."""
    with request_in_mode(shared_txn=False):
        plain_key = dedup_key(make_connection())
    with request_in_mode(shared_txn=True):
        shared_key = dedup_key(make_connection())
    assert plain_key == shared_key


# ------------------------------------------------------------------- DB-backed behaviour
# Only the DB-backed tests below carry the `pg` marker (per test, not module-wide) so the
# no-DB dedup tests above still run under `-m 'not pg'`.


@pytest_asyncio.fixture
async def seeded():
    """(Re)seed ``grafast_demo`` with a fresh engine, and restore it on teardown."""
    await dispose_engine()
    await setup_demo_schema()
    yield
    await dispose_engine()
    await setup_demo_schema()
    await dispose_engine()


# probes run on the held connection to observe its identity / isolation / txn.
BACKEND_PID = select(func.pg_backend_pid().label("pid"))
ISOLATION = select(
    func.current_setting("transaction_isolation", True).label("iso")
)
TXID = select(func.txid_current().label("txid"))


@pytest.mark.pg
@pytest.mark.asyncio
async def test_shared_txn_runs_all_statements_on_one_connection(seeded):
    """Every statement in a shared_txn request runs on the SAME backend (one connection)."""
    executor = SQLAlchemyExecutor(get_engine())
    async with pg_request_context_async(executor, shared_txn=True):
        first = (await executor.run(BACKEND_PID, None))[0]["pid"]
        second = (await executor.run(BACKEND_PID, None))[0]["pid"]
        third = (await executor.run(BACKEND_PID, None))[0]["pid"]
    assert first == second == third


@pytest.mark.pg
@pytest.mark.asyncio
async def test_shared_txn_is_one_repeatable_read_transaction(seeded):
    """All statements share ONE transaction id and read at ``repeatable read``."""
    executor = SQLAlchemyExecutor(get_engine())
    async with pg_request_context_async(executor, shared_txn=True):
        iso_a = (await executor.run(ISOLATION, None))[0]["iso"]
        txid_a = (await executor.run(TXID, None))[0]["txid"]
        iso_b = (await executor.run(ISOLATION, None))[0]["iso"]
        txid_b = (await executor.run(TXID, None))[0]["txid"]
    # the isolation level is the pinned REPEATABLE READ for every statement...
    assert iso_a == iso_b == "repeatable read"
    # ...and the txid is identical across statements (a single transaction).
    assert txid_a == txid_b


@pytest.mark.pg
@pytest.mark.asyncio
async def test_default_mode_is_unchanged(seeded):
    """Without shared_txn each read runs on its own connection at the default isolation.

    The default per-statement path opens a fresh connection (no enclosing transaction), so
    consecutive reads land on DIFFERENT backends and read at the server default
    (``read committed``) — proving the new mode does not perturb the default path.
    """
    executor = SQLAlchemyExecutor(get_engine())
    with pg_request_context(executor):  # shared_txn defaults to False
        iso = (await executor.run(ISOLATION, None))[0]["iso"]
    assert iso == "read committed"


@pytest.mark.pg
@pytest.mark.asyncio
async def test_set_transaction_issued_once_for_the_whole_request(seeded):
    """The isolation-level SET LOCAL is issued ONCE per request, not per statement.

    Three data statements run inside the shared transaction, but the ``SET TRANSACTION
    ISOLATION LEVEL REPEATABLE READ`` appears exactly once in the executed SQL — it is set
    at txn open, never re-issued.
    """
    executor = SQLAlchemyExecutor(get_engine())
    with count_sql(get_engine()) as counter:
        async with pg_request_context_async(executor, shared_txn=True):
            await executor.run(BACKEND_PID, None)
            await executor.run(BACKEND_PID, None)
            await executor.run(BACKEND_PID, None)
    set_txn = [s for s in counter.statements if s == SET_TRANSACTION_REPEATABLE_READ]
    assert len(set_txn) == 1


@pytest.mark.pg
@pytest.mark.asyncio
async def test_settings_applied_once_at_txn_open(seeded):
    """pgSettings ride the held txn for every statement, applied once at open.

    ``set_config(..., true)`` runs ONCE at txn open, yet the GUC is readable by EVERY later
    statement in the same transaction (it is transaction-local). The ``select set_config``
    statement appears exactly once across the three reads.
    """
    executor = SQLAlchemyExecutor(get_engine())
    probe = select(func.current_setting("app.demo", True).label("demo"))
    with count_sql(get_engine()) as counter:
        async with pg_request_context_async(
            executor, settings={"app.demo": "shared"}, shared_txn=True
        ):
            one = (await executor.run(probe, None, settings={"app.demo": "shared"}))[0]
            two = (await executor.run(probe, None, settings={"app.demo": "shared"}))[0]
    assert one["demo"] == "shared"
    assert two["demo"] == "shared"
    set_config = [s for s in counter.statements if s.startswith("select set_config")]
    assert len(set_config) == 1


@pytest.mark.pg
@pytest.mark.asyncio
async def test_concurrent_first_statements_open_exactly_one_connection(seeded):
    """Two statements racing to be the request's FIRST share one connection (no open race).

    The lazy open is a check-then-act over an ``await``, so without serialisation two
    concurrent first statements (the sibling-field gather fan-out) would each open a separate
    connection+transaction and leak one. Firing the two probes with NO warm-up, on one held
    transaction, they must land on the SAME backend pid AND the SAME txid, and the pool must
    report no leaked (checked-out) connection after the request exits.
    """
    engine = get_engine()
    executor = SQLAlchemyExecutor(engine)
    async with pg_request_context_async(executor, shared_txn=True):
        # both probes are the FIRST statements of the request — they contend for the open.
        pids_a, pids_b = await asyncio.gather(
            executor.run(BACKEND_PID, None),
            executor.run(TXID, None),
        )
        first_pid = pids_a[0]["pid"]
        first_txid = pids_b[0]["txid"]
        second_pid = (await executor.run(BACKEND_PID, None))[0]["pid"]
        second_txid = (await executor.run(TXID, None))[0]["txid"]
    # one connection (same backend) and one transaction (same txid) across the racers.
    assert first_pid == second_pid
    assert first_txid == second_txid
    # the held connection was rolled back and returned: nothing leaked from the open race.
    assert engine.sync_engine.pool.checkedout() == 0


@pytest.mark.pg
@pytest.mark.asyncio
async def test_shared_txn_threads_through_select_steps(seeded):
    """Real pg STEPS (not raw executor calls) run on the one held connection too.

    A root collection step and a hasMany step driven inside the shared transaction both
    observe the SAME backend pid as a direct probe — the steps thread through
    ``request.executor.run``, which honours the held connection.
    """
    executor = SQLAlchemyExecutor(get_engine())
    _registry, authors, posts, _comments = build_registry()
    async with pg_request_context_async(executor, shared_txn=True):
        probe_pid = (await executor.run(BACKEND_PID, None))[0]["pid"]
        all_step = PgSelectAllStep(authors, order_by=["id"]).for_parent(constant(None))
        await all_step.execute(1, [[None]])
        many = PgSelectStep(posts, constant(None), "author_id", order_by=["id"])
        await many.execute(1, [[1]])
        single = PgSelectSingleStep(authors, constant(None), "id")
        await single.execute(1, [[1]])
        # the connection is still the held one after the steps ran.
        after_pid = (await executor.run(BACKEND_PID, None))[0]["pid"]
    assert probe_pid == after_pid


@pytest.mark.pg
@pytest.mark.asyncio
async def test_read_only_shared_txn_rolls_back_on_exit(seeded):
    """A read-only shared_txn request persists nothing: the held txn rolls back on close.

    Driving an INSERT step with ``commit`` NOT requested (commit only comes from the
    mutation seam) leaves the row un-persisted — the held read transaction is rolled back on
    block exit, and a fresh request does not see the row.
    """
    executor = SQLAlchemyExecutor(get_engine())
    _registry, _authors, posts, _comments = build_registry()
    # build the insert via the mutation step, but run it WITHOUT commit (a plain
    # executor.run, not the committing mutation path), so the held read txn rolls it back.
    insert = PgInsertSingleStep(
        posts, {"id": 9100, "author_id": 1, "title": "rollback me"}
    )
    built, built_params = insert.build_statement([], 0)
    async with pg_request_context_async(executor, shared_txn=True):
        await executor.run(built, built_params)  # commit=False -> rolled back on exit

    # a NEW request must not see the un-committed row.
    check = select(func.count().label("n")).select_from(
        posts_table(posts)
    ).where(column("id") == 9100)
    async with pg_request_context_async(executor, shared_txn=True):
        n = (await executor.run(check, None))[0]["n"]
    assert n == 0


@pytest.mark.pg
@pytest.mark.asyncio
async def test_mutation_commits_the_shared_txn(seeded):
    """Under shared_txn a mutation step (commit=True) COMMITS the held transaction.

    The insert step runs with ``commit=True`` on the serial mutation seam; under shared_txn
    that commits the single request transaction, so a SUBSEQUENT request (a fresh shared
    txn) sees the persisted row.
    """
    executor = SQLAlchemyExecutor(get_engine())
    _registry, _authors, posts, _comments = build_registry()
    insert = PgInsertSingleStep(
        posts, {"id": 9200, "author_id": 1, "title": "committed via shared txn"}
    )
    async with pg_request_context_async(executor, shared_txn=True):
        out = await insert.execute(1, [])
    assert out[0]["id"] == 9200
    assert out[0]["title"] == "committed via shared txn"

    check = select(func.count().label("n")).select_from(
        posts_table(posts)
    ).where(column("id") == 9200)
    async with pg_request_context_async(executor, shared_txn=True):
        n = (await executor.run(check, None))[0]["n"]
    assert n == 1


def posts_table(posts):
    """A Core ``table`` over the posts resource (for the verification COUNT)."""
    from sqlalchemy import table

    return table(
        posts.table,
        *[column(c) for c in posts.columns],
        schema=posts.schema,
    )


# --------------------------------------------------------- sync contextmanager guard rail


def test_sync_context_rejects_shared_txn_at_entry():
    """The SYNC ``pg_request_context`` refuses ``shared_txn`` BEFORE opening anything.

    Its held connection is async and cannot be awaited closed from a sync ``finally``, so
    the guard rejects at ENTRY (nothing is ever opened to leak) and steers callers to
    :func:`pg_request_context_async`. No DB needed: it fails before any connection.
    """
    executor = SQLAlchemyExecutor.__new__(SQLAlchemyExecutor)
    with pytest.raises(RuntimeError, match="pg_request_context_async"):
        with pg_request_context(executor, shared_txn=True):
            pass
