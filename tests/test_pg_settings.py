"""Per-request pgSettings / RLS and the bring-your-own-pool finalize.

These prove grafast-py's pgSettings responsibility: the request-scoped executor applies
each pgSetting with ``set_config(key, value, true)`` in the SAME transaction as the
query (``is_local=true`` → transaction-scoped GUCs that auto-clear at commit, never
leaking onto the pooled connection), and EVERY pg step type threads ``request.settings``
into ``executor.run`` so none silently bypasses RLS.

The set_config MECHANISM is verified RLS-independently by selecting
``current_setting('app.demo', true)`` (the ``grafast_demo.setting_probe`` view): the GUC
value rides back on every row regardless of the connecting role. RLS ENFORCEMENT (a
policy actually filtering rows) is gated on the role being subject to RLS — the local
scratch-DB user is a SUPERUSER / BYPASSRLS, which bypasses any policy unconditionally, so
that one filtering assertion is SKIPPED honestly (creating a non-bypass role is forbidden
here — it is a server-global object outside ``grafast_demo``).

DB tests are marked ``pg`` (deselectable ``-m 'not pg'``); they touch ONLY the
``grafast_demo`` schema of ``grafast_py_test`` via the dedicated ``setting_probe`` /
``secret_notes`` fixtures and do not alter authors/posts/comments.
"""

import pytest
import pytest_asyncio
from graphql import graphql
from sqlalchemy import any_, bindparam, column, func, select, table, text

from grafast_py.context import GrafastExecutionContext
from grafast_py.core_steps import access, constant
from grafast_py.schema import make_grafast_schema
from grafast_py.pg.connection import PgConnectionStep
from grafast_py.pg.engine import count_sql, dispose_engine, get_engine
from grafast_py.pg.executor import SQLAlchemyExecutor, pg_request_context
from grafast_py.pg.resource import PgRegistry, PgResource
from grafast_py.pg.steps import PgSelectAllStep, PgSelectSingleStep, PgSelectStep
from examples.demo_schema import build_demo_schema
from examples.seed import (
    setup_demo_schema,
    setup_rls_table,
    setup_settings_probe_view,
)

pytestmark = pytest.mark.pg

# The setting_probe view exposes current_setting('app.demo', true) as the `demo` column,
# keyed by id/owner_id (1 and 2) — see examples.seed.setup_settings_probe_view.
PROBE_COLUMNS = ["id", "owner_id", "demo"]


def make_probe() -> PgResource:
    """A fresh resource over the ``setting_probe`` view (its own registry)."""
    registry = PgRegistry()
    return PgResource(
        "setting_probe", "grafast_demo", "setting_probe", PROBE_COLUMNS,
        registry=registry,
    )


@pytest_asyncio.fixture
async def probe_seeded():
    """(Re)seed ``grafast_demo`` + the ``setting_probe`` view (fresh engine per test)."""
    await dispose_engine()
    await setup_demo_schema()
    await setup_settings_probe_view()
    yield
    await dispose_engine()


async def is_rls_bypass(engine) -> bool:
    """Whether the connecting role bypasses RLS (superuser or BYPASSRLS).

    Such a role is NEVER subject to a policy, so an RLS-filtering test cannot bite and
    must be skipped honestly (creating a non-bypass role is forbidden here).
    """
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "select rolsuper, rolbypassrls from pg_roles"
                    " where rolname = current_user"
                )
            )
        ).mappings().one()
    return bool(row["rolsuper"] or row["rolbypassrls"])


# ------------------------------------------------------ set_config MECHANISM (executor)


@pytest.mark.asyncio
async def test_set_config_applies_in_query_txn_and_clears_after(probe_seeded):
    """``set_config`` is set in the query's OWN txn and cleared on the next checkout.

    Selecting ``current_setting('app.demo', true)`` returns the per-request value WITH
    settings and ``''`` WITHOUT — and a second no-settings query does NOT see the prior
    value, proving the GUC is transaction-local (never leaks across the pool).
    """
    engine = get_engine()
    executor = SQLAlchemyExecutor(engine)
    stmt = select(func.current_setting("app.demo", True).label("demo"))

    with_settings = await executor.run(stmt, None, settings={"app.demo": "hello"})
    assert with_settings == [{"demo": "hello"}]

    no_settings = await executor.run(stmt, None, settings=None)
    assert no_settings == [{"demo": ""}]  # unset GUC reads '' (not None) on a fresh txn

    # a no-settings query AFTER a settings one does NOT see the prior value (txn-local).
    await executor.run(stmt, None, settings={"app.demo": "hello"})
    after = await executor.run(stmt, None)
    assert after == [{"demo": ""}]


@pytest.mark.asyncio
async def test_multiple_settings_applied_in_one_statement(probe_seeded):
    """ALL pgSettings apply in one round-trip; each is readable in the query's txn."""
    engine = get_engine()
    executor = SQLAlchemyExecutor(engine)
    stmt = select(
        func.current_setting("app.one", True).label("one"),
        func.current_setting("app.two", True).label("two"),
    )
    rows = await executor.run(
        stmt, None, settings={"app.one": "alpha", "app.two": "beta"}
    )
    assert rows == [{"one": "alpha", "two": "beta"}]


# --------------------------------------------- set_config MECHANISM via GraphQL end-to-end


def build_probe_schema() -> "object":
    """A tiny GraphQL schema whose ``setting`` field returns the per-request GUC.

    ``Query.settings`` is a root collection over the ``setting_probe`` view;
    ``Setting.demo`` projects ``current_setting('app.demo', true)`` (a view column), so a
    GraphQL query observes the GUC the executor set for the request.
    """
    sdl = """
    type Query { settings: [Setting!]! }
    type Setting { id: Int!  demo: String! }
    """
    probe = make_probe()

    def plan_settings(parent_step, args, info):
        return PgSelectAllStep(probe, order_by=["id"]).for_parent(parent_step)

    def leaf(key):
        def plan(parent_step, args, info):
            return access(parent_step, (key,))

        return plan

    return make_grafast_schema(
        sdl,
        {
            "Query": {"settings": plan_settings},
            "Setting": {"id": leaf("id"), "demo": leaf("demo")},
        },
    )


@pytest.mark.asyncio
async def test_graphql_query_sees_request_pg_settings(probe_seeded):
    """An end-to-end GraphQL query reflects the per-request pgSettings, then clears.

    With ``settings={'app.demo': 'hello'}`` every row's ``demo`` is ``'hello'``; without
    settings it is ``''`` — proving the GUC is set in the query's transaction and the
    pg steps thread it through, with no leak between the two requests.
    """
    schema = build_probe_schema()
    query = "{ settings { id demo } }"

    with pg_request_context(
        SQLAlchemyExecutor(get_engine()), settings={"app.demo": "hello"}
    ):
        with_result = await graphql(
            schema, query, execution_context_class=GrafastExecutionContext
        )
    assert with_result.errors is None
    assert {r["demo"] for r in with_result.data["settings"]} == {"hello"}

    with pg_request_context(SQLAlchemyExecutor(get_engine())):
        without_result = await graphql(
            schema, query, execution_context_class=GrafastExecutionContext
        )
    assert without_result.errors is None
    assert {r["demo"] for r in without_result.data["settings"]} == {""}


# ------------------------------------------- settings threaded to EVERY step type


@pytest.mark.asyncio
async def test_settings_threaded_to_select_all_step(probe_seeded):
    """``PgSelectAllStep`` (root collection) threads request.settings into the query."""
    with pg_request_context(
        SQLAlchemyExecutor(get_engine()), settings={"app.demo": "root"}
    ):
        step = PgSelectAllStep(make_probe(), order_by=["id"]).for_parent(constant(None))
        out = await step.execute(1, [[None]])
    assert {r["demo"] for r in out[0]} == {"root"}


@pytest.mark.asyncio
async def test_settings_threaded_to_select_step_hasmany(probe_seeded):
    """``PgSelectStep`` (hasMany ``= ANY(:keys)``) threads request.settings."""
    with pg_request_context(
        SQLAlchemyExecutor(get_engine()), settings={"app.demo": "many"}
    ):
        step = PgSelectStep(make_probe(), constant(None), "owner_id", order_by=["id"])
        out = await step.execute(2, [[1, 2]])
    # owner_id 1 -> row 1, owner_id 2 -> row 2; each row carries the GUC value.
    assert out[0][0]["demo"] == "many"
    assert out[1][0]["demo"] == "many"


@pytest.mark.asyncio
async def test_settings_threaded_to_select_single_step(probe_seeded):
    """``PgSelectSingleStep`` (hasOne / get) threads request.settings."""
    with pg_request_context(
        SQLAlchemyExecutor(get_engine()), settings={"app.demo": "single"}
    ):
        step = PgSelectSingleStep(make_probe(), constant(None), "id")
        out = await step.execute(2, [[1, 2]])
    assert out[0]["demo"] == "single"
    assert out[1]["demo"] == "single"


@pytest.mark.asyncio
async def test_settings_threaded_to_connection_step(probe_seeded):
    """``PgConnectionStep`` (Relay) threads request.settings into the windowed query."""
    with pg_request_context(
        SQLAlchemyExecutor(get_engine()), settings={"app.demo": "conn"}
    ):
        step = PgConnectionStep(
            make_probe(), constant(None), "owner_id", order_by=["id"], first=5
        )
        out = await step.execute(2, [[1, 2]])
    # each parent's connection nodes carry the GUC value in `demo`.
    assert out[0]["nodes"][0]["demo"] == "conn"
    assert out[1]["nodes"][0]["demo"] == "conn"


# ------------------------------------------------------------------- RLS end-to-end


@pytest_asyncio.fixture
async def rls_seeded():
    """(Re)seed ``grafast_demo`` + the ``secret_notes`` RLS fixture (fresh engine)."""
    await dispose_engine()
    await setup_demo_schema()
    await setup_rls_table()
    yield
    await dispose_engine()


@pytest.mark.asyncio
async def test_rls_policy_filters_by_request_setting(rls_seeded):
    """With ``settings={'app.owner': '1'}`` an RLS policy returns only owner-1 rows.

    Gated on the connecting role being subject to RLS: the local scratch-DB user is a
    SUPERUSER / BYPASSRLS that bypasses any policy unconditionally (creating a non-bypass
    role is forbidden here), so the filtering assertion is SKIPPED honestly. The
    set_config-mechanism tests above already prove grafast-py's responsibility (injecting
    the GUC into the query's transaction); the lack of filtering under a superuser is the
    bypass, not a defect in the pgSettings path.
    """
    engine = get_engine()
    if await is_rls_bypass(engine):
        pytest.skip(
            "connecting role bypasses RLS (superuser / BYPASSRLS), so a policy cannot "
            "filter rows; creating a non-bypass role is forbidden in this scratch DB. "
            "The set_config mechanism is proven by the current_setting tests above."
        )

    notes = PgResource(
        "secret_notes", "grafast_demo", "secret_notes", ["id", "owner", "body"],
        registry=PgRegistry(),
    )
    with pg_request_context(
        SQLAlchemyExecutor(engine), settings={"app.owner": "1"}
    ):
        step = PgSelectAllStep(notes, order_by=["id"]).for_parent(constant(None))
        out = await step.execute(1, [[None]])
    # only owner-1 rows survive the policy USING (owner = current_setting('app.owner')).
    assert [r["id"] for r in out[0]] == [1, 2]
    assert {r["owner"] for r in out[0]} == {1}


# ------------------------------------------- batching unchanged when settings is None


@pytest_asyncio.fixture
async def demo_schema():
    """Build the demo schema after (re)seeding ``grafast_demo`` (fresh engine per test)."""
    await dispose_engine()
    await setup_demo_schema()
    schema = build_demo_schema()
    yield schema
    await dispose_engine()


@pytest.mark.asyncio
async def test_no_settings_keeps_o_depth_batching(demo_schema):
    """With ``settings=None`` the executor takes the no-txn path; counts stay O(depth).

    The same nested query asserted in ``test_pg_datasource`` issues one batched statement
    per resource-layer — the no-settings path adds NO extra round-trips (no set_config, no
    transaction wrapper that the counter would see).
    """
    query = """
    {
      authors {
        id
        name
        posts {
          id
          title
          author { id name }
          comments { id body author { name } }
        }
      }
    }
    """
    with count_sql(get_engine()) as counter:
        with pg_request_context(SQLAlchemyExecutor(get_engine())):
            result = await graphql(
                demo_schema, query, execution_context_class=GrafastExecutionContext
            )
    assert result.errors is None
    assert counter.count == 5  # authors + posts + post.author + comments + comment.author
