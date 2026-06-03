"""Phase E hardening gtests: opt-in limits, tracing hooks, and pg pool config.

Each control is OPT-IN via a `GrafastConfig` attached to a *subclass* of
`GrafastExecutionContext` (never the shared base class, so the conformance harness's
global install is unaffected). The defaults are exercised implicitly by every other
suite (which uses the bare base context with `DEFAULT_CONFIG` = no limits).

Covered: execution timeout fires (async path), the operation/plan/step-batch tracing
hooks are called, and the pg engine honours a configured pool size. (Query cost/depth
limiting is intentionally a validation-layer concern, not part of this engine.)
"""

import asyncio
from contextlib import contextmanager

import pytest
from graphql import execute, parse

from grafast_py import (
    GrafastExecutionContext,
    access,
    load_one,
    make_grafast_schema,
)
from grafast_py.config import (
    GrafastConfig,
    GrafastTimeoutError,
)

SDL = """
type Query {
  user: User
}
type User {
  id: Int!
  name: String!
  best: User
}
"""

# a tiny self-referential object so we can nest `best` arbitrarily deep.
ROOT_USER = {"id": 1, "name": "Luke", "best": {"id": 2, "name": "Han", "best": None}}


def build_schema():
    def plan_user(parent, args, info):
        from grafast_py import constant

        return load_one(constant(0), lambda keys: [ROOT_USER])

    def plan_id(parent, args, info):
        return access(parent, ["id"])

    def plan_name(parent, args, info):
        return access(parent, ["name"])

    def plan_best(parent, args, info):
        return access(parent, ["best"])

    return make_grafast_schema(
        SDL,
        {
            "Query": {"user": plan_user},
            "User": {"id": plan_id, "name": plan_name, "best": plan_best},
        },
    )


def context_with(config: GrafastConfig):
    """A throwaway context subclass carrying `config` (keeps the base class clean)."""

    class _Ctx(GrafastExecutionContext):
        grafast_config = config

    return _Ctx


def run(query, config, variables=None):
    return execute(
        build_schema(),
        parse(query),
        execution_context_class=context_with(config),
        variable_values=variables,
    )


# --------------------------------------------------------------- exec timeout
def test_timeout_fires_on_slow_async_resolver():
    slow_sdl = "type Query { slow: Int }"

    def plan_slow(parent, args, info):
        from grafast_py import lambda_step, constant

        async def sleep_then(_):
            await asyncio.sleep(0.5)
            return 1

        return lambda_step(constant(0), sleep_then)

    schema = make_grafast_schema(slow_sdl, {"Query": {"slow": plan_slow}})

    async def go():
        return await execute(
            schema,
            parse("{ slow }"),
            execution_context_class=context_with(
                GrafastConfig(execution_timeout_s=0.05)
            ),
        )

    result = asyncio.run(go())
    assert result.errors, "expected a timeout error"
    err = result.errors[0]
    assert isinstance(err.original_error or err, GrafastTimeoutError)
    assert "execution timeout" in err.message


def test_no_timeout_when_under_budget():
    slow_sdl = "type Query { slow: Int }"

    def plan_slow(parent, args, info):
        from grafast_py import lambda_step, constant

        async def quick(_):
            await asyncio.sleep(0.0)
            return 7

        return lambda_step(constant(0), quick)

    schema = make_grafast_schema(slow_sdl, {"Query": {"slow": plan_slow}})

    async def go():
        return await execute(
            schema,
            parse("{ slow }"),
            execution_context_class=context_with(
                GrafastConfig(execution_timeout_s=5.0)
            ),
        )

    result = asyncio.run(go())
    assert not result.errors, result.errors
    assert result.data == {"slow": 7}


# --------------------------------------------------------------- tracing hook
def test_step_batch_tracing_hook_is_called():
    seen = []

    @contextmanager
    def span(step, count):
        seen.append((type(step).__name__, count))
        yield

    config = GrafastConfig(on_step_batch=lambda step, count: span(step, count))
    result = run("{ user { id name } }", config)
    assert not result.errors, result.errors
    # the bucket DAG ran: at least the user LoadOne + the access steps were traced,
    # each via the hook (one call per step batch).
    assert seen, "tracing hook was never called"
    names = {n for n, _ in seen}
    assert "LoadOneStep" in names


def test_operation_and_plan_hooks_are_called():
    calls = {"op": 0, "plan": 0}

    @contextmanager
    def op_span(context, operation):
        calls["op"] += 1
        yield

    @contextmanager
    def plan_span(context, operation):
        calls["plan"] += 1
        yield

    config = GrafastConfig(
        on_operation=lambda c, o: op_span(c, o),
        on_plan=lambda c, o: plan_span(c, o),
    )
    result = run("{ user { id } }", config)
    assert not result.errors, result.errors
    assert calls["op"] == 1
    assert calls["plan"] == 1


# ----------------------------------------------------------------- pg pool size
@pytest.mark.pg
def test_pg_pool_size_is_honored():
    from grafast_py.pg.engine import configure_engine, dispose_engine

    async def go():
        await dispose_engine()
        engine = configure_engine(pool_size=7, max_overflow=3)
        pool = engine.pool
        assert pool.size() == 7
        # effective ceiling = pool_size + max_overflow
        assert pool._max_overflow == 3
        await dispose_engine()

    asyncio.run(go())
