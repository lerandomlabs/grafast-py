"""Hardening tests: opt-in limits, tracing hooks, and pg pool config.

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
from inspect import getcoroutinestate, iscoroutine

import pytest
from graphql import build_schema as build_graphql_schema, execute, parse

from grafast_py import (
    GrafastExecutionContext,
    access,
    constant,
    each,
    encode_global_id,
    first_step,
    lambda_step,
    list_step,
    load_one,
    make_grafast_schema,
    node,
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


@pytest.mark.parametrize("limit", [1, 2, 5])
@pytest.mark.parametrize("mode", ["resolver", "batch", "list"])
def test_nested_async_work_completes_within_concurrency_limit(limit, mode):
    """A parent waiting for children must not occupy an active-work permit.

    This is the artwork → tokens → owners → address → profile failure shape:
    nesting exceeds the configured limit, and multiple branches still compete
    for the same per-operation budget. No database is needed to reproduce it.
    """

    async def go():
        active = 0
        peak = 0

        async def load(keys):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0)
                return [{"id": key + 1} for key in keys]
            finally:
                active -= 1

        fields = "a: Node b: Node c: Node"
        if mode == "list":
            fields = "a: [Node!] b: [Node!] c: [Node!]"
        sdl = "type Query { " + fields + " } type Node { id: Int! next: Node }"
        if mode == "batch":
            from grafast_py import constant

            schema = make_grafast_schema(
                sdl,
                {
                    "Query": {
                        name: lambda p, a, i: constant({"id": 0}) for name in "abc"
                    },
                    "Node": {
                        "id": lambda p, a, i: access(p, ["id"]),
                        "next": lambda p, a, i: load_one(access(p, ["id"]), load),
                    },
                },
            )
        else:
            schema = build_graphql_schema(sdl)

            async def root(_parent, _info):
                return [{"id": 0}, {"id": 0}] if mode == "list" else {"id": 0}

            async def next_node(parent, _info):
                return (await load([parent["id"]]))[0]

            for field in schema.query_type.fields.values():
                field.resolve = root
            schema.get_type("Node").fields["next"].resolve = next_node

        selection = "id"
        expected = {"id": 8}
        for depth in reversed(range(8)):
            selection = "id next { " + selection + " }"
            expected = {"id": depth, "next": expected}
        result = await asyncio.wait_for(
            execute(
                schema,
                parse(
                    "{ "
                    + " ".join(name + " { " + selection + " }" for name in "abc")
                    + " }"
                ),
                execution_context_class=context_with(
                    GrafastConfig(max_step_concurrency=limit)
                ),
            ),
            timeout=1,
        )
        assert result.errors is None
        assert result.data == {
            name: [expected, expected] if mode == "list" else expected for name in "abc"
        }
        assert active == 0
        assert peak <= limit
        # Independent branches should fill the available budget, rather than
        # accidentally fixing the deadlock by serializing all execution.
        assert peak == min(6 if mode == "list" else 3, limit)

    asyncio.run(go())


@pytest.mark.parametrize("predicate", [asyncio.iscoroutine, iscoroutine])
@pytest.mark.parametrize("mode", ["batch", "each", "node"])
def test_cancel_nested_loads_closes_queued_coroutines(mode, predicate):
    async def go():
        started = asyncio.Event()
        blocked = asyncio.Event()
        created = []
        active = 0

        async def load(keys):
            nonlocal active
            active += 1
            started.set()
            try:
                await blocked.wait()
                return [{"id": key + 1} for key in keys]
            finally:
                active -= 1

        def track_load(keys):
            coroutine = load(keys)
            created.append(coroutine)
            return coroutine

        def plan_next(parent, _args, _info):
            if mode == "each":
                return first_step(
                    each(
                        list_step([parent]),
                        lambda item: load_one(access(item, ["id"]), track_load),
                    )
                )
            if mode == "node":
                return node(
                    lambda_step(
                        access(parent, ["id"]),
                        lambda key: encode_global_id("Node", key),
                    ),
                    {"Node": track_load},
                )
            return load_one(access(parent, ["id"]), track_load)

        schema = make_grafast_schema(
            "type Query { a: Node b: Node c: Node } type Node { id: Int! next: Node }",
            {
                "Query": {name: lambda p, a, i: constant({"id": 0}) for name in "abc"},
                "Node": {"id": lambda p, a, i: access(p, ["id"]), "next": plan_next},
            },
        )

        async def query():
            return await execute(
                schema,
                parse("{ a { next { id } } b { next { id } } c { next { id } } }"),
                execution_context_class=context_with(
                    GrafastConfig(max_step_concurrency=1)
                ),
                is_awaitable=predicate,
            )

        operation = asyncio.create_task(query())
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
        finally:
            operation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await operation
        assert active == 0
        # Closing the queued wrapper must also close the loader it owns; otherwise
        # cancelling a request leaves unawaited coroutines and runtime warnings.
        assert created
        assert all(
            getcoroutinestate(coroutine) == "CORO_CLOSED" for coroutine in created
        )

    asyncio.run(go())


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
