"""UNARY-MODEL + purity-contract invariants (Option B: plan lambdas are assumed pure).

Under the Grafast PURITY CONTRACT, a plan transform (``lambda_step`` / ``filter_step``) is a
deterministic function of its input, so over a request-CONSTANT input it is value-constant
(``_is_unary``); over a per-entry input it narrows to batch. SHARE-optimization (hoist / run-once,
which compute the value once and copy it to every row) needs BOTH purity AND a CONCRETE value —
a coroutine is single-await and cannot be copied. ``filter_step`` is sync, so it is optimized; a
``lambda_step`` is decided PER INSTANCE — ``LambdaStep`` inspects ``fn`` and a SYNC lambda is
optimized while an ASYNC one runs per entry (its result is a per-row coroutine). For async I/O use
a LOAD step, which batches AND is share-safe. Impure work uses a plain resolver (the escape hatch).

The flags are the explicit IMPURE / side-effecting OPT-OUT: a plain resolver (``ResolveStep``)
sets ``dedupable=False`` / ``hoistable=False`` / ``_is_unary=False`` so an impure per-entry
resolver is never merged, hoisted, or run once. If you need impure / side-effecting per-entry
work, use a plain resolver, NOT a plan lambda.

These assert the flag semantics (a pure transform over a constant is unary+hoistable; over a
per-entry input it narrows; the resolver escape hatch stays barriered) plus a byte-identity
oracle (a pure filter's result is identical whether or not it is optimized). No DB required.

(History: authored as red xfail tests against the pre-UnaryModel engine, then rewritten when the
UnaryModel landed and again when the purity contract was adopted — plan lambdas/filters are now
optimizable rather than barriered, matching upstream.)
"""

import asyncio

import pytest

from grafast_py.config import GrafastConfig
from grafast_py.core_steps import (
    FilterStep,
    LambdaStep,
    RootStep,
    access,
    constant,
    filter_step,
    lambda_step,
    load_one,
)
from grafast_py.entry import grafast_execute
from grafast_py.schema import make_grafast_schema
from grafast_py.steps import ResolveStep


# A two-level list nesting (orgs -> people) so a request-constant transform lives under a
# per-item child bucket — the shape where run-once / hoisting is observable.
TWO_LEVEL_SDL = """
type Query { orgs: [Org!]! }
type Org { id: Int! people: [Person!]! }
type Person { id: Int! picks: [Int!]! }
"""


def leaf(key):
    """A plan resolver that reads ``key`` off the per-child boundary (the parent row)."""

    def plan(parent_step, args, info):
        return access(parent_step, (key,))

    return plan


def test_share_eligibility_is_per_fn_sync_ness():
    """Share-eligibility (hoist / run-once) needs BOTH purity AND sync-ness, and ``LambdaStep``
    decides sync-ness PER INSTANCE by inspecting ``fn``.

    A value can only be shared (computed once, copied to every row) if it is a CONCRETE result —
    a coroutine is single-await and cannot be copied. So a SYNC ``lambda_step`` is value-constant
    AND share-eligible (``is_sync_and_safe`` + ``hoistable``); an ASYNC ``lambda_step`` is still
    value-constant (the unary flag — the purity half) but NOT share-eligible (it would alias a
    coroutine across rows). A sync ``filter_step`` is also share-eligible.
    """
    sync_lam = lambda_step(constant(5), lambda v: v + 1)
    async_lam = lambda_step(constant(5), _an_async_fn)
    filt = filter_step(constant([1, 2, 3]), lambda x: x > 1)

    # all are value-constant over a constant input (the unary flag — the purity half)
    assert sync_lam._is_unary is True
    assert async_lam._is_unary is True
    assert filt._is_unary is True

    # but only the SYNC ones are share-eligible (concrete value); the async lambda is not
    assert sync_lam.is_sync_and_safe is True and sync_lam.hoistable is True
    assert async_lam.is_sync_and_safe is False and async_lam.hoistable is False
    assert filt.is_sync_and_safe is True and filt.hoistable is True

    # unariness NARROWS over a per-entry (non-unary) input — the value differs per entry.
    root = RootStep()  # a batch source (its column IS the bucket of parents)
    assert lambda_step(root, lambda v: v)._is_unary is False


async def _an_async_fn(v):
    return v


def test_step_classification_pure_sync_pure_async_and_the_impure_escape_hatch():
    """Three host-code categories, by their merge / hoist / unary flags:

    * pure + SYNC (a sync ``lambda_step`` / ``FilterStep``): dedupable + hoistable + _is_unary —
      fully share-optimizable.
    * pure + ASYNC (an async ``lambda_step``): dedupable + _is_unary (the purity half) but NOT
      hoistable / is_sync_and_safe — its result is a per-row coroutine that cannot be shared, so it
      runs per entry (decided PER INSTANCE).
    * impure / side-effecting (``ResolveStep``, a plain resolver): the escape hatch — never merged,
      hoisted, or run once.
    """
    # the impure escape hatch is fully barriered (class-level — every resolver is the same)
    assert ResolveStep.dedupable is False
    assert ResolveStep.hoistable is False
    assert ResolveStep._is_unary is False
    # a pure SYNC transform is fully share-optimizable
    assert FilterStep.dedupable is True
    assert FilterStep.hoistable is True and FilterStep._is_unary is True
    # a lambda is dedupable + value-constant by contract, but share-eligibility is PER INSTANCE:
    assert LambdaStep.dedupable is True
    sync_lam = lambda_step(constant(0), lambda v: v)
    async_lam = lambda_step(constant(0), _an_async_fn)
    assert sync_lam.is_sync_and_safe is True and sync_lam.hoistable is True
    assert async_lam.is_sync_and_safe is False and async_lam.hoistable is False


def test_pure_filter_over_constant_is_byte_identical_when_optimized():
    """A PURE filter over a request-constant list yields identical data hoist ON vs OFF.

    The filter is now run-once-capable (Option B), so under hoist it may fire once and fan the
    result; because the predicate is PURE, the data is byte-identical to the per-entry layout —
    the suite-wide hoist byte-identity oracle, now extended to a host predicate.
    """
    orgs = [{"id": 10}, {"id": 20}]
    people = [{"id": 1}, {"id": 2}]

    def run(hoist):
        def plan_orgs(p, a, i):
            return constant(orgs)

        def plan_people(p, a, i):
            return load_one(constant("p"), lambda keys: [people for _ in keys])

        def plan_picks(p, a, i):
            return filter_step(constant([1, 2, 3, 4]), lambda x: x % 2 == 0)  # PURE

        schema = make_grafast_schema(
            TWO_LEVEL_SDL,
            {
                "Query": {"orgs": plan_orgs},
                "Org": {"id": leaf("id"), "people": plan_people},
                "Person": {"id": leaf("id"), "picks": plan_picks},
            },
        )
        result = grafast_execute(
            schema,
            "{ orgs { id people { id picks } } }",
            config=GrafastConfig(hoist=hoist),
        )
        assert result.errors is None, result.errors
        return [[pp["picks"] for pp in o["people"]] for o in result.data["orgs"]]

    off = run(False)
    on = run(True)
    assert off == [[[2, 4], [2, 4]], [[2, 4], [2, 4]]]
    assert on == off


@pytest.mark.asyncio
async def test_unary_async_lambda_runs_per_entry_not_broadcast():
    """A unary ASYNC lambda must NOT be run-once: broadcasting one coroutine across parents would
    alias a single awaitable (a coroutine is single-await; the @stream path, which completes each
    parent separately, would then raise "cannot reuse already awaited coroutine"). The engine only
    run-once-broadcasts provably-SYNC steps (``is_sync_and_safe``), so an async ``lambda_step`` over
    a constant runs per entry — distinct coroutines — with correct data. Regression for the
    cross-parent coroutine-reuse hazard.
    """
    calls = {"n": 0}

    async def async_tag(_):
        calls["n"] += 1
        await asyncio.sleep(0)
        return 7

    schema = make_grafast_schema(
        "type Query { items: [Item!]! }\ntype Item { tag: Int! }",
        {
            "Query": {"items": lambda p, a, i: constant([{}, {}, {}])},  # 3 parents
            "Item": {"tag": lambda p, a, i: lambda_step(constant("k"), async_tag)},
        },
    )
    result = await grafast_execute(
        schema, "{ items { tag } }", config=GrafastConfig(hoist=False)
    )
    assert result.errors is None, result.errors
    assert result.data == {"items": [{"tag": 7}, {"tag": 7}, {"tag": 7}]}
    # per entry: 3 DISTINCT coroutines, not one broadcast (which would be 1 call + an aliased coro).
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_async_callable_object_is_detected_and_not_shared():
    """A callable OBJECT whose ``__call__`` is async is detected (via ``__call__``) and runs per
    entry — not share-optimized into one coroutine (which would trip the share-point guard).

    Regression for the ``asyncio.iscoroutinefunction(instance) == False`` blind spot: the instance
    is not itself a coroutine function (only its bound ``__call__`` is), so ``LambdaStep`` inspects
    ``__call__`` too.
    """

    class AsyncTag:
        def __init__(self):
            self.calls = 0

        async def __call__(self, _):
            self.calls += 1
            await asyncio.sleep(0)
            return 7

    class SyncTag:
        def __call__(self, _):
            return 9

    # detected async via __call__ -> NOT share-eligible; a sync callable object stays eligible
    assert lambda_step(constant("k"), AsyncTag()).is_sync_and_safe is False
    assert lambda_step(constant("k"), AsyncTag()).hoistable is False
    assert lambda_step(constant("k"), SyncTag()).is_sync_and_safe is True

    # and it executes correctly over multiple parents (per entry, distinct coroutines, no crash)
    tagger = AsyncTag()
    schema = make_grafast_schema(
        "type Query { items: [Item!]! }\ntype Item { tag: Int! }",
        {
            "Query": {"items": lambda p, a, i: constant([{}, {}, {}])},
            "Item": {"tag": lambda p, a, i: lambda_step(constant("k"), tagger)},
        },
    )
    result = await grafast_execute(
        schema, "{ items { tag } }", config=GrafastConfig(hoist=True)
    )
    assert result.errors is None, result.errors
    assert result.data == {"items": [{"tag": 7}, {"tag": 7}, {"tag": 7}]}
    assert tagger.calls == 3  # per entry, not aliased into one shared coroutine
