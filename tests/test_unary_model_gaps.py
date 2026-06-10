"""UNARY-MODEL + purity-contract invariants (Option B: plan lambdas are assumed pure).

Under the Grafast PURITY CONTRACT, a plan transform (``lambda_step`` / ``filter_step``) is a
deterministic function of its input, so over a request-CONSTANT input it is value-constant
(``_is_unary``); over a per-entry input it narrows to batch. The hoist / run-once OPTIMIZATION,
however, applies only to provably-SYNC steps (``is_sync_and_safe``) — a ``filter_step`` is sync,
so it is hoisted and run once; a ``lambda_step`` is async-capable (``fn`` may be a coroutine
function), so it is NEVER hoisted or run once (fanning one coroutine across rows would alias a
single-await awaitable, which the @stream path would then double-await). A lambda therefore keeps
the purity contract (dedup, the resolver escape hatch) but runs per entry.

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


def test_pure_plan_transforms_unary_flag_and_optimization_eligibility():
    """A lambda / filter over a constant is value-constant (``_is_unary``); only the provably-SYNC
    filter is eligible for the hoist / run-once optimization. A lambda is async-capable, so it is
    never hoisted/run-once (fanning one coroutine across rows would alias a single-await awaitable)
    — it keeps the purity contract (dedup) but not the optimization."""
    lam_const = lambda_step(constant(5), lambda v: v + 1)
    filt_const = filter_step(constant([1, 2, 3]), lambda x: x > 1)
    # both are value-constant over a constant input (the unary flag — the purity contract)
    assert lam_const._is_unary is True
    assert filt_const._is_unary is True
    # but only the SYNC filter is optimization-eligible; the async-capable lambda is not hoistable
    assert filt_const.hoistable is True
    assert lam_const.hoistable is False

    # unariness NARROWS over a per-entry (non-unary) input — the value differs per entry.
    root = RootStep()  # a batch source (its column IS the bucket of parents)
    assert lambda_step(root, lambda v: v)._is_unary is False


def test_step_classification_pure_sync_pure_async_and_the_impure_escape_hatch():
    """Three host-code categories, by their merge / hoist / unary flags:

    * pure + SYNC (``FilterStep``): dedupable + hoistable + _is_unary — fully optimizable.
    * pure + ASYNC-capable (``LambdaStep``): dedupable + _is_unary (the purity contract) but NOT
      hoistable — it may emit raw per-entry coroutines, so it is never hoisted/run-once (fan-out
      would alias a single-await awaitable).
    * impure / side-effecting (``ResolveStep``, a plain resolver): the escape hatch — never merged,
      hoisted, or run once.
    """
    # the impure escape hatch is fully barriered
    assert ResolveStep.dedupable is False
    assert ResolveStep.hoistable is False
    assert ResolveStep._is_unary is False
    # a pure SYNC transform is fully optimizable
    assert FilterStep.dedupable is True
    assert FilterStep.hoistable is True and FilterStep._is_unary is True
    # a pure ASYNC-capable transform keeps the purity contract (dedup + unary flag) but is not
    # hoisted/run-once (it may emit per-entry awaitables that fan-out would alias)
    assert LambdaStep.dedupable is True
    assert LambdaStep._is_unary is True
    assert LambdaStep.hoistable is False


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
