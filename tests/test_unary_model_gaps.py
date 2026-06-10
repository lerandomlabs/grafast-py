"""UNARY-MODEL + purity-contract invariants (Option B: plan lambdas are assumed pure).

Under the Grafast PURITY CONTRACT, a plan transform (``lambda_step`` / ``filter_step``) is a
deterministic function of its input. So — like upstream Grafast's unary-value model — such a
step over a request-CONSTANT input is ``_is_unary`` (run ONCE, the result fanned to every child)
and ``hoistable`` (liftable to a shallower layer); over a per-entry input it narrows to batch.

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


def test_pure_plan_transforms_are_unary_and_hoistable_over_a_constant():
    """A lambda / filter over a request-constant input is unary + hoistable (the purity default)."""
    lam_const = lambda_step(constant(5), lambda v: v + 1)
    filt_const = filter_step(constant([1, 2, 3]), lambda x: x > 1)
    # pure-by-default: no barrier, so both are run-once-capable AND liftable.
    assert lam_const._is_unary is True and lam_const.hoistable is True
    assert filt_const._is_unary is True and filt_const.hoistable is True

    # but unariness NARROWS over a per-entry (non-unary) input — the value differs per entry, so
    # the engine treats it as batch (narrowed at the add_dependency chokepoint).
    root = RootStep()  # a batch source (its column IS the bucket of parents)
    lam_per_entry = lambda_step(root, lambda v: v)
    assert lam_per_entry._is_unary is False


def test_resolver_is_the_impure_escape_hatch():
    """Plan lambdas are PURE-by-default; the impure / side-effecting path is a plain resolver.

    A ``ResolveStep`` (the adapter for a plain graphql-core resolver) is the single host-code path
    the engine treats as possibly impure / side-effecting: never merged (``dedupable=False``),
    never hoisted (``hoistable=False``) and never run once (``_is_unary=False``). Plan
    lambdas/filters set none of these — they inherit the pure-by-default class flags.
    """
    # the escape hatch is fully barriered
    assert ResolveStep.dedupable is False
    assert ResolveStep.hoistable is False
    assert ResolveStep._is_unary is False
    # the pure plan transforms are NOT barriered (class defaults: hoistable=True, _is_unary=True)
    assert LambdaStep.hoistable is True and LambdaStep._is_unary is True
    assert FilterStep.hoistable is True and FilterStep._is_unary is True


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
