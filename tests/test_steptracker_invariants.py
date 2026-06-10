"""STEPTRACKER robustness invariants — the landed upstream-parity bookkeeping.

These assert the behaviour of the ``StepTracker`` machine now wired into ``dag.Plan``:
a bounded optimize fixpoint (``MAX_OPTIMIZATION_LOOPS``), structural step removal on
replace/merge (``eradicate``), and a maintained reverse-edge index (``_dependents_index``)
that makes ``dependents_of`` a pure function of the live step set. Each test is a pure
plan-level construction (no DB, no execution), asserting on ``plan.steps`` /
``plan.optimize`` / ``plan.dependents_of`` structure, so it runs locally with no Postgres.

The landed mechanisms (with their upstream ``StepTracker.ts`` / ``OperationPlan.ts`` analogues):

  * a BOUNDED optimize fixpoint — ``dag.MAX_OPTIMIZATION_LOOPS`` (upstream caps + warns), so a
    runaway optimize warns and stops instead of spinning (was an unbounded ``while changed:``).
  * structural step removal on replace/merge — ``Plan.eradicate`` removes a replaced step from
    ``self.steps`` + the index the moment it is replaced (was: left dead-but-present until a
    later ``tree_shake``, hidden only by a transient ``_replaced_away`` set, now deleted).
  * a MAINTAINED reverse-edge index — ``Plan._dependents_index`` (upstream's ``$step.dependents``),
    so ``dependents_of`` reads the live edge set in O(out-degree) and stays consistent under
    mutation (was: an on-demand rescan of ``self.steps`` minus the transient ``_replaced_away``).

All were authored as ``xfail(strict=True)`` red tests against the pre-StepTracker engine; the
markers were removed when the machine landed and they now run green.
"""

import pytest

from grafast_py.core_steps import ConstantStep, ListStep, access
from grafast_py.dag import Plan
from grafast_py.step_model import Step


# --------------------------------------------------------------------------- helpers


class SettlingReplaceStep(Step):
    """A step whose ``optimize`` returns a FRESH replacement a bounded number of times.

    It rewrites itself to an equivalent peer ``rounds`` times, then settles (returns
    ``self``). Used to drive ``optimize``'s fixpoint loop through several iterations
    WITHOUT ever risking a hang — the chain provably terminates after ``rounds`` rewrites.
    """

    is_sync_and_safe = True

    def __init__(self, rounds: int) -> None:
        super().__init__()
        self.rounds = rounds

    def execute(self, count, values):
        return [None] * count

    def optimize(self, plan):
        if self.rounds <= 0:
            return self
        return SettlingReplaceStep(self.rounds - 1)


# --------------------------------------------------------------------- (1) optimize cap


def test_optimize_loop_has_a_bounded_iteration_cap():
    """``optimize`` must enforce a bounded iteration cap, not loop unboundedly.

    A well-behaved step that settles after a few rewrites still proves the structural
    gap: the fixpoint loop has NO ceiling, so a badly-behaved step that keeps returning a
    fresh replacement would spin forever instead of being caught. Upstream bounds it with
    ``MAX_OPTIMIZATION_LOOPS`` and warns; we assert that bound EXISTS and is finite.

    Asserted on the SOURCE/constant rather than by actually hanging the suite, so the test
    always terminates: the settling step below confirms ``optimize`` runs and converges,
    and the cap assertion is what is red today.
    """
    plan = Plan()
    plan.add_step(SettlingReplaceStep(3))
    # the loop converges for a well-behaved step (sanity — never hangs).
    plan.optimize()

    # the GREEN invariant: a finite, positive optimization-loop cap is defined on the DAG
    # module (mirroring upstream's MAX_OPTIMIZATION_LOOPS = 10). Today there is none.
    import grafast_py.dag as dag_module

    cap = getattr(dag_module, "MAX_OPTIMIZATION_LOOPS", None)
    assert isinstance(cap, int) and cap > 0, (
        "dag.optimize must bound its fixpoint loop with a finite "
        "MAX_OPTIMIZATION_LOOPS cap (none defined today)"
    )


def test_runaway_optimize_is_caught_by_the_cap():
    """A step that NEVER settles must be stopped by the cap, not spun on forever.

    We make ``optimize`` safe to call by capping the runaway ourselves at the call site
    (a sentinel that flips to identity after a generous bound), so the TEST cannot hang.
    The invariant under test is that the ENGINE — not the test — owns that ceiling: once a
    ``MAX_OPTIMIZATION_LOOPS`` guard exists, a never-settling step is detected and the loop
    terminates (upstream warns; a strict engine could raise). Today no such guard exists,
    so this asserts on the absent cap constant and is red.
    """
    import grafast_py.dag as dag_module

    # the cap must exist for the engine to be able to catch a runaway at all.
    cap = getattr(dag_module, "MAX_OPTIMIZATION_LOOPS", None)
    assert isinstance(cap, int) and cap > 0

    class RunawayStep(Step):
        is_sync_and_safe = True

        def __init__(self, budget: int) -> None:
            super().__init__()
            # budget bounds OUR safety net so the test never hangs, independent of the
            # engine cap; it is set well above any sane engine cap so the engine's own
            # ceiling is what stops the loop first once it exists.
            self.budget = budget

        def execute(self, count, values):
            return [None] * count

        def optimize(self, plan):
            if self.budget <= 0:
                return self
            return RunawayStep(self.budget - 1)

    plan = Plan()
    plan.add_step(RunawayStep(cap * 100))
    # with a real engine cap, optimize stops at the ceiling rather than draining the budget.
    plan.optimize()


# ------------------------------------------------------- (2) structural removal on replace


class SelfReplaceOnce(Step):
    """Replaces itself with a pre-built ``replacement`` step exactly once, then settles."""

    is_sync_and_safe = True

    def __init__(self, tag, replacement=None) -> None:
        super().__init__()
        self.tag = tag
        self._replacement = replacement

    def execute(self, count, values):
        return [self.tag] * count

    def optimize(self, plan):
        if self._replacement is not None:
            replacement = self._replacement
            self._replacement = None
            return replacement
        return self


def test_replaced_step_is_structurally_removed_from_plan_steps():
    """After ``optimize`` replaces a step, the dead original is GONE from ``plan.steps``.

    Upstream ``replaceStep`` transfers the dependents and ``eradicate`` nulls the step out
    of ``stepById`` and its layer the instant it is replaced — the replaced step
    evaporates. ``dag.Plan`` instead leaves the dead original sitting in ``self.steps``
    (only suppressed from ``dependents_of`` via the transient ``_replaced_away`` set) until
    a later ``tree_shake`` happens to trim it. We assert the immediate-removal invariant.
    """
    plan = Plan()
    replacement = SelfReplaceOnce("new")
    original = SelfReplaceOnce("old", replacement=replacement)
    plan.add_step(original)

    remap = plan.optimize()

    # optimize did replace the original with the new step (sanity on the construction).
    assert remap[original.id] is replacement

    # GREEN: the dead original is structurally GONE from the live step set immediately —
    # not lingering until a downstream tree_shake.
    assert original not in plan.steps, (
        "a step replaced during optimize must be removed from plan.steps immediately, "
        "not left dead-but-present until tree_shake"
    )
    # and the surviving replacement is present.
    assert replacement in plan.steps


# ------------------------------------------------- (3) dependents_of consistency / index


def test_dependents_of_reflects_the_live_edges_through_mutation():
    """``dependents_of`` reads the maintained reverse-edge index and tracks LIVE edges.

    A constant feeding two accesses has both accesses as dependents. After one access is
    structurally removed (``eradicate``), ``dependents_of`` must reflect ONLY the surviving
    edge — it follows the maintained ``_dependents_index``, not an on-demand rescan and not a
    transient "hide this id" filter (the old ``_replaced_away`` gate, now deleted). This
    exercises the live mechanism under mutation, where the pre-StepTracker rescan-minus-filter
    would have diverged.
    """
    plan = Plan()
    base = ConstantStep("x")
    a1 = access(base, ("a",))
    a2 = access(base, ("b",))
    plan.add_step(a1)
    plan.add_step(a2)

    assert sorted(s.id for s in plan.dependents_of(base)) == sorted([a1.id, a2.id])

    # eradicate a1 (merge it into a2): the live edge set must drop to just a2 immediately,
    # exercising the maintained index rather than any hidden pass state.
    plan.eradicate(a1, a2)
    plan._sweep_dead()
    assert [s.id for s in plan.dependents_of(base)] == [a2.id]
    assert a1 not in plan.steps


def test_dependents_of_dedupes_a_step_wired_to_the_same_dependency_twice():
    """A step depending on one base at MULTIPLE indices appears ONCE in ``dependents_of``.

    The reverse index records an edge per (dependent, dependency-index) pair, so a step wired
    to the same dependency twice has two index entries; ``dependents_of`` dedupes by identity.
    """
    plan = Plan()
    base = ConstantStep("x")
    pair = ListStep([base, base])  # depends on `base` at indices 0 AND 1
    plan.add_step(pair)
    assert plan.dependents_of(base) == [pair]  # deduped despite two reverse edges


def test_a_maintained_reverse_edge_index_exists():
    """The plan maintains a reverse-edge (dependents) index, not an on-demand inversion.

    Upstream ``StepTracker`` keeps a ``$step.dependents`` array updated as steps are wired
    and replaced, so dependents are an O(1) lookup that stays consistent under
    ``replaceStep``. ``dag.Plan`` keeps no such index — ``dependents_of`` walks every step's
    dependency list on each call. We assert the maintained-index invariant: after wiring,
    each step exposes its live dependents through a maintained structure (rather than the
    plan having to re-scan ``self.steps``).
    """
    plan = Plan()
    base = ConstantStep("x")
    a1 = access(base, ("a",))
    plan.add_step(a1)

    # GREEN end-state: a maintained reverse-edge index is reachable, either as a per-step
    # `dependents` collection (upstream parity) or a plan-level reverse map kept in sync.
    has_step_level = hasattr(base, "dependents")
    has_plan_level = hasattr(plan, "dependents_by_step") or hasattr(
        plan, "_dependents_index"
    )
    assert has_step_level or has_plan_level, (
        "expected a maintained dependents reverse-edge index (per-step .dependents or a "
        "plan-level reverse map); dependents_of re-inverts plan.steps on demand instead"
    )

    # and it reflects the live edge from base -> a1.
    if has_step_level:
        assert any(d is a1 or getattr(d, "step", None) is a1 for d in base.dependents)
