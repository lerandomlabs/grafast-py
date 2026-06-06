"""Tests for the Wave 3a optimizer substrate on the plan DAG.

`Plan.optimize` runs each step's `optimize(plan)` hook to a fixpoint, rewiring every
reference to a replaced step through the same survivor-chain machinery `deduplicate`
uses. `Plan.tree_shake(roots)` trims steps that are neither reachable from the
consumption roots NOR side-effecting. `Plan.dependents_of(step)` inverts the
dependency edges so the future inlining optimizer can find (and absorb) its consumers.

The DEFINING property gated here is the NO-OP SAFETY INVARIANT: with the shipped
default identity `Step.optimize`, optimize + tree-shake leave the finalized plan
byte-identical — the real behaviour change (query inlining) is a separate later wave,
so the substrate is validated here only with a TOY optimizer.
"""

from typing import Any, List

from grafast_py.core_steps import AccessStep, ConstantStep, RootStep
from grafast_py.dag import (
    Plan,
    _as_survivors,
    _collapse_chain,
    _compose_remaps,
    order_steps,
)
from grafast_py.step_model import Step, run_steps


def never_awaitable(_value: Any) -> bool:
    return False


class DoubleStep(Step):
    """A toy 1-dependency step: doubles its dependency's column, entry-wise."""

    def __init__(self, source: Step) -> None:
        super().__init__()
        self.add_dependency(source)

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        return [v * 2 for v in values[0]]


class ConstFoldDouble(DoubleStep):
    """A toy optimizer: a DoubleStep over a ConstantStep folds to a single constant.

    Stands in for the future query-inlining optimizer — it ABSORBS its dependency by
    returning a replacement (a ConstantStep of the doubled value), which orphans the
    folded ConstantStep so tree-shake can drop it.
    """

    def optimize(self, plan: Plan) -> Step:
        source = self.dependencies[0]
        if isinstance(source, ConstantStep):
            return ConstantStep(source.data * 2)
        return self


# ---------------------------------------------------------------------------
# No-op safety invariant: default identity optimize() changes nothing.
# ---------------------------------------------------------------------------


def test_optimize_default_identity_is_a_no_op():
    plan = Plan()
    root = RootStep()
    plan.add_step(root)
    a = plan.add_step(AccessStep(root, ["id"]))
    plan.add_step(DoubleStep(a))

    before = list(plan.steps)
    remap = plan.optimize()

    assert remap == {}  # nothing rewrote itself
    assert plan.steps == before  # same objects, same order, byte-identical


def test_tree_shake_keeps_every_consumed_step_default():
    plan = Plan()
    root = RootStep()
    plan.add_step(root)
    a = plan.add_step(AccessStep(root, ["id"]))
    doubled = plan.add_step(DoubleStep(a))

    before = list(plan.steps)
    # every planner step hangs off a consumed FieldPlan.step (here `doubled`):
    plan.tree_shake([doubled])
    assert plan.steps == before


# ---------------------------------------------------------------------------
# dependents_of: the read accessor the inlining optimizer uses.
# ---------------------------------------------------------------------------


def test_dependents_of_inverts_dependency_edges():
    plan = Plan()
    root = plan.add_step(RootStep())
    a = plan.add_step(AccessStep(root, ["id"]))
    b = plan.add_step(AccessStep(root, ["name"]))
    da = plan.add_step(DoubleStep(a))

    deps = plan.dependents_of(root)
    assert set(map(id, deps)) == {id(a), id(b)}
    assert plan.dependents_of(a) == [da]
    assert plan.dependents_of(da) == []


# ---------------------------------------------------------------------------
# A TOY optimizer actually rewriting: replacement is wired + the fold is correct.
# ---------------------------------------------------------------------------


def test_optimize_replaces_step_and_rewires_dependents():
    plan = Plan()
    const = plan.add_step(ConstantStep(5))
    folded = plan.add_step(ConstFoldDouble(const))
    consumer = plan.add_step(DoubleStep(folded))

    remap = plan.optimize()

    # the foldable step rewrote itself to a ConstantStep(10)
    survivor = remap[folded.id]
    assert isinstance(survivor, ConstantStep)
    assert survivor.data == 10
    # the consumer's dependency edge was rewired to the replacement, not the old step
    assert consumer.dependencies[0] is survivor

    # and it executes to the folded value end-to-end
    ordered = order_steps([consumer])
    results = run_steps(1, ordered, never_awaitable)
    assert results[consumer.id] == [20]


def test_optimize_fixpoint_chains_rewrites():
    """One fold enabling another: ConstFoldDouble over a folded constant folds again."""
    plan = Plan()
    const = plan.add_step(ConstantStep(3))
    inner = plan.add_step(ConstFoldDouble(const))  # -> ConstantStep(6)
    outer = plan.add_step(ConstFoldDouble(inner))  # only foldable AFTER inner folds

    remap = plan.optimize()

    outer_survivor = remap[outer.id]
    assert isinstance(outer_survivor, ConstantStep)
    assert outer_survivor.data == 12  # 3 -> 6 -> 12, reached at the fixpoint


# ---------------------------------------------------------------------------
# tree_shake: drops orphaned non-side-effecting steps; NEVER side-effecting ones.
# ---------------------------------------------------------------------------


def test_tree_shake_drops_orphaned_step_after_optimize():
    plan = Plan()
    const = plan.add_step(ConstantStep(5))
    folded = plan.add_step(ConstFoldDouble(const))

    remap = plan.optimize()
    survivor = remap[folded.id]
    # after folding, the original ConstantStep is orphaned (the survivor is a fresh
    # constant with no dependency on it); it must be shaken out.
    plan.tree_shake([survivor])

    surviving_ids = {s.id for s in plan.steps}
    assert survivor.id in surviving_ids
    assert const.id not in surviving_ids
    assert folded.id not in surviving_ids


class WriteStep(Step):
    """A toy side-effecting step (mutation marker): runs for effect, never merged.

    `execute` records each run in `writes` so the effect is observable only when the step
    actually runs — a structural keep that the executor never runs would leave `writes`
    empty. `passthrough` carries the source column so the step is also correct end-to-end.
    """

    dedupable = False

    def __init__(self, source: Step, writes: List[Any]) -> None:
        super().__init__()
        self.writes = writes
        self.add_dependency(source)

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        self.writes.append(list(values[0]))  # the WRITE
        return list(values[0])


def test_tree_shake_never_removes_side_effecting_step_even_if_unconsumed():
    writes: List[Any] = []
    plan = Plan()
    root = plan.add_step(RootStep())
    value_source = plan.add_step(AccessStep(root, ["id"]))
    write = plan.add_step(WriteStep(value_source, writes))
    consumed = plan.add_step(ConstantStep(1))

    # the write's value is NOT among the consumption roots — only `consumed` is.
    orphaned = plan.tree_shake([consumed])

    surviving_ids = {s.id for s in plan.steps}
    # the side-effecting write is force-kept, AND its value-source dependency with it.
    assert write.id in surviving_ids
    assert value_source.id in surviving_ids
    assert root.id in surviving_ids
    assert consumed.id in surviving_ids
    # tree-shake REPORTS the orphaned side-effecting step so finalize can give it a run
    # target — a structural keep the executor ignores would silently drop the write.
    assert orphaned == [write]


def test_tree_shake_force_kept_write_actually_runs_for_effect():
    """BEHAVIOURAL gate: the force-kept orphan executes, and reverting that drops the write.

    Structural survival in `plan.steps` is vacuous unless something RUNS the kept step. The
    executor runs a bucket's reachable steps PLUS the orphaned side-effecting steps tree-shake
    reports (the `effect_steps` wiring). Here we reproduce that at the DAG level: running the
    consumed roots AND the reported orphan records the write; running ONLY the consumed roots
    (reverting the force-keep) leaves the write empty — proving the keep is load-bearing.
    """
    writes: List[Any] = []
    plan = Plan()
    root = plan.add_step(RootStep())
    root.seed([{"id": 7}])
    value_source = plan.add_step(AccessStep(root, ["id"]))
    plan.add_step(WriteStep(value_source, writes))
    consumed = plan.add_step(ConstantStep(1))

    orphaned = plan.tree_shake([consumed])

    # reverting the force-keep: run ONLY the consumed roots — the orphan never runs.
    run_steps(1, order_steps([consumed]), never_awaitable)
    assert writes == [], "the write ran without any run target — test is not isolating it"

    # the fix: run the consumed roots PLUS the reported orphan (what finalize attaches as
    # the bucket's effect step) — now the write executes for effect.
    run_steps(1, order_steps([consumed, *orphaned]), never_awaitable)
    assert writes == [[7]], (
        "the force-kept orphaned write did not run for effect — its write was lost"
    )


def test_tree_shake_does_not_renumber_ids():
    plan = Plan()
    root = plan.add_step(RootStep())
    a = plan.add_step(AccessStep(root, ["id"]))
    keep = plan.add_step(DoubleStep(a))
    orphan = plan.add_step(ConstantStep(99))  # unconsumed, not side-effecting

    ids_before = {id(s): s.id for s in (root, a, keep)}
    plan.tree_shake([keep])

    surviving_ids = {s.id for s in plan.steps}
    assert orphan.id not in surviving_ids
    # the kept steps retain their original plan-time ids (no renumber → executor stays in sync)
    for step, old_id in ids_before.items():
        survivor = next(s for s in plan.steps if id(s) == step)
        assert survivor.id == old_id


# ---------------------------------------------------------------------------
# Helper unit tests: survivor lifting, chain collapse, remap composition.
# ---------------------------------------------------------------------------


def test_as_survivors_fills_unreplaced_ids_with_identity():
    plan = Plan()
    s0 = plan.add_step(ConstantStep(1))
    s1 = plan.add_step(ConstantStep(2))
    replacement = ConstantStep(99)
    replacement.id = 7

    survivors = _as_survivors(plan.steps, {s0.id: replacement})
    assert survivors[s0.id] is replacement  # replaced id maps to its replacement
    assert survivors[s1.id] is s1  # unreplaced id is its own survivor


def test_collapse_chain_resolves_transitive_replacements():
    plan = Plan()
    a = plan.add_step(ConstantStep(1))  # id 0
    b = plan.add_step(ConstantStep(2))  # id 1
    c = plan.add_step(ConstantStep(3))  # id 2
    # a -> b, b -> c : collapsing must give a -> c and b -> c directly.
    collapsed = _collapse_chain(plan.steps, {a.id: b, b.id: c})
    assert collapsed[a.id] is c
    assert collapsed[b.id] is c


def test_compose_remaps_chains_optimize_then_dedup():
    plan = Plan()
    a = plan.add_step(ConstantStep(1))  # optimize replaces a -> mid
    mid = plan.add_step(ConstantStep(2))  # dedup then merges mid -> final
    final = plan.add_step(ConstantStep(3))

    opt_remap = {a.id: mid}
    dedup_remap = {mid.id: final}
    composed = _compose_remaps(opt_remap, dedup_remap)
    # a's optimize replacement is followed forward through dedup to the final survivor
    assert composed[a.id] is final
    # an id only dedup touched is carried through unchanged
    assert composed[mid.id] is final


# ---------------------------------------------------------------------------
# Optimize + dedup interplay: optimize-produced duplicates get merged by dedup.
# ---------------------------------------------------------------------------


def test_optimize_then_deduplicate_merges_folded_duplicates():
    plan = Plan()
    c1 = plan.add_step(ConstantStep(5))
    c2 = plan.add_step(ConstantStep(5))
    f1 = plan.add_step(ConstFoldDouble(c1))  # -> ConstantStep(10)
    f2 = plan.add_step(ConstFoldDouble(c2))  # -> ConstantStep(10), structurally identical

    opt_remap = plan.optimize()
    dedup_remap = plan.deduplicate()
    remap = _compose_remaps(opt_remap, dedup_remap)
    # the two independently folded constants are structurally identical post-rewrite,
    # so dedup collapses them to one survivor — exactly the order finalize_plan uses.
    assert remap[f1.id] is remap[f2.id]


def test_optimize_does_not_fold_double_over_non_constant():
    """The toy optimizer only folds over ConstantStep; over an access it is identity."""
    plan = Plan()
    root = plan.add_step(RootStep())
    a = plan.add_step(AccessStep(root, ["n"]))
    d = plan.add_step(ConstFoldDouble(a))

    remap = plan.optimize()
    assert remap == {}  # nothing folded
    assert plan.steps[-1] is d
