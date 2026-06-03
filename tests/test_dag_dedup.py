"""Tests for the plan DAG: topological ordering and cross-step deduplication.

`order_steps` must return steps deps-first; `Plan.deduplicate` must merge
structurally identical steps (same class + deps + params) so a value accessed or
loaded twice is computed once. The decisive case is two LoadOne steps over the same
key against the same loader collapsing to a single step.
"""

from typing import Any, List

from grafast_py.core_steps import AccessStep, ConstantStep, LoadOneStep, RootStep
from grafast_py.dag import Plan, order_steps
from grafast_py.step_model import run_steps


def never_awaitable(_value: Any) -> bool:
    return False


def test_order_steps_returns_dependencies_first():
    root = RootStep()
    a = AccessStep(root, ["id"])
    b = AccessStep(a, ["x"])
    ordered = order_steps([b])
    positions = {id(s): i for i, s in enumerate(ordered)}
    assert positions[id(root)] < positions[id(a)] < positions[id(b)]


def test_dedup_merges_two_identical_access_steps():
    plan = Plan()
    root = RootStep()
    plan.add_step(root)
    a1 = plan.add_step(AccessStep(root, ["id"]))
    a2 = plan.add_step(AccessStep(root, ["id"]))
    assert a1 is not a2

    remap = plan.deduplicate()
    # both access steps collapse to a single survivor
    assert remap[a1.id] is remap[a2.id]


def test_dedup_keeps_distinct_access_paths_separate():
    plan = Plan()
    root = RootStep()
    plan.add_step(root)
    a = plan.add_step(AccessStep(root, ["id"]))
    b = plan.add_step(AccessStep(root, ["name"]))

    remap = plan.deduplicate()
    assert remap[a.id] is not remap[b.id]


def test_dedup_merges_two_loaders_over_the_same_key_and_loader():
    calls = {"n": 0}

    def loader(keys: List[Any]) -> List[Any]:
        calls["n"] += 1
        return [k for k in keys]

    plan = Plan()
    root = RootStep()
    plan.add_step(root)
    key1 = plan.add_step(AccessStep(root, ["id"]))
    key2 = plan.add_step(AccessStep(root, ["id"]))
    load1 = plan.add_step(LoadOneStep(key1, loader))
    load2 = plan.add_step(LoadOneStep(key2, loader))

    remap = plan.deduplicate()
    # the two key accesses merge, so the two loaders (now sharing a dep + loader)
    # also merge — a value loaded twice loads once
    assert remap[load1.id] is remap[load2.id]

    # run the deduped DAG and confirm a single batch call
    survivor = remap[load1.id]
    root.seed([{"id": 7}, {"id": 8}])
    ordered = order_steps([survivor])
    results = run_steps(2, ordered, never_awaitable)
    assert results[survivor.id] == [7, 8]
    assert calls["n"] == 1


def test_dedup_does_not_merge_loaders_with_different_loaders():
    def loader_a(keys):
        return list(keys)

    def loader_b(keys):
        return list(keys)

    plan = Plan()
    root = RootStep()
    plan.add_step(root)
    key = plan.add_step(AccessStep(root, ["id"]))
    la = plan.add_step(LoadOneStep(key, loader_a))
    lb = plan.add_step(LoadOneStep(key, loader_b))

    remap = plan.deduplicate()
    assert remap[la.id] is not remap[lb.id]


def test_dedup_merges_equal_constants():
    plan = Plan()
    c1 = plan.add_step(ConstantStep(5))
    c2 = plan.add_step(ConstantStep(5))
    c3 = plan.add_step(ConstantStep(6))

    remap = plan.deduplicate()
    assert remap[c1.id] is remap[c2.id]
    assert remap[c1.id] is not remap[c3.id]
