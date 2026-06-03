"""The plan DAG: id assignment, topological ordering, and cross-step dedup.

A :class:`Plan` collects steps as the planner builds them, assigns each a unique
``id``, and provides the two passes the executor needs: a topological ordering (so
every step runs after its dependencies) and a structural *deduplication* pass that
merges identical steps (same class, same dependency winner ids, same ``peer_key`` /
``dedup_params``) so a value computed or loaded twice is computed/loaded once.

``order_steps`` is also usable standalone (e.g. for a transient sub-DAG inside
``EachStep``): it walks the dependency graph reachable from a set of target steps
and returns them in dependency order, assigning fresh ids if unset.
"""

from typing import Dict, List, Sequence, Set

from .step_model import Step


class Plan:
    """A growing collection of steps forming the operation's step DAG.

    ``add_step`` registers a step (and, transitively, any of its already-wired
    dependencies not yet registered) and assigns ids. ``deduplicate`` collapses
    structurally identical steps and returns a remap from old step -> survivor so
    the planner can rewrite the ``FieldPlan.step`` references it holds.
    """

    def __init__(self) -> None:
        self.steps: List[Step] = []
        self._seen: set[int] = set()

    def add_step(self, step: Step) -> Step:
        """Register ``step`` and its transitive dependencies; assign ids; return it."""
        self._register(step)
        return step

    def _register(self, step: Step) -> None:
        if id(step) in self._seen:
            return
        for dep in step.dependencies:
            self._register(dep)
        self._seen.add(id(step))
        step.id = len(self.steps)
        self.steps.append(step)

    def topo_order(self) -> List[Step]:
        """Return all registered steps in dependency order."""
        return order_steps(self.steps)

    def deduplicate(self) -> Dict[int, Step]:
        """Merge structurally identical steps; return ``old id -> survivor`` remap.

        Two steps are structurally identical iff they share class, ``peer_key``, and
        their dependencies (after remap) point at the same survivor ids in the same
        order, and their ``dedup_params`` match. The lowest-id step in a group wins;
        every reference (other steps' ``dependencies`` and the returned remap) is
        rewired to it. Iterated to a fixpoint because merging deps can make their
        dependents newly identical.
        """
        survivors: Dict[int, Step] = {s.id: s for s in self.steps}

        changed = True
        while changed:
            changed = False
            by_key: Dict[tuple, Step] = {}
            for step in order_steps(self.steps):
                if survivors.get(step.id) is not step:
                    continue  # already merged away
                key = _structural_key(step, survivors)
                winner = by_key.get(key)
                if winner is None:
                    by_key[key] = step
                    continue
                # merge `step` into `winner`
                step.deduplicated_with(winner)
                survivors[step.id] = winner
                changed = True

            if changed:
                _rewire_dependencies(self.steps, survivors)

        # collapse transitive survivor chains so callers get a direct mapping
        remap: Dict[int, Step] = {}
        for step in self.steps:
            remap[step.id] = _resolve(survivors, step.id)
        return remap


def order_steps(targets: Sequence[Step]) -> List[Step]:
    """Topologically sort the sub-DAG reachable from ``targets`` (deps first).

    Performs a DFS post-order over ``dependencies``, deduplicating shared nodes by
    object identity, and assigns a fresh contiguous id to any step whose id is still
    unset (``-1``). The returned list is safe to feed to ``run_steps``.
    """
    ordered: List[Step] = []
    visited: set[int] = set()

    def visit(step: Step) -> None:
        if id(step) in visited:
            return
        visited.add(id(step))
        for dep in step.dependencies:
            visit(dep)
        ordered.append(step)

    for target in targets:
        visit(target)

    next_id = 0
    used = {s.id for s in ordered if s.id >= 0}
    for step in ordered:
        if step.id < 0:
            while next_id in used:
                next_id += 1
            step.id = next_id
            used.add(next_id)
    return ordered


def order_steps_within(targets: Sequence[Step], boundary_ids: Set[int]) -> List[Step]:
    """Topologically sort the sub-DAG from ``targets`` down to (but excluding) a boundary.

    Used by the executor to run ONE bucket's plan-resolver steps in isolation: the
    bucket's parents are produced by a step whose id is in ``boundary_ids`` (the
    operation root, or a parent object field's step). That boundary step's output is
    seeded directly, so the walk stops there — it is neither descended into nor
    included — and only the steps strictly *between* the boundary and ``targets``
    (this bucket's own access/load/lambda steps) are returned, deps-first.

    Steps already in this layer keep their plan-time ids; the seeded boundary columns
    are matched by those same ids in :func:`grafast_py.step_model.run_steps`.
    """
    ordered: List[Step] = []
    visited: set[int] = set()

    def visit(step: Step) -> None:
        if id(step) in visited:
            return
        visited.add(id(step))
        if step.id in boundary_ids:
            return  # a seeded source for this bucket: do not descend or include it
        for dep in step.dependencies:
            visit(dep)
        ordered.append(step)

    for target in targets:
        visit(target)
    return ordered


def _structural_key(step: Step, survivors: Dict[int, Step]) -> tuple:
    """A hashable key identifying a step's structural identity for dedup."""
    dep_ids = tuple(_resolve(survivors, dep.id).id for dep in step.dependencies)
    return (type(step), step.peer_key, dep_ids, step.dedup_params())


def _rewire_dependencies(steps: List[Step], survivors: Dict[int, Step]) -> None:
    """Point every step's dependency list at the current survivors."""
    for step in steps:
        if survivors.get(step.id) is not step:
            continue
        step.dependencies = [_resolve(survivors, dep.id) for dep in step.dependencies]


def _resolve(survivors: Dict[int, Step], step_id: int) -> Step:
    """Follow a survivor chain to the final winning step."""
    winner = survivors[step_id]
    while winner.id != step_id and survivors[winner.id] is not winner:
        step_id = winner.id
        winner = survivors[step_id]
    return winner


__all__ = ["Plan", "order_steps", "order_steps_within"]
