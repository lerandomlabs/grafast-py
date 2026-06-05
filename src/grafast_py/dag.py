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

    def optimize(self) -> Dict[int, Step]:
        """Run each step's `optimize` hook to a fixpoint; return `old id -> replacement`.

        Walks the steps deps-first (`order_steps`) so a step's dependencies are already
        in their optimized form when its own `optimize` runs, then iterates to a fixpoint
        because one rewrite can enable another (an absorbed dependent can leave its
        dependency newly foldable). When `optimize` returns a replacement, the change is
        recorded and every reference to the old step is rewired to the replacement via the
        SAME survivor-chain machinery `deduplicate` uses (`_rewire_dependencies`/`_resolve`).

        With the shipped default identity `Step.optimize`, the loop runs exactly once, no
        replacement is recorded, no rewire fires, and the returned remap is empty — a
        provable no-op over the finalized plan.
        """
        replaced: Dict[int, Step] = {}  # old id -> replacement step
        changed = True
        while changed:
            changed = False
            for step in order_steps(self.steps):
                if replaced.get(step.id, step) is not step:
                    continue  # already replaced away
                new = step.optimize(self)
                if new is step:
                    continue
                replaced[step.id] = new
                if new.id < 0:  # a freshly built replacement not yet registered
                    self.add_step(new)
                changed = True
            if changed:
                _rewire_dependencies(self.steps, _as_survivors(self.steps, replaced))
        return _collapse_chain(self.steps, replaced)

    def dependents_of(self, step: Step) -> List[Step]:
        """Return every registered step that lists `step` among its dependencies.

        The read accessor the future inlining optimizer uses inside its `optimize`
        hook to find (and absorb) the steps consuming its output. Computed by
        inverting `step.dependencies` over `self.steps` on demand — no eager
        reverse-edge map is maintained, since a pure-substrate wave has no consumer
        of one beyond this lookup.
        """
        return [s for s in self.steps if any(dep is step for dep in s.dependencies)]

    def tree_shake(self, consumption_roots: List[Step]) -> List[Step]:
        """Drop steps unreachable from `consumption_roots` AND not side-effecting.

        `consumption_roots` is the executor's consumption surface for the finalized
        plan — every `FieldPlan.step` plus each `ObjectPlan.parent_step`, across the
        whole (transitively nested) ObjectPlan tree — computed by the caller, which
        holds the ObjectPlan (`dag.py` stays ObjectPlan-free). From those roots, the
        transitively-needed step ids are exactly `order_steps`' reachable set over
        `dependencies`.

        Side-effecting steps (`dedupable=False`, e.g. a pg mutation) run FOR EFFECT and
        may legitimately be unconsumed (a write whose return value is not selected still
        must execute), so they AND their transitive dependencies are force-kept. Ids are
        NOT renumbered: the executor matches seeded boundary columns and `FieldPlan.step`
        by plan-time id, so renumbering would desync the bucket runs. `_seen` is left
        intact (it gates `add_step` idempotence by object identity; trimming does not
        re-add).

        Returns the side-effecting steps that were force-kept *despite* being
        unreachable from any consumption root — the writes an optimizer orphaned by
        inlining their return value. The caller (`finalize_plan`) attaches each to the
        bucket that must RUN it for effect; without that the orphan would sit in the DAG
        but never execute (the executor only runs steps reachable from `FieldPlan.step`),
        silently losing the write. With the default identity `optimize`, every step the
        planner built hangs off a `FieldPlan.step`, so the reachable set is all steps,
        nothing is orphaned, and both the trim and the returned list are empty no-ops.
        """
        reachable = {s.id for s in order_steps(consumption_roots)}
        side_effecting = [s for s in self.steps if not s.dedupable]
        keep = reachable | {s.id for s in order_steps(side_effecting)}
        self.steps = [s for s in self.steps if s.id in keep]
        return [s for s in side_effecting if s.id not in reachable]

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
                if not step.dedupable:
                    continue  # side-effecting (e.g. mutation): never merge with a peer
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


def _as_survivors(steps: List[Step], replaced: Dict[int, Step]) -> Dict[int, Step]:
    """Lift a sparse `old id -> replacement` map into a full survivors map.

    `optimize`'s `replaced` only records the steps that rewrote themselves, but
    `_rewire_dependencies`/`_resolve` require a winner for EVERY id they touch (an
    unreplaced step is its own survivor). This fills the gaps so the same chain-
    following primitives `deduplicate` uses apply unchanged to the optimize remap.
    """
    survivors: Dict[int, Step] = {s.id: s for s in steps}
    survivors.update(replaced)
    return survivors


def _collapse_chain(steps: List[Step], replaced: Dict[int, Step]) -> Dict[int, Step]:
    """Collapse transitive replacement chains into a direct `old id -> final` map.

    One optimize iteration can replace A with B and a later one replace B with C;
    callers want `A -> C` directly. Resolves each replaced id through the survivor
    chain (filling unreplaced gaps via `_as_survivors`) and returns only the entries
    that actually moved.
    """
    survivors = _as_survivors(steps, replaced)
    return {old_id: _resolve(survivors, old_id) for old_id in replaced}


def _compose_remaps(first: Dict[int, Step], second: Dict[int, Step]) -> Dict[int, Step]:
    """Chain two `old id -> survivor` remaps applied in order (`first` then `second`).

    `finalize_plan` runs `optimize` then `deduplicate`; the ObjectPlan must be rewritten
    once to the FINAL survivor. For an id replaced by optimize, follow that replacement
    forward through dedup's remap (by the replacement's id); ids dedup touched but
    optimize did not are carried through directly. Returns a single `old id -> final`
    map so `remap_object_plan` rewrites the tree in one pass.
    """
    composed: Dict[int, Step] = {}
    for old_id, mid in first.items():
        composed[old_id] = second.get(mid.id, mid)
    for old_id, survivor in second.items():
        composed.setdefault(old_id, survivor)
    return composed


__all__ = ["Plan", "order_steps", "order_steps_within"]
