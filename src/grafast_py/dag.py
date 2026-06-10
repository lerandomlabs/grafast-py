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

from typing import Dict, List, Sequence, Set, Tuple

from .config import log
from .step_model import Step

# Upstream ``OperationPlan.optimizeSteps`` bounds its fixpoint with ``MAX_OPTIMIZATION_LOOPS``
# (10) and warns on overrun rather than spinning. grafast-py's ``optimize`` is a per-step
# fixpoint that can need more iterations on deep inline chains, so the cap is set generously
# above any real convergence depth; exceeding it means a step's ``optimize`` never settles
# (a bug), and we warn + stop with the plan in its last consistent state rather than hang.
MAX_OPTIMIZATION_LOOPS = 1000


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
        # plan-level inlining decision (one operation = one decision): the
        # `GrafastConfig.inline_relations` flag, stashed here by `plan_operation` /
        # `abstract_child_plan` so a pg step's `optimize(self, plan)` can read it
        # without plumbing the whole execution context. Default `False` so a `Plan`
        # built without a config (e.g. unit tests) never inlines — the no-op
        # invariant holds: every step's `optimize` short-circuits to identity.
        self.inline_relations: bool = False
        # plan-level placeholder/caching decisions (one operation = one decision each),
        # the `GrafastConfig.placeholders` / `cache_plans` flags stashed here by
        # `plan_operation` (mirroring `inline_relations`) so the planner reads one
        # constant instead of plumbing the whole execution context. `placeholders` gates
        # whether the planner computes per-argument variable provenance and threads it
        # into `FieldArgs`; `cache_plans` gates the cross-request plan cache. Both default
        # `False` so a `Plan` built without a config (e.g. unit tests) computes no
        # provenance and caches nothing — both features stay dark.
        self.placeholders: bool = False
        self.cache_plans: bool = False
        # whether this finalized plan is VALUE-INDEPENDENT and so safe to cache across
        # requests of the same document. Defaults True; the planner flips it False when a
        # GraphQL `$variable` value was INLINED as a plan-time literal (the value is baked
        # into the SQL text, so the plan is value-specific and reusing it across requests
        # would serve the wrong value). A value-agnostic plan — every SQL-affecting variable
        # value is either a same-every-request literal or a source-tagged placeholder — stays
        # True and may be cached. Only consulted when `cache_plans` is on (default-off => the
        # cache is never read/written, so the flag is inert and the planning path is unaffected).
        self.cacheable: bool = True
        # plan-level hoisting decision (one operation = one decision): the
        # `GrafastConfig.hoist` flag, stashed here by `plan_operation` /
        # `abstract_child_plan` so `finalize_plan` can run the cross-parent hoist pass
        # without plumbing the whole execution context. Default `False` so a `Plan`
        # built without a config (e.g. unit tests) never hoists — the finalize pass is a
        # byte-identical no-op (nothing is lifted, `populate_layers` sees empty
        # `hoisted_in`/`hoisted_out_ids` and runs exactly as before).
        self.hoist: bool = False
        # whether this plan was built for a MUTATION operation. Hoisting is disabled under
        # mutations (the mutation root runs its fields serially and must not be reordered),
        # so `finalize_plan` gates the hoist pass on `not is_mutation`. Set by
        # `plan_operation` from the operation type; left `False` for abstract concrete-type
        # subtrees (which finalize against their own ROOT plan and are reached only from a
        # query/event walk).
        self.is_mutation: bool = False
        # whether this plan partitions @defer'd groups and reads @stream markers during
        # the walk. Only set True (by `plan_operation` / `abstract_child_plan`) on graphql-core
        # 3.3 when the operation carries the incremental directives. False (the default, and
        # always on 3.2) => the legacy collection seam runs and the plan is byte-identical.
        self.incremental: bool = False
        # SIDE replacements an `optimize` hook records beyond the one step it returns.
        # A `Step.optimize` rewrites only ITSELF (its return value), but a DEPENDENT-
        # absorbing optimizer (the LATERAL inliner) must ALSO rewrite the children it
        # folded — each absorbed child relation step becomes a `NestedExtractStep` reading
        # the parent's nested column. The hook pushes `(old_child, replacement)` here via
        # `record_replacement`; `optimize` drains the buffer into its `replaced` map after
        # each hook runs, so the SAME survivor-chain rewire repoints every reference to the
        # folded child (the child bucket's parent_step + the AccessSteps reading its rows).
        # Empty for the default identity optimize, so it is a no-op there.
        self._optimize_side_replacements: List[tuple[Step, Step]] = []
        # the MAINTAINED reverse-edge index (upstream ``StepTracker``'s ``$step.dependents``):
        # ``dependency step id -> list of (dependent step, dependency index)``. Kept in sync at
        # registration (`_register`) and at every structural rewrite (`eradicate`), so
        # `dependents_of` is an O(out-degree) lookup that is a pure function of the LIVE step
        # set — never an on-demand rescan of `self.steps`, and never filtered by transient
        # pass state. A step REPLACED/MERGED away is `eradicate`d: its reverse edges move to the
        # survivor and it leaves this index immediately, so a dependent-absorbing optimizer (the
        # LATERAL inliner) can never re-fold an already-folded child — structural removal, not a
        # hidden "already replaced" filter, prevents the runaway.
        self._dependents_index: Dict[int, List[Tuple[Step, int]]] = {}
        # ids eradicated during the running pass, trimmed from `self.steps` once at pass end
        # (the index is updated immediately; the `self.steps` list is filtered in one sweep to
        # keep eradication O(1) amortised rather than O(n) per removal). Empty between passes.
        self._dead_ids: Set[int] = set()

    def record_replacement(self, old: Step, new: Step) -> None:
        """Record a SIDE replacement (`old` -> `new`) for the running optimize pass.

        Used by a dependent-absorbing `optimize` hook to rewrite steps OTHER than the one
        it returns: the LATERAL inliner's parent `optimize` returns its replacement parent
        but ALSO folds child relation steps, recording each `child -> NestedExtractStep`
        here so `optimize` rewires every reference to the folded child to the extract step.
        `new` must already be registered (`add_step`) so it carries an id before the rewire.
        """
        self._optimize_side_replacements.append((old, new))

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
        # id = current length: collision-free because, in the finalize flow, every `add_step`
        # happens BEFORE the pass's single trailing `_sweep_dead` (which shrinks `self.steps`).
        # A future pass that registered a step AFTER a sweep would need a monotonic counter.
        step.id = len(self.steps)
        self.steps.append(step)
        # record this step's forward edges in the reverse index (deps are registered first,
        # so each already carries an id). Keeps the index a faithful inverse of the live DAG.
        for idx, dep in enumerate(step.dependencies):
            self._dependents_index.setdefault(dep.id, []).append((step, idx))

    def topo_order(self) -> List[Step]:
        """Return all registered steps in dependency order."""
        return order_steps(self.steps)

    def optimize(self) -> Dict[int, Step]:
        """Run each step's `optimize` hook to a fixpoint; return `old id -> replacement`.

        Walks the steps deps-first (`order_steps`) so a step's dependencies are already
        in their optimized form when its own `optimize` runs, then iterates to a fixpoint
        because one rewrite can enable another (an absorbed dependent can leave its
        dependency newly foldable). When `optimize` returns a replacement, the old step is
        `eradicate`d — its dependents repointed onto the replacement and the step removed
        from the live set and reverse index — the SAME structural-removal primitive
        `deduplicate` uses.

        A hook may ALSO record SIDE replacements via `record_replacement` (the LATERAL
        inliner folds child relation steps into the parent it returns, rewriting each
        `child -> NestedExtractStep`); those are drained into `replaced` right after the
        hook runs, so the same rewire repoints every reference to a folded child.

        With the default identity `Step.optimize`, the loop runs exactly once, no
        replacement is recorded, nothing is eradicated, and the returned remap is empty — a
        provable no-op over the finalized plan.

        The fixpoint is BOUNDED by `MAX_OPTIMIZATION_LOOPS` (upstream parity): a step whose
        `optimize` never settles is a bug, so we warn and stop with the plan in its last
        consistent state rather than spin forever.
        """
        merged: Dict[int, Step] = {}  # old id -> immediate replacement
        changed = True
        loops = 0
        while changed:
            changed = False
            loops += 1
            if loops > MAX_OPTIMIZATION_LOOPS:
                log.warning(
                    "optimize did not converge; stopping at cap",
                    cap=MAX_OPTIMIZATION_LOOPS,
                    steps=len(self.steps),
                )
                break
            for step in order_steps(self.steps):
                if step.id in self._dead_ids:
                    continue  # eradicated earlier this pass
                new = step.optimize(self)
                # drain SIDE replacements the hook recorded (the inliner's folded children)
                # BEFORE handling its return value, so a parent that returns a replacement
                # AND folds children eradicates both in one pass. Each replacement is already
                # registered by the inliner (record_replacement requires it).
                if self._optimize_side_replacements:
                    for old_child, replacement in self._optimize_side_replacements:
                        if replacement.id < 0:
                            self.add_step(replacement)
                        self.eradicate(old_child, replacement)
                        merged[old_child.id] = replacement
                    self._optimize_side_replacements = []
                    changed = True
                if new is step:
                    continue
                if new.id < 0:  # a freshly built replacement not yet registered
                    self.add_step(new)
                self.eradicate(step, new)
                merged[step.id] = new
                changed = True
        self._sweep_dead()
        return _resolve_merged(merged)

    def dependents_of(self, step: Step) -> List[Step]:
        """Return every LIVE registered step that lists `step` among its dependencies.

        Reads the maintained reverse-edge index (`_dependents_index`) — an O(out-degree)
        lookup that is a pure function of the live step set, deduped by identity (a step
        depending on `step` at several indices appears once). An `eradicate`d step is absent
        from the index, so it is never surfaced; there is no transient-pass-state filter.
        This is the read accessor the LATERAL inlining optimizer uses inside its `optimize`
        hook to find (and absorb) the steps consuming its output.
        """
        out: List[Step] = []
        seen: Set[int] = set()
        for dependent, _idx in self._dependents_index.get(step.id, []):
            if id(dependent) not in seen:
                seen.add(id(dependent))
                out.append(dependent)
        return out

    def eradicate(self, dead: Step, survivor: Step) -> None:
        """Structurally remove `dead`, transferring its dependents to `survivor`.

        The single rewrite primitive `deduplicate` and `optimize` share (upstream's
        ``StepTracker.replaceStep`` + ``eradicate``): every step that depended on `dead` is
        repointed to `survivor` (its `dependencies[idx]` and the reverse edge both move),
        `dead`'s own forward edges leave its dependencies' reverse lists, and `dead` leaves the
        index immediately. `dead` is marked for removal from `self.steps` (trimmed once at pass
        end via `_sweep_dead`, keeping eradication O(1) amortised). After this, no live step
        references `dead` and `dependents_of` can never surface it — so a dependent-absorbing
        optimizer cannot re-fold an already-folded child.
        """
        if dead is survivor:
            return
        moved = self._dependents_index.pop(dead.id, [])
        survivor_edges = self._dependents_index.setdefault(survivor.id, [])
        for dependent, idx in moved:
            dependent.dependencies[idx] = survivor
            survivor_edges.append((dependent, idx))
            # a rewire onto a non-unary survivor lowers the dependent (monotone narrowing —
            # a merge/replace can only ever LOWER unariness, never raise it).
            if not survivor._is_unary:
                dependent._is_unary = False
        for idx, dep in enumerate(dead.dependencies):
            edges = self._dependents_index.get(dep.id)
            if edges is not None:
                self._dependents_index[dep.id] = [
                    (s, i) for (s, i) in edges if not (s is dead and i == idx)
                ]
        self._dead_ids.add(dead.id)

    def _sweep_dead(self) -> None:
        """Trim eradicated steps from `self.steps` in one pass; clear the working set."""
        if not self._dead_ids:
            return
        dead = self._dead_ids
        self.steps = [s for s in self.steps if s.id not in dead]
        self._dead_ids = set()

    def tree_shake(self, consumption_roots: List[Step]) -> List[Step]:
        """Drop steps unreachable from `consumption_roots` AND not side-effecting.

        `consumption_roots` is the executor's consumption surface for the finalized
        plan — every `FieldPlan.step` plus each `ObjectPlan.layer.parent_step`, across the
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
        # prune the reverse index to the kept set (both dependency keys and dependent edges)
        # so the maintained index stays a faithful inverse of the live DAG after the trim.
        self._dependents_index = {
            dep_id: [(s, i) for (s, i) in edges if s.id in keep]
            for dep_id, edges in self._dependents_index.items()
            if dep_id in keep
        }
        return [s for s in side_effecting if s.id not in reachable]

    def deduplicate(self) -> Dict[int, Step]:
        """Merge structurally identical steps; return ``old id -> survivor`` remap.

        Two steps are structurally identical iff they share class, ``peer_key``, and
        their dependencies (after remap) point at the same survivor ids in the same
        order, and their ``dedup_params`` match. The lowest-id step in a group wins;
        every reference (other steps' ``dependencies`` and the returned remap) is
        rewired to it. Iterated to a fixpoint because merging deps can make their
        dependents newly identical.

        A merged step is `eradicate`d (its dependents repointed onto the winner, the step
        removed from `self.steps` and the reverse index) rather than hidden in a survivors
        map and trimmed later — so `dependents_of` reflects the merge immediately.
        """
        entry_steps = list(self.steps)  # snapshot for the full old-id -> final remap
        merged: Dict[int, Step] = {}  # old id -> immediate winner
        changed = True
        while changed:
            changed = False
            by_key: Dict[tuple, Step] = {}
            for step in order_steps(self.steps):
                if step.id in self._dead_ids:
                    continue  # already merged away this pass
                if not step.dedupable:
                    continue  # side-effecting (e.g. mutation): never merge with a peer
                key = _structural_key(step)
                winner = by_key.get(key)
                if winner is None:
                    by_key[key] = step
                    continue
                # merge `step` into `winner`: notify, then eradicate (repoint its dependents)
                step.deduplicated_with(winner)
                self.eradicate(step, winner)
                merged[step.id] = winner
                changed = True
        self._sweep_dead()
        # the full old-id -> final remap over every step present at entry, chains collapsed
        remap: Dict[int, Step] = {}
        for step in entry_steps:
            final = step
            while final.id in merged and merged[final.id] is not final:
                final = merged[final.id]
            remap[step.id] = final
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


def _structural_key(step: Step) -> tuple:
    """A hashable key identifying a step's structural identity for dedup.

    Reads the step's dependency ids DIRECTLY: `eradicate` repoints a merged step's
    dependents onto the winner immediately, so every live step's `dependencies` already
    point at the current survivors — no survivor-map resolution is needed.
    """
    dep_ids = tuple(dep.id for dep in step.dependencies)
    return (type(step), step.peer_key, dep_ids, step.dedup_params())


def _resolve_merged(merged: Dict[int, Step]) -> Dict[int, Step]:
    """Collapse transitive `old id -> immediate replacement` links into `old id -> final`.

    One pass can replace A with B and a later one replace B with C; callers want `A -> C`
    directly. Follows each chain to the final survivor and returns only the entries that
    actually moved (an unchanged step is absent — matching the old `_collapse_chain`).
    """
    out: Dict[int, Step] = {}
    for old_id, immediate in merged.items():
        final = immediate
        while final.id in merged and merged[final.id] is not final:
            final = merged[final.id]
        out[old_id] = final
    return out


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
