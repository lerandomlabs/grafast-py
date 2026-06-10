"""The Step base class and the bucket step executor.

This is the heart of the Grafast plan-then-execute model: a *step* is a node in a
plan DAG, built once at plan time and executed ONCE over a whole *bucket* (a batch
of `count` entries flowing through a layer together). `execute(count, values)`
returns a list of length `count` — one output per bucket position — and `values[d]`
is the already-computed output column (also length `count`) of the step's `d`-th
dependency. Running a step once per bucket — rather than re-entering a resolver per
(field, parent) pair — is what makes batching automatic: a `loadMany` step sees EVERY
key in its bucket in a single `execute`, so it can issue one batch call.

This module ships the base class and the executor. The concrete value steps (access,
lambda, constant, list, object, loadOne/loadMany), the plan-resolver API, and the
resolver-adapter (`ResolveStep`, in `steps.py`) all subclass `Step` and obey the same
`execute` contract, so the executor here is the single place steps are run. A step may
declare `wants_extra` to receive a per-invocation `BucketExtra` (request context +
per-parent paths) as a third `execute` argument; pure value/load steps do not.
"""

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from .dag import Plan


class Step:
    """A node in the plan DAG, executed once per bucket.

    Subclasses MUST implement `execute(count, values) -> list[Any]` returning a
    list of EXACTLY `count` entries (entry `i` is this step's result for bucket
    position `i`). The list may instead be returned as a coroutine resolving to
    such a list (async steps); the executor awaits it.

    Dependencies on other steps are wired with `add_dependency`, which returns the
    integer dependency index. NEVER store another `Step` instance for use at
    execute time — steps are ephemeral and may be replaced during
    deduplicate/optimize; hold the dependency index and read `values[index]` in
    `execute`. (`self.dependencies` is owned by the planner for DAG traversal and is
    rewritten in place when steps are merged.)
    """

    # assigned by the planner when the step is added to the plan; -1 until then.
    id: int

    # whether the dedup pass may MERGE this step into a structurally-identical peer.
    # True for pure value/read steps (merging is a safe optimization — the same value
    # computed/loaded twice is computed once). SIDE-EFFECTING steps (e.g. mutations) set
    # this False so two distinct writes are never collapsed into one.
    dedupable: bool = True

    # whether the cross-parent hoist pass may LIFT this step to a shallower layer (so it runs once
    # over a batch instead of once-per-child-bucket). Requires the step to be a DETERMINISTIC,
    # entry-INDEPENDENT function of its inputs WHOSE COLUMN HOLDS CONCRETE VALUES: firing once and
    # fanning the result to every child must equal firing per child. Defaults True for the pure sync
    # transforms (constant / access / list / object / first / last / reverse / filter) and the
    # column-resolving batch steps (load / node / each, whose coroutine-of-column run_steps resolves
    # before any fan-out). (Hoisting is SEPARATE from ``_is_unary`` run-once: a load is hoistable but
    # not unary.) It is the explicit OPT-OUT, set False by a plain resolver (``ResolveStep`` — impure
    # / side-effecting) AND ``LambdaStep`` (async-capable: its column may hold raw per-entry
    # coroutines that fan-out would alias across rows — see core_steps).
    hoistable: bool = True

    # whether `run_steps` passes the per-bucket-invocation BucketExtra (request context +
    # per-parent paths) as a third execute() argument. False for every pure value/load step
    # (a step's column is a function of its dependency columns alone); the resolver-adapter
    # sets it True because it needs the request's field_resolver/middleware/build_resolve_info
    # and per-parent paths, which are per-invocation and MUST NOT be stored on the shared step.
    wants_extra: bool = False

    # whether this step's value is the SAME for every entry in its bucket — a request-constant
    # (upstream's ``_isUnary``). Defaults True and is NARROWED to False the moment the step gains
    # a non-unary dependency (`add_dependency`) or a rewire repoints it onto one (`Plan.eradicate`),
    # so narrowing is monotone — it can only ever LOWER unariness, never raise it. Pure value /
    # shape / transform steps (constant / access / list / object / first / last / reverse / lambda /
    # filter) inherit the default and stay unary OVER unary inputs (the purity contract). The batch
    # SOURCES and impure / I/O steps force it False as a class attribute: ``RootStep`` (its column
    # IS the bucket of parents) and ``ItemStep`` (a per-entry exploded list element); and the steps
    # whose run-once equivalence is not assumed — ``ResolveStep`` (an arbitrary host resolver),
    # ``EachStep``, the batch load / node steps, and every pg step (each does its own batching / IO).
    # A unary step is executed ONCE per bucket and its single value broadcast to all entries —
    # byte-identical to the per-entry run, fewer invocations (see :func:`run_steps`). FOOTGUN: a
    # NEW 0-dependency step whose column is genuinely PER-ENTRY (a batch source) MUST set
    # ``_is_unary = False`` (like ``RootStep`` / ``ItemStep``); the default True would otherwise run
    # it once and broadcast entry 0 to every position, with no loud failure.
    _is_unary: bool = True

    # whether this step's ``execute`` is guaranteed SYNC — its column holds concrete values, never a
    # per-entry awaitable. The concrete steps set this explicitly; the base default is conservative
    # (False) so a step is run-once-broadcast (`_bucket_unariness`) only when PROVEN sync — otherwise
    # broadcasting could alias one coroutine across parents (single-await). Run-once is the only
    # consumer; correctness never depends on a False here, only the run-once optimization does.
    is_sync_and_safe: bool = False

    def __init__(self) -> None:
        self.dependencies: List["Step"] = []
        self.id = -1

    def add_dependency(self, step: "Step") -> int:
        """Wire a dependency on `step`; returns its integer dependency index.

        This is the construction-time edge chokepoint: it also NARROWS unariness — a step that
        depends on a non-unary step is itself non-unary. Narrowing only ever sets False (monotone).
        The per-instance `_is_unary` flag is ADVISORY, not authoritative: a later rewrite
        (`Plan.eradicate`) can lower a dependent without propagating transitively, and the actual
        run-once decision is recomputed per bucket (seed-aware) in :func:`run_steps`
        (`_bucket_unariness`), which is conservative — it never marks a step unary unless every
        dependency is effectively unary in that bucket.
        """
        index = len(self.dependencies)
        self.dependencies.append(step)
        if not step._is_unary:
            self._is_unary = False
        return index

    @property
    def dependency_count(self) -> int:
        return len(self.dependencies)

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        """Run once over a bucket of `count` entries; return a list of length `count`.

        `values[d]` is the output column (length `count`) of dependency `d`.
        """
        raise NotImplementedError

    @property
    def peer_key(self) -> str:
        """A cheap pre-filter for dedup: steps with different keys are never peers.

        Defaults to the class name, so a subclass that does not override it (and has
        no params) deduplicates purely on class + dependencies. Concrete value/load
        steps override it to fold in their params (path, fn identity, load fn).
        """
        return type(self).__name__

    def dedup_params(self) -> tuple:
        """The dedup-relevant config beyond class + deps (default: none)."""
        return ()

    def deduplicate(self, peers: List["Step"]) -> List["Step"]:
        """Return the subset of `peers` truly equivalent to `self` (default: none).

        Peers are pre-filtered by the planner to share this step's class and
        dependencies; `deduplicate` is the per-step check that remaining params
        (path / fn identity / load fn / data) also match. Returning `[]` blocks
        deduplication.
        """
        return []

    def deduplicated_with(self, winner: "Step") -> None:
        """Called on the losing step when merged into `winner` (default: no-op)."""

    def optimize(self, plan: "Plan") -> "Step":
        """Self-rewrite during the optimize pass: return `self` to keep, or a
        replacement `Step` to be wired in for `self`. Default: identity.

        Passed the owning `Plan` so a dependent-absorbing optimizer (the
        query-inlining step) can find ITS dependents via `plan.dependents_of(self)`
        and register a freshly built replacement step. A replacement returned here is
        re-wired into the DAG by the pass; per the class contract above, an optimizer
        must NOT stash `Step` references for execute time — only dependency indices.
        """
        return self

    def finalize(self) -> None:
        """Last-chance precompute before execution (default: no-op)."""


def run_steps(
    count: int,
    ordered_steps: List[Step],
    is_awaitable: Callable[[Any], bool],
    seed: Optional[Dict[int, List[Any]]] = None,
    on_step_batch: Optional[Callable[[Step, int], Any]] = None,
    extra=None,
):
    """Execute a topologically ordered list of steps once each over a bucket.

    `ordered_steps` must be sorted so every step's dependencies appear before it.
    Each step is run exactly once with its dependency columns gathered from the
    already-computed results, and its output column (length `count`) is stored.

    `seed` pre-populates `step.id -> column` for steps whose output is already known
    (the bucket's parent column: a child object bucket reuses the parent field
    step's already-computed output as its source rather than recomputing it). Seeded
    steps must be excluded from `ordered_steps`; their dependencies are never read.

    `on_step_batch(step, count)` is the opt-in tracing hook (no-op by default): it is
    called around each step's `execute` and may return a context manager (a span) or
    None. It wraps the batch boundary — the exact point Grafast batches a layer.

    `extra` is an optional per-invocation `BucketExtra` (request context + per-parent
    paths) passed to a step's `execute` only when it declares `wants_extra`; it is a
    fresh argument per call, never shared mutable state.

    Returns a dict mapping `step.id -> output column`. When any step's `execute`
    returns a coroutine (an async *column*, e.g. a batched load), the whole
    call returns a coroutine resolving to that dict; steps are then awaited in
    dependency order so a dependent never runs before its dependency's column is
    materialised. `is_awaitable` is the context's awaitable predicate.
    """
    results: dict[int, List[Any]] = dict(seed) if seed else {}
    # effective per-bucket unariness: a SEEDED boundary column is per-entry (it IS the bucket of
    # parents, e.g. a constant LIST exploded into a child object bucket), so a step is run ONCE
    # only if it is unary-capable AND every dependency is effectively unary IN THIS BUCKET. This
    # makes unariness bucket-relative — a globally-unary step seeded as a child boundary is
    # correctly treated as batch there. Precomputed once over the topologically-ordered steps.
    unary_here = _bucket_unariness(ordered_steps, set(results), count)

    for index, step in enumerate(ordered_steps):
        # a unary (request-constant) step runs ONCE over a single representative entry; its
        # length-1 output is then broadcast to `count` at store time so the bucket store keeps
        # its "every column length count" invariant and downstream completion is byte-identical.
        unary = unary_here[step.id]
        step_count = 1 if unary else count
        cols = _gather_cols(results, step, unary)
        span = _span(on_step_batch, step, count)
        span.__enter__()
        try:
            out = (
                step.execute(step_count, cols, extra)
                if step.wants_extra
                else step.execute(step_count, cols)
            )
        except BaseException:
            # a synchronous raise must still close the span (no leak across the error).
            span.__exit__(None, None, None)
            raise
        if is_awaitable(out):
            # the real work is in the await; keep the span open across it.
            return _run_steps_async(
                count,
                ordered_steps,
                results,
                index,
                step,
                out,
                span,
                unary,
                unary_here,
                is_awaitable,
                on_step_batch,
                extra,
            )
        span.__exit__(None, None, None)
        _store(results, step, count, out, unary)

    return results


def _bucket_unariness(
    ordered_steps: List[Step], seeded: set, count: int
) -> Dict[int, bool]:
    """Effective unariness of each step IN THIS BUCKET (`step.id -> bool`).

    A step is run-once-and-broadcast only if ALL of:
      * it is unary-capable (`step._is_unary`) — its value is request-constant; AND
      * it is provably SYNC (`is_sync_and_safe`) — so its column is CONCRETE values, never a raw
        per-entry awaitable. Broadcasting an awaitable would alias ONE coroutine across every
        parent (a coroutine is single-await; the @stream path, which completes parents
        separately, would then raise "cannot reuse already awaited coroutine"). An async
        ``lambda_step`` over a constant therefore runs per entry (distinct coroutines), not once; AND
      * none of its dependencies is a per-entry source in this bucket — every dependency is itself
        run-once here (so its column is concrete + all-equal) and not a SEEDED boundary (a seeded
        column IS the bucket of parents, so anything reading it is per-entry).
    Conservative: any uncertainty falls to batch.
    """
    unary_here: Dict[int, bool] = {}
    if count <= 0:
        return {step.id: False for step in ordered_steps}
    for step in ordered_steps:
        eff = step._is_unary and step.is_sync_and_safe
        if eff:
            for dep in step.dependencies:
                if dep.id in seeded or not unary_here.get(dep.id, False):
                    eff = False
                    break
        unary_here[step.id] = eff
    return unary_here


def _span(on_step_batch, step, count):
    """Build the tracing span context manager for one step batch (no-op when unset)."""
    if on_step_batch is None:
        return nullcontext()
    span = on_step_batch(step, count)
    return span if span is not None else nullcontext()


async def _run_steps_async(
    count, ordered_steps, results, index, pending_step, pending, pending_span, pending_unary,
    unary_here, is_awaitable, on_step_batch=None, extra=None,
):
    """Finish a step run that hit an awaitable column, awaiting in dep order."""
    try:
        out = await pending
    except BaseException:
        # the span carried across the sync->async handoff must still close if the awaited step
        # raises (mirrors the sync path) — otherwise the failure-time span leaks.
        pending_span.__exit__(None, None, None)
        raise
    pending_span.__exit__(None, None, None)
    _store(results, pending_step, count, out, pending_unary)

    for step in ordered_steps[index + 1 :]:
        unary = unary_here[step.id]
        step_count = 1 if unary else count
        cols = _gather_cols(results, step, unary)
        with _span(on_step_batch, step, count):
            out = (
                step.execute(step_count, cols, extra)
                if step.wants_extra
                else step.execute(step_count, cols)
            )
            if is_awaitable(out):
                out = await out
        _store(results, step, count, out, unary)

    return results


def _gather_cols(results: dict, step: Step, unary: bool) -> List[List[Any]]:
    """Gather a step's dependency columns, sliced to one representative entry when unary.

    A unary step's dependencies are all unary too (narrowing), so the first (broadcast) value
    of each dependency column IS the representative value the single run needs.
    """
    if unary:
        return [[results[dep.id][0]] for dep in step.dependencies]
    return [results[dep.id] for dep in step.dependencies]


def _store(
    results: dict, step: Step, count: int, out: List[Any], unary: bool = False
) -> None:
    """Record a step's output column, broadcasting a unary step's single value to `count`.

    A unary step is run once (`step_count == 1`) and must return exactly one value; that value
    is broadcast to every bucket entry so the store keeps its length-`count` invariant. A batch
    step's output is stored as-is under the hard length contract.
    """
    if unary:
        if len(out) != 1:
            raise AssertionError(
                f"unary step {type(step).__name__}#{step.id} returned {len(out)} values"
                f" (a unary step runs once and must return exactly 1)"
            )
        # broadcast the single value to the bucket: `out * count` stores `count` REFERENCES to the
        # one computed value (not clones) — safe because bucket values are read-only downstream, and
        # identical to what ``ConstantStep.execute`` ([data] * count) has always produced.
        results[step.id] = out * count
        return
    if len(out) != count:
        raise AssertionError(
            f"step {type(step).__name__}#{step.id} returned {len(out)} values"
            f" for a bucket of {count}"
        )
    results[step.id] = out


__all__ = ["Step", "run_steps"]
