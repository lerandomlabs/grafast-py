"""The Step base class and the bucket step executor.

This is the heart of the Grafast plan-then-execute model: a *step* is a node in a
plan DAG, built once at plan time and executed ONCE over a whole *bucket* (a batch
of `count` entries flowing through a layer together). `execute(count, values)`
returns a list of length `count` — one output per bucket position — and `values[d]`
is the already-computed output column (also length `count`) of the step's `d`-th
dependency. Running a step once per bucket — rather than re-entering a resolver per
(field, parent) pair — is what makes batching automatic: a future `loadMany` step
sees EVERY key in its bucket in a single `execute`, so it can issue one batch call.

This module ships the base class and the executor plus the legacy-resolver adapter
(`ResolveStep`). The concrete value steps (access, lambda, constant, list, object,
loadOne/loadMany) and the plan-resolver API arrive in the next build stage; they all
subclass `Step` and obey the same `execute` contract, so the executor here is the
single place steps are run.
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

    def __init__(self) -> None:
        self.dependencies: List["Step"] = []
        self.id = -1

    def add_dependency(self, step: "Step") -> int:
        """Wire a dependency on `step`; returns its integer dependency index."""
        index = len(self.dependencies)
        self.dependencies.append(step)
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

        Passed the owning `Plan` so a dependent-absorbing optimizer (the future
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

    Returns a dict mapping `step.id -> output column`. When any step's `execute`
    returns a coroutine (an async *column*, e.g. a future batched load), the whole
    call returns a coroutine resolving to that dict; steps are then awaited in
    dependency order so a dependent never runs before its dependency's column is
    materialised. `is_awaitable` is the context's awaitable predicate.
    """
    results: dict[int, List[Any]] = dict(seed) if seed else {}

    for index, step in enumerate(ordered_steps):
        cols = [results[dep.id] for dep in step.dependencies]
        span = _span(on_step_batch, step, count)
        span.__enter__()
        out = step.execute(count, cols)
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
                is_awaitable,
                on_step_batch,
            )
        span.__exit__(None, None, None)
        _store(results, step, count, out)

    return results


def _span(on_step_batch, step, count):
    """Build the tracing span context manager for one step batch (no-op when unset)."""
    if on_step_batch is None:
        return nullcontext()
    span = on_step_batch(step, count)
    return span if span is not None else nullcontext()


async def _run_steps_async(
    count, ordered_steps, results, index, pending_step, pending, pending_span,
    is_awaitable, on_step_batch=None,
):
    """Finish a step run that hit an awaitable column, awaiting in dep order."""
    out = await pending
    pending_span.__exit__(None, None, None)
    _store(results, pending_step, count, out)

    for step in ordered_steps[index + 1 :]:
        cols = [results[dep.id] for dep in step.dependencies]
        with _span(on_step_batch, step, count):
            out = step.execute(count, cols)
            if is_awaitable(out):
                out = await out
        _store(results, step, count, out)

    return results


def _store(results: dict, step: Step, count: int, out: List[Any]) -> None:
    """Record a step's output column, asserting the hard length contract."""
    if len(out) != count:
        raise AssertionError(
            f"step {type(step).__name__}#{step.id} returned {len(out)} values"
            f" for a bucket of {count}"
        )
    results[step.id] = out


__all__ = ["Step", "run_steps"]
