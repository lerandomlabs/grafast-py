"""Tests for the Step base class and the bucket step executor (build:step-model).

These exercise the core plan-then-execute primitive directly: a `Step.execute`
runs ONCE over a whole bucket of `count` entries and returns a column of length
`count`, and `run_steps` runs an ordered DAG once each in dependency order, feeding
each step its dependencies' already-computed columns. The decisive property — that
a step sees the WHOLE bucket in a single `execute` call — is proven here with a call
counter; the loadMany batching gate in the next stage builds directly on it.
"""

import asyncio
from typing import Any, List

import pytest

from grafast_py.step_model import Step, run_steps


def never_awaitable(_value: Any) -> bool:
    return False


class SourceStep(Step):
    """0-dependency source seeded with a fixed column."""

    # a per-entry batch source (like RootStep/ItemStep), so never unary.
    _is_unary = False

    def __init__(self, column: List[Any]) -> None:
        super().__init__()
        self.column = column

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        return self.column


class CountingMapStep(Step):
    """Maps its single dependency's column through `fn`, counting execute calls.

    Crucially, it runs `fn` over the whole column in ONE `execute`, so `self.calls`
    counts batched passes, not per-entry invocations — the batching invariant.
    """

    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn
        self.calls = 0

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        self.calls += 1
        col = values[0]
        return [self.fn(v) for v in col]


def assign_ids(steps: List[Step]) -> None:
    for i, step in enumerate(steps):
        step.id = i


def test_run_steps_runs_each_step_once_in_dependency_order():
    source = SourceStep([1, 2, 3, 4])
    doubler = CountingMapStep(lambda v: v * 2)
    doubler.add_dependency(source)

    assign_ids([source, doubler])
    results = run_steps(4, [source, doubler], never_awaitable)

    assert results[source.id] == [1, 2, 3, 4]
    assert results[doubler.id] == [2, 4, 6, 8]
    # the whole bucket flowed through a SINGLE execute call (not one per entry)
    assert doubler.calls == 1


def test_step_over_n_entries_calls_execute_exactly_once():
    """The executor invariant the loadMany gate rests on: ONE pass per bucket."""
    n = 50
    source = SourceStep(list(range(n)))
    mapper = CountingMapStep(lambda v: v + 100)
    mapper.add_dependency(source)

    assign_ids([source, mapper])
    results = run_steps(n, [source, mapper], never_awaitable)

    assert results[mapper.id] == [v + 100 for v in range(n)]
    assert mapper.calls == 1


def test_run_steps_threads_columns_through_a_chain():
    source = SourceStep([10, 20, 30])
    add_one = CountingMapStep(lambda v: v + 1)
    add_one.add_dependency(source)
    times_two = CountingMapStep(lambda v: v * 2)
    times_two.add_dependency(add_one)

    assign_ids([source, add_one, times_two])
    results = run_steps(3, [source, add_one, times_two], never_awaitable)

    assert results[times_two.id] == [22, 42, 62]
    assert add_one.calls == 1
    assert times_two.calls == 1


def test_run_steps_asserts_column_length_contract():
    class WrongLengthStep(Step):
        _is_unary = False  # a batch step: must return one value per bucket entry

        def execute(self, count, values):
            return [0]  # returns 1 value for a bucket of 3 → contract violation

    bad = WrongLengthStep()
    bad.id = 0
    with pytest.raises(AssertionError):
        run_steps(3, [bad], never_awaitable)


@pytest.mark.asyncio
async def test_run_steps_supports_an_async_column():
    """A step may return a coroutine resolving to its whole column."""

    class AsyncSourceStep(Step):
        _is_unary = False  # a batch source: returns a per-entry column

        async def execute(self, count, values):
            await asyncio.sleep(0)
            return [7, 8, 9]

    def is_awaitable(v):
        return asyncio.iscoroutine(v)

    source = AsyncSourceStep()
    mapper = CountingMapStep(lambda v: v - 1)
    mapper.add_dependency(source)
    assign_ids([source, mapper])

    pending = run_steps(3, [source, mapper], is_awaitable)
    assert asyncio.iscoroutine(pending)
    results = await pending

    assert results[source.id] == [7, 8, 9]
    assert results[mapper.id] == [6, 7, 8]
    assert mapper.calls == 1
