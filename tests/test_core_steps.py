"""Tests for the core plan-resolver steps in isolation (build:core-steps-api).

Each step is exercised directly through `run_steps` over a bucket so the hard
contract is checked at the source: `execute` runs ONCE per bucket and returns a
column of length `count`. The batch steps (loadOne/loadMany) additionally prove the
Phase-A payoff with a call counter — a load over N parents invokes the batch
callback EXACTLY ONCE — which is the gate the next stage's full executor builds on.
"""

import asyncio
from typing import Any, List

import pytest

from grafast_py.core_steps import (
    AccessStep,
    ConstantStep,
    EachStep,
    LambdaStep,
    ListStep,
    LoadManyStep,
    LoadOneStep,
    ObjectStep,
    get_in,
)
from grafast_py.dag import order_steps
from grafast_py.step_model import Step, run_steps


def never_awaitable(_value: Any) -> bool:
    return False


def is_coro(value: Any) -> bool:
    return asyncio.iscoroutine(value)


class SourceStep(Step):
    """0-dependency source seeded with a fixed column (test scaffold)."""

    def __init__(self, column: List[Any]) -> None:
        super().__init__()
        self.column = column

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        return self.column


def run(targets: List[Step], count: int, awaitable=never_awaitable):
    ordered = order_steps(targets)
    return run_steps(count, ordered, awaitable)


# --------------------------------------------------------------------- get_in
def test_get_in_walks_dicts_then_attributes_with_fallback():
    class Obj:
        def __init__(self):
            self.name = "leia"

    assert get_in({"a": {"b": 5}}, ["a", "b"]) == 5
    assert get_in({"a": None}, ["a", "b"], fallback="x") == "x"
    assert get_in(Obj(), ["name"]) == "leia"
    assert get_in({}, ["missing"], fallback=42) == 42
    assert get_in(None, ["a"], fallback="z") == "z"


# ------------------------------------------------------------------- constant
def test_constant_step_fills_the_whole_bucket():
    c = ConstantStep("hi")
    results = run([c], 3)
    assert results[c.id] == ["hi", "hi", "hi"]


# --------------------------------------------------------------------- access
def test_access_step_extracts_a_key_per_entry():
    source = SourceStep([{"id": 1}, {"id": 2}, {"id": 3}])
    acc = AccessStep(source, ["id"])
    results = run([acc], 3)
    assert results[acc.id] == [1, 2, 3]


def test_access_step_get_chains_deeper():
    source = SourceStep([{"a": {"b": 10}}, {"a": {"b": 20}}])
    acc = AccessStep(source, ["a"]).get("b")
    results = run([acc], 2)
    assert results[acc.id] == [10, 20]


def test_access_step_uses_fallback_for_missing():
    source = SourceStep([{"x": 1}, {}])
    acc = AccessStep(source, ["x"], fallback=-1)
    results = run([acc], 2)
    assert results[acc.id] == [1, -1]


# --------------------------------------------------------------------- lambda
def test_lambda_step_maps_each_entry_once():
    source = SourceStep([1, 2, 3, 4])
    lam = LambdaStep(source, lambda v: v * 10)
    results = run([lam], 4)
    assert results[lam.id] == [10, 20, 30, 40]


# ----------------------------------------------------------------------- list
def test_list_step_bundles_dependency_columns_per_entry():
    a = SourceStep([1, 2])
    b = SourceStep(["x", "y"])
    lst = ListStep([a, b])
    results = run([lst], 2)
    assert results[lst.id] == [[1, "x"], [2, "y"]]


# --------------------------------------------------------------------- object
def test_object_step_builds_a_dict_per_entry():
    name = SourceStep(["luke", "han"])
    age = SourceStep([19, 32])
    obj = ObjectStep({"name": name, "age": age})
    results = run([obj], 2)
    assert results[obj.id] == [
        {"name": "luke", "age": 19},
        {"name": "han", "age": 32},
    ]


def test_object_step_get_projects_back_to_the_stored_step():
    name = SourceStep(["luke"])
    obj = ObjectStep({"name": name})
    assert obj.get("name") is name


# ------------------------------------------------------------------ load_one
def test_load_one_calls_batch_fn_exactly_once_over_the_bucket():
    calls = {"n": 0}

    def batch_load(keys: List[Any]) -> List[Any]:
        calls["n"] += 1
        return [{"id": k, "name": f"user{k}"} for k in keys]

    source = SourceStep([1, 2, 3, 1])  # note the duplicate key 1
    loader = LoadOneStep(source, batch_load)
    results = run([loader], 4)

    assert calls["n"] == 1  # THE GATE: one batch call for the whole bucket
    assert results[loader.id] == [
        {"id": 1, "name": "user1"},
        {"id": 2, "name": "user2"},
        {"id": 3, "name": "user3"},
        {"id": 1, "name": "user1"},  # duplicate key scattered from the same result
    ]


def test_load_one_coalesces_duplicate_keys_into_one_batch_entry():
    seen_keys = {}

    def batch_load(keys: List[Any]) -> List[Any]:
        seen_keys["keys"] = list(keys)
        return [k * 100 for k in keys]

    source = SourceStep([5, 5, 7, 5])
    loader = LoadOneStep(source, batch_load)
    results = run([loader], 4)

    # duplicate key 5 appears ONCE in the batch, scattered to all three slots
    assert seen_keys["keys"] == [5, 7]
    assert results[loader.id] == [500, 500, 700, 500]


# ----------------------------------------------------------------- load_many
def test_load_many_calls_batch_fn_once_and_scatters_lists():
    calls = {"n": 0}

    def batch_load(keys: List[Any]) -> List[Any]:
        calls["n"] += 1
        # one LIST of records per key
        return [[f"{k}-a", f"{k}-b"] for k in keys]

    source = SourceStep(["p1", "p2", "p3"])
    loader = LoadManyStep(source, batch_load)
    results = run([loader], 3)

    assert calls["n"] == 1  # THE GATE for loadMany over N parents
    assert results[loader.id] == [
        ["p1-a", "p1-b"],
        ["p2-a", "p2-b"],
        ["p3-a", "p3-b"],
    ]


@pytest.mark.asyncio
async def test_load_many_supports_an_async_batch_fn_called_once():
    calls = {"n": 0}

    async def batch_load(keys: List[Any]) -> List[Any]:
        calls["n"] += 1
        await asyncio.sleep(0)
        return [[k, k] for k in keys]

    source = SourceStep([1, 2, 3])
    loader = LoadManyStep(source, batch_load)
    pending = run([loader], 3, awaitable=is_coro)
    assert asyncio.iscoroutine(pending)
    results = await pending

    assert calls["n"] == 1
    assert results[loader.id] == [[1, 1], [2, 2], [3, 3]]


def test_load_many_asserts_result_alignment():
    def bad_load(keys: List[Any]) -> List[Any]:
        return [["only one"]]  # returns 1 result for 3 unique keys

    source = SourceStep(["a", "b", "c"])
    loader = LoadManyStep(source, bad_load)
    with pytest.raises(AssertionError):
        run([loader], 3)


# ------------------------------------------------------------------------ each
def test_each_maps_a_list_step_batching_the_mapper_across_all_items():
    calls = {"n": 0}

    def batch_double(keys: List[Any]) -> List[Any]:
        calls["n"] += 1
        return [k * 2 for k in keys]

    # two parents, each with a list of ids; `each` explodes into one item bucket
    list_source = SourceStep([[1, 2], [3]])
    mapper_loader = []

    def mapper(item_step):
        loader = LoadOneStep(item_step, batch_double)
        mapper_loader.append(loader)
        return loader

    eacher = EachStep(list_source, mapper)
    results = run([eacher], 2, awaitable=is_coro)
    if asyncio.iscoroutine(results):
        results = asyncio.get_event_loop().run_until_complete(results)

    # the mapper's loader ran ONCE over all three items (1,2,3) across both parents
    assert calls["n"] == 1
    assert results[eacher.id] == [[2, 4], [6]]
