"""Tests for the generic core steps: Relay global ids, ``node(id)``, list transforms.

All NO-DB and core-agnostic — these steps never touch SQLAlchemy. Each step is run
through ``run_steps`` over a bucket so the hard contract holds (``execute`` runs ONCE
and returns a column of length ``count``). The :class:`NodeStep` tests additionally
prove the batching payoff: a ``node`` query spanning several ids of the same type
invokes that type's loader EXACTLY ONCE (the Grafast batching gate carried through the
node interface), routing per typename so a mixed-type query batches once PER type.

A dedup-correctness test ships for every step whose config changes its result:
``NodeStep`` (the loader registry), ``FilterStep`` (the predicate). It proves both
directions — different config => NOT peers, identical config => peers — through the
planner's real ``Plan.deduplicate`` pass, the same gate the engine relies on.
"""

import asyncio
from typing import Any, List

import pytest

from grafast_py.core_steps import (
    ConstantStep,
    FilterStep,
    FirstStep,
    LastStep,
    NodeStep,
    ReverseStep,
    decode_global_id,
    encode_global_id,
    filter_step,
    first_step,
    last_step,
    node,
    reverse_step,
)
from grafast_py.dag import Plan, order_steps
from grafast_py.step_model import Step, run_steps


def never_awaitable(_value: Any) -> bool:
    return False


def is_coro(value: Any) -> bool:
    return asyncio.iscoroutine(value)


class SourceStep(Step):
    """0-dependency source seeded with a fixed column (test scaffold)."""

    # a per-entry batch source (like RootStep/ItemStep), so never unary.
    _is_unary = False

    def __init__(self, column: List[Any]) -> None:
        super().__init__()
        self.column = column

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        return self.column


def run(targets: List[Step], count: int, awaitable=never_awaitable):
    ordered = order_steps(targets)
    return run_steps(count, ordered, awaitable)


def dedup_key(step: Step):
    """The class + peer_key + dedup_params slice of the planner's structural key."""
    return (type(step), step.peer_key, step.dedup_params())


# ----------------------------------------------------------- global id encode/decode
def test_global_id_round_trips_single_part():
    gid = encode_global_id("User", 42)
    assert isinstance(gid, str)
    assert decode_global_id(gid) == ("User", (42,))


def test_global_id_round_trips_composite_parts():
    gid = encode_global_id("Store", 7, "north")
    assert decode_global_id(gid) == ("Store", (7, "north"))


def test_global_id_is_opaque_base64_carrying_the_typename():
    # the id must be url-printable base64; the typename rides inside (so node(id)
    # can dispatch) but is not plainly readable.
    gid = encode_global_id("Post", 1)
    import base64

    assert base64.b64decode(gid).decode().startswith('["Post"')


def test_encode_global_id_rejects_empty_id():
    with pytest.raises(ValueError):
        encode_global_id("User")


def test_decode_global_id_rejects_garbage():
    with pytest.raises(ValueError):
        decode_global_id("!!!not base64!!!")


def test_decode_global_id_rejects_non_array_payload():
    import base64
    import json

    forged = base64.b64encode(json.dumps({"typename": "User"}).encode()).decode()
    with pytest.raises(ValueError):
        decode_global_id(forged)


def test_decode_global_id_rejects_missing_id_part():
    import base64
    import json

    forged = base64.b64encode(json.dumps(["User"]).encode()).decode()
    with pytest.raises(ValueError):
        decode_global_id(forged)


# --------------------------------------------------------------------------- node
def test_node_resolves_via_per_type_loader_batched_once():
    calls = {"User": 0, "Post": 0}

    def load_users(keys: List[Any]) -> List[Any]:
        calls["User"] += 1
        return [{"__typename": "User", "id": k} for k in keys]

    def load_posts(keys: List[Any]) -> List[Any]:
        calls["Post"] += 1
        return [{"__typename": "Post", "id": k} for k in keys]

    ids = SourceStep(
        [
            encode_global_id("User", 1),
            encode_global_id("Post", 9),
            encode_global_id("User", 2),
            encode_global_id("User", 1),  # duplicate id, same type
        ]
    )
    n = NodeStep(ids, {"User": load_users, "Post": load_posts})
    results = run([n], 4)

    # THE GATE: one batched load PER type for the whole bucket (not one-per-id).
    assert calls == {"User": 1, "Post": 1}
    assert results[n.id] == [
        {"__typename": "User", "id": 1},
        {"__typename": "Post", "id": 9},
        {"__typename": "User", "id": 2},
        {"__typename": "User", "id": 1},  # duplicate scattered from the same batch
    ]


def test_node_loader_receives_keys_in_entry_order():
    seen = {}

    def load_users(keys: List[Any]) -> List[Any]:
        seen["keys"] = list(keys)
        return [k * 10 for k in keys]

    ids = SourceStep([encode_global_id("User", k) for k in (3, 1, 2)])
    n = NodeStep(ids, {"User": load_users})
    results = run([n], 3)

    assert seen["keys"] == [3, 1, 2]
    assert results[n.id] == [30, 10, 20]


def test_node_routes_composite_id_as_a_tuple_key():
    seen = {}

    def load_stores(keys: List[Any]) -> List[Any]:
        seen["keys"] = list(keys)
        return [f"store{k}" for k in keys]

    ids = SourceStep([encode_global_id("Store", 1, 2), encode_global_id("Store", 3, 4)])
    n = NodeStep(ids, {"Store": load_stores})
    results = run([n], 2)

    # a composite id is passed to the loader as the tuple of its parts.
    assert seen["keys"] == [(1, 2), (3, 4)]
    assert results[n.id] == ["store(1, 2)", "store(3, 4)"]


def test_node_carries_unknown_type_as_a_per_entry_error_without_poisoning_siblings():
    def load_users(keys: List[Any]) -> List[Any]:
        return [{"id": k} for k in keys]

    ids = SourceStep(
        [encode_global_id("User", 1), encode_global_id("Ghost", 99)]
    )
    n = NodeStep(ids, {"User": load_users})
    results = run([n], 2)

    assert results[n.id][0] == {"id": 1}  # the known type still resolves
    assert isinstance(results[n.id][1], KeyError)  # unknown type carried per entry


def test_node_carries_a_malformed_id_as_a_per_entry_error():
    def load_users(keys: List[Any]) -> List[Any]:
        return [{"id": k} for k in keys]

    ids = SourceStep([encode_global_id("User", 1), "garbage-not-a-global-id"])
    n = NodeStep(ids, {"User": load_users})
    results = run([n], 2)

    assert results[n.id][0] == {"id": 1}
    assert isinstance(results[n.id][1], ValueError)


def test_node_carries_a_non_string_id_as_a_per_entry_error_without_poisoning_siblings():
    """A non-string id (None / int) is carried per-entry, not poisoning sibling entries.

    ``decode_global_id`` calls ``.encode()`` on the id; a non-string would raise
    ``AttributeError`` out of ``execute`` and poison EVERY sibling entry (across all types),
    contradicting the per-entry isolation guarantee. The non-string guard makes it a
    ``ValueError`` carried as that one entry's value while a real id of another type resolves.
    """
    def load_users(keys: List[Any]) -> List[Any]:
        return [{"id": k} for k in keys]

    for bad_id in (None, 12345, b"bytes"):
        ids = SourceStep([encode_global_id("User", 1), bad_id])
        n = NodeStep(ids, {"User": load_users})
        results = run([n], 2)
        # the real id of another type still resolves; the non-string entry carries its error.
        assert results[n.id][0] == {"id": 1}
        assert isinstance(results[n.id][1], ValueError)


def test_decode_global_id_rejects_non_string_with_value_error():
    """``decode_global_id`` rejects a non-string id with a ValueError (not AttributeError)."""
    for bad_id in (None, 12345, b"bytes"):
        with pytest.raises(ValueError, match="expected a string"):
            decode_global_id(bad_id)


def test_node_asserts_loader_result_alignment():
    def bad_load(keys: List[Any]) -> List[Any]:
        return [{"id": "only one"}]  # 1 result for 2 keys

    ids = SourceStep([encode_global_id("User", 1), encode_global_id("User", 2)])
    n = NodeStep(ids, {"User": bad_load})
    with pytest.raises(AssertionError):
        run([n], 2)


@pytest.mark.asyncio
async def test_node_supports_an_async_loader_called_once():
    calls = {"n": 0}

    async def load_users(keys: List[Any]) -> List[Any]:
        calls["n"] += 1
        await asyncio.sleep(0)
        return [{"id": k} for k in keys]

    ids = SourceStep([encode_global_id("User", k) for k in (1, 2, 1)])
    n = NodeStep(ids, {"User": load_users})
    pending = run([n], 3, awaitable=is_coro)
    assert asyncio.iscoroutine(pending)
    results = await pending

    assert calls["n"] == 1
    assert results[n.id] == [{"id": 1}, {"id": 2}, {"id": 1}]


def test_node_helper_snapshots_the_registry():
    def load_users(keys):
        return [k for k in keys]

    registry = {"User": load_users}
    n = node(SourceStep([encode_global_id("User", 1)]), registry)
    registry["User"] = lambda keys: ["mutated"]  # must not affect the built step

    results = run([n], 1)
    assert results[n.id] == [1]


# ----------------------------------------------------------------- node dedup
def test_node_same_loaders_are_peers():
    """Two node steps over the SAME loader set + same id step merge to one."""
    def load_users(keys):
        return keys

    ids = SourceStep([encode_global_id("User", 1)])
    a = NodeStep(ids, {"User": load_users})
    b = NodeStep(ids, {"User": load_users})
    assert dedup_key(a) == dedup_key(b)

    plan = Plan()
    plan.add_step(a)
    plan.add_step(b)
    remap = plan.deduplicate()
    # the lower-id step wins; both references resolve to one survivor.
    assert remap[a.id] is remap[b.id]


def test_node_different_loaders_are_not_peers():
    """Different loader registries change which rows each id resolves to => not peers."""
    def load_a(keys):
        return keys

    def load_b(keys):
        return keys

    ids = SourceStep([encode_global_id("User", 1)])
    a = NodeStep(ids, {"User": load_a})
    b = NodeStep(ids, {"User": load_b})
    assert a.peer_key != b.peer_key
    assert a.dedup_params() != b.dedup_params()
    assert dedup_key(a) != dedup_key(b)

    plan = Plan()
    plan.add_step(a)
    plan.add_step(b)
    remap = plan.deduplicate()
    assert remap[a.id] is a
    assert remap[b.id] is b


def test_node_dedup_is_order_independent_over_the_registry():
    """The loader set, not its insertion order, fixes identity."""
    def load_users(keys):
        return keys

    def load_posts(keys):
        return keys

    ids = SourceStep([encode_global_id("User", 1)])
    a = NodeStep(ids, {"User": load_users, "Post": load_posts})
    b = NodeStep(ids, {"Post": load_posts, "User": load_users})
    assert dedup_key(a) == dedup_key(b)


# ---------------------------------------------------------------------- filter
def test_filter_step_keeps_matching_items_per_entry():
    source = SourceStep([[1, 2, 3, 4], [5, 6], []])
    even = FilterStep(source, lambda v: v % 2 == 0)
    results = run([even], 3)
    assert results[even.id] == [[2, 4], [6], []]


def test_filter_step_passes_through_none_and_exception_entries():
    err = RuntimeError("upstream boom")
    source = SourceStep([[1, 2, 3], None, err])
    odd = FilterStep(source, lambda v: v % 2 == 1)
    results = run([odd], 3)
    assert results[odd.id][0] == [1, 3]
    assert results[odd.id][1] is None
    assert results[odd.id][2] is err


def test_filter_step_predicate_raise_is_carried_per_entry_not_poisoning_the_bucket():
    """A predicate that raises for one entry's item errors ONLY that entry; siblings complete."""
    # the predicate reads d["ok"]; entry 1 has an item missing that key -> KeyError.
    source = SourceStep([[{"ok": True}, {"ok": False}], [{"ok": True}, {"nope": 1}]])
    kept = FilterStep(source, lambda d: d["ok"])
    results = run([kept], 2)
    # entry 0 filters cleanly; entry 1's predicate raise is carried as THAT entry's value,
    # so the bucket is not poisoned (entry 0 still completed).
    assert results[kept.id][0] == [{"ok": True}]
    assert isinstance(results[kept.id][1], KeyError)


def test_filter_dedup_same_predicate_peers_different_not():
    """Filter dedup keys on predicate identity (a closure is unique), both directions."""
    pred = lambda v: v > 0  # noqa: E731
    source = SourceStep([[1, -1]])
    a = FilterStep(source, pred)
    b = FilterStep(source, pred)
    assert dedup_key(a) == dedup_key(b)

    c = FilterStep(source, lambda v: v > 0)  # a DIFFERENT object: not a peer
    assert dedup_key(a) != dedup_key(c)

    plan = Plan()
    plan.add_step(a)
    plan.add_step(b)
    plan.add_step(c)
    remap = plan.deduplicate()
    assert remap[a.id] is remap[b.id]  # same predicate object: merged
    assert remap[c.id] is c  # different object: kept


# ----------------------------------------------------------------------- first
def test_first_step_takes_the_head_or_none():
    source = SourceStep([[10, 20], [], [30]])
    head = FirstStep(source)
    results = run([head], 3)
    assert results[head.id] == [10, None, 30]


def test_first_step_passes_through_exception_entries():
    err = ValueError("boom")
    source = SourceStep([[1, 2], err])
    head = FirstStep(source)
    results = run([head], 2)
    assert results[head.id] == [1, err]


def test_first_steps_over_the_same_list_dedup():
    source = SourceStep([[1, 2]])
    a = FirstStep(source)
    b = FirstStep(source)
    assert dedup_key(a) == dedup_key(b)
    plan = Plan()
    plan.add_step(a)
    plan.add_step(b)
    remap = plan.deduplicate()
    assert remap[a.id] is remap[b.id]


# ------------------------------------------------------------------------ last
def test_last_step_takes_the_tail_or_none():
    source = SourceStep([[10, 20], [], [30]])
    tail = LastStep(source)
    results = run([tail], 3)
    assert results[tail.id] == [20, None, 30]


# --------------------------------------------------------------------- reverse
def test_reverse_step_reverses_each_entry_without_mutating_source():
    original = [1, 2, 3]
    source = SourceStep([original, [9]])
    rev = ReverseStep(source)
    results = run([rev], 2)
    assert results[rev.id] == [[3, 2, 1], [9]]
    assert original == [1, 2, 3]  # the source list was not mutated


def test_reverse_step_passes_through_none_entries():
    source = SourceStep([[1, 2], None])
    rev = ReverseStep(source)
    results = run([rev], 2)
    assert results[rev.id] == [[2, 1], None]


# ----------------------------------------------------- list-transform constructors
def test_list_transform_helpers_build_the_right_step():
    source = SourceStep([[1, 2, 3]])
    assert isinstance(filter_step(source, lambda v: True), FilterStep)
    assert isinstance(first_step(source), FirstStep)
    assert isinstance(last_step(source), LastStep)
    assert isinstance(reverse_step(source), ReverseStep)


def test_list_transforms_compose_over_a_list_producing_step():
    # filter -> reverse -> first, all in one pass over the per-entry lists.
    source = SourceStep([[1, 2, 3, 4, 5, 6]])
    evens = filter_step(source, lambda v: v % 2 == 0)
    reversed_evens = reverse_step(evens)
    head = first_step(reversed_evens)
    results = run([head], 1)
    assert results[head.id] == [6]  # last even, via reverse+first


def test_constant_step_feeds_a_list_transform():
    # a list_producing constant works as the source of a transform (no SQL needed).
    source = ConstantStep([3, 1, 2])
    rev = reverse_step(source)
    results = run([rev], 2)
    assert results[rev.id] == [[2, 1, 3], [2, 1, 3]]
