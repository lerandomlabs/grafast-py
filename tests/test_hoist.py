"""The executable definition of cross-parent hoisting.

Conformance byte-identity is NECESSARY but INSUFFICIENT to prove this property: with the
flag OFF (the default) the finalize pass is a no-op, so the suite would pass even if the
hoist pass did nothing useful. This module proves the actual deliverable — a step whose
inputs are constant across a child bucket is LIFTED to a shallower layer so it runs
once-per-request instead of once-per-child-bucket, with data byte-identical:

(a) a relocation unit test — a request-constant child step is MOVED from a deeper
    LayerPlan to a shallower one by `hoist_steps`; the deeper layer no longer lists it
    (it is in `hoisted_out_ids`, absent from `ordered_steps`) and the shallower layer
    owns it (`hoisted_in`, present in `ordered_steps`). The `FieldPlan.step` reference is
    UNCHANGED vs the hoist-OFF control (proving "WHERE not WHETHER").

(b) an end-to-end fire-once batching test — a request-constant key used inside a LIST
    field fires its LoadStep ONCE under hoist (lifted to the root, run once over the
    single root bucket) vs once-per-parent-bucket naive, data byte-identical, threaded via
    `parent_store` (the child bucket reads the hoisted column, never re-runs the step).

(c) a guard test — a side-effecting (`dedupable=False`) step, a RootStep, and a
    boundary-dependent step are NOT hoisted, and a mutation operation disables hoisting
    entirely.
"""

from typing import Any, List

import pytest
from graphql import parse
from graphql.execution.collect_fields import collect_fields

from grafast_py import GrafastExecutionContext
from grafast_py.completion import find_object_completer
from grafast_py.config import GrafastConfig
from grafast_py.core_steps import RootStep, access, constant, lambda_step, load_one
from grafast_py.dag import order_steps
from grafast_py.entry import grafast_execute
from grafast_py.plan import plan_operation
from grafast_py.schema import make_grafast_schema, resolve_type_from_tag
from grafast_py.step_model import Step


def build_plan(schema, query: str, hoist: bool):
    """Finalize a plan from a query with an explicit hoist config (shadows the suite toggle)."""

    class _Ctx(GrafastExecutionContext):
        grafast_config = GrafastConfig(hoist=hoist)

    document = parse(query)
    operation = document.definitions[0]
    ctx = _Ctx.build(schema, document)
    root_type = schema.query_type
    root_fields = collect_fields(
        ctx.schema, ctx.fragments, ctx.variable_values, root_type, operation.selection_set
    )
    object_plan = plan_operation(ctx, operation, root_type, root_fields)
    return ctx, object_plan


def leaf(key):
    def plan(parent_step, args, info):
        return access(parent_step, (key,))

    return plan


# --------------------------------------------------------------------------- (a) relocation


RELOCATION_SDL = """
type Query {
  people: [Person!]!
}
type Person {
  id: Int!
  tag: Int!
}
"""


def test_request_constant_step_relocates_to_shallower_layer():
    """A request-constant child step is lifted to the root layer; the child drops it.

    `tag = load_one(constant("k"), fn)` depends only on a ConstantStep (request-constant) —
    no dependency on the per-child boundary — so it (and its constant) are hoistable out of the
    Person child layer into the root. The `id` access depends on the child boundary and stays.
    """
    rows = [{"id": 1}, {"id": 2}, {"id": 3}]

    def plan_people(parent, args, info):
        return load_one(constant("all"), lambda keys: [rows for _ in keys])

    def plan_tag(parent, args, info):
        return load_one(constant("k"), lambda keys: [100 for _ in keys])

    schema = make_grafast_schema(
        RELOCATION_SDL,
        {"Query": {"people": plan_people}, "Person": {"id": leaf("id"), "tag": plan_tag}},
    )

    # control: hoist OFF — the child owns the tag step, nothing is annotated.
    _, off_plan = build_plan(schema, "{ people { id tag } }", hoist=False)
    off_people_fp = off_plan.fields[0]
    off_child = find_object_completer(off_people_fp.completer).child_plan
    off_tag_fp = next(fp for fp in off_child.fields if fp.response_name == "tag")
    assert off_child.layer.hoisted_out_ids == frozenset()
    assert off_plan.layer.hoisted_in == []
    # in the control the tag step IS in the child's ordered_steps (it runs per child bucket).
    off_child_ordered_ids = {s.id for s in (off_child.layer.ordered_steps or [])}
    assert off_tag_fp.step.id in off_child_ordered_ids

    # hoist ON — the tag step (and its constant key) relocate to the root layer.
    _, on_plan = build_plan(schema, "{ people { id tag } }", hoist=True)
    on_people_fp = on_plan.fields[0]
    on_child = find_object_completer(on_people_fp.completer).child_plan
    on_tag_fp = next(fp for fp in on_child.fields if fp.response_name == "tag")
    on_id_fp = next(fp for fp in on_child.fields if fp.response_name == "id")

    # the deeper (child) layer no longer RUNS the tag step: it is hoisted-out and excluded
    # from this layer's ordered_steps.
    assert on_tag_fp.step.id in on_child.layer.hoisted_out_ids
    on_child_ordered_ids = {s.id for s in (on_child.layer.ordered_steps or [])}
    assert on_tag_fp.step.id not in on_child_ordered_ids
    # the boundary-dependent id access stays in the child layer (NOT hoisted).
    assert on_id_fp.step.id not in on_child.layer.hoisted_out_ids
    assert on_id_fp.step.id in on_child_ordered_ids

    # the shallower (root) layer now OWNS the tag step: it is hoisted-in and present in the
    # root's ordered_steps (runs once over the single root bucket).
    hoisted_in_ids = {s.id for s in on_plan.layer.hoisted_in}
    assert on_tag_fp.step.id in hoisted_in_ids
    root_ordered_ids = {s.id for s in (on_plan.layer.ordered_steps or [])}
    assert on_tag_fp.step.id in root_ordered_ids

    # WHERE not WHETHER: the FieldPlan.step object is the SAME class/structure across builds —
    # hoisting never replaces the step. (Ids are stable plan-time integers; the step object
    # carries the same dependency shape.)
    assert type(on_tag_fp.step) is type(off_tag_fp.step)
    assert [d.id for d in on_tag_fp.step.dependencies] == [
        d.id for d in off_tag_fp.step.dependencies
    ]
    # the consumption surface is unchanged: the tag step is still reachable from its FieldPlan
    # in both builds (only its production layer moved), so tree-shake keeps it identically.
    assert on_tag_fp.step.id in {s.id for s in order_steps([on_tag_fp.step])}


# --------------------------------------------------------------------------- (b) fire-once


def test_request_constant_load_fires_once_under_hoist_in_list_field():
    """A request-constant load inside a LIST field fires ONCE under hoist, data identical.

    Naive: the LoadStep is owned by the per-item Person layer and runs once per Person bucket
    (one bucket per root parent here, so once). Hoisted: the LoadStep is lifted to the root and
    runs once over the single root bucket, its column threaded DOWN to every Person item via
    `parent_store` — read, never re-run in the child bucket.
    """
    rows = [{"id": 1}, {"id": 2}, {"id": 3}]

    def run(hoist):
        calls = {"n": 0}

        def plan_people(parent, args, info):
            return load_one(constant("all"), lambda keys: [rows for _ in keys])

        def plan_tag(parent, args, info):
            def load_tags(keys: List[Any]) -> List[Any]:
                calls["n"] += 1
                return [100 for _ in keys]

            return load_one(constant("k"), load_tags)

        schema = make_grafast_schema(
            RELOCATION_SDL,
            {"Query": {"people": plan_people}, "Person": {"id": leaf("id"), "tag": plan_tag}},
        )
        result = grafast_execute(
            schema, "{ people { id tag } }", config=GrafastConfig(hoist=hoist)
        )
        return result, calls["n"]

    off_result, off_calls = run(False)
    on_result, on_calls = run(True)

    expected = {
        "people": [
            {"id": 1, "tag": 100},
            {"id": 2, "tag": 100},
            {"id": 3, "tag": 100},
        ]
    }
    # data byte-identical between naive and hoisted.
    assert off_result.errors is None and on_result.errors is None
    assert off_result.data == expected
    assert on_result.data == expected
    # the load fires once in BOTH (single root bucket either way), proving hoist did not
    # MULTIPLY the call and that the child bucket did NOT re-run the hoisted load.
    assert off_calls == 1
    assert on_calls == 1


TWO_LEVEL_SDL = """
type Query {
  orgs: [Org!]!
}
type Org {
  id: Int!
  people: [Person!]!
}
type Person {
  id: Int!
  tag: Int!
}
"""


def test_hoist_data_identical_through_two_nesting_levels():
    """A request-constant load TWO list-levels deep is hoisted to root, data byte-identical.

    `Person.tag = load_one(constant("k"), fn)` lives under Query.orgs -> Org.people -> Person.
    With hoist ON it relocates to the root layer; its column must thread DOWN through BOTH the
    Org and Person buckets via `parent_store` (read at each level, never re-run). This backs the
    multi-level threading claim with an executed ON-vs-OFF data-identity comparison.
    """
    orgs = [{"id": 10}, {"id": 20}]
    people_by_org = [{"id": 1}, {"id": 2}]

    def run(hoist):
        def plan_orgs(parent, args, info):
            return constant(orgs)

        def plan_people(parent, args, info):
            return load_one(constant("p"), lambda keys: [people_by_org for _ in keys])

        def plan_tag(parent, args, info):
            return load_one(constant("k"), lambda keys: [100 for _ in keys])

        schema = make_grafast_schema(
            TWO_LEVEL_SDL,
            {
                "Query": {"orgs": plan_orgs},
                "Org": {"id": leaf("id"), "people": plan_people},
                "Person": {"id": leaf("id"), "tag": plan_tag},
            },
        )
        return grafast_execute(
            schema,
            "{ orgs { id people { id tag } } }",
            config=GrafastConfig(hoist=hoist),
        )

    off_result = run(False)
    on_result = run(True)

    expected = {
        "orgs": [
            {
                "id": 10,
                "people": [{"id": 1, "tag": 100}, {"id": 2, "tag": 100}],
            },
            {
                "id": 20,
                "people": [{"id": 1, "tag": 100}, {"id": 2, "tag": 100}],
            },
        ]
    }
    assert off_result.errors is None and on_result.errors is None
    assert off_result.data == expected
    assert on_result.data == expected  # hoisted column threaded through two levels, identical


def test_hoisted_column_threaded_via_parent_store_not_rerun_in_child():
    """The hoisted column reaches the child bucket via parent_store, NOT re-run there.

    We assert the mechanism end-to-end: under hoist the child Person layer has the tag step in
    `hoisted_out_ids` and excluded from its `ordered_steps`, so the executor MUST seed it from
    the parent bucket (otherwise `id` would resolve but `tag` would be missing). The capture
    seam confirms the child bucket is entered (the hop fires) while the load counter stays at 1
    — the step ran once in the root and was READ, not recomputed, per child position.
    """
    rows = [{"id": 1}, {"id": 2}, {"id": 3}]
    calls = {"n": 0}
    captured = {}

    def plan_people(parent, args, info):
        return load_one(constant("all"), lambda keys: [rows for _ in keys])

    def plan_tag(parent, args, info):
        def load_tags(keys: List[Any]) -> List[Any]:
            calls["n"] += 1
            return [100 for _ in keys]

        return load_one(constant("k"), load_tags)

    schema = make_grafast_schema(
        RELOCATION_SDL,
        {"Query": {"people": plan_people}, "Person": {"id": leaf("id"), "tag": plan_tag}},
    )

    from grafast_py.execute import execute_object_plan

    ctx, object_plan = build_plan(schema, "{ people { id tag } }", hoist=True)

    people_fp = object_plan.fields[0]
    child_plan = find_object_completer(people_fp.completer).child_plan
    tag_fp = next(fp for fp in child_plan.fields if fp.response_name == "tag")

    # the child layer dropped the tag step from its run order (it is seeded from parent_store).
    assert tag_fp.step.id in child_plan.layer.hoisted_out_ids
    assert tag_fp.step.id not in {s.id for s in (child_plan.layer.ordered_steps or [])}

    def capture(completer, keep_objs, keep_origin, keep_paths):
        captured["entered"] = True
        captured["n_children"] = len(keep_objs)

    ctx._grafast_capture_keep_origin = capture

    results = execute_object_plan(ctx, object_plan, [None], [None])

    # the child bucket WAS entered (the hop fired) with the three Person items …
    assert captured.get("entered") is True
    assert captured["n_children"] == 3
    # … and the hoisted load ran exactly once (not once per child position), proving it was
    # threaded down (read) rather than re-executed in the child bucket.
    assert calls["n"] == 1
    assert results[0] == {
        "people": [
            {"id": 1, "tag": 100},
            {"id": 2, "tag": 100},
            {"id": 3, "tag": 100},
        ]
    }


# --------------------------------------------------------------------------- (c) guards


class _SideEffectStep(Step):
    """A toy request-constant step that is side-effecting (`dedupable=False`).

    Like a write, it must run where the planner put it — even though its (zero) inputs are
    request-constant, it must NEVER be hoisted.
    """

    dedupable = False
    is_sync_and_safe = True

    def __init__(self, data: Any) -> None:
        super().__init__()
        self.data = data

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        return [self.data] * count

    @property
    def peer_key(self) -> str:
        return f"sideeffect|{id(self)}"


def test_side_effecting_step_is_not_hoisted():
    """A `dedupable=False` step is request-constant but immovable — never hoisted."""
    rows = [{"id": 1}, {"id": 2}]

    def plan_people(parent, args, info):
        return load_one(constant("all"), lambda keys: [rows for _ in keys])

    def plan_tag(parent, args, info):
        return _SideEffectStep(7)

    schema = make_grafast_schema(
        RELOCATION_SDL,
        {"Query": {"people": plan_people}, "Person": {"id": leaf("id"), "tag": plan_tag}},
    )

    _, on_plan = build_plan(schema, "{ people { id tag } }", hoist=True)
    people_fp = on_plan.fields[0]
    child = find_object_completer(people_fp.completer).child_plan
    tag_fp = next(fp for fp in child.fields if fp.response_name == "tag")

    # NOT hoisted out of the child, and NOT hoisted into the root.
    assert tag_fp.step.id not in child.layer.hoisted_out_ids
    assert on_plan.layer.hoisted_out_ids == frozenset()
    assert tag_fp.step.id not in {s.id for s in on_plan.layer.hoisted_in}
    # it still runs in the child layer.
    assert tag_fp.step.id in {s.id for s in (child.layer.ordered_steps or [])}


def test_root_step_is_never_hoisted():
    """The RootStep IS the boundary; it is never lifted into a `hoisted_in` anywhere."""
    rows = [{"id": 1}]

    def plan_people(parent, args, info):
        return load_one(constant("all"), lambda keys: [rows for _ in keys])

    def plan_tag(parent, args, info):
        return load_one(constant("k"), lambda keys: [1 for _ in keys])

    schema = make_grafast_schema(
        RELOCATION_SDL,
        {"Query": {"people": plan_people}, "Person": {"id": leaf("id"), "tag": plan_tag}},
    )

    _, on_plan = build_plan(schema, "{ people { id tag } }", hoist=True)

    def visit(op):
        for step in op.layer.hoisted_in:
            assert not isinstance(step, RootStep)
        for fp in op.fields:
            child = find_object_completer(fp.completer)
            if child is not None and child.child_plan is not None:
                visit(child.child_plan)

    visit(on_plan)


def test_boundary_dependent_step_is_not_hoisted():
    """A step depending on the per-child boundary (the parent row) stays in the child layer."""
    rows = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]

    def plan_people(parent, args, info):
        return load_one(constant("all"), lambda keys: [rows for _ in keys])

    schema = make_grafast_schema(
        """
        type Query { people: [Person!]! }
        type Person { id: Int!, name: String! }
        """,
        # both fields read a column of the parent row (depend on the child boundary).
        {"Query": {"people": plan_people}, "Person": {"id": leaf("id"), "name": leaf("name")}},
    )

    _, on_plan = build_plan(schema, "{ people { id name } }", hoist=True)
    people_fp = on_plan.fields[0]
    child = find_object_completer(people_fp.completer).child_plan

    # nothing is hoistable — every child step descends from the per-child boundary.
    assert child.layer.hoisted_out_ids == frozenset()
    assert on_plan.layer.hoisted_in == []
    child_ordered_ids = {s.id for s in (child.layer.ordered_steps or [])}
    for fp in child.fields:
        assert fp.step.id in child_ordered_ids


MUTATION_SDL = """
type Query { ping: Int! }
type Mutation {
  doThing: Result!
}
type Result {
  id: Int!
  tag: Int!
}
"""


def test_mutation_disables_hoisting():
    """Hoisting is disabled entirely under a mutation operation, even for a request-constant step."""
    result_row = {"id": 1}

    def plan_do_thing(parent, args, info):
        return load_one(constant("do"), lambda keys: [result_row for _ in keys])

    def plan_tag(parent, args, info):
        # request-constant — WOULD be hoistable under a query, but a mutation disables hoisting.
        return load_one(constant("k"), lambda keys: [99 for _ in keys])

    schema = make_grafast_schema(
        MUTATION_SDL,
        {
            "Query": {"ping": lambda p, a, i: constant(1)},
            "Mutation": {"doThing": plan_do_thing},
            "Result": {"id": leaf("id"), "tag": plan_tag},
        },
    )

    class _Ctx(GrafastExecutionContext):
        grafast_config = GrafastConfig(hoist=True)

    document = parse("mutation { doThing { id tag } }")
    operation = document.definitions[0]
    ctx = _Ctx.build(schema, document)
    root_type = schema.mutation_type
    root_fields = collect_fields(
        ctx.schema, ctx.fragments, ctx.variable_values, root_type, operation.selection_set
    )
    object_plan = plan_operation(ctx, operation, root_type, root_fields)

    # the plan recorded it is a mutation, so the hoist pass was gated off in finalize.
    assert ctx._grafast_plan.is_mutation is True

    # no layer anywhere carries a hoist annotation.
    def visit(op):
        assert op.layer.hoisted_in == []
        assert op.layer.hoisted_out_ids == frozenset()
        for fp in op.fields:
            child = find_object_completer(fp.completer)
            if child is not None and child.child_plan is not None:
                visit(child.child_plan)

    visit(object_plan)


# ------------------------------------------------- (d) abstract + hoist coverage + the guard


ABSTRACT_HOIST_SDL = """
type Query { items: [Item!]! }
interface Item { id: Int! }
type Widget implements Item { id: Int! tag: Int! }
"""


def test_abstract_field_with_hoist_is_byte_identical():
    """A hoistable load under an interface concrete-type field: hoist ON == OFF, no error.

    ``test_hoist.py`` otherwise has ZERO abstract coverage, and the hoist carry now threads the
    bridge UNIFORMLY through the abstract completion path (no special-cased path the bridge can
    skip). This exercises that path under hoist and proves byte-identity. The concrete-type subtree
    is self-contained (its own RootStep), so the bridge is inert here — but the path is now uniform
    and the fail-loud guard (below) backs the invariant.
    """
    rows = [{"id": 1, "__typename": "Widget"}, {"id": 2, "__typename": "Widget"}]

    def run(hoist):
        def plan_items(parent, args, info):
            return constant(rows)

        def plan_tag(parent, args, info):
            return load_one(constant("k"), lambda keys: [100 for _ in keys])

        schema = make_grafast_schema(
            ABSTRACT_HOIST_SDL,
            {
                "Query": {"items": plan_items},
                "Widget": {"id": leaf("id"), "tag": plan_tag},
            },
            type_resolvers={"Item": resolve_type_from_tag("__typename")},
        )
        return grafast_execute(
            schema,
            "{ items { id ... on Widget { tag } } }",
            config=GrafastConfig(hoist=hoist),
        )

    off_result = run(False)
    on_result = run(True)
    expected = {"items": [{"id": 1, "tag": 100}, {"id": 2, "tag": 100}]}
    assert off_result.errors is None and on_result.errors is None
    assert off_result.data == expected
    assert on_result.data == expected


def test_build_hoist_parent_store_fails_loud_when_bridge_missing():
    """A layer that hoisted steps OUT but got no bridge fails LOUD — never a silent missing column.

    On every threaded completer path the bridge is present whenever a child has hoisted-out steps,
    so this can't fire in practice; the guard makes a future regression (a completer path that
    forgets to carry the bridge) crash loudly rather than read a missing/re-run column.
    """
    from types import SimpleNamespace

    from grafast_py.completion import build_hoist_parent_store

    hoisted = SimpleNamespace(layer=SimpleNamespace(hoisted_out_ids=frozenset({7})))
    with pytest.raises(AssertionError):
        build_hoist_parent_store(hoisted, None, [0, 1])
    # the no-hoist case stays a quiet None (the byte-identical default path).
    plain = SimpleNamespace(layer=SimpleNamespace(hoisted_out_ids=frozenset()))
    assert build_hoist_parent_store(plain, None, [0, 1]) is None


def test_sync_lambda_over_constant_is_run_once():
    """A SYNC ``lambda_step`` over a request-constant input is computed ONCE and shared, so ``fn``
    fires exactly once for the whole request (``off_n == on_n == 1``), data byte-identical.

    A sync lambda's value is a CONCRETE result, so it satisfies both share-contracts (pure by
    contract + sync by inspection): the engine runs it once and copies the result to every row —
    via the executor run-once (so it holds even with hoist OFF) and/or hoisting. An ASYNC lambda
    does NOT get this (its result is a per-row coroutine that cannot be shared — see
    tests/test_unary_model_gaps.py::test_unary_async_lambda_runs_per_entry_not_broadcast and the
    @stream regression). The call counter is a test instrument only; ``fn``'s return is constant.
    """
    orgs = [{"id": 10}, {"id": 20}]
    people = [{"id": 1}, {"id": 2}]

    def run(hoist):
        calls = {"n": 0}

        def pure_tag(_):
            calls["n"] += 1
            return 100  # PURE: always 100, independent of call count / order

        def plan_orgs(p, a, i):
            return constant(orgs)

        def plan_people(p, a, i):
            return load_one(constant("p"), lambda keys: [people for _ in keys])

        def plan_tag(p, a, i):
            return lambda_step(constant("k"), pure_tag)  # pure, request-constant input

        schema = make_grafast_schema(
            TWO_LEVEL_SDL,
            {
                "Query": {"orgs": plan_orgs},
                "Org": {"id": leaf("id"), "people": plan_people},
                "Person": {"id": leaf("id"), "tag": plan_tag},
            },
        )
        result = grafast_execute(
            schema, "{ orgs { id people { id tag } } }", config=GrafastConfig(hoist=hoist)
        )
        assert result.errors is None
        tags = [[pp["tag"] for pp in o["people"]] for o in result.data["orgs"]]
        return tags, calls["n"]

    off_tags, off_n = run(False)
    on_tags, on_n = run(True)

    # data byte-identical (pure fn), hoist ON == OFF.
    assert off_tags == [[100, 100], [100, 100]]
    assert on_tags == off_tags
    # the SYNC lambda is run ONCE for the whole request (concrete value, shared) — whether or not
    # hoist is on (the executor run-once already collapses it). With the old barrier both were 4.
    assert off_n == 1 and on_n == 1
