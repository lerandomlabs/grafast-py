"""The executable definition of lifecycle detachment.

Conformance byte-identity is NECESSARY but INSUFFICIENT to prove this property: the
inline completion path is unchanged, so the suite would pass even if a child layer
still secretly required the live parent walk to materialise its seed. This module
proves the actual deliverable — a child object layer is runnable from
``(parent value column, index_map)`` ALONE, decoupled from the parent walk:

- run the ROOT layer to get the root store (no output walk);
- take the parent value column the child bucket descends into;
- run the CHILD layer DIRECTLY via ``run_child_layer_from_store`` /
  ``run_layer(parent_store=..., index_map projection)`` WITHOUT calling
  ``walk_output`` on the parent;
- assert the detached child store columns equal the inline nested execution's
  child store columns, that the index_map IS the completion-time
  ``keep_origin``/``origin`` scatter list, and that the loader-call counter equals
  the inline path (batching is sacrosanct).

The seed projection is ``[parent_column[o] for o in index_map]``, which equals the
transient ``keep_objs``/``objs`` the inline path materialises element-for-element —
so the IDENTICAL child step DAG runs over the IDENTICAL bucket.

Three cases cover the three descent shapes that lifecycle detachment generalises:
nested singular object, flattened LIST item, and abstract concrete-type GROUP. The
index_map indexes the FLATTENED item column for the list case, and the abstract
group's ``parent_step`` is a FRESH ``RootStep`` (not in the operation's outer store),
so its parent column is the abstract field's own resolved value column projected by
the group ``origin``.
"""

from typing import Any, List

from graphql import parse
from graphql.execution.collect_fields import collect_fields

from grafast_py import GrafastExecutionContext
from grafast_py.completion import find_object_completer
from grafast_py.core_steps import RootStep, access, constant, load_one
from grafast_py.execute import run_child_layer_from_store, run_layer
from grafast_py.plan import plan_operation
from grafast_py.schema import (
    make_grafast_schema,
    resolve_type_from_tag,
)


def build_plan(schema, query: str):
    """Finalize a plan from a query (mirrors tests/test_plan_finalize.build_plan)."""
    document = parse(query)
    operation = document.definitions[0]
    ctx = GrafastExecutionContext.build(schema, document)
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


# --------------------------------------------------------------------------- nested


NESTED_SDL = """
type Query {
  hero: Person
}
type Person {
  id: Int!
  name: String!
}
"""


def test_nested_object_layer_runs_from_store_column_and_index_map():
    """A singular object field's child layer runs from (parent column, index_map) alone."""
    from grafast_py.execute import execute_object_plan

    hero_row = {"id": 7, "name": "Luke"}
    name_calls = {"n": 0, "keys": None}

    def plan_hero(parent, args, info):
        # one batched load of the single hero row (a loadOne over a constant key)
        return load_one(constant("hero"), lambda keys: [hero_row for _ in keys])

    def plan_name(parent, args, info):
        # the leaf load whose batch-call counter proves batching is preserved
        def load_names(keys: List[Any]) -> List[Any]:
            name_calls["n"] += 1
            name_calls["keys"] = list(keys)
            return [k["name"] for k in keys]

        return load_one(parent, load_names)

    schema = make_grafast_schema(
        NESTED_SDL,
        {"Query": {"hero": plan_hero}, "Person": {"id": leaf("id"), "name": plan_name}},
    )

    # capture the hero->name hop's child seed + index map from the live inline walk.
    captured = {}

    def capture(completer, keep_objs, keep_origin, keep_paths):
        captured["objs"] = list(keep_objs)
        captured["origin"] = list(keep_origin)
        captured["paths"] = list(keep_paths)

    ctx, object_plan = build_plan(schema, "{ hero { name } }")
    ctx._grafast_capture_keep_origin = capture

    hero_fp = object_plan.fields[0]
    child_completer = find_object_completer(hero_fp.completer)
    child_plan = child_completer.child_plan
    # the child layer's parent_step IS the hero field's own step (child_parent_step).
    assert child_plan.layer.parent_step is hero_fp.step

    # 1. FULL inline execution of the root plan: fires the capture at the hop. (This is the
    #    live parent walk we are decoupling from.)
    execute_object_plan(ctx, object_plan, [None], [None])
    assert captured, "the hero->name hop did not fire the capture seam"

    # the inline child store is exactly what the hop ran: the child plan over keep_objs.
    name_calls["n"] = 0
    inline_child = execute_object_plan(ctx, child_plan, captured["objs"], captured["paths"])
    inline_name_calls = name_calls["n"]

    # 2. detached run: run ONLY the root layer (no walk), take the hero value column out of
    #    the store, and drive the child layer directly from (column, index_map).
    root_store = run_layer(ctx, object_plan.layer, [None], [None])
    parent_column = root_store[hero_fp.step.id]

    # one root parent, non-null hero, no is_type_of -> keep_origin == [0].
    index_map = list(range(len(parent_column)))
    assert index_map == captured["origin"]
    # the projection equals the inline keep_objs element-for-element.
    assert [parent_column[o] for o in index_map] == captured["objs"]

    name_calls["n"] = 0
    parent_store = {child_plan.layer.parent_step.id: parent_column}
    detached_child = run_child_layer_from_store(
        ctx, child_plan, parent_store, index_map, captured["paths"]
    )

    assert detached_child == inline_child
    # the leaf loader fired EXACTLY as many times as the inline child path (once).
    assert name_calls["n"] == inline_name_calls == 1


# --------------------------------------------------------------------------- list


LIST_SDL = """
type Query {
  people: [Person!]!
}
type Person {
  id: Int!
  name: String!
}
"""


def test_list_item_layer_runs_from_flattened_column_and_index_map():
    """A list-of-objects child layer runs from the FLATTENED item column + index_map."""
    from grafast_py.execute import execute_object_plan

    # the people field returns a LIST, so the child bucket is the FLATTENED items — the
    # subtle indexing the index_map must honour (not the per-parent list).
    rows = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}, {"id": 3, "name": "c"}]
    name_calls = {"n": 0, "keys": None}

    def plan_people(parent, args, info):
        return load_one(constant("all"), lambda keys: [rows for _ in keys])

    def plan_name(parent, args, info):
        def load_names(keys: List[Any]) -> List[Any]:
            name_calls["n"] += 1
            name_calls["keys"] = list(keys)
            return [k["name"] for k in keys]

        return load_one(parent, load_names)

    schema = make_grafast_schema(
        LIST_SDL,
        {"Query": {"people": plan_people}, "Person": {"id": leaf("id"), "name": plan_name}},
    )

    captured = {}

    def capture(completer, keep_objs, keep_origin, keep_paths):
        captured["objs"] = list(keep_objs)
        captured["origin"] = list(keep_origin)
        captured["paths"] = list(keep_paths)

    ctx, object_plan = build_plan(schema, "{ people { name } }")
    ctx._grafast_capture_keep_origin = capture

    people_fp = object_plan.fields[0]
    child_completer = find_object_completer(people_fp.completer)
    child_plan = child_completer.child_plan
    assert child_plan.layer.parent_step is people_fp.step

    # 1. FULL inline execution: fires the capture at the people-item -> name hop.
    execute_object_plan(ctx, object_plan, [None], [None])
    assert captured, "the people-item->name hop did not fire the capture seam"

    name_calls["n"] = 0
    inline_child = execute_object_plan(ctx, child_plan, captured["objs"], captured["paths"])
    inline_name_calls = name_calls["n"]

    # 2. detached run: the parent value column the index_map indexes is the FLATTENED item
    #    column, NOT the per-root list-of-people column held under people_fp.step.id.
    root_store = run_layer(ctx, object_plan.layer, [None], [None])
    per_root_lists = root_store[people_fp.step.id]
    flattened_column = [item for sublist in per_root_lists for item in sublist]

    # no nulls / no is_type_of -> keep_origin enumerates the flattened items in order.
    index_map = list(range(len(flattened_column)))
    assert index_map == captured["origin"]
    assert [flattened_column[o] for o in index_map] == captured["objs"]

    name_calls["n"] = 0
    parent_store = {child_plan.layer.parent_step.id: flattened_column}
    detached_child = run_child_layer_from_store(
        ctx, child_plan, parent_store, index_map, captured["paths"]
    )

    assert detached_child == inline_child
    assert name_calls["n"] == inline_name_calls == 1


# --------------------------------------------------------------------------- abstract


ABSTRACT_SDL = """
type Query {
  results: [SearchResult!]!
}
union SearchResult = Image | Video
type Image {
  id: Int!
  width: Int!
}
type Video {
  id: Int!
  durationSeconds: Int!
}
"""


def unwrap_abstract(completer):
    """Reach the AbstractCompleter under any List!/NonNull wrapper chain."""
    from grafast_py.completion import AbstractCompleter, ListCompleter, NonNullCompleter

    if isinstance(completer, AbstractCompleter):
        return completer
    if isinstance(completer, NonNullCompleter):
        return unwrap_abstract(completer.inner)
    if isinstance(completer, ListCompleter):
        return unwrap_abstract(completer.item_completer)
    return None


def test_abstract_group_layer_runs_from_value_column_and_origin():
    """An abstract concrete-type group runs from (abstract value column, origin) alone."""
    from grafast_py.execute import execute_object_plan

    rows = [
        {"__t": "Image", "id": 1, "width": 1920},
        {"__t": "Video", "id": 2, "duration_seconds": 42},
        {"__t": "Image", "id": 3, "width": 512},
    ]
    width_calls = {"n": 0}

    def plan_results(parent, args, info):
        return constant(rows)

    def plan_width(parent, args, info):
        def load_widths(keys: List[Any]) -> List[Any]:
            width_calls["n"] += 1
            return [k["width"] for k in keys]

        return load_one(parent, load_widths)

    schema = make_grafast_schema(
        ABSTRACT_SDL,
        {
            "Query": {"results": plan_results},
            "Image": {"id": leaf("id"), "width": plan_width},
            "Video": {"id": leaf("id"), "durationSeconds": leaf("duration_seconds")},
        },
        type_resolvers={"SearchResult": resolve_type_from_tag("__t")},
    )

    query = """
    { results { ... on Image { id width } ... on Video { id durationSeconds } } }
    """

    # 1. inline run: capture the Image group's seed + origin + its child_plan via the
    #    completion-time seam, while completing the abstract field for real so the inline
    #    child store and loader-call count are produced.
    captured_groups = {}

    def capture(object_type, child_plan, objs, origin, group_paths, values):
        captured_groups[object_type.name] = {
            "child_plan": child_plan,
            "objs": list(objs),
            "origin": list(origin),
            "paths": list(group_paths),
            "values": list(values),
        }

    ctx, object_plan = build_plan(schema, query)
    ctx._grafast_capture_group_origin = capture

    results_fp = object_plan.fields[0]
    abstract_completer = unwrap_abstract(results_fp.completer)
    assert abstract_completer is not None

    root_store = run_layer(ctx, object_plan.layer, [None], [None])
    abstract_value_column = [
        item for sublist in root_store[results_fp.step.id] for item in sublist
    ]
    paths = [None] * len(abstract_value_column)
    infos = [None] * len(abstract_value_column)

    # complete the abstract field over its whole value column: this fires the capture for
    # every concrete-type group and runs each group's child layer inline.
    from grafast_py.completion import complete_abstract_values

    complete_abstract_values(
        ctx, abstract_completer, list(abstract_value_column), paths, infos, results_fp.field_nodes
    )
    inline_width_calls = width_calls["n"]

    image = captured_groups["Image"]
    child_plan = image["child_plan"]
    # the group's parent_step is a FRESH RootStep (abstract_child_plan), NOT a key in the
    # outer operation store — the parent column is the abstract field's own value column.
    assert isinstance(child_plan.layer.parent_step, RootStep)

    origin = image["origin"]
    # the group seed equals the abstract value column projected by origin == the group objs.
    assert [abstract_value_column[o] for o in origin] == image["objs"]

    # the inline child store for the Image group, recomputed for comparison (the abstract
    # plan_cache makes child_plan stable, so the same step DAG runs over the same bucket).
    inline_child = execute_object_plan(ctx, child_plan, image["objs"], image["paths"])

    # 2. detached run: drive the group's child layer DIRECTLY from (abstract value column,
    #    origin) without re-entering abstract dispatch.
    width_calls["n"] = 0
    parent_store = {child_plan.layer.parent_step.id: abstract_value_column}
    detached_child = run_child_layer_from_store(
        ctx, child_plan, parent_store, origin, image["paths"]
    )

    assert detached_child == inline_child
    # inline ran the Image group ONCE (capture) + once (recompute) = 2; the detached run
    # adds one more. Each individual group run fires the width loader exactly once.
    assert inline_width_calls == 1
    assert width_calls["n"] == 1
