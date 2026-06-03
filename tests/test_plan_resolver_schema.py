"""Tests for the plan-resolver API and the planner's step-DAG construction.

Covers the three attachment surfaces (`set_field_plan`, `make_grafast_schema`,
`GrafastSchemaBindable`) and the reader (`get_field_plan`), then proves the planner,
for a field WITH a plan resolver, calls it with the parent step + coerced args,
builds a step depending on the parent, and recurses into object subfields with that
step as the new `$parent`. Fields WITHOUT a plan resolver must contribute no steps
(legacy resolver-adapter path), which keeps the conformance suite green.
"""

from typing import Any, List

from graphql import GraphQLObjectType, build_schema, parse
from graphql.execution.collect_fields import collect_fields

from grafast_py import GrafastExecutionContext
from grafast_py.core_steps import (
    AccessStep,
    LoadManyStep,
    RootStep,
    get,
    load_many,
)
from grafast_py.plan import plan_operation
from grafast_py.schema import (
    FieldArgs,
    GrafastSchemaBindable,
    get_field_plan,
    make_grafast_schema,
    set_field_plan,
)


def find_object_completer(completer):
    from grafast_py.completion import (
        ListCompleter,
        NonNullCompleter,
        ObjectCompleter,
    )

    if isinstance(completer, ObjectCompleter):
        return completer
    if isinstance(completer, NonNullCompleter):
        return find_object_completer(completer.inner)
    if isinstance(completer, ListCompleter):
        return find_object_completer(completer.item_completer)
    return None


SDL = """
type Query {
  people: [Person]
}
type Person {
  name: String
  friends: [Person]
}
"""


# ---------------------------------------------------- attachment / reader API
def test_set_and_get_field_plan_roundtrip():
    schema = build_schema(SDL)
    person = schema.type_map["Person"]

    def name_plan(parent, args, info):
        return get(parent, "name")

    field = person.fields["name"]
    assert get_field_plan(field) is None
    set_field_plan(field, name_plan)
    assert get_field_plan(field) is name_plan
    assert field.extensions["grafast"]["plan"] is name_plan


def test_make_grafast_schema_attaches_plans_from_a_map():
    def people_plan(parent, args, info):
        return get(parent, "people")

    schema = make_grafast_schema(SDL, {"Query": {"people": people_plan}})
    query = schema.type_map["Query"]
    assert get_field_plan(query.fields["people"]) is people_plan
    # untouched field stays on the legacy path
    assert get_field_plan(schema.type_map["Person"].fields["name"]) is None


def test_make_grafast_schema_rejects_unknown_field():
    import pytest

    with pytest.raises(KeyError):
        make_grafast_schema(SDL, {"Query": {"nope": lambda *a: None}})


def test_grafast_schema_bindable_binds_plans():
    def people_plan(parent, args, info):
        return get(parent, "people")

    schema = build_schema(SDL)
    bindable = GrafastSchemaBindable({"Query": {"people": people_plan}})
    bindable.bind_to_schema(schema)
    assert get_field_plan(schema.type_map["Query"].fields["people"]) is people_plan


def test_field_args_exposes_coerced_values():
    args = FieldArgs({"limit": 10, "after": "x"})
    assert args.get("limit") == 10
    assert args["after"] == "x"
    assert "limit" in args
    assert args.get("missing", 99) == 99


# ------------------------------------------------ planner builds a step DAG
def build_plan(schema, query: str):
    document = parse(query)
    operation = document.definitions[0]
    ctx = GrafastExecutionContext.build(schema, document)
    root_type = schema.query_type
    root_fields = collect_fields(
        ctx.schema, ctx.fragments, ctx.variable_values, root_type, operation.selection_set
    )
    object_plan = plan_operation(ctx, operation, root_type, root_fields)
    return ctx, object_plan


def test_planner_builds_a_step_for_a_plan_field_depending_on_root():
    def people_plan(parent, args, info):
        # parent is the RootStep; build a step that depends on it
        return get(parent, "people")

    schema = make_grafast_schema(SDL, {"Query": {"people": people_plan}})
    ctx, object_plan = build_plan(schema, "{ people { name } }")

    people_fp = object_plan.fields[0]
    assert people_fp.plan_fn is people_plan
    assert isinstance(people_fp.step, AccessStep)
    # the field step depends on the root step (the planner threaded $parent in)
    assert isinstance(ctx._grafast_root_step, RootStep)
    assert people_fp.step.dependencies[0] is ctx._grafast_root_step


def test_planner_recurses_with_the_field_step_as_child_parent():
    captured = {}

    def people_plan(parent, args, info):
        step = load_many(get(parent, "people_ids"), lambda keys: [[] for _ in keys])
        captured["people_step"] = step
        return step

    def name_plan(parent, args, info):
        # `parent` here MUST be the people field's step (the child $parent)
        captured["name_parent"] = parent
        return get(parent, "name")

    schema = make_grafast_schema(
        SDL,
        {"Query": {"people": people_plan}, "Person": {"name": name_plan}},
    )
    _ctx, object_plan = build_plan(schema, "{ people { name } }")

    people_fp = object_plan.fields[0]
    assert isinstance(people_fp.step, LoadManyStep)
    # the child name plan received the people step as its $parent → DAG is stitched
    assert captured["name_parent"] is captured["people_step"]

    child_completer = find_object_completer(people_fp.completer)
    name_fp = child_completer.child_plan.fields[0]
    assert isinstance(name_fp.step, AccessStep)
    assert name_fp.step.dependencies[0] is captured["people_step"]


def test_legacy_field_without_plan_contributes_no_step():
    schema = build_schema(SDL)  # no plans attached
    _ctx, object_plan = build_plan(schema, "{ people { name } }")
    people_fp = object_plan.fields[0]
    assert people_fp.plan_fn is None
    assert people_fp.step is None
