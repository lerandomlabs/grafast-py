"""Tests for the shared finalize path of the planner pipeline.

`finalize_plan` is the single optimize → dedup → tree-shake helper that BOTH the
operation root (`plan_operation`) and every abstract/object child subtree
(`completion.abstract_child_plan`) run, so abstract subtrees optimize identically to
the root. `collect_consumption_root_steps` walks the finalized ObjectPlan tree to the
executor's consumption surface (every `FieldPlan.step` + each `ObjectPlan.layer.parent_step`,
across transitively nested child plans) that tree-shake measures reachability against.

With the default identity `Step.optimize`, the whole finalize is a no-op: the result of
`plan_operation` is the same ObjectPlan with the same step DAG, which is what keeps the
conformance suite byte-identical.
"""

from graphql import parse
from graphql.execution.collect_fields import collect_fields

from grafast_py import GrafastExecutionContext
from grafast_py.core_steps import AccessStep, RootStep, get, load_many
from grafast_py.plan import (
    collect_consumption_root_steps,
    finalize_plan,
    plan_operation,
)
from grafast_py.schema import make_grafast_schema

SDL = """
type Query {
  people: [Person]
}
type Person {
  name: String
  friends: [Person]
}
"""


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


def find_object_completer(completer):
    from grafast_py.completion import find_object_completer as foc

    return foc(completer)


def test_finalize_plan_default_is_a_no_op_over_a_real_operation():
    """No step subclass overrides optimize → finalize keeps every step the planner built."""

    def people_plan(parent, args, info):
        return get(parent, "people")

    def name_plan(parent, args, info):
        return get(parent, "name")

    schema = make_grafast_schema(
        SDL, {"Query": {"people": people_plan}, "Person": {"name": name_plan}}
    )
    ctx, _object_plan = build_plan(schema, "{ people { name } }")

    plan = ctx._grafast_plan
    # root + people access + name access — all consumed, none orphaned.
    assert any(isinstance(s, RootStep) for s in plan.steps)
    assert sum(isinstance(s, AccessStep) for s in plan.steps) == 2
    # tree-shake kept everything: the operation's plan still has all its steps.
    assert len(plan.steps) == 3


def test_collect_consumption_roots_includes_nested_child_plan_steps():
    """The consumption surface spans the whole tree, incl. nested object child plans."""

    def people_plan(parent, args, info):
        return load_many(get(parent, "people_ids"), lambda keys: [[] for _ in keys])

    def name_plan(parent, args, info):
        return get(parent, "name")

    schema = make_grafast_schema(
        SDL, {"Query": {"people": people_plan}, "Person": {"name": name_plan}}
    )
    _ctx, object_plan = build_plan(schema, "{ people { name } }")

    roots = collect_consumption_root_steps(object_plan)
    root_ids = {s.id for s in roots}

    people_fp = object_plan.fields[0]
    child = find_object_completer(people_fp.completer)
    name_fp = child.child_plan.fields[0]

    # the field steps at BOTH levels are consumption roots ...
    assert people_fp.step.id in root_ids
    assert name_fp.step.id in root_ids
    # ... and so is the root parent_step (the bucket boundary the executor seeds).
    assert object_plan.layer.parent_step.id in root_ids


def test_finalize_plan_is_idempotent_when_reapplied():
    """A second finalize over an already-finalized plan/ObjectPlan changes nothing."""

    def people_plan(parent, args, info):
        return get(parent, "people")

    schema = make_grafast_schema(SDL, {"Query": {"people": people_plan}})
    ctx, object_plan = build_plan(schema, "{ people { name } }")

    plan = ctx._grafast_plan
    steps_before = list(plan.steps)
    again = finalize_plan(plan, object_plan)

    assert plan.steps == steps_before
    assert again == object_plan
