"""End-to-end gate for the Wave 3a optimizer substrate through the real pipeline.

`test_dag_optimize.py` exercises `Plan.optimize`/`tree_shake`/`dependents_of` in
isolation and `test_plan_finalize.py` checks the shared `finalize_plan` helper. This
file is the INTEGRATION oracle: it drives the substrate through the genuine
plan-then-execute path (`plan_operation` → `finalize_plan` → the executor), where a
Step subclass that overrides `optimize` rewrites a real operation's DAG and the
EXECUTED result is what gets asserted.

The defining property gated here is the NO-OP SAFETY INVARIANT: with the shipped
default identity `Step.optimize`, the optimize pass + tree-shake leave the executed
result byte-identical (same `data`, same `errors`) — the same oracle the graphql-core
conformance suite enforces at scale, asserted here at the unit level. A TOY optimizer
(installed as a real plan resolver) then proves the hook + tree-shake actually rewrite
THROUGH finalize while keeping execution correct, and a side-effecting (`dedupable=False`)
step proves the preservation rule: an unconsumed mutation is never shaken out.
"""

from typing import Any, Dict, List

import pytest
from graphql import (
    GraphQLField,
    GraphQLInt,
    GraphQLObjectType,
    GraphQLSchema,
    GraphQLString,
    execute,
    graphql,
    parse,
)
from graphql.execution.collect_fields import collect_fields
from graphql.execution.execute import ExecutionContext

from grafast_py import GrafastExecutionContext, constant, get, make_grafast_schema
from grafast_py.core_steps import ConstantStep, RootStep
from grafast_py.dag import Plan, order_steps
from grafast_py.plan import collect_consumption_root_steps, plan_operation
from grafast_py.step_model import Step

# --------------------------------------------------------------------------- data
# A tiny static graph: a single `config` object whose leaves are served by plan
# resolvers. No database — the whole point is that the optimizer substrate is a pure
# step-DAG transform, observable end-to-end without any IO.
CONFIG: Dict[str, Any] = {"name": "grafast", "answer": 42}


# ----------------------------------------------------- a TOY optimizing step + helper
class FoldableConstStep(Step):
    """A toy 1-dependency passthrough whose `optimize` folds a constant dependency.

    Stands in for the future query-inlining optimizer: when its single dependency is
    a `ConstantStep`, it ABSORBS it by returning a fresh `ConstantStep` of the same
    value, which orphans the folded constant so tree-shake drops it. Over a
    non-constant dependency it is identity (returns `self`), so it is a faithful
    passthrough on the legacy path. `execute` mirrors the dependency's column so the
    unfolded form is itself correct end-to-end.
    """

    is_sync_and_safe = True

    def __init__(self, source: Step) -> None:
        super().__init__()
        self.add_dependency(source)

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        return list(values[0])

    def optimize(self, plan) -> Step:
        source = self.dependencies[0]
        if isinstance(source, ConstantStep):
            return ConstantStep(source.data)
        return self


def fold_const(value: Any):
    """Plan-resolver factory: a FoldableConstStep over a constant of `value`."""

    def plan_fn(parent, args, info):
        return FoldableConstStep(constant(value))

    return plan_fn


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


# =========================================================================
# 1. NO-OP SAFETY: default identity optimize leaves the EXECUTED result
#    byte-identical to the plain graphql-core path.
# =========================================================================

CONFIG_SDL = """
type Query {
  config: Config
}
type Config {
  name: String
  answer: Int
}
"""


def make_plain_schema() -> GraphQLSchema:
    """The same graph with ordinary `resolve=` functions (the no-op oracle)."""
    config_type = GraphQLObjectType(
        "Config",
        {
            "name": GraphQLField(GraphQLString, resolve=lambda obj, info: obj["name"]),
            "answer": GraphQLField(GraphQLInt, resolve=lambda obj, info: obj["answer"]),
        },
    )
    query_type = GraphQLObjectType(
        "Query",
        {"config": GraphQLField(config_type, resolve=lambda obj, info: CONFIG)},
    )
    return GraphQLSchema(query=query_type)


def make_plan_schema() -> GraphQLSchema:
    """The same graph where every field carries a plan resolver (the substrate path)."""

    def config_plan(parent, args, info):
        return constant(CONFIG)

    def name_plan(parent, args, info):
        return get(parent, "name")

    def answer_plan(parent, args, info):
        return get(parent, "answer")

    return make_grafast_schema(
        CONFIG_SDL,
        {
            "Query": {"config": config_plan},
            "Config": {"name": name_plan, "answer": answer_plan},
        },
    )


def test_default_optimize_keeps_executed_result_byte_identical():
    """With no `optimize` override, the plan path's data+errors match plain core."""
    plain_schema = make_plain_schema()
    plan_schema = make_plan_schema()
    query = "{ config { name answer } }"

    plain = execute(plain_schema, parse(query), execution_context_class=ExecutionContext)
    plan = execute(plan_schema, parse(query), execution_context_class=GrafastExecutionContext)

    assert plain.errors is None
    assert plan.errors is None
    assert plan.data == plain.data == {"config": {"name": "grafast", "answer": 42}}


def test_default_finalize_shakes_out_nothing():
    """finalize_plan over a real default-optimize operation keeps every planned step."""
    plan_schema = make_plan_schema()
    ctx, _object_plan = build_plan(plan_schema, "{ config { name answer } }")

    plan = ctx._grafast_plan
    # root + config constant + name access + answer access — all consumed, none orphaned.
    assert len(plan.steps) == 4
    assert any(isinstance(s, RootStep) for s in plan.steps)
    assert sum(isinstance(s, ConstantStep) for s in plan.steps) == 1


# =========================================================================
# 2. A TOY optimizer rewriting THROUGH finalize_plan + the executor.
# =========================================================================


def make_folding_schema() -> GraphQLSchema:
    """A schema whose `answer` field plan is a FoldableConstStep over a constant.

    The fold collapses that step to a bare ConstantStep at finalize time; we assert
    both the structural rewrite AND that the field still executes to the constant.
    """
    sdl = """
    type Query {
      answer: Int
    }
    """
    return make_grafast_schema(sdl, {"Query": {"answer": fold_const(7)}})


def test_toy_optimizer_folds_step_through_finalize():
    """The FoldableConstStep is rewritten to a ConstantStep by finalize_plan's optimize."""
    schema = make_folding_schema()
    _ctx, object_plan = build_plan(schema, "{ answer }")

    answer_fp = object_plan.fields[0]
    # the FieldPlan.step the executor consumes is the FOLDED survivor, not the toy step.
    assert isinstance(answer_fp.step, ConstantStep)
    assert answer_fp.step.data == 7


def test_toy_optimizer_orphan_is_tree_shaken_through_finalize():
    """The constant the fold absorbed is orphaned, so finalize's tree-shake drops it.

    Before folding the plan holds {RootStep, ConstantStep(7) source, FoldableConstStep}.
    The fold replaces the FoldableConstStep with a NEW ConstantStep(7) that does not
    depend on the source, orphaning it; tree-shake removes the orphan but keeps the
    consumed survivor. RootStep stays (a finalized ObjectPlan.parent_step is a
    consumption root even when no field step depends on it).
    """
    schema = make_folding_schema()
    ctx, object_plan = build_plan(schema, "{ answer }")
    plan = ctx._grafast_plan

    survivor = object_plan.fields[0].step
    constant_steps = [s for s in plan.steps if isinstance(s, ConstantStep)]
    # exactly ONE constant survives — the fold survivor; the absorbed source was shaken.
    assert constant_steps == [survivor]
    assert not any(isinstance(s, FoldableConstStep) for s in plan.steps)
    # the survivor is still in the plan and reachable as a consumption root.
    assert survivor.id in {s.id for s in collect_consumption_root_steps(object_plan)}


def test_toy_optimizer_folded_field_still_executes_correctly():
    """End-to-end: the folded field returns the constant value through the executor."""
    schema = make_folding_schema()
    result = execute(schema, parse("{ answer }"), execution_context_class=GrafastExecutionContext)

    assert result.errors is None
    assert result.data == {"answer": 7}


def test_toy_optimizer_is_identity_over_non_constant_dependency():
    """Over a non-constant dependency the toy `optimize` keeps the step (no fold)."""

    def passthrough_plan(parent, args, info):
        # source is a RootStep access, NOT a ConstantStep → optimize() returns self.
        return FoldableConstStep(get(parent, "name"))

    sdl = """
    type Query {
      name: String
    }
    """
    schema = make_grafast_schema(sdl, {"Query": {"name": passthrough_plan}})
    _ctx, object_plan = build_plan(schema, "{ name }")

    # not folded: the FieldPlan.step is still the toy passthrough step.
    assert isinstance(object_plan.fields[0].step, FoldableConstStep)


# =========================================================================
# 3. dedupable=False preservation: an unconsumed side-effecting step is
#    NEVER tree-shaken (it runs for effect), even through finalize_plan.
# =========================================================================


# A toy mutation-marked step orphaned by inlining. `execute` records a REAL side effect
# (appends to `writes`) and returns a constant; an optimizer absorbs its value, leaving it
# unconsumed. It is defined at module scope (not nested) so the same class — and its
# `optimize`/`execute` — is reused by every test below, and `writes` is reset per test.
writes: List[int] = []


class RecordingWriteStep(ConstantStep):
    """A toy mutation: side-effecting (`dedupable=False`), records each execute as a write."""

    dedupable = False

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        writes.append(self.data)  # the WRITE: observable only if execute actually runs
        return super().execute(count, values)


class InlineAbsorbingStep(Step):
    """Folds to a fresh ConstantStep, orphaning its side-effecting source dependency.

    Models the future query-inlining optimizer: it inlines the source's value into a bare
    constant and drops the dependency, so the side-effecting source is left unconsumed.
    """

    is_sync_and_safe = True

    def __init__(self, source: Step) -> None:
        super().__init__()
        self.add_dependency(source)

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        return list(values[0])

    def optimize(self, plan) -> Step:
        return ConstantStep(self.dependencies[0].data)


def build_inlined_write_schema() -> GraphQLSchema:
    """A schema whose `value` field inlines a side-effecting write, orphaning it."""

    def plan_fn(parent, args, info):
        return InlineAbsorbingStep(RecordingWriteStep(3))

    sdl = """
    type Query {
      value: Int
    }
    """
    return make_grafast_schema(sdl, {"Query": {"value": plan_fn}})


def test_finalize_never_shakes_unconsumed_side_effecting_step():
    """A `dedupable=False` step orphaned by an optimizer is force-kept by tree-shake.

    The optimizer folds the field's value step to a constant (orphaning the side-effecting
    source), but the source carries `dedupable=False` — the mutation marker. Such a step
    runs FOR EFFECT and may be unconsumed (its return value is not selected), so tree-shake
    must keep it (and its transitive deps) regardless, AND attach it to a bucket so the
    executor still runs it. Structural survival alone is vacuous — the behavioural proof is
    `test_finalize_runs_unconsumed_side_effecting_step` below.
    """
    schema = build_inlined_write_schema()
    ctx, object_plan = build_plan(schema, "{ value }")
    plan = ctx._grafast_plan

    survivor = object_plan.fields[0].step
    assert isinstance(survivor, ConstantStep)
    assert survivor.data == 3
    assert not isinstance(survivor, RecordingWriteStep)  # the survivor is NOT the write

    surviving_ids = {s.id for s in plan.steps}
    source = next(s for s in plan.steps if isinstance(s, RecordingWriteStep))
    # the side-effecting source is unconsumed (the folded survivor does NOT depend on
    # it) yet force-kept — the defining preservation invariant of tree-shake.
    assert source.id in surviving_ids
    assert source not in collect_consumption_root_steps(object_plan)
    # ... and force-kept STRUCTURALLY is not enough: it must be attached to a bucket as an
    # effect step so the executor has a run target for it (issue #1's fix).
    assert any(source in op_effect for op_effect in _all_effect_steps(object_plan))


def _all_effect_steps(object_plan) -> List[List[Step]]:
    """Every `effect_steps` list across the (nested) ObjectPlan tree."""
    from grafast_py.completion import find_object_completer

    out: List[List[Step]] = [object_plan.effect_steps]
    for fp in object_plan.fields:
        child = find_object_completer(fp.completer)
        if child is not None and child.child_plan is not None:
            out.extend(_all_effect_steps(child.child_plan))
    return out


def test_finalize_runs_unconsumed_side_effecting_step():
    """BEHAVIOURAL gate: the orphaned write actually EXECUTES through the executor.

    The optimizer inlines the write's value so no field consumes it; the executor must
    still run it for effect (`writes` gains an entry). This is the faithful preservation
    proof: structural survival in `plan.steps` is vacuous unless the executor has a path to
    run the kept step. The data still returns the inlined constant.
    """
    writes.clear()
    schema = build_inlined_write_schema()
    result = execute(schema, parse("{ value }"), execution_context_class=GrafastExecutionContext)

    assert result.errors is None
    assert result.data == {"value": 3}
    assert writes == [3], (
        "the orphaned side-effecting step did not run for effect — its write was "
        f"silently lost (writes={writes!r})"
    )


def test_reverting_force_keep_makes_the_write_disappear():
    """The litmus: reverting tree-shake's force-keep changes observable writes.

    With the side-effecting orphan NOT force-kept (and not attached as an effect step), the
    executor has no run target for it, so its write never happens — `writes` stays empty
    while the inlined constant is still returned. This is exactly the silent data-loss the
    force-keep + effect-step wiring prevents: the side effect is genuinely load-bearing,
    not observationally dead scaffolding.
    """
    writes.clear()
    schema = build_inlined_write_schema()

    # revert the force-keep: tree-shake drops side-effecting orphans and attaches nothing.
    def no_force_keep(self, consumption_roots):
        reachable = {s.id for s in order_steps(consumption_roots)}
        self.steps = [s for s in self.steps if s.id in reachable]
        return []

    original = Plan.tree_shake
    Plan.tree_shake = no_force_keep
    try:
        result = execute(
            schema, parse("{ value }"), execution_context_class=GrafastExecutionContext
        )
    finally:
        Plan.tree_shake = original

    assert result.errors is None
    assert result.data == {"value": 3}  # the inlined constant is unchanged ...
    assert writes == [], (
        "reverting the force-keep left the write running — the force-keep is then "
        "observationally dead, not load-bearing"
    )


# =========================================================================
# 4. The MUTATION serial path runs an orphaned write for effect (the canonical
#    side-effecting case), both when the write is sync and when it is async.
# =========================================================================


def build_inlined_mutation_schema(write_step_factory) -> GraphQLSchema:
    """A Mutation whose `apply` field inlines a side-effecting write, orphaning it.

    `write_step_factory()` builds the orphaned side-effecting step (sync or async); the
    field's plan wraps it in an InlineAbsorbingStep so an optimizer drops the dependency.
    """

    def plan_fn(parent, args, info):
        return InlineAbsorbingStep(write_step_factory())

    sdl = """
    type Query {
      ping: Int
    }
    type Mutation {
      apply: Int
    }
    """
    return make_grafast_schema(
        sdl,
        {"Query": {"ping": lambda p, a, i: constant(1)}, "Mutation": {"apply": plan_fn}},
    )


def test_mutation_serial_path_runs_orphaned_sync_write_for_effect():
    """A mutation whose return value was inlined still WRITES through the serial path."""
    writes.clear()
    schema = build_inlined_mutation_schema(lambda: RecordingWriteStep(5))
    result = execute(
        schema, parse("mutation { apply }"), execution_context_class=GrafastExecutionContext
    )

    assert result.errors is None
    assert result.data == {"apply": 5}
    assert writes == [5], "the orphaned mutation write was silently dropped on the serial path"


class AsyncRecordingWriteStep(Step):
    """A side-effecting step whose execute returns a coroutine (an async write).

    Models the real pg-mutation shape: the write happens inside an awaited coroutine, so
    the serial path must await the effect run before completing fields. `value` is read by
    `AsyncInlineAbsorbingStep.optimize` when it inlines (and orphans) this step.
    """

    dedupable = False
    is_sync_and_safe = False

    def __init__(self, value: int) -> None:
        super().__init__()
        self.value = value

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        async def run():
            writes.append(self.value)  # the async WRITE
            return [self.value] * count

        return run()


class AsyncInlineAbsorbingStep(InlineAbsorbingStep):
    """Inlines an `AsyncRecordingWriteStep`'s value, orphaning the async write."""

    def optimize(self, plan) -> Step:
        return ConstantStep(self.dependencies[0].value)


@pytest.mark.asyncio
async def test_mutation_serial_path_runs_orphaned_async_write_for_effect():
    """An ASYNC orphaned mutation (the real pg shape) is awaited for effect, serially."""
    writes.clear()

    def plan_fn(parent, args, info):
        return AsyncInlineAbsorbingStep(AsyncRecordingWriteStep(8))

    schema = make_grafast_schema(
        """
        type Query { ping: Int }
        type Mutation { apply: Int }
        """,
        {
            "Query": {"ping": lambda p, a, i: constant(1)},
            "Mutation": {"apply": plan_fn},
        },
    )
    result = await graphql(
        schema, "mutation { apply }", execution_context_class=GrafastExecutionContext
    )

    assert result.errors is None
    assert result.data == {"apply": 8}
    assert writes == [8], "the orphaned ASYNC mutation write was silently dropped"


# =========================================================================
# 5. An orphaned write in a NESTED object bucket runs for effect — the
#    per-bucket effect-step attribution, not just the root bucket.
# =========================================================================


class ParentKeyedWriteStep(Step):
    """A side-effecting step keyed off its bucket's parent column (a realistic write).

    One dependency: the bucket's parent access. `execute` records the parent values it
    saw, proving it ran in the right (nested) bucket over the right parents. Its single
    value is inlined (and the step orphaned) by `InlineAbsorbingStep`.
    """

    dedupable = False
    is_sync_and_safe = True

    def __init__(self, source: Step, marker: str) -> None:
        super().__init__()
        self.marker = marker
        self.add_dependency(source)

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        writes.append((self.marker, list(values[0])))  # the WRITE, with parent context
        return list(values[0])

    @property
    def data(self):
        # InlineAbsorbingStep.optimize folds `self.dependencies[0].data`; expose a stable
        # inlined value so the field still returns a constant.
        return self.marker


def test_orphaned_write_in_nested_bucket_runs_for_effect():
    """A write orphaned inside a NESTED object bucket still executes there for effect.

    `inner.tag`'s plan inlines a `ParentKeyedWriteStep` that keys off the `inner` object's
    own bucket parent (the `outer` row), orphaning the write. finalize must attach it to
    the DEEPEST bucket (the inner one), and `run_bucket_steps` must run it there — over the
    inner bucket's parents, not the root's. The data still returns the inlined constant.
    """
    writes.clear()

    def outer_plan(parent, args, info):
        return constant({"label": "row-A"})

    def tag_plan(parent, args, info):
        # parent is the inner bucket's parent step (the `outer` object row).
        return InlineAbsorbingStep(ParentKeyedWriteStep(get(parent, "label"), "nested"))

    sdl = """
    type Query { outer: Outer }
    type Outer { tag: String }
    """
    schema = make_grafast_schema(
        sdl, {"Query": {"outer": outer_plan}, "Outer": {"tag": tag_plan}}
    )
    result = execute(
        schema, parse("{ outer { tag } }"), execution_context_class=GrafastExecutionContext
    )

    assert result.errors is None
    assert result.data == {"outer": {"tag": "nested"}}
    # the write ran in the nested bucket, over the `outer` row's `label` column.
    assert writes == [("nested", ["row-A"])], (
        f"the orphaned nested-bucket write did not run over its bucket parents: {writes!r}"
    )
