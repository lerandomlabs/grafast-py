"""The planner: walk an operation selection set into an OutputPlan tree.

The OutputPlan describes, statically, how to turn a parent object into output for
one selection set: a list of field plans, each carrying the field's resolver,
arguments (coerced once at plan time), return type, and a **completer** — the
pre-computed wrapping-type descent (NonNull/List/leaf/object/abstract) the executor
drives over a batch of resolved values.

The planner builds a genuine native plan for EVERY GraphQL output shape: leaf
(scalar/enum), object, abstract (interface/union), and arbitrarily nested
List/NonNull wrappers of those. There is no deferral path — the engine is an
unconditional drop-in. Fields without a plan resolver still take the per-parent
resolver-adapter (`ResolveStep`) route, which the planner wires into the same step
DAG so plain-resolver schemas run unchanged.
"""

from typing import Any, Callable, Dict, List, NamedTuple, Optional

from graphql.execution.execute import get_field_def
from graphql.execution.values import get_argument_values
from graphql.language import FieldNode, OperationDefinitionNode
from graphql.pyutils import Path
from graphql.type import GraphQLField, GraphQLObjectType, GraphQLOutputType

from .completion import (
    Completer,
    build_completer,
    find_object_completer,
)
from .dag import Plan, _compose_remaps, order_steps
from .schema import FieldArgs, get_field_plan
from .step_model import Step


class FieldPlan(NamedTuple):
    """A single field within an object selection set.

    `response_name` is the alias-or-name key emitted in `data`. `args` are coerced
    eagerly; coercion errors are deferred to execution so they surface as located
    field errors (matching graphql-core's in-resolve-block coercion). `completer`
    is the wrapping-type completion tree for `return_type` (always present — every
    output-type kind has a native completer). `field_label` is "ParentType.fieldName"
    for synthetic non-null / iterable error messages.
    """

    response_name: str
    field_nodes: List[FieldNode]
    field_def: GraphQLField
    parent_type: GraphQLObjectType
    return_type: GraphQLOutputType
    completer: Completer
    field_label: str
    args: Optional[Dict[str, Any]]
    args_error: Optional[Exception]
    # plan-resolver path: `plan_fn` is the field's plan resolver (or None for the
    # legacy resolver-adapter path); `step` is the step it produced, registered in
    # the operation's Plan. When `plan_fn` is None, `step` is None and the field
    # takes the existing ResolveStep path so conformance stays green.
    plan_fn: Optional[Callable] = None
    step: Optional[Step] = None


class ObjectPlan(NamedTuple):
    """An output plan for one object selection set.

    `parent_step` is the step whose per-bucket output column IS this bucket's parent
    objects (the operation `RootStep` at the root; an enclosing plan field's step for
    a nested object; `None` for a bucket reached purely under legacy resolver fields,
    where no plan-resolver steps run). The executor seeds `parent_step.id` with the
    live parents and runs this bucket's plan-field steps from there.

    `effect_steps` are side-effecting steps (`dedupable=False`, e.g. a pg mutation)
    that this bucket must run FOR EFFECT even though no `FieldPlan.step` consumes
    them — the case where an optimizer absorbed a mutation's return value (inlined it)
    and orphaned the write. tree-shake force-keeps such a step in the plan DAG, but the
    executor only runs steps reachable from `fields`, so the orphan needs a run target;
    `effect_steps` IS that target. With the default identity optimize nothing is ever
    orphaned, so this list is always empty and the executor path is byte-identical.
    """

    parent_type: GraphQLObjectType
    fields: List[FieldPlan]
    parent_step: Optional[Step] = None
    effect_steps: List[Step] = []


def plan_object(
    context,
    parent_type: GraphQLObjectType,
    fields: Dict[str, List[FieldNode]],
    parent_step: Optional[Step] = None,
    plan: Optional[Plan] = None,
) -> ObjectPlan:
    """Plan one object selection set into an ObjectPlan.

    `fields` is the already-`collect_fields`-filtered response map, so @skip /
    @include and fragment conditions are honoured before planning.

    `parent_step` is the step whose output is the bucket of parents for this object
    (the root value's step at the operation root; an enclosing plan field's step for
    a nested object). `plan` is the operation's step DAG; both are threaded so that a
    field WITH a plan resolver builds a genuine step depending on `parent_step` and
    passes that step down as `$parent` for its sub-selection. Fields WITHOUT a plan
    resolver ignore `parent_step` and keep the legacy resolver-adapter path; their
    object children inherit `parent_step` unchanged so a plan-resolver DESCENDANT
    under a legacy parent still has a sensible source step.
    """
    field_plans: List[FieldPlan] = []
    for response_name, field_nodes in fields.items():
        field_def = get_field_def(context.schema, parent_type, field_nodes[0])
        if not field_def:
            # unknown field — dropped from output, like execute_field's Undefined
            continue

        return_type = field_def.type

        args: Optional[Dict[str, Any]] = None
        args_error: Optional[Exception] = None
        try:
            args = get_argument_values(
                field_def, field_nodes[0], context.variable_values
            )
        except Exception as raw_error:  # coercion error → located at exec time
            args_error = raw_error

        plan_fn = get_field_plan(field_def)
        field_step: Optional[Step] = None
        if plan_fn is not None and plan is not None and args_error is None:
            info = context.build_resolve_info(
                field_def,
                field_nodes,
                parent_type,
                Path(None, response_name, parent_type.name),
            )
            field_step = plan_fn(parent_step, FieldArgs(args), info)
            plan.add_step(field_step)

        # an object/plan field passes its step down as the child bucket's parent;
        # a legacy field (or a plan field over leaves) passes the inherited step.
        child_parent_step = field_step if field_step is not None else parent_step

        completer = build_completer(context, return_type, field_nodes)
        object_completer = find_object_completer(completer)
        if object_completer is not None:
            sub_fields = context.collect_subfields(
                object_completer.object_type, field_nodes
            )
            child_plan = plan_object(
                context,
                object_completer.object_type,
                sub_fields,
                parent_step=child_parent_step,
                plan=plan,
            )
            completer = attach_child_plan(completer, child_plan)

        field_plans.append(
            FieldPlan(
                response_name=response_name,
                field_nodes=field_nodes,
                field_def=field_def,
                parent_type=parent_type,
                return_type=return_type,
                completer=completer,
                field_label=f"{parent_type.name}.{field_nodes[0].name.value}",
                args=args,
                args_error=args_error,
                plan_fn=plan_fn,
                step=field_step,
            )
        )
    return ObjectPlan(
        parent_type=parent_type, fields=field_plans, parent_step=parent_step
    )


def attach_child_plan(completer: Completer, child_plan: ObjectPlan) -> Completer:
    """Rebuild a completer chain with the leaf ObjectCompleter's child plan set."""
    from .completion import ListCompleter, NonNullCompleter, ObjectCompleter

    if isinstance(completer, ObjectCompleter):
        return completer._replace(child_plan=child_plan)
    if isinstance(completer, NonNullCompleter):
        return completer._replace(inner=attach_child_plan(completer.inner, child_plan))
    if isinstance(completer, ListCompleter):
        return completer._replace(
            item_completer=attach_child_plan(completer.item_completer, child_plan)
        )
    return completer


def plan_operation(context, operation: OperationDefinitionNode, root_type, root_fields):
    """Build the top-level ObjectPlan for an operation's root selection set.

    Also builds the operation's step DAG (`Plan`): a `RootStep` seeds the root
    value as the parent bucket, and every plan-resolver field hangs its step off it
    (recursively). After the tree is planned the DAG is deduplicated and the
    surviving `Plan` plus the `RootStep` are stashed on the context for the executor
    (`context._grafast_plan` / `context._grafast_root_step`). Fields without a plan
    resolver contribute no steps, so for the conformance suite the DAG is empty and
    the legacy path is entirely unaffected.
    """
    from .core_steps import RootStep

    plan = Plan()
    # thread the plan-level inlining decision off the context's config so each pg
    # step's `optimize(self, plan)` reads one constant (`plan.inline_relations`)
    # instead of the whole context. Default-OFF config => no-op optimize pass.
    plan.inline_relations = type(context).grafast_config.inline_relations
    root_step = RootStep()
    plan.add_step(root_step)

    object_plan = plan_object(
        context, root_type, root_fields, parent_step=root_step, plan=plan
    )

    object_plan = finalize_plan(plan, object_plan)

    context._grafast_plan = plan
    context._grafast_root_step = root_step
    return object_plan


def finalize_plan(plan: Plan, object_plan: ObjectPlan) -> ObjectPlan:
    """Optimize → dedup → tree-shake the plan, then remap the ObjectPlan to survivors.

    The single finalize path shared by the operation root (`plan_operation`) and every
    abstract/object child subtree (`completion.abstract_child_plan`), so abstract
    subtrees optimize identically to the operation root.

    Order, with reasons:
      * `optimize` FIRST — its rewrites may produce structurally identical steps, which
        dedup then merges; running dedup first would not see them.
      * `deduplicate` SECOND — operates on the post-rewrite DAG (it fixpoints internally).
      * `tree_shake` LAST — an absorbed dependent can leave its old dependency unconsumed;
        only after optimize+dedup is the consumed set final, so shaking last drops exactly
        the now-orphaned, non-side-effecting steps and nothing a later pass still needed.

    The two remaps are composed so the ObjectPlan tree is rewritten to its FINAL survivor
    in one `remap_object_plan` pass. Consumption roots are collected from the *remapped*
    tree (post-survivor) so tree-shake measures reachability against the steps the
    executor will actually consume.

    With the shipped default identity `Step.optimize`, `optimize` returns an empty remap,
    `deduplicate` behaves exactly as before, every step stays reachable from a
    `FieldPlan.step`, and tree-shake keeps everything — a byte-identical no-op.
    """
    opt_remap = plan.optimize()
    dedup_remap = plan.deduplicate()
    remap = _compose_remaps(opt_remap, dedup_remap)
    object_plan = remap_object_plan(object_plan, remap)
    roots = collect_consumption_root_steps(object_plan)
    orphaned_effects = plan.tree_shake(roots)
    if orphaned_effects:
        object_plan = attach_effect_steps(object_plan, orphaned_effects)
    return object_plan


def collect_consumption_root_steps(object_plan: ObjectPlan) -> List[Step]:
    """Collect every step the executor consumes from a finalized ObjectPlan tree.

    The executor derives each bucket's run targets from `fp.step for fp in plan.fields`
    and seeds the bucket's `parent_step`; the union of all (transitively nested) buckets'
    targets and boundaries IS the plan's consumption surface. This walks the tree exactly
    as `remap_object_plan` does — recursing into child plans via `find_object_completer`
    — and returns the consumed steps (deduplicated by id) for `Plan.tree_shake`.
    """
    seen: Dict[int, Step] = {}

    def visit(op: ObjectPlan) -> None:
        if op.parent_step is not None:
            seen[op.parent_step.id] = op.parent_step
        for fp in op.fields:
            if fp.step is not None:
                seen[fp.step.id] = fp.step
            child = find_object_completer(fp.completer)
            if child is not None and child.child_plan is not None:
                visit(child.child_plan)

    visit(object_plan)
    return list(seen.values())


def attach_effect_steps(
    object_plan: ObjectPlan, orphaned_effects: List[Step]
) -> ObjectPlan:
    """Attach each orphaned side-effecting step to the bucket that must RUN it.

    An optimizer that inlines a mutation's return value leaves the write `dedupable=False`
    step in the plan DAG but unconsumed by any `FieldPlan.step`; tree-shake force-keeps it
    (a write runs for effect), yet the executor only runs steps reachable from a bucket's
    fields, so the orphan needs an explicit run target. Each such step descends from
    exactly one bucket's `parent_step` (the planner built it from that bucket's parent),
    so its owner is the DEEPEST ObjectPlan whose `parent_step` is in the step's transitive
    dependency set — the same bucket whose boundary the executor seeds. We rebuild that
    ObjectPlan with the step appended to `effect_steps`; the executor then runs it for
    effect alongside the bucket's consumed fields.

    With the default identity optimize nothing is ever orphaned, so `orphaned_effects`
    is empty and this function is never called — a no-op for the conformance path.
    """
    owner_id_for: Dict[int, int] = {}
    for effect in orphaned_effects:
        boundary_ids = {dep.id for dep in order_steps([effect])}
        owner_id_for[effect.id] = _deepest_owner_parent_id(object_plan, boundary_ids)

    def rebuild(op: ObjectPlan, depth: int) -> ObjectPlan:
        new_fields: List[FieldPlan] = []
        for fp in op.fields:
            new_completer = fp.completer
            child = find_object_completer(new_completer)
            if child is not None and child.child_plan is not None:
                rebuilt_child = rebuild(child.child_plan, depth + 1)
                new_completer = attach_child_plan(new_completer, rebuilt_child)
            new_fields.append(fp._replace(completer=new_completer))
        mine = [
            effect
            for effect in orphaned_effects
            if op.parent_step is not None
            and owner_id_for[effect.id] == op.parent_step.id
        ]
        return op._replace(
            fields=new_fields, effect_steps=[*op.effect_steps, *mine] if mine else op.effect_steps
        )

    return rebuild(object_plan, 0)


def _deepest_owner_parent_id(object_plan: ObjectPlan, boundary_ids: set) -> int:
    """Find the parent_step id of the deepest bucket the orphaned step descends from.

    Walks the ObjectPlan tree; a bucket owns the step when its `parent_step` is among the
    step's transitive dependency ids (`boundary_ids`). Nested buckets all qualify (a child's
    parent_step descends from its parent's), so the DEEPEST match wins — that is the bucket
    whose boundary the executor seeds with the parents the write actually keys off.

    A step that depends on NO bucket boundary (a 0-dependency write, or one keyed only off
    constants) has no parent column to key off; it belongs to the ROOT bucket, where it runs
    once over the operation's single root entry. We fall back to the root `parent_step` for
    that case.
    """
    best_id = -1
    best_depth = -1

    def visit(op: ObjectPlan, depth: int) -> None:
        nonlocal best_id, best_depth
        if (
            op.parent_step is not None
            and op.parent_step.id in boundary_ids
            and depth > best_depth
        ):
            best_id = op.parent_step.id
            best_depth = depth
        for fp in op.fields:
            child = find_object_completer(fp.completer)
            if child is not None and child.child_plan is not None:
                visit(child.child_plan, depth + 1)

    visit(object_plan, 0)
    if best_id >= 0:
        return best_id
    if object_plan.parent_step is not None:
        return object_plan.parent_step.id  # boundary-less write → the root bucket runs it
    raise AssertionError(
        "orphaned side-effecting step has no bucket to run it for effect "
        "(the root ObjectPlan has no parent_step)"
    )


def remap_object_plan(object_plan: ObjectPlan, remap: Dict[int, Step]) -> ObjectPlan:
    """Rewrite every `FieldPlan.step` to its dedup survivor (NamedTuples are rebuilt).

    Dedup may merge two field steps into one survivor; the `FieldPlan.step`
    references the planner stored must point at the survivor so the executor reads
    the right output column. Recurses through object/list/non-null completers into
    child plans, since a child field's step is also subject to dedup.
    """
    new_fields: List[FieldPlan] = []
    for fp in object_plan.fields:
        new_step = remap.get(fp.step.id, fp.step) if fp.step is not None else None
        new_completer = fp.completer
        child = find_object_completer(new_completer)
        if child is not None and child.child_plan is not None:
            remapped_child = remap_object_plan(child.child_plan, remap)
            new_completer = attach_child_plan(new_completer, remapped_child)
        new_fields.append(fp._replace(step=new_step, completer=new_completer))
    new_parent_step = object_plan.parent_step
    if new_parent_step is not None:
        new_parent_step = remap.get(new_parent_step.id, new_parent_step)
    return object_plan._replace(fields=new_fields, parent_step=new_parent_step)
