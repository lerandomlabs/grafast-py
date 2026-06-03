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
from .dag import Plan
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
    live parents and runs this bucket's plan-field steps from there. `step_ids` are
    those plan-field step ids — kept so abstract / re-planned child buckets that share
    no plan steps stay a no-op.
    """

    parent_type: GraphQLObjectType
    fields: List[FieldPlan]
    parent_step: Optional[Step] = None


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
    root_step = RootStep()
    plan.add_step(root_step)

    object_plan = plan_object(
        context, root_type, root_fields, parent_step=root_step, plan=plan
    )

    remap = plan.deduplicate()
    object_plan = remap_object_plan(object_plan, remap)

    context._grafast_plan = plan
    context._grafast_root_step = root_step
    return object_plan


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
