"""The planner: walk an operation selection set into an OutputPlan tree.

The OutputPlan describes, statically, how to turn a parent object into output for
one selection set: a list of field plans, each carrying the field's resolver,
arguments (coerced once at plan time), return type, and a **completer** — the
pre-computed wrapping-type descent (NonNull/List/leaf/object/abstract) the executor
drives over a batch of resolved values.

The planner builds a genuine native plan for EVERY GraphQL output shape: leaf
(scalar/enum), object, abstract (interface/union), and arbitrarily nested
List/NonNull wrappers of those. There is no deferral path — the engine is an
unconditional drop-in. EVERY field carries a `FieldPlan.step`: a plan step for a
plan-resolver field, or a `ResolveStep` (the resolver-adapter) for a plain-resolver
field. Both live in the operation's step DAG and depend on the bucket parent_step,
so the executor reads each field's value column uniformly from the bucket store —
there is no `step is None` path and no separate per-parent resolver machine.
"""

from enum import Enum
from typing import Any, Callable, Dict, FrozenSet, List, NamedTuple, Optional, Set, Tuple

from graphql.execution.execute import get_field_def
from graphql.execution.values import get_argument_values
from graphql.language import FieldNode, OperationDefinitionNode, VariableNode
from graphql.pyutils import Path
from graphql.type import GraphQLField, GraphQLObjectType, GraphQLOutputType

from .completion import (
    Completer,
    build_completer,
    find_object_completer,
)
from .dag import Plan, _compose_remaps, order_steps, order_steps_within
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
    # `plan_fn` is the field's plan resolver (or None for the resolver-adapter path).
    # `step` is the field's value step registered in the operation's Plan — a plan step
    # when `plan_fn` is set, otherwise a `ResolveStep` adapter. EVERY field carries a
    # step now; `step is None` only in the (unreachable) no-plan-no-parent guard case.
    plan_fn: Optional[Callable] = None
    step: Optional[Step] = None


class LayerReason(Enum):
    """Why a bucket is a batch boundary (a *layer*).

    P1 enumerates only the reasons that have a real, distinct construction site in the
    planner TODAY: the operation root (and the structurally-identical abstract
    concrete-type subtree, which is its own ``RootStep``-seeded bucket), and a nested
    object field's bucket. Reasons that exist only at execution time (a mutation field
    runs the root layer serially) or only at completion time (a list item) are NOT
    distinct plan-time layers yet, so they get no member here — they arrive with the
    later phases that give them their own layer. Nothing branches on ``reason`` in P1;
    it only records what used to be implicit.
    """

    ROOT = "root"
    NESTED = "nested"


class LayerPlan(NamedTuple):
    """The reason-tagged batch boundary for one bucket (execution-only state).

    Owns what `ObjectPlan` used to carry as bucket-boundary state, split out from its
    output shape so the two can diverge in later phases (the OutputPlan/LayerPlan port).

    `parent_step` is the step whose per-bucket output column IS this bucket's parent
    objects (the operation `RootStep` at the root; an enclosing plan field's step for a
    nested object; `None` for a bucket reached purely under legacy resolver fields, where
    no plan-resolver steps run). The executor seeds `parent_step.id` with the live parents
    and runs this bucket's plan-field steps from there.

    `effect_steps` are side-effecting steps (`dedupable=False`, e.g. a pg mutation) that
    this bucket must run FOR EFFECT even though no `FieldPlan.step` consumes them — the
    case where an optimizer absorbed a mutation's return value (inlined it) and orphaned
    the write. tree-shake force-keeps such a step in the plan DAG, but the executor only
    runs steps reachable from `fields`, so the orphan needs a run target; `effect_steps`
    IS that target. With the default identity optimize nothing is ever orphaned, so this
    list is always empty and the executor path is byte-identical.

    `run_steps` are the steps this bucket runs once over its parents — the field value
    steps PLUS `effect_steps` — and `ordered_steps` is their dependency-ordered form
    (`order_steps_within(run_steps, {parent_step.id})`). Both are materialised by
    `finalize_plan` (`populate_layers`). Holding them on the layer is what DE-FUSES
    execution from serialization: `run_layer` runs a bucket from its LayerPlan ALONE and
    never reads the output shape (`fields`/completers) to decide what to run. They are
    empty/None until finalize; every executed plan is finalized, so the executor always
    sees them populated.
    """

    reason: LayerReason
    parent_step: Optional[Step] = None
    effect_steps: List[Step] = []
    run_steps: List[Step] = []
    ordered_steps: Optional[List[Step]] = None


class ObjectPlan(NamedTuple):
    """An output plan for one object selection set, paired with its batch boundary.

    `parent_type` and `fields` are the output shape; `layer` is the reason-tagged batch
    boundary (`parent_step`/`effect_steps`) the executor seeds and runs this bucket from.
    """

    parent_type: GraphQLObjectType
    fields: List[FieldPlan]
    layer: LayerPlan


def plan_object(
    context,
    parent_type: GraphQLObjectType,
    fields: Dict[str, List[FieldNode]],
    parent_step: Optional[Step] = None,
    plan: Optional[Plan] = None,
    reason: LayerReason = LayerReason.NESTED,
) -> ObjectPlan:
    """Plan one object selection set into an ObjectPlan.

    `fields` is the already-`collect_fields`-filtered response map, so @skip /
    @include and fragment conditions are honoured before planning.

    `parent_step` is the step whose output is the bucket of parents for this object
    (the root value's step at the operation root; an enclosing field's step for a
    nested object). `plan` is the operation's step DAG; both are threaded so that a
    field WITH a plan resolver builds a genuine step depending on `parent_step`, and a
    field WITHOUT one builds a `ResolveStep` adapter that ALSO depends on `parent_step`
    — every field gets a step and passes it down as `$parent` for its sub-selection.
    """
    # function-local to break the plan.py <-> steps.py import cycle (steps.py imports
    # FieldPlan from this module at module load).
    from .steps import ResolveStep

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
            # per-argument variable provenance (placeholders): walk this field's
            # argument AST so a plan resolver can tell a `$variable`-derived value from a
            # plan-time literal. Computed when `placeholders` is on OR when `cache_plans` is on
            # — caching NEEDS the provenance so an INLINED variable (read raw, not
            # placeholdered) is detected below and the plan marked non-cacheable; without it a
            # `cache_plans=True` / `placeholders=False` host would bleed the first request's
            # value across requests. Both off (the default) => empty provenance =>
            # `FieldArgs.is_variable` is always False => every host inlines literals by value.
            variable_args, variable_sources = (
                variable_provenance(field_nodes[0])
                if (plan.placeholders or plan.cache_plans)
                else (None, None)
            )
            field_args = FieldArgs(
                args, variable_args=variable_args, variable_sources=variable_sources
            )
            field_step = plan_fn(parent_step, field_args, info)
            plan.add_step(field_step)
            # cacheability: if the host INLINED a variable-derived arg's value as a plan-time
            # literal (read its raw value WITHOUT taking its placeholder source) the plan is
            # value-specific and must NOT be cached across requests. `FieldArgs` tracks the
            # raw reads and the placeholdered sources and nets them; a fully-placeholdered or
            # all-literal field leaves `cacheable` True.
            if field_args.inlined_variable_args():
                plan.cacheable = False
        else:
            # no-plan resolver field (plan_fn is None): the resolver-adapter now lives IN
            # the operation plan as a ResolveStep depending on the bucket parent_step, so
            # every field carries a FieldPlan.step and completion reads it uniformly from
            # the bucket store. (`step is None` only survives in the impossible
            # no-plan-no-parent case below, where it falls back to the inherited parent.)
            if plan is not None and parent_step is not None:
                field_step = ResolveStep(
                    field_def, parent_type, field_nodes, response_name, args, args_error
                )
                field_step.add_dependency(parent_step)
                plan.add_step(field_step)
                if plan.cache_plans and args_error is None:
                    # KEEP the legacy cacheability guard: the coerced args are FROZEN onto
                    # FieldPlan.args from this request, and a cache HIT replays them — so a
                    # resolver reading a `$variable`-derived arg would serve a later request
                    # the FIRST request's value, and a ResolveStep has no placeholder to
                    # re-point. Refuse to cache a plan carrying any variable-derived resolver
                    # arg (re-plan per request); flipping this to cacheable is deferred to P5.
                    legacy_variable_args, _ = variable_provenance(field_nodes[0])
                    if legacy_variable_args:
                        plan.cacheable = False

        # every field passes its step down as the child bucket's parent (a plan field's
        # step, or a resolver field's ResolveStep); only the impossible no-plan-no-parent
        # guard leaves `field_step` None, in which case the inherited step is passed.
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
        parent_type=parent_type,
        fields=field_plans,
        layer=LayerPlan(reason=reason, parent_step=parent_step),
    )


def variable_provenance(
    field_node: FieldNode,
) -> Tuple[FrozenSet[str], Dict[str, str]]:
    """Compute per-argument variable provenance from a field's argument AST.

    graphql-core's ``get_argument_values`` coerces a ``$variable`` argument to its
    runtime value before a plan resolver runs, so the coerced ``args`` dict alone cannot
    tell a literal from a variable. The seam is the AST: an ``ArgumentNode`` whose
    ``.value`` is a :class:`~graphql.language.VariableNode` came from a variable, and the
    variable name is ``arg.value.name.value``.

    Returns the SET of variable-derived argument names plus a mapping arg-name ->
    GraphQL-variable-name, which :class:`FieldArgs` turns into the stable ``"var:<name>"``
    source tag a placeholder dedups by. Arguments given as literals (or as a list/object
    literal) are not included, so a host inlines them by value. This is pure
    ``graphql.language`` — no execute-internals dependency.
    """
    variable_args: Set[str] = set()
    variable_sources: Dict[str, str] = {}
    for arg in field_node.arguments:
        if isinstance(arg.value, VariableNode):
            arg_name = arg.name.value
            variable_args.add(arg_name)
            variable_sources[arg_name] = arg.value.name.value
    return frozenset(variable_args), variable_sources


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
    resolver contribute no steps, so a pure plain-resolver schema yields an empty DAG
    and the legacy resolver path is entirely unaffected.

    Cross-request plan cache (opt-in)
    ---------------------------------
    When `GrafastConfig.cache_plans` is on, a finalized VALUE-INDEPENDENT plan is cached by
    `(schema identity, document text, operation name, variable fingerprint)` and reused
    across requests of the same document — a HIT skips the whole plan build, RE-BINDS each
    placeholder to THIS request's variables (`rebind_cached_plan`), and stashes the cached
    triple. A MISS plans normally and, if the result is cacheable (no variable value was
    inlined as a literal), stores it. With `cache_plans` off (the default) the cache is never
    touched, so the path below the cache block plans exactly as it would with no cache present.
    """
    from .core_steps import RootStep

    config = type(context).grafast_config
    if config.cache_plans:
        cached = lookup_cached_plan(context, operation, config)
        if cached is not None:
            return cached

    plan = Plan()
    # thread the plan-level inlining decision off the context's config so each pg
    # step's `optimize(self, plan)` reads one constant (`plan.inline_relations`)
    # instead of the whole context. Default-OFF config => no-op optimize pass.
    plan.inline_relations = config.inline_relations
    # plan-level placeholder/caching decisions, threaded off the SAME config the same
    # way: `placeholders` gates whether `plan_object` computes per-argument variable
    # provenance (and threads it into `FieldArgs`); `cache_plans` gates the cross-request
    # plan cache. Both default-OFF => no provenance computed, nothing cached, literals
    # inlined by value.
    plan.placeholders = config.placeholders
    plan.cache_plans = config.cache_plans
    root_step = RootStep()
    plan.add_step(root_step)

    object_plan = plan_object(
        context,
        root_type,
        root_fields,
        parent_step=root_step,
        plan=plan,
        reason=LayerReason.ROOT,
    )

    object_plan = finalize_plan(plan, object_plan)

    # an ABSTRACT field's per-concrete-type subtree is planned LAZILY at execute time (in
    # `completion.abstract_child_plan`), AFTER this operation plan is stored, and its steps are
    # held on the completer (NOT in `plan.steps`), so the operation-level placeholder rebind
    # never reaches them. A host that inlines a `$variable` under a concrete type would therefore
    # bake the FIRST request's value into the completer-cached subtree and a later cache HIT would
    # serve it the wrong value. The subtree's cacheability cannot be known here (it is not built
    # yet), so the operation conservatively refuses to cache when it owns ANY abstract field —
    # such an operation re-plans per request, rebuilding its subtrees fresh. (A non-abstract
    # operation is unaffected: every SQL-affecting step lives in `plan.steps` and rebinds.)
    if owns_abstract_field(object_plan):
        plan.cacheable = False

    context._grafast_plan = plan
    context._grafast_root_step = root_step

    if config.cache_plans and plan.cacheable:
        store_cached_plan(context, operation, config, object_plan, root_step, plan)

    return object_plan


def owns_abstract_field(object_plan: ObjectPlan) -> bool:
    """Whether any field in the (transitively nested) ObjectPlan tree returns an abstract type.

    Walks the completer tree of every field, recursing through List / NonNull wrappers and into
    object-field child plans, and returns True at the first :class:`AbstractCompleter` (an
    interface / union field). Used by `plan_operation` to refuse caching an operation whose
    abstract subtrees are planned lazily at execute time (their placeholder steps live on the
    completer, beyond the operation-level rebind's reach) — see the cacheability note there.
    """
    from .completion import AbstractCompleter

    def completer_has_abstract(completer: Any) -> bool:
        if isinstance(completer, AbstractCompleter):
            return True
        inner = getattr(completer, "inner", None)
        if inner is not None and completer_has_abstract(inner):
            return True
        item = getattr(completer, "item_completer", None)
        if item is not None and completer_has_abstract(item):
            return True
        return False

    def visit(op: ObjectPlan) -> bool:
        for fp in op.fields:
            if completer_has_abstract(fp.completer):
                return True
            child = find_object_completer(fp.completer)
            if child is not None and child.child_plan is not None and visit(child.child_plan):
                return True
        return False

    return visit(object_plan)


def lookup_cached_plan(context, operation: OperationDefinitionNode, config):
    """Return the cached ObjectPlan for this request (re-bound to its variables), or None.

    A cache HIT ISOLATES the shared cached entry into a per-request deep copy
    (`isolate_cached_plan`) and re-points every placeholder on that COPY to THIS request's
    variable values (the cached SQL is value-agnostic, so only the bound values move), then
    re-stashes the COPY's triple on the context for the executor. Isolating per request keeps
    the shared cache entry IMMUTABLE, so two concurrent requests of the same document with
    different variables each rebind+execute their own copy and never bleed values into each
    other — no request-level serialization needed. A MISS returns None so `plan_operation`
    plans normally. Only called when `cache_plans` is on.
    """
    from .cache import compute_cache_key, isolate_cached_plan, rebind_cached_plan

    cache = config.plan_cache if config.plan_cache is not None else _process_cache()
    key = compute_cache_key(context.schema, operation, context.fragments, config)
    cached = cache.get(key)
    if cached is None:
        return None
    if cached.schema is not context.schema:
        # a stale `id(schema)` collision (a freed schema's id reused by this one) — treat as a
        # miss so we never serve a plan built against a different schema.
        return None
    # deep-copy into a per-request plan BEFORE rebinding, so the rebind mutates this request's
    # own copy and never the shared cache entry (the concurrency-safety guarantee).
    request_plan = isolate_cached_plan(cached)
    rebind_cached_plan(request_plan, context.variable_values, operation)
    context._grafast_plan = request_plan.plan
    context._grafast_root_step = request_plan.root_step
    return request_plan.object_plan


def store_cached_plan(
    context, operation: OperationDefinitionNode, config, object_plan, root_step, plan
):
    """Store a freshly-finalized, value-independent plan under its cache key.

    Called only on a MISS when `cache_plans` is on AND `plan.cacheable` (no variable value
    was inlined as a plan-time literal). A value-specific plan is never stored — reusing it
    would serve a later request the earlier request's value.
    """
    from .cache import CachedPlan, compute_cache_key

    cache = config.plan_cache if config.plan_cache is not None else _process_cache()
    key = compute_cache_key(context.schema, operation, context.fragments, config)
    cache.put(
        key,
        CachedPlan(
            object_plan=object_plan,
            root_step=root_step,
            plan=plan,
            schema=context.schema,
        ),
    )


def _process_cache():
    """The process-global plan cache (lazy), used when the config supplies none."""
    from .cache import default_cache

    return default_cache()


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

    With the default identity `Step.optimize` (a step that does not optimize returns itself
    unchanged), `optimize` returns an empty remap, `deduplicate` sees the original DAG, every
    step stays reachable from a `FieldPlan.step`, and tree-shake keeps everything — so the
    finalized plan equals the planned one.
    """
    opt_remap = plan.optimize()
    dedup_remap = plan.deduplicate()
    remap = _compose_remaps(opt_remap, dedup_remap)
    object_plan = remap_object_plan(object_plan, remap)
    roots = collect_consumption_root_steps(object_plan)
    orphaned_effects = plan.tree_shake(roots)
    if orphaned_effects:
        object_plan = attach_effect_steps(object_plan, orphaned_effects)
    # materialise each bucket's self-contained run set onto its LayerPlan, LAST — after
    # every step is final (post optimize/dedup/tree-shake/effect-attach) — so the executor
    # runs a bucket from its layer alone, never reading the output shape.
    object_plan = populate_layers(object_plan)
    return object_plan


def populate_layers(object_plan: ObjectPlan) -> ObjectPlan:
    """Fill each bucket's `LayerPlan.run_steps`/`ordered_steps` across the OutputPlan tree.

    This is the single place the "fields → execution targets" coupling is turned into DATA
    on the layer: `run_steps` is the bucket's field value steps plus its `effect_steps`, and
    `ordered_steps` is their dependency order from the boundary — exactly what the executor
    used to derive per-bucket at run time from `plan.fields`. Computing it once here lets
    `run_layer` read the LayerPlan alone (the de-fusion). Walks the tree like
    `remap_object_plan`, rebuilding child completers so nested layers are populated too.
    """
    new_fields: List[FieldPlan] = []
    for fp in object_plan.fields:
        new_completer = fp.completer
        child = find_object_completer(new_completer)
        if child is not None and child.child_plan is not None:
            rebuilt_child = populate_layers(child.child_plan)
            new_completer = attach_child_plan(new_completer, rebuilt_child)
        new_fields.append(fp._replace(completer=new_completer))
    layer = object_plan.layer
    run_steps = [fp.step for fp in new_fields if fp.step is not None]
    run_steps.extend(layer.effect_steps)
    ordered_steps = (
        order_steps_within(run_steps, {layer.parent_step.id})
        if layer.parent_step is not None and run_steps
        else None
    )
    new_layer = layer._replace(run_steps=run_steps, ordered_steps=ordered_steps)
    return object_plan._replace(fields=new_fields, layer=new_layer)


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
        if op.layer.parent_step is not None:
            seen[op.layer.parent_step.id] = op.layer.parent_step
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
    is empty and this function is never called.
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
            if op.layer.parent_step is not None
            and owner_id_for[effect.id] == op.layer.parent_step.id
        ]
        new_layer = (
            op.layer._replace(effect_steps=[*op.layer.effect_steps, *mine])
            if mine
            else op.layer
        )
        return op._replace(fields=new_fields, layer=new_layer)

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
            op.layer.parent_step is not None
            and op.layer.parent_step.id in boundary_ids
            and depth > best_depth
        ):
            best_id = op.layer.parent_step.id
            best_depth = depth
        for fp in op.fields:
            child = find_object_completer(fp.completer)
            if child is not None and child.child_plan is not None:
                visit(child.child_plan, depth + 1)

    visit(object_plan, 0)
    if best_id >= 0:
        return best_id
    if object_plan.layer.parent_step is not None:
        # boundary-less write → the root bucket runs it
        return object_plan.layer.parent_step.id
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
    new_parent_step = object_plan.layer.parent_step
    if new_parent_step is not None:
        new_parent_step = remap.get(new_parent_step.id, new_parent_step)
    new_layer = object_plan.layer._replace(parent_step=new_parent_step)
    return object_plan._replace(fields=new_fields, layer=new_layer)
