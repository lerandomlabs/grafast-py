"""The plan-resolver API: attach and read field plan resolvers.

A field carries its plan resolver at ``field.extensions["grafast"]["plan"]``, a
callable ``plan($parent_step, field_args, info) -> Step``. ``$parent_step`` is the
PARENT step (the root value's step at the operation root; the parent field's
returned step for nested fields); ``field_args`` exposes coerced argument values
via ``field_args.get("name")`` / ``field_args["name"]``; ``info`` is graphql-core's
``GraphQLResolveInfo`` for the field. The returned step becomes the field's value
AND the ``$parent_step`` for the field's sub-selection plans — that is how the DAG
is stitched down the tree.

This module provides three attachment surfaces:

- :func:`set_field_plan` — programmatic: stuff a plan into an existing
  ``GraphQLField``'s extensions.
- :func:`get_field_plan` — the planner's reader.
- :func:`make_grafast_schema` — build a ``GraphQLSchema`` from SDL plus a
  ``{"Type": {"field": plan}}`` plan map, attaching each plan into the matching
  field's extensions. Also exposed as a :class:`GrafastSchemaBindable` for Ariadne's
  ``make_executable_schema(type_defs, bindable)`` flow.
"""

from typing import Any, Callable, Dict, FrozenSet, Mapping, Optional, Set

from graphql import (
    GraphQLAbstractType,
    GraphQLField,
    GraphQLObjectType,
    GraphQLSchema,
)
from graphql.utilities import build_schema

from .constraints import values_equal

# the plan-resolver callable signature: ($parent_step, field_args, info) -> Step
PlanResolver = Callable[..., Any]

# the resolve_type bridge signature graphql-core's completion calls per value:
# (value, info, abstract_type) -> the runtime concrete object type NAME (a str), the
# value being a plain row dict here. graphql-core validates that name against the
# schema's possible types, so an unknown/non-member name fails loud at completion.
TypeResolver = Callable[..., Optional[str]]


class FieldArgs:
    """A thin read accessor over a field's coerced argument values, plus per-argument
    variable provenance.

    ``field_args.get("name")`` / ``field_args["name"]`` return the already-coerced
    argument value (NOT a step — plan resolvers treat args as plain values and lift
    them with ``constant`` when a step is needed). ``raw`` exposes the whole coerced dict.

    Provenance (placeholders)
    -------------------------
    graphql-core's ``get_argument_values`` coerces a ``$variable`` argument to its
    runtime value before a plan resolver ever runs, so ``raw`` alone "cannot tell a
    literal from a variable". The seam is the field AST: an argument whose value node is
    a ``VariableNode`` came from a variable. The planner walks the field's argument nodes
    (when ``placeholders`` is enabled) and threads the SET of variable-derived argument
    names here as ``variable_args``, with the corresponding GraphQL variable name per arg
    in ``variable_sources``.

    A host plan resolver then asks ``field_args.is_variable("status")`` and, for a
    variable-derived value, builds a value-agnostic pg placeholder keyed by
    ``field_args.source("status")`` (a stable ``"var:<variable_name>"`` tag) instead of
    inlining the literal. The source tag is REQUEST-STABLE (the variable name, not a
    per-request id), so two requests of the same document produce the same key and share
    one cached plan, while two different variable sources never merge.

    The single-argument constructor ``FieldArgs(args)`` is supported: ``variable_args``
    defaults to empty, so ``is_variable`` is always ``False`` and every host falls back to
    literal inlining (the behaviour when no provenance is threaded in).

    Eval semantics (plan cache)
    ---------------------------
    Reading a variable-derived arg's VALUE at plan time is an **eval** — the value enters
    the plan, so the plan is only correct for requests carrying the same value. This
    accessor records every such observation (``__getitem__`` / ``get`` / ``raw`` for
    values, ``in`` / ``eval_has`` for presence, ``eval_is`` for a comparison outcome) and
    the planner converts the bookkeeping into per-variable CONSTRAINTS after the resolver
    runs (see ``plan.record_field_args_constraints``): the plan then caches as a per-value
    VARIANT — never stale, split per distinct value. Reading a LITERAL arg records nothing
    (its value is document-pinned, identical every request). The constraint-free channel
    is ``source()``: taking a variable arg's placeholder source (value never observed)
    nets out a raw read of the same arg, so the placeholder path records nothing at all.

    This mirrors the context gate (``info.context``) with one deliberate asymmetry: a
    bare context read yields a value-less token (every context read is per-request),
    while a bare arg read yields the value (most args are literals, and the variable case
    is recorded as the eval it is). Same vocabulary, two defaults — see AGENTS.md
    "purposeful distinctions from upstream".
    """

    def __init__(
        self,
        args: Optional[Mapping[str, Any]],
        variable_args: Optional[FrozenSet[str]] = None,
        variable_sources: Optional[Mapping[str, str]] = None,
        nested_variable_args: Optional[Mapping[str, FrozenSet[str]]] = None,
    ) -> None:
        self._args: Dict[str, Any] = dict(args or {})
        self.variable_args: FrozenSet[str] = (
            frozenset(variable_args) if variable_args else frozenset()
        )
        # arg name -> the GraphQL variable name it resolved from; used to build the
        # stable source tag. Defaults empty so source() falls back to the arg name.
        self.variable_sources: Dict[str, str] = dict(variable_sources or {})
        # arg name -> variable names NESTED inside the arg's list/object literal
        # (``where: {status: $s}``). Not placeholderable as a whole (the surrounding
        # structure is a literal), so a raw read of such an arg observes those variables'
        # values and constrains on each.
        self.nested_variable_args: Dict[str, FrozenSet[str]] = dict(
            nested_variable_args or {}
        )
        # the bookkeeping the planner nets after the resolver runs: variable-derived arg
        # names whose value was OBSERVED (read raw — direct or nested), those whose
        # placeholder ``source()`` was taken (value-agnostic; nets out an observation of
        # the same arg), presence checks, and eval_is outcomes. All empty when no
        # provenance was threaded (placeholders + caching off), so the default path
        # records nothing.
        self.literal_variable_reads: Set[str] = set()
        self.placeholdered_variable_args: Set[str] = set()
        self.membership_checks: Set[str] = set()
        self.equality_checks: list = []

    def get(self, name: str, default: Any = None) -> Any:
        self._note_literal_read(name)
        return self._args.get(name, default)

    def __getitem__(self, name: str) -> Any:
        self._note_literal_read(name)
        return self._args[name]

    @property
    def raw(self) -> Dict[str, Any]:
        """The whole coerced dict — observing EVERY variable-derived arg at once.

        The dict hands out values unrecorded, so taking it counts as a raw read of every
        variable-derived arg (direct and nested); placeholdered args still net out.
        """
        self.literal_variable_reads |= self.variable_args
        self.literal_variable_reads |= self.nested_variable_args.keys()
        return self._args

    def _note_literal_read(self, name: str) -> None:
        """Record a value observation of a variable-derived arg (direct or nested).

        Only variable-derived args matter — reading a plan-time literal arg's value is
        always safe (it is the same every request). An observed arg is a CANDIDATE
        inline; ``inlined_variable_args`` subtracts those the host also placeholdered
        (where the raw read only fed the placeholder's bound value).
        """
        if name in self.variable_args or name in self.nested_variable_args:
            self.literal_variable_reads.add(name)

    def inlined_variable_args(self) -> Set[str]:
        """Variable-derived args whose VALUE the host observed without placeholdering.

        The planner converts each to a per-variable value constraint after the resolver
        runs: the plan caches as a per-value variant. Reading a variable's raw value to
        BIND it onto a placeholder (``source()`` was also taken) does not count — that
        path is value-agnostic and records nothing.
        """
        return self.literal_variable_reads - self.placeholdered_variable_args

    def __contains__(self, name: str) -> bool:
        # presence of a DIRECT variable arg tracks whether the variable was provided —
        # a per-request fact a host may branch on, so it is recorded (a presence
        # constraint). A literal or nested-literal arg's presence is document-pinned.
        if name in self.variable_args:
            self.membership_checks.add(name)
        return name in self._args

    def eval(self, name: str, default: Any = None) -> Any:
        """The explicit spelling of a value observation (identical to ``args[name]``).

        Named for symmetry with ``info.context.eval``: the value is read NOW and the
        plan constrains on it (one cached variant per distinct value).
        """
        return self.get(name, default)

    def eval_is(self, name: str, expected: Any) -> bool:
        """Whether arg ``name`` equals ``expected``, constraining only the OUTCOME.

        The split-reducing read (mirrors ``info.context.eval_is``): every request on the
        same side of the comparison shares one cached plan variant. Only a DIRECT
        variable arg gets the outcome treatment — the comparison observes the COERCED
        arg, which tracks the variable only while the variable is present, so the
        recorded check is paired with a presence constraint at conversion (see
        ``plan.record_field_args_constraints``). An arg with NESTED variables has no
        single outcome to pin (the comparison saw every nested value), so it is recorded
        as a full value observation instead.
        """
        # the SAME comparison the hit-time replay uses (type-pinned ==): a build/replay
        # divergence (True == 1 here, type-mismatch there) would record a constraint that
        # fails for its own builder — a permanent-miss churn.
        passed = values_equal(self._args.get(name), expected)
        if name in self.variable_args:
            self.equality_checks.append((name, expected, passed))
        elif name in self.nested_variable_args:
            self.literal_variable_reads.add(name)
        return passed

    def eval_has(self, name: str) -> bool:
        """Whether arg ``name`` is present, constraining only the presence."""
        return self.__contains__(name)

    def is_variable(self, name: str) -> bool:
        """True iff argument ``name`` originated DIRECTLY from a GraphQL ``$variable``.

        Always ``False`` when no provenance was threaded in (the default / placeholders
        off), so a host then sees every arg as a literal and inlines it by value. An arg
        whose variables are NESTED inside a literal (``where: {status: $s}``) is not
        "a variable" — it cannot ride a single placeholder; reading it is an eval.
        """
        return name in self.variable_args

    def source(self, name: str) -> str:
        """The stable placeholder source tag for variable-derived argument ``name``.

        Returns ``"var:<variable_name>"`` — request-stable across requests of the same
        document, so identical-source placeholders dedup/merge and produce a cache hit,
        while different variable sources never merge. Falls back to the argument name
        when the underlying variable name was not threaded in (e.g. provenance built from
        ``variable_args`` alone).

        Taking a variable arg's source records it as PLACEHOLDERED, so a raw value-read of the
        same arg (to bind it onto the placeholder) does not record a constraint — the
        value never enters the SQL text, only a value-agnostic ``%(name)s`` does.

        Fails loud for an arg whose variables are NESTED inside a literal: the composite
        value cannot ride one placeholder, so handing out a source tag would silently
        resolve to nothing at render.
        """
        if name in self.nested_variable_args:
            raise ValueError(
                f"argument {name!r} cannot be placeholdered: its variables "
                f"({sorted(self.nested_variable_args[name])}) are nested inside a "
                "literal structure — read the value (the plan then caches per value) "
                "or restructure the argument as a direct $variable"
            )
        if name in self.variable_args:
            self.placeholdered_variable_args.add(name)
        var_name = self.variable_sources.get(name, name)
        return f"var:{var_name}"

    def __repr__(self) -> str:
        if self.variable_args:
            return f"FieldArgs({self._args!r}, variable_args={sorted(self.variable_args)!r})"
        return f"FieldArgs({self._args!r})"


def set_field_plan(field: GraphQLField, plan: PlanResolver) -> None:
    """Attach ``plan`` to ``field`` at ``extensions['grafast']['plan']``.

    Creates the ``extensions`` dict and the ``grafast`` sub-dict if absent, leaving
    any other extension data intact.
    """
    extensions = field.extensions
    if extensions is None:
        extensions = {}
        field.extensions = extensions
    grafast = extensions.get("grafast")
    if grafast is None:
        grafast = {}
        extensions["grafast"] = grafast
    grafast["plan"] = plan


def get_field_plan(field: GraphQLField) -> Optional[PlanResolver]:
    """Read a field's plan resolver, or ``None`` if it carries none."""
    extensions = field.extensions
    if not extensions:
        return None
    grafast = extensions.get("grafast")
    if not grafast:
        return None
    plan = grafast.get("plan")
    return plan if callable(plan) else None


PlanMap = Mapping[str, Mapping[str, PlanResolver]]


def attach_plans(schema: GraphQLSchema, plans: PlanMap) -> GraphQLSchema:
    """Attach a ``{"Type": {"field": plan}}`` map into ``schema``'s field extensions.

    Unknown type or field names raise (fail loud) rather than silently no-op, so a
    typo in a plan map surfaces immediately instead of leaving a field on the
    resolver-adapter path by accident.
    """
    for type_name, field_plans in plans.items():
        type_ = schema.type_map.get(type_name)
        if not isinstance(type_, GraphQLObjectType):
            raise KeyError(
                f"plan map references type {type_name!r} which is not an object type"
                " in the schema"
            )
        for field_name, plan in field_plans.items():
            field = type_.fields.get(field_name)
            if field is None:
                raise KeyError(
                    f"plan map references unknown field {type_name}.{field_name}"
                )
            set_field_plan(field, plan)
    return schema


def resolve_type_from_discriminator(
    column: str, mapping: Mapping[Any, str]
) -> TypeResolver:
    """A resolve_type bridge that maps a row's discriminator column to a typename.

    ``column`` is the key on the (row dict) value carrying the discriminator — e.g. a
    ``kind`` column holding ``"image"`` / ``"video"`` — and ``mapping`` maps each raw
    discriminator value to the GraphQL concrete type NAME (``{"image": "Image"}``). The
    returned callable has the ``(value, info, abstract_type)`` signature graphql-core's
    completion calls; it reads ``value[column]`` and returns the mapped typename.

    An unmapped discriminator value fails loud (``KeyError``) rather than returning
    ``None`` and bubbling a generic "must resolve to an Object type" error far from the
    cause — a value the host did not enumerate is a wiring bug. Reads nothing but dict
    keys, so this module stays free of any pg/sqlalchemy dependency.
    """

    def resolve_type(value: Any, info: Any, abstract_type: Any) -> str:
        discriminator = value[column]
        try:
            return mapping[discriminator]
        except KeyError:
            raise KeyError(
                f"discriminator value {discriminator!r} (column {column!r}) is not in "
                f"the type mapping {sorted(map(repr, mapping))}"
            ) from None

    return resolve_type


def resolve_type_from_tag(column: str) -> TypeResolver:
    """A resolve_type bridge that reads the concrete typename DIRECTLY off a tag column.

    For the common pgUnionAll shape where each member branch already tags its rows with
    its own GraphQL typename (a ``__typename`` / ``type`` column built into the union
    SQL), so no separate value->name mapping is needed: ``value[column]`` IS the
    typename. graphql-core then validates that name against the abstract type's possible
    types, so a bogus tag still fails loud at completion. Reads only dict keys.
    """

    def resolve_type(value: Any, info: Any, abstract_type: Any) -> str:
        return value[column]

    return resolve_type


TypeResolverMap = Mapping[str, TypeResolver]


def attach_type_resolvers(
    schema: GraphQLSchema, type_resolvers: TypeResolverMap
) -> GraphQLSchema:
    """Wire each ``{abstract_type_name: bridge}`` onto that type's ``resolve_type``.

    The completion engine reads ``abstract_type.resolve_type`` first (falling back to the
    context ``type_resolver``), so attaching the bridge here is all the wiring a
    Postgres-backed interface/union needs — the existing completion-time abstract
    dispatch does the per-concrete-type grouping and sub-selection planning unchanged.

    Mirrors :func:`attach_plans`'s fail-loud style: a name that is not an abstract
    (interface/union) type in the schema raises ``KeyError`` rather than silently
    no-op'ing, so a typo surfaces immediately.
    """
    for type_name, bridge in type_resolvers.items():
        type_ = schema.type_map.get(type_name)
        if not isinstance(type_, GraphQLAbstractType):
            raise KeyError(
                f"type_resolvers references type {type_name!r} which is not an "
                "interface or union type in the schema"
            )
        type_.resolve_type = bridge
    return schema


def make_grafast_schema(
    type_defs: str,
    plans: Optional[PlanMap] = None,
    type_resolvers: Optional[TypeResolverMap] = None,
) -> GraphQLSchema:
    """Build a schema from SDL and attach plan resolvers from a plan map.

    ``type_defs`` is GraphQL SDL; ``plans`` is ``{"Type": {"field": plan}}``. The
    result is a plain ``GraphQLSchema`` whose plan fields carry their resolver in
    ``extensions['grafast']['plan']`` — ready for the planner to pick up.

    ``type_resolvers`` is the optional ``{abstract_type_name: bridge}`` map for
    Postgres-backed interfaces/unions: each bridge is wired onto the abstract type's
    ``resolve_type`` so completion-time dispatch resolves each row's concrete type. Left
    ``None`` (the default) the signature is unchanged for existing callers.
    """
    schema = build_schema(type_defs)
    if plans:
        attach_plans(schema, plans)
    if type_resolvers:
        attach_type_resolvers(schema, type_resolvers)
    return schema


class GrafastSchemaBindable:
    """An Ariadne-style bindable that attaches plan resolvers to a built schema.

    Usage: ``make_executable_schema(type_defs, GrafastSchemaBindable(plan_map))``.
    Ariadne calls ``bind_to_schema(schema)`` after building the schema from SDL; we
    attach the plans into field extensions there.
    """

    def __init__(self, plans: PlanMap) -> None:
        self.plans = plans

    def bind_to_schema(self, schema: GraphQLSchema) -> None:
        attach_plans(schema, self.plans)


__all__ = [
    "FieldArgs",
    "PlanResolver",
    "set_field_plan",
    "get_field_plan",
    "attach_plans",
    "make_grafast_schema",
    "GrafastSchemaBindable",
    # resolve_type bridges for Postgres-backed interfaces/unions (completion-time dispatch)
    "TypeResolver",
    "resolve_type_from_discriminator",
    "resolve_type_from_tag",
    "attach_type_resolvers",
]
