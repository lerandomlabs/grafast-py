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

from typing import Any, Callable, Dict, Mapping, Optional

from graphql import GraphQLField, GraphQLObjectType, GraphQLSchema
from graphql.utilities import build_schema

# the plan-resolver callable signature: ($parent_step, field_args, info) -> Step
PlanResolver = Callable[..., Any]


class FieldArgs:
    """A thin read accessor over a field's coerced argument values.

    ``field_args.get("name")`` / ``field_args["name"]`` return the already-coerced
    argument value (NOT a step, in this Phase-A surface — plan resolvers treat args
    as plain values and lift them with ``constant`` when a step is needed). ``raw``
    exposes the whole coerced dict.
    """

    def __init__(self, args: Optional[Mapping[str, Any]]) -> None:
        self.raw: Dict[str, Any] = dict(args or {})

    def get(self, name: str, default: Any = None) -> Any:
        return self.raw.get(name, default)

    def __getitem__(self, name: str) -> Any:
        return self.raw[name]

    def __contains__(self, name: str) -> bool:
        return name in self.raw

    def __repr__(self) -> str:
        return f"FieldArgs({self.raw!r})"


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


def make_grafast_schema(
    type_defs: str, plans: Optional[PlanMap] = None
) -> GraphQLSchema:
    """Build a schema from SDL and attach plan resolvers from a plan map.

    ``type_defs`` is GraphQL SDL; ``plans`` is ``{"Type": {"field": plan}}``. The
    result is a plain ``GraphQLSchema`` whose plan fields carry their resolver in
    ``extensions['grafast']['plan']`` — ready for the planner to pick up.
    """
    schema = build_schema(type_defs)
    if plans:
        attach_plans(schema, plans)
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
]
