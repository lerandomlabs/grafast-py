"""Derive :class:`PgResource` descriptors from SQLAlchemy declarative ORM models.

The host still writes the GraphQL SDL and the plan resolvers; this module only
saves them re-typing table / column / relation metadata that already lives on
their mapped models. It produces plain :class:`PgResource` descriptors (name,
schema, table, ordered column list, primary key, relations) — NO codecs, NO
column-type mapping, NO GraphQL type generation. The core resource model stays
ORM-neutral; SQLAlchemy is imported only here.

Two entry points:

- :func:`resource_from_model` builds ONE resource (relations not wired).
- :func:`resources_from_models` builds a whole batch in two passes: first a
  resource per model (kept in a model->resource identity map), then the relations
  between models that are both in the batch. Relation targets resolve through that
  identity map (keyed on the model class, not its derived name), so wiring is
  independent of how the resources are named.
"""

from typing import Dict, Iterable, Optional, Sequence, Type

from sqlalchemy import inspect as sa_inspect
from sqlalchemy.orm import RelationshipProperty

from ..config import log
from .resource import PgRegistry, PgResource


def resource_from_model(
    model: Type,
    *,
    registry: Optional[PgRegistry] = None,
    name: Optional[str] = None,
    schema: Optional[str] = None,
    columns: Optional[Sequence[str]] = None,
    primary_key: Optional[str] = None,
) -> PgResource:
    """Build one :class:`PgResource` from a mapped declarative ``model``.

    Relations are NOT wired here (a single model carries no notion of which other
    models are in scope); use :func:`resources_from_models` for that. Overrides win
    over the derived values. Raises ``ValueError`` if ``model`` is not a mapped
    declarative model, or has a composite/absent primary key and no ``primary_key``
    override.
    """
    table = getattr(model, "__table__", None)
    if table is None:
        raise ValueError(
            f"{model!r} is not a mapped declarative model (no __table__); "
            "pass a SQLAlchemy declarative model class"
        )

    resource_name = name or model.__tablename__
    resource_schema = schema or table.schema or "public"
    resource_columns = (
        list(columns) if columns is not None else [c.name for c in table.columns]
    )
    resource_pk = primary_key or derive_primary_key(model, table)

    return PgResource(
        resource_name,
        resource_schema,
        table.name,
        resource_columns,
        primary_key=resource_pk,
        registry=registry,
    )


def derive_primary_key(model: Type, table) -> str:
    """Return the single primary-key column name, else raise ``ValueError``.

    A composite or absent primary key cannot be expressed by :class:`PgResource`
    (its ``primary_key`` is a single column), so callers must pass a ``primary_key``
    override in that case — this fails loudly to surface it.
    """
    pk_columns = list(table.primary_key.columns)
    if len(pk_columns) == 1:
        return pk_columns[0].name
    detail = "composite" if pk_columns else "absent"
    raise ValueError(
        f"model {model.__name__!r} has a {detail} primary key "
        f"({[c.name for c in pk_columns]}); pass primary_key=... to "
        "resource_from_model / resources_from_models"
    )


def resources_from_models(
    models: Iterable[Type],
    *,
    registry: Optional[PgRegistry] = None,
    relations: bool = True,
    strict: bool = False,
) -> PgRegistry:
    """Build a :class:`PgRegistry` of resources for a batch of mapped models.

    Two passes: (1) create a resource per model and remember a model->resource
    identity map; (2) if ``relations`` is set, wire the relations whose target model
    is also in the batch. Returns the registry (a fresh :class:`PgRegistry` if none
    is passed).

    Each relation is wired only when it is a plain single-column hasOne / hasMany to
    an in-batch target. Many-to-many relations, composite-FK relations, and relations
    whose target is out of the batch are skipped with a ``log.warning``; with
    ``strict=True`` every such skip is raised as a ``ValueError`` instead.
    """
    registry = registry if registry is not None else PgRegistry()
    model_list = list(models)

    model_to_resource: Dict[Type, PgResource] = {}
    for model in model_list:
        model_to_resource[model] = resource_from_model(model, registry=registry)

    if relations:
        for model in model_list:
            wire_relations(model, model_to_resource, strict=strict)

    return registry


def wire_relations(
    model: Type,
    model_to_resource: Dict[Type, PgResource],
    *,
    strict: bool,
) -> None:
    """Wire ``model``'s single-column relations onto its resource.

    Iterates ``sa_inspect(model).relationships`` and adds a hasOne / hasMany relation
    for every plain single-column relation whose target model is in
    ``model_to_resource``. Unsupported shapes (many-to-many, composite FK, out-of-batch
    target) are skipped with a warning, or raised when ``strict``.
    """
    resource = model_to_resource[model]
    for rel in sa_inspect(model).relationships:
        add_relation(resource, rel, model_to_resource, strict=strict)


def add_relation(
    resource: PgResource,
    rel: RelationshipProperty,
    model_to_resource: Dict[Type, PgResource],
    *,
    strict: bool,
) -> None:
    """Add a single relationship ``rel`` to ``resource`` (or skip it loudly).

    ``rel.local_remote_pairs[0]`` is ``(local, remote)`` oriented relative to the
    relationship's OWN parent class — local is a column on this resource's table,
    remote a column on the target's table — which matches
    :class:`~grafast_py.pg.resource.PgRelation`'s convention directly, so no
    re-orientation is needed. Kind is ``has_one`` when ``rel.uselist`` is ``False``
    (covers many-to-one AND one-to-one) and ``has_many`` otherwise.
    """
    if rel.direction.name == "MANYTOMANY":
        if strict:
            raise ValueError(
                f"resource {resource.name!r}: cannot derive many-to-many relation "
                f"{rel.key!r} (use an explicit join resource)"
            )
        log.warning(
            "skip many-to-many relation", resource=resource.name, relation=rel.key
        )
        return

    if len(rel.local_remote_pairs) != 1:
        if strict:
            raise ValueError(
                f"resource {resource.name!r}: cannot derive composite-FK relation "
                f"{rel.key!r} ({len(rel.local_remote_pairs)} column pairs)"
            )
        log.warning(
            "skip composite foreign key relation",
            resource=resource.name,
            relation=rel.key,
            pairs=len(rel.local_remote_pairs),
        )
        return

    target_resource = model_to_resource.get(rel.mapper.class_)
    if target_resource is None:
        if strict:
            raise ValueError(
                f"resource {resource.name!r}: relation {rel.key!r} targets "
                f"{rel.mapper.class_.__name__!r}, which is not in the batch"
            )
        log.warning(
            "skip relation to out-of-batch model",
            resource=resource.name,
            relation=rel.key,
            target=rel.mapper.class_.__name__,
        )
        return

    local_col, remote_col = rel.local_remote_pairs[0]
    if rel.uselist is False:
        resource.has_one(
            rel.key,
            target=target_resource,
            local_column=local_col.name,
            remote_column=remote_col.name,
        )
    else:
        resource.has_many(
            rel.key,
            target=target_resource,
            local_column=local_col.name,
            remote_column=remote_col.name,
        )


__all__ = ["resource_from_model", "resources_from_models"]
