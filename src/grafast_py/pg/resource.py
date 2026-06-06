"""PgResource: a table/source abstraction with columns, primary key and relations.

Mirrors Grafast's ``@dataplan/pg`` resource concept. A :class:`PgResource` names a
table, its ATTRIBUTES (column descriptors), its primary key, and the relations (hasOne /
hasMany) that link it to other resources by FK columns. Plan resolvers use the resource's
factory methods (:meth:`PgResource.get_single`, :meth:`PgResource.find`) and relation
helpers to build the batched pg steps in :mod:`grafast_py.pg.steps`.

Attributes (:class:`PgColumn`) carry a column NAME plus an optional per-attribute
:class:`PgCodec` (a ``to_py`` decode hook applied on read), nullability/default flags
(metadata, e.g. from the ORM), and an optional ``expression``. An attribute WITH an
expression is a COMPUTED column (``pgClassExpression``): a host-authored Core SQL
expression over the table columns, projected as an extra labelled column in the SAME
batched SELECT — never a separate statement, never request data. An attribute WITHOUT an
expression is a stored table column (selectable directly).

Computed columns are PROJECTION-ONLY in v1: a SELECT-list alias cannot be referenced in
``row_number() OVER (ORDER BY ...)`` or a keyset comparator, so ordering by one raises a
clear error (:meth:`PgResource.assert_order_terms_stored`); to order or filter, use a
stored column or inline the SQL expression. Orderable/filterable computed columns are
deferred.

A bare string in ``columns`` is sugar for ``PgColumn(name=str)`` — so the historical
``columns=["id", "name"]`` form keeps working unchanged, and ``resource.columns`` still
returns the ordered list of TABLE-column names (the stored, selectable ones; computed
attributes are excluded). ``resource.attributes`` is the full descriptor map (incl.
computed) and ``resource.computed`` the names whose attribute has an expression.

Resources register in a :class:`PgRegistry` so relations resolve their target by
name regardless of declaration order.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

from sqlalchemy.sql import ColumnElement
from sqlalchemy.types import TypeEngine

from ..core_steps import access
from ..step_model import Step
from .ordering import OrderTerm

# A resource select-customizer: context -> list of Core predicates ANDed onto EVERY
# batched select for the resource (the selectAuth analogue for soft-delete / tenant
# scoping / visibility). Resolved once per planned step against the per-request context.
SelectCustomizer = Callable[[Any], Sequence[Any]]


@dataclass(frozen=True)
class PgCodec:
    """A minimal per-attribute codec: a READ decode hook + a WRITE encode hook.

    ``to_py`` decodes a fetched value on READ (applied in every row-materialisation path —
    plain selects, window slices, connection nodes); ``to_pg`` encodes a Python value back
    to a bind on WRITE (mutations). Each is a pass-through no-op when ``None`` — a codec is
    only needed for a domain transform, since asyncpg/SQLAlchemy already handle common scalars.
    """

    to_py: Optional[Callable[[Any], Any]] = None
    to_pg: Optional[Callable[[Any], Any]] = None


@dataclass(frozen=True)
class PgColumn:
    """One resource ATTRIBUTE descriptor: a column name plus optional codec/expression.

    ``name`` is the attribute name (a stored column's name, or the label of a computed
    column). ``codec`` is the optional per-attribute :class:`PgCodec` (its ``to_py`` is
    applied on read). ``not_null`` / ``has_default`` are metadata flags (populated e.g.
    from an ORM model) used by mutations later; they do not affect reads. ``expression``,
    when set, marks the attribute COMPUTED: a host-authored Core SQL expression (over the
    table columns) projected as ``expression.label(name)`` — so a computed attribute is
    NOT a stored table column and is excluded from :attr:`PgResource.columns`.
    """

    name: str
    codec: Optional[PgCodec] = None
    not_null: bool = False
    has_default: bool = False
    expression: Optional[ColumnElement] = field(default=None, compare=False)

    @property
    def is_computed(self) -> bool:
        """Whether this attribute is a computed column (has an expression)."""
        return self.expression is not None


def as_column(entry: Union[str, PgColumn]) -> PgColumn:
    """Coerce a column spec to a :class:`PgColumn` (a bare ``str`` -> ``PgColumn(name)``)."""
    return entry if isinstance(entry, PgColumn) else PgColumn(name=entry)


class PgRelation:
    """A FK link from one resource to another.

    ``local_column`` is the column on the *owning* resource whose value identifies the
    related rows; ``remote_column`` is the matched column on ``target``. For a hasOne
    (``Post.author``) the local column is the FK (``author_id``) and the remote is the
    target's key (``id``). For a hasMany (``Author.posts``) the local column is the
    owner's key (``id``) and the remote is the FK on the target (``author_id``).
    """

    def __init__(
        self,
        name: str,
        target: "PgResource",
        local_column: str,
        remote_column: str,
        kind: str,
    ) -> None:
        self.name = name
        self.target = target
        self.local_column = local_column
        self.remote_column = remote_column
        self.kind = kind  # "has_one" | "has_many"


class PgResource:
    """One Postgres table/source: name, columns, primary key, relations."""

    def __init__(
        self,
        name: str,
        schema: str,
        table: str,
        columns: Sequence[Union[str, "PgColumn"]],
        primary_key: str = "id",
        registry: Optional["PgRegistry"] = None,
        select_customizer: Optional[SelectCustomizer] = None,
        column_types: Optional[Mapping[str, TypeEngine]] = None,
    ) -> None:
        self.name = name
        self.schema = schema
        self.table = table
        # Each column spec becomes a PgColumn descriptor (a bare string is sugar for a
        # plain stored column). The attributes map preserves declaration order (incl.
        # computed attributes); self.columns below derives the stored-column NAME list.
        self.attributes: Dict[str, PgColumn] = {}
        for entry in columns:
            attribute = as_column(entry)
            # the attributes dict is keyed by name, so a duplicate would silently collapse
            # (the old list form preserved duplicates). A duplicate attribute name is a
            # declaration bug — reject it loudly rather than drop a column.
            if attribute.name in self.attributes:
                raise ValueError(
                    f"resource {name!r} declares attribute {attribute.name!r} more than "
                    "once; attribute names must be unique"
                )
            self.attributes[attribute.name] = attribute
        self.primary_key = primary_key
        # OPTIONAL per-column SQL types, consulted ONLY by the keyset cursor path to cast
        # a text-origin cursor value (an ISO datetime / decimal string) back to a
        # non-native column type. Native int/text columns need no entry (they bind
        # directly), so this stays empty for the common case — NOT a full type map.
        self.column_types: Mapping[str, TypeEngine] = column_types or {}
        self.relations: Dict[str, PgRelation] = {}
        # the selectAuth analogue: context -> list[Core predicate], resolved ONCE per
        # planned step and ANDed onto every batched select for this resource. None means
        # no default scoping. NOT an RLS framework — flat AND-combined predicates only.
        self.select_customizer = select_customizer
        if registry is not None:
            registry.add(self)

    @property
    def qualified_table(self) -> str:
        """``schema.table`` for use in SQL."""
        return f"{self.schema}.{self.table}"

    @property
    def columns(self) -> List[str]:
        """The ordered list of STORED table-column names (computed attributes excluded).

        This is the selectable-column list every ``build_query`` iterates as ``column(c)``
        — kept name-for-name identical to the historical ``columns`` list so all existing
        select paths (and the from_sqlalchemy parity test) are unchanged. Computed
        attributes (those with an ``expression``) are projected separately, never as a
        plain ``column(name)``, so they are not here.
        """
        return [a.name for a in self.attributes.values() if not a.is_computed]

    @property
    def computed(self) -> List[str]:
        """The names of computed attributes (those carrying an ``expression``)."""
        return [a.name for a in self.attributes.values() if a.is_computed]

    def assert_order_terms_stored(self, order_terms: Sequence[OrderTerm]) -> None:
        """Fail loud if any ORDER BY term names a COMPUTED attribute (projection-only).

        A computed attribute is emitted only as a SELECT-list alias, so Postgres cannot
        reference it inside ``row_number() OVER (ORDER BY ...)`` or a keyset comparator —
        it would raise a raw "column does not exist" error deep in execution. Computed
        columns are projection-only here; surface a clear, column-named error at order-set
        time so a host orders by a stored column or inlines the SQL expression instead.
        """
        for term in order_terms:
            attribute = self.attributes.get(term.column)
            if attribute is not None and attribute.is_computed:
                raise ValueError(
                    f"cannot order by computed column {term.column!r}; computed columns "
                    "are projection-only — order by a stored column, or inline the SQL "
                    "expression"
                )

    def computed_projections(self) -> List[ColumnElement]:
        """Core ``expression.label(name)`` projections for every computed attribute.

        Host-authored, identifier-class SQL over the TABLE columns — projected as extra
        labelled columns in the SAME batched SELECT (in the table-scope/INNER select for
        the window-sliced and connection paths), never a separate statement and never
        carrying request data. Empty for a resource with no computed attributes.
        """
        return [
            a.expression.label(a.name)
            for a in self.attributes.values()
            if a.is_computed
        ]

    def decode_row(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        """Apply each attribute's codec ``to_py`` to a fetched row dict (decode on read).

        Returns a new dict where every attribute carrying a codec with a ``to_py`` hook
        has its value decoded; columns without a codec (the common case) pass through
        unchanged. Applied uniformly wherever rows are materialised — plain selects,
        window slices, connection nodes — so decoding is consistent across all paths.
        """
        if not self.has_decoders:
            return dict(row)
        out = dict(row)
        for name, attribute in self.attributes.items():
            codec = attribute.codec
            if codec is not None and codec.to_py is not None and name in out:
                out[name] = codec.to_py(out[name])
        return out

    def decode_rows(self, rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        """Decode a list of fetched rows via :meth:`decode_row` (the read materialiser)."""
        if not self.has_decoders:
            return [dict(r) for r in rows]
        return [self.decode_row(r) for r in rows]

    @property
    def has_decoders(self) -> bool:
        """Whether any attribute carries a ``to_py`` decode hook (fast-path else copy-only)."""
        return any(
            a.codec is not None and a.codec.to_py is not None
            for a in self.attributes.values()
        )

    def has_one(
        self, name: str, target: "PgResource", local_column: str, remote_column: str
    ) -> PgRelation:
        """Register a hasOne relation (one related row, matched remote == local)."""
        relation = PgRelation(name, target, local_column, remote_column, "has_one")
        self.relations[name] = relation
        return relation

    def has_many(
        self, name: str, target: "PgResource", local_column: str, remote_column: str
    ) -> PgRelation:
        """Register a hasMany relation (a list of related rows)."""
        relation = PgRelation(name, target, local_column, remote_column, "has_many")
        self.relations[name] = relation
        return relation

    def get_relation(self, name: str) -> PgRelation:
        """Return a registered relation by name (fail loud on a typo)."""
        relation = self.relations.get(name)
        if relation is None:
            raise KeyError(f"resource {self.name!r} has no relation {name!r}")
        return relation

    # ------------------------------------------------------------- step factories
    # The plan-resolver-facing surface. Each builds a batched pg step keyed on a
    # per-entry key step; the SQL emission lives in grafast_py.pg.steps.

    def get_single(self, key_step: Step, match_column: Optional[str] = None) -> Step:
        """A :class:`PgSelectSingleStep`: one row where ``match_column`` == key.

        ``match_column`` defaults to the primary key (``resource.get(id)``).
        """
        from .steps import PgSelectSingleStep

        return PgSelectSingleStep(self, key_step, match_column or self.primary_key)

    def find(
        self,
        key_step: Step,
        match_column: str,
        order_by: Optional[Sequence[Union[str, OrderTerm]]] = None,
        order_is_unique: bool = False,
        first: Optional[int] = None,
        offset: int = 0,
    ) -> Step:
        """A :class:`PgSelectStep`: rows where ``match_column`` == key.

        Used for hasMany relations (``match_column`` is the FK on this resource).
        ``first``/``offset`` apply a PER-PARENT page slice (the in-SQL window slice),
        not a bucket-wide LIMIT.
        """
        from .steps import PgSelectStep

        return PgSelectStep(
            self, key_step, match_column,
            order_by=order_by, order_is_unique=order_is_unique,
            first=first, offset=offset,
        )

    def related_single(self, parent_row_step: Step, relation_name: str) -> Step:
        """Plan a hasOne relation off ``parent_row_step`` (the parent row step)."""
        relation = self.get_relation(relation_name)
        key = access(parent_row_step, (relation.local_column,))
        return relation.target.get_single(key, relation.remote_column)

    def related_many(
        self,
        parent_row_step: Step,
        relation_name: str,
        order_by: Optional[Sequence[Union[str, OrderTerm]]] = None,
        order_is_unique: bool = False,
        first: Optional[int] = None,
        offset: int = 0,
    ) -> Step:
        """Plan a hasMany relation off ``parent_row_step`` (the parent row step).

        ``first``/``offset`` page EACH parent's related rows via the in-SQL window
        slice (one batched statement across the bucket), not a bucket-wide LIMIT.
        """
        relation = self.get_relation(relation_name)
        key = access(parent_row_step, (relation.local_column,))
        default_order = order_by or [relation.target.primary_key]
        return relation.target.find(
            key, relation.remote_column,
            order_by=default_order, order_is_unique=order_is_unique,
            first=first, offset=offset,
        )


class PgRegistry:
    """A name -> :class:`PgResource` map so relations resolve their targets."""

    def __init__(self) -> None:
        self.resources: Dict[str, PgResource] = {}

    def add(self, resource: PgResource) -> None:
        self.resources[resource.name] = resource

    def __getitem__(self, name: str) -> PgResource:
        return self.resources[name]


__all__ = [
    "PgResource",
    "PgRelation",
    "PgRegistry",
    "PgColumn",
    "PgCodec",
    "as_column",
]
