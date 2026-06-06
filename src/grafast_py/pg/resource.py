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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from sqlalchemy.sql import ColumnElement
from sqlalchemy.types import TypeEngine

from ..core_steps import access, list_step
from ..step_model import Step
from .ordering import OrderTerm

# A resource select-customizer: context -> list of Core predicates ANDed onto EVERY
# batched select for the resource (the selectAuth analogue for soft-delete / tenant
# scoping / visibility). Resolved once per planned step against the per-request context.
SelectCustomizer = Callable[[Any], Sequence[Any]]


@dataclass(frozen=True)
class PgCodec:
    """A minimal per-attribute codec: a READ decode hook + a WRITE encode hook + a SQL type.

    ``to_py`` decodes a fetched value on READ (applied in every row-materialisation path —
    plain selects, window slices, connection nodes); ``to_pg`` encodes a Python value back
    to a bind on WRITE (mutations). Each is a pass-through no-op when ``None`` — a codec is
    only needed for a domain transform, since asyncpg/SQLAlchemy already handle common scalars.

    ``sql_type`` is the SQL type the codec's column declares for the KEYSET cursor path: a
    text-origin cursor value (an ISO datetime / decimal / array string) is cast back to this
    type so Postgres coerces it (see :mod:`grafast_py.pg.cursor`). It is fed into the
    resource ``column_types`` map at construction. A NATIVE scalar (int/text/bool — asyncpg
    already returns the right Python type and it binds directly) leaves it ``None``; only a
    non-native type (numeric/timestamptz/array/range) needs it. ``sql_type`` is compared by
    identity off the dedup path: a codec rides the post-grouping decode, so it never enters a
    step's peer_key / dedup_params (decode is dedup-neutral).
    """

    to_py: Optional[Callable[[Any], Any]] = None
    to_pg: Optional[Callable[[Any], Any]] = None
    sql_type: Optional[TypeEngine] = field(default=None, compare=False)


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


def match_columns_tuple(
    match_column: Optional[str],
    match_columns: Optional[Sequence[str]],
) -> Tuple[str, ...]:
    """Coerce a single ``match_column`` OR a ``match_columns`` tuple to one column tuple.

    The step factories accept EITHER a single column name (the common single-column FK /
    primary-key lookup) or an explicit tuple (a COMPOSITE key). Exactly one must be given;
    passing both — or neither — is a wiring bug and fails loud.
    """
    if match_columns is not None:
        if match_column is not None:
            raise ValueError(
                "pass match_column OR match_columns, not both"
            )
        columns = tuple(match_columns)
        if not columns:
            raise ValueError("match_columns must be non-empty")
        return columns
    if match_column is None:
        raise ValueError("pass a match_column or match_columns")
    return (match_column,)


def relation_columns_pair(
    local_column: Optional[str],
    remote_column: Optional[str],
    local_columns: Optional[Sequence[str]],
    remote_columns: Optional[Sequence[str]],
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Coerce single ``*_column`` OR tuple ``*_columns`` relation args to column tuples.

    A relation is declared with EITHER single ``local_column``/``remote_column`` strings
    (a single-column FK) or ``local_columns``/``remote_columns`` tuples (a composite FK);
    mixing the two forms, or supplying neither, is a declaration bug and fails loud.
    """
    if local_columns is not None or remote_columns is not None:
        if local_column is not None or remote_column is not None:
            raise ValueError(
                "declare a relation with single columns OR column tuples, not both"
            )
        if local_columns is None or remote_columns is None:
            raise ValueError(
                "a composite relation needs BOTH local_columns and remote_columns"
            )
        return tuple(local_columns), tuple(remote_columns)
    if local_column is None or remote_column is None:
        raise ValueError(
            "a relation needs local_column and remote_column (or the *_columns tuples)"
        )
    return (local_column,), (remote_column,)


def relation_key_step(parent_row_step: Step, local_columns: Sequence[str]) -> Step:
    """The per-entry key step for a relation: one column access, else a tuple of them.

    For a single-column FK this is a plain :class:`AccessStep` projecting the local column
    off the parent row (the scalar key the batched ``= ANY(:keys)`` matches). For a
    COMPOSITE FK it is a :class:`ListStep` of one access per local column, so each entry's
    key is the column TUPLE the tuple-IN skeleton matches as a whole.
    """
    columns = tuple(local_columns)
    if len(columns) == 1:
        return access(parent_row_step, (columns[0],))
    return list_step([access(parent_row_step, (c,)) for c in columns])


class PgRelation:
    """A FK link from one resource to another, over ONE OR MORE matched column pairs.

    ``local_columns`` are the columns on the *owning* resource whose values identify the
    related rows; ``remote_columns`` are the matched columns on ``target`` (positionally
    aligned). For a hasOne (``Post.author``) the local column is the FK (``author_id``)
    and the remote is the target's key (``id``). For a hasMany (``Author.posts``) the
    local column is the owner's key (``id``) and the remote is the FK on the target
    (``author_id``). A COMPOSITE foreign key carries several pairs (e.g.
    ``(org_id, item_id)`` matched against ``(org_id, item_id)``); the columns then form a
    tuple key matched as a whole.

    The single-column case is the fast path: ``local_column`` / ``remote_column`` return
    the lone column name (and raise if the relation is composite), so existing
    single-column call sites read unchanged.
    """

    def __init__(
        self,
        name: str,
        target: "PgResource",
        local_columns: Sequence[str],
        remote_columns: Sequence[str],
        kind: str,
    ) -> None:
        self.name = name
        self.target = target
        self.local_columns: Tuple[str, ...] = tuple(local_columns)
        self.remote_columns: Tuple[str, ...] = tuple(remote_columns)
        if not self.local_columns or len(self.local_columns) != len(self.remote_columns):
            raise ValueError(
                f"relation {name!r} needs matching non-empty local/remote column "
                f"tuples; got local={self.local_columns} remote={self.remote_columns}"
            )
        self.kind = kind  # "has_one" | "has_many"

    @property
    def is_composite(self) -> bool:
        """Whether the FK spans more than one column pair (the tuple-key path)."""
        return len(self.local_columns) > 1

    @property
    def local_column(self) -> str:
        """The lone local column name (single-column fast path; raises if composite)."""
        if self.is_composite:
            raise ValueError(
                f"relation {self.name!r} is composite ({self.local_columns}); use "
                "local_columns"
            )
        return self.local_columns[0]

    @property
    def remote_column(self) -> str:
        """The lone remote column name (single-column fast path; raises if composite)."""
        if self.is_composite:
            raise ValueError(
                f"relation {self.name!r} is composite ({self.remote_columns}); use "
                "remote_columns"
            )
        return self.remote_columns[0]


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
        #
        # An attribute whose codec declares a ``sql_type`` (a non-native scalar, or an
        # array/range whose element/own type the codec carries) seeds that entry here, so
        # the keyset CAST keeps working for a codec-typed order column with no separate
        # column_types declaration. An EXPLICIT column_types entry WINS (it is the host's
        # deliberate override), so codec-derived types only fill the gaps.
        derived_types = self.codec_column_types()
        if column_types:
            derived_types.update(column_types)
        self.column_types: Mapping[str, TypeEngine] = derived_types
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

    def decode_value(self, name: str, value: Any) -> Any:
        """Decode ONE named column's value via its attribute codec (a single-value decode).

        The per-row :meth:`decode_row` decodes a whole materialised node; this decodes a
        SINGLE column value the same way, for the connection aggregate path where a value
        is surfaced OUTSIDE a node row (an ``aggregateGroups`` group-by key, or a ``min`` /
        ``max`` over a codec'd column). A column with no codec ``to_py`` (or an unknown name)
        passes through unchanged, so node and aggregate output stay consistently typed.
        """
        attribute = self.attributes.get(name)
        if attribute is None or attribute.codec is None or attribute.codec.to_py is None:
            return value
        return attribute.codec.to_py(value)

    @property
    def has_decoders(self) -> bool:
        """Whether any attribute carries a ``to_py`` decode hook (fast-path else copy-only)."""
        return any(
            a.codec is not None and a.codec.to_py is not None
            for a in self.attributes.values()
        )

    def codec_column_types(self) -> Dict[str, TypeEngine]:
        """Per-column SQL types DERIVED from attribute codecs (for the keyset CAST).

        Each attribute whose codec declares a ``sql_type`` (a non-native scalar, or an
        array/range whose codec carries the cast type) contributes ``name -> sql_type``, so
        a codec-typed order column casts a text-origin cursor value back to its type without
        a separate ``column_types`` declaration. Seeded into :attr:`column_types` at
        construction (an explicit entry overrides). Off the dedup path — purely a read/cast
        concern.
        """
        return {
            name: attribute.codec.sql_type
            for name, attribute in self.attributes.items()
            if attribute.codec is not None and attribute.codec.sql_type is not None
        }

    def codec_for(
        self,
        pg_type_name: str,
        *,
        composite_fields: Optional[Sequence] = None,
        enum_labels: Optional[Sequence[str]] = None,
    ) -> PgCodec:
        """Look up the :class:`PgCodec` for a Postgres type NAME (the codec registry seam).

        Delegates to :func:`grafast_py.pg.codecs.codec_for`, so a host derives an attribute's
        codec by Postgres type (``"timestamptz"``, ``"int4[]"``, ``"int4range"``) rather than
        hand-building the ``to_py`` / ``sql_type`` pair — recursing into array/range/composite
        element types. Composite and enum types pass their structure via
        ``composite_fields`` / ``enum_labels``. Imported lazily to keep ``codecs`` (which
        imports :class:`PgCodec` from here) free of a circular import at module load.
        """
        from .codecs import codec_for

        return codec_for(
            pg_type_name,
            composite_fields=composite_fields,
            enum_labels=enum_labels,
        )

    def has_one(
        self,
        name: str,
        target: "PgResource",
        local_column: Optional[str] = None,
        remote_column: Optional[str] = None,
        *,
        local_columns: Optional[Sequence[str]] = None,
        remote_columns: Optional[Sequence[str]] = None,
    ) -> PgRelation:
        """Register a hasOne relation (one related row, matched remote == local).

        Pass single ``local_column``/``remote_column`` strings for the common
        single-column FK, or ``local_columns``/``remote_columns`` tuples for a COMPOSITE
        FK (the columns then match as a whole tuple).
        """
        local, remote = relation_columns_pair(
            local_column, remote_column, local_columns, remote_columns
        )
        relation = PgRelation(name, target, local, remote, "has_one")
        self.relations[name] = relation
        return relation

    def has_many(
        self,
        name: str,
        target: "PgResource",
        local_column: Optional[str] = None,
        remote_column: Optional[str] = None,
        *,
        local_columns: Optional[Sequence[str]] = None,
        remote_columns: Optional[Sequence[str]] = None,
    ) -> PgRelation:
        """Register a hasMany relation (a list of related rows).

        Single ``local_column``/``remote_column`` for a single-column FK, or
        ``local_columns``/``remote_columns`` tuples for a COMPOSITE FK.
        """
        local, remote = relation_columns_pair(
            local_column, remote_column, local_columns, remote_columns
        )
        relation = PgRelation(name, target, local, remote, "has_many")
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

    def get_single(
        self,
        key_step: Step,
        match_column: Optional[str] = None,
        *,
        match_columns: Optional[Sequence[str]] = None,
    ) -> Step:
        """A :class:`PgSelectSingleStep`: one row where the match key(s) == the key step.

        ``match_column`` (single) defaults to the primary key (``resource.get(id)``);
        pass ``match_columns`` for a COMPOSITE key (the key step then supplies a tuple).
        """
        from .steps import PgSelectSingleStep

        # default to the primary key ONLY when neither form is given (the bare
        # ``resource.get(id)`` lookup); a composite ``match_columns`` must not also default
        # the single ``match_column``.
        if match_column is None and match_columns is None:
            match_column = self.primary_key
        columns = match_columns_tuple(match_column, match_columns)
        return PgSelectSingleStep(self, key_step, columns)

    def find(
        self,
        key_step: Step,
        match_column: Optional[str] = None,
        order_by: Optional[Sequence[Union[str, OrderTerm]]] = None,
        order_is_unique: bool = False,
        first: Optional[int] = None,
        offset: int = 0,
        *,
        match_columns: Optional[Sequence[str]] = None,
    ) -> Step:
        """A :class:`PgSelectStep`: rows where the match key(s) == the key step.

        Used for hasMany relations (``match_column``/``match_columns`` are the FK on this
        resource). ``first``/``offset`` apply a PER-PARENT page slice (the in-SQL window
        slice), not a bucket-wide LIMIT.
        """
        from .steps import PgSelectStep

        columns = match_columns_tuple(match_column, match_columns)
        return PgSelectStep(
            self, key_step, columns,
            order_by=order_by, order_is_unique=order_is_unique,
            first=first, offset=offset,
        )

    def related_single(self, parent_row_step: Step, relation_name: str) -> Step:
        """Plan a hasOne relation off ``parent_row_step`` (the parent row step)."""
        relation = self.get_relation(relation_name)
        key = relation_key_step(parent_row_step, relation.local_columns)
        return relation.target.get_single(
            key, match_columns=relation.remote_columns
        )

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
        key = relation_key_step(parent_row_step, relation.local_columns)
        default_order = order_by or [relation.target.primary_key]
        return relation.target.find(
            key, match_columns=relation.remote_columns,
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
    "match_columns_tuple",
    "relation_columns_pair",
    "relation_key_step",
]
