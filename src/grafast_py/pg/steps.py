"""Batch pg steps: one parameterised SQL per bucket, scattered back to parents.

:class:`PgSelectStep` (hasMany / collections) and :class:`PgSelectSingleStep`
(hasOne / single row) are ordinary :class:`grafast_py.step_model.Step` subclasses, so
they ride the SAME bucket executor, dedup and ``EachStep`` flattening as the core
steps — no second executor. The Grafast batching property is realised in
``execute``: it receives EVERY lookup key across the bucket at once (its dep-0 key
column), folds them into a single ``WHERE match = ANY($1)`` statement run ONCE on the
async engine, then scatters the rows to the parents whose key matched. The match key is
ONE column (the fast path ``= ANY``) OR a COMPOSITE tuple of columns
(``(c1, c2, ...) IN (...)``); the grouping/scatter key is the scalar value for a single
column and the column TUPLE for a composite (see :func:`grouping_key`).

Hence a depth-D nested query issues ~D SQL statements total (one per resource-layer):
``authors`` (1) -> for ALL authors' posts ``WHERE author_id = ANY(...)`` (1) -> for
ALL those posts' comments ``WHERE post_id = ANY(...)`` (1). The flattening of list
items into one bucket (the core ``EachStep`` / list completion) is what lets the inner
layer see every key at once.

Rows come back as plain dicts keyed by column name so the existing ``AccessStep``
projects leaf columns with no special-casing.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from sqlalchemy import any_, bindparam, column, select, table, tuple_

from ..config import log
from ..core_steps import access
from ..step_model import Step
from .customize import PgCustomizable
from .executor import current_pg_request
from .ordering import OrderTerm, normalize_order, order_clauses
from .pagination import window_slice, window_slice_params
from .resource import PgResource


def as_match_columns(match: Union[str, Sequence[str]]) -> Tuple[str, ...]:
    """Coerce a step's ``match`` argument to a column tuple (single str -> 1-tuple).

    The step constructors accept EITHER a single column name (the single-column fast
    path) or an already-built tuple of names (a COMPOSITE key); both collapse to the one
    ``match_columns`` tuple every emission/grouping path reads.
    """
    if isinstance(match, str):
        return (match,)
    columns = tuple(match)
    if not columns:
        raise ValueError("a pg step needs at least one match column")
    return columns


def grouping_key(row: Mapping[str, Any], match_columns: Tuple[str, ...]) -> Any:
    """The dict key a row groups/scatters under: the scalar value, or the column tuple.

    Single-column keeps the cheap scalar (``row[col]``) so the key step's scalar value
    matches directly; a COMPOSITE key reduces to ``tuple(row[c] for c in cols)`` so a row
    groups under the same tuple the key step supplies. Computed on the RAW row (before any
    codec decode) so a codec on a match column cannot misgroup (see the connection path).
    """
    if len(match_columns) == 1:
        return row[match_columns[0]]
    return tuple(row[c] for c in match_columns)


def normalize_lookup_key(raw_key: Any, composite: bool) -> Any:
    """Reduce a per-entry key-step value to its hashable lookup/grouping form, or ``None``.

    Single column: the scalar key passes through (``None`` looks up nothing). COMPOSITE: a
    :class:`ListStep` hands each entry a LIST of the local-column values; tuple-ify it so it
    is hashable and aligns with :func:`grouping_key`. A composite key with ANY ``None``
    component (or a missing whole key) cannot match a row (the FK is not fully present), so
    it normalises to ``None`` — scattering to null/empty, never to a partial match.
    """
    if not composite:
        return raw_key
    if raw_key is None:
        return None
    key = tuple(raw_key)
    if any(component is None for component in key):
        return None
    return key


def _coalesce_keys(keys: List[Any], count: int) -> Tuple[List[Any], List[Optional[int]]]:
    """Collapse a bucket's key column to unique non-null keys + per-entry slot.

    Returns ``(unique_keys, entry_slot)`` where ``entry_slot[i]`` is the index into
    ``unique_keys`` for entry ``i`` (or ``None`` when that entry's key is ``None`` and
    therefore looks up nothing — missing key -> null/empty).
    """
    unique_keys: List[Any] = []
    slot_for_key: Dict[Any, int] = {}
    entry_slot: List[Optional[int]] = [None] * count
    for i in range(count):
        key = keys[i]
        if key is None:
            continue
        slot = slot_for_key.get(key)
        if slot is None:
            slot = len(unique_keys)
            slot_for_key[key] = slot
            unique_keys.append(key)
        entry_slot[i] = slot
    return unique_keys, entry_slot


class PgSelectStep(PgCustomizable):
    """Batched key-matched ``SELECT`` returning a LIST per entry.

    One dependency: the per-entry key step (dep 0), e.g. an author id for an
    ``Author.posts`` hasMany. ``execute`` gathers every key in the bucket, runs ONE
    statement, groups rows by their match key, and scatters each entry's list of matching
    rows (missing key -> empty list).

    The key match has two skeletons selected by the match-column count: a SINGLE column is
    the fast path ``match = ANY($1)`` (a ``$1::T[]`` array param); a COMPOSITE key emits
    ``(c1, c2, ...) IN (:keys)`` over a tuple, the list-of-tuples baked onto the bindparam
    at build time. The grouping key follows suit: a scalar for single, the column tuple for
    composite (see :func:`grouping_key`).

    Host customization: UNIFORM WHERE predicates AND-combined onto the batched WHERE —
    the resource ``select_customizer`` (resolved once against the per-request context)
    plus per-plan ``.where()``s via :meth:`builder`. The skeleton (key match / window
    partition) stays ours; hosts add only uniform additions.
    """

    is_sync_and_safe = False

    def __init__(
        self,
        resource: PgResource,
        key_step: Step,
        match: Union[str, Sequence[str]],
        order_by: Optional[Sequence[Union[str, OrderTerm]]] = None,
        order_is_unique: bool = False,
        first: Optional[int] = None,
        offset: int = 0,
    ) -> None:
        super().__init__()
        self.resource = resource
        # ``match`` is a single column name (the fast path) OR a tuple of names (a
        # COMPOSITE key); collapse both to the one column tuple every path reads.
        self.match_columns: Tuple[str, ...] = as_match_columns(match)
        self.order_is_unique = order_is_unique
        self.order_by: Tuple[OrderTerm, ...] = normalize_order(
            order_by, primary_key=resource.primary_key, order_is_unique=order_is_unique
        )
        resource.assert_order_terms_stored(self.order_by)
        # structured page bounds, applied PER PARENT via the window slice (never a
        # bucket-wide LIMIT, which would limit the whole ANY($1) result across parents).
        self.first = first
        self.offset = offset
        # dep 0 is the key step; values[0] is the key column at execute time.
        self.add_dependency(key_step)
        # seed the WHERE list with the resource's select-customizer predicates (resolved
        # ONCE against the per-request context), then per-plan .where()s append after.
        self.seed_resource_customization(resource)

    @property
    def is_limited(self) -> bool:
        """Whether a per-parent page bound is set (window slice) vs plain select."""
        return self.first is not None or self.offset != 0

    @property
    def is_composite(self) -> bool:
        """Whether the match key spans more than one column (the tuple-IN path)."""
        return len(self.match_columns) > 1

    @property
    def match_column(self) -> str:
        """The lone match column (single-column fast path; raises if composite)."""
        if self.is_composite:
            raise ValueError(
                f"select on {self.resource.name!r} is composite "
                f"({self.match_columns}); use match_columns"
            )
        return self.match_columns[0]

    def add_order_term(self, term: Union[str, OrderTerm]) -> None:
        """Append a UNIFORM ordering term (re-normalised with the PK tie-break)."""
        existing = [t for t in self.order_by]
        existing.append(term if isinstance(term, OrderTerm) else OrderTerm(term))
        self.order_by = normalize_order(
            existing,
            primary_key=self.resource.primary_key,
            order_is_unique=self.order_is_unique,
        )
        self.resource.assert_order_terms_stored(self.order_by)

    def set_first(self, first: Optional[int]) -> None:
        """Set the structured per-parent page size (window slice; never a raw LIMIT)."""
        self.first = first

    def set_offset(self, offset: int) -> None:
        """Set the structured per-parent page offset."""
        self.offset = offset

    def match_predicate(self, unique_keys: Optional[List[Any]] = None):
        """The batched key-match predicate: ``= ANY(:keys)`` (single) or tuple-IN (composite).

        Single column keeps the cheap ``match = ANY(:keys)`` (``$1::T[]`` array, keys bound
        as a flat list via the execute-time params) — it ignores ``unique_keys``, so the
        plain skeleton renders without any keys. A COMPOSITE key emits
        ``(c1, c2, ...) IN (:keys)`` over a ``tuple_`` with ``expanding=True``; the
        list-of-tuples value is baked onto the bindparam at BUILD time because the
        RawExecutor ``render_postcompile`` path needs the value present to expand the IN
        list (a value-less expanding bind cannot postcompile). The SQLAlchemyExecutor path
        reads the same baked value, so binding it here serves both executors.
        """
        if not self.is_composite:
            match = column(self.match_columns[0])
            return match == any_(bindparam("keys", expanding=False))
        cols = [column(c) for c in self.match_columns]
        return tuple_(*cols).in_(
            bindparam("keys", value=unique_keys or [], expanding=True)
        )

    def build_query(self, unique_keys: Optional[List[Any]] = None):
        """Build the batched key-matched SELECT via SQLAlchemy Core.

        With no page bound (``first is None and offset == 0``) this is the plain
        ORDER BY select — no window overhead. With a bound it is the shared per-parent
        window slice (``row_number() OVER (PARTITION BY match)`` then ``__rn`` filtered),
        so only each parent's page rows come back in the single bucket statement. The
        key-match clause is single (``= ANY``) or composite (tuple-IN) per
        :meth:`match_predicate`; ``unique_keys`` is consumed only by the composite branch
        (it bakes the IN list at build time), and ignored by the single fast path.
        """
        if not self.is_limited:
            tbl = table(
                self.resource.table,
                *[column(c) for c in self.resource.columns],
                schema=self.resource.schema,
            )
            # project the stored columns PLUS each computed attribute's labelled
            # expression (over the table columns) in the SAME select — no extra statement.
            stmt = select(tbl, *self.resource.computed_projections()).where(
                self.match_predicate(unique_keys)
            )
            # AND each host predicate onto the batched WHERE alongside the skeleton.
            for predicate in self.where_predicates:
                stmt = stmt.where(predicate)
            if self.order_by:
                stmt = stmt.order_by(*order_clauses(self.order_by))
            return stmt

        return window_slice(
            schema=self.resource.schema,
            table_name=self.resource.table,
            columns=self.resource.columns,
            match_columns=self.match_columns,
            unique_keys=unique_keys,
            order_by=self.order_by,
            first=self.first,
            offset=self.offset,
            where_predicates=self.where_predicates,
            computed=self.resource.computed_projections(),
        )

    async def run_query(self, unique_keys: List[Any]) -> List[Dict[str, Any]]:
        """Run the batched statement once and return the RAW (undecoded) rows.

        Codec decode is applied AFTER grouping (see :meth:`group_and_decode`), not here:
        a codec on the match column would otherwise change the grouping value and the rows
        would group under the decoded value while the key step supplies the raw one,
        scattering to nobody. Grouping on the raw value first keeps that safe.

        The single-column path binds ``keys`` as the flat array param at execute time; the
        composite path already baked the list-of-tuples onto the IN bindparam at build
        time, so it passes no ``keys`` param (a re-bind would clash with the expanded
        per-element binds).
        """
        request = current_pg_request()
        params: Dict[str, Any] = {}
        if not self.is_composite:
            params["keys"] = unique_keys
        if self.is_limited:
            params.update(window_slice_params(self.first, self.offset))
        rows = await request.executor.run(
            self.build_query(unique_keys), params, settings=request.settings
        )
        log.debug(
            "pg batch select",
            resource=self.resource.name,
            keys=len(unique_keys),
            rows=len(rows),
        )
        return rows

    def group_rows(self, rows: List[Dict[str, Any]]) -> Dict[Any, List[Dict[str, Any]]]:
        """Group rows by their RAW match key — the scalar value or the column tuple."""
        grouped: Dict[Any, List[Dict[str, Any]]] = {}
        match_columns = self.match_columns
        for row in rows:
            grouped.setdefault(grouping_key(row, match_columns), []).append(row)
        return grouped

    def group_and_decode(
        self, rows: List[Dict[str, Any]]
    ) -> Dict[Any, List[Dict[str, Any]]]:
        """Group rows on the RAW match value, then codec-decode each group.

        Grouping keys on the raw ``match_column`` (what the key step supplied) BEFORE any
        decode, so a codec on the match column cannot misgroup rows; the per-group decode
        then materialises the presented (decoded) values. Mirrors the connection path
        (group raw, decode after). When no attribute has a decode hook this is just the
        raw grouping (no per-row copy).
        """
        grouped = self.group_rows(rows)
        if not self.resource.has_decoders:
            return grouped
        return {key: self.resource.decode_rows(group) for key, group in grouped.items()}

    def normalized_keys(self, keys: List[Any]) -> List[Any]:
        """Each entry's lookup/grouping key (scalar or tuple), ``None`` when it matches none."""
        composite = self.is_composite
        return [normalize_lookup_key(k, composite) for k in keys]

    def scatter(
        self,
        grouped: Dict[Any, List[Dict[str, Any]]],
        keys: List[Any],
        count: int,
    ) -> List[Any]:
        """Per-entry list of matching rows (missing key -> empty list).

        ``keys`` are the NORMALIZED keys (scalar or tuple), aligned with the grouping key.
        """
        return [grouped.get(keys[i], []) for i in range(count)]

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        keys = self.normalized_keys(values[0])
        unique_keys, _slot = _coalesce_keys(keys, count)
        if not unique_keys:
            return [[] for _ in range(count)]

        async def run():
            rows = await self.run_query(unique_keys)
            return self.scatter(self.group_and_decode(rows), keys, count)

        return run()

    @property
    def peer_key(self) -> str:
        # the normalized order_by tuple already bakes in the order_is_unique decision
        # (whether the PK tie-break was appended), so it alone captures every
        # SQL-relevant ordering difference — folding the raw flag too would only
        # over-discriminate (miss merging byte-identical statements). first/offset
        # change the page slice (and thus the SQL), so they must discriminate too. The
        # match_columns tuple discriminates single vs composite (and which columns) — it
        # changes the SQL skeleton (= ANY vs tuple-IN) and the grouping key. The
        # customization signature folds the host WHERE predicates so the cheap pre-filter
        # never merges byte-different statements.
        return (
            f"pg_select|{self.resource.qualified_table}|{self.match_columns!r}"
            f"|{self.order_by!r}|{self.first}|{self.offset}"
            f"|{self.customization_signature()!r}"
        )

    def dedup_params(self) -> Tuple[Any, ...]:
        return (
            self.resource.qualified_table,
            self.match_columns,
            self.order_by,
            self.first,
            self.offset,
            self.customization_signature(),
        )


class PgSelectAllStep(PgCustomizable):
    """Batched ``SELECT ... [ORDER BY]`` (no key WHERE) returning ALL rows per entry.

    The root-collection step (``Query.authors`` / ``Query.posts``): one dependency,
    the bucket's parent step (the operation root), used only to size the bucket — the
    same row list is returned for every entry. A root list has a single bucket entry
    (the root value), so this is ONE statement; the relation layers chain off the
    returned row steps.

    Host customization: UNIFORM WHERE predicates AND-combined onto the select — the
    resource ``select_customizer`` (resolved once against the per-request context) plus
    per-plan ``.where()``s via :meth:`builder`. ``first``/``offset`` page the single
    root result with a plain ``LIMIT``/``OFFSET`` (a root list has ONE bucket entry, so
    there is no per-parent fan-out a bucket-wide LIMIT could corrupt — unlike a hasMany).
    """

    is_sync_and_safe = False

    def __init__(
        self,
        resource: PgResource,
        order_by: Optional[Sequence[Union[str, OrderTerm]]] = None,
        order_is_unique: bool = False,
        first: Optional[int] = None,
        offset: int = 0,
    ) -> None:
        super().__init__()
        self.resource = resource
        self.order_is_unique = order_is_unique
        self.order_by: Tuple[OrderTerm, ...] = normalize_order(
            order_by, primary_key=resource.primary_key, order_is_unique=order_is_unique
        )
        resource.assert_order_terms_stored(self.order_by)
        # page bounds for the single root result (plain LIMIT/OFFSET; see class doc).
        self.first = first
        self.offset = offset
        self.seed_resource_customization(resource)

    def for_parent(self, parent_step: Step) -> "PgSelectAllStep":
        """Wire the bucket-sizing parent dependency and return self.

        The parent is dep 0, used only to size the bucket (the same row list is returned
        for every entry).
        """
        self.add_dependency(parent_step)
        return self

    def set_first(self, first: Optional[int]) -> None:
        """Set the root page size (plain LIMIT on the single root result)."""
        self.first = first

    def set_offset(self, offset: int) -> None:
        """Set the root page offset (plain OFFSET on the single root result)."""
        self.offset = offset

    def add_order_term(self, term: Union[str, OrderTerm]) -> None:
        """Append a UNIFORM ordering term (re-normalised with the PK tie-break)."""
        existing = list(self.order_by)
        existing.append(term if isinstance(term, OrderTerm) else OrderTerm(term))
        self.order_by = normalize_order(
            existing,
            primary_key=self.resource.primary_key,
            order_is_unique=self.order_is_unique,
        )
        self.resource.assert_order_terms_stored(self.order_by)

    def build_query(self):
        tbl = table(
            self.resource.table,
            *[column(c) for c in self.resource.columns],
            schema=self.resource.schema,
        )
        # project the stored columns PLUS each computed attribute's labelled expression
        # in the SAME select (no extra statement).
        stmt = select(tbl, *self.resource.computed_projections())
        # AND each host predicate onto the select (uniform across the whole result).
        for predicate in self.where_predicates:
            stmt = stmt.where(predicate)
        if self.order_by:
            stmt = stmt.order_by(*order_clauses(self.order_by))
        if self.offset:
            stmt = stmt.offset(self.offset)
        if self.first is not None:
            stmt = stmt.limit(self.first)
        return stmt

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        async def run():
            request = current_pg_request()
            rows = await request.executor.run(
                self.build_query(), None, settings=request.settings
            )
            # decode each row's codec-bearing attributes once; the same decoded list is
            # returned for every bucket entry (a root list has one bucket entry).
            decoded = self.resource.decode_rows(rows)
            return [list(decoded) for _ in range(count)]

        return run()

    @property
    def peer_key(self) -> str:
        return (
            f"pg_select_all|{self.resource.qualified_table}"
            f"|{self.order_by!r}|{self.first}|{self.offset}"
            f"|{self.customization_signature()!r}"
        )

    def dedup_params(self) -> Tuple[Any, ...]:
        return (
            self.resource.qualified_table,
            self.order_by,
            self.first,
            self.offset,
            self.customization_signature(),
        )


class PgSelectSingleStep(PgSelectStep):
    """Batched key-matched ``SELECT`` returning ONE row per entry.

    Same single-statement batching as :class:`PgSelectStep` (the single ``= ANY`` or the
    composite tuple-IN skeleton); each entry's result is its single matching row (or
    ``None`` for a missing key) rather than a list. Used for hasOne relations
    (``Post.author``) and ``resource.get(id)``.
    """

    def scatter(
        self,
        grouped: Dict[Any, List[Dict[str, Any]]],
        keys: List[Any],
        count: int,
    ) -> List[Any]:
        out: List[Any] = []
        for i in range(count):
            rows = grouped.get(keys[i])
            out.append(rows[0] if rows else None)
        return out

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        keys = self.normalized_keys(values[0])
        unique_keys, _slot = _coalesce_keys(keys, count)
        if not unique_keys:
            return [None] * count

        async def run():
            rows = await self.run_query(unique_keys)
            return self.scatter(self.group_and_decode(rows), keys, count)

        return run()

    @property
    def peer_key(self) -> str:
        return (
            f"pg_select_single|{self.resource.qualified_table}|{self.match_columns!r}"
            f"|{self.order_by!r}|{self.customization_signature()!r}"
        )

    def get(self, attr: Any) -> Step:
        """Lazily access a column of the loaded row (an :class:`AccessStep`)."""
        return access(self, (attr,))


# -------------------------------------------------------------- plan-helper API
# Free-function constructors mirroring Grafast's pgSelect / pgSelectSingle, so a plan
# resolver reads naturally. Each merely CONSTRUCTS a step (no SQL at plan time).


def pg_select(
    resource: PgResource,
    key_step: Step,
    match: Union[str, Sequence[str]],
    order_by: Optional[Sequence[Union[str, OrderTerm]]] = None,
    order_is_unique: bool = False,
    first: Optional[int] = None,
    offset: int = 0,
) -> PgSelectStep:
    """A batched select keyed on ``match`` (a single column name or a composite tuple)."""
    return PgSelectStep(
        resource, key_step, match,
        order_by=order_by, order_is_unique=order_is_unique,
        first=first, offset=offset,
    )


def pg_select_single(
    resource: PgResource,
    key_step: Step,
    match: Optional[Union[str, Sequence[str]]] = None,
) -> PgSelectSingleStep:
    """A batched single-row select keyed on ``match`` (defaults to the primary key)."""
    return PgSelectSingleStep(
        resource, key_step, match if match is not None else resource.primary_key
    )


__all__ = [
    "PgSelectStep",
    "PgSelectSingleStep",
    "PgSelectAllStep",
    "pg_select",
    "pg_select_single",
]
