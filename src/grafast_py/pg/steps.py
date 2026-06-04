"""Batch pg steps: one parameterised SQL per bucket, scattered back to parents.

:class:`PgSelectStep` (hasMany / collections) and :class:`PgSelectSingleStep`
(hasOne / single row) are ordinary :class:`grafast_py.step_model.Step` subclasses, so
they ride the SAME bucket executor, dedup and ``EachStep`` flattening as the core
steps — no second executor. The Grafast batching property is realised in
``execute``: it receives EVERY lookup key across the bucket at once (its dep-0 key
column), folds them into a single ``WHERE match_column = ANY($1)`` statement run ONCE
on the async engine, then scatters the rows to the parents whose key matched.

Hence a depth-D nested query issues ~D SQL statements total (one per resource-layer):
``authors`` (1) -> for ALL authors' posts ``WHERE author_id = ANY(...)`` (1) -> for
ALL those posts' comments ``WHERE post_id = ANY(...)`` (1). The flattening of list
items into one bucket (the core ``EachStep`` / list completion) is what lets the inner
layer see every key at once.

Rows come back as plain dicts keyed by column name so the existing ``AccessStep``
projects leaf columns with no special-casing.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from sqlalchemy import any_, bindparam, column, select, table

from ..config import log
from ..core_steps import access
from ..step_model import Step
from .customize import PgCustomizable
from .executor import current_pg_request
from .ordering import OrderTerm, normalize_order, order_clauses
from .pagination import window_slice, window_slice_params
from .resource import PgResource


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
    """Batched ``SELECT ... WHERE match_column = ANY($1)`` returning a LIST per entry.

    One dependency: the per-entry key step (dep 0), e.g. an author id for an
    ``Author.posts`` hasMany. ``execute`` gathers every key in the bucket, runs ONE
    statement, groups rows by their ``match_column`` value, and scatters each entry's
    list of matching rows (missing key -> empty list).

    Host customization: UNIFORM WHERE predicates AND-combined onto the batched WHERE —
    the resource ``select_customizer`` (resolved once against the per-request context)
    plus per-plan ``.where()``s via :meth:`builder`. The skeleton (``= ANY($1)`` / window
    partition) stays ours; hosts add only uniform additions.
    """

    is_sync_and_safe = False

    def __init__(
        self,
        resource: PgResource,
        key_step: Step,
        match_column: str,
        order_by: Optional[Sequence[Union[str, OrderTerm]]] = None,
        order_is_unique: bool = False,
        first: Optional[int] = None,
        offset: int = 0,
    ) -> None:
        super().__init__()
        self.resource = resource
        self.match_column = match_column
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

    def build_query(self):
        """Build the batched ``= ANY($1)`` SELECT via SQLAlchemy Core.

        With no page bound (``first is None and offset == 0``) this is the plain
        ORDER BY select — no window overhead. With a bound it is the shared per-parent
        window slice (``row_number() OVER (PARTITION BY match)`` then ``__rn`` filtered),
        so only each parent's page rows come back in the single bucket statement.
        """
        if not self.is_limited:
            tbl = table(
                self.resource.table,
                *[column(c) for c in self.resource.columns],
                schema=self.resource.schema,
            )
            match = column(self.match_column)
            # project the stored columns PLUS each computed attribute's labelled
            # expression (over the table columns) in the SAME select — no extra statement.
            stmt = select(tbl, *self.resource.computed_projections()).where(
                match == any_(bindparam("keys", expanding=False))
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
            match_column=self.match_column,
            order_by=self.order_by,
            first=self.first,
            offset=self.offset,
            where_predicates=self.where_predicates,
            computed=self.resource.computed_projections(),
        )

    async def run_query(self, unique_keys: List[Any]) -> List[Dict[str, Any]]:
        """Run the batched statement once and return the RAW (undecoded) rows.

        Codec decode is applied AFTER grouping (see :meth:`group_and_decode`), not here:
        a codec on the match column would otherwise change ``row[match_column]`` and the
        rows would group under the decoded value while the key step supplies the raw one,
        scattering to nobody. Grouping on the raw value first keeps that safe.
        """
        request = current_pg_request()
        params: Dict[str, Any] = {"keys": unique_keys}
        if self.is_limited:
            params.update(window_slice_params(self.first, self.offset))
        rows = await request.executor.run(
            self.build_query(), params, settings=request.settings
        )
        log.debug(
            "pg batch select",
            resource=self.resource.name,
            keys=len(unique_keys),
            rows=len(rows),
        )
        return rows

    def group_rows(self, rows: List[Dict[str, Any]]) -> Dict[Any, List[Dict[str, Any]]]:
        """Group rows by their RAW ``match_column`` value (preserving query order)."""
        grouped: Dict[Any, List[Dict[str, Any]]] = {}
        col = self.match_column
        for row in rows:
            grouped.setdefault(row[col], []).append(row)
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

    def scatter(
        self,
        grouped: Dict[Any, List[Dict[str, Any]]],
        keys: List[Any],
        count: int,
    ) -> List[Any]:
        """Per-entry list of matching rows (missing key -> empty list)."""
        return [grouped.get(keys[i], []) for i in range(count)]

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        keys = values[0]
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
        # customization signature folds the host WHERE predicates so the cheap pre-filter
        # never merges byte-different statements.
        return (
            f"pg_select|{self.resource.qualified_table}|{self.match_column}"
            f"|{self.order_by!r}|{self.first}|{self.offset}"
            f"|{self.customization_signature()!r}"
        )

    def dedup_params(self) -> Tuple[Any, ...]:
        return (
            self.resource.qualified_table,
            self.match_column,
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
    """Batched ``SELECT ... WHERE match_column = ANY($1)`` returning ONE row per entry.

    Same single-statement batching as :class:`PgSelectStep`; each entry's result is
    its single matching row (or ``None`` for a missing key) rather than a list. Used
    for hasOne relations (``Post.author``) and ``resource.get(id)``.
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
        keys = values[0]
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
            f"pg_select_single|{self.resource.qualified_table}|{self.match_column}"
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
    match_column: str,
    order_by: Optional[Sequence[Union[str, OrderTerm]]] = None,
    order_is_unique: bool = False,
    first: Optional[int] = None,
    offset: int = 0,
) -> PgSelectStep:
    return PgSelectStep(
        resource, key_step, match_column,
        order_by=order_by, order_is_unique=order_is_unique,
        first=first, offset=offset,
    )


def pg_select_single(
    resource: PgResource, key_step: Step, match_column: Optional[str] = None
) -> PgSelectSingleStep:
    return PgSelectSingleStep(
        resource, key_step, match_column or resource.primary_key
    )


__all__ = [
    "PgSelectStep",
    "PgSelectSingleStep",
    "PgSelectAllStep",
    "pg_select",
    "pg_select_single",
]
