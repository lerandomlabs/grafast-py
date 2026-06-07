"""Relay connection step: KEYSET-paged, sliced in SQL, batched across parents.

:class:`PgConnectionStep` wraps a hasMany lookup and produces, per bucket entry, a Relay
connection dict ``{"edges", "nodes", "totalCount", "pageInfo"}``. It pages EVERY parent's
connection in ONE SQL statement: a ``row_number() OVER (PARTITION BY match)`` window whose
inner WHERE carries the KEYSET ``after``/``before`` seek predicate (NOT an offset),
fetching ONE extra row per partition so ``hasNextPage`` needs no per-partition count.
Cursors are keyset (seek) cursors (:mod:`grafast_py.pg.cursor`) — a cursor minted under a
different ordering is rejected loudly, never misapplied.

``totalCount`` is a SEPARATE batched aggregate (``match, count(*) GROUP BY match``) issued
only when the selection set asks for it (``needs_total``); being its own query it stays
correct even on an EMPTY terminal page. So a connection layer is at most TWO statements
across all parents — the page query plus the optional count (O(depth)).

DOMAIN AGGREGATES (``sum``/``avg``/``min``/``max``/``count`` over a column, optionally
GROUPed) are a THIRD optional batched statement built the same way: ``match, <aggs>
GROUP BY match[, <group_by>]`` (:meth:`PgConnectionStep.build_aggregate_query`). It mirrors
``build_count_query`` exactly — same key match, same host WHERE predicates (so it AND-s the
SAME ``where_predicates``, INCLUDING any compiled filter Condition, as the page query),
WITHOUT the keyset cursor predicate (it aggregates the FULL per-parent set, correct even on
an empty terminal page). It is issued only when the selection set asks for an aggregate
(:func:`connection_aggregates`). The aggregate SPEC (which functions over which columns,
plus the optional extra grouping) folds into the connection dedup key alongside
``needs_total`` + ``customization_signature`` — two byte-different aggregate statements never
merge. A connection layer is therefore at most THREE statements across all parents (page +
optional count + optional aggregate), still O(depth).

Forward paging is ``first``/``after``; reverse is ``last``/``before`` (keyset on the
reversed order, ``last+1`` then re-reversed in Python). Sub-fields are plain
:class:`AccessStep` projections into the per-entry dict, and ``edges[].node`` is the row
dict, so nested relations under a node batch exactly like a plain row.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from graphql.language import FieldNode, FragmentSpreadNode, InlineFragmentNode
from sqlalchemy import any_, bindparam, column, func, select, table, tuple_
from sqlalchemy.sql import ColumnElement

from ..core_steps import access
from ..step_model import Step
from .customize import PgCustomizable
from .cursor import (
    decode_keyset_cursor,
    effective_nulls,
    encode_keyset_cursor,
    keyset_where,
)
from .executor import current_pg_request
from .ordering import OrderTerm, normalize_order, order_clauses
from .placeholders import (
    Placeholder,
    placeholder_source_tag,
    rebind_pagination_value,
    unwrap_placeholder,
)
from .resource import PgResource
from .steps import as_match_columns, grouping_key, normalize_lookup_key


def connection_needs_total(info: Any) -> bool:
    """Whether the connection field's selection set asks for ``totalCount``.

    Walks the connection field's own sub-selection (resolving fragment spreads via
    ``info.fragments`` and inline fragments) for a field named ``totalCount``. The plan
    resolver calls this once to decide whether the SEPARATE count aggregate is issued, so
    a connection that does not select ``totalCount`` skips the count query entirely.
    """
    fragments = getattr(info, "fragments", {}) or {}
    for field_node in info.field_nodes:
        if _selects_total(field_node.selection_set, fragments):
            return True
    return False


def _selects_total(selection_set: Any, fragments: Dict[str, Any]) -> bool:
    """Recurse a selection set (and its fragments) for a ``totalCount`` field."""
    if selection_set is None:
        return False
    for selection in selection_set.selections:
        if isinstance(selection, FieldNode):
            if selection.name.value == "totalCount":
                return True
        elif isinstance(selection, InlineFragmentNode):
            if _selects_total(selection.selection_set, fragments):
                return True
        elif isinstance(selection, FragmentSpreadNode):
            fragment = fragments.get(selection.name.value)
            if fragment is not None and _selects_total(
                fragment.selection_set, fragments
            ):
                return True
    return False


# The aggregate functions a connection can compute over a per-parent set, mapped to the
# Core function each emits. ``count`` over a column counts NON-NULL values of it; ``count``
# with no column is the per-parent row count (count(*) — totalCount already covers the
# unconditional case, so a column-less ``count`` here is for a filtered/distinct variant a
# host wires explicitly). The set mirrors Grafast's pgAggregate built-ins.
AGGREGATE_FUNCTIONS = frozenset({"sum", "avg", "min", "max", "count"})


@dataclass(frozen=True)
class PgAggregate:
    """One aggregate over a connection's per-parent set: ``<function>(<column>) AS <alias>``.

    ``function`` is one of :data:`AGGREGATE_FUNCTIONS`; ``column`` is the stored column it
    aggregates (``None`` only for a bare ``count`` -> ``count(*)``); ``alias`` is the output
    key the aggregate value lands under in the per-parent aggregate dict. The triple is the
    UNIT the connection dedup key folds over — two aggregate specs differing in any of
    function / column / alias must not merge — so the dataclass is frozen (hashable) and
    every field participates in equality.
    """

    function: str
    column: Optional[str] = None
    alias: str = ""

    def __post_init__(self) -> None:
        if self.function not in AGGREGATE_FUNCTIONS:
            raise ValueError(
                f"unsupported aggregate function {self.function!r}; supported: "
                f"{', '.join(sorted(AGGREGATE_FUNCTIONS))}"
            )
        if self.column is None and self.function != "count":
            raise ValueError(
                f"aggregate {self.function!r} needs a column (only count may omit one)"
            )
        if not self.alias:
            raise ValueError("aggregate needs an output alias")

    def projection(self) -> ColumnElement:
        """The labelled Core aggregate expression (``<function>(<column>) AS <alias>``).

        A bare ``count`` (no column) emits ``count(*)``; every other form applies the
        function to ``column(self.column)``. Projected into the aggregate SELECT alongside
        the match columns, so each per-parent group carries its aggregate value under
        ``alias``.
        """
        if self.column is None:
            return func.count().label(self.alias)
        return getattr(func, self.function)(column(self.column)).label(self.alias)


def connection_aggregates(info: Any, aggregate_fields: Mapping[str, PgAggregate]) -> List[PgAggregate]:
    """The aggregates the connection's selection set asks for (gated, like ``needs_total``).

    ``aggregate_fields`` maps a connection sub-field NAME to the :class:`PgAggregate` it
    denotes (the host's binding of GraphQL aggregate fields to SQL aggregates). This walks
    the connection field's own sub-selection — resolving fragment spreads and inline
    fragments exactly like :func:`connection_needs_total` — and returns the aggregates whose
    field is selected, in declaration (mapping) order so the spec is deterministic. The plan
    resolver calls it once to decide whether (and which) aggregate statement is issued, so a
    connection selecting no aggregate field skips the aggregate query entirely.
    """
    fragments = getattr(info, "fragments", {}) or {}
    selected: set = set()
    for field_node in info.field_nodes:
        _collect_aggregate_fields(field_node.selection_set, fragments, selected)
    return [agg for name, agg in aggregate_fields.items() if name in selected]


def _collect_aggregate_fields(
    selection_set: Any, fragments: Dict[str, Any], selected: set
) -> None:
    """Collect the field NAMES selected under a selection set (and its fragments)."""
    if selection_set is None:
        return
    for selection in selection_set.selections:
        if isinstance(selection, FieldNode):
            selected.add(selection.name.value)
        elif isinstance(selection, InlineFragmentNode):
            _collect_aggregate_fields(selection.selection_set, fragments, selected)
        elif isinstance(selection, FragmentSpreadNode):
            fragment = fragments.get(selection.name.value)
            if fragment is not None:
                _collect_aggregate_fields(fragment.selection_set, fragments, selected)


class PgConnectionStep(PgCustomizable):
    """Batched Relay connection over a hasMany lookup keyed on ``match_columns``.

    Host customization: UNIFORM WHERE predicates AND-combined onto the INNER WHERE
    (before ``row_number()`` materialises) — the resource ``select_customizer`` (resolved
    once against the per-request context) plus per-plan ``.where()``s via :meth:`builder`.
    Paging is the construction-time keyset cursor; the builder rejects
    ``set_offset`` with a clear error (no offset surface here).

    Forward paging is ``first``/``after``; reverse paging is ``last``/``before``. Exactly
    one direction is in play per step (a connection field is forward XOR reverse).
    ``needs_total`` (set by the plan resolver from the selection set) gates the separate
    count aggregate; ``aggregates`` (likewise selection-gated) gates the separate domain
    aggregate, optionally grouped by ``aggregate_group_by`` beyond the match key.
    """

    is_sync_and_safe = False

    def __init__(
        self,
        resource: PgResource,
        key_step: Step,
        match: Union[str, Sequence[str]],
        order_by: Sequence[Union[str, OrderTerm]],
        first: Optional[Union[int, Placeholder]] = None,
        after: Optional[Union[str, Placeholder]] = None,
        last: Optional[Union[int, Placeholder]] = None,
        before: Optional[Union[str, Placeholder]] = None,
        order_is_unique: bool = False,
        needs_total: bool = False,
        aggregates: Sequence[PgAggregate] = (),
        aggregate_group_by: Sequence[str] = (),
    ) -> None:
        super().__init__()
        self.resource = resource
        # single column name (fast path) OR a tuple (a COMPOSITE key); the page/count
        # queries partition+group over the same column(s).
        self.match_columns: Tuple[str, ...] = as_match_columns(match)
        self.order_is_unique = order_is_unique
        self.order_by: Tuple[OrderTerm, ...] = normalize_order(
            order_by, primary_key=resource.primary_key, order_is_unique=order_is_unique
        )
        resource.assert_order_terms_stored(self.order_by)
        # page size: a plan-time literal int/None (value-included key) OR a ``Placeholder``
        # (variable-derived, source-keyed). The keyset page binds page_limit as a param
        # regardless, so a placeholder changes only the dedup key — the page math unwraps it.
        self.first = first
        self.last = last
        self.needs_total = needs_total
        # the stable source tags of a variable-derived after/before cursor (``None`` for a
        # literal cursor). The dedup key emits these in place of the decoded cursor VALUES so
        # a variable cursor keys value-agnostically (a cache hit across requests of the same
        # document); the decoded values still drive the keyset SQL, bound as params as before.
        self.after_source: Optional[str] = placeholder_source_tag(after)
        self.before_source: Optional[str] = placeholder_source_tag(before)
        # the domain aggregates (selection-gated) and the OPTIONAL extra GROUP BY columns
        # (beyond the match key, e.g. a per-status sub-total). Stored as tuples so the spec
        # is hashable and folds straight into the dedup key; empty ``aggregates`` means no
        # aggregate statement is issued.
        self.aggregates: Tuple[PgAggregate, ...] = tuple(aggregates)
        self.aggregate_group_by: Tuple[str, ...] = tuple(aggregate_group_by)
        # A connection pages forward XOR reverse; supplying BOTH a forward (first/after)
        # and a reverse (last/before) arg is undefined under the Relay spec. Reject it
        # loudly at construction rather than derive `reverse` loosely and silently page one
        # way (which would ignore the other args). Per Relay guidance, forward and reverse
        # paging args must not be combined.
        forward_arg = first is not None or after is not None
        reverse_arg = last is not None or before is not None
        if forward_arg and reverse_arg:
            raise ValueError(
                "ambiguous Relay pagination: forward (first/after) and reverse "
                "(last/before) args were both supplied; a connection pages forward XOR "
                "reverse — pass only one direction"
            )
        # reverse paging when last/before is in play; otherwise forward. The guard above
        # ensures the two are mutually exclusive here.
        self.reverse = reverse_arg
        # keep the raw after/before (a literal cursor str or a ``Placeholder``) so the decode
        # can be RE-RUN whenever the order mutates (add_order_term) OR a variable-derived cursor
        # is re-bound for the next cached request (rebind_placeholders) — decoding only at
        # construction would seek with stale values against a changed order or value. A literal
        # cursor's rebind is a no-op; ``None`` when the arg was absent.
        self.after = after
        self.before = before
        self._decode_cursors()
        # dep 0 is the key step; values[0] is the key column at execute time.
        self.add_dependency(key_step)
        self.seed_resource_customization(resource)

    @property
    def is_composite(self) -> bool:
        """Whether the match key spans more than one column (the tuple-IN path)."""
        return len(self.match_columns) > 1

    @property
    def match_column(self) -> str:
        """The lone match column (single-column fast path; raises if composite)."""
        if self.is_composite:
            raise ValueError(
                f"connection on {self.resource.name!r} is composite "
                f"({self.match_columns}); use match_columns"
            )
        return self.match_columns[0]

    def match_predicate(self, unique_keys: Optional[List[Any]] = None):
        """The batched key-match predicate: ``= ANY(:keys)`` (single) or tuple-IN (composite).

        Mirrors :meth:`PgSelectStep.match_predicate`: the composite tuple-IN bakes the
        list-of-tuples onto the bindparam at build time so the RawExecutor postcompile path
        can expand it; the single fast path keeps ``= ANY`` ($1::T[]) with ``keys`` bound at
        execute time and ignores ``unique_keys``.
        """
        if not self.is_composite:
            return column(self.match_columns[0]) == any_(
                bindparam("keys", expanding=False)
            )
        cols = [column(c) for c in self.match_columns]
        return tuple_(*cols).in_(
            bindparam("keys", value=unique_keys or [], expanding=True)
        )

    def _decode_cursors(self) -> None:
        """Decode the after/before cursors LOUDLY against the CURRENT order and bound values.

        Run at construction, whenever the order is mutated (add_order_term), AND whenever a
        variable-derived cursor is re-bound for a cached request (rebind_placeholders), so the
        seek values are always validated against the order actually emitted and the value
        actually bound. A ``Placeholder`` cursor is unwrapped to its current value first. The
        decode is digest-validated: a cursor minted under a different ordering — or a garbage
        cursor — raises a clear "minted for a different ordering" ``ValueError`` at plan time
        rather than seeking with stale values (an IndexError, or worse a SILENT mis-seek for a
        same-length order change). The decoded VALUES discriminate the dedup key ONLY for a
        LITERAL cursor; a variable-derived cursor keys off its source tag instead.
        """
        after_cursor = unwrap_placeholder(self.after)
        before_cursor = unwrap_placeholder(self.before)
        self.after_values: Optional[List[Any]] = (
            decode_keyset_cursor(after_cursor, self.order_by) if after_cursor else None
        )
        self.before_values: Optional[List[Any]] = (
            decode_keyset_cursor(before_cursor, self.order_by) if before_cursor else None
        )

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
        # the order just changed, so any after/before cursor decoded under the OLD order is
        # now invalid; re-decode against the new order. A now-incompatible cursor (different
        # digest) is rejected LOUDLY here instead of silently seeking the wrong page.
        self._decode_cursors()

    def set_first(self, first: Optional[Union[int, Placeholder]]) -> None:
        """Set the structured per-parent forward page size (literal int or ``Placeholder``)."""
        self.first = first

    def rebind_placeholders(self, values_by_source: Mapping[str, Any]) -> None:
        """Re-point this connection's WHERE, page-size and CURSOR placeholders to a cached request.

        Extends the base WHERE rebind (``super``) with the Relay pagination placeholders:

        * ``first`` / ``last`` page sizes — a variable-derived size is a :class:`Placeholder`
          re-pointed by source (a literal int is a no-op);
        * ``after`` / ``before`` cursors — a variable-derived cursor is a :class:`Placeholder`
          carrying the FIRST request's cursor string; the cached step decoded it into
          ``after_values`` / ``before_values`` at build, so on a HIT we re-point the sentinel
          AND RE-DECODE the new cursor against the current order (digest-validated, so a cursor
          from a different ordering still fails loud). A literal cursor is left verbatim and its
          decoded values stand — a no-op.
        """
        super().rebind_placeholders(values_by_source)
        self.first = rebind_pagination_value(self.first, values_by_source)
        self.last = rebind_pagination_value(self.last, values_by_source)
        self.after = rebind_pagination_value(self.after, values_by_source)
        self.before = rebind_pagination_value(self.before, values_by_source)
        # re-decode the (possibly re-pointed) cursors against the current order — the shared
        # helper unwraps the new placeholder value and digest-validates, so a cursor from a
        # different ordering still fails loud.
        self._decode_cursors()

    # ------------------------------------------------------------------ SQL build

    def effective_order(self) -> Tuple[OrderTerm, ...]:
        """The ORDER BY to emit: the request order, reversed for ``last``/``before``.

        Reverse paging walks the connection from the end: we flip every term's direction
        (and NULLS placement) so the window numbers rows from the tail, take the first
        ``last+1`` of that, then re-reverse the page in Python to restore the requested
        order.
        """
        if not self.reverse:
            return self.order_by
        return tuple(
            OrderTerm(t.column, not t.descending, _reverse_nulls(t))
            for t in self.order_by
        )

    def page_size(self) -> Optional[int]:
        """The active page size as a runtime int, unwrapping a variable-derived ``Placeholder``.

        ``last`` for reverse paging, ``first`` for forward; a ``Placeholder`` yields its
        request value. The page arithmetic (``page_limit`` / the one-extra-row probe in
        ``build_connection``) reads THIS, so a placeholder page size pages exactly like a
        literal — only the dedup key (which keeps the sentinel) differs.
        """
        return unwrap_placeholder(self.last if self.reverse else self.first)

    def page_limit(self) -> Optional[int]:
        """The per-partition row cap: the page size PLUS ONE extra (for hasNextPage)."""
        size = self.page_size()
        return None if size is None else size + 1

    def cursor_predicate(self):
        """The keyset WHERE predicate for the active cursor, or ``None`` when unpaged.

        Forward: rows strictly AFTER the ``after`` cursor. Reverse: rows strictly BEFORE
        the ``before`` cursor (built against the request order; the comparator flips
        internally). The values are plan-time-known (decoded in ``__init__``).
        """
        if self.reverse:
            if self.before_values is None:
                return None
            return keyset_where(
                self.order_by,
                self.before_values,
                after=False,
                column_types=self.resource.column_types,
            )
        if self.after_values is None:
            return None
        return keyset_where(
            self.order_by,
            self.after_values,
            after=True,
            column_types=self.resource.column_types,
        )

    def inner_predicates(self) -> List[Any]:
        """Host customization predicates AND the keyset predicate for the INNER WHERE."""
        predicates = list(self.where_predicates)
        cursor = self.cursor_predicate()
        if cursor is not None:
            predicates.append(cursor)
        return predicates

    def build_page_query(self, unique_keys: Optional[List[Any]] = None):
        """Build the batched window-sliced PAGE SELECT (ONE statement across parents).

        The inner query filters the key match (``= ANY(:keys)`` single / tuple-IN
        composite) AND the host predicates AND the keyset cursor predicate, then numbers
        each parent's rows ``row_number() OVER (PARTITION BY match ORDER BY <effective
        order>) AS __rn``. The outer keeps only ``__rn <= :page_limit`` (page size + 1
        extra, per partition) and orders by match column(s) then ``__rn`` so each parent's
        page rows come back in window order. Only the page rows (plus the one-extra probe)
        return — the slice is IN SQL. ``unique_keys`` supplies the composite IN list at
        build time (ignored by the single fast path).

        Computed attributes (``expression.label(name)`` over the table columns) are
        projected in the INNER (table-scope) select and re-selected by label in the outer,
        so each node row carries the computed value with no extra statement.
        """
        match_cols = [column(c) for c in self.match_columns]
        cols = [column(c) for c in self.resource.columns]
        computed = self.resource.computed_projections()
        computed_names = [c.name for c in computed]
        tbl = table(
            self.resource.table,
            *[column(c) for c in self.resource.columns],
            schema=self.resource.schema,
        )
        rn = (
            func.row_number()
            .over(
                partition_by=match_cols,
                order_by=order_clauses(self.effective_order()),
            )
            .label("__rn")
        )
        inner = (
            select(*cols, *computed, rn)
            .select_from(tbl)
            .where(self.match_predicate(unique_keys))
        )
        for predicate in self.inner_predicates():
            inner = inner.where(predicate)
        inner = inner.subquery()

        outer_cols = [inner.c[c] for c in self.resource.columns] + [
            inner.c[name] for name in computed_names
        ]
        # order by every match column (the partition key) then __rn so each parent's page
        # is contiguous and in window order.
        stmt = select(*outer_cols).order_by(
            *[inner.c[c] for c in self.match_columns], inner.c["__rn"]
        )
        if self.page_limit() is not None:
            # per-partition upper bound: __rn <= page_limit (bound param; page + 1 extra).
            stmt = stmt.where(inner.c["__rn"] <= bindparam("page_limit"))
        return stmt

    def build_count_query(self, unique_keys: Optional[List[Any]] = None):
        """Build the SEPARATE batched total aggregate: ``match, count(*) GROUP BY match``.

        Counts the FULL per-parent set (NOT affected by first/after) under the same host
        customization predicates as the page — but WITHOUT the keyset cursor predicate, so
        the total is the parent's whole count even on an empty terminal page. The match
        column(s) are projected each under their own name (so a composite key reconstructs
        its tuple) and grouped over; one statement for the whole bucket. ``unique_keys``
        supplies the composite IN list at build time.
        """
        match_cols = [column(c) for c in self.match_columns]
        tbl = table(
            self.resource.table,
            *[column(c) for c in self.resource.columns],
            schema=self.resource.schema,
        )
        stmt = (
            select(*match_cols, func.count().label("__total"))
            .select_from(tbl)
            .where(self.match_predicate(unique_keys))
            .group_by(*match_cols)
        )
        for predicate in self.where_predicates:
            stmt = stmt.where(predicate)
        return stmt

    @property
    def has_aggregates(self) -> bool:
        """Whether any domain aggregate is selected (gates the aggregate statement)."""
        return bool(self.aggregates)

    def aggregate_spec(self) -> Tuple[Tuple[str, Optional[str], str], ...]:
        """The aggregate SPEC as a hashable tuple of ``(function, column, alias)`` triples.

        The dedup discriminator for the aggregate statement: two connections whose
        aggregates differ in any function / column / alias — or whose ordering differs —
        produce DIFFERENT specs, so byte-different aggregate SQL never merges. Folded into
        both ``peer_key`` and ``dedup_params`` alongside the extra grouping.
        """
        return tuple((a.function, a.column, a.alias) for a in self.aggregates)

    def build_aggregate_query(self, unique_keys: Optional[List[Any]] = None):
        """Build the SEPARATE batched DOMAIN aggregate, mirroring :meth:`build_count_query`.

        ``match[, <group_by>], <agg projections> GROUP BY match[, <group_by>]`` over the
        FULL per-parent set: the SAME key match (single ``= ANY`` / composite tuple-IN) and
        the SAME host ``where_predicates`` (INCLUDING any compiled filter Condition) as the
        page query, but WITHOUT the keyset cursor predicate — so the aggregate covers the
        parent's whole set, correct even on an empty terminal page (exactly like the count).
        The match column(s) are projected each under their own name (so a composite key
        reconstructs its tuple); the optional ``aggregate_group_by`` columns are added to
        BOTH the projection and the GROUP BY, so a grouped aggregate yields one row per
        ``(match, group_by...)`` bucket. ``unique_keys`` supplies the composite IN list at
        build time.
        """
        match_cols = [column(c) for c in self.match_columns]
        group_cols = [column(c) for c in self.aggregate_group_by]
        projections = [a.projection() for a in self.aggregates]
        tbl = table(
            self.resource.table,
            *[column(c) for c in self.resource.columns],
            schema=self.resource.schema,
        )
        stmt = (
            select(*match_cols, *group_cols, *projections)
            .select_from(tbl)
            .where(self.match_predicate(unique_keys))
            .group_by(*match_cols, *group_cols)
        )
        for predicate in self.where_predicates:
            stmt = stmt.where(predicate)
        return stmt

    # ------------------------------------------------------------------- execute

    def count_row_key(self, row: Dict[str, Any]) -> Any:
        """The per-parent total's lookup key from a count row: scalar or column tuple."""
        return grouping_key(row, self.match_columns)

    def aggregate_values(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """The aggregate alias -> value mapping carried by one aggregate row.

        Projects out only the aggregate aliases (NOT the match / group_by key columns), so a
        per-parent aggregate dict carries just the computed values under their output keys. A
        ``min`` / ``max`` returns a value OF its source column's type, so it rides the SAME
        resource codec decode a page node gets (a min/max over a codec'd range/enum/timestamptz
        column is decoded, not surfaced raw). ``sum`` / ``avg`` / ``count`` produce a numeric
        result that is NOT the column's type, so they pass through undecoded.
        """
        return {
            a.alias: (
                self.resource.decode_value(a.column, row[a.alias])
                if a.function in ("min", "max") and a.column is not None
                else row[a.alias]
            )
            for a in self.aggregates
        }

    async def run_queries(
        self, unique_keys: List[Any]
    ) -> Tuple[List[Dict[str, Any]], Dict[Any, int], Dict[Any, List[Dict[str, Any]]]]:
        """Run the page query (always), the count (iff ``needs_total``), the aggregate (iff selected).

        Returns the page rows, a per-parent ``match -> total`` map (empty when ``needs_total``
        is unset — totalCount then reports 0 for unselected counts), and a per-parent
        ``match -> [aggregate row dicts]`` map (empty when no aggregate is selected; one entry
        per GROUP BY bucket, so an UNGROUPED aggregate is a single-element list). The
        single-column path binds ``keys`` at execute time; the composite path bakes the IN
        list at build time, so it passes no ``keys`` param. The aggregate is a THIRD statement
        at most — issued only when the selection set asks for one.
        """
        request = current_pg_request()
        params: Dict[str, Any] = {}
        if not self.is_composite:
            params["keys"] = unique_keys
        if self.page_limit() is not None:
            params["page_limit"] = self.page_limit()
        rows = await request.executor.run(
            self.build_page_query(unique_keys), params, settings=request.settings
        )

        totals: Dict[Any, int] = {}
        if self.needs_total:
            count_params: Dict[str, Any] = {}
            if not self.is_composite:
                count_params["keys"] = unique_keys
            count_rows = await request.executor.run(
                self.build_count_query(unique_keys),
                count_params,
                settings=request.settings,
            )
            totals = {self.count_row_key(r): r["__total"] for r in count_rows}

        aggregates: Dict[Any, List[Dict[str, Any]]] = {}
        if self.has_aggregates:
            agg_params: Dict[str, Any] = {}
            if not self.is_composite:
                agg_params["keys"] = unique_keys
            agg_rows = await request.executor.run(
                self.build_aggregate_query(unique_keys),
                agg_params,
                settings=request.settings,
            )
            # group the aggregate rows under the MATCH key (the scatter key); an extra
            # GROUP BY yields several rows per parent (one bucket each), an ungrouped
            # aggregate exactly one.
            for row in agg_rows:
                aggregates.setdefault(self.count_row_key(row), []).append(row)
        return rows, totals, aggregates

    def build_connection(
        self,
        rows: List[Dict[str, Any]],
        total: int,
        aggregate_rows: Sequence[Dict[str, Any]] = (),
    ) -> Dict[str, Any]:
        """Assemble one parent's page rows + total + aggregates into a Relay connection dict.

        ``rows`` are this parent's page rows in window order (already filtered + limited
        in SQL). With the one-extra probe present (``len(rows) > size``), the extra row is
        dropped and the "more" flag set. For reverse paging the page is re-reversed to
        restore the requested order; ``hasNextPage``/``hasPreviousPage`` are assigned per
        direction. ``total`` is the full per-parent count (from the separate aggregate),
        so it is correct even when the page is empty.

        ``aggregate_rows`` are this parent's rows from the SEPARATE aggregate statement (one
        per GROUP BY bucket; empty when no aggregate is selected). When ungrouped, the lone
        row's aggregate values are surfaced flat under ``aggregates``; with an extra GROUP BY
        each bucket's group_by key columns AND aggregate values are surfaced under
        ``aggregateGroups``. Both are absent (``None``) when no aggregate is selected, so an
        unaggregated connection's shape is byte-identical to one built with aggregates off.
        """
        # the active page size as a runtime int (unwrapping a variable-derived Placeholder),
        # so the one-extra-row probe is computed against the real size.
        size = self.page_size()
        has_extra = size is not None and len(rows) > size
        page = rows[:size] if has_extra else rows
        # reverse paging numbered from the tail; restore the requested (forward) order.
        if self.reverse:
            page = list(reversed(page))

        # the node carries every materialised attribute: the stored columns AND the
        # computed labels, each decoded via the resource codec (to_py). The cursor is
        # encoded from the RAW row (its order-column values), so encode it before decode.
        node_keys = list(self.resource.columns) + self.resource.computed
        edges = []
        for row in page:
            cursor = encode_keyset_cursor(self.order_by, row)
            decoded = self.resource.decode_row(row)
            node = {c: decoded[c] for c in node_keys}
            edges.append({"node": node, "cursor": cursor})
        nodes = [e["node"] for e in edges]

        if self.reverse:
            # reverse: the extra row probes for an EARLIER page (hasPreviousPage); a
            # `before` cursor means later rows exist (hasNextPage).
            has_previous = has_extra
            has_next = self.before_values is not None
        else:
            # forward: the extra row probes for a LATER page (hasNextPage); an `after`
            # cursor means earlier rows exist (hasPreviousPage).
            has_next = has_extra
            has_previous = self.after_values is not None

        page_info = {
            "hasNextPage": has_next,
            "hasPreviousPage": has_previous,
            "startCursor": edges[0]["cursor"] if edges else None,
            "endCursor": edges[-1]["cursor"] if edges else None,
        }
        connection: Dict[str, Any] = {
            "edges": edges,
            "nodes": nodes,
            "totalCount": total,
            "pageInfo": page_info,
        }
        if self.has_aggregates:
            # ungrouped: surface the lone bucket's aggregate values flat under `aggregates`
            # (an empty per-parent set has NO aggregate row, so sum/avg/min/max are null and
            # a column-less count is 0 — Postgres elides the GROUP row, we synthesise it).
            # grouped: one entry per bucket under `aggregateGroups`, each carrying its
            # group_by key columns plus the aggregate values.
            if self.aggregate_group_by:
                # the group-by key columns ride the SAME resource codec decode page nodes get,
                # so a GROUP BY over a codec'd (enum/range/timestamptz) column surfaces the
                # decoded key, not the raw asyncpg value — consistent with the node's column.
                connection["aggregateGroups"] = [
                    {
                        **{
                            c: self.resource.decode_value(c, row[c])
                            for c in self.aggregate_group_by
                        },
                        **self.aggregate_values(row),
                    }
                    for row in aggregate_rows
                ]
            else:
                connection["aggregates"] = (
                    self.aggregate_values(aggregate_rows[0])
                    if aggregate_rows
                    else self.empty_aggregates()
                )
        return connection

    def empty_aggregates(self) -> Dict[str, Any]:
        """The aggregate dict for a parent with NO matching rows (the empty-set defaults).

        Postgres returns no GROUP row for an empty set, so the ungrouped aggregate has no
        row to read; synthesise the SQL-faithful defaults — ANY ``count`` (column-less
        ``count(*)`` OR ``count(col)``) is 0, since count is the one aggregate immune to the
        empty-set→NULL rule (it counts rows/non-NULL values and returns bigint 0 over zero
        rows). Every other aggregate (``sum``/``avg``/``min``/``max``) is the SQL ``NULL`` of
        an empty aggregate.
        """
        return {
            a.alias: 0 if a.function == "count" else None
            for a in self.aggregates
        }

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        composite = self.is_composite
        keys = [normalize_lookup_key(k, composite) for k in values[0]]
        unique_keys = [k for k in dict.fromkeys(keys) if k is not None]
        if not unique_keys:
            empty = self.build_connection([], 0)
            return [dict(empty) for _ in range(count)]

        async def run():
            rows, totals, aggregates = await self.run_queries(unique_keys)
            by_key: Dict[Any, List[Dict[str, Any]]] = {}
            for row in rows:
                by_key.setdefault(grouping_key(row, self.match_columns), []).append(row)
            return [
                self.build_connection(
                    by_key.get(keys[i], []),
                    totals.get(keys[i], 0),
                    aggregates.get(keys[i], []),
                )
                for i in range(count)
            ]

        return run()

    def cursor_key(self, source: Optional[str], values: Optional[List[Any]]) -> Any:
        """The dedup-key component for a cursor: its SOURCE tag (variable) or its VALUES (literal).

        A variable-derived cursor keys off its stable ``source`` tag (``var:after``) so two
        requests of the same document share one key (a cache hit) while two different sources
        never merge — and the runtime value never enters the key. A literal cursor keeps its
        decoded VALUES (list -> tuple, hashable) so two pages differing only by a literal
        ``after``/``before`` still get different keys, exactly as before. ``None`` (unpaged on
        this side) keys as ``None``. The source path tags the tuple (``("var", source)``) so a
        source string can never coincidentally collide with a same-shaped values tuple.
        """
        if source is not None:
            return ("var", source)
        return tuple(values) if values is not None else None

    @property
    def peer_key(self) -> str:
        # the aggregate spec + extra grouping is APPENDED last (after needs_total +
        # customization_signature), per the unified APPEND-never-insert convention, so two
        # connections differing only by which aggregates they compute (or how they group)
        # never merge — they emit a byte-different aggregate statement. first/last render
        # their Placeholder source (not the value) when variable-derived; after/before key by
        # source tag (variable) or decoded values (literal) via cursor_key.
        return (
            f"pg_connection|{self.resource.qualified_table}|{self.match_columns!r}"
            f"|{self.order_by!r}|{self.first}|{self.last}"
            f"|{self.cursor_key(self.after_source, self.after_values)!r}"
            f"|{self.cursor_key(self.before_source, self.before_values)!r}|{self.needs_total}"
            f"|{self.customization_signature()!r}"
            f"|{self.aggregate_spec()!r}|{self.aggregate_group_by!r}"
        )

    def dedup_params(self) -> Tuple[Any, ...]:
        return (
            self.resource.qualified_table,
            self.match_columns,
            self.order_by,
            self.first,
            self.last,
            # the cursor key component: a variable cursor's stable source tag (value-agnostic,
            # so a cache hit across requests of the same document) or a literal cursor's
            # decoded VALUES (so two pages differing only by `after`/`before` never merge).
            self.cursor_key(self.after_source, self.after_values),
            self.cursor_key(self.before_source, self.before_values),
            self.needs_total,
            self.customization_signature(),
            # the aggregate spec + extra grouping fold in last (APPEND, never insert): the
            # aggregate statement's SQL is determined by which functions over which columns,
            # plus the GROUP BY, so two specs that differ MUST get different keys.
            self.aggregate_spec(),
            self.aggregate_group_by,
        )

    def get(self, attr: Any) -> Step:
        """Project a connection sub-field (``totalCount`` / ``pageInfo`` / ...)."""
        return access(self, (attr,))


def _reverse_nulls(term: OrderTerm) -> Optional[str]:
    """Flip a term's EFFECTIVE nulls placement for the reversed (last/before) order.

    Reversing the order reverses NULL placement too; emit it EXPLICITLY so the reversed
    ORDER BY and the BEFORE keyset comparator agree on where NULLs sit. Same rule as
    cursor._flip_nulls (kept local: a private one-liner is not worth a cross-module import).
    """
    return "last" if effective_nulls(term) == "first" else "first"


def connection(
    resource: PgResource,
    key_step: Step,
    match: Union[str, Sequence[str]],
    order_by: Sequence[Union[str, OrderTerm]],
    first: Optional[Union[int, Placeholder]] = None,
    after: Optional[Union[str, Placeholder]] = None,
    last: Optional[Union[int, Placeholder]] = None,
    before: Optional[Union[str, Placeholder]] = None,
    order_is_unique: bool = False,
    needs_total: bool = False,
    aggregates: Sequence[PgAggregate] = (),
    aggregate_group_by: Sequence[str] = (),
) -> PgConnectionStep:
    """Plan-helper: a batched, keyset-paged Relay connection over a hasMany lookup.

    ``match`` is the FK key — a single column name (fast path) or a composite tuple.
    ``aggregates`` (selection-gated via :func:`connection_aggregates`) requests the separate
    domain aggregate, optionally grouped by ``aggregate_group_by`` beyond the match key.
    """
    return PgConnectionStep(
        resource, key_step, match, order_by,
        first=first, after=after, last=last, before=before,
        order_is_unique=order_is_unique, needs_total=needs_total,
        aggregates=aggregates, aggregate_group_by=aggregate_group_by,
    )


__all__ = [
    "PgConnectionStep",
    "PgAggregate",
    "AGGREGATE_FUNCTIONS",
    "connection",
    "connection_needs_total",
    "connection_aggregates",
]
