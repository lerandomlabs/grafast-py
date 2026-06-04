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

Forward paging is ``first``/``after``; reverse is ``last``/``before`` (keyset on the
reversed order, ``last+1`` then re-reversed in Python). Sub-fields are plain
:class:`AccessStep` projections into the per-entry dict, and ``edges[].node`` is the row
dict, so nested relations under a node batch exactly like a plain row.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from graphql.language import FieldNode, FragmentSpreadNode, InlineFragmentNode
from sqlalchemy import any_, bindparam, column, func, select, table

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
from .resource import PgResource


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


class PgConnectionStep(PgCustomizable):
    """Batched Relay connection over a hasMany lookup keyed on ``match_column``.

    Host customization: UNIFORM WHERE predicates AND-combined onto the INNER WHERE
    (before ``row_number()`` materialises) — the resource ``select_customizer`` (resolved
    once against the per-request context) plus per-plan ``.where()``s via :meth:`builder`.
    Paging is the construction-time keyset cursor; the builder rejects
    ``set_offset`` with a clear error (no offset surface here).

    Forward paging is ``first``/``after``; reverse paging is ``last``/``before``. Exactly
    one direction is in play per step (a connection field is forward XOR reverse).
    ``needs_total`` (set by the plan resolver from the selection set) gates the separate
    count aggregate.
    """

    is_sync_and_safe = False

    def __init__(
        self,
        resource: PgResource,
        key_step: Step,
        match_column: str,
        order_by: Sequence[Union[str, OrderTerm]],
        first: Optional[int] = None,
        after: Optional[str] = None,
        last: Optional[int] = None,
        before: Optional[str] = None,
        order_is_unique: bool = False,
        needs_total: bool = False,
    ) -> None:
        super().__init__()
        self.resource = resource
        self.match_column = match_column
        self.order_is_unique = order_is_unique
        self.order_by: Tuple[OrderTerm, ...] = normalize_order(
            order_by, primary_key=resource.primary_key, order_is_unique=order_is_unique
        )
        resource.assert_order_terms_stored(self.order_by)
        self.first = first
        self.last = last
        self.needs_total = needs_total
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
        # decode the cursor LOUDLY against the current order (digest-validated); a cursor
        # minted under a different ordering — or a garbage cursor — raises at plan time
        # rather than being silently misapplied. The decoded VALUES are plan-time content
        # that discriminate the dedup key.
        self.after_values: Optional[List[Any]] = (
            decode_keyset_cursor(after, self.order_by) if after else None
        )
        self.before_values: Optional[List[Any]] = (
            decode_keyset_cursor(before, self.order_by) if before else None
        )
        # dep 0 is the key step; values[0] is the key column at execute time.
        self.add_dependency(key_step)
        self.seed_resource_customization(resource)

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

    def set_first(self, first: Optional[int]) -> None:
        """Set the structured per-parent forward page size."""
        self.first = first

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

    def page_limit(self) -> Optional[int]:
        """The per-partition row cap: the page size PLUS ONE extra (for hasNextPage)."""
        size = self.last if self.reverse else self.first
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

    def build_page_query(self):
        """Build the batched window-sliced PAGE SELECT (ONE statement across parents).

        The inner query filters ``match = ANY(:keys)`` AND the host predicates AND the
        keyset cursor predicate, then numbers each parent's rows ``row_number() OVER
        (PARTITION BY match ORDER BY <effective order>) AS __rn``. The outer keeps only
        ``__rn <= :page_limit`` (page size + 1 extra, per partition) and orders by match
        then ``__rn`` so each parent's page rows come back in window order. Only the page
        rows (plus the one-extra probe) return — the slice is IN SQL.

        Computed attributes (``expression.label(name)`` over the table columns) are
        projected in the INNER (table-scope) select and re-selected by label in the outer,
        so each node row carries the computed value with no extra statement.
        """
        match = column(self.match_column)
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
                partition_by=match,
                order_by=order_clauses(self.effective_order()),
            )
            .label("__rn")
        )
        inner = (
            select(*cols, *computed, rn)
            .select_from(tbl)
            .where(match == any_(bindparam("keys", expanding=False)))
        )
        for predicate in self.inner_predicates():
            inner = inner.where(predicate)
        inner = inner.subquery()

        outer_cols = [inner.c[c] for c in self.resource.columns] + [
            inner.c[name] for name in computed_names
        ]
        stmt = select(*outer_cols).order_by(
            inner.c[self.match_column], inner.c["__rn"]
        )
        if self.page_limit() is not None:
            # per-partition upper bound: __rn <= page_limit (bound param; page + 1 extra).
            stmt = stmt.where(inner.c["__rn"] <= bindparam("page_limit"))
        return stmt

    def build_count_query(self):
        """Build the SEPARATE batched total aggregate: ``match, count(*) GROUP BY match``.

        Counts the FULL per-parent set (NOT affected by first/after) under the same host
        customization predicates as the page — but WITHOUT the keyset cursor predicate, so
        the total is the parent's whole count even on an empty terminal page. One statement
        for the whole bucket.
        """
        match = column(self.match_column)
        tbl = table(
            self.resource.table,
            *[column(c) for c in self.resource.columns],
            schema=self.resource.schema,
        )
        stmt = (
            select(match.label("__match"), func.count().label("__total"))
            .select_from(tbl)
            .where(match == any_(bindparam("keys", expanding=False)))
            .group_by(match)
        )
        for predicate in self.where_predicates:
            stmt = stmt.where(predicate)
        return stmt

    # ------------------------------------------------------------------- execute

    async def run_queries(
        self, unique_keys: List[Any]
    ) -> Tuple[List[Dict[str, Any]], Dict[Any, int]]:
        """Run the page query (always) and the count query (iff ``needs_total``).

        Returns the page rows and a per-parent ``match -> total`` map (empty when
        ``needs_total`` is unset — totalCount then reports 0 for unselected counts).
        """
        request = current_pg_request()
        params: Dict[str, Any] = {"keys": unique_keys}
        if self.page_limit() is not None:
            params["page_limit"] = self.page_limit()
        rows = await request.executor.run(
            self.build_page_query(), params, settings=request.settings
        )

        totals: Dict[Any, int] = {}
        if self.needs_total:
            count_rows = await request.executor.run(
                self.build_count_query(),
                {"keys": unique_keys},
                settings=request.settings,
            )
            totals = {r["__match"]: r["__total"] for r in count_rows}
        return rows, totals

    def build_connection(
        self, rows: List[Dict[str, Any]], total: int
    ) -> Dict[str, Any]:
        """Assemble one parent's page rows + total into a Relay connection dict.

        ``rows`` are this parent's page rows in window order (already filtered + limited
        in SQL). With the one-extra probe present (``len(rows) > size``), the extra row is
        dropped and the "more" flag set. For reverse paging the page is re-reversed to
        restore the requested order; ``hasNextPage``/``hasPreviousPage`` are assigned per
        direction. ``total`` is the full per-parent count (from the separate aggregate),
        so it is correct even when the page is empty.
        """
        size = self.last if self.reverse else self.first
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
        return {
            "edges": edges,
            "nodes": nodes,
            "totalCount": total,
            "pageInfo": page_info,
        }

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        keys = values[0]
        unique_keys = [k for k in dict.fromkeys(keys) if k is not None]
        if not unique_keys:
            empty = self.build_connection([], 0)
            return [dict(empty) for _ in range(count)]

        async def run():
            rows, totals = await self.run_queries(unique_keys)
            by_key: Dict[Any, List[Dict[str, Any]]] = {}
            for row in rows:
                by_key.setdefault(row[self.match_column], []).append(row)
            return [
                self.build_connection(
                    by_key.get(keys[i], []), totals.get(keys[i], 0)
                )
                for i in range(count)
            ]

        return run()

    @property
    def peer_key(self) -> str:
        return (
            f"pg_connection|{self.resource.qualified_table}|{self.match_column}"
            f"|{self.order_by!r}|{self.first}|{self.last}"
            f"|{self.after_values!r}|{self.before_values!r}|{self.needs_total}"
            f"|{self.customization_signature()!r}"
        )

    def dedup_params(self) -> Tuple[Any, ...]:
        return (
            self.resource.qualified_table,
            self.match_column,
            self.order_by,
            self.first,
            self.last,
            # decoded cursor VALUES are plan-time content, so they discriminate the key
            # (two pages differing only by `after`/`before` must not merge); list ->
            # tuple to stay hashable in the dedup tuple.
            tuple(self.after_values) if self.after_values is not None else None,
            tuple(self.before_values) if self.before_values is not None else None,
            self.needs_total,
            self.customization_signature(),
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
    match_column: str,
    order_by: Sequence[Union[str, OrderTerm]],
    first: Optional[int] = None,
    after: Optional[str] = None,
    last: Optional[int] = None,
    before: Optional[str] = None,
    order_is_unique: bool = False,
    needs_total: bool = False,
) -> PgConnectionStep:
    """Plan-helper: a batched, keyset-paged Relay connection over a hasMany lookup."""
    return PgConnectionStep(
        resource, key_step, match_column, order_by,
        first=first, after=after, last=last, before=before,
        order_is_unique=order_is_unique, needs_total=needs_total,
    )


__all__ = [
    "PgConnectionStep",
    "connection",
    "connection_needs_total",
]
