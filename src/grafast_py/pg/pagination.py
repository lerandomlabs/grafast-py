"""The shared per-parent window slice: push FIRST/OFFSET into ONE batched statement.

A limited relation/connection must fetch ONLY each parent's page rows — never
fetch-all-then-slice-in-Python, and never a bucket-wide ``LIMIT`` (which would limit
the whole ``= ANY($1)`` result across ALL parents, not per parent). The fix is a
``row_number()`` window PARTITIONED BY the match column: each parent is numbered
independently, so a single outer ``WHERE __rn > :offset [AND __rn <= :offset + :first]``
slices every parent's page in ONE statement across the bucket.

:func:`window_slice` is that one emitter, used by the plain ``PgSelectStep`` (limited
find / relation). ``first`` / ``offset`` are bound as PARAMETERS (never inlined) and are
PER-PARTITION (per parent). ``__rn`` is an internal slicing/ordering column — it is NOT
projected into the result rows. (The Relay connection has its own keyset-sliced page query
in :mod:`grafast_py.pg.connection` plus a separate ``totalCount`` aggregate, so it does not
use this offset window.)
"""

from typing import Any, Optional, Sequence

from sqlalchemy import any_, bindparam, column, func, select, table
from sqlalchemy.sql import ColumnElement

from .ordering import OrderTerm, order_clauses


def window_slice(
    *,
    schema: str,
    table_name: str,
    columns: Sequence[str],
    match_column: str,
    order_by: Sequence[OrderTerm],
    first: Optional[int] = None,
    offset: int = 0,
    where_predicates: Sequence[ColumnElement] = (),
    computed: Sequence[ColumnElement] = (),
):
    """Build the per-parent window-sliced ``= ANY(:keys)`` SELECT (ONE statement).

    The inner query numbers each parent's rows ``row_number() OVER (PARTITION BY match
    ORDER BY <order>) AS __rn``; the outer keeps the page rows ``__rn > :offset`` (and
    ``__rn <= :offset + :first`` when ``first`` is set) and ORDERS BY ``__rn`` so each
    parent's page comes back in window order — a subquery alone does not guarantee output
    order. ``__rn`` is internal: it is used for the slice + order but NOT projected, so it
    never leaks into the row dicts ``AccessStep`` reads. The partition slices every parent
    independently in the single bucket statement.

    ``where_predicates`` are host customization predicates AND-combined onto the INNER
    WHERE alongside ``match = ANY(:keys)`` — BEFORE ``row_number()`` materialises, so each
    parent's page is numbered over the FILTERED set and per-parent paging respects the
    customization.

    ``computed`` are the resource's computed-column projections (``expression.label(name)``)
    over the TABLE columns: because they reference table columns they are evaluated in the
    INNER (table-scope) select and projected OUT through the subquery to the outer by their
    label, so each page row carries the computed value under its name with no extra
    statement.

    ``offset`` / ``first`` bind as the ``offset`` / ``first`` params (never inlined).
    """
    cols = [column(c) for c in columns]
    match = column(match_column)
    tbl = table(table_name, *[column(c) for c in columns], schema=schema)

    rn = (
        func.row_number()
        .over(partition_by=match, order_by=order_clauses(order_by))
        .label("__rn")
    )
    # computed expressions reference the TABLE columns, so they are projected in the INNER
    # (table-scope) select; the outer then re-selects them by label.
    inner_select = (
        select(*cols, *computed, rn)
        .select_from(tbl)
        .where(match == any_(bindparam("keys", expanding=False)))
    )
    for predicate in where_predicates:
        inner_select = inner_select.where(predicate)
    inner = inner_select.subquery()

    # project the resource columns AND the computed labels; __rn is internal (slice +
    # order), so the row dicts carry no window bookkeeping into downstream AccessStep /
    # nested relations.
    computed_names = [c.name for c in computed]
    outer_cols = [inner.c[c] for c in columns] + [
        inner.c[name] for name in computed_names
    ]
    stmt = (
        select(*outer_cols)
        .where(inner.c["__rn"] > bindparam("offset"))
        .order_by(inner.c["__rn"])
    )
    if first is not None:
        # per-partition upper bound: offset < __rn <= offset + first (bound params).
        stmt = stmt.where(
            inner.c["__rn"] <= bindparam("offset") + bindparam("first")
        )
    return stmt


def window_slice_params(first: Optional[int], offset: int) -> dict:
    """Return the bind values for :func:`window_slice` (``offset`` always, ``first`` iff set)."""
    params: dict[str, Any] = {"offset": offset}
    if first is not None:
        params["first"] = first
    return params


__all__ = ["window_slice", "window_slice_params"]
