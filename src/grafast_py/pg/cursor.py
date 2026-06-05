"""Keyset (seek) cursors and the keyset WHERE comparator for Relay connections.

A keyset cursor encodes the ORDER-COLUMN VALUES of a row plus a short digest of the
order spec it was minted under, base64-wrapped over a JSON array:

    cursor = base64(json([digest, v1, v2, ..., vn]))

where ``v1..vn`` are the row's values for the n order columns (in order) and ``digest``
is a short sha256 hex of the canonical order spec (the tuple of ``(column, descending,
effective-nulls)`` for every term). Decoding VALIDATES the digest against the current
order and the value count against n, raising LOUDLY on mismatch — a cursor minted for a
different ordering is REJECTED, never silently misapplied (no decode-to-0).

:func:`keyset_where` turns a decoded cursor into a SQLAlchemy predicate selecting rows
strictly AFTER / BEFORE the cursor row in the order. It is direction- AND nulls-aware,
built as the recursive per-key comparator

    (k1 AFTER v1) OR (k1 NOTDISTINCT v1 AND ((k2 AFTER v2) OR (k2 NOTDISTINCT v2 ...)))

which is correct for MIXED direction + nullable columns (a row-wise ``tuple_()``
comparison is not). Values bind as PARAMS (never inlined): native ints/strings bind
directly with the column's type; text-origin non-native values (datetime/Decimal,
stringified losslessly at encode time) bind via ``cast(bindparam(String) AS <coltype>)``
so Postgres coerces the text to the column type at execution.
"""

import base64
import hashlib
import json
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, List, Mapping, Optional, Sequence

from sqlalchemy import and_, bindparam, cast, column, or_
from sqlalchemy.sql import ColumnElement
from sqlalchemy.types import TypeEngine

from .ordering import OrderTerm

# A keyset cursor binds its values under per-column param names; prefix them so they can
# never collide with the batched skeleton's reserved binds (keys/first/offset) or a host
# predicate's binds. One predicate is built per bucket (batch-uniform), so static names
# are safe.
_KEYSET_BIND_PREFIX = "__ks_"


def effective_nulls(term: OrderTerm) -> str:
    """Resolve a term's EFFECTIVE NULLS placement (``"first"`` / ``"last"``).

    Postgres defaults are ASC => NULLS LAST, DESC => NULLS FIRST; an explicit
    ``OrderTerm.nulls`` overrides. The keyset comparator and the emitted ORDER BY must
    agree on this, so it is resolved on BOTH sides from the same rule.
    """
    if term.nulls is not None:
        return term.nulls
    return "first" if term.descending else "last"


def order_digest(order_terms: Sequence[OrderTerm]) -> str:
    """A short sha256 hex digest of the canonical order spec.

    Derived from each term's ``(column, descending, effective-nulls)`` so two cursors
    minted under different orderings carry different digests — a cursor used under a
    DIFFERENT order is rejected at decode time. Short (12 hex chars) keeps the cursor
    compact while staying collision-safe for the handful of orderings a field can have.
    """
    canonical = [
        (term.column, term.descending, effective_nulls(term)) for term in order_terms
    ]
    raw = json.dumps(canonical, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def encode_value(value: Any) -> Any:
    """JSON-serialise one order-column value losslessly.

    int/str/bool/None pass through as native JSON; non-native types stringify without
    loss — datetime/date/time via ``.isoformat()`` (ISO-8601, tz offset included),
    Decimal via ``str`` (NOT float — avoids binary rounding). The text strings round-trip
    back as-is and are bound through ``cast(text AS <coltype>)`` so Postgres coerces them.
    """
    if value is None or isinstance(value, (int, str, bool)):
        return value
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    # any other type: stringify losslessly and let PG cast it back at bind time.
    return str(value)


def encode_keyset_cursor(order_terms: Sequence[OrderTerm], row: dict) -> str:
    """Encode a keyset cursor for ``row`` under ``order_terms``.

    The cursor carries the order digest followed by the row's value for each order
    column (in order). Base64 over a compact JSON array keeps it opaque per the Relay
    contract.
    """
    payload: List[Any] = [order_digest(order_terms)]
    for term in order_terms:
        if term.column not in row:
            # the cursor is built from the row's order-column VALUES, so every order
            # column must be projected into the row. A missing one means the connection
            # ordered by a column that is not a stored/selected attribute — name it here
            # rather than let a bare KeyError surface deep in execute.
            raise ValueError(
                f"cannot encode keyset cursor: order column {term.column!r} is not "
                f"present in the row (available: {sorted(row)}); order only by columns "
                "projected into the resource"
            )
        payload.append(encode_value(row[term.column]))
    raw = json.dumps(payload, separators=(",", ":")).encode()
    return base64.b64encode(raw).decode()


def decode_keyset_cursor(
    cursor: str, order_terms: Sequence[OrderTerm]
) -> List[Any]:
    """Decode + VALIDATE a keyset cursor against the current order; return its values.

    Raises ``ValueError`` LOUDLY on any mismatch — a malformed/garbage cursor, a digest
    that does not match the current order (a cursor minted under a DIFFERENT ordering),
    or the wrong number of values. There is NO silent decode-to-0: a cursor that does not
    belong to this ordering must be rejected, not misapplied.
    """
    try:
        raw = base64.b64decode(cursor.encode(), validate=True)
        payload = json.loads(raw.decode())
    except (ValueError, UnicodeDecodeError) as err:
        raise ValueError(f"malformed keyset cursor {cursor!r}: {err}") from err

    if not isinstance(payload, list) or not payload:
        raise ValueError(f"malformed keyset cursor {cursor!r}: not a value array")

    digest, *values = payload
    expected = order_digest(order_terms)
    if digest != expected:
        raise ValueError(
            "keyset cursor was minted for a different ordering "
            f"(digest {digest!r} != current {expected!r}); reject rather than misapply"
        )
    if len(values) != len(order_terms):
        raise ValueError(
            f"keyset cursor has {len(values)} values but the order has "
            f"{len(order_terms)} columns"
        )
    return values


def bind_value(
    value: Any,
    index: int,
    sql_type: Optional[TypeEngine],
) -> ColumnElement:
    """Bind one decoded cursor value as a param for the order column at ``index``.

    When the column declares a SQL type (``sql_type``, e.g. timestamptz / numeric for a
    non-native column), bind the value and CAST it to that type — the encoder stringified
    datetime/Decimal losslessly, so PG coerces the ISO/decimal TEXT (``CAST(:p AS
    TIMESTAMP WITH TIME ZONE)`` / ``CAST(:p AS NUMERIC)``) back to the column type at
    execution. With no declared type (the common case — int/text columns) the decoded
    value binds NATIVELY: a native int/str matches the column directly, so no cast is
    needed and none is emitted. Values always bind as PARAMS, never inlined.
    """
    name = f"{_KEYSET_BIND_PREFIX}{index}"
    if sql_type is not None:
        return cast(bindparam(name, value=value), sql_type)
    return bindparam(name, value=value)


def column_after(
    term: OrderTerm,
    bound: ColumnElement,
    value: Any,
    column_expr: Optional[ColumnElement] = None,
) -> ColumnElement:
    """The NULL-aware STRICTLY-AFTER predicate for one column in the order's direction.

    AFTER means "comes later in the ORDER BY". Resolves the term's effective NULLS
    placement, then per the probe-confirmed truth table:

      ASC  + NULLS LAST  : v non-null => (col IS NULL OR col > v);   v NULL => FALSE
      ASC  + NULLS FIRST : v non-null => (col IS NOT NULL AND col > v); v NULL => col IS NOT NULL
      DESC + NULLS FIRST : v non-null => (col IS NOT NULL AND col < v); v NULL => col IS NOT NULL
      DESC + NULLS LAST  : v non-null => (col IS NULL OR col < v);   v NULL => FALSE

    The BEFORE direction (reverse paging) is obtained by the caller flipping each term's
    ``descending`` before building the comparator, so this one function serves both.
    ``column_expr`` overrides the column the comparison runs against — used by the
    pgUnionAll keyset, where the ``__typename`` discriminator term is a per-branch LITERAL
    (``literal('Article')``), not a table column, so the seek runs against that branch's
    constant tag rather than a non-existent ``__typename`` column.
    """
    col = column(term.column) if column_expr is None else column_expr
    placement = effective_nulls(term)
    value_is_null = value is None

    if not term.descending and placement == "last":  # ASC NULLS LAST
        if value_is_null:
            return col != col  # FALSE: a trailing NULL has nothing strictly after it
        return or_(col.is_(None), col > bound)
    if not term.descending and placement == "first":  # ASC NULLS FIRST
        if value_is_null:
            return col.isnot(None)
        return and_(col.isnot(None), col > bound)
    if term.descending and placement == "first":  # DESC NULLS FIRST
        if value_is_null:
            return col.isnot(None)
        return and_(col.isnot(None), col < bound)
    # DESC NULLS LAST
    if value_is_null:
        return col != col  # FALSE
    return or_(col.is_(None), col < bound)


def keyset_where(
    order_terms: Sequence[OrderTerm],
    values: Sequence[Any],
    *,
    after: bool,
    column_types: Optional[Mapping[str, TypeEngine]] = None,
    column_exprs: Optional[Mapping[str, ColumnElement]] = None,
) -> ColumnElement:
    """Build the keyset predicate selecting rows strictly AFTER/BEFORE the cursor row.

    ``after=True`` selects rows that come LATER in ``order_terms`` than the cursor
    (forward paging); ``after=False`` selects rows that come EARLIER (reverse paging,
    against the already-reversed order). The recursive per-key comparator

      gt0 OR (eq0 AND (gt1 OR (eq1 AND (... gtN))))

    is NULL-safe on both the row column and the cursor value: the equality rungs use
    ``IS NOT DISTINCT FROM`` (so NULL==NULL ties let the recursion descend and a NULL
    cursor key degenerates to ``col IS NULL``), and each ``gt`` is :func:`column_after`'s
    direction- and nulls-aware strictly-after predicate. SQL precedence (AND > OR) makes
    SQLAlchemy's unparenthesised top-level OR-chain parse as the intended nesting.

    Values bind as params; for ``after=False`` each term's direction is flipped so the
    SAME comparator yields the BEFORE predicate. ``column_types`` maps a column name to
    its SQL type for non-native (datetime/Decimal) columns so the text-origin cursor value
    is cast back to the column type (see :func:`bind_value`); native columns need none.
    ``column_exprs`` maps an order column NAME to the column expression the comparison runs
    against, overriding the default ``column(name)`` — the pgUnionAll keyset uses it for the
    per-branch ``__typename`` discriminator term, which is a constant ``literal(type_name)``
    on each leg rather than a real table column.
    """
    types = column_types or {}
    exprs = column_exprs or {}
    terms = (
        list(order_terms)
        if after
        else [
            OrderTerm(t.column, not t.descending, _flip_nulls(t))
            for t in order_terms
        ]
    )
    bounds = [
        bind_value(values[i], i, types.get(term.column))
        for i, term in enumerate(terms)
    ]
    return _recursive_after(terms, list(values), bounds, 0, exprs)


def _recursive_after(
    terms: Sequence[OrderTerm],
    values: Sequence[Any],
    bounds: Sequence[ColumnElement],
    index: int,
    column_exprs: Mapping[str, ColumnElement],
) -> ColumnElement:
    """Recursively build ``gtI OR (eqI AND rest)`` from term ``index`` onward."""
    term = terms[index]
    expr = column_exprs.get(term.column)
    col = column(term.column) if expr is None else expr
    gt = column_after(term, bounds[index], values[index], expr)
    if index == len(terms) - 1:
        return gt
    eq = col.is_not_distinct_from(bounds[index])
    rest = _recursive_after(terms, values, bounds, index + 1, column_exprs)
    return or_(gt, and_(eq, rest))


def _flip_nulls(term: OrderTerm) -> Optional[str]:
    """Flip a term's EFFECTIVE nulls placement for the reversed (BEFORE) order.

    Reversing the order reverses NULL placement too, so the BEFORE comparator built on
    the flipped-direction terms must carry the flipped effective placement EXPLICITLY
    (the default for the flipped direction would otherwise differ). Returns ``"first"`` /
    ``"last"`` (always explicit) so :func:`effective_nulls` does not re-derive a default.
    Same rule as connection._reverse_nulls (kept local: a private one-liner is not worth a
    cross-module import).
    """
    return "last" if effective_nulls(term) == "first" else "first"


__all__ = [
    "effective_nulls",
    "order_digest",
    "encode_value",
    "encode_keyset_cursor",
    "decode_keyset_cursor",
    "keyset_where",
    "column_after",
]
