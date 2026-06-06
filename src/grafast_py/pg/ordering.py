"""Structured ORDER BY specs: the single source of truth for row ordering.

A :class:`OrderTerm` is one ``ORDER BY`` column with its direction and optional
NULLS placement. :func:`normalize_order` turns the loose, host-facing order spec
(a sequence of column names and/or :class:`OrderTerm`s) into a canonical
``tuple[OrderTerm, ...]`` — string shorthand becomes an ascending term, and unless
the order is declared unique it APPENDS the primary key as a final ascending
tie-break. That canonical tuple is what every pg step stores: it drives the emitted
SQL (:func:`order_clauses`, used by BOTH the plain-select ORDER BY and the connection
window ORDER BY) AND the dedup keys, so two selects differing only in ordering never
merge while two with the same effective order always do.

The PK tie-break is not cosmetic: a non-unique order produces non-deterministic row
output, breaks the connection ``row_number()`` window's stability, and would make a
future keyset cursor ambiguous. Appending the PK makes the total order deterministic.
:class:`OrderTerm` is frozen (hashable, stable ``repr``) so the normalized tuple is a
reliable component of the structural dedup key.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

from sqlalchemy import column


@dataclass(frozen=True)
class OrderTerm:
    """One ORDER BY column: its name, direction, and optional NULLS placement.

    ``nulls`` is ``None`` (DB default placement), ``"first"`` or ``"last"``. Frozen so
    it is hashable and has a stable ``repr`` — both required for it to participate in
    the step dedup key.
    """

    column: str
    descending: bool = False
    nulls: Optional[str] = None

    def __post_init__(self) -> None:
        if self.nulls not in (None, "first", "last"):
            raise ValueError(
                f"OrderTerm.nulls must be None, 'first' or 'last', got {self.nulls!r}"
            )


def normalize_order(
    order_by: Optional[Sequence[Union[str, OrderTerm]]],
    *,
    primary_key: str,
    order_is_unique: bool = False,
) -> Tuple[OrderTerm, ...]:
    """Canonicalise a loose order spec to a ``tuple[OrderTerm, ...]``.

    Each entry is a column name (-> ascending :class:`OrderTerm`) or an explicit
    :class:`OrderTerm`. Unless ``order_is_unique`` is set, the ``primary_key`` is
    appended as a final ascending tie-break when no term already orders by it — making
    the total order deterministic for correct window ``ROW_NUMBER`` and future keyset
    cursors. The result is never empty: even with ``order_is_unique`` and no terms the
    primary key is appended, so a select is never left unordered (an empty ORDER BY is
    non-deterministic and breaks the window ``ROW_NUMBER`` / keyset).
    """
    terms: List[OrderTerm] = []
    for entry in order_by or ():
        terms.append(entry if isinstance(entry, OrderTerm) else OrderTerm(entry))

    if not order_is_unique and not any(t.column == primary_key for t in terms):
        terms.append(OrderTerm(primary_key))
    if not terms:  # floor: a select is never left unordered
        terms.append(OrderTerm(primary_key))

    return tuple(terms)


def order_clauses(terms: Sequence[OrderTerm]) -> list:
    """Emit SQLAlchemy ordering elements for ``terms`` (one shared emitter).

    Used by both the plain-select ORDER BY and the connection window ORDER BY so the
    direction / NULLS handling lives in exactly one place.
    """
    clauses = []
    for term in terms:
        clause = column(term.column)
        if term.descending:
            clause = clause.desc()
        if term.nulls == "first":
            clause = clause.nulls_first()
        elif term.nulls == "last":
            clause = clause.nulls_last()
        clauses.append(clause)
    return clauses


__all__ = ["OrderTerm", "normalize_order", "order_clauses"]
