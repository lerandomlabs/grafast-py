"""A small filter Condition AST that compiles to a Core boolean predicate.

A host expresses a structured WHERE filter — the GraphQL ``filter:`` arg shape
(``{and: [...]}`` / ``{or: [...]}`` / ``{not: ...}`` / ``{field: {eq: value}}``) — as a
:class:`Condition` tree rather than hand-building a SQLAlchemy expression. Each node
COMPILES to an ordinary Core ``ColumnElement``:

- :class:`Compare` — one column op value (``eq`` / ``ne`` / ``lt`` / ``le`` / ``gt`` /
  ``ge`` / ``in`` / ``like`` / ``ilike`` / ``is_null``), the leaf comparison;
- :class:`And` / :class:`Or` — the n-ary boolean combinators (an empty ``And`` is the
  always-true identity, an empty ``Or`` the always-false identity, so a degenerate filter
  arg compiles to a constant rather than failing);
- :class:`Not` — negation of one child.

The compiled tree is JUST a Core predicate, so it folds onto the EXISTING ``where_predicates``
list via :func:`grafast_py.pg.customize.check_predicate` exactly like a hand-built
``.where()``: it AND-combines onto the batched WHERE, and inherits the value-discriminated
dedup of :func:`grafast_py.pg.customize.predicate_key` for free (the compiled condition is
just another peer in ``where_predicates``, so ``customization_signature()`` already
discriminates two differently-valued filters). A compiled condition therefore changes NO
skeleton and adds NO new dedup discriminator — it rides the where path the composite-key
work already made tuple-aware.

Values are INLINED into the predicate at COMPILE time (the supported value-discriminated
form — a literal known at plan time, re-parameterised at execute), never a value-agnostic
placeholder. A leaf column name is bound to ``column(name)``; the predicate's own
:func:`check_predicate` validation (raw-string / unbound-bind / reserved-name guards) runs
when it folds into the WHERE list, so an injection seam cannot reach the query through a
Condition any more than through a raw ``.where()`` — provided ``value`` is a plain Python
literal (the shape an untrusted GraphQL scalar arrives as). A ``Compare`` value must NOT be a
SQLAlchemy SQL construct (``text()`` / ``literal_column(...)``): a literal binds as a
parameter, but a hand-passed SQL construct compiles as raw SQL exactly as it would through a
raw ``.where()`` — so never forward an untrusted value as a SQL construct.
"""

from typing import Any, List, Sequence

from sqlalchemy import and_, column, false, not_, or_, true
from sqlalchemy.sql import ColumnElement

# The leaf comparison operators a Compare node supports, mapped to the Core expression each
# emits. ``is_null`` takes no value (its truthiness picks IS NULL vs IS NOT NULL); every
# other operator inlines the value as a literal (value-discriminated in the dedup key).
_BINARY_OPS = {
    "eq": lambda col, value: col == value,
    "ne": lambda col, value: col != value,
    "lt": lambda col, value: col < value,
    "le": lambda col, value: col <= value,
    "gt": lambda col, value: col > value,
    "ge": lambda col, value: col >= value,
    "in": lambda col, value: col.in_(value),
    "like": lambda col, value: col.like(value),
    "ilike": lambda col, value: col.ilike(value),
}


class Condition:
    """Base of the filter AST: a node that COMPILES to a Core boolean ``ColumnElement``.

    Subclasses (:class:`Compare`, :class:`And`, :class:`Or`, :class:`Not`) implement
    :meth:`to_predicate`. A host never instantiates this base directly; it builds a tree of
    the concrete nodes and hands the root to ``builder.where_tree(...)`` (or a step's
    ``.where_tree(...)``), which compiles it and folds the result onto the batched WHERE.
    """

    def to_predicate(self) -> ColumnElement:
        """Compile this node to a Core boolean predicate (the WHERE fragment it denotes)."""
        raise NotImplementedError


class Compare(Condition):
    """A leaf comparison: ``column <op> value`` (or ``column IS [NOT] NULL`` for ``is_null``).

    ``field`` is the (stored) column name, bound to ``column(field)`` so it resolves against
    the batched table scope exactly like a hand-built ``.where(column(field) == ...)``.
    ``op`` is one of the supported operators; ``value`` is the literal it compares against
    (inlined at compile, value-discriminated in the dedup key). ``is_null`` ignores ``value``
    except for its truthiness: a truthy value emits ``IS NULL``, a falsy one ``IS NOT NULL``.
    ``in`` takes a sequence and emits an ``IN`` over its members; ``like`` / ``ilike`` emit a
    (case-sensitive / case-insensitive) ``LIKE`` over the string pattern ``value``.
    """

    def __init__(self, field: str, op: str, value: Any = None) -> None:
        if op != "is_null" and op not in _BINARY_OPS:
            raise ValueError(
                f"unsupported filter operator {op!r}; supported: "
                f"{', '.join(sorted([*_BINARY_OPS, 'is_null']))}"
            )
        self.field = field
        self.op = op
        self.value = value

    def to_predicate(self) -> ColumnElement:
        col = column(self.field)
        if self.op == "is_null":
            # the truthiness of value selects IS NULL vs IS NOT NULL (a structured filter's
            # ``{is_null: true}`` vs ``{is_null: false}``); no literal is inlined.
            return col.is_(None) if self.value else col.isnot(None)
        return _BINARY_OPS[self.op](col, self.value)


class And(Condition):
    """The n-ary AND of its child conditions (an empty AND is the always-true identity)."""

    def __init__(self, children: Sequence[Condition]) -> None:
        self.children: List[Condition] = list(children)

    def to_predicate(self) -> ColumnElement:
        # an empty AND is the identity TRUE so a degenerate ``{and: []}`` filter compiles to
        # a constant predicate rather than an empty (invalid) and_() — it then folds onto the
        # WHERE as a no-op clause.
        if not self.children:
            return true()
        return and_(*[c.to_predicate() for c in self.children])


class Or(Condition):
    """The n-ary OR of its child conditions (an empty OR is the always-false identity)."""

    def __init__(self, children: Sequence[Condition]) -> None:
        self.children: List[Condition] = list(children)

    def to_predicate(self) -> ColumnElement:
        # an empty OR is the identity FALSE (an ``{or: []}`` matches nothing), mirroring the
        # empty-AND identity above.
        if not self.children:
            return false()
        return or_(*[c.to_predicate() for c in self.children])


class Not(Condition):
    """The negation of a single child condition (``NOT (child)``)."""

    def __init__(self, child: Condition) -> None:
        self.child = child

    def to_predicate(self) -> ColumnElement:
        return not_(self.child.to_predicate())


def compile_condition(condition: Condition) -> ColumnElement:
    """Compile a :class:`Condition` tree to its Core boolean predicate.

    The single entry the step builders call: it just delegates to the root node's
    :meth:`Condition.to_predicate`. The result is an ordinary ``ColumnElement`` the
    customization path then validates (:func:`check_predicate`) and folds onto the WHERE —
    so a compiled condition is indistinguishable from a hand-built predicate downstream.
    """
    return condition.to_predicate()


__all__ = [
    "Condition",
    "Compare",
    "And",
    "Or",
    "Not",
    "compile_condition",
]
