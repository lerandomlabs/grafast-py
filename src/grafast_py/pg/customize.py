"""Batch-uniform query customization: WHERE predicates and the host seam.

A host narrows OUR batched skeleton (``match = ANY(:keys)`` or the ``row_number() OVER
(PARTITION BY match)`` window) without touching it: only UNIFORM additions, applied
identically to every parent's rows in the one statement. WHERE predicates are SQLAlchemy
Core ``ColumnElement`` expressions AND-combined onto the batched WHERE (never a raw
string — :func:`check_predicate` rejects that injection seam); ordering / ``first`` /
``offset`` forward to the step's structured surfaces. There is no raw-``LIMIT`` surface (a
bucket-wide ``LIMIT`` would limit the whole ``= ANY($1)`` result across parents).

A host inlines a GraphQL-arg value into the predicate (``.where(column("status") ==
args["status"])``): known at PLAN time, re-parameterised at EXECUTE time, and — crucially
— what DISCRIMINATES the dedup key. Two selects differing only by a host predicate must
NOT dedup-merge, but ``ColumnElement``s have no stable repr/hash, so the dedup key uses
:func:`predicate_key`: the predicate compiled with ``literal_binds`` so every value renders
INLINE (``status = 'published'`` vs ``status = 'draft'`` differ; a value-free compile would
collapse both to ``status = %(status_1)s`` and wrongly merge them). That compile is for the
KEY only — the step EXECUTES the predicate with its bindparams intact, never with
``literal_binds``. ``literal_binds`` is valid only because every bind carries a plan-time
value, which :func:`check_predicate` enforces (an unbound bind, or a bind reusing a
reserved skeleton name, fails loud).
"""

from typing import Any, Callable, List, Sequence, Union

from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import CompileError
from sqlalchemy.sql import ColumnElement, visitors
from sqlalchemy.sql.elements import BindParameter

from ..config import log
from ..step_model import Step
from .conditions import Condition, compile_condition
from .executor import current_pg_request
from .ordering import OrderTerm

# The bind names the batched skeleton itself owns; a host predicate may not reuse them
# (it would shadow the skeleton's own value at execute time).
RESERVED_BIND_NAMES = frozenset({"keys", "first", "offset"})


def check_predicate(predicate: Any) -> ColumnElement:
    """Validate a host WHERE predicate: a Core expression, fully bound, no reserved binds.

    Three fail-loud guards (raw string / unbound bind / reserved bind name) protect the
    batched statement; they apply identically to a per-plan ``.where()`` and to a
    resource ``select_customizer`` predicate:

    - a raw string (or any non-:class:`ColumnElement`) would be interpolated as opaque
      SQL — an injection seam — so it fails loud rather than reaching the query;
    - an UNBOUND bindparam carries a value not known at plan time. The dedup key renders
      values inline (``literal_binds``), which can only see plan-time values, so an
      unbound bind is unsupported — pass the value inline (``== args["x"]``) instead;
    - a bind reusing a RESERVED skeleton name (``keys`` / ``first`` / ``offset``) would
      collide with the batched skeleton's own params at execute time.
    """
    if isinstance(predicate, str):
        raise TypeError(
            "pg where() takes a SQLAlchemy Core predicate (a ColumnElement, e.g. "
            "column('status') == args['status']), never a raw string "
            f"(injection risk); got {predicate!r}"
        )
    if not isinstance(predicate, ColumnElement):
        raise TypeError(
            "pg where() takes a SQLAlchemy Core ColumnElement predicate; got "
            f"{type(predicate).__name__}"
        )
    for bind in visitors.iterate(predicate):
        if not isinstance(bind, BindParameter):
            continue
        if bind.key in RESERVED_BIND_NAMES:
            raise ValueError(
                f"pg where() predicate uses bind name {bind.key!r}, which collides "
                f"with a reserved skeleton bind ({', '.join(sorted(RESERVED_BIND_NAMES))}); "
                "rename the bind"
            )
        if bind.required:
            raise ValueError(
                f"pg where() predicate has an unbound bindparam {bind.key!r} (no "
                "plan-time value); a value-agnostic placeholder is not supported — "
                "pass the value inline, e.g. column('x') == args['x']"
            )
    return predicate


def predicate_key(predicate: ColumnElement) -> str:
    """A stable, content-based dedup key for a Core predicate (VALUE-included).

    Compiled with the Postgres dialect and ``literal_binds`` so every bound value renders
    INLINE: ``status = 'published'`` and ``status = 'draft'`` produce DIFFERENT strings,
    so two differently-filtered selects never dedup-merge, while identical predicates
    yield the identical string (and DO merge). Valid only because every bind carries a
    plan-time value — :func:`check_predicate` guarantees that before a predicate reaches
    here. The result is a hashable ``str`` that slots into the step's ``dedup_params``
    tuple (which dag.py places in a dict key). This compile is for the KEY only; the
    step executes the predicate with its bindparams intact, so execution stays
    parameterised (never run with ``literal_binds``).

    An exotic literal that no dialect can render inline (e.g. a non-UTF8 ``bytes`` value
    against a ``bytea`` column) raises ``CompileError``; rather than crash PLANNING we
    fall back to :func:`structural_predicate_key` — the value-free SQL plus the bound
    values' repr — which still distinguishes two different exotic predicates.
    """
    try:
        return str(
            predicate.compile(
                dialect=postgresql.dialect(),
                compile_kwargs={"literal_binds": True},
            )
        )
    except CompileError:
        return structural_predicate_key(predicate)


def structural_predicate_key(predicate: ColumnElement) -> str:
    """A literal-binds-free fallback dedup key: value-free SQL plus the bound values.

    Compiles the predicate WITHOUT ``literal_binds`` (so an unrenderable literal cannot
    raise) and appends a stable repr of the bound values. Two predicates with the same
    shape but different values yield different keys, so dedup stays correct for the exotic
    literals :func:`predicate_key` cannot inline.
    """
    compiled = predicate.compile(dialect=postgresql.dialect())
    params = tuple(sorted((name, repr(value)) for name, value in compiled.params.items()))
    return f"{compiled}|{params!r}"


class PgCustomizable(Step):
    """Shared customization state for a batched pg select / connection step.

    Carries the host's UNIFORM WHERE predicates — folded identically into every parent's
    rows of the one batched statement — plus the dedup signature that keeps two
    differently-customized selects from merging. ``seed_resource_customization`` seeds the
    predicates from the resource's ``select_customizer`` (resolved once against the
    per-request context) so resource-level scoping is present from construction.

    Subclasses (the concrete steps) own the SQL emission and per-parent paging surfaces;
    this base owns only what is identical across them.
    """

    def init_customization(self, predicates: Sequence[ColumnElement] = ()) -> None:
        """Seed the customization list (call from the subclass ``__init__``)."""
        # AND-combined onto the batched WHERE in insertion order (and_(...) operand order
        # is preserved in the compiled string, so a stable order keeps the dedup key
        # stable). Resource-customizer predicates come first, then per-plan .where()s.
        self.where_predicates: List[ColumnElement] = list(predicates)
        # the customization_signature tuple is content-derived and read by BOTH peer_key
        # and dedup_params; cache it and invalidate on every where_predicates mutation.
        self._signature_cache: Union[tuple, None] = None

    def seed_resource_customization(self, resource: Any) -> None:
        """Seed the WHERE list from a resource's ``select_customizer`` (call from ``__init__``).

        Resolves the customizer ONCE against the per-request context. The context is only
        read when the resource actually has a customizer, so a step built outside a request
        (e.g. a no-DB dedup test over an un-customized resource) needs no bound context.
        """
        customizer = resource.select_customizer
        if customizer is None:
            self.init_customization()
            return
        context = current_pg_request().context
        self.init_customization(resolve_customizer_predicates(customizer, context))

    def add_where(self, predicate: ColumnElement) -> None:
        """AND a validated UNIFORM Core predicate onto the batched WHERE."""
        self.where_predicates.append(check_predicate(predicate))
        self._signature_cache = None

    def where_tree(self, condition: "Condition") -> None:
        """Compile a structured filter :class:`Condition` and AND it onto the batched WHERE.

        A thin adaptor: it compiles the condition tree to a Core boolean predicate and
        folds it via :meth:`add_where`, so a compiled condition is JUST another peer in
        ``where_predicates``. It therefore inherits the value-discriminated dedup of
        :func:`predicate_key` unchanged — no new dedup discriminator, no skeleton change —
        and runs the same :func:`check_predicate` validation as a hand-built ``.where()``.
        """
        self.add_where(compile_condition(condition))

    def builder(self) -> "PgSelectQueryBuilder":
        """The host-facing query-level customization wrapper over this step."""
        return PgSelectQueryBuilder(self)

    def customization_signature(self) -> tuple:
        """A content-based, hashable dedup component for the customization (cached).

        A tuple of per-predicate :func:`predicate_key` strings in insertion order
        (resource-customizer predicates first, then per-plan ``.where()``s): equal
        predicate lists yield equal tuples and different VALUES differ, so byte-different
        statements never merge. Computed once per step (both ``peer_key`` and
        ``dedup_params`` read it) and invalidated when a ``.where()`` mutates the list.
        """
        if self._signature_cache is None:
            self._signature_cache = tuple(
                predicate_key(p) for p in self.where_predicates
            )
        return self._signature_cache


# Capability check for the builder: a step advertises which structured surfaces it
# supports by defining the corresponding method; the builder raises a CLEAR error (not a
# bare AttributeError) when a host calls a surface the wrapped step genuinely cannot
# support (e.g. a connection cannot set_offset — its paging is the after-cursor).
def require_capability(step: "Step", method: str, reason: str) -> None:
    if not hasattr(step, method):
        raise TypeError(
            f"{type(step).__name__} does not support {method}; {reason}"
        )


class PgSelectQueryBuilder:
    """The host-facing query-level customization seam over a pg select/connection step.

    A THIN wrapper exposing ONLY uniform, query-level mutations — there is structurally
    no way to add a per-parent condition or a raw ``LIMIT`` because those surfaces simply
    do not exist here. Every method mutates the wrapped step in place and returns ``self``
    so calls chain. The step folds each mutation into its batched SQL and, crucially, into
    its dedup key (so two differently-customized selects never merge).

    Hosts reach this via the step's ``.apply(callback)`` seam (``callback(builder)``); the
    builder is the only customization surface they are handed. A surface the wrapped step
    cannot support raises a CLEAR error naming the step, never a bare ``AttributeError``.
    """

    def __init__(self, step: "Step") -> None:
        self._step = step

    def where(self, predicate: ColumnElement) -> "PgSelectQueryBuilder":
        """AND a UNIFORM Core predicate onto the batched WHERE (raw string fails loud)."""
        self._step.add_where(check_predicate(predicate))
        return self

    def where_tree(self, condition: Condition) -> "PgSelectQueryBuilder":
        """Compile a structured filter :class:`Condition` and AND it onto the batched WHERE.

        The structured-filter counterpart to :meth:`where`: it hands the condition tree to
        the wrapped step's :meth:`PgCustomizable.where_tree`, which compiles and folds it
        through the SAME validated ``add_where`` path — so a compiled filter is just another
        uniform WHERE peer, value-discriminated in the dedup key like any other predicate.
        """
        self._step.where_tree(condition)
        return self

    def order_by(self, term: Union[str, OrderTerm]) -> "PgSelectQueryBuilder":
        """Append a UNIFORM ordering term (shared across the whole batch)."""
        self._step.add_order_term(term)
        return self

    def set_first(self, first: Union[int, None]) -> "PgSelectQueryBuilder":
        """Set the structured per-parent page size (never a bucket-wide LIMIT)."""
        require_capability(
            self._step,
            "set_first",
            "this step has no page-size surface",
        )
        self._step.set_first(first)
        return self

    def set_offset(self, offset: int) -> "PgSelectQueryBuilder":
        """Set the structured per-parent page offset."""
        require_capability(
            self._step,
            "set_offset",
            "its paging is the construction-time after-cursor; pass after= instead",
        )
        self._step.set_offset(offset)
        return self

    def apply(
        self, callback: Callable[["PgSelectQueryBuilder"], Any]
    ) -> "PgSelectQueryBuilder":
        """Hand the host this builder to apply its customizations, then return self."""
        callback(self)
        return self


def resolve_customizer_predicates(
    customizer: Union[Callable[[Any], Sequence[Any]], None],
    context: Any,
) -> List[ColumnElement]:
    """Resolve a resource ``select_customizer`` against the per-request context, ONCE.

    The customizer is the selectAuth analogue (soft-delete / tenant scoping / visibility):
    ``context -> list[Core predicate]``. Resolved once per planned step (at construction),
    every predicate validated as a Core expression (never a raw string, fully bound, no
    reserved bind name) and AND-combined onto EVERY batched select for the resource.
    ``None`` customizer yields no predicates.
    """
    if customizer is None:
        return []
    produced = customizer(context)
    predicates = [check_predicate(p) for p in produced]
    log.debug(
        "pg resolve select customizer",
        predicates=len(predicates),
        has_context=context is not None,
    )
    return predicates


__all__ = [
    "RESERVED_BIND_NAMES",
    "check_predicate",
    "predicate_key",
    "structural_predicate_key",
    "PgCustomizable",
    "PgSelectQueryBuilder",
    "resolve_customizer_predicates",
]
