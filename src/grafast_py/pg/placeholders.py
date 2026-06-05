"""Value-agnostic WHERE placeholders: a source-tagged, value-LESS-in-the-key bindparam.

A host filters by a GraphQL arg value by INLINING it (``.where(column("status") ==
args["status"])``): plan-time-known and value-DISCRIMINATED in the dedup key (two values
-> two keys -> never merge). That is correct but VALUE-specific: the compiled key carries
the literal, so the plan cannot be reused across requests with a different value.

When the value came from a GraphQL ``$variable`` (``field_args.is_variable("status")``),
the host instead builds a PLACEHOLDER: ``.where(column("status") == pg_placeholder(
field_args.source("status"), args["status"]))``. A placeholder is a SQLAlchemy
``bindparam`` that

  * carries the request's value (``value=``) so it rides ``compiled.params`` through BOTH
    executors at EXECUTE time with nothing new needed — exactly like today's inlined value,
    only never rendered inline; and
  * carries a stable SOURCE tag (``"var:status"`` — the GraphQL variable name, NOT a
    per-request-unique id) so the dedup key discriminates by the placeholder's IDENTITY,
    never by its runtime value.

The dedup key for a placeholder predicate is the VALUE-AGNOSTIC compile (``status =
%(name)s``) plus the sorted source tags — so two requests of the same document produce the
SAME key (they share one plan / cache entry), two DIFFERENT variable sources produce
DIFFERENT keys (never merge), and a placeholder never merges with a coincidentally
equal-valued inlined literal (``$1`` vs an inlined ``'published'`` are different SQL). A
LITERAL predicate keeps today's value-included ``literal_binds`` key UNCHANGED — so every
existing merge/count is byte-identical; only a placeholder-bearing predicate takes the new
path. See :func:`grafast_py.pg.customize.predicate_key` for where the two paths diverge.

The source tag lives on the bindparam via a side attribute (``_grafast_placeholder_source``)
that :meth:`PgCustomizable.add_where` reads to populate the per-step ``placeholder_binds``
registry; the predicate-key compile then knows which binds are placeholders (value-agnostic)
and which are ordinary literals (value-included).
"""

import itertools
from typing import Any, Mapping, Optional

from sqlalchemy import bindparam
from sqlalchemy.sql.elements import BindParameter


class Placeholder:
    """A source-tagged sentinel for a NON-predicate pagination value (``first`` / ``offset``
    / ``after`` / ``before``) that came from a GraphQL ``$variable``.

    Unlike a WHERE value (which the host wraps as a Core :func:`pg_placeholder` bindparam),
    a pagination value is a plain Python scalar (an ``int`` ``first`` / ``offset``) or a
    cursor ``str`` (``after`` / ``before``) stored directly on the step — it is NOT a
    ``ColumnElement`` that can carry a side attribute, and the step already binds it as a
    PARAMETER at execute time (``window_slice`` binds ``first`` / ``offset``; the keyset
    comparator binds the decoded cursor values). So the SQL is ALREADY value-agnostic for
    these; the only thing that carries the literal is the DEDUP KEY (``first`` / ``offset``
    in ``peer_key`` / ``dedup_params``, the decoded ``after_values`` / ``before_values``).

    This sentinel makes that key value-AGNOSTIC when the value is variable-derived: it
    carries the request's ``value`` (so the step unwraps it at SQL-build / execute time, the
    page math, and the cursor decode, all unchanged) but its ``__eq__`` / ``__hash__`` /
    ``__repr__`` key ONLY off the stable ``source`` tag (``"var:limit"`` — the variable name).
    So two requests of the same document produce the SAME dedup key (a cache hit / merge),
    two DIFFERENT variable sources never merge, and a placeholder never equals a literal
    ``int`` / cursor of a coincidentally equal value (a ``Placeholder`` is never ``==`` a bare
    ``int``). A LITERAL ``first=5`` stays the bare ``int`` 5 (value-included) so literal pages
    never merge with placeholder pages — exactly the WHERE-predicate split, applied to the
    pagination surface.

    A step UNWRAPS via :func:`unwrap_placeholder` everywhere it needs the runtime value, and
    keeps the sentinel itself in ``peer_key`` / ``dedup_params`` so the key stays source-keyed.
    """

    __slots__ = ("source", "value")

    def __init__(self, source: str, value: Any) -> None:
        self.source = source
        self.value = value

    def __eq__(self, other: Any) -> bool:
        # key off the SOURCE tag ONLY (never the runtime value): two same-source
        # placeholders are equal (they merge / share a cache entry) regardless of value, and
        # a placeholder is NEVER equal to a bare literal value — `Placeholder("var:x", 5) !=
        # 5` — so a placeholder page never merges with a coincidentally equal literal page.
        if not isinstance(other, Placeholder):
            return NotImplemented
        return self.source == other.source

    def __hash__(self) -> int:
        # hash off the source tag so the sentinel dict-keys / set-members by source — the
        # union/connection per-parent grouping and the planner's dedup tuple both rely on it.
        return hash(self.source)

    def __repr__(self) -> str:
        # the dedup key folds steps through their repr (peer_key is an f-string); render the
        # SOURCE, never the value, so two same-source placeholders' keys are byte-identical
        # and the value never leaks into a plan-cache key.
        return f"Placeholder({self.source!r})"


def rebind_pagination_value(value: Any, values_by_source: Mapping[str, Any]) -> Any:
    """Re-point a pagination value's :class:`Placeholder` to a cached request's variable value.

    The pagination counterpart of :func:`rebind_placeholder_value` (which re-points a WHERE
    bindparam): a variable-derived ``first`` / ``offset`` / ``after`` / ``before`` is a
    :class:`Placeholder` carrying the FIRST request's value; on a cache HIT for a different
    request, re-point it to THIS request's value by its stable source tag. Returns a FRESH
    ``Placeholder`` (same source, new value) when the source is supplied, the original when it
    is not, and a plain literal (a non-``Placeholder``) unchanged — so a step assigns the
    return value back and a literal page is a no-op. A fresh object (rather than mutating in
    place) keeps the sentinel's ``__hash__``/``__eq__`` — which key off the source — stable.
    """
    if not isinstance(value, Placeholder):
        return value
    if value.source not in values_by_source:
        return value
    return Placeholder(value.source, values_by_source[value.source])


def unwrap_placeholder(value: Any) -> Any:
    """The runtime value of a pagination value, unwrapping a :class:`Placeholder`.

    A step calls this at every point that needs the ACTUAL value (the SQL build, the page
    arithmetic, the execute-time params, the cursor decode): a variable-derived value arrives
    as a :class:`Placeholder` carrying the request's value, a plan-time literal arrives as the
    bare scalar / cursor string. Either way this returns the value the step computes with, so
    only the dedup key (which keeps the sentinel) differs between the two.
    """
    return value.value if isinstance(value, Placeholder) else value


def placeholder_source_tag(value: Any) -> Optional[str]:
    """The stable source tag of a pagination value, or ``None`` for a plain literal.

    The pagination-surface counterpart of :func:`placeholder_source` (which reads a WHERE
    bindparam): a :class:`Placeholder` reports its variable source, a bare ``int`` / cursor
    ``str`` reports ``None`` (it stays on the value-included literal key path).
    """
    return value.source if isinstance(value, Placeholder) else None

# A side attribute stamped on a placeholder bindparam carrying its STABLE source tag
# (e.g. "var:status"). `add_where` reads it off each bind in a predicate to populate the
# step's `placeholder_binds` registry; an ordinary (literal) bindparam lacks it, so it
# stays on the value-included key path. Kept as a plain attribute (not a SQLAlchemy
# annotation) so it survives the predicate as the host hands it over, before any compile.
PLACEHOLDER_SOURCE_ATTR = "_grafast_placeholder_source"

# Per-process counter giving every placeholder bindparam a UNIQUE bind NAME, so two
# placeholders in one statement (or two predicates sharing a source) never collide on the
# compiled `%(name)s` param. The name is execution-only plumbing; the dedup key uses the
# SOURCE tag, not the name, so a fresh name per call does NOT change the key (two requests
# of the same document still key identically off the stable source tag).
_placeholder_counter = itertools.count()


def pg_placeholder(source: str, value: Any, *, type_: Optional[Any] = None) -> BindParameter:
    """Build a source-tagged, value-carrying placeholder bindparam for a WHERE predicate.

    ``source`` is the STABLE source tag the dedup key discriminates by — for a
    variable-derived arg, ``field_args.source("status")`` (``"var:<variable_name>"``),
    request-stable so two requests of the same document key identically (a cache hit) while
    two different variable sources never merge. ``value`` is THIS request's runtime value,
    bound on the param so it rides ``compiled.params`` through execution unchanged (nothing
    new is needed at execute time — the bind already carries the value). ``type_`` is the
    optional SQLAlchemy type for the column (the host owns the column type; the engine never
    guesses it), forwarded to ``bindparam`` so the value is adapted/cast correctly.

    The returned bind has a UNIQUE name (so two placeholders never collide in one compiled
    statement) and the source tag stamped on :data:`PLACEHOLDER_SOURCE_ATTR`, which
    :meth:`grafast_py.pg.customize.PgCustomizable.add_where` reads to register it as a
    placeholder (value-agnostic in the dedup key). A host only ever gets a placeholder by
    calling this — there is no auto-placeholdering — because only the host knows the column
    type and owns the predicate construction.
    """
    name = f"grafast_ph_{next(_placeholder_counter)}"
    bind = bindparam(name, value=value, type_=type_)
    setattr(bind, PLACEHOLDER_SOURCE_ATTR, source)
    return bind


def placeholder_source(bind: BindParameter) -> Optional[str]:
    """Return the stable source tag of a placeholder bindparam, or ``None`` for a literal.

    The discriminator :meth:`PgCustomizable.add_where` and :func:`predicate_key` use to tell
    a placeholder (value-agnostic, source-tagged) from an ordinary literal bind (value-
    included): a placeholder carries :data:`PLACEHOLDER_SOURCE_ATTR`; a plain ``bindparam``
    (the literal SQLAlchemy emits for ``column("x") == value``) does not.
    """
    return getattr(bind, PLACEHOLDER_SOURCE_ATTR, None)


def rebind_placeholder_value(bind: BindParameter, value: Any) -> None:
    """Re-point a placeholder bindparam's bound value to ``value`` (the plan-cache rebind).

    A cached predicate's ``pg_placeholder`` bind carries the FIRST request's value; on a cache
    HIT for a different request the bind is re-pointed to THIS request's variable value so the
    shared value-agnostic SQL executes with the right value. SQLAlchemy keeps the bound value
    on ``BindParameter.value`` (and the ``callable`` form when deferred); set both so the value
    rides ``compiled.params`` unchanged at execute time.
    """
    bind.value = value
    bind.callable = None


__all__ = [
    "PLACEHOLDER_SOURCE_ATTR",
    "pg_placeholder",
    "placeholder_source",
    "rebind_placeholder_value",
    "rebind_pagination_value",
    "Placeholder",
    "unwrap_placeholder",
    "placeholder_source_tag",
]
