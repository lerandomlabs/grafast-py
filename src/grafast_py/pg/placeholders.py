"""Value-agnostic WHERE placeholders: a source-tagged, value-LESS bindparam.

A host filters by a GraphQL arg value by INLINING it (``.where(column("status") ==
args["status"])``): plan-time-known and value-DISCRIMINATED in the dedup key (two values
-> two keys -> never merge). That is correct but VALUE-specific: the compiled key carries
the literal, so the plan cannot be reused across requests with a different value.

When the value came from a GraphQL ``$variable`` (``field_args.is_variable("status")``),
the host instead builds a PLACEHOLDER: ``.where(column("status") == pg_placeholder(
field_args.source("status"), args["status"]))``. A placeholder is a SQLAlchemy
``bindparam`` that

  * carries NO baked value — the bind is built value-LESS so the cached statement stays
    immutable and value-independent; THIS request's runtime value is injected into the
    compiled statement's ``params`` at RENDER time (keyed by the bind name), by its source
    tag, so two concurrent cache HITs of the same statement run with their OWN values and
    can never bleed. (The ``value`` arg to :func:`pg_placeholder` is kept only so a step
    can map the bind name to its source -> value when rendering the FIRST/uncached request;
    it is NOT bound onto the bindparam.) and
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
from typing import Any, Callable, Mapping, Optional

from sqlalchemy import bindparam
from sqlalchemy.sql.elements import BindParameter


class Placeholder:
    """A source-tagged, value-LESS sentinel for a NON-predicate pagination value (``first`` /
    ``offset`` / ``after`` / ``before``) that came from a GraphQL ``$variable``.

    Unlike a WHERE value (which the host wraps as a Core :func:`pg_placeholder` bindparam),
    a pagination value is a plain Python scalar (an ``int`` ``first`` / ``offset``) or a
    cursor ``str`` (``after`` / ``before``) stored directly on the step — it is NOT a
    ``ColumnElement`` that can carry a side attribute, and the step already binds it as a
    PARAMETER at execute time (``window_slice`` binds ``first`` / ``offset``; the keyset
    comparator binds the decoded cursor values). So the SQL is ALREADY value-agnostic for
    these; the only thing that carries the literal is the DEDUP KEY (``first`` / ``offset``
    in ``peer_key`` / ``dedup_params``, the decoded ``after_values`` / ``before_values``).

    The sentinel itself lives on the SHARED cached step, so it carries NO request value: the
    runtime value is resolved per request from ``BucketExtra.source_values`` by the stable
    ``source`` tag at SQL-render / page-math / cursor-decode time (see
    :func:`resolve_placeholder`). Its ``__eq__`` / ``__hash__`` / ``__repr__`` key ONLY off
    that ``source`` tag (``"var:limit"`` — the variable name), so two requests of the same
    document produce the SAME dedup key (a cache hit / merge), two DIFFERENT variable sources
    never merge, and a placeholder never equals a literal ``int`` / cursor of a coincidentally
    equal value (a ``Placeholder`` is never ``==`` a bare ``int``). A LITERAL ``first=5`` stays
    the bare ``int`` 5 (value-included) so literal pages never merge with placeholder pages —
    exactly the WHERE-predicate split, applied to the pagination surface.

    A step RESOLVES via :func:`resolve_placeholder` (against ``source_values``) everywhere it
    needs the runtime value, and keeps the sentinel itself in ``peer_key`` / ``dedup_params``
    so the key stays source-keyed.
    """

    __slots__ = ("source",)

    def __init__(self, source: str) -> None:
        # also accepts a plan-time context token (`info.context["page_size"]`) — a token IS
        # a source tag, so `Placeholder(info.context["page_size"])` paginates by a
        # per-request context value the same way `Placeholder(args.source("first"))` does
        # by a variable.
        self.source = getattr(source, "source", source)

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


def resolve_placeholder(value: Any, source_values: Mapping[str, Any]) -> Any:
    """The runtime value of a pagination value, resolving a value-LESS :class:`Placeholder`.

    A step calls this at every point that needs the ACTUAL value (the SQL shape decision,
    the page arithmetic, the execute-time params, the cursor decode): a variable-derived
    value arrives as a value-less :class:`Placeholder`, resolved against this request's
    ``source_values`` (the source-tag -> value map threaded via ``BucketExtra``); a plan-time
    literal arrives as the bare scalar / cursor string and passes through unchanged. Either
    way this returns the value the step computes with, so only the dedup key (which keeps the
    sentinel) differs between the two. A ``var:`` source absent from the map resolves to
    ``None`` (an omitted no-default variable), matching ``values_by_source``'s
    omitted-variable handling.

    A ``ctx:`` source (``Placeholder(info.context["page_size"])``) resolves FRESH from the
    pg request context — the same rules as the WHERE render seam (``where_params``): a
    Mapping context by key, anything else by attribute, and a MISSING key fails LOUD (a
    page size silently resolving to None would drop the window bound and widen the page).
    """
    if isinstance(value, Placeholder):
        if value.source.startswith("ctx:"):
            from .executor import current_pg_request

            key = value.source[4:]
            context = current_pg_request().context
            if isinstance(context, Mapping):
                return context[key]
            return getattr(context, key)
        return source_values.get(value.source)
    return value


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

# A side attribute carrying a placeholder's optional RUNTIME TRANSFORM — a pure callable
# applied to the resolved source value AT RENDER time (``transform(context[key])``), so a
# context-DERIVED scoping value (``status.upper()``, ``tenant + 1000``) rides a value-AGNOSTIC
# bind whose value is computed PER REQUEST rather than baked as a plan-time literal. The bind
# stays value-LESS (its STRUCTURE is fixed at plan time, like a bare ``ctx:`` placeholder); the
# transform is the grafast-py analogue of upstream ``lambda($context, fn)`` feeding a predicate.
# Absent on a bare placeholder (the resolved value passes through unchanged).
PLACEHOLDER_TRANSFORM_ATTR = "_grafast_placeholder_transform"

# Per-process counter giving every placeholder bindparam a UNIQUE bind NAME, so two
# placeholders in one statement (or two predicates sharing a source) never collide on the
# compiled `%(name)s` param. The name is execution-only plumbing; the dedup key uses the
# SOURCE tag, not the name, so a fresh name per call does NOT change the key (two requests
# of the same document still key identically off the stable source tag).
_placeholder_counter = itertools.count()


def pg_placeholder(
    source: str,
    value: Any = None,
    *,
    type_: Optional[Any] = None,
    transform: Optional[Callable[[Any], Any]] = None,
) -> BindParameter:
    """Build a source-tagged, value-LESS placeholder bindparam for a WHERE predicate.

    ``source`` is the STABLE source tag the dedup key discriminates by — for a
    variable-derived arg, ``field_args.source("status")`` (``"var:<variable_name>"``),
    request-stable so two requests of the same document key identically (a cache hit) while
    two different variable sources never merge. ``value`` is accepted for backward call
    compatibility (the host still passes ``args["status"]``) but is NOT bound onto the
    bindparam: the bind is built value-LESS so the cached statement stays immutable and the
    runtime value is injected into the compiled statement's ``params`` per request at RENDER
    time (keyed by this bind's name, resolved by source from ``BucketExtra.source_values``).
    Binding the value here is exactly what the deepcopy-free cache must avoid — a baked value
    on a SHARED bind would bleed across concurrent requests. ``type_`` is the optional
    SQLAlchemy type for the column (the host owns the column type; the engine never guesses
    it), forwarded to ``bindparam`` so the injected value is adapted/cast correctly.

    ``transform`` is an optional PURE callable applied to the resolved source value AT RENDER
    time (``transform(context[key])``): a context-DERIVED scoping value (``status.upper()``,
    ``tenant + 1000``) thus rides a value-AGNOSTIC bind whose value is computed PER REQUEST,
    never baked as a plan-time literal. The bind stays value-LESS — its STRUCTURE is fixed at
    plan time exactly like a bare placeholder — and the transform is the grafast-py analogue of
    upstream ``lambda($context, fn)`` feeding a predicate. It runs in the render seam
    (``where_params`` / ``member_where_params``), so it must stay pure (no I/O, no side effects).

    The bind is built with ``required=False`` so :func:`grafast_py.pg.customize.check_predicate`
    accepts it (a value-less ``required=True`` bind would be rejected as "unbound"); the value
    rides ``params`` at execute, never the bind. The returned bind has a UNIQUE name (so two
    placeholders never collide in one compiled statement) and the source tag stamped on
    :data:`PLACEHOLDER_SOURCE_ATTR`, which
    :meth:`grafast_py.pg.customize.PgCustomizable.add_where` reads to register it as a
    placeholder (value-agnostic in the dedup key). A host only ever gets a placeholder by
    calling this — there is no auto-placeholdering — because only the host knows the column
    type and owns the predicate construction.

    ``source`` also accepts a plan-time context token (``info.context["tenant_id"]``, a
    :class:`~grafast_py.constraints.ContextToken`) — a token IS a source tag
    (``"ctx:tenant_id"``), so ``pg_placeholder(info.context["tenant_id"], type_=...)`` is
    the typed / transformed form of the bare ``column == token`` coercion.
    """
    source_tag = getattr(source, "source", source)
    name = f"grafast_ph_{next(_placeholder_counter)}"
    bind = bindparam(name, type_=type_, required=False)
    setattr(bind, PLACEHOLDER_SOURCE_ATTR, source_tag)
    if transform is not None:
        setattr(bind, PLACEHOLDER_TRANSFORM_ATTR, transform)
    return bind


def placeholder_source(bind: BindParameter) -> Optional[str]:
    """Return the stable source tag of a placeholder bindparam, or ``None`` for a literal.

    The discriminator :meth:`PgCustomizable.add_where` and :func:`predicate_key` use to tell
    a placeholder (value-agnostic, source-tagged) from an ordinary literal bind (value-
    included): a placeholder carries :data:`PLACEHOLDER_SOURCE_ATTR`; a plain ``bindparam``
    (the literal SQLAlchemy emits for ``column("x") == value``) does not.
    """
    return getattr(bind, PLACEHOLDER_SOURCE_ATTR, None)


def placeholder_transform(bind: BindParameter) -> Optional[Callable[[Any], Any]]:
    """Return the render-time transform of a placeholder bindparam, or ``None``.

    A placeholder built with ``transform=`` carries a pure callable on
    :data:`PLACEHOLDER_TRANSFORM_ATTR`; the render seam applies it to the resolved source
    value (``transform(context[key])``) so a context-DERIVED value is computed per request. A
    bare placeholder (or an ordinary literal bind) carries none, so the resolved value passes
    through unchanged.
    """
    return getattr(bind, PLACEHOLDER_TRANSFORM_ATTR, None)


def transform_key(fn: Callable[[Any], Any]) -> str:
    """A cache-STABLE, content-DISTINGUISHING identity for a placeholder transform callable.

    Two requirements pull against ``id(fn)``:

    * STABLE across re-invocation — a 2-arg ``select_customizer`` is re-invoked on every cache-hit
      to revalidate its constraints, minting a FRESH function object each time. An id-based key
      would never match the stored key and would silently disable plan caching for any customizer
      using a ``transform=``. A code object is created once at def-time and SHARED across
      invocations, so content drawn from it is stable.
    * DISTINGUISHING by BEHAVIOUR, not source location — two distinct transforms on the SAME line
      (``lambda v: v + 1`` vs ``lambda v: v + 2``) must not collapse, else same-source placeholders
      would dedup/cache as if identical and apply the wrong transform to one of them.

    So for a Python callable we key on its EXECUTABLE CONTENT: bytecode + constants + defaults
    (positional AND keyword-only) + the closure cell values — together the COMPLETE set of what
    parameterizes a function's behaviour given its call. Equivalent transforms merge, different
    ones differ (incl. closures over different values; a transform closing over per-request state
    then forces a safe re-plan rather than reusing the first request's captured value). A callable
    with no ``__code__`` keys by its own qualified name (a builtin/method descriptor like
    ``str.upper``) or, for a callable INSTANCE, by its type plus ``__dict__`` state (so ``AddN(1)``
    and ``AddN(2)`` differ).
    """
    code = getattr(fn, "__code__", None)
    if code is not None:
        closure = tuple(repr(cell.cell_contents) for cell in (fn.__closure__ or ()))
        return (
            f"code:{code.co_code!r}:{code.co_consts!r}:"
            f"{fn.__defaults__!r}:{fn.__kwdefaults__!r}:{closure!r}"
        )
    qualname = getattr(fn, "__qualname__", None)
    if qualname is not None:
        return f"named:{getattr(fn, '__module__', '')}.{qualname}"
    state = getattr(fn, "__dict__", {})
    return (
        f"obj:{type(fn).__module__}.{type(fn).__qualname__}:"
        f"{tuple(sorted((k, repr(v)) for k, v in state.items()))!r}"
    )


__all__ = [
    "PLACEHOLDER_SOURCE_ATTR",
    "PLACEHOLDER_TRANSFORM_ATTR",
    "pg_placeholder",
    "placeholder_source",
    "placeholder_transform",
    "transform_key",
    "Placeholder",
    "resolve_placeholder",
    "placeholder_source_tag",
]
