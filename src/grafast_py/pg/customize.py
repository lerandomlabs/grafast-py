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

A PLACEHOLDER is the value-AGNOSTIC counterpart: when the value came from a GraphQL
``$variable``, the host wraps it as ``pg_placeholder(field_args.source("status"),
args["status"])`` — a bindparam that carries the request's value (so it still rides
``compiled.params`` at execute, unchanged) but is tagged with a STABLE source (``var:status``,
the variable name). A predicate containing such a bind takes a DIFFERENT key path: compiled
WITHOUT ``literal_binds`` (value-agnostic ``status = %(name)s``) plus the sorted source tags,
so two requests of the same document key IDENTICALLY (one shared plan), two different variable
sources never merge, and a placeholder never merges with a coincidentally equal-valued inlined
literal (``$1`` vs an inlined ``'published'`` are different SQL). A predicate with NO
placeholder binds is UNAFFECTED — it keeps the value-included ``literal_binds`` key, so a
purely literal predicate keys exactly as it would without placeholder support. The
discriminator is membership in the per-step
``placeholder_binds`` registry, populated by :meth:`PgCustomizable.add_where` when it sees a
source-tagged bind (see :mod:`grafast_py.pg.placeholders`).
"""

import collections.abc
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from sqlalchemy import literal_column
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import CompileError
from sqlalchemy.sql import ColumnElement, visitors
from sqlalchemy.sql.elements import BindParameter

from ..config import log
from ..step_model import Step
from .conditions import Condition, compile_condition
from .executor import current_pg_request
from .ordering import OrderTerm
from .placeholders import pg_placeholder, placeholder_source

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


def predicate_key(
    predicate: ColumnElement,
    placeholder_binds: Optional[Mapping[str, str]] = None,
) -> str:
    """A stable, content-based dedup key for a Core predicate.

    ``placeholder_binds`` maps a placeholder bindparam's NAME to its stable source tag
    (e.g. ``{"grafast_ph_3": "var:status"}``), as populated by
    :meth:`PgCustomizable.add_where`. It splits the predicate into two key regimes:

    LITERAL (no placeholder binds — the default, and every purely literal predicate): compiled
    with the Postgres dialect and ``literal_binds`` so every bound value renders INLINE —
    ``status = 'published'`` and ``status = 'draft'`` produce DIFFERENT strings, so two
    differently-filtered selects never dedup-merge, while identical predicates yield the
    identical string (and DO merge). Valid only because every bind carries a plan-time value
    — :func:`check_predicate` guarantees that. This path is BYTE-IDENTICAL whether or not
    placeholder support is in play, so every literal-predicate merge/count is preserved.

    PLACEHOLDER (one or more binds are in ``placeholder_binds``): the value is NOT known at
    plan time (it is a per-request variable), so the key MUST NOT inline it. Compiled WITHOUT
    ``literal_binds`` (value-agnostic ``status = %(name)s``) and SUFFIXED with the sorted
    source tags of the predicate's placeholder binds. This discriminates by placeholder
    IDENTITY, never by runtime value: two predicates over the SAME source merge; over
    DIFFERENT sources do not; and a placeholder predicate never equals a literal one of a
    coincidentally equal value (a placeholder renders a ``<<ph:source>>`` sentinel + a ``|ph=``
    suffix; an ordinary literal renders inline). Only the PLACEHOLDER binds are replaced by a
    source sentinel before the compile — every co-located ORDINARY literal still renders inline
    by value (so two predicates differing only by such a literal do NOT merge), and the unique
    ``grafast_ph_N`` name is erased so two same-source predicates converge.

    The result is a hashable ``str`` that slots into the step's ``dedup_params`` tuple. This
    compile is for the KEY only; the step executes the predicate with its bindparams intact
    (carrying their values), so execution stays parameterised.

    An exotic literal that no dialect can render inline (e.g. a non-UTF8 ``bytes`` value
    against a ``bytea`` column) raises ``CompileError``; rather than crash PLANNING we fall
    back to :func:`structural_predicate_key` — the value-free SQL plus the bound values'
    repr — which still distinguishes two different exotic predicates.
    """
    if placeholder_binds:
        return placeholder_predicate_key(predicate, placeholder_binds)
    try:
        return str(
            predicate.compile(
                dialect=postgresql.dialect(),
                compile_kwargs={"literal_binds": True},
            )
        )
    except CompileError:
        return structural_predicate_key(predicate)


def placeholder_predicate_key(
    predicate: ColumnElement, placeholder_binds: Mapping[str, str]
) -> str:
    """The dedup key for a predicate that mixes placeholder and ordinary literal binds.

    Each PLACEHOLDER bind is replaced IN THE AST by a stable source-derived sentinel
    (``literal_column("<<ph:var:status>>")``) BEFORE compiling, then the result is compiled
    with ``literal_binds`` so every REMAINING bind (an ordinary co-located literal that rides
    the same ``and_(...)``) renders INLINE by its value. The AST replacement, not a post-compile
    string rewrite, serves three ends:

    - it ERASES the placeholder's per-call bind NAME, so two same-source predicates (differing
      only by the fresh ``grafast_ph_N``) produce the IDENTICAL SQL and merge (a cache hit
      across requests of the same document) — for BOTH the scalar ``%(name)s`` form AND the
      expanding ``IN (__[POSTCOMPILE_name])`` form a string rewrite of ``%(name)s`` would miss;
    - it pins each source POSITIONALLY in the SQL (``a = <<ph:var:x>> AND b = <<ph:var:y>>``),
      so a source-to-column swap stays a DISTINCT statement, never a spurious merge; and
    - it removes the placeholder's bound VALUE from the literal-binds compile, so the placeholder
      stays value-agnostic while a NON-placeholder literal beside it keeps its value (``title =
      'alpha'`` vs ``title = 'beta'`` differ, so two requests filtering by different co-located
      literals never wrongly merge — the cross-value-corruption guard).

    A trailing sorted source-tag suffix is appended so the key is human-legible and a
    placeholder never collides with a literal-only predicate of a coincidentally equal value.
    The result is ``(<source-positioned, literal-inlined SQL>)|ph=<sorted source tags>``.

    An exotic co-located literal no dialect can render inline raises ``CompileError`` (as in the
    literal path); we fall back to :func:`structural_placeholder_predicate_key`, which keeps the
    same placeholder sentinels but reprs the remaining bound values instead of inlining them.
    """
    sentinelled = sentinel_placeholders(predicate)
    sources = sorted(placeholder_binds.values())
    try:
        agnostic_sql = str(
            sentinelled.compile(
                dialect=postgresql.dialect(),
                compile_kwargs={"literal_binds": True},
            )
        )
    except CompileError:
        return structural_placeholder_predicate_key(sentinelled, sources)
    return f"{agnostic_sql}|ph={sources!r}"


def sentinel_placeholders(predicate: ColumnElement) -> ColumnElement:
    """Return ``predicate`` with every placeholder bind replaced by a source sentinel.

    Walks the predicate AST and swaps each source-tagged placeholder
    :class:`~sqlalchemy.sql.elements.BindParameter` for a
    ``literal_column("<<ph:<source>>>")`` — a value-LESS, source-positioned token. Ordinary
    literal binds are left intact so a subsequent ``literal_binds`` compile inlines THEIR
    values. Used to build the value-agnostic placeholder dedup key (see
    :func:`placeholder_predicate_key`).
    """

    def replace(element: Any) -> Optional[ColumnElement]:
        if isinstance(element, BindParameter):
            source = placeholder_source(element)
            if source is not None:
                return literal_column(f"<<ph:{source}>>")
        return None

    return visitors.replacement_traverse(predicate, {}, replace)


def structural_placeholder_predicate_key(
    sentinelled: ColumnElement, sources: Sequence[str]
) -> str:
    """A ``literal_binds``-free fallback for a placeholder predicate with an exotic literal.

    The placeholder binds are already sentinelled out of ``sentinelled``, so the only binds
    left are ordinary co-located literals. Compiles WITHOUT ``literal_binds`` (so an
    unrenderable literal cannot raise) and appends a stable repr of those remaining bound
    values, keeping a co-located literal value-discriminated while the placeholders stay
    source-keyed. Mirrors :func:`structural_predicate_key` for the placeholder path.
    """
    compiled = sentinelled.compile(dialect=postgresql.dialect())
    params = tuple(sorted((name, repr(value)) for name, value in compiled.params.items()))
    return f"{compiled}|{params!r}|ph={sources!r}"


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


def placeholder_binds_in(predicate: ColumnElement) -> Dict[str, str]:
    """The placeholder binds (name -> source tag) a predicate carries, walked from its AST.

    The registry-free way to discover a predicate's placeholders: walk its bindparams and
    collect those carrying a source tag (a :func:`pg_placeholder`). Used where no per-step
    ``placeholder_binds`` registry is available — the inline fold signature, which keys a
    reproduced child predicate the same way the standalone child does. Returns ``{}`` for a
    literal-only predicate, so a caller passes ``None`` (the literal key path) for it.
    """
    binds: Dict[str, str] = {}
    for bind in visitors.iterate(predicate):
        if not isinstance(bind, BindParameter):
            continue
        source = placeholder_source(bind)
        if source is not None:
            binds[bind.key] = source
    return binds


class ContextSources:
    """The plan-time placeholder factory handed to a 2-arg ``select_customizer``.

    The cache-safe counterpart of the 1-arg ``customizer(context)`` form (whose values are
    plan-time LITERALS). ``placeholder(key)`` mints a value-LESS ``pg_placeholder`` tagged
    ``ctx:<key>``; its value is read from ``current_pg_request().context[key]`` PER REQUEST at
    execute time (in :meth:`PgCustomizable.where_params`), so the predicate's STRUCTURE is fixed
    at plan time while its VALUE is supplied per request — the grafast-py analogue of upstream
    ``selectAuth`` embedding a runtime step. A customizer built this way is value-INDEPENDENT, so
    its plan is cacheable and a cache HIT re-binds each request's OWN context value rather than
    serving the first request's baked literal.
    """

    def placeholder(self, key: str, *, type_: Optional[Any] = None) -> BindParameter:
        """A value-LESS placeholder whose value is ``context[key]``, read per request at execute.

        ``key`` names the per-request context entry to bind, read at execute from
        ``current_pg_request().context`` (``context[key]`` for a Mapping, else
        ``getattr(context, key)``); ``type_`` is the column's SQLAlchemy type, so the injected
        value is cast correctly. No value is taken here — the plan stays value-independent and the
        value rides the compiled statement's params per request, resolved by the ``ctx:<key>``
        source tag. A derived/transformed scoping value (not a bare context entry) is not
        expressible this way; expose it under its own context key, or use the 1-arg form (which
        inlines a literal and is non-cacheable).
        """
        return pg_placeholder(f"ctx:{key}", type_=type_)


def predicate_bakes_literal(predicate: ColumnElement) -> bool:
    """Whether a customizer predicate carries a non-placeholder (plan-time literal) bind.

    Such a bound value may be PER-REQUEST (resolved from the context at plan time), so a plan
    carrying it cannot be shared across requests — it forces the plan non-cacheable (the safety
    floor). A predicate whose binds are ALL placeholders (``ctx:`` / ``var:`` source-tagged), or
    which has no binds at all (a static ``deleted_at IS NULL``), is value-independent and stays
    cacheable. ``check_predicate`` already rejects an unbound bind, so every bind here is either a
    source-tagged placeholder or a plan-time literal.
    """
    for bind in visitors.iterate(predicate):
        if isinstance(bind, BindParameter) and placeholder_source(bind) is None:
            return True
    return False


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

    # True when the resource customizer baked a plan-time LITERAL into the WHERE (so the plan
    # carries a possibly-per-request value and must not be cached — the cache safety floor reads
    # this duck-typed). Set on EVERY instance by init_customization, so its absence on a
    # PgCustomizable would signal a missed init rather than silently defaulting to cacheable.
    customizer_bakes_literal: bool = False
    # how many of the leading where_predicates came from the resource customizer (the rest are
    # per-plan .where()s); has_literal_customization inspects only these.
    _customizer_predicate_count: int = 0

    def init_customization(
        self,
        predicates: Sequence[ColumnElement] = (),
        *,
        bakes_literal: bool = False,
        customizer_predicate_count: int = 0,
    ) -> None:
        """Seed the customization list (call from the subclass ``__init__``).

        ``bakes_literal`` records that the resource customizer contributed a plan-time literal
        (so the cache safety floor forces the plan non-cacheable); ``customizer_predicate_count``
        is how many leading predicates are the customizer's (vs per-plan ``.where()``s).
        """
        # AND-combined onto the batched WHERE in insertion order (and_(...) operand order
        # is preserved in the compiled string, so a stable order keeps the dedup key
        # stable). Resource-customizer predicates come first, then per-plan .where()s.
        self.where_predicates: List[ColumnElement] = list(predicates)
        self.customizer_bakes_literal = bakes_literal
        self._customizer_predicate_count = customizer_predicate_count
        # per-step placeholder registry: placeholder bind NAME -> stable source tag (e.g.
        # "var:status"), populated as add_where sees a source-tagged bind. Read by
        # predicate_key to take the value-agnostic key path for a placeholder-bearing
        # predicate; EMPTY for a literal-only step (the default), so its key stays the
        # byte-identical value-included literal key. Seed it from any predicates passed in
        # (a resource customizer could itself build a placeholder, though that is unusual).
        self.placeholder_binds: Dict[str, str] = {}
        for predicate in self.where_predicates:
            self._register_placeholder_binds(predicate)
        # the customization_signature tuple is content-derived and read by BOTH peer_key
        # and dedup_params; cache it and invalidate on every where_predicates mutation.
        self._signature_cache: Union[tuple, None] = None

    def seed_resource_customization(self, resource: Any) -> None:
        """Seed the WHERE list from a resource's ``select_customizer`` (call from ``__init__``).

        Resolves the customizer ONCE against the per-request context. The context is only
        read when the resource actually has a customizer, so a step built outside a request
        (e.g. a no-DB dedup test over an un-customized resource) needs no bound context.

        A 1-arg customizer (``customizer(context)``) inlines values as plan-time LITERALS; a
        2-arg customizer (``customizer(context, sources)``) uses ``sources.placeholder(key)`` to
        emit value-LESS ``ctx:`` placeholders whose value is read per request at execute (in
        :meth:`where_params`). A customizer that baked a literal marks the plan non-cacheable
        (the safety floor); a pure-placeholder customizer stays cacheable.
        """
        customizer = resource.select_customizer
        if customizer is None:
            self.init_customization()
            return
        context = current_pg_request().context
        predicates, bakes_literal = resolve_customizer_predicates(
            customizer, context, resource.select_customizer_arity
        )
        self.init_customization(
            predicates,
            bakes_literal=bakes_literal,
            customizer_predicate_count=len(predicates),
        )

    def add_where(self, predicate: ColumnElement) -> None:
        """AND a validated UNIFORM Core predicate onto the batched WHERE.

        Also registers any PLACEHOLDER binds the predicate carries (a source-tagged
        ``pg_placeholder``) into ``placeholder_binds`` so :func:`predicate_key` takes the
        value-agnostic key path for this predicate. A predicate with only ordinary literal
        binds adds nothing to the registry, so its key stays the byte-identical literal key.
        """
        self.where_predicates.append(check_predicate(predicate))
        self._register_placeholder_binds(predicate)
        self._signature_cache = None

    def _register_placeholder_binds(self, predicate: ColumnElement) -> None:
        """Record each source-tagged placeholder bind in ``predicate`` (name -> source tag).

        Registers those of the predicate's binds carrying a placeholder source tag (a
        ``pg_placeholder``); ordinary literal binds carry none and are skipped, so the
        registry holds ONLY placeholder binds. A bind name is unique per ``pg_placeholder``
        call, so no two placeholders collide in the registry.
        """
        self.placeholder_binds.update(placeholder_binds_in(predicate))

    def where_params(self, source_values: Mapping[str, Any]) -> Dict[str, Any]:
        """The execute-time params for this step's WHERE placeholder binds (name -> value).

        The deepcopy-free render seam: a ``pg_placeholder`` bind is value-LESS, so its runtime
        value is supplied per request in the compiled statement's ``params`` rather than baked on
        the SHARED bind. Each placeholder bind resolves by its stable SOURCE tag:

        * a ``var:`` source (a GraphQL ``$variable``) resolves from ``source_values`` — the
          source-tag -> value map threaded via ``BucketExtra`` (a source absent from the map
          resolves to ``None``, an omitted no-default variable);
        * a ``ctx:`` source (a resource ``select_customizer`` value) resolves from THIS request's
          context (``context[key]`` for a Mapping, else ``getattr(context, key)`` for an
          object/dataclass context), read FRESH per request — so a cache HIT over the SHARED plan
          binds THIS request's context value, never the one the plan was built with. A missing
          context key fails LOUD (``KeyError`` / ``AttributeError`` propagate), not a silent
          ``None`` that would render ``col = NULL`` and silently widen the scope.

        A literal-only step has an empty ``placeholder_binds`` registry, so this returns ``{}`` —
        a byte-identical no-op for the default cache-off path; ``current_pg_request()`` is touched
        only when a ``ctx:`` bind is actually present (so var:-only steps and no-request unit tests
        are unchanged). The subclass seeds these into the ``params`` it hands ``executor.run``
        alongside ``keys`` / pagination params, so the value rides ``compiled.params`` per request
        and never mutates the cached statement.
        """
        if not self.placeholder_binds:
            return {}
        context = None
        if any(source.startswith("ctx:") for source in self.placeholder_binds.values()):
            context = current_pg_request().context
        params: Dict[str, Any] = {}
        for name, source in self.placeholder_binds.items():
            if source.startswith("ctx:"):
                key = source[len("ctx:") :]
                if isinstance(context, collections.abc.Mapping):
                    params[name] = context[key]
                else:
                    params[name] = getattr(context, key)
            else:
                params[name] = source_values.get(source)
        return params

    def copy_customization_from(self, other: "PgCustomizable") -> None:
        """Copy ``other``'s already-resolved customization onto this step verbatim.

        An inlining clone reproduces the parent's WHERE predicate-for-predicate rather than
        re-resolving the resource customizer; it must copy the PLACEHOLDER registry alongside
        ``where_predicates`` so a placeholder-bearing predicate keeps its value-agnostic key
        on the clone (a registry left empty would silently revert that predicate to the
        value-included literal path, desyncing the clone's key from the original's). The cache
        safety-floor decision (``customizer_bakes_literal`` + the customizer predicate count)
        travels too, so a clone of a literal-baking step stays non-cacheable rather than silently
        re-enabling caching. Resets the signature cache so the clone recomputes against the copied
        state.
        """
        self.where_predicates = list(other.where_predicates)
        self.placeholder_binds = dict(other.placeholder_binds)
        self.customizer_bakes_literal = other.customizer_bakes_literal
        self._customizer_predicate_count = other._customizer_predicate_count
        self._signature_cache = None

    def has_literal_customization(self) -> bool:
        """Whether the RESOURCE customizer contributed a plan-time literal (so it can't be cached).

        Inspects only the leading customizer-origin predicates (not per-plan ``.where()``s):
        True if any carries a non-placeholder bind. Equivalent to the ``customizer_bakes_literal``
        flag set at seed time; exposed for tests and as the readable discriminator.
        """
        return any(
            predicate_bakes_literal(p)
            for p in self.where_predicates[: self._customizer_predicate_count]
        )

    def customizer_structure_matches(self) -> bool:
        """Whether the resource customizer yields the SAME predicate STRUCTURE for the CURRENT
        request as this (cached) step holds — the cache-HIT structural-divergence guard.

        Re-resolves the resource ``select_customizer`` against THIS request's context and compares
        the VALUE-AGNOSTIC keys of its predicates (placeholder sources sentinelled, fresh bind
        names erased; a plain literal's value still included) to the cached step's customizer
        predicates. A value-only change — a placeholder bound to a different per-request value —
        keeps the SAME key, so a well-behaved value-varying customizer still hits. A STRUCTURAL
        change — different columns / predicate count, e.g. a customizer that returns no filter for
        an admin and a scoped filter for a user — yields a DIFFERENT key, so the caller treats the
        hit as a miss and re-plans rather than letting this request inherit another request's
        customizer-decided structure. A step with no resource customizer trivially matches.
        """
        customizer = self.resource.select_customizer
        if customizer is None:
            return True
        fresh, _ = resolve_customizer_predicates(
            customizer,
            current_pg_request().context,
            self.resource.select_customizer_arity,
        )
        fresh_keys = tuple(
            predicate_key(p, placeholder_binds_in(p) or None) for p in fresh
        )
        cached_keys = self.customization_signature()[: self._customizer_predicate_count]
        return fresh_keys == cached_keys

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

        Each predicate's key is computed against ITS OWN placeholder binds (not the whole
        step registry) so a placeholder in one predicate never leaks its source tag into
        another predicate's key. A literal-only step has an empty registry and so every
        ``predicate_key`` call takes the unchanged value-included literal path.
        """
        if self._signature_cache is None:
            self._signature_cache = tuple(
                predicate_key(p, self._placeholder_binds_for(p))
                for p in self.where_predicates
            )
        return self._signature_cache

    def _placeholder_binds_for(
        self, predicate: ColumnElement
    ) -> Optional[Dict[str, str]]:
        """The subset of ``placeholder_binds`` whose binds actually appear in ``predicate``.

        ``predicate_key`` keys a placeholder predicate off the SOURCE tags of ITS binds, so
        it must see only the binds in this predicate — a step-wide registry would append
        another predicate's source tag spuriously. Returns ``None`` when the predicate has no
        placeholder binds, so :func:`predicate_key` takes the unchanged literal path (this is
        the common case: an empty registry yields ``None`` for every predicate).
        """
        if not self.placeholder_binds:
            return None
        present: Dict[str, str] = {}
        for bind in visitors.iterate(predicate):
            if isinstance(bind, BindParameter) and bind.key in self.placeholder_binds:
                present[bind.key] = self.placeholder_binds[bind.key]
        return present or None


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
    customizer: Optional[Callable[..., Sequence[Any]]],
    context: Any,
    arity: int,
) -> Tuple[List[ColumnElement], bool]:
    """Resolve a resource ``select_customizer`` against the per-request context, ONCE.

    The customizer is the selectAuth analogue (soft-delete / tenant scoping / visibility),
    applied to READS only — never to mutations (see :mod:`grafast_py.pg.mutations`; write
    authorization is Postgres RLS via pgSettings). Two forms, by ``arity``:

    * 1-arg ``customizer(context) -> [Core predicate]`` — the legacy form; values are inlined as
      plan-time LITERALS, so the plan is value-specific (forced non-cacheable by the safety floor).
    * 2-arg ``customizer(context, sources) -> [Core predicate]`` — the cacheable form;
      ``sources.placeholder(key)`` mints a value-LESS ``ctx:`` placeholder whose value is read per
      request at execute, so the plan stays value-INDEPENDENT (structure at plan time, value per
      request — the convergence to upstream ``selectAuth``).

    A 2-arg customizer that varies only its VALUES across requests of a resource (the common
    tenant case) shares one cached plan and re-binds per request. One that varies its predicate
    STRUCTURE by context (e.g. no filter for an admin vs a scoped filter for a user) stays correct:
    the cache-hit structural-divergence guard (:meth:`PgCustomizable.customizer_structure_matches`)
    re-resolves it per request and re-plans on a shape change, so a request never inherits another
    request's structure — it just does not share a cached plan across the differing shapes. Every
    predicate is validated as a Core expression (never a raw string, fully bound, no reserved bind
    name) and AND-combined onto EVERY batched select for the resource. Returns ``(predicates,
    bakes_literal)`` where ``bakes_literal`` is True if ANY predicate carries a non-placeholder
    (plan-time literal) bind — the cache safety-floor signal for value baking.
    """
    if customizer is None:
        return [], False
    produced = (
        customizer(context) if arity == 1 else customizer(context, ContextSources())
    )
    predicates = [check_predicate(p) for p in produced]
    bakes_literal = any(predicate_bakes_literal(p) for p in predicates)
    log.debug(
        "pg resolve select customizer",
        predicates=len(predicates),
        cacheable=not bakes_literal,
        has_context=context is not None,
    )
    return predicates, bakes_literal


__all__ = [
    "RESERVED_BIND_NAMES",
    "check_predicate",
    "predicate_key",
    "placeholder_predicate_key",
    "sentinel_placeholders",
    "structural_placeholder_predicate_key",
    "structural_predicate_key",
    "predicate_bakes_literal",
    "ContextSources",
    "PgCustomizable",
    "PgSelectQueryBuilder",
    "resolve_customizer_predicates",
]
