"""LATERAL relation inlining: the fold spec and the nested-row extract step.

Inlining is an OPPORTUNISTIC, EQUIVALENCE-PRESERVING optimization: a parent pg select
absorbs a safe-to-fold child relation into its OWN statement via a ``LEFT JOIN LATERAL``
whose nested ``json_agg`` (hasMany) / ``to_jsonb`` (hasOne) rows the child bucket later
reads off the parent row — fewer SQL statements, byte-identical data. This module owns the
two data structures that survive the fold into execution:

- :class:`InlineSpec` — the immutable record of ONE fold (which child relation was inlined
  into the parent, under which nested column, faithfully reproducing the child's columns,
  ordering, customization and codecs). The parent step's ``build_query`` reads it to emit
  the LATERAL; the parent's ``peer_key`` / ``dedup_params`` fold it in (two parents that
  inlined differently must never dedup-merge, since the SQL text genuinely differs).
- :class:`NestedExtractStep` — the step the folded CHILD relation step is rewritten into.
  Its dep 0 is the parent ROW step (the same column the child bucket is seeded from); at
  execute time it reads the parent row's nested column (a list/dict asyncpg already
  materialised from the LATERAL's json), decodes it through the SAME ``resource.decode_rows``
  the batched path uses, and scatters per entry — a list (defaulting ``[]`` when the column
  is null/absent) for a hasMany, the single dict (or ``None``) for a hasOne. Because it does
  NO database work (the parent already fetched the rows), it is a pure synchronous step.

The wiring that DECIDES a fold (the parent's ``optimize`` and the strict safety predicate)
and that EMITS the LATERAL (the parent's ``build_query``) live on the parent pg steps; this
module is the substrate they build on.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Sequence, Tuple

from sqlalchemy import (
    and_,
    column,
    func,
    literal_column,
    select,
    table,
    text,
)
from sqlalchemy.dialects.postgresql import aggregate_order_by
from sqlalchemy.sql import ColumnElement
from sqlalchemy.sql.selectable import Lateral

from ..config import log
from ..core_steps import AccessStep
from ..step_model import Step
from .ordering import OrderTerm, order_clauses
from .resource import PgResource

if TYPE_CHECKING:
    from ..dag import Plan

# the relation kinds, matching PgRelation.kind exactly so a spec built off a relation
# carries the relation's own kind string unchanged.
KIND_HAS_ONE = "has_one"
KIND_HAS_MANY = "has_many"


@dataclass(frozen=True)
class InlineSpec:
    """One inlined child relation: everything the parent LATERAL must reproduce faithfully.

    Built by the parent step's ``optimize`` when a child relation PASSES the safety
    predicate, and carried on the replacement parent step so ``build_query`` can emit the
    ``LEFT JOIN LATERAL (...) AS <nested_alias> ON true`` and the folded child step can be
    rewritten into a :class:`NestedExtractStep` reading ``row[nested_alias]``.

    Equivalence rides on every field being a faithful copy of what the STANDALONE child
    would carry, so the nested rows are byte-identical to the batched child's rows:

    ``resource``
        the CHILD resource — its ``decode_rows`` decodes the nested rows, its ``columns`` /
        ``computed_projections`` shape the LATERAL's inner SELECT.
    ``kind``
        ``"has_one"`` (scatter the single dict / ``None``) or ``"has_many"`` (scatter the
        list / ``[]``); matches :class:`PgRelation` ``kind`` verbatim.
    ``local_columns`` / ``remote_columns``
        the FK correlation: the parent's local key column(s) equated to the child's remote
        column(s) inside the LATERAL's ``WHERE`` (``parent.local = child.remote``), the
        textbook single-column or composite-tuple correlation.
    ``order_by``
        the child's normalized ORDER BY (incl. the PK tie-break) reproduced inside
        ``json_agg(... ORDER BY ...)`` so the nested list order matches the standalone child.
    ``where_predicates``
        the child's UNIFORM WHERE predicates (resource ``select_customizer`` + any per-plan
        ``.where()``), reproduced verbatim inside the LATERAL — only folded when identical to
        what the standalone child carries (the safety predicate enforces this).
    ``computed``
        the child's computed projections (host SQL over the child's columns) emitted inside
        the LATERAL so a computed column survives the json round-trip identically.
    ``nested_alias``
        the extra column name the LATERAL projects onto the parent row; the
        :class:`NestedExtractStep` reads exactly this key.

    Frozen so it is a stable, hashable component of the parent step's dedup key (two parents
    inlining different children, orders or filters never merge). ``where_predicates`` and
    ``computed`` are SQLAlchemy ``ColumnElement``s (no stable hash); they ride
    ``compare=False`` so the dataclass stays hashable, and the parent's dedup key folds their
    content-based keys (the customization signature) separately, exactly as a standalone
    child would.
    """

    resource: PgResource = field(compare=False)
    kind: str
    nested_alias: str
    local_columns: Tuple[str, ...]
    remote_columns: Tuple[str, ...]
    order_by: Tuple[OrderTerm, ...] = ()
    where_predicates: Tuple[Any, ...] = field(default=(), compare=False)
    computed: Tuple[Any, ...] = field(default=(), compare=False)

    def __post_init__(self) -> None:
        if self.kind not in (KIND_HAS_ONE, KIND_HAS_MANY):
            raise ValueError(
                f"InlineSpec.kind must be {KIND_HAS_ONE!r} or {KIND_HAS_MANY!r}, "
                f"got {self.kind!r}"
            )
        if not self.local_columns or len(self.local_columns) != len(self.remote_columns):
            raise ValueError(
                "InlineSpec needs matching non-empty local/remote column tuples; got "
                f"local={self.local_columns} remote={self.remote_columns}"
            )

    @property
    def is_has_many(self) -> bool:
        """Whether the folded relation scatters a LIST per parent (else a single row)."""
        return self.kind == KIND_HAS_MANY


class NestedExtractStep(Step):
    """Read a folded child relation's rows off the PARENT row's nested column — no DB work.

    The step a folded child relation step is rewritten into once its parent absorbed it
    into a ``LEFT JOIN LATERAL``. Its single dependency (dep 0) is the parent ROW step — the
    SAME column the child object bucket is seeded from — so after the optimize pass repoints
    the child bucket's ``parent_step`` to this step, the bucket is seeded with exactly the
    row dicts the standalone child query would have produced.

    ``execute`` reads each parent row's ``alias`` column (a Python ``list`` / ``dict`` asyncpg
    already materialised from the LATERAL's json), runs ``resource.decode_rows`` so every
    attribute codec decodes identically to the batched path, and scatters per entry:

    - hasMany -> the decoded list, defaulting to ``[]`` when the column is ``null`` / absent
      (a parent with no children — exactly the batched path's empty-list scatter);
    - hasOne -> the single decoded dict, or ``None`` when the column is ``null`` / absent
      (a parent whose FK points nowhere — exactly the batched path's ``None`` scatter).

    No database round-trip happens here (the parent's LATERAL already fetched the rows), so
    the step is purely synchronous: ``execute`` returns a plain list, never a coroutine.
    """

    is_sync_and_safe = True

    def __init__(
        self,
        parent_row_step: Step,
        alias: str,
        resource: PgResource,
        kind: str,
    ) -> None:
        super().__init__()
        if kind not in (KIND_HAS_ONE, KIND_HAS_MANY):
            raise ValueError(
                f"NestedExtractStep.kind must be {KIND_HAS_ONE!r} or {KIND_HAS_MANY!r}, "
                f"got {kind!r}"
            )
        self.resource = resource
        # the nested column the parent's LATERAL projected; execute reads row[alias].
        self.alias = alias
        self.kind = kind
        # dep 0 is the parent ROW step — the same row dicts the child bucket is seeded from.
        self.add_dependency(parent_row_step)

    @property
    def is_has_many(self) -> bool:
        """Whether this extracts a LIST per parent (else a single row / None)."""
        return self.kind == KIND_HAS_MANY

    def extract_many(self, row: Any) -> List[Any]:
        """Decode the nested list off ONE parent row (null / absent -> ``[]``).

        The LATERAL emits ``coalesce(json_agg(...), '[]')`` so a present column is already a
        list, but a row that never got the column (a parent dropped from the LATERAL join, or
        a fake test row) is treated as no children — the batched path's empty-list scatter.
        """
        if isinstance(row, Mapping):
            nested = row.get(self.alias)
        else:
            nested = None
        if nested is None:
            return []
        return self.resource.decode_rows(nested)

    def extract_one(self, row: Any) -> Optional[Any]:
        """Decode the single nested row off ONE parent row (null / absent -> ``None``).

        The LATERAL emits ``to_jsonb(child) ... ORDER BY <order> LIMIT 1`` (or ``null`` when
        the FK points nowhere), so a present column is a single row dict; a missing column or
        an explicit
        ``null`` scatters ``None`` — the batched path's missing-hasOne scatter.
        """
        if isinstance(row, Mapping):
            nested = row.get(self.alias)
        else:
            nested = None
        if nested is None:
            return None
        # decode_rows materialises a one-row list; unwrap to the single decoded dict.
        return self.resource.decode_rows([nested])[0]

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        rows = values[0]
        if self.is_has_many:
            return [self.extract_many(rows[i]) for i in range(count)]
        return [self.extract_one(rows[i]) for i in range(count)]

    @property
    def peer_key(self) -> str:
        # two extracts merge only when they read the SAME nested column off the SAME child
        # resource as the same kind; the alias is the discriminator the parent's LATERAL
        # assigned, so two folds of different children under different aliases never merge.
        return (
            f"nested_extract|{self.resource.qualified_table}"
            f"|{self.alias}|{self.kind}"
        )

    def dedup_params(self) -> Tuple[Any, ...]:
        return (self.resource.qualified_table, self.alias, self.kind)


def inline_spec_from_relation(
    relation: Any,
    nested_alias: str,
    order_by: Sequence[OrderTerm] = (),
    where_predicates: Sequence[Any] = (),
    computed: Sequence[Any] = (),
) -> InlineSpec:
    """Build an :class:`InlineSpec` from a :class:`PgRelation` plus the resolved child state.

    A small convenience the parent's ``optimize`` reaches for once a child relation passes
    the safety predicate: it lifts the relation's ``target`` resource, ``kind`` and FK
    column tuples into the spec, alongside the child step's resolved ordering / customization /
    computed projections (which ``optimize`` reads off the child step, not the relation).
    """
    return InlineSpec(
        resource=relation.target,
        kind=relation.kind,
        nested_alias=nested_alias,
        local_columns=tuple(relation.local_columns),
        remote_columns=tuple(relation.remote_columns),
        order_by=tuple(order_by),
        where_predicates=tuple(where_predicates),
        computed=tuple(computed),
    )


def nested_alias_for(child_resource: PgResource, remote_columns: Sequence[str]) -> str:
    """A deterministic nested-column name for a fold of ``child_resource`` on ``remote``.

    The LATERAL projects exactly one extra column onto the parent row under this name, and
    the :class:`NestedExtractStep` reads ``row[<alias>]`` back. Derived from the child table
    and its correlated remote column(s) so the SAME fold always picks the SAME alias (the
    optimize pass is a deterministic plan transform) and two DIFFERENT folds onto one parent
    (e.g. ``Author.posts`` and ``Author.comments``) get distinct columns. The leading
    ``__inline_`` is a reserved prefix no real table column uses, so it cannot shadow a
    stored/computed attribute already in the parent SELECT-list.
    """
    return f"__inline_{child_resource.table}_{'_'.join(remote_columns)}"


def access_column_path(step: Step) -> Optional[Tuple[str, ...]]:
    """The single-column path a step projects off its parent, or ``None`` if it is not one.

    A relation's per-entry key step (the child's dep 0) is, for a SINGLE-column FK, a plain
    :class:`AccessStep` projecting ONE column off the parent row (``access(parent, (col,))``).
    This returns that ``(col,)`` path when ``step`` is exactly such an access — a one-segment
    path with no fallback — and ``None`` otherwise (a composite key is a
    :class:`grafast_py.core_steps.ListStep` of accesses, a deeper/fallback access, or any
    other step), so the safety predicate folds only the textbook single-column correlation
    it currently supports.
    """
    if not isinstance(step, AccessStep):
        return None
    if len(step.path) != 1:
        return None
    if step.fallback is not None:
        # a fallback means the access substitutes a value for a missing key; that is not the
        # plain FK projection the LATERAL correlation reproduces, so it is not a fold key.
        return None
    return step.path


def find_inline_candidates(
    parent: Step, plan: "Plan"
) -> List[Tuple[Step, InlineSpec]]:
    """The inlining SAFETY PREDICATE: the child folds of ``parent`` that pass EVERY condition.

    A PURE function (no DAG mutation, no SQL): it inspects ``parent`` and its dependents in
    ``plan`` and returns ``[(child_step, InlineSpec), ...]`` for each child relation step that
    is PROVABLY equivalent to fold into ``parent``'s statement via a ``LEFT JOIN LATERAL``.
    The parent's ``optimize`` consumes this list to build the replacement parent + rewrite
    each folded child into a :class:`NestedExtractStep`; here we only DECIDE, conservatively.
    Returns ``[]`` when inlining is off, or when no child passes — the no-op invariant
    (``optimize`` then returns ``self`` unchanged).

    Inlining is opportunistic and equivalence-preserving: the result with a fold MUST be
    byte-identical to the batched ``= ANY($1)`` path. So a fold of child C into parent P is
    taken ONLY when EVERY enumerated condition holds; ANY false condition (or any
    uncertainty) SKIPS C (P keeps the batched child — a redundant extra statement, never wrong
    data). The conditions, in order:

    1. TOGGLE: ``plan.inline_relations`` is True AND neither P's nor C's resource opted out
       (``opt_out_inline``).
    2. SAME SCHEMA/EXECUTOR, DISTINCT TABLE: P and C share a schema binding (the request
       executor is request-scoped and identical for every step, so the only cross-executor
       risk is a resource in a different schema/database — not foldable), AND C is not the
       SAME table as P (a SELF-referential relation: the flat ``build_lateral`` cannot alias
       the inner table apart from the outer parent, so its correlation would collapse — SKIP
       and keep the batched path; aliasing the inner select apart is not yet supported).
    3. FK-CORRELATABLE: C is a relation select whose dep 0 is a single-column
       :class:`AccessStep` reading ONE column off P's rows (the local FK), with C's
       ``match_columns`` the remote columns — the textbook ``parent.local = child.remote``.
       A composite key is a :class:`ListStep` and is not inlined (kept on the batched path).
    4. NOT SIDE-EFFECTING: both P and C are ``dedupable`` (a mutation is ``dedupable=False`` —
       NEVER inlined).
    5. UNPAGINATED hasMany: C is not window-sliced (``is_limited`` False); a hasOne is
       implicitly ``LIMIT 1`` and fine. A limited hasMany SKIPS (fall back).
    6. ORDERING FAITHFUL: C's ``order_by`` is expressible inside ``json_agg`` — it names only
       stored / computed columns (``assert_order_terms_stored`` already forbids ordering by a
       projection-only computed column); a violation SKIPS.
    7. CUSTOMIZATION REPRODUCIBLE: any child carrying a host WHERE predicate (a
       ``select_customizer`` scope or a per-plan ``.where()``) is not inlined — reproducing
       an arbitrary correlated predicate inside the LATERAL with a proven-identical
       customization signature is not yet supported. An UNFILTERED child folds with
       ``where_predicates=()``.
    8. COLUMNS JSON-STABLE: every folded child column survives ``to_jsonb`` -> JSON -> ``to_py``
       to the SAME Python value as the batched row decode (``resource.is_inline_json_safe``);
       a non-native codec OR a non-native / UNKNOWN-typed bare column (timestamptz / numeric /
       bytea / array / range / composite) SKIPS — an unprovable column is never assumed native.
    9. NO CONNECTION: C is a relation select, never a :class:`PgConnectionStep` (paginated /
       aggregate / keyset) — never inlined. (Enforced by the type check in 3.)
    """
    # local imports break the steps <-> inline import cycle (steps imports inline at module
    # load; inline only needs the concrete classes here, inside the predicate).
    from .connection import PgConnectionStep
    from .steps import PgSelectAllStep, PgSelectSingleStep, PgSelectStep

    # 1. TOGGLE — off, or this parent table opted out: no fold at all (the no-op path).
    if not plan.inline_relations:
        return []
    if parent.resource.opt_out_inline:
        return []
    # 4 (parent side). a side-effecting parent is never a fold root — fail safe.
    if not parent.dedupable:
        return []

    candidates: List[Tuple[Step, InlineSpec]] = []
    used_aliases: set = set()
    # the FK key steps projecting ONE column off the parent's rows; each such access's own
    # consumers are the candidate child relation selects keyed on it.
    for access in plan.dependents_of(parent):
        local_path = access_column_path(access)
        if local_path is None:
            continue  # not a single-column projection off the parent (e.g. a composite key)
        for child in plan.dependents_of(access):
            spec = inline_candidate_for(
                parent, child, access, local_path, used_aliases,
                connection_cls=PgConnectionStep,
                single_cls=PgSelectSingleStep,
                many_cls=PgSelectStep,
                all_cls=PgSelectAllStep,
            )
            if spec is not None:
                candidates.append((child, spec))
                used_aliases.add(spec.nested_alias)
    return candidates


def inline_candidate_for(
    parent: Step,
    child: Step,
    access: Step,
    local_path: Tuple[str, ...],
    used_aliases: set,
    *,
    connection_cls: type,
    single_cls: type,
    many_cls: type,
    all_cls: type,
) -> Optional[InlineSpec]:
    """Decide ONE (parent, child) fold: return its :class:`InlineSpec`, or ``None`` to SKIP.

    Applies conditions 3-9 (the toggle/parent conditions are pre-checked by
    :func:`find_inline_candidates`) to a single child consumer of a single-column key
    ``access`` (the FK key step projecting ``local_path`` off the parent). Each SKIP returns
    ``None`` with a debug log naming the reason, so an operator can see WHY a fold did not
    fire; a pass returns the spec the LATERAL is built from. Pure: it never mutates ``parent``
    or ``child``.
    """
    # 9 / 3 (type): a connection is paginated/aggregate/keyset — never folded; and only a
    # relation SELECT (single = hasOne, plain = hasMany) is a fold candidate. A
    # PgSelectAllStep is a ROOT collection (no key match), never a child relation.
    if isinstance(child, connection_cls):
        log.debug("inline skip", reason="connection", child=child.resource.name)
        return None
    if isinstance(child, all_cls) or not isinstance(child, many_cls):
        # not a key-matched relation select (root collection / non-pg consumer); skip.
        return None
    is_has_one = isinstance(child, single_cls)

    # 3 (correlation): the child's dep 0 must be EXACTLY this key access (so the child is
    # keyed off the parent's local column), and the child's match_columns the remote side —
    # the textbook `parent.local = child.remote`. A child depending on the access in any other
    # position is a shape the NestedExtractStep could not reproduce; SKIP.
    if child.dependencies[0] is not access:
        return None
    if child.is_composite:
        log.debug("inline skip", reason="composite", child=child.resource.name)
        return None

    # 2. SAME SCHEMA/EXECUTOR — a child in another schema/database is a different executor
    # binding; not foldable into the parent's statement.
    if child.resource.schema != parent.resource.schema:
        log.debug(
            "inline skip", reason="cross_schema",
            parent=parent.resource.name, child=child.resource.name,
        )
        return None

    # 2b. SAME TABLE (self-referential relation) — the child's resource is the SAME
    # schema+table as the parent's (e.g. employees.has_many("reports", employees, ...)).
    # build_lateral emits the LATERAL child as a bare `table(spec.resource.table)` with the
    # SAME unaliased name as the outer parent, so the `parent_table.c[local]` correlation
    # would resolve to the INNER table inside the subquery — collapsing the outer
    # correlation into a within-row comparison so EVERY parent silently gets [] / None.
    # Without an inner-select alias distinct from the outer parent, SKIP a self-relation and
    # keep the proven batched `= ANY($1)` path (one extra statement), matching the
    # limited/filtered/composite skips.
    if child.resource.table == parent.resource.table:
        log.debug(
            "inline skip", reason="self_relation",
            parent=parent.resource.name, child=child.resource.name,
        )
        return None

    # 1 (child side). a child table that opted out is never folded.
    if child.resource.opt_out_inline:
        log.debug("inline skip", reason="opt_out", child=child.resource.name)
        return None

    # 4. NOT SIDE-EFFECTING — a mutation child (dedupable=False) is never inlined.
    if not child.dedupable:
        log.debug("inline skip", reason="mutation", child=child.resource.name)
        return None

    # 5. UNPAGINATED hasMany — a per-parent window slice cannot ride a json_agg.
    if child.is_limited:
        log.debug("inline skip", reason="limited", child=child.resource.name)
        return None

    # 7. CUSTOMIZATION — never fold a customizer-bearing child, and never fold a per-plan-filtered
    # child; only an UNFILTERED, un-customized relation is inlined.
    #
    # A child whose resource carries a select_customizer is request-SCOPED: the predicates it
    # produced THIS request may differ next request — in particular a structure-branching customizer
    # can return NO predicates this request (e.g. an admin branch) and a scoping filter the next. If
    # we folded it, the child leaves `plan.steps` (it becomes a NestedExtractStep on the parent and
    # tree_shake drops the original), so the cache-hit structural-divergence guard
    # (lookup_cached_plan / customizer_structure_matches) could no longer re-check it, and a later
    # request would inherit THIS request's (here unfiltered) child rows — a cross-context leak under
    # cache_plans + inline_relations. So keep ANY customizer-bearing child on the batched path, where
    # its step stays in `plan.steps` and the guard runs. (An un-customized child with a per-plan
    # `.where()` carries those predicates directly, so the `where_predicates` check below skips it.)
    if child.resource.select_customizer is not None:
        log.debug("inline skip", reason="customizer", child=child.resource.name)
        return None
    if child.where_predicates:
        log.debug("inline skip", reason="filtered", child=child.resource.name)
        return None

    # 8. COLUMNS JSON-STABLE — every folded column must survive the json round-trip: a
    # non-native codec OR a non-native/unknown-typed bare column is not provably stable.
    if not child.resource.is_inline_json_safe:
        log.debug("inline skip", reason="unsafe_column", child=child.resource.name)
        return None

    # 6. ORDERING FAITHFUL — fail loud if the child orders by a projection-only computed
    # column (it cannot live inside json_agg's ORDER BY); reuse the existing guard. The order
    # is otherwise reproduced verbatim (normalize_order already baked in the PK tie-break at
    # construction), so the nested list order matches the standalone child byte-for-byte.
    child.resource.assert_order_terms_stored(child.order_by)

    remote_columns = tuple(child.match_columns)
    alias = nested_alias_for(child.resource, remote_columns)
    if alias in used_aliases:
        # two distinct folds collided on a derived alias (same child table + remote columns
        # under one parent — an unusual duplicate selection); disambiguate deterministically.
        alias = f"{alias}_{len(used_aliases)}"
    kind = KIND_HAS_ONE if is_has_one else KIND_HAS_MANY
    return InlineSpec(
        resource=child.resource,
        kind=kind,
        nested_alias=alias,
        local_columns=tuple(local_path),
        remote_columns=remote_columns,
        order_by=tuple(child.order_by),
        # condition 7 guaranteed an unfiltered child, so no predicates ride the LATERAL;
        # computed projections are emitted from the child resource at build time.
        where_predicates=(),
        computed=tuple(child.resource.computed_projections()),
    )


def lateral_correlation(
    parent_table: Any,
    local_columns: Sequence[str],
    remote_columns: Sequence[str],
) -> ColumnElement:
    """The LATERAL ON-correlation: ``child.remote[i] = parent.local[i]`` ANDed over the FK.

    A single-column FK yields one equality; a COMPOSITE FK ANDs one equality per matched
    pair — the textbook ``parent.local = child.remote`` correlation the standalone child's
    ``= ANY(:keys)`` / tuple-IN reproduces row-for-row. The child columns are bare
    :func:`column` refs (resolved against the child table inside the inner select); the
    parent columns reference ``parent_table`` so they stay in the OUTER scope when the inner
    select is correlated (see :func:`build_lateral`).
    """
    equalities = [
        column(remote) == parent_table.c[local]
        for local, remote in zip(local_columns, remote_columns)
    ]
    if len(equalities) == 1:
        return equalities[0]
    return and_(*equalities)


def lateral_order_clauses(
    inner: Any, order_by: Sequence[OrderTerm]
) -> List[ColumnElement]:
    """ORDER BY elements for the json_agg, bound to the inner subquery's columns.

    The nested list order must match the standalone child's ``ORDER BY`` byte-for-byte
    (incl. the PK tie-break already baked into ``order_by`` by ``normalize_order``). Because
    the ordering lives INSIDE ``json_agg(... ORDER BY ...)`` — a plain subquery ORDER BY does
    not survive aggregation in Postgres — each term references the inner subquery column
    (``inner.c[name]``) rather than a bare table column, with the same direction / NULLS
    placement :func:`grafast_py.pg.ordering.order_clauses` emits.
    """
    clauses: List[ColumnElement] = []
    for term in order_by:
        clause = inner.c[term.column]
        if term.descending:
            clause = clause.desc()
        if term.nulls == "first":
            clause = clause.nulls_first()
        elif term.nulls == "last":
            clause = clause.nulls_last()
        clauses.append(clause)
    return clauses


def build_lateral(spec: InlineSpec, parent_table: Any) -> Lateral:
    """Emit the correlated ``LEFT JOIN LATERAL`` subquery that folds ONE child relation.

    Returns a SQLAlchemy Core :class:`~sqlalchemy.sql.selectable.Lateral` the parent's
    ``build_query`` outer-joins ``ON true`` and projects as one extra column
    ``spec.nested_alias`` on the parent row. The shape, per kind, is:

    - hasMany -> ``SELECT coalesce(json_agg(to_jsonb(child) ORDER BY <order>), '[]'::json)``
      so a parent with no children yields ``[]`` (the batched path's empty-list scatter),
      and the nested list is in the child's exact order;
    - hasOne  -> ``SELECT to_jsonb(child)`` over the inner select with the child's
      ``ORDER BY <order>`` (so a non-unique remote FK picks the SAME row the batched
      ``rows[0]`` does — the PK-floored lowest row) then ``LIMIT 1``, yielding the single
      row dict or ``NULL`` (the batched path's ``None`` scatter) when the FK points nowhere.

    The inner select projects the child's STORED columns plus its computed projections (so a
    computed column survives the json round-trip identically), correlated on
    ``child.remote = parent.local`` and AND-ed with the child's reproduced ``where_predicates``
    — all INSIDE the LATERAL, so the nested rows are byte-identical to the standalone child's
    rows. ``parent_table`` is correlated OUT (it lives in the outer FROM), which is what makes
    this a LATERAL rather than a cross join.
    """
    child = table(
        spec.resource.table,
        *[column(c) for c in spec.resource.columns],
        schema=spec.resource.schema,
    )
    # project the child's STORED columns PLUS each computed attribute's labelled expression,
    # so to_jsonb(row) carries every attribute the standalone child would have decoded.
    inner_select = (
        select(child, *spec.computed)
        .where(
            lateral_correlation(parent_table, spec.local_columns, spec.remote_columns)
        )
        # keep parent_table in the OUTER scope: without this SQLAlchemy auto-FROMs the
        # referenced parent table INTO the inner FROM (a cross join), not a LATERAL.
        .correlate(parent_table)
    )
    for predicate in spec.where_predicates:
        inner_select = inner_select.where(predicate)
    if not spec.is_has_many:
        # hasOne: a single related row. The batched PgSelectSingleStep floors its order to
        # ORDER BY <pk> (normalize_order) and takes rows[0], so on a NON-UNIQUE remote FK
        # (several rows match the correlation) it deterministically returns the lowest-PK
        # row. A bare LIMIT 1 here would return an arbitrary HEAP row instead — a silent
        # divergence — so reproduce the child's ORDER BY inside the inner select BEFORE the
        # LIMIT 1. Unlike the hasMany json_agg (where a subquery ORDER BY does not survive
        # aggregation, hence lateral_order_clauses inside the aggregate), a plain subquery
        # ORDER BY here survives into LIMIT 1, so order by the child columns directly.
        if spec.order_by:
            inner_select = inner_select.order_by(*order_clauses(spec.order_by))
        inner_select = inner_select.limit(1)
    inner = inner_select.subquery(f"{spec.nested_alias}_src")

    # to_jsonb over the WHOLE inner row, referenced by the subquery's name so Postgres
    # serialises every projected column (stored + computed) in declaration order.
    row_json = func.to_jsonb(literal_column(inner.name).self_group())

    if spec.is_has_many:
        # json_agg with the order INSIDE the aggregate (a subquery ORDER BY does not survive
        # aggregation), coalesced to '[]' so an empty relation is [] not NULL.
        ordered = aggregate_order_by(row_json, *lateral_order_clauses(inner, spec.order_by))
        nested = func.coalesce(func.json_agg(ordered), text("'[]'::json")).label(
            spec.nested_alias
        )
    else:
        nested = row_json.label(spec.nested_alias)

    return select(nested).select_from(inner).lateral(f"{spec.nested_alias}_lat")


def fold_inline_candidates(
    parent: Step,
    candidates: List[Tuple[Step, InlineSpec]],
    plan: "Plan",
) -> Step:
    """Perform the fold: build the replacement parent + rewrite each folded child.

    The shared body of every pg parent step's ``optimize`` (the per-class wrapper only
    decides the candidate list and supplies the class-specific clone). Given the children
    the safety predicate chose to fold (``candidates``), this:

    1. builds the REPLACEMENT parent — the same step carrying the same skeleton PLUS the
       :class:`InlineSpec`\\ s (via ``parent.clone_with_inline_specs``), whose ``build_query``
       grows one ``LEFT JOIN LATERAL`` per spec, so the child rows ride the parent's ONE
       statement;
    2. rewrites each folded child relation step into a :class:`NestedExtractStep` keyed on
       the REPLACEMENT parent (dep 0 — the same parent row dicts the child bucket is seeded
       from), reading the nested column the LATERAL projected and scattering per entry; and
    3. records each ``child -> NestedExtractStep`` via ``plan.record_replacement`` so the
       optimize pass's survivor-chain rewire repoints every reference to the folded child
       (the child object bucket's ``parent_step`` AND the AccessSteps reading child columns)
       to the extract step — which produces byte-identical rows off the parent's LATERAL.

    Returns the replacement parent (what ``optimize`` returns). The orphaned key
    :class:`AccessStep` (the child's old dep 0, no longer consumed once the child reads off
    the parent row) is left for ``finalize_plan``'s ``tree_shake`` to drop. Composes for
    nesting: a parent re-optimizes after its child folded (the optimize fixpoint), so a
    3-level fold collapses into one statement.
    """
    replacement = parent.clone_with_inline_specs(
        [spec for _child, spec in candidates]
    )
    plan.add_step(replacement)
    for child, spec in candidates:
        extract = NestedExtractStep(
            replacement, spec.nested_alias, spec.resource, spec.kind
        )
        plan.add_step(extract)
        # repoint every reference to the folded child to the extract step; the optimize
        # pass drains this into its `replaced` map and runs the same rewire dedup uses.
        plan.record_replacement(child, extract)
    log.debug(
        "inline fold",
        parent=parent.resource.name,
        folded=len(candidates),
        aliases=[spec.nested_alias for _c, spec in candidates],
    )
    return replacement


__all__ = [
    "InlineSpec",
    "NestedExtractStep",
    "inline_spec_from_relation",
    "build_lateral",
    "lateral_correlation",
    "lateral_order_clauses",
    "find_inline_candidates",
    "inline_candidate_for",
    "fold_inline_candidates",
    "access_column_path",
    "nested_alias_for",
    "KIND_HAS_ONE",
    "KIND_HAS_MANY",
]
