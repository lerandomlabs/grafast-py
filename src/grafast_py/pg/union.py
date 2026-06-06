"""pgUnionAll: a keyset-paged Relay connection over N member tables via ``UNION ALL``.

The CROSS-TABLE polymorphism shape: a GraphQL union (or interface) whose concrete types
live in SEPARATE tables — ``SearchResult = Post | Comment``, each its own
:class:`PgResource`. :class:`PgUnionAllStep` fetches all members in ONE ``UNION ALL``
statement whose branches project a SHARED, NULL-PADDED column shape plus a ``__typename``
literal tag, so each returned row already carries its concrete type. The
``resolve_type_from_tag('__typename')`` bridge then resolves each row at COMPLETION time
and the existing completion-time abstract dispatch groups rows by concrete type and plans
each group's sub-selection like a normal object field — so a member's type-specific fields
and nested pg relations batch per concrete-type group, with NO plan-time polymorphism and
NO core-engine change (the SAME machinery the single-table discriminator shape rides).

The union is ordered by SHARED order columns (present in EVERY member), keyset-sliced
exactly like :class:`grafast_py.pg.connection.PgConnectionStep`: a keyset ``after``/``before``
seek predicate AND-ed onto every branch's WHERE, then a ``row_number() OVER (PARTITION BY
match ORDER BY <order>)`` window over the union, fetching ONE extra row per partition for
``hasNextPage`` — so the page slice is IN SQL and cursors are seek (not offset) cursors
rejected loudly under a different ordering. ``totalCount`` is a SEPARATE batched
``count(*)`` over the union (grouped by the match key in per-parent mode), issued only when
the selection set asks for it — so a union layer is at most TWO statements regardless of
member count.

TWO modes, mirroring the plain selects:

- ROOT collection (``Query.search``): no key match — every branch scans its whole table
  (under its per-branch WHERE), one synthetic bucket entry, plain ``LIMIT`` on the page.
- PER-PARENT (``Author.activity``): each member matches ``<match> = ANY(:keys)`` against
  the shared key step (an FK on each member), partitioned by the match key so every
  parent's page slices independently in the one batched statement, then scattered back.

DEDUP: every SQL-affecting input folds into ``peer_key`` + ``dedup_params`` with
value-included ``literal_binds`` (the member set — each member's qualified table, typename,
match columns and per-branch WHERE signature — plus the shared projection, the order, the
cursor VALUES, ``first``/``last`` and ``needs_total``). Two union steps differing only in any
such input get different dedup keys; identical ones merge. The per-branch WHERE predicates
ride :func:`grafast_py.pg.customize.predicate_key` exactly like a select's customization, so
two unions whose branches differ only by a filter VALUE never merge.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from sqlalchemy import (
    any_,
    bindparam,
    column,
    func,
    literal,
    null,
    select,
    table,
    tuple_,
    union_all,
)
from sqlalchemy.sql import ColumnElement, visitors
from sqlalchemy.sql.elements import BindParameter

from ..core_steps import access
from ..step_model import Step
from .connection import connection_needs_total
from .customize import check_predicate, placeholder_binds_in, predicate_key
from .cursor import decode_keyset_cursor, effective_nulls, encode_keyset_cursor, keyset_where
from .executor import current_pg_request
from .ordering import OrderTerm, normalize_order
from .placeholders import (
    Placeholder,
    placeholder_source,
    placeholder_source_tag,
    resolve_placeholder,
)
from .resource import PgResource
from .steps import as_match_columns, grouping_key, normalize_lookup_key

# the column carrying each row's concrete GraphQL type name, projected as a literal into
# every branch (``literal('Post') AS __typename``). The resolve_type_from_tag bridge reads
# this key to dispatch each row at completion time, so it is the union's discriminator
# column. Named with the GraphQL meta-field spelling so a host wiring the tag bridge uses
# the obvious key.
TYPENAME_COLUMN = "__typename"

# sentinel distinguishing "caller omitted the resolved cursor values" (default to the step's
# plan-time literal decode) from "caller passed None" (genuinely no cursor on that side).
_UNSET: Any = object()


@dataclass(frozen=True)
class PgUnionMember:
    """One branch of a ``UNION ALL``: a member table tagged with its concrete type name.

    ``resource`` is the member's table source; ``type_name`` is the GraphQL concrete type
    NAME tagged onto every row of this branch (the ``__typename`` literal the tag bridge
    reads). ``match`` is the member's key column(s) for PER-PARENT mode — a single column
    name (the FK matched ``= ANY(:keys)``) or a composite tuple; ``None`` in ROOT mode
    (no key match). ``where`` are per-branch UNIFORM Core predicates AND-ed onto THIS
    branch's WHERE only (so one member can be scoped — e.g. a soft-delete filter — without
    touching its peers).

    NOTE: a member resource's ``select_customizer`` is NOT auto-applied by the union (only
    PgSelect auto-applies its resource's auth scope; pgUnionAll requires host-supplied member
    scoping — upstream pgUnionAll parity). A scoped member (soft-delete / tenant / visibility)
    must RESTATE that predicate here via ``where=``; the union will not add it for you.

    Frozen + hashable so the member tuple folds straight into the union step's dedup key;
    ``where`` is excluded from the dataclass identity (Core predicates have no stable hash —
    they are value-discriminated via :func:`predicate_key` in the step's signature instead)
    and stored as a tuple so the dataclass stays hashable.
    """

    resource: PgResource
    type_name: str
    match: Optional[Union[str, Sequence[str]]] = None
    where: Tuple[ColumnElement, ...] = field(default=(), compare=False)

    def __post_init__(self) -> None:
        # validate the per-branch predicates at construction (a raw string / unbound bind /
        # reserved skeleton name fails loud here, never at the query), mirroring the select
        # customization seam. Re-store the validated tuple so a list arg is normalised.
        object.__setattr__(
            self, "where", tuple(check_predicate(p) for p in self.where)
        )

    @property
    def match_columns(self) -> Tuple[str, ...]:
        """The member's key column tuple for per-parent mode (empty in root mode)."""
        return () if self.match is None else as_match_columns(self.match)

    @property
    def is_composite(self) -> bool:
        """Whether this member's key spans more than one column (the tuple-IN path)."""
        return len(self.match_columns) > 1

    def where_signature(self) -> Tuple[str, ...]:
        """The dedup signature of this branch's WHERE predicates.

        A LITERAL predicate is compiled with ``literal_binds`` (via :func:`predicate_key`) so
        two members differing only by a filter VALUE produce different signatures and never
        merge — the same value-discrimination the select customization uses. A PLACEHOLDER
        predicate (a ``pg_placeholder`` over a variable-derived value) keys value-agnostically
        off its stable source tag instead, so two requests of the same document share one key
        while two different sources never merge — exactly as a select's customization does.
        """
        return tuple(
            predicate_key(p, placeholder_binds_in(p) or None) for p in self.where
        )


class PgUnionAllStep(Step):
    """Batched keyset-paged Relay connection over N member tables joined by ``UNION ALL``.

    Each member projects the SHARED column shape (NULL-padding the columns it lacks) plus
    its ``__typename`` literal tag; the branches are ``UNION ALL``-ed, keyset-sliced over the
    SHARED order columns, and (per-parent mode) partitioned by the match key so every parent
    pages independently in one statement. ``needs_total`` gates a SEPARATE batched
    ``count(*)`` over the union. The result per bucket entry is a Relay connection dict
    ``{edges, nodes, totalCount, pageInfo}`` — identical in shape to
    :class:`grafast_py.pg.connection.PgConnectionStep`, so its sub-fields project the same
    way and ``edges[].node`` is the tagged row dict the completion-time dispatch resolves.

    ROOT mode (no ``key_step``): every branch scans its table under its per-branch WHERE,
    one synthetic bucket entry, plain ``LIMIT`` page. PER-PARENT mode (a ``key_step``): each
    member matches ``<match> = ANY(:keys)`` and the window partitions by the match key.
    """

    is_sync_and_safe = False
    # needs the per-request source-tag -> value map (BucketExtra.source_values) to resolve its
    # page-size / cursor / per-member WHERE placeholders into the compiled statement's params at
    # render time (and to digest-validate a variable cursor per request).
    wants_extra = True

    def __init__(
        self,
        members: Sequence[PgUnionMember],
        shared_columns: Sequence[str],
        order_by: Sequence[Union[str, OrderTerm]],
        key_step: Optional[Step] = None,
        first: Optional[Union[int, Placeholder]] = None,
        after: Optional[Union[str, Placeholder]] = None,
        last: Optional[Union[int, Placeholder]] = None,
        before: Optional[Union[str, Placeholder]] = None,
        order_is_unique: bool = False,
        needs_total: bool = False,
    ) -> None:
        super().__init__()
        if not members:
            raise ValueError("a pgUnionAll needs at least one member")
        self.members: Tuple[PgUnionMember, ...] = tuple(members)
        # the columns present in EVERY member, projected by name in each branch — the
        # selectable shape the order/cursor columns and every interface field read from.
        # Each member additionally NULL-pads the union-wide member-specific columns so the
        # branches share one column list (see member_columns).
        self.shared_columns: Tuple[str, ...] = tuple(shared_columns)
        if not self.shared_columns:
            raise ValueError("a pgUnionAll needs at least one shared column")
        # the union-wide ordered set of MEMBER-specific columns (a column some member has
        # that is NOT shared): every branch projects each either as its own column or
        # NULL-padded, so the UNION ALL legs are union-compatible. Order is stable
        # (shared first, then first-seen member columns) so the projection — and the dedup
        # key — is deterministic.
        self.member_columns: Tuple[str, ...] = self.compute_member_columns()
        self.order_is_unique = order_is_unique
        # the order spec drives the keyset window AND the cursor. normalize_order appends the
        # FIRST member's primary key as a tie-break, but that id is unique only WITHIN one
        # member table, NOT across the UNION ALL — two rows from different members can share
        # the same (order_cols.., id), making the order non-total and the seek drop/duplicate
        # a tied row at the page boundary. So we ALSO append the __typename tag as the FINAL
        # tie-break: it is a member-constant literal projected into every branch, so
        # (order_cols.., id, __typename) is union-wide TOTAL (no two distinct rows share it —
        # two members never share a type name, and within a member id is unique). The keyset
        # comparator and the cursor carry it too (see cursor_predicate / build_connection).
        base_order = normalize_order(
            order_by,
            primary_key=self.members[0].resource.primary_key,
            order_is_unique=order_is_unique,
        )
        self.order_by: Tuple[OrderTerm, ...] = base_order + (OrderTerm(TYPENAME_COLUMN),)
        self.assert_order_columns_shared()
        # page size: a plan-time literal int/None (value-included key) OR a ``Placeholder``
        # (variable-derived, source-keyed). The page math unwraps the runtime value; the
        # keyset window binds page_limit as a param regardless, so a placeholder changes only
        # the dedup key, never the SQL.
        self.first = first
        self.last = last
        self.needs_total = needs_total
        # the stable source tags of a variable-derived after/before cursor (``None`` for a
        # literal cursor); the dedup key emits these in place of the decoded cursor VALUES so
        # a variable cursor keys value-agnostically (a cache hit), while the decoded values
        # still drive the keyset SQL bound as params.
        self.after_source: Optional[str] = placeholder_source_tag(after)
        self.before_source: Optional[str] = placeholder_source_tag(before)
        # a connection pages forward XOR reverse; both forward (first/after) and reverse
        # (last/before) args is undefined under Relay — reject it loudly rather than page
        # one way and silently ignore the other (mirrors PgConnectionStep). A Placeholder is
        # never ``None``, so a variable-derived arg counts toward its direction.
        forward_arg = first is not None or after is not None
        reverse_arg = last is not None or before is not None
        if forward_arg and reverse_arg:
            raise ValueError(
                "ambiguous Relay pagination: forward (first/after) and reverse "
                "(last/before) args were both supplied; a pgUnionAll pages forward XOR "
                "reverse — pass only one direction"
            )
        self.reverse = reverse_arg
        # keep the raw after/before (a literal cursor str or a ``Placeholder``) so a LITERAL
        # cursor can be re-decoded on an order mutation and a VARIABLE cursor resolved + decoded
        # per request at render. ``None`` when the arg was absent.
        self.after = after
        self.before = before
        # decode LITERAL cursors at plan time (request-stable, and the dedup key reads their
        # decoded VALUES); a VARIABLE (``Placeholder``) cursor stores no decoded values on the
        # shared step — it is resolved against this request's source map and decoded at render
        # (see :meth:`resolve_cursor_values`), keying value-agnostically off its source tag.
        # The decode is digest-validated: a literal cursor minted under a different ordering —
        # or garbage — raises a clear "minted for a different ordering" ``ValueError``.
        self.after_values, self.before_values = self.decode_literal_cursors()
        # PER-PARENT mode iff a key step is wired. Every member then needs a match key, and
        # all members must agree on the match arity (the window partitions by ONE shared
        # set of match columns across the union). ROOT mode has no key match.
        self.is_per_parent = key_step is not None
        if key_step is not None:
            self.assert_members_have_match()
            self.match_columns: Tuple[str, ...] = self.members[0].match_columns
            # dep 0 is the key step; values[0] is the key column at execute time.
            self.add_dependency(key_step)
        else:
            self.assert_members_have_no_match()
            self.match_columns = ()

    # ------------------------------------------------------------------ validation

    def compute_member_columns(self) -> Tuple[str, ...]:
        """The union-wide ordered set of member-specific (non-shared) columns.

        Every member's stored columns minus the shared set, first-seen order preserved so
        the projection (and the dedup key) is deterministic. A branch projects each of
        these either as its own column or NULL-padded, making the UNION ALL legs
        union-compatible.
        """
        shared = set(self.shared_columns)
        seen: Dict[str, None] = {}
        for member in self.members:
            for col in member.resource.columns:
                if col not in shared and col not in seen:
                    seen[col] = None
        return tuple(seen)

    def assert_order_columns_shared(self) -> None:
        """Fail loud if any ORDER BY column is not a SHARED column.

        The keyset orders the WHOLE union, so every order column must exist (by name) in
        every branch — i.e. be a shared column. An order column some member lacks would
        reference a NULL-padded alias and silently mis-sort that member's rows; reject it
        with a clear, column-named error instead. The ``__typename`` discriminator is
        EXEMPT: it is the union's own per-branch literal tie-break (appended to make the
        order total), projected into every leg, so it is always present even though it is
        not a stored shared column.
        """
        shared = set(self.shared_columns) | {TYPENAME_COLUMN}
        for term in self.order_by:
            if term.column not in shared:
                raise ValueError(
                    f"pgUnionAll order column {term.column!r} is not a shared column "
                    f"(shared: {sorted(shared)}); order only by columns present in every "
                    "member"
                )

    def assert_members_have_match(self) -> None:
        """Per-parent mode: every member needs the SAME match key, drawn from shared columns.

        The step keeps a SINGLE set of match columns (the first member's) and uses those
        names for the page-window PARTITION/ORDER, the Python SCATTER (``grouping_key``) and
        the COUNT projection — a ONE-window design (we deliberately do NOT port upstream's
        per-identifier LATERAL machinery). So a member whose match columns differ in NAME or
        ORDER from the first member's would be partitioned/grouped/counted under columns it
        does not project: its rows would group under ``None`` and be silently dropped from
        the page, and the count leg would project a column that member lacks (a Postgres
        ``column ... does not exist`` error). Likewise a match column that is not SHARED is
        NULL-padded in the peers that lack it, so the partition/scatter key is ``None`` there.
        Require, fail-loud at construction, that every member declares the IDENTICAL match
        columns and that each match column is a shared (real, identically-named, non-NULL)
        projection in every branch.
        """
        expected = self.members[0].match_columns
        if not expected:
            raise ValueError(
                "per-parent pgUnionAll (a key_step was supplied) needs each member to "
                "declare a match column; the first member has none"
            )
        shared = set(self.shared_columns)
        for col in expected:
            if col not in shared:
                raise ValueError(
                    f"pgUnionAll match column {col!r} is not a shared column "
                    f"(shared: {sorted(shared)}); the match key partitions/scatters/counts "
                    "the whole union, so it must be a shared column projected by every member"
                )
        for member in self.members:
            if len(member.match_columns) != len(expected):
                raise ValueError(
                    f"pgUnionAll members disagree on match arity: {member.type_name!r} "
                    f"has {len(member.match_columns)} match column(s), expected "
                    f"{len(expected)}; every member must match the same number of key columns"
                )
            if member.match_columns != expected:
                raise ValueError(
                    f"pgUnionAll members disagree on match columns: {member.type_name!r} "
                    f"matches {list(member.match_columns)}, expected {list(expected)}; "
                    "every member must match the SAME key column(s) (same names, same order) "
                    "— the step partitions/scatters/counts under one shared match key, so a "
                    "divergent name would silently drop that member's rows"
                )

    def assert_members_have_no_match(self) -> None:
        """Root mode: no member may declare a match key (there is no key to match)."""
        for member in self.members:
            if member.match_columns:
                raise ValueError(
                    f"root pgUnionAll (no key_step) must not declare a member match; "
                    f"{member.type_name!r} declares {member.match_columns}"
                )

    @property
    def is_composite(self) -> bool:
        """Whether the per-parent match key spans more than one column."""
        return len(self.match_columns) > 1

    def decode_literal_cursors(self) -> Tuple[Optional[List[Any]], Optional[List[Any]]]:
        """Decode the LITERAL after/before cursors against the current order (plan time).

        A LITERAL cursor (a bare str) is request-stable and its decoded VALUES discriminate the
        dedup key, so it is decoded here at construction. A VARIABLE (``Placeholder``) cursor is
        NOT decoded here — its value is resolved + decoded per request at RENDER time (see
        :meth:`resolve_cursor_values`), keying value-agnostically off its source tag. The decode
        is digest-validated: a literal cursor minted under a different ordering — or garbage —
        raises a clear ``ValueError`` rather than seeking with stale values.
        """
        after_values = (
            decode_keyset_cursor(self.after, self.order_by)
            if isinstance(self.after, str) and self.after
            else None
        )
        before_values = (
            decode_keyset_cursor(self.before, self.order_by)
            if isinstance(self.before, str) and self.before
            else None
        )
        return after_values, before_values

    def resolve_cursor_values(
        self, source_values: Mapping[str, Any]
    ) -> Tuple[Optional[List[Any]], Optional[List[Any]]]:
        """The decoded after/before seek values for THIS request (digest-validated at render).

        A LITERAL cursor's values were decoded at plan time; a VARIABLE (``Placeholder``)
        cursor is resolved to THIS request's cursor string from ``source_values`` and decoded
        HERE — never stored on the shared step, so two concurrent cache HITs decode their OWN
        cursor with no bleed. The decode is digest-validated per request (a variable cursor
        minted under a different ordering still fails loud, at render time).
        """
        after_values = self.after_values
        before_values = self.before_values
        if isinstance(self.after, Placeholder):
            after_cursor = resolve_placeholder(self.after, source_values)
            after_values = (
                decode_keyset_cursor(after_cursor, self.order_by)
                if after_cursor
                else None
            )
        if isinstance(self.before, Placeholder):
            before_cursor = resolve_placeholder(self.before, source_values)
            before_values = (
                decode_keyset_cursor(before_cursor, self.order_by)
                if before_cursor
                else None
            )
        return after_values, before_values

    def member_where_params(self, source_values: Mapping[str, Any]) -> Dict[str, Any]:
        """The execute-time params for every member branch's WHERE placeholder binds.

        Unlike a plain select the union subclasses :class:`~grafast_py.step_model.Step` directly
        (not ``PgCustomizable``), so it gathers its own placeholder params: each member's
        per-branch ``where`` predicate may carry value-LESS ``pg_placeholder`` binds, whose
        runtime value is supplied per request in the compiled statement's ``params`` (keyed by
        the bind name, resolved by its source tag against ``source_values``) rather than baked
        on the SHARED bind. A union with only literal member predicates returns ``{}``. The same
        bind names appear in BOTH the page leg and the count leg (the union rebuilds the same
        member predicates each render), so these params serve both.
        """
        params: Dict[str, Any] = {}
        for member in self.members:
            for predicate in member.where:
                for bind in visitors.iterate(predicate):
                    if not isinstance(bind, BindParameter):
                        continue
                    source = placeholder_source(bind)
                    if source is not None:
                        params[bind.key] = source_values.get(source)
        return params

    # ------------------------------------------------------------------ SQL build

    def effective_order(self) -> Tuple[OrderTerm, ...]:
        """The ORDER BY to emit: the request order, reversed for ``last``/``before``.

        Reverse paging walks from the tail: flip every term's direction (and NULLS
        placement) so the window numbers rows from the end, take the first ``last+1`` of
        that, then re-reverse the page in Python — exactly the PgConnectionStep rule.
        """
        if not self.reverse:
            return self.order_by
        return tuple(
            OrderTerm(t.column, not t.descending, _reverse_nulls(t))
            for t in self.order_by
        )

    def page_size(self, source_values: Mapping[str, Any] = {}) -> Optional[int]:
        """The active page size as a runtime int, resolving a variable-derived ``Placeholder``.

        ``last`` for reverse, ``first`` for forward; a ``Placeholder`` yields its request
        value from ``source_values``. The page arithmetic (``page_limit`` + the one-extra-row
        probe in ``build_connection``) reads THIS, so a placeholder page size pages exactly
        like a literal — only the dedup key (which keeps the sentinel) differs.
        """
        return resolve_placeholder(self.last if self.reverse else self.first, source_values)

    def page_limit(self, source_values: Mapping[str, Any] = {}) -> Optional[int]:
        """The per-partition row cap: the page size PLUS ONE extra (for hasNextPage)."""
        size = self.page_size(source_values)
        return None if size is None else size + 1

    def cursor_predicate(
        self,
        member: PgUnionMember,
        after_values: Optional[List[Any]],
        before_values: Optional[List[Any]],
    ) -> Optional[ColumnElement]:
        """The keyset WHERE predicate for ``member``'s branch, or ``None`` when unpaged.

        Built over the SHARED order columns PLUS the ``__typename`` discriminator, so it
        AND-s onto this branch's WHERE (each branch carries the shared columns; the
        ``__typename`` term compares against this member's constant ``literal(type_name)``
        rather than a non-existent ``__typename`` column — that final tie-break is what
        makes the seek total across the union, so a cross-branch ``(order, id)`` collision
        no longer drops or duplicates a tied row). Forward selects rows strictly AFTER
        ``after``; reverse strictly BEFORE ``before``. The seek values are passed in (resolved
        per request via :meth:`resolve_cursor_values`), never read off the shared step. The
        cursor column types come from the FIRST member's resource (the order columns are
        shared, so their type is the same in every member's source).
        """
        column_types = self.members[0].resource.column_types
        column_exprs = {TYPENAME_COLUMN: literal(member.type_name)}
        if self.reverse:
            if before_values is None:
                return None
            return keyset_where(
                self.order_by,
                before_values,
                after=False,
                column_types=column_types,
                column_exprs=column_exprs,
            )
        if after_values is None:
            return None
        return keyset_where(
            self.order_by,
            after_values,
            after=True,
            column_types=column_types,
            column_exprs=column_exprs,
        )

    def match_predicate(self, member: PgUnionMember, unique_keys: Optional[List[Any]]):
        """One member's batched key-match predicate (single ``= ANY`` or composite tuple-IN).

        Mirrors :meth:`PgSelectStep.match_predicate`: the single fast path keeps
        ``match = ANY(:keys)`` with ``keys`` bound at execute time; the composite path bakes
        the list-of-tuples onto the IN bindparam at build time so the RawExecutor
        postcompile path can expand it. The bind name ``keys`` is shared across all branches
        (they all match the SAME key set in one statement), so it binds once at execute time.
        """
        cols = member.match_columns
        if len(cols) == 1:
            return column(cols[0]) == any_(bindparam("keys", expanding=False))
        return tuple_(*[column(c) for c in cols]).in_(
            bindparam("keys", value=unique_keys or [], expanding=True)
        )

    def branch_select(
        self,
        member: PgUnionMember,
        unique_keys: Optional[List[Any]],
        after_values: Optional[List[Any]] = None,
        before_values: Optional[List[Any]] = None,
    ):
        """Build ONE member's ``SELECT`` leg of the union (shared + NULL-padded + tag).

        Projects the shared columns by name, the union-wide member columns (its own where
        present else ``NULL AS <col>``), and ``literal(type_name) AS __typename`` — so every
        leg has the identical column list and tags its concrete type. The keyset cursor
        predicate (built from this request's resolved seek values) and the per-parent key
        match (per-parent mode) AND onto this branch's WHERE alongside the member's own
        per-branch predicates.
        """
        own = set(member.resource.columns)
        shared_proj: List[ColumnElement] = [column(c) for c in self.shared_columns]
        member_proj: List[ColumnElement] = [
            column(c) if c in own else null().label(c) for c in self.member_columns
        ]
        tbl = table(
            member.resource.table,
            *[column(c) for c in member.resource.columns],
            schema=member.resource.schema,
        )
        stmt = select(
            *shared_proj,
            *member_proj,
            literal(member.type_name).label(TYPENAME_COLUMN),
        ).select_from(tbl)
        if self.is_per_parent:
            stmt = stmt.where(self.match_predicate(member, unique_keys))
        cursor = self.cursor_predicate(member, after_values, before_values)
        if cursor is not None:
            stmt = stmt.where(cursor)
        for predicate in member.where:
            stmt = stmt.where(predicate)
        return stmt

    def union_subquery(
        self,
        unique_keys: Optional[List[Any]],
        after_values: Optional[List[Any]] = None,
        before_values: Optional[List[Any]] = None,
    ):
        """The ``UNION ALL`` of every member's leg, as a subquery to window/order over."""
        legs = [
            self.branch_select(m, unique_keys, after_values, before_values)
            for m in self.members
        ]
        return union_all(*legs).subquery()

    def projected_columns(self) -> List[str]:
        """The full union row column list (shared + member-specific + the typename tag)."""
        return list(self.shared_columns) + list(self.member_columns) + [TYPENAME_COLUMN]

    def build_page_query(
        self,
        unique_keys: Optional[List[Any]] = None,
        source_values: Mapping[str, Any] = {},
        after_values: Any = _UNSET,
        before_values: Any = _UNSET,
    ):
        """Build the batched keyset-sliced PAGE SELECT over the union (ONE statement).

        Per-parent: numbers each parent's union rows ``row_number() OVER (PARTITION BY
        match ORDER BY <effective order>) AS __rn`` and keeps ``__rn <= :page_limit``
        (page size + 1 extra) per partition, ordered by match then ``__rn`` so each
        parent's page is contiguous and in window order. Root: a plain ``ORDER BY <order>``
        over the union with a plain ``LIMIT :page_limit`` (one synthetic bucket entry — no
        partition to corrupt). The keyset cursor predicate lives on each branch's WHERE
        (see :meth:`branch_select`), built from this request's resolved seek values, so the
        slice is fully IN SQL.

        ``after_values`` / ``before_values`` are this request's decoded seek values; when
        omitted (a direct build call) they default to the step's plan-time-decoded LITERAL
        values, so a literal cursor still seeks.
        """
        if after_values is _UNSET:
            after_values = self.after_values
        if before_values is _UNSET:
            before_values = self.before_values
        union = self.union_subquery(unique_keys, after_values, before_values)
        projected = self.projected_columns()
        order = order_clauses_over(union, self.effective_order())

        if not self.is_per_parent:
            stmt = select(*[union.c[c] for c in projected]).order_by(*order)
            if self.page_limit(source_values) is not None:
                stmt = stmt.limit(bindparam("page_limit"))
            return stmt

        match_cols = [union.c[c] for c in self.match_columns]
        rn = (
            func.row_number()
            .over(partition_by=match_cols, order_by=order)
            .label("__rn")
        )
        inner = select(union, rn).subquery()
        outer_cols = [inner.c[c] for c in projected]
        stmt = select(*outer_cols).order_by(
            *[inner.c[c] for c in self.match_columns], inner.c["__rn"]
        )
        if self.page_limit(source_values) is not None:
            stmt = stmt.where(inner.c["__rn"] <= bindparam("page_limit"))
        return stmt

    def build_count_query(self, unique_keys: Optional[List[Any]] = None):
        """Build the SEPARATE batched total over the union (WITHOUT the cursor predicate).

        Per-parent: ``match, count(*) GROUP BY match`` over the union of the members'
        key-matched legs (each under its per-branch WHERE), so the total is each parent's
        whole count even on an empty terminal page. Root: a single ``count(*)`` over the
        whole union. The cursor predicate is OMITTED (the count covers the full set), so a
        dedicated NO-cursor union is built for the count leg.
        """
        legs = [self.count_branch_select(m, unique_keys) for m in self.members]
        union = union_all(*legs).subquery()
        if not self.is_per_parent:
            return select(func.count().label("__total")).select_from(union)
        match_cols = [union.c[c] for c in self.match_columns]
        return (
            select(*match_cols, func.count().label("__total"))
            .select_from(union)
            .group_by(*match_cols)
        )

    def count_branch_select(self, member: PgUnionMember, unique_keys: Optional[List[Any]]):
        """One member's leg for the COUNT union: key columns only, no cursor predicate.

        Projects only the match columns (per-parent — so the outer can GROUP BY them) under
        the member's key match AND per-branch WHERE, but WITHOUT the keyset cursor predicate
        so the count covers each parent's FULL set. Root mode projects a constant so the
        legs stay union-compatible with nothing to group.
        """
        tbl = table(
            member.resource.table,
            *[column(c) for c in member.resource.columns],
            schema=member.resource.schema,
        )
        if self.is_per_parent:
            proj: List[ColumnElement] = [column(c) for c in self.match_columns]
        else:
            # root: no key to project; a single literal keeps every leg one-column and
            # union-compatible, and the outer just count(*)s the rows.
            proj = [literal(1).label("__one")]
        stmt = select(*proj).select_from(tbl)
        if self.is_per_parent:
            stmt = stmt.where(self.match_predicate(member, unique_keys))
        for predicate in member.where:
            stmt = stmt.where(predicate)
        return stmt

    # ------------------------------------------------------------------- execute

    def count_row_key(self, row: Dict[str, Any]) -> Any:
        """The per-parent total's lookup key from a count row: scalar or column tuple."""
        return grouping_key(row, self.match_columns)

    async def run_queries(
        self, unique_keys: Optional[List[Any]], source_values: Mapping[str, Any] = {}
    ) -> Tuple[List[Dict[str, Any]], Dict[Any, int]]:
        """Run the page query (always) and the count (iff ``needs_total``).

        Returns the page rows and a per-parent ``match -> total`` map (empty when
        ``needs_total`` is unset; a single ``None`` key in root mode). The ``keys`` param
        binds the shared single-column match list at execute time; a composite match bakes
        its IN list at build time so it passes no ``keys`` param. ``source_values`` resolves
        this request's variable-derived cursors (decoded here, never on the shared step), page
        size, and per-member WHERE placeholder values, injected into the params so a value-LESS
        placeholder bind executes with this request's value; the count leg AND-s the same
        member predicates, so it carries the same member WHERE params.
        """
        request = current_pg_request()
        after_values, before_values = self.resolve_cursor_values(source_values)
        member_params = self.member_where_params(source_values)
        params: Dict[str, Any] = dict(member_params)
        if self.is_per_parent and not self.is_composite:
            params["keys"] = unique_keys
        if self.page_limit(source_values) is not None:
            params["page_limit"] = self.page_limit(source_values)
        rows = await request.executor.run(
            self.build_page_query(
                unique_keys, source_values, after_values, before_values
            ),
            params,
            settings=request.settings,
        )

        totals: Dict[Any, int] = {}
        if self.needs_total:
            count_params: Dict[str, Any] = dict(member_params)
            if self.is_per_parent and not self.is_composite:
                count_params["keys"] = unique_keys
            count_rows = await request.executor.run(
                self.build_count_query(unique_keys),
                count_params,
                settings=request.settings,
            )
            if self.is_per_parent:
                totals = {self.count_row_key(r): r["__total"] for r in count_rows}
            else:
                # root: a single count(*) row keyed under the synthetic None bucket key.
                totals = {None: count_rows[0]["__total"] if count_rows else 0}
        return rows, totals

    def decode_member_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Codec-decode one union row through the resource of its concrete type.

        The row's ``__typename`` tag names the member it came from; that member's resource
        decodes the row (its own columns get their codec ``to_py``; the NULL-padded
        off-member columns and the typename pass through). So a member with a codec'd column
        surfaces the decoded value just like a plain select of that table.
        """
        member = self.member_by_type.get(row[TYPENAME_COLUMN])
        if member is None:
            # the tag came from one of our own literal projections, so this cannot happen
            # unless the union shape was corrupted — fail loud rather than scatter a row
            # with no concrete type (a silent skip would drop it from the page).
            raise KeyError(
                f"union row carries unknown __typename {row[TYPENAME_COLUMN]!r}; "
                f"members tag {sorted(self.member_by_type)}"
            )
        return member.resource.decode_row(row)

    def build_connection(
        self,
        rows: List[Dict[str, Any]],
        total: int,
        source_values: Mapping[str, Any] = {},
        after_values: Optional[List[Any]] = None,
        before_values: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Assemble one parent's page rows + total into a Relay connection dict.

        ``rows`` are this parent's union page rows in window order (already keyset-filtered
        + limited in SQL). With the one-extra probe present (``len(rows) > size``) the extra
        row is dropped and the "more" flag set; reverse paging re-reverses the page to the
        requested order. Each node is the codec-decoded tagged row (carrying ``__typename``),
        so completion-time dispatch resolves its concrete type. The cursor encodes from the
        RAW row's order-column values, so encode before decode.

        ``source_values`` resolves the variable-derived page size for the one-extra-row probe;
        ``after_values`` / ``before_values`` (this request's decoded seek values) drive the
        ``hasNextPage``/``hasPreviousPage`` presence flags — both per request.
        """
        # the active page size as a runtime int (resolving a variable-derived Placeholder).
        size = self.page_size(source_values)
        has_extra = size is not None and len(rows) > size
        page = rows[:size] if has_extra else rows
        if self.reverse:
            page = list(reversed(page))

        node_keys = self.projected_columns()
        edges = []
        for row in page:
            cursor = encode_keyset_cursor(self.order_by, row)
            decoded = self.decode_member_row(row)
            node = {c: decoded[c] for c in node_keys}
            edges.append({"node": node, "cursor": cursor})
        nodes = [e["node"] for e in edges]

        if self.reverse:
            has_previous = has_extra
            has_next = before_values is not None
        else:
            has_next = has_extra
            has_previous = after_values is not None

        return {
            "edges": edges,
            "nodes": nodes,
            "totalCount": total,
            "pageInfo": {
                "hasNextPage": has_next,
                "hasPreviousPage": has_previous,
                "startCursor": edges[0]["cursor"] if edges else None,
                "endCursor": edges[-1]["cursor"] if edges else None,
            },
        }

    @property
    def member_by_type(self) -> Dict[str, PgUnionMember]:
        """``type_name -> member`` map for decoding a tagged row through its resource."""
        return {m.type_name: m for m in self.members}

    def execute(self, count: int, values: List[List[Any]], extra=None) -> List[Any]:
        # `extra` is always supplied in production; the default covers a direct execute() call
        # in a unit test with no placeholders.
        source_values = extra.source_values if extra is not None else {}
        if not self.is_per_parent:
            # ROOT: one statement for the whole union; the SAME connection is returned for
            # every bucket entry (a root list has one logical result, sized to the bucket).
            async def run_root():
                rows, totals = await self.run_queries(None, source_values)
                after_values, before_values = self.resolve_cursor_values(source_values)
                conn = self.build_connection(
                    rows, totals.get(None, 0),
                    source_values=source_values,
                    after_values=after_values, before_values=before_values,
                )
                return [dict(conn) for _ in range(count)]

            return run_root()

        composite = self.is_composite
        keys = [normalize_lookup_key(k, composite) for k in values[0]]
        unique_keys = [k for k in dict.fromkeys(keys) if k is not None]
        if not unique_keys:
            after_values, before_values = self.resolve_cursor_values(source_values)
            empty = self.build_connection(
                [], 0, source_values=source_values,
                after_values=after_values, before_values=before_values,
            )
            return [dict(empty) for _ in range(count)]

        async def run():
            rows, totals = await self.run_queries(unique_keys, source_values)
            after_values, before_values = self.resolve_cursor_values(source_values)
            by_key: Dict[Any, List[Dict[str, Any]]] = {}
            for row in rows:
                by_key.setdefault(grouping_key(row, self.match_columns), []).append(row)
            return [
                self.build_connection(
                    by_key.get(keys[i], []), totals.get(keys[i], 0),
                    source_values=source_values,
                    after_values=after_values, before_values=before_values,
                )
                for i in range(count)
            ]

        return run()

    # ------------------------------------------------------------------- dedup

    def member_signature(self) -> Tuple[Tuple[str, str, Tuple[str, ...], Tuple[str, ...]], ...]:
        """The hashable, value-included dedup signature of the member set.

        Per member: ``(qualified_table, type_name, match_columns, where_signature)`` — so
        two unions differing in WHICH tables, their tagged type names, the per-member match
        key, or a per-branch filter VALUE (the value-included :func:`predicate_key`) get
        DIFFERENT signatures and never merge; identical member sets (in the same order)
        match. Member ORDER matters (it is the UNION ALL leg order, which is the SQL), so the
        tuple preserves it.
        """
        return tuple(
            (
                m.resource.qualified_table,
                m.type_name,
                m.match_columns,
                m.where_signature(),
            )
            for m in self.members
        )

    def cursor_key(self, source: Optional[str], values: Optional[List[Any]]) -> Any:
        """The dedup-key component for a cursor: its SOURCE tag (variable) or its VALUES (literal).

        A variable-derived cursor keys off its stable ``source`` tag (``var:after``) — value-
        agnostic, so two requests of the same document share one key (a cache hit) while two
        different sources never merge. A literal cursor keeps its decoded VALUES (list ->
        tuple, hashable) so two pages differing only by a literal ``after``/``before`` get
        different keys, as before. ``None`` (unpaged on this side) keys as ``None``; the source
        path is tagged (``("var", source)``) so it never collides with a same-shaped tuple.
        """
        if source is not None:
            return ("var", source)
        return tuple(values) if values is not None else None

    @property
    def peer_key(self) -> str:
        # first/last render their Placeholder source (not the value) when variable-derived;
        # after/before key by source tag (variable) or decoded values (literal) via cursor_key.
        return (
            f"pg_union_all|{self.member_signature()!r}"
            f"|{self.shared_columns!r}|{self.member_columns!r}"
            f"|{self.match_columns!r}|{self.is_per_parent}"
            f"|{self.order_by!r}|{self.first}|{self.last}"
            f"|{self.cursor_key(self.after_source, self.after_values)!r}"
            f"|{self.cursor_key(self.before_source, self.before_values)!r}|{self.needs_total}"
        )

    def dedup_params(self) -> Tuple[Any, ...]:
        return (
            self.member_signature(),
            self.shared_columns,
            self.member_columns,
            self.match_columns,
            self.is_per_parent,
            self.order_by,
            self.first,
            self.last,
            # the cursor key component: a variable cursor's stable source tag (value-agnostic,
            # a cache hit across requests of the same document) or a literal cursor's decoded
            # VALUES (so two pages differing only by after/before never merge).
            self.cursor_key(self.after_source, self.after_values),
            self.cursor_key(self.before_source, self.before_values),
            self.needs_total,
        )

    def get(self, attr: Any) -> Step:
        """Project a connection sub-field (``totalCount`` / ``pageInfo`` / ``nodes`` / ...)."""
        return access(self, (attr,))


def order_clauses_over(subquery: Any, terms: Sequence[OrderTerm]) -> list:
    """Emit ORDER BY elements bound to a SUBQUERY's columns (direction + NULLS aware).

    :func:`grafast_py.pg.ordering.order_clauses` emits bare ``column(name)`` references; over
    a union subquery the order must reference the subquery's OWN columns (``subquery.c[name]``)
    so it binds to the union's projected shape, not an ambiguous outer name. Mirrors
    ``order_clauses``' direction/NULLS handling against the resolved column.
    """
    clauses = []
    for term in terms:
        clause = subquery.c[term.column]
        if term.descending:
            clause = clause.desc()
        if term.nulls == "first":
            clause = clause.nulls_first()
        elif term.nulls == "last":
            clause = clause.nulls_last()
        clauses.append(clause)
    return clauses


def _reverse_nulls(term: OrderTerm) -> Optional[str]:
    """Flip a term's EFFECTIVE nulls placement for the reversed (last/before) order.

    Reversing the order reverses NULL placement too; emit it EXPLICITLY so the reversed
    ORDER BY and the BEFORE keyset comparator agree on where NULLs sit. Same rule as
    connection._reverse_nulls / cursor._flip_nulls (kept local: a private one-liner is not
    worth a cross-module import).
    """
    return "last" if effective_nulls(term) == "first" else "first"


def pg_union_all(
    members: Sequence[PgUnionMember],
    shared_columns: Sequence[str],
    order_by: Sequence[Union[str, OrderTerm]],
    key_step: Optional[Step] = None,
    first: Optional[Union[int, Placeholder]] = None,
    after: Optional[Union[str, Placeholder]] = None,
    last: Optional[Union[int, Placeholder]] = None,
    before: Optional[Union[str, Placeholder]] = None,
    order_is_unique: bool = False,
    needs_total: bool = False,
) -> PgUnionAllStep:
    """Plan-helper: a batched, keyset-paged Relay connection over N member tables.

    ``members`` are the ``UNION ALL`` legs (each a :class:`PgUnionMember` tagging its
    concrete type); ``shared_columns`` are the columns present in every member (the order
    columns must be among them). Pass ``key_step`` for PER-PARENT mode (each member then
    declares a ``match``) or omit it for the ROOT collection. ``needs_total`` (selection-gated
    via :func:`grafast_py.pg.connection.connection_needs_total`) requests the separate count.
    """
    return PgUnionAllStep(
        members,
        shared_columns,
        order_by,
        key_step=key_step,
        first=first,
        after=after,
        last=last,
        before=before,
        order_is_unique=order_is_unique,
        needs_total=needs_total,
    )


def union_all_connection(
    members: Sequence[PgUnionMember],
    shared_columns: Sequence[str],
    order_by: Sequence[Union[str, OrderTerm]],
    info: Any,
    key_step: Optional[Step] = None,
    first: Optional[Union[int, Placeholder]] = None,
    after: Optional[Union[str, Placeholder]] = None,
    last: Optional[Union[int, Placeholder]] = None,
    before: Optional[Union[str, Placeholder]] = None,
    order_is_unique: bool = False,
) -> PgUnionAllStep:
    """Plan-helper that gates ``needs_total`` off the field's selection set (like a connection).

    The selection-set-aware sibling of :func:`pg_union_all`: it reads ``info`` for a
    ``totalCount`` selection (:func:`grafast_py.pg.connection.connection_needs_total`) so the
    separate count is issued ONLY when the field asks for it — the same gating a plain
    connection plan resolver does. Pass ``key_step`` for per-parent mode.
    """
    return pg_union_all(
        members,
        shared_columns,
        order_by,
        key_step=key_step,
        first=first,
        after=after,
        last=last,
        before=before,
        order_is_unique=order_is_unique,
        needs_total=connection_needs_total(info),
    )


__all__ = [
    "PgUnionMember",
    "PgUnionAllStep",
    "TYPENAME_COLUMN",
    "pg_union_all",
    "union_all_connection",
]
