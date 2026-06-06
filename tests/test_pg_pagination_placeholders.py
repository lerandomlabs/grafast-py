"""Value-agnostic pagination placeholders: the value-LESS ``Placeholder`` sentinel for
``first`` / ``offset`` / ``after`` / ``before`` when variable-derived (Wave 4 / P5).

A WHERE value is a Core ``ColumnElement`` the host wraps as a :func:`pg_placeholder`
bindparam (see ``test_pg_placeholders``); a PAGINATION value is a plain Python scalar (an
``int`` ``first`` / ``offset``) or a cursor ``str`` (``after`` / ``before``) stored DIRECTLY
on the step. The step binds these as PARAMETERS at execute time (``window_slice`` binds
``first`` / ``offset``; the keyset comparator binds the decoded cursor values), so the SQL is
ALREADY value-agnostic — the ONLY thing that carries the literal is the DEDUP KEY.

The :class:`Placeholder` sentinel makes that key value-AGNOSTIC for a variable-derived value.
Under the deepcopy-free cache (P5) the sentinel itself is value-LESS — it lives on the SHARED
cached step and carries no request value; the runtime value is resolved per request from the
``BucketExtra.source_values`` map at SQL-render time (:func:`resolve_placeholder`) and injected
into the compiled statement's ``params``, never bound onto the shared step. Its ``__eq__`` /
``__hash__`` / ``__repr__`` key ONLY off the stable source tag (the variable name). This module
gates the CRUX — the four placeholder dedup behaviours, both directions through
``dag.Plan.deduplicate()`` with NO DB — for EACH pagination surface (select first/offset,
connection first + after-cursor, union first + after-cursor, root-collection LIMIT/OFFSET):

  1. two steps over the SAME source MERGE;
  2. two steps over DIFFERENT sources do NOT merge;
  3. a placeholder step and a literal step of a COINCIDENTALLY equal value do NOT merge;
  4. an existing literal-only step keeps its value-included key UNCHANGED (no regression).

Plus the NO-REGRESSION oracle (a placeholder page returns byte-identical rows + statement
count to the inlined literal) and the ANTI-CORRUPTION gate (the same value-agnostic statement
run with TWO different variable values, supplied per request via ``source_values``, returns
each its OWN correct rows — the property a shared/cached plan depends on).

DB tests are marked ``pg`` (deselectable ``-m 'not pg'``); they touch ONLY the
``grafast_demo`` ``posts`` fixture and alter nothing else.
"""

import pytest
import pytest_asyncio
from sqlalchemy import column

from grafast_py.core_steps import constant
from grafast_py.dag import Plan
from grafast_py.steps import BucketExtra
from grafast_py.pg.connection import PgConnectionStep
from grafast_py.pg.cursor import encode_keyset_cursor
from grafast_py.pg.engine import count_sql, dispose_engine, get_engine
from grafast_py.pg.executor import SQLAlchemyExecutor, pg_request_context
from grafast_py.pg.ordering import normalize_order
from grafast_py.pg.placeholders import (
    Placeholder,
    pg_placeholder,
    placeholder_source_tag,
    resolve_placeholder,
)
from grafast_py.pg.resource import PgRegistry, PgResource
from grafast_py.pg.steps import PgSelectAllStep, PgSelectStep
from grafast_py.pg.union import PgUnionAllStep, PgUnionMember
from examples.seed import setup_demo_schema


def make_posts() -> PgResource:
    """A fresh ``posts`` resource (its own registry) for the pagination placeholder tests."""
    registry = PgRegistry()
    return PgResource(
        "posts", "grafast_demo", "posts", ["id", "author_id", "title"], registry=registry
    )


def make_comments() -> PgResource:
    """A fresh ``comments`` resource — the second union member for the union tests."""
    registry = PgRegistry()
    return PgResource(
        "comments",
        "grafast_demo",
        "comments",
        ["id", "post_id", "author_id", "body"],
        registry=registry,
    )


def cursor_for(resource: PgResource, id_value: int) -> str:
    """A keyset cursor over ``order_by=['id']`` pinned to row ``id_value``."""
    order = normalize_order(["id"], primary_key=resource.primary_key)
    return encode_keyset_cursor(order, {"id": id_value})


@pytest_asyncio.fixture
async def seeded():
    """(Re)seed ``grafast_demo`` (fresh engine per test)."""
    await dispose_engine()
    await setup_demo_schema()
    yield
    await dispose_engine()


def dedup_key(step):
    """The class + peer_key + dedup_params slice of the planner's structural key.

    A pagination value adds no dependency, so the placeholder discriminator lives entirely
    in ``peer_key`` / ``dedup_params`` (the dependency survivor ids are identical here).
    """
    return (type(step), step.peer_key, step.dedup_params())


def extra_with(source_values):
    """A minimal per-invocation ``BucketExtra`` carrying ``source_values`` for a step execute.

    The pg value-steps read only ``extra.source_values`` (to resolve their value-less
    placeholders into params at render); context / parent_paths are unused on this path.
    """
    return BucketExtra(context=None, parent_paths=[], source_values=source_values)


# ----------------------------------------------------- the Placeholder sentinel itself


def test_placeholder_keys_off_source_only():
    """A value-LESS ``Placeholder`` is equal/hashes by SOURCE tag only."""
    a = Placeholder("var:limit")
    b = Placeholder("var:limit")  # same source
    c = Placeholder("var:other")  # different source
    assert a == b and hash(a) == hash(b)  # same source -> equal/merge (crux 1)
    assert a != c  # different source -> distinct (crux 2)


def test_placeholder_never_equals_a_bare_literal():
    """A ``Placeholder`` is never ``==`` a bare scalar (crux 3 at the unit)."""
    ph = Placeholder("var:limit")
    assert ph != 5 and 5 != ph
    # so a placeholder page never merges with a coincidentally equal literal page.


def test_placeholder_repr_carries_source_not_value():
    """The repr (folded into the f-string peer_key) renders the SOURCE only.

    Two same-source placeholders' keys are then byte-identical and the runtime value can
    never leak into a plan-cache key (the sentinel carries no value).
    """
    ph = Placeholder("var:limit")
    assert repr(ph) == "Placeholder('var:limit')"


def test_resolve_and_source_tag_helpers():
    """``resolve_placeholder`` yields the request value from the source map; tag the source.

    A value-less ``Placeholder`` resolves against ``source_values`` (None when absent); a bare
    literal passes through unchanged and reports ``None`` source — the value-included key path.
    """
    ph = Placeholder("var:n")
    assert resolve_placeholder(ph, {"var:n": 7}) == 7
    assert resolve_placeholder(ph, {}) is None  # omitted no-default variable -> None
    assert resolve_placeholder(7, {}) == 7
    assert placeholder_source_tag(ph) == "var:n"
    assert placeholder_source_tag(7) is None


# ----------------------------------------------------- select first/offset: the four crux behaviours


def test_select_first_same_source_merges():
    """CRUX 1: two selects whose ``first`` is the SAME source MERGE through ``deduplicate()``.

    The value-less ``Placeholder`` keys off its source, so the lowest-id step wins and both
    references resolve to it — the value-agnostic window slice is rendered per request.
    """
    a = PgSelectStep(make_posts(), constant(None), "author_id", order_by=["id"],
                     first=Placeholder("var:n"))
    b = PgSelectStep(make_posts(), constant(None), "author_id", order_by=["id"],
                     first=Placeholder("var:n"))
    assert dedup_key(a) == dedup_key(b)

    plan = Plan()
    plan.add_step(a)
    plan.add_step(b)
    remap = plan.deduplicate()
    assert remap[a.id] is remap[b.id]  # one survivor


def test_select_first_different_source_does_not_merge():
    """CRUX 2: two selects whose ``first`` is a DIFFERENT source do NOT merge."""
    a = PgSelectStep(make_posts(), constant(None), "author_id", order_by=["id"],
                     first=Placeholder("var:n"))
    b = PgSelectStep(make_posts(), constant(None), "author_id", order_by=["id"],
                     first=Placeholder("var:m"))
    assert dedup_key(a) != dedup_key(b)

    plan = Plan()
    plan.add_step(a)
    plan.add_step(b)
    remap = plan.deduplicate()
    assert remap[a.id] is a and remap[b.id] is b


def test_select_first_placeholder_never_merges_with_literal():
    """CRUX 3: a placeholder ``first`` never merges with a literal ``first`` of equal value.

    ``LIMIT :first`` (a value-agnostic window slice, source-keyed) vs an inlined literal
    page are different plans — re-rendering the placeholder would otherwise serve a literal-
    pinned plan a runtime size it never planned for.
    """
    ph = PgSelectStep(make_posts(), constant(None), "author_id", order_by=["id"],
                      first=Placeholder("var:n"))
    lit = PgSelectStep(make_posts(), constant(None), "author_id", order_by=["id"], first=2)
    assert dedup_key(ph) != dedup_key(lit)

    plan = Plan()
    plan.add_step(ph)
    plan.add_step(lit)
    remap = plan.deduplicate()
    assert remap[ph.id] is ph and remap[lit.id] is lit


def test_select_offset_placeholder_keys_by_source():
    """A placeholder ``offset`` keys by source too — and is always limited (value-agnostic).

    A variable-derived offset is value-agnostic, so the plan cannot know its value this
    request: it rides the window slice unconditionally and keys distinctly from a literal
    ``offset=0`` plain select (different SQL — paginated vs plain). Two same-source offsets
    merge regardless of value.
    """
    a = PgSelectStep(make_posts(), constant(None), "author_id", order_by=["id"],
                     offset=Placeholder("var:o"))
    b = PgSelectStep(make_posts(), constant(None), "author_id", order_by=["id"],
                     offset=Placeholder("var:o"))
    plain = PgSelectStep(make_posts(), constant(None), "author_id", order_by=["id"])
    assert a.is_limited is True  # a Placeholder offset is always limited (value-agnostic)
    assert dedup_key(a) == dedup_key(b)  # same source merges
    assert dedup_key(a) != dedup_key(plain)  # never merges with a literal offset=0 plain


def test_select_literal_first_offset_unchanged():
    """CRUX 4: literal-only ``first`` / ``offset`` keep their value-discriminated dedup.

    The regression gate from ``test_pg_pagination`` re-asserted under the placeholder-aware
    code: identical literals merge; different literal sizes do not.
    """
    a = PgSelectStep(make_posts(), constant(None), "author_id", order_by=["id"], first=2)
    b = PgSelectStep(make_posts(), constant(None), "author_id", order_by=["id"], first=2)
    c = PgSelectStep(make_posts(), constant(None), "author_id", order_by=["id"], first=3)
    assert dedup_key(a) == dedup_key(b)
    assert dedup_key(a) != dedup_key(c)

    plan = Plan()
    for step in (a, b, c):
        plan.add_step(step)
    remap = plan.deduplicate()
    assert remap[a.id] is remap[b.id]  # identical literals merge
    assert remap[c.id] is c  # the different literal stays its own survivor


# ----------------------------- root collection (PgSelectAllStep) LIMIT/OFFSET placeholder


def test_root_collection_placeholder_emits_value_agnostic_sql():
    """A placeholder root LIMIT/OFFSET emits a VALUE-LESS bound param; a literal inlines.

    So the cached root statement is value-agnostic and the runtime value lives only in the
    per-request ``run_params(source_values)`` — never inlined into a cacheable plan.
    """
    ph = PgSelectAllStep(make_posts(), order_by=["id"],
                         first=Placeholder("var:n"),
                         offset=Placeholder("var:o")).for_parent(constant(None))
    sql = str(ph.build_query())
    assert ":root_first" in sql and ":root_offset" in sql
    assert "5" not in sql and "3" not in sql  # the values are NOT inlined
    assert ph.run_params({"var:n": 5, "var:o": 3}) == {"root_first": 5, "root_offset": 3}


def test_root_collection_literal_sql_unchanged():
    """A literal root LIMIT/OFFSET inlines exactly as before — ``run_params`` is ``None``."""
    lit = PgSelectAllStep(make_posts(), order_by=["id"], first=5, offset=3).for_parent(
        constant(None)
    )
    sql = str(lit.build_query())
    # SQLAlchemy renders the inlined literal as an auto-named bound (value baked on), so
    # no explicit per-request params are needed — byte-identical to pre-Wave-4.
    assert ":root_first" not in sql and ":root_offset" not in sql
    assert lit.run_params() is None


def test_root_collection_placeholder_crux():
    """The four crux behaviours on the root collection's LIMIT/OFFSET, through deduplicate()."""
    a = PgSelectAllStep(make_posts(), order_by=["id"],
                        first=Placeholder("var:n")).for_parent(constant(None))
    b = PgSelectAllStep(make_posts(), order_by=["id"],
                        first=Placeholder("var:n")).for_parent(constant(None))
    c = PgSelectAllStep(make_posts(), order_by=["id"],
                        first=Placeholder("var:m")).for_parent(constant(None))
    lit = PgSelectAllStep(make_posts(), order_by=["id"], first=5).for_parent(constant(None))
    assert dedup_key(a) == dedup_key(b)  # crux 1: same source merges
    assert dedup_key(a) != dedup_key(c)  # crux 2: different source no-merge
    assert dedup_key(a) != dedup_key(lit)  # crux 3: placeholder vs literal no-merge

    plan = Plan()
    for step in (a, b, c, lit):
        plan.add_step(step)
    remap = plan.deduplicate()
    assert remap[a.id] is remap[b.id]
    assert remap[c.id] is c and remap[lit.id] is lit


# ------------------------- connection first + after-cursor: the four crux behaviours


def test_connection_first_placeholder_crux():
    """The four crux behaviours on a connection's ``first`` page size, through deduplicate()."""
    def conn(first):
        return PgConnectionStep(make_posts(), constant(None), "author_id",
                                order_by=["id"], first=first)
    a = conn(Placeholder("var:n"))
    b = conn(Placeholder("var:n"))
    c = conn(Placeholder("var:m"))
    lit = conn(2)
    assert dedup_key(a) == dedup_key(b)  # crux 1
    assert dedup_key(a) != dedup_key(c)  # crux 2
    assert dedup_key(a) != dedup_key(lit)  # crux 3

    plan = Plan()
    for step in (a, b, c, lit):
        plan.add_step(step)
    remap = plan.deduplicate()
    assert remap[a.id] is remap[b.id]
    assert remap[c.id] is c and remap[lit.id] is lit


def test_connection_after_cursor_placeholder_crux():
    """The four crux behaviours on a connection's ``after`` cursor, through deduplicate().

    A VARIABLE cursor is value-LESS on the shared step (it decodes per request at render); the
    dedup KEY carries only the stable source tag — so two requests of the same document share
    one plan while the value never enters the key, and each resolves ITS OWN seek values from
    the per-request source map.
    """
    posts = make_posts()
    cur_a = cursor_for(posts, 10)
    cur_b = cursor_for(posts, 20)

    def conn(after):
        return PgConnectionStep(make_posts(), constant(None), "author_id",
                                order_by=["id"], first=2, after=after)
    a = conn(Placeholder("var:after"))
    b = conn(Placeholder("var:after"))  # SAME source
    c = conn(Placeholder("var:other"))  # different source
    lit = conn(cur_a)  # literal cursor
    assert dedup_key(a) == dedup_key(b)  # crux 1: same source merges...
    # ...yet each request resolves ITS OWN decoded seek values from the source map for SQL.
    assert a.resolve_cursor_values({"var:after": cur_a})[0] != \
        a.resolve_cursor_values({"var:after": cur_b})[0]
    assert dedup_key(a) != dedup_key(c)  # crux 2: different source no-merge
    assert dedup_key(a) != dedup_key(lit)  # crux 3: placeholder cursor vs literal cursor
    # crux 4: the literal cursor keeps its value-included key (the decoded values)
    assert lit.cursor_key(lit.after_source, lit.after_values) == tuple(lit.after_values)

    plan = Plan()
    for step in (a, b, c, lit):
        plan.add_step(step)
    remap = plan.deduplicate()
    assert remap[a.id] is remap[b.id]
    assert remap[c.id] is c and remap[lit.id] is lit


def test_connection_literal_cursor_unchanged():
    """CRUX 4: literal ``after`` cursors keep their value-discriminated dedup (no regression).

    Identical literal cursors merge; different literal cursors do not — the pre-Wave-4
    behaviour, re-asserted under the placeholder-aware key path.
    """
    posts = make_posts()
    cur1 = cursor_for(posts, 10)
    cur2 = cursor_for(posts, 20)

    def conn(after):
        return PgConnectionStep(make_posts(), constant(None), "author_id",
                                order_by=["id"], first=2, after=after)
    a = conn(cur1)
    b = conn(cur1)
    c = conn(cur2)
    assert dedup_key(a) == dedup_key(b)
    assert dedup_key(a) != dedup_key(c)


# ----------------------------------- pgUnionAll first + after-cursor: the four crux


def union_members():
    """Two union members (posts + comments) sharing the ``id`` order column."""
    return [
        PgUnionMember(make_posts(), "Post", match="author_id"),
        PgUnionMember(make_comments(), "Comment", match="author_id"),
    ]


def test_union_first_placeholder_crux():
    """The four crux behaviours on a pgUnionAll ``first`` page size, through deduplicate()."""
    def union(first):
        return PgUnionAllStep(
            union_members(), shared_columns=["id", "author_id"], order_by=["id"],
            key_step=constant(None), first=first,
        )
    a = union(Placeholder("var:n"))
    b = union(Placeholder("var:n"))
    c = union(Placeholder("var:m"))
    lit = union(2)
    assert dedup_key(a) == dedup_key(b)  # crux 1
    assert dedup_key(a) != dedup_key(c)  # crux 2
    assert dedup_key(a) != dedup_key(lit)  # crux 3

    plan = Plan()
    for step in (a, b, c, lit):
        plan.add_step(step)
    remap = plan.deduplicate()
    assert remap[a.id] is remap[b.id]
    assert remap[c.id] is c and remap[lit.id] is lit


def test_union_after_cursor_placeholder_crux():
    """The four crux behaviours on a pgUnionAll ``after`` cursor, through deduplicate()."""
    posts = make_posts()
    # the union order is (id, <first member pk>, __typename); build the cursor through the
    # step so the digest matches.
    probe = PgUnionAllStep(
        union_members(), shared_columns=["id", "author_id"], order_by=["id"],
        key_step=constant(None), first=2,
    )
    row_a = {"id": 10, posts.primary_key: 10, "__typename": "Post"}
    row_b = {"id": 20, posts.primary_key: 20, "__typename": "Post"}
    cur_a = encode_keyset_cursor(probe.order_by, row_a)
    cur_b = encode_keyset_cursor(probe.order_by, row_b)

    def union(after):
        return PgUnionAllStep(
            union_members(), shared_columns=["id", "author_id"], order_by=["id"],
            key_step=constant(None), first=2, after=after,
        )
    a = union(Placeholder("var:after"))
    b = union(Placeholder("var:after"))
    c = union(Placeholder("var:other"))
    lit = union(cur_a)
    assert dedup_key(a) == dedup_key(b)  # crux 1: same source merges
    # ...yet each request resolves its own decoded seek values from the source map.
    assert a.resolve_cursor_values({"var:after": cur_a})[0] != \
        a.resolve_cursor_values({"var:after": cur_b})[0]
    assert dedup_key(a) != dedup_key(c)  # crux 2
    assert dedup_key(a) != dedup_key(lit)  # crux 3

    plan = Plan()
    for step in (a, b, c, lit):
        plan.add_step(step)
    remap = plan.deduplicate()
    assert remap[a.id] is remap[b.id]
    assert remap[c.id] is c and remap[lit.id] is lit


def test_union_renders_page_cursor_and_member_where_per_request():
    """RENDER-INJECTION (deepcopy-free): a pgUnionAll resolves page size, cursor AND member-WHERE
    placeholders from the per-request source map — never off the shared step.

    A cached pgUnionAll carries value-LESS variable-derived ``first`` (Placeholder), ``after``
    cursor (Placeholder) and per-member WHERE ``pg_placeholder`` binds. The deepcopy-free model
    resolves all three per request from ``source_values`` at render: the SHARED step holds no
    value, so two requests (here, two source maps) render distinct page sizes, distinct decoded
    cursor seek values, and distinct member-WHERE param values — the cross-request bleed the
    cache must never allow.
    """
    posts = make_posts()
    # build cursors through a probe step so the digest matches the union's (id, __typename) order.
    probe = PgUnionAllStep(
        union_members(), shared_columns=["id", "author_id"], order_by=["id"],
        key_step=constant(None), first=2,
    )
    cur_a = encode_keyset_cursor(probe.order_by, {"id": 3, posts.primary_key: 3, "__typename": "Post"})
    cur_b = encode_keyset_cursor(probe.order_by, {"id": 99, posts.primary_key: 99, "__typename": "Post"})

    def members_with_where():
        return [
            PgUnionMember(
                make_posts(), "Post", match="author_id",
                where=(column("title") == pg_placeholder("var:status", "published"),),
            ),
            PgUnionMember(make_comments(), "Comment", match="author_id"),
        ]

    step = PgUnionAllStep(
        members_with_where(), shared_columns=["id", "author_id"], order_by=["id"],
        key_step=constant(None),
        first=Placeholder("var:n"), after=Placeholder("var:after"),
    )
    # the SHARED step carries NO request value (value-less placeholders / no decoded cursor).
    assert step.after_values is None and step.before_values is None
    member_bind = step.members[0].where[0].right
    assert member_bind.value is None  # the WHERE placeholder bind is value-LESS

    # request A renders its OWN page size, decoded cursor and member-WHERE value.
    a_values = {"var:n": 2, "var:after": cur_a, "var:status": "published"}
    assert step.page_size(a_values) == 2
    assert step.resolve_cursor_values(a_values)[0] == decode_for(step, cur_a)
    assert step.member_where_params(a_values)[member_bind.key] == "published"

    # request B (the same SHARED step) renders DIFFERENT values — no bleed from A.
    b_values = {"var:n": 10, "var:after": cur_b, "var:status": "draft"}
    assert step.page_size(b_values) == 10
    assert step.resolve_cursor_values(b_values)[0] == decode_for(step, cur_b)
    assert step.member_where_params(b_values)[member_bind.key] == "draft"


def decode_for(step, cursor):
    """The decoded keyset values for ``cursor`` under ``step``'s order (for the render assertion)."""
    from grafast_py.pg.cursor import decode_keyset_cursor

    return decode_keyset_cursor(cursor, step.order_by)


# ----------------------------------------------- no-regression + anti-corruption DB oracle


@pytest.mark.pg
@pytest.mark.asyncio
async def test_placeholder_first_byte_identical_to_literal(seeded):
    """NO-REGRESSION: a placeholder ``first`` returns the SAME rows + SAME count as inlining.

    The old path passes a literal ``first=2``; the new path wraps it as a value-less
    ``Placeholder`` and supplies the value via ``source_values``. Same one windowed statement,
    same rows — the placeholder changes only the dedup key, never the executed page.
    """
    engine = get_engine()
    with pg_request_context(SQLAlchemyExecutor(engine)):
        literal = PgSelectStep(make_posts(), constant(None), "author_id",
                               order_by=["id"], first=2)
        with count_sql(engine) as lit_counter:
            lit_out = await literal.execute(3, [[1, 2, 3]], extra_with({}))

    with pg_request_context(SQLAlchemyExecutor(engine)):
        placeheld = PgSelectStep(make_posts(), constant(None), "author_id",
                                 order_by=["id"], first=Placeholder("var:n"))
        with count_sql(engine) as ph_counter:
            ph_out = await placeheld.execute(3, [[1, 2, 3]], extra_with({"var:n": 2}))

    assert ph_counter.count == lit_counter.count == 1
    assert ph_out == lit_out
    assert [[r["id"] for r in bucket] for bucket in ph_out] == [[1, 2], [3, 4], [6, 7]]
    # the value is NOT inlined in the placeholder statement — first/offset bind as params.
    sql = str(placeheld.build_query(source_values={"var:n": 2}))
    assert "__rn <= :offset + :first" in sql


@pytest.mark.pg
@pytest.mark.asyncio
async def test_same_placeholder_first_serves_two_sizes_correctly(seeded):
    """ANTI-CORRUPTION: one value-agnostic windowed statement, run with TWO page sizes.

    A placeholder-bearing select is value-agnostic SQL rendered per request from
    ``source_values``. Running the same SHARED step with ``first=1`` then ``first=3`` must
    return each its OWN correct page — the property a shared/cached plan depends on (a cache
    hit must never serve one request the size of another).
    """
    engine = get_engine()
    step = PgSelectStep(make_posts(), constant(None), "author_id",
                        order_by=["id"], first=Placeholder("var:n"))
    with pg_request_context(SQLAlchemyExecutor(engine)):
        one_out = await step.execute(3, [[1, 2, 3]], extra_with({"var:n": 1}))

    with pg_request_context(SQLAlchemyExecutor(engine)):
        three_out = await step.execute(3, [[1, 2, 3]], extra_with({"var:n": 3}))

    # the two requests share one value-agnostic statement SHAPE (the first/offset binds are
    # the same named params) but get distinct pages.
    assert str(step.build_query(source_values={"var:n": 1})) == \
        str(step.build_query(source_values={"var:n": 3}))
    assert [[r["id"] for r in b] for b in one_out] == [[1], [3], [6]]
    # author 1 owns posts 1,2 (capped at 2 by its row count); author 2 owns 3,4,5; author
    # 3 owns 6,7,8,9 (capped at 3 by first=3).
    assert [[r["id"] for r in b] for b in three_out] == [[1, 2], [3, 4, 5], [6, 7, 8]]


@pytest.mark.pg
@pytest.mark.asyncio
async def test_root_collection_placeholder_byte_identical_to_literal(seeded):
    """NO-REGRESSION: a placeholder root LIMIT/OFFSET returns the SAME rows + count as inlining.

    The root collection's plain LIMIT/OFFSET: the old path inlines ``first=2, offset=1``; the
    new path wraps each as a value-less ``Placeholder`` (bound as a value-less param, supplied
    via ``source_values``). Same one statement, same rows.
    """
    engine = get_engine()
    with pg_request_context(SQLAlchemyExecutor(engine)):
        literal = PgSelectAllStep(make_posts(), order_by=["id"], first=2, offset=1).for_parent(
            constant(None)
        )
        with count_sql(engine) as lit_counter:
            lit_out = await literal.execute(1, [[None]], extra_with({}))

    with pg_request_context(SQLAlchemyExecutor(engine)):
        placeheld = PgSelectAllStep(
            make_posts(), order_by=["id"],
            first=Placeholder("var:n"), offset=Placeholder("var:o"),
        ).for_parent(constant(None))
        with count_sql(engine) as ph_counter:
            ph_out = await placeheld.execute(1, [[None]], extra_with({"var:n": 2, "var:o": 1}))

    assert ph_counter.count == lit_counter.count == 1
    assert ph_out == lit_out
    # offset 1 skips post 1, first 2 -> posts 2,3.
    assert [r["id"] for r in ph_out[0]] == [2, 3]
