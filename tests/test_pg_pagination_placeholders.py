"""Value-agnostic pagination placeholders: the ``Placeholder`` sentinel for ``first`` /
``offset`` / ``after`` / ``before`` when variable-derived (Wave 4, step 6).

A WHERE value is a Core ``ColumnElement`` the host wraps as a :func:`pg_placeholder`
bindparam (see ``test_pg_placeholders``); a PAGINATION value is a plain Python scalar (an
``int`` ``first`` / ``offset``) or a cursor ``str`` (``after`` / ``before``) stored DIRECTLY
on the step. The step already binds these as PARAMETERS at execute time (``window_slice``
binds ``first`` / ``offset``; the keyset comparator binds the decoded cursor values), so the
SQL is ALREADY value-agnostic — the ONLY thing that carries the literal is the DEDUP KEY.

The :class:`Placeholder` sentinel makes that key value-AGNOSTIC for a variable-derived value:
it carries the request's value (so the step unwraps it at build/execute) but its ``__eq__`` /
``__hash__`` / ``__repr__`` key ONLY off the stable source tag (the variable name). This
module gates the CRUX — the four placeholder dedup behaviours, both directions through
``dag.Plan.deduplicate()`` with NO DB — for EACH pagination surface (select first/offset,
connection first + after-cursor, union first + after-cursor, root-collection LIMIT/OFFSET):

  1. two steps over the SAME source MERGE;
  2. two steps over DIFFERENT sources do NOT merge;
  3. a placeholder step and a literal step of a COINCIDENTALLY equal value do NOT merge;
  4. an existing literal-only step keeps its value-included key UNCHANGED (no regression).

Plus the NO-REGRESSION oracle (a placeholder page returns byte-identical rows + statement
count to the inlined literal) and the ANTI-CORRUPTION gate (the same value-agnostic statement
run with TWO different variable values returns each its OWN correct rows — the property a
shared/cached plan depends on).

DB tests are marked ``pg`` (deselectable ``-m 'not pg'``); they touch ONLY the
``grafast_demo`` ``posts`` fixture and alter nothing else.
"""

import pytest
import pytest_asyncio
from sqlalchemy import column

from grafast_py.core_steps import constant
from grafast_py.dag import Plan
from grafast_py.pg.connection import PgConnectionStep
from grafast_py.pg.cursor import encode_keyset_cursor
from grafast_py.pg.engine import count_sql, dispose_engine, get_engine
from grafast_py.pg.executor import SQLAlchemyExecutor, pg_request_context
from grafast_py.pg.ordering import normalize_order
from grafast_py.pg.placeholders import (
    Placeholder,
    pg_placeholder,
    placeholder_source_tag,
    unwrap_placeholder,
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


# ----------------------------------------------------- the Placeholder sentinel itself


def test_placeholder_keys_off_source_not_value():
    """A ``Placeholder`` is equal/hashes by SOURCE tag only — never by its runtime value."""
    a = Placeholder("var:limit", 5)
    b = Placeholder("var:limit", 999)  # same source, DIFFERENT value
    c = Placeholder("var:other", 5)  # different source, same value
    assert a == b and hash(a) == hash(b)  # same source -> equal/merge (crux 1)
    assert a != c  # different source -> distinct (crux 2)


def test_placeholder_never_equals_a_bare_literal():
    """A ``Placeholder`` is never ``==`` a bare scalar of the same value (crux 3 at the unit)."""
    ph = Placeholder("var:limit", 5)
    assert ph != 5 and 5 != ph
    # so a placeholder page never merges with a coincidentally equal literal page.


def test_placeholder_repr_carries_source_not_value():
    """The repr (folded into the f-string peer_key) renders the SOURCE, never the value.

    Two same-source placeholders' keys are then byte-identical and the runtime value can
    never leak into a plan-cache key.
    """
    ph = Placeholder("var:limit", 5)
    assert repr(ph) == "Placeholder('var:limit')"
    assert "5" not in repr(ph)


def test_unwrap_and_source_tag_helpers():
    """``unwrap_placeholder`` yields the runtime value; ``placeholder_source_tag`` the source.

    A bare literal passes through ``unwrap_placeholder`` and reports ``None`` source — so it
    stays on the value-included key path.
    """
    ph = Placeholder("var:n", 7)
    assert unwrap_placeholder(ph) == 7
    assert unwrap_placeholder(7) == 7
    assert placeholder_source_tag(ph) == "var:n"
    assert placeholder_source_tag(7) is None


# ----------------------------------------- select first/offset: the four crux behaviours


def test_select_first_same_source_merges():
    """CRUX 1: two selects whose ``first`` is the SAME source MERGE through ``deduplicate()``.

    Even with DIFFERENT runtime sizes (the value-agnostic window slice is re-bound per
    request), so the lowest-id step wins and both references resolve to it.
    """
    a = PgSelectStep(make_posts(), constant(None), "author_id", order_by=["id"],
                     first=Placeholder("var:n", 2))
    b = PgSelectStep(make_posts(), constant(None), "author_id", order_by=["id"],
                     first=Placeholder("var:n", 9))
    assert dedup_key(a) == dedup_key(b)

    plan = Plan()
    plan.add_step(a)
    plan.add_step(b)
    remap = plan.deduplicate()
    assert remap[a.id] is remap[b.id]  # one survivor
    # the value never enters the key
    assert "2" not in a.peer_key.split("Placeholder")[1][:20]


def test_select_first_different_source_does_not_merge():
    """CRUX 2: two selects whose ``first`` is a DIFFERENT source do NOT merge."""
    a = PgSelectStep(make_posts(), constant(None), "author_id", order_by=["id"],
                     first=Placeholder("var:n", 2))
    b = PgSelectStep(make_posts(), constant(None), "author_id", order_by=["id"],
                     first=Placeholder("var:m", 2))
    assert dedup_key(a) != dedup_key(b)

    plan = Plan()
    plan.add_step(a)
    plan.add_step(b)
    remap = plan.deduplicate()
    assert remap[a.id] is a and remap[b.id] is b


def test_select_first_placeholder_never_merges_with_literal():
    """CRUX 3: a placeholder ``first`` never merges with a literal ``first`` of equal value.

    ``LIMIT :first`` (a value-agnostic window slice, source-keyed) vs an inlined literal
    page are different plans — re-binding the placeholder would otherwise serve a literal-
    pinned plan a runtime size it never planned for.
    """
    ph = PgSelectStep(make_posts(), constant(None), "author_id", order_by=["id"],
                      first=Placeholder("var:n", 2))
    lit = PgSelectStep(make_posts(), constant(None), "author_id", order_by=["id"], first=2)
    assert dedup_key(ph) != dedup_key(lit)

    plan = Plan()
    plan.add_step(ph)
    plan.add_step(lit)
    remap = plan.deduplicate()
    assert remap[ph.id] is ph and remap[lit.id] is lit


def test_select_offset_placeholder_keys_by_source():
    """A placeholder ``offset`` keys by source too — even when its current value is 0.

    A variable-derived offset is value-agnostic, so the plan cannot know it is 0 this
    request: it rides the window slice unconditionally and keys distinctly from a literal
    ``offset=0`` plain select (different SQL — paginated vs plain). Two same-source offsets
    merge regardless of value.
    """
    a = PgSelectStep(make_posts(), constant(None), "author_id", order_by=["id"],
                     offset=Placeholder("var:o", 0))
    b = PgSelectStep(make_posts(), constant(None), "author_id", order_by=["id"],
                     offset=Placeholder("var:o", 5))
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
    per-request ``run_params`` — never inlined into a cacheable plan.
    """
    ph = PgSelectAllStep(make_posts(), order_by=["id"],
                         first=Placeholder("var:n", 5),
                         offset=Placeholder("var:o", 3)).for_parent(constant(None))
    sql = str(ph.build_query())
    assert ":root_first" in sql and ":root_offset" in sql
    assert "5" not in sql and "3" not in sql  # the values are NOT inlined
    assert ph.run_params() == {"root_first": 5, "root_offset": 3}


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
                        first=Placeholder("var:n", 5)).for_parent(constant(None))
    b = PgSelectAllStep(make_posts(), order_by=["id"],
                        first=Placeholder("var:n", 99)).for_parent(constant(None))
    c = PgSelectAllStep(make_posts(), order_by=["id"],
                        first=Placeholder("var:m", 5)).for_parent(constant(None))
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
    a = conn(Placeholder("var:n", 2))
    b = conn(Placeholder("var:n", 9))
    c = conn(Placeholder("var:m", 2))
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

    The decoded cursor VALUES still drive the keyset SQL (a and b decode different rows), but
    the dedup KEY carries only the stable source tag — so two requests of the same document
    share one plan while the value never enters the key.
    """
    posts = make_posts()
    cur_a = cursor_for(posts, 10)
    cur_b = cursor_for(posts, 20)

    def conn(after):
        return PgConnectionStep(make_posts(), constant(None), "author_id",
                                order_by=["id"], first=2, after=after)
    a = conn(Placeholder("var:after", cur_a))
    b = conn(Placeholder("var:after", cur_b))  # SAME source, DIFFERENT cursor value
    c = conn(Placeholder("var:other", cur_a))  # different source
    lit = conn(cur_a)  # literal cursor of the SAME value as a
    assert dedup_key(a) == dedup_key(b)  # crux 1: same source merges...
    assert a.after_values != b.after_values  # ...yet each decodes ITS OWN values for SQL
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
    a = union(Placeholder("var:n", 2))
    b = union(Placeholder("var:n", 9))
    c = union(Placeholder("var:m", 2))
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
    a = union(Placeholder("var:after", cur_a))
    b = union(Placeholder("var:after", cur_b))
    c = union(Placeholder("var:other", cur_a))
    lit = union(cur_a)
    assert dedup_key(a) == dedup_key(b)  # crux 1: same source merges
    assert a.after_values != b.after_values  # ...yet each decodes its own values
    assert dedup_key(a) != dedup_key(c)  # crux 2
    assert dedup_key(a) != dedup_key(lit)  # crux 3

    plan = Plan()
    for step in (a, b, c, lit):
        plan.add_step(step)
    remap = plan.deduplicate()
    assert remap[a.id] is remap[b.id]
    assert remap[c.id] is c and remap[lit.id] is lit


def test_union_rebind_repoints_page_cursor_and_member_where():
    """REGRESSION: a cache HIT re-binds a pgUnionAll's page size, cursor AND member-WHERE placeholders.

    A cached pgUnionAll carries the FIRST request's variable-derived ``first`` (Placeholder),
    ``after`` cursor (Placeholder) and per-member WHERE ``pg_placeholder`` value. Without a
    ``rebind_placeholders`` override the union would execute a LATER request with the FIRST
    request's values — the exact cross-request value bleed the cache must never allow. This
    drives the override directly: after rebinding to a second request's variables, the page
    size, the decoded cursor values AND the member-WHERE bound value all move to the new request.
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
        first=Placeholder("var:n", 2), after=Placeholder("var:after", cur_a),
    )
    # the FIRST request's values are bound on the cached step.
    assert step.first.value == 2
    assert step.after_values == decode_for(step, cur_a)
    assert step.members[0].where[0].right.value == "published"

    # a cache HIT for a SECOND request re-points every placeholder by its source tag.
    step.rebind_placeholders({"var:n": 10, "var:after": cur_b, "var:status": "draft"})
    assert step.first.value == 10
    assert step.after_values == decode_for(step, cur_b)
    assert step.members[0].where[0].right.value == "draft"


def decode_for(step, cursor):
    """The decoded keyset values for ``cursor`` under ``step``'s order (for the rebind assertion)."""
    from grafast_py.pg.cursor import decode_keyset_cursor

    return decode_keyset_cursor(cursor, step.order_by)


# ----------------------------------------------- no-regression + anti-corruption DB oracle


@pytest.mark.pg
@pytest.mark.asyncio
async def test_placeholder_first_byte_identical_to_literal(seeded):
    """NO-REGRESSION: a placeholder ``first`` returns the SAME rows + SAME count as inlining.

    The old path passes a literal ``first=2``; the new path wraps it as a ``Placeholder``.
    Same one windowed statement, same rows — the placeholder changes only the dedup key,
    never the executed page.
    """
    engine = get_engine()
    with pg_request_context(SQLAlchemyExecutor(engine)):
        literal = PgSelectStep(make_posts(), constant(None), "author_id",
                               order_by=["id"], first=2)
        with count_sql(engine) as lit_counter:
            lit_out = await literal.execute(3, [[1, 2, 3]])

    with pg_request_context(SQLAlchemyExecutor(engine)):
        placeheld = PgSelectStep(make_posts(), constant(None), "author_id",
                                 order_by=["id"], first=Placeholder("var:n", 2))
        with count_sql(engine) as ph_counter:
            ph_out = await placeheld.execute(3, [[1, 2, 3]])

    assert ph_counter.count == lit_counter.count == 1
    assert ph_out == lit_out
    assert [[r["id"] for r in bucket] for bucket in ph_out] == [[1, 2], [3, 4], [6, 7]]
    # the value is NOT inlined in the placeholder statement — first/offset bind as params.
    sql = str(placeheld.build_query())
    assert "__rn <= :offset + :first" in sql


@pytest.mark.pg
@pytest.mark.asyncio
async def test_same_placeholder_first_serves_two_sizes_correctly(seeded):
    """ANTI-CORRUPTION: one value-agnostic windowed statement, run with TWO page sizes.

    A placeholder-bearing select is value-agnostic SQL re-bound per request. Running the same
    statement shape with ``first=1`` then ``first=3`` must return each its OWN correct page —
    the property a shared/cached plan depends on (a cache hit must never serve one request the
    size of another).
    """
    engine = get_engine()
    with pg_request_context(SQLAlchemyExecutor(engine)):
        one = PgSelectStep(make_posts(), constant(None), "author_id",
                           order_by=["id"], first=Placeholder("var:n", 1))
        one_out = await one.execute(3, [[1, 2, 3]])

    with pg_request_context(SQLAlchemyExecutor(engine)):
        three = PgSelectStep(make_posts(), constant(None), "author_id",
                             order_by=["id"], first=Placeholder("var:n", 3))
        three_out = await three.execute(3, [[1, 2, 3]])

    # the two requests share one value-agnostic statement SHAPE (the first/offset binds are
    # the same named params) but get distinct pages.
    assert str(one.build_query()) == str(three.build_query())
    assert [[r["id"] for r in b] for b in one_out] == [[1], [3], [6]]
    # author 1 owns posts 1,2 (capped at 2 by its row count); author 2 owns 3,4,5; author
    # 3 owns 6,7,8,9 (capped at 3 by first=3).
    assert [[r["id"] for r in b] for b in three_out] == [[1, 2], [3, 4, 5], [6, 7, 8]]


@pytest.mark.pg
@pytest.mark.asyncio
async def test_root_collection_placeholder_byte_identical_to_literal(seeded):
    """NO-REGRESSION: a placeholder root LIMIT/OFFSET returns the SAME rows + count as inlining.

    The root collection's plain LIMIT/OFFSET: the old path inlines ``first=2, offset=1``; the
    new path wraps each as a ``Placeholder`` (bound as a value-less param). Same one statement,
    same rows.
    """
    engine = get_engine()
    with pg_request_context(SQLAlchemyExecutor(engine)):
        literal = PgSelectAllStep(make_posts(), order_by=["id"], first=2, offset=1).for_parent(
            constant(None)
        )
        with count_sql(engine) as lit_counter:
            lit_out = await literal.execute(1, [[None]])

    with pg_request_context(SQLAlchemyExecutor(engine)):
        placeheld = PgSelectAllStep(
            make_posts(), order_by=["id"],
            first=Placeholder("var:n", 2), offset=Placeholder("var:o", 1),
        ).for_parent(constant(None))
        with count_sql(engine) as ph_counter:
            ph_out = await placeheld.execute(1, [[None]])

    assert ph_counter.count == lit_counter.count == 1
    assert ph_out == lit_out
    # offset 1 skips post 1, first 2 -> posts 2,3.
    assert [r["id"] for r in ph_out[0]] == [2, 3]
