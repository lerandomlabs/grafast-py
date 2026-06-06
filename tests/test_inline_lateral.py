"""The correlated LATERAL build (Wave 3b, step 4): build_lateral + apply_laterals.

Step 3 added the fold DATA structures (:class:`InlineSpec` / :class:`NestedExtractStep`);
this step adds the SQL: a parent pg select's ``build_query`` LEFT JOINs one
``json_agg`` / ``to_jsonb`` LATERAL per carried :class:`InlineSpec`. There is no optimize
wiring yet (a default-built step carries no specs), so this is gated behind the presence of
specs — a parent with no specs emits BYTE-IDENTICAL SQL to the batched ``= ANY($1)`` path.

These tests pin three layers:

- the GOLDEN SQL of :func:`build_lateral` for hasOne and hasMany (the emitted correlated
  subquery string), and that ``build_query`` wraps it in ``LEFT OUTER JOIN LATERAL ... ON
  true`` so a child-less parent survives;
- the NO-OP invariant: an empty ``inline_specs`` leaves the statement (and the dedup key)
  exactly as the pre-Wave-3b batched build, while a present spec DISCRIMINATES the dedup key
  (two parents inlining different children never merge);
- DB-BACKED EQUIVALENCE (marked ``pg``): a manually-inlined parent + :class:`NestedExtractStep`
  scatters BYTE-IDENTICAL child rows to the standalone batched child step, in ONE FEWER
  statement (proving the LATERAL fires and is equivalence-preserving — the inlining invariant).

The non-DB golden/dedup tests run with ``-m 'not pg'``; the equivalence test is ``pg`` and
touches ONLY the ``grafast_demo`` schema of ``grafast_py_test``.
"""

import pytest
import pytest_asyncio
from sqlalchemy.dialects import postgresql

from grafast_py.core_steps import constant
from grafast_py.pg.inline import (
    KIND_HAS_MANY,
    KIND_HAS_ONE,
    InlineSpec,
    NestedExtractStep,
    build_lateral,
)
from grafast_py.pg.ordering import OrderTerm, normalize_order
from grafast_py.pg.resource import PgRegistry, PgResource
from grafast_py.pg.steps import PgSelectAllStep, PgSelectSingleStep, PgSelectStep


def render(stmt) -> str:
    """Compile a Core statement to its Postgres SQL string (whitespace-normalised)."""
    sql = str(stmt.compile(dialect=postgresql.dialect()))
    return " ".join(sql.split())


def make_blog_registry():
    """authors(id, name) <-hasMany-> posts(id, author_id, title) <-hasOne-> authors."""
    registry = PgRegistry()
    authors = PgResource(
        "authors", "grafast_demo", "authors", ["id", "name"], registry=registry
    )
    posts = PgResource(
        "posts", "grafast_demo", "posts", ["id", "author_id", "title"], registry=registry
    )
    return registry, authors, posts


def posts_many_spec(posts: PgResource) -> InlineSpec:
    """authors.posts hasMany fold: local author key ``id`` == remote FK ``author_id``."""
    return InlineSpec(
        resource=posts,
        kind=KIND_HAS_MANY,
        nested_alias="__posts",
        local_columns=("id",),
        remote_columns=("author_id",),
        order_by=normalize_order(["id"], primary_key="id"),
    )


def author_one_spec(authors: PgResource) -> InlineSpec:
    """posts.author hasOne fold: local FK ``author_id`` == remote key ``id``."""
    return InlineSpec(
        resource=authors,
        kind=KIND_HAS_ONE,
        nested_alias="__author",
        local_columns=("author_id",),
        remote_columns=("id",),
        order_by=normalize_order(None, primary_key="id"),
    )


def render_correlated_lateral(spec: InlineSpec, parent) -> str:
    """Render :func:`build_lateral`'s subquery JOINED to its parent (the real context).

    A LATERAL only correlates out the parent table when that table is in the enclosing
    FROM; rendered alone, SQLAlchemy would (correctly) auto-FROM the parent INTO the inner
    select. So the golden joins the lateral to the parent ``ON true`` — exactly how the
    parent's ``build_query`` uses it — to pin the correlated json subquery byte-for-byte.
    """
    from sqlalchemy import select, true

    lateral = build_lateral(spec, parent)
    return render(
        select(lateral.c[spec.nested_alias]).select_from(parent.outerjoin(lateral, true()))
    )


# --------------------------------------------------- golden build_lateral (direct)


def test_build_lateral_emits_has_many_json_agg_subquery():
    """The hasMany LATERAL body: coalesce(json_agg(to_jsonb(child) ORDER BY <order>), '[]')."""
    from sqlalchemy import column, table

    registry, authors, posts = make_blog_registry()
    parent = table("authors", column("id"), column("name"), schema="grafast_demo")
    body = render_correlated_lateral(posts_many_spec(posts), parent)
    assert body == (
        "SELECT __posts_lat.__posts "
        "FROM grafast_demo.authors LEFT OUTER JOIN LATERAL ("
        "SELECT coalesce(json_agg(to_jsonb(__posts_src) ORDER BY __posts_src.id), "
        "'[]'::json) AS __posts "
        "FROM (SELECT grafast_demo.posts.id AS id, "
        "grafast_demo.posts.author_id AS author_id, "
        "grafast_demo.posts.title AS title "
        "FROM grafast_demo.posts "
        "WHERE author_id = grafast_demo.authors.id) AS __posts_src"
        ") AS __posts_lat ON true"
    )


def test_build_lateral_emits_has_one_to_jsonb_subquery():
    """The hasOne LATERAL body: to_jsonb(child) over the inner select with LIMIT 1."""
    from sqlalchemy import column, table

    registry, authors, posts = make_blog_registry()
    parent = table(
        "posts", column("id"), column("author_id"), column("title"), schema="grafast_demo"
    )
    body = render_correlated_lateral(author_one_spec(authors), parent)
    assert body == (
        "SELECT __author_lat.__author "
        "FROM grafast_demo.posts LEFT OUTER JOIN LATERAL ("
        "SELECT to_jsonb(__author_src) AS __author "
        "FROM (SELECT grafast_demo.authors.id AS id, "
        "grafast_demo.authors.name AS name "
        "FROM grafast_demo.authors "
        "WHERE id = grafast_demo.posts.author_id "
        "ORDER BY id LIMIT %(param_1)s) AS __author_src"
        ") AS __author_lat ON true"
    )


# ------------------------------------------------------- golden build_query SQL


def test_build_lateral_has_many_golden_sql():
    """hasMany emits coalesce(json_agg(to_jsonb(child) ORDER BY <order>), '[]') correlated.

    The ordering lives INSIDE json_agg (a subquery ORDER BY does not survive aggregation),
    the correlation is ``child.remote = parent.local``, and the coalesce yields ``[]`` for a
    parent with no children.
    """
    registry, authors, posts = make_blog_registry()
    root = PgSelectAllStep(authors, order_by=["id"], inline_specs=[posts_many_spec(posts)])
    sql = render(root.build_query())
    assert sql == (
        "SELECT grafast_demo.authors.id, grafast_demo.authors.name, "
        "__posts_lat.__posts "
        "FROM grafast_demo.authors LEFT OUTER JOIN LATERAL ("
        "SELECT coalesce(json_agg(to_jsonb(__posts_src) ORDER BY __posts_src.id), "
        "'[]'::json) AS __posts "
        "FROM (SELECT grafast_demo.posts.id AS id, "
        "grafast_demo.posts.author_id AS author_id, "
        "grafast_demo.posts.title AS title "
        "FROM grafast_demo.posts "
        "WHERE author_id = grafast_demo.authors.id) AS __posts_src"
        ") AS __posts_lat ON true ORDER BY id"
    )


def test_build_lateral_has_one_golden_sql():
    """hasOne emits to_jsonb(child) over the inner select with LIMIT 1, correlated.

    A single related row (or NULL when the FK points nowhere) — the batched path's
    single-row / ``None`` scatter, reproduced inside the LATERAL.
    """
    registry, authors, posts = make_blog_registry()
    rel = PgSelectStep(
        posts, constant(None), "author_id",
        order_by=["id"], inline_specs=[author_one_spec(authors)],
    )
    sql = render(rel.build_query())
    assert sql == (
        "SELECT grafast_demo.posts.id, grafast_demo.posts.author_id, "
        "grafast_demo.posts.title, __author_lat.__author "
        "FROM grafast_demo.posts LEFT OUTER JOIN LATERAL ("
        "SELECT to_jsonb(__author_src) AS __author "
        "FROM (SELECT grafast_demo.authors.id AS id, "
        "grafast_demo.authors.name AS name "
        "FROM grafast_demo.authors "
        "WHERE id = grafast_demo.posts.author_id "
        "ORDER BY id LIMIT %(param_1)s) AS __author_src"
        ") AS __author_lat ON true "
        "WHERE author_id = ANY (%(keys)s) ORDER BY id"
    )


def test_build_lateral_has_one_on_single_step():
    """A PgSelectSingleStep (hasOne parent) folds a nested hasOne LATERAL the same way."""
    registry, authors, posts = make_blog_registry()
    single = PgSelectSingleStep(
        posts, constant(None), "id", inline_specs=[author_one_spec(authors)]
    )
    sql = render(single.build_query())
    assert "LEFT OUTER JOIN LATERAL" in sql
    assert "to_jsonb(__author_src) AS __author" in sql
    assert "WHERE id = grafast_demo.posts.author_id" in sql
    # the single step keeps its own key match alongside the fold.
    assert "WHERE id = ANY (%(keys)s)" in sql


def test_build_lateral_has_many_order_descending_with_pk_tiebreak():
    """A non-PK descending order is reproduced inside json_agg with the PK tie-break."""
    registry, authors, posts = make_blog_registry()
    spec = InlineSpec(
        resource=posts,
        kind=KIND_HAS_MANY,
        nested_alias="__posts",
        local_columns=("id",),
        remote_columns=("author_id",),
        order_by=normalize_order([OrderTerm("title", descending=True)], primary_key="id"),
    )
    sql = render(PgSelectAllStep(authors, inline_specs=[spec]).build_query())
    # the title DESC then the id tie-break, both bound to the inner subquery columns.
    assert "ORDER BY __posts_src.title DESC, __posts_src.id" in sql


def test_build_lateral_composite_correlation_ands_each_pair():
    """A composite FK correlates each matched pair AND-combined in the LATERAL WHERE."""
    registry = PgRegistry()
    parent_res = PgResource(
        "lines", "grafast_demo", "lines", ["org_id", "item_id"],
        primary_key="org_id", registry=registry,
    )
    child_res = PgResource(
        "prices", "grafast_demo", "prices", ["org_id", "item_id", "amount"],
        primary_key="org_id", registry=registry,
    )
    spec = InlineSpec(
        resource=child_res,
        kind=KIND_HAS_MANY,
        nested_alias="__prices",
        local_columns=("org_id", "item_id"),
        remote_columns=("org_id", "item_id"),
        order_by=normalize_order(["amount"], primary_key="org_id"),
    )
    sql = render(PgSelectAllStep(parent_res, inline_specs=[spec]).build_query())
    assert (
        "WHERE org_id = grafast_demo.lines.org_id "
        "AND item_id = grafast_demo.lines.item_id"
    ) in sql


# ----------------------------------------------------------- the no-op invariant


def test_empty_specs_is_byte_identical_to_batched_build():
    """A default-built parent (no specs) emits the EXACT pre-Wave-3b SQL — no LATERAL."""
    registry, authors, posts = make_blog_registry()
    plain = render(PgSelectAllStep(authors, order_by=["id"]).build_query())
    assert "LATERAL" not in plain
    assert plain == (
        "SELECT grafast_demo.authors.id, grafast_demo.authors.name "
        "FROM grafast_demo.authors ORDER BY id"
    )

    rel_plain = render(
        PgSelectStep(posts, constant(None), "author_id", order_by=["id"]).build_query()
    )
    assert "LATERAL" not in rel_plain
    assert rel_plain == (
        "SELECT grafast_demo.posts.id, grafast_demo.posts.author_id, "
        "grafast_demo.posts.title FROM grafast_demo.posts "
        "WHERE author_id = ANY (%(keys)s) ORDER BY id"
    )


def test_window_sliced_parent_rejects_specs():
    """A limited (window-sliced) parent cannot carry a fold — it fails loud, never silent.

    The safety predicate must not fold a child into a paginated parent (the LATERAL would
    have to attach to the post-slice outer query); a spec here is a wiring bug.
    """
    registry, authors, posts = make_blog_registry()
    step = PgSelectStep(
        posts, constant(None), "author_id",
        order_by=["id"], first=2, inline_specs=[author_one_spec(authors)],
    )
    with pytest.raises(AssertionError, match="window-sliced"):
        step.build_query()


# ------------------------------------------------------------ dedup discrimination


def test_specs_discriminate_the_dedup_key():
    """Two parents inlining different children emit different SQL, so they must not merge."""
    registry, authors, posts = make_blog_registry()
    no_fold = PgSelectAllStep(authors, order_by=["id"])
    with_fold = PgSelectAllStep(
        authors, order_by=["id"], inline_specs=[posts_many_spec(posts)]
    )
    assert no_fold.peer_key != with_fold.peer_key
    assert no_fold.dedup_params() != with_fold.dedup_params()


def test_no_specs_dedup_key_unchanged_from_baseline():
    """The empty-spec dedup key is byte-identical to a baseline build (no over-discrimination).

    inline_signature() of () is an empty tuple, so a default-built step's key is the same it
    was pre-Wave-3b — the no-op invariant extends to dedup.
    """
    registry, authors, posts = make_blog_registry()
    a = PgSelectAllStep(authors, order_by=["id"])
    b = PgSelectAllStep(authors, order_by=["id"])
    assert a.peer_key == b.peer_key
    assert a.dedup_params() == b.dedup_params()
    assert a.inline_signature() == ()


def test_different_fold_orders_discriminate():
    """The LATERALs join left-to-right, so a different fold order is a different statement."""
    registry, authors, posts = make_blog_registry()
    a_spec = author_one_spec(authors)
    # a second distinct fold (alias differs) so the order is observable.
    b_spec = InlineSpec(
        resource=authors,
        kind=KIND_HAS_ONE,
        nested_alias="__editor",
        local_columns=("author_id",),
        remote_columns=("id",),
        order_by=normalize_order(None, primary_key="id"),
    )
    ab = PgSelectStep(posts, constant(None), "author_id", inline_specs=[a_spec, b_spec])
    ba = PgSelectStep(posts, constant(None), "author_id", inline_specs=[b_spec, a_spec])
    assert ab.peer_key != ba.peer_key


# --------------------------------------------------- DB-backed equivalence (pg)


@pytest_asyncio.fixture
async def reseeded_engine():
    """(Re)seed grafast_demo and yield a fresh engine; dispose after."""
    from grafast_py.pg.engine import dispose_engine, get_engine
    from examples.seed import setup_demo_schema

    await dispose_engine()
    await setup_demo_schema()
    yield get_engine()
    await dispose_engine()


@pytest.mark.pg
@pytest.mark.asyncio
async def test_inlined_lateral_is_equivalent_to_batched_in_one_fewer_statement(
    reseeded_engine,
):
    """The inlined LATERAL scatters BYTE-IDENTICAL child rows to the batched path, in 1 stmt.

    Drives both paths over the SAME seed and asserts:
    - the batched child (standalone ``PgSelectStep`` keyed ``= ANY``) runs 2 statements
      (parent authors + child posts) and scatters each author's posts list;
    - the inlined parent (authors with a posts LATERAL) + ``NestedExtractStep`` runs 1
      statement and scatters the IDENTICAL lists — same rows, same order, same shape.
    This is the inlining invariant at the SQL level: fewer statements, identical data.
    """
    from grafast_py.pg.engine import count_sql
    from grafast_py.pg.executor import SQLAlchemyExecutor, pg_request_context

    registry, authors, posts = make_blog_registry()
    engine = reseeded_engine

    # --- batched baseline: authors (1) then posts WHERE author_id = ANY (1) = 2 statements.
    with count_sql(engine) as batched_counter:
        with pg_request_context(SQLAlchemyExecutor(engine)):
            root = PgSelectAllStep(authors, order_by=["id"])
            author_rows = (await root.execute(1, [[None]]))[0]
            author_ids = [r["id"] for r in author_rows]
            child = PgSelectStep(posts, constant(None), "author_id", order_by=["id"])
            batched_lists = await child.execute(len(author_ids), [author_ids])
    assert batched_counter.count == 2

    # --- inlined: authors with a posts LATERAL (1) + NestedExtractStep (no DB) = 1 statement.
    with count_sql(engine) as inlined_counter:
        with pg_request_context(SQLAlchemyExecutor(engine)):
            inlined_root = PgSelectAllStep(
                authors, order_by=["id"], inline_specs=[posts_many_spec(posts)]
            )
            # the parent statement now carries the LATERAL; its row dicts gain a __posts
            # json column. The NestedExtractStep reads that column off the parent row — no
            # DB work — and decodes/scatters per author, exactly like the batched child.
            parent_rows_per_entry = (await inlined_root.execute(1, [[None]]))[0]
            extract = NestedExtractStep(inlined_root, "__posts", posts, KIND_HAS_MANY)
            # the child bucket is seeded with one entry per parent row (what completion
            # does after the optimize pass repoints the bucket's parent_step).
            inlined_lists = extract.execute(
                len(parent_rows_per_entry), [parent_rows_per_entry]
            )
    assert inlined_counter.count == 1

    # the inlined parent's nested posts decode to the SAME per-author lists the batched
    # child scattered — byte-identical data, one fewer statement.
    assert inlined_lists == batched_lists
    # and the batched path actually returned non-trivial data (so the equality is meaningful).
    assert sum(len(group) for group in batched_lists) == 9  # 2 + 3 + 4 posts
