"""DB-backed equivalence battery for LATERAL relation inlining (Wave 3b, step 6).

The DEFINING invariant of inlining is that it is an OPPORTUNISTIC, EQUIVALENCE-PRESERVING
optimization: for ANY operation, the result with inlining ENABLED must be BYTE-IDENTICAL to
the result with it DISABLED (the batched ``= ANY($1)`` path, which is the correctness
baseline). Inlining changes only the NUMBER of SQL statements, never the data.

So every case here runs the SAME query TWICE — once with ``GrafastConfig(inline_relations=
False)`` (baseline) and once ``True`` (inlined) — and asserts:

(a) ``result.data`` and ``result.errors`` are BYTE-IDENTICAL between the two runs (deep
    equality on the JSON, including list order and null/[] shape); and
(b) the inlined run issues STRICTLY FEWER SQL statements (proving the fold actually FIRED
    and is not silently a no-op), by the expected amount.

The matrix covers the shapes the safety predicate either FOLDS (hasOne, unpaginated hasMany,
multi-level single-level nesting, explicit non-PK order, json-stable codec) or SKIPS but must
keep byte-identical (NULL/empty children, non-json-safe codec, composite key, filtered child,
paginated connection). A SKIP case asserts the count is UNCHANGED (the fallback fired) while
the data stays identical — unsafe-but-skipped is always correct.

Marked ``pg`` (DB-backed) and touches ONLY the ``grafast_demo`` schema of
``grafast_py_test`` via the idempotent, drop-first seed fixtures.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from graphql import graphql
from sqlalchemy import column, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import DateTime, Integer, Text

from grafast_py.config import GrafastConfig
from grafast_py.context import GrafastExecutionContext
from grafast_py.core_steps import access
from grafast_py.pg.engine import count_sql, dispose_engine, get_engine
from grafast_py.pg.executor import pg_request_context
from grafast_py.pg.resource import PgCodec, PgColumn, PgRegistry, PgResource
from grafast_py.schema import make_grafast_schema
from examples.demo_schema import build_demo_schema
from examples.seed import DEMO_SCHEMA, setup_demo_schema

pytestmark = pytest.mark.pg

# Native column types so the inlining safety predicate can PROVE each column json-stable
# native and fold it; a bare untyped column is UNKNOWN-typed and the predicate fails safe
# (see PgResource.is_inline_json_safe). These mirror the types the ORM bridge records, so the
# hand-declared battery resources fold exactly as a model-derived registry would.
AUTHORS_COLS = [PgColumn("id", sql_type=Integer()), PgColumn("name", sql_type=Text())]
POSTS_COLS = [
    PgColumn("id", sql_type=Integer()),
    PgColumn("author_id", sql_type=Integer()),
    PgColumn("title", sql_type=Text()),
]
COMMENTS_COLS = [
    PgColumn("id", sql_type=Integer()),
    PgColumn("post_id", sql_type=Integer()),
    PgColumn("author_id", sql_type=Integer()),
    PgColumn("body", sql_type=Text()),
]


def context_class(inline: bool):
    """A throwaway context subclass carrying ``GrafastConfig(inline_relations=inline)``."""

    class _Ctx(GrafastExecutionContext):
        grafast_config = GrafastConfig(inline_relations=inline)

    return _Ctx


class HeldConnExecutor:
    """A :class:`PgExecutor` that runs every statement on ONE held :class:`AsyncConnection`.

    The equivalence battery runs the SAME query twice (inline OFF then ON) and asserts
    byte-identical data. For that to be a valid oracle BOTH reads must observe the IDENTICAL
    committed DB state — but the default per-statement path opens a fresh pooled connection per
    statement, each with its own implicit snapshot, so a SIBLING test's committed INSERT
    (``Empty Author`` / a fixture row) landing between the two reads — visible across the
    process-global pool under function-scoped event loops — makes the second read see an extra
    row the first did not (the historical ~33%-in-isolation flake). Pinning both reads to ONE
    connection inside ONE ``REPEATABLE READ`` transaction gives them a SINGLE frozen snapshot,
    so no concurrently-committed sibling row can drift between them — the assertion compares
    inline-vs-batched on genuinely identical data. It is a pure read harness (rolled back on
    close); the product code is unchanged.
    """

    def __init__(self, conn) -> None:
        self.conn = conn

    async def run(self, statement, params, *, settings=None, commit=False):
        result = await self.conn.execute(statement, params or {})
        return [dict(row) for row in result.mappings().all()]


@asynccontextmanager
async def repeatable_read_snapshot(engine):
    """Open one ``REPEATABLE READ`` connection so both equivalence reads share one snapshot.

    Yields a :class:`HeldConnExecutor` over a single connection whose transaction is pinned to
    ``REPEATABLE READ`` (issued before any query, as Postgres requires) — both the batched and
    inlined reads run on it, observing one frozen snapshot. Rolled back and closed on exit (the
    battery persists nothing).
    """
    conn = await engine.connect()
    try:
        txn = await conn.begin()
        try:
            await conn.execute(text("set transaction isolation level repeatable read"))
            yield HeldConnExecutor(conn)
        finally:
            await txn.rollback()
    finally:
        await conn.close()


async def run_counted(schema, query, *, inline: bool, executor, variables=None):
    """Run ``query`` under the inline toggle on ``executor``; return ``(result, count)``.

    Binds the request-scoped ``executor`` (the shared ``REPEATABLE READ`` connection, so both
    reads see one snapshot — see :func:`repeatable_read_snapshot`) and counts the SQL the
    operation issues — the two halves of every equivalence case (data identity + count delta).
    The engine is the SAME instance ``count_sql`` is attached to, so the held connection's
    statements are still counted.
    """
    with count_sql(get_engine()) as counter:
        with pg_request_context(executor):
            result = await graphql(
                schema,
                query,
                variable_values=variables,
                execution_context_class=context_class(inline),
            )
    return result, counter.count


async def assert_inlined_equivalent(
    schema, query, *, expected_saving: int, variables=None
):
    """Run ``query`` inlined vs batched; assert byte-identical data + the count delta.

    ``expected_saving`` is how many FEWER statements the inlined run must issue (the number
    of folds that fired). ``0`` is a SKIP case (the fallback): data identical, count equal —
    unsafe-but-skipped is always correct. A positive value proves the fold fired.

    Both reads run on ONE shared ``REPEATABLE READ`` snapshot (one held connection) so a
    sibling test's concurrently-committed row cannot drift between them — the data assertion
    compares inline-vs-batched against genuinely identical committed state.
    """
    engine = get_engine()
    async with repeatable_read_snapshot(engine) as executor:
        batched, batched_count = await run_counted(
            schema, query, inline=False, executor=executor, variables=variables
        )
        inlined, inlined_count = await run_counted(
            schema, query, inline=True, executor=executor, variables=variables
        )
    # the inlining invariant: BYTE-IDENTICAL data and errors under both flag states.
    assert batched.errors == inlined.errors
    assert batched.data == inlined.data
    # and the fold actually changed the statement count by exactly the expected amount.
    assert inlined_count == batched_count - expected_saving, (
        f"expected {expected_saving} fewer statements inlined; "
        f"batched={batched_count} inlined={inlined_count}"
    )
    return batched, batched_count, inlined_count


# the broad oracle: a matrix of demo-schema operations that must be BYTE-IDENTICAL under
# both flag states. Each is run inlined vs batched and deep-equality-checked — the same
# data assertion every existing pg test makes, but across the flag toggle, so flipping
# inlining on for any of these shapes can never change the answer. ``saving`` is the
# minimum statements the inlined run must save (0 for a pure-SKIP shape; >0 proves a fold).
DEMO_MATRIX = [
    ("hasone", "{ posts { id author { id name } } }", 1),
    ("hasmany", "{ authors { id posts { id title } } }", 1),
    ("hasone_and_hasmany", "{ posts { id author { name } comments { id } } }", 2),
    ("two_level", "{ authors { id posts { id comments { id body } } } }", 1),
    ("author_by_id", "{ author(id: 2) { id name posts { id } } }", 1),
    ("missing_author", "{ author(id: 999) { id name } }", 0),
    ("posts_root", "{ posts { id title } }", 0),
    (
        "connection_skips",
        "{ authors { id postsConnection(first: 2) { totalCount edges { node { id } } } } }",
        0,
    ),
    # posts->author folds (1); author->posts is a second level under the absorbed author,
    # so it falls back batched off the author extract — single-level saving of 1.
    ("deep_author_chain", "{ posts { author { name posts { id } } } }", 1),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query,saving", [(q, s) for _name, q, s in DEMO_MATRIX],
    ids=[name for name, _q, _s in DEMO_MATRIX],
)
async def test_demo_matrix_byte_identical_under_toggle(demo_schema, query, saving):
    """The broad oracle: every demo-schema shape is byte-identical inlined vs batched.

    Flipping ``inline_relations`` on must never change the answer for ANY of these shapes —
    the same exact-data assertion the existing pg suite makes, now swept across the toggle.
    A foldable shape (``saving > 0``) also proves the fold fired (fewer statements); a
    SKIP shape (``saving == 0``) proves the fallback stayed byte-identical.
    """
    await assert_inlined_equivalent(demo_schema, query, expected_saving=saving)


@pytest_asyncio.fixture
async def demo_schema():
    """(Re)seed ``grafast_demo`` and yield the demo GraphQL schema; dispose after."""
    await dispose_engine()
    await setup_demo_schema()
    schema = build_demo_schema()
    yield schema
    await dispose_engine()


# ============================================================ FOLD: the core shapes


@pytest.mark.asyncio
async def test_has_one_folds_byte_identical(demo_schema):
    """Post.author (hasOne) folds into the posts statement: 2 -> 1, data identical."""
    query = "{ posts { id title author { id name } } }"
    _b, batched_count, _i = await assert_inlined_equivalent(
        demo_schema, query, expected_saving=1
    )
    assert batched_count == 2  # posts + authors batched


@pytest.mark.asyncio
async def test_has_many_unpaginated_folds_byte_identical(demo_schema):
    """Author.posts (unpaginated hasMany) folds into the authors statement: 2 -> 1."""
    query = "{ authors { id name posts { id title } } }"
    _b, batched_count, _i = await assert_inlined_equivalent(
        demo_schema, query, expected_saving=1
    )
    assert batched_count == 2  # authors + posts batched


@pytest.mark.asyncio
async def test_multi_level_single_fold_byte_identical(demo_schema):
    """authors -> posts -> comments: ONE fold (posts into authors); comments stays batched.

    Nested LATERAL is unsupported this wave (a flat ``build_lateral``), so the deeper
    comments relation falls back to its batched path off the extracted post rows — still
    byte-identical, only one fewer statement (3 -> 2), not two.
    """
    query = "{ authors { id posts { id comments { id body } } } }"
    _b, batched_count, _i = await assert_inlined_equivalent(
        demo_schema, query, expected_saving=1
    )
    assert batched_count == 3  # authors + posts + comments batched


@pytest.mark.asyncio
async def test_mixed_hasone_and_hasmany_under_posts(demo_schema):
    """Post.author (hasOne) AND Post.comments (hasMany) both fold into posts: 3 -> 1.

    Two distinct children of one parent fold into TWO LATERALs on the same posts statement;
    authors + comments collapse into the posts query. (comments.author is a third layer that
    stays batched off the comment extract, so the total saving is the two posts-level folds.)
    """
    query = "{ posts { id author { name } comments { id body } } }"
    batched, batched_count, inlined_count = await assert_inlined_equivalent(
        demo_schema, query, expected_saving=2
    )
    assert batched_count == 3  # posts + authors + comments
    assert inlined_count == 1


@pytest.mark.asyncio
async def test_deep_chain_each_layer_folds_one_level(demo_schema):
    """The O(depth) datasource query stays byte-identical and drops statements when inlined.

    The broadest oracle: the exact nested query from test_pg_datasource (authors -> posts ->
    {author, comments -> author}) run inlined vs batched. Each parent absorbs its immediate
    relation children into one LATERAL layer; data must be byte-identical and the count
    strictly lower. The exact saving is computed against the batched baseline (some deeper
    layers fall back), so we only assert it dropped.

    Batched is 5 statements (authors, posts, post.author, comments, comment.author). Inlined:
    authors absorbs posts (and posts absorbs its own author + comments into the same LATERAL
    layer), so several layers collapse — strictly fewer than 5.
    """
    query = """
    {
      authors {
        id
        name
        posts {
          id
          title
          author { id name }
          comments { id body author { name } }
        }
      }
    }
    """
    # both reads share one REPEATABLE READ snapshot so no sibling commit drifts between them.
    async with repeatable_read_snapshot(get_engine()) as executor:
        batched, batched_count = await run_counted(
            demo_schema, query, inline=False, executor=executor
        )
        inlined, inlined_count = await run_counted(
            demo_schema, query, inline=True, executor=executor
        )
    assert batched.errors == inlined.errors
    assert batched.data == inlined.data
    assert inlined_count < batched_count
    assert batched_count == 5  # the test_pg_datasource baseline


# ============================================================ explicit ORDER BY fold


def build_ordered_posts_schema():
    """A schema whose Author.posts orders by title DESC (non-PK) with the PK tie-break.

    Proves the nested ``json_agg(... ORDER BY title DESC, id)`` reproduces the standalone
    child's normalized order byte-for-byte.
    """
    registry = PgRegistry()
    authors = PgResource(
        "authors", DEMO_SCHEMA, "authors", AUTHORS_COLS, registry=registry
    )
    posts = PgResource(
        "posts", DEMO_SCHEMA, "posts", POSTS_COLS, registry=registry
    )
    authors.has_many("posts", posts, local_column="id", remote_column="author_id")

    from grafast_py.pg.ordering import OrderTerm
    from grafast_py.pg.steps import PgSelectAllStep

    sdl = """
    type Query { authors: [Author!]! }
    type Author { id: Int! name: String! posts: [Post!]! }
    type Post { id: Int! title: String! }
    """

    def authors_plan(parent, args, info):
        return PgSelectAllStep(authors, order_by=["id"]).for_parent(parent)

    def posts_plan(parent, args, info):
        return authors.related_many(
            parent, "posts", order_by=[OrderTerm("title", descending=True)]
        )

    def leaf(key):
        def plan(parent, args, info):
            return access(parent, (key,))

        return plan

    return make_grafast_schema(
        sdl,
        {
            "Query": {"authors": authors_plan},
            "Author": {"id": leaf("id"), "name": leaf("name"), "posts": posts_plan},
            "Post": {"id": leaf("id"), "title": leaf("title")},
        },
    )


@pytest.mark.asyncio
async def test_non_pk_order_fold_preserves_order(demo_schema):
    """A hasMany ordered by title DESC (+PK tie-break) folds with the exact nested order."""
    schema = build_ordered_posts_schema()
    query = "{ authors { id posts { id title } } }"
    batched, batched_count, _i = await assert_inlined_equivalent(
        schema, query, expected_saving=1
    )
    # sanity: the order is genuinely non-PK (titles descending within each author).
    titles = [p["title"] for p in batched.data["authors"][0]["posts"]]
    assert titles == sorted(titles, reverse=True)


# ============================================================ NULL / empty children


@pytest_asyncio.fixture
async def null_empty_schema():
    """A schema with an author owning ZERO posts and a post whose author FK points nowhere.

    Exercises the NULL/empty scatter: an empty hasMany must yield ``[]`` (coalesce json_agg)
    and a hasOne whose FK matches no row must yield ``null`` — identical to the batched path.

    It ALSO seeds a deeper empty-hasMany (a post owning ZERO comments) so the same
    ``coalesce(json_agg, '[]')`` empty scatter is proven one level down — the task's
    "a post with no comments" sub-case — alongside the top-level empty author.
    """
    await dispose_engine()
    await setup_demo_schema()
    engine = get_engine()
    async with engine.begin() as conn:
        # author 5 owns no posts (empty hasMany); a post pointing at a missing author id is
        # impossible under the FK, so model the null-hasOne with a separate nullable-FK table.
        # (id 5 leaves the base seed's 1..3 untouched; the post-with-no-comments is id 4's.)
        await conn.execute(
            text(f"INSERT INTO {DEMO_SCHEMA}.authors (id, name) VALUES (5, 'Empty Author')")
        )
        # a post owned by author 1 carrying ZERO comments (the base seed gives every post 2),
        # so Post.comments must scatter [] for it — the deeper empty-hasMany sub-case.
        await conn.execute(
            text(
                f"INSERT INTO {DEMO_SCHEMA}.posts (id, author_id, title)"
                " VALUES (99, 1, 'Commentless Post')"
            )
        )
        await conn.execute(text(f"DROP TABLE IF EXISTS {DEMO_SCHEMA}.notes"))
        await conn.execute(
            text(
                f"""
                CREATE TABLE {DEMO_SCHEMA}.notes (
                    id        integer PRIMARY KEY,
                    author_id integer NULL,
                    body      text NOT NULL
                )
                """
            )
        )
        # note 1 references author 1; note 2's author_id is NULL (hasOne points nowhere).
        await conn.execute(
            text(
                f"INSERT INTO {DEMO_SCHEMA}.notes (id, author_id, body)"
                " VALUES (1, 1, 'a'), (2, NULL, 'b')"
            )
        )

    registry = PgRegistry()
    authors = PgResource(
        "authors", DEMO_SCHEMA, "authors", AUTHORS_COLS, registry=registry
    )
    posts = PgResource(
        "posts", DEMO_SCHEMA, "posts", POSTS_COLS, registry=registry
    )
    comments = PgResource(
        "comments", DEMO_SCHEMA, "comments", COMMENTS_COLS, registry=registry,
    )
    notes = PgResource(
        "notes", DEMO_SCHEMA, "notes",
        [
            PgColumn("id", sql_type=Integer()),
            PgColumn("author_id", sql_type=Integer()),
            PgColumn("body", sql_type=Text()),
        ],
        registry=registry,
    )
    authors.has_many("posts", posts, local_column="id", remote_column="author_id")
    posts.has_many("comments", comments, local_column="id", remote_column="post_id")
    notes.has_one("author", authors, local_column="author_id", remote_column="id")

    from grafast_py.pg.steps import PgSelectAllStep

    sdl = """
    type Query { authors: [Author!]! notes: [Note!]! }
    type Author { id: Int! name: String! posts: [Post!]! }
    type Post { id: Int! title: String! comments: [Comment!]! }
    type Comment { id: Int! body: String! }
    type Note { id: Int! body: String! author: Author }
    """

    def leaf(key):
        def plan(parent, args, info):
            return access(parent, (key,))

        return plan

    schema = make_grafast_schema(
        sdl,
        {
            "Query": {
                "authors": lambda p, a, i: PgSelectAllStep(
                    authors, order_by=["id"]
                ).for_parent(p),
                "notes": lambda p, a, i: PgSelectAllStep(
                    notes, order_by=["id"]
                ).for_parent(p),
            },
            "Author": {
                "id": leaf("id"),
                "name": leaf("name"),
                "posts": lambda p, a, i: authors.related_many(p, "posts"),
            },
            "Post": {
                "id": leaf("id"),
                "title": leaf("title"),
                "comments": lambda p, a, i: posts.related_many(p, "comments"),
            },
            "Comment": {"id": leaf("id"), "body": leaf("body")},
            "Note": {
                "id": leaf("id"),
                "body": leaf("body"),
                "author": lambda p, a, i: notes.related_single(p, "author"),
            },
        },
    )
    yield schema
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP TABLE IF EXISTS {DEMO_SCHEMA}.notes"))
    await dispose_engine()


@pytest.mark.asyncio
async def test_empty_hasmany_folds_to_empty_list(null_empty_schema):
    """An author with no posts yields [] inlined exactly as batched (coalesce json_agg)."""
    query = "{ authors { id posts { id } } }"
    batched, _bc, _ic = await assert_inlined_equivalent(
        null_empty_schema, query, expected_saving=1
    )
    empty_author = next(a for a in batched.data["authors"] if a["id"] == 5)
    assert empty_author["posts"] == []


@pytest.mark.asyncio
async def test_post_with_no_comments_folds_to_empty_list(null_empty_schema):
    """A post owning zero comments yields [] one level DOWN, inlined exactly as batched.

    The deeper empty-hasMany: ``authors -> posts -> comments`` where post 99 has no comments.
    posts folds into authors (1 fewer statement); the inner comments relation falls back
    batched off the extracted post rows (a flat LATERAL this wave), so the saving is the one
    posts fold — but the empty post's ``comments`` must still scatter ``[]`` identically.
    """
    query = "{ authors { id posts { id comments { id } } } }"
    batched, _bc, _ic = await assert_inlined_equivalent(
        null_empty_schema, query, expected_saving=1
    )
    commentless = next(
        p
        for a in batched.data["authors"]
        for p in a["posts"]
        if p["id"] == 99
    )
    assert commentless["comments"] == []


@pytest.mark.asyncio
async def test_null_hasone_folds_to_null(null_empty_schema):
    """A hasOne whose FK points nowhere yields null inlined exactly as batched."""
    query = "{ notes { id author { id name } } }"
    batched, _bc, _ic = await assert_inlined_equivalent(
        null_empty_schema, query, expected_saving=1
    )
    by_id = {n["id"]: n for n in batched.data["notes"]}
    assert by_id[1]["author"]["name"] == "Ada Lovelace"
    assert by_id[2]["author"] is None  # the NULL-FK note's author folds to null


# ====================================================== hasOne over a NON-UNIQUE remote FK


@pytest_asyncio.fixture
async def non_unique_hasone_schema():
    """A hasOne whose remote column is NON-UNIQUE: several rows match the correlation.

    The canonical hasOne points at the target's PK (at most one match), so a bare
    ``LIMIT 1`` inside the LATERAL is indistinguishable from the batched ``rows[0]``. But
    the engine permits declaring a hasOne over a non-PK, non-unique remote column, where
    SEVERAL child rows match the FK. The batched ``PgSelectSingleStep`` floors its order to
    ``ORDER BY <pk>`` and takes ``rows[0]`` -> the lowest-PK matching row, deterministically.
    The fold MUST reproduce that exact row, so the inner select carries the child's ORDER BY
    BEFORE ``LIMIT 1``; an order-less ``LIMIT 1`` would pick an arbitrary heap row and diverge.

    Tags are inserted in REVERSE-PK physical order under one ``group_id`` so a heap-order
    ``LIMIT 1`` would pick the HIGH-PK row, making any divergence visible.
    """
    await dispose_engine()
    await setup_demo_schema()
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP TABLE IF EXISTS {DEMO_SCHEMA}.ptag"))
        await conn.execute(text(f"DROP TABLE IF EXISTS {DEMO_SCHEMA}.pnote"))
        await conn.execute(
            text(
                f"""
                CREATE TABLE {DEMO_SCHEMA}.ptag (
                    id       integer PRIMARY KEY,
                    group_id integer NOT NULL,
                    label    text NOT NULL
                )
                """
            )
        )
        await conn.execute(
            text(
                f"""
                CREATE TABLE {DEMO_SCHEMA}.pnote (
                    id       integer PRIMARY KEY,
                    group_id integer NOT NULL
                )
                """
            )
        )
        # tags inserted high-PK first so a heap-order LIMIT 1 picks id=30 ('high'), while the
        # batched ORDER BY id, rows[0] picks id=10 ('low') — the divergence the fold must avoid.
        await conn.execute(
            text(
                f"INSERT INTO {DEMO_SCHEMA}.ptag (id, group_id, label) VALUES"
                " (30, 7, 'high'), (20, 7, 'mid'), (10, 7, 'low')"
            )
        )
        await conn.execute(
            text(f"INSERT INTO {DEMO_SCHEMA}.pnote (id, group_id) VALUES (1, 7)")
        )

    registry = PgRegistry()
    notes = PgResource(
        "pnote", DEMO_SCHEMA, "pnote",
        [PgColumn("id", sql_type=Integer()), PgColumn("group_id", sql_type=Integer())],
        registry=registry,
    )
    tags = PgResource(
        "ptag", DEMO_SCHEMA, "ptag",
        [
            PgColumn("id", sql_type=Integer()),
            PgColumn("group_id", sql_type=Integer()),
            PgColumn("label", sql_type=Text()),
        ],
        registry=registry,
    )
    # a hasOne over the NON-UNIQUE remote column group_id (three tags match group_id=7).
    notes.has_one("tag", tags, local_column="group_id", remote_column="group_id")

    from grafast_py.pg.steps import PgSelectAllStep

    sdl = """
    type Query { notes: [Note!]! }
    type Note { id: Int! tag: Tag }
    type Tag { id: Int! label: String! }
    """

    def leaf(key):
        def plan(parent, args, info):
            return access(parent, (key,))

        return plan

    schema = make_grafast_schema(
        sdl,
        {
            "Query": {
                "notes": lambda p, a, i: PgSelectAllStep(
                    notes, order_by=["id"]
                ).for_parent(p)
            },
            "Note": {
                "id": leaf("id"),
                "tag": lambda p, a, i: notes.related_single(p, "tag"),
            },
            "Tag": {"id": leaf("id"), "label": leaf("label")},
        },
    )
    yield schema
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP TABLE IF EXISTS {DEMO_SCHEMA}.pnote"))
        await conn.execute(text(f"DROP TABLE IF EXISTS {DEMO_SCHEMA}.ptag"))
    await dispose_engine()


@pytest.mark.asyncio
async def test_non_unique_hasone_fold_picks_same_row(non_unique_hasone_schema):
    """A hasOne over a non-unique remote FK folds to the SAME row the batched path picks.

    The inner select reproduces the child's ``ORDER BY <pk>`` before ``LIMIT 1``, so the
    fold returns the lowest-PK matching tag (id=10, 'low') byte-for-byte with the batched
    ``rows[0]`` — not the arbitrary heap-order row (id=30, 'high') a bare ``LIMIT 1`` gives.
    """
    query = "{ notes { id tag { id label } } }"
    batched, _bc, _ic = await assert_inlined_equivalent(
        non_unique_hasone_schema, query, expected_saving=1
    )
    # the chosen tag is genuinely the lowest-PK of several matches (not the heap-order row).
    assert batched.data["notes"][0]["tag"] == {"id": 10, "label": "low"}


# ============================================ SKIP: bare (codec-less) non-native columns


@pytest_asyncio.fixture
async def bare_non_native_schema():
    """A hasMany whose child has BARE numeric + timestamptz columns (no codec, no type).

    The blocker-1 corruption vector: a codec-less ``numeric`` / ``timestamptz`` column whose
    ``to_jsonb`` -> JSON form differs from the asyncpg row value (``12.5000`` -> lossy
    ``12.5``; a UTC datetime -> a server-tz-shifted string). With no declared type the
    predicate cannot prove the column native, so the fold SKIPS and the batched child stays —
    keeping the data byte-identical. (A host who declares the types, e.g. via the ORM bridge,
    gets the same SKIP through the type-aware branch.)
    """
    await dispose_engine()
    await setup_demo_schema()
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP TABLE IF EXISTS {DEMO_SCHEMA}.bare_evt"))
        await conn.execute(
            text(
                f"""
                CREATE TABLE {DEMO_SCHEMA}.bare_evt (
                    id        integer PRIMARY KEY,
                    author_id integer NOT NULL,
                    amount    numeric(12, 4) NOT NULL,
                    ts        timestamptz NOT NULL
                )
                """
            )
        )
        await conn.execute(
            text(
                f"INSERT INTO {DEMO_SCHEMA}.bare_evt (id, author_id, amount, ts)"
                " VALUES (1, 1, 12.5000, '2024-01-02 03:04:05.123456+00')"
            )
        )

    registry = PgRegistry()
    authors = PgResource(
        "authors", DEMO_SCHEMA, "authors", AUTHORS_COLS, registry=registry
    )
    # amount / ts are BARE: no codec, no sql_type -> UNKNOWN, so the fold must SKIP.
    events = PgResource(
        "bare_evt", DEMO_SCHEMA, "bare_evt",
        ["id", "author_id", "amount", "ts"],
        registry=registry,
    )
    authors.has_many("evts", events, local_column="id", remote_column="author_id")

    from grafast_py.pg.steps import PgSelectAllStep

    sdl = """
    type Query { authors: [Author!]! }
    type Author { id: Int! evts: [Evt!]! }
    type Evt { id: Int! amount: String! ts: String! }
    """

    def leaf(key):
        def plan(parent, args, info):
            return access(parent, (key,))

        return plan

    schema = make_grafast_schema(
        sdl,
        {
            "Query": {
                "authors": lambda p, a, i: PgSelectAllStep(
                    authors, order_by=["id"]
                ).for_parent(p)
            },
            "Author": {
                "id": leaf("id"),
                "evts": lambda p, a, i: authors.related_many(p, "evts"),
            },
            "Evt": {"id": leaf("id"), "amount": leaf("amount"), "ts": leaf("ts")},
        },
    )
    yield schema
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP TABLE IF EXISTS {DEMO_SCHEMA}.bare_evt"))
    await dispose_engine()


@pytest.mark.asyncio
async def test_bare_non_native_column_skips_but_identical(bare_non_native_schema):
    """A bare numeric/timestamptz child column SKIPS the fold; data stays byte-identical.

    Without the guard the fold would corrupt: ``amount`` ``12.5000`` -> ``12.5`` (lost
    precision) and ``ts`` would tz-shift to the server zone. The SKIP keeps the batched child,
    so both paths return the SAME strings — unsafe-but-skipped is always correct.
    """
    query = "{ authors { id evts { id amount ts } } }"
    batched, _bc, _ic = await assert_inlined_equivalent(
        bare_non_native_schema, query, expected_saving=0
    )
    evt = batched.data["authors"][0]["evts"][0]
    # the trailing precision and the UTC offset survive (the corruption the fold would cause).
    assert evt["amount"] == "12.5000"
    assert evt["ts"] == "2024-01-02 03:04:05.123456+00:00"


# ============================================================ CODEC: json-stable folds


@pytest_asyncio.fixture
async def labels_schema_factory():
    """A factory yielding a labels schema whose ``code`` codec is configurable.

    ``json_safe=True`` -> a NATIVE-typed (text) ``to_py`` uppercasing codec (foldable);
    ``json_safe=False`` -> a timestamptz-typed codec column (SKIPPED, must stay batched).
    Both decode through the SAME ``decode_rows`` on read, so the presented data is identical
    either way — the SKIP is only about whether the json round-trip is provably stable.
    """
    await dispose_engine()
    await setup_demo_schema()
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP TABLE IF EXISTS {DEMO_SCHEMA}.author_labels"))
        await conn.execute(
            text(
                f"""
                CREATE TABLE {DEMO_SCHEMA}.author_labels (
                    id        integer PRIMARY KEY,
                    author_id integer NOT NULL,
                    code      text NOT NULL,
                    seen_at   timestamptz NOT NULL
                )
                """
            )
        )
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        await conn.execute(
            text(
                f"INSERT INTO {DEMO_SCHEMA}.author_labels (id, author_id, code, seen_at)"
                " VALUES (:id, :author_id, :code, :seen_at)"
            ),
            [
                {"id": 1, "author_id": 1, "code": "alpha", "seen_at": base},
                {"id": 2, "author_id": 1, "code": "bravo", "seen_at": base},
                {"id": 3, "author_id": 2, "code": "charlie", "seen_at": base},
            ],
        )

    def build(json_safe: bool):
        registry = PgRegistry()
        authors = PgResource(
            "authors", DEMO_SCHEMA, "authors", AUTHORS_COLS, registry=registry
        )
        label_id = PgColumn("id", sql_type=Integer())
        label_author_id = PgColumn("author_id", sql_type=Integer())
        if json_safe:
            cols = [
                label_id,
                label_author_id,
                PgColumn("code", codec=PgCodec(to_py=str.upper)),
            ]
            fields = "type Label { id: Int! code: String! }"
            sub = "code"
        else:
            cols = [
                label_id,
                label_author_id,
                PgColumn("code", sql_type=Text()),
                # the ONLY non-json-stable column: a timestamptz codec column makes the fold SKIP.
                PgColumn(
                    "seen_at", codec=PgCodec(sql_type=DateTime(timezone=True))
                ),
            ]
            fields = "type Label { id: Int! code: String! }"
            sub = "code"
        labels = PgResource(
            "author_labels", DEMO_SCHEMA, "author_labels", cols, registry=registry
        )
        authors.has_many(
            "labels", labels, local_column="id", remote_column="author_id"
        )

        from grafast_py.pg.steps import PgSelectAllStep

        sdl = f"""
        type Query {{ authors: [Author!]! }}
        type Author {{ id: Int! labels: [Label!]! }}
        {fields}
        """

        def leaf(key):
            def plan(parent, args, info):
                return access(parent, (key,))

            return plan

        return make_grafast_schema(
            sdl,
            {
                "Query": {
                    "authors": lambda p, a, i: PgSelectAllStep(
                        authors, order_by=["id"]
                    ).for_parent(p)
                },
                "Author": {
                    "id": leaf("id"),
                    "labels": lambda p, a, i: authors.related_many(p, "labels"),
                },
                "Label": {"id": leaf("id"), sub: leaf(sub)},
            },
        )

    yield build
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP TABLE IF EXISTS {DEMO_SCHEMA}.author_labels"))
    await dispose_engine()


@pytest.mark.asyncio
async def test_json_stable_codec_column_folds(labels_schema_factory):
    """A native-text uppercasing codec is json-stable: the labels hasMany folds, 2 -> 1."""
    schema = labels_schema_factory(json_safe=True)
    query = "{ authors { id labels { id code } } }"
    batched, _bc, _ic = await assert_inlined_equivalent(
        schema, query, expected_saving=1
    )
    # the codec ran on decode (uppercased) identically on both paths.
    assert batched.data["authors"][0]["labels"][0]["code"] == "ALPHA"


@pytest.mark.asyncio
async def test_non_json_safe_codec_column_skips_but_identical(labels_schema_factory):
    """A timestamptz codec column is NOT json-stable: the fold SKIPS, data stays identical.

    The fallback: an unsafe codec keeps the batched child (count unchanged), and the result
    is byte-identical either way — unsafe-but-skipped is always correct.
    """
    schema = labels_schema_factory(json_safe=False)
    query = "{ authors { id labels { id code } } }"
    await assert_inlined_equivalent(schema, query, expected_saving=0)


@pytest_asyncio.fixture
async def jsonb_docs_schema():
    """Author.docs hasMany over a ``jsonb`` column the HOST proves json-stable, plus an
    author owning zero docs.

    The non-native codec column the safety predicate folds only under a host OVERRIDE: a
    ``jsonb`` column carries ``sql_type=JSONB()`` (so the DEFAULT
    ``PgCodec.is_json_stable`` derives UNSAFE — a non-native type the predicate would SKIP),
    but the host sets ``json_stable=True`` because a ``jsonb`` value provably round-trips
    ``to_jsonb`` -> JSON -> Python to the IDENTICAL ``dict`` it decodes off a batched asyncpg
    row (unlike ``numeric``, which JSON-renders ``12.50`` to a lossy float ``12.5``, or
    ``timestamptz``, which JSON-renders to a tz-shifted string — both genuinely NOT
    json-stable, hence correctly SKIPPED by ``test_non_json_safe_codec_column_skips_but_identical``).
    So this exercises the ``json_stable=True`` OVERRIDE branch of ``is_json_stable``
    end-to-end: a non-native column the host has proven stable DOES fold, byte-identically.

    Author 3 owns zero docs, so the same fixture also pins the empty-hasMany scatter under a
    codec column (``[]`` inlined exactly as batched).
    """
    await dispose_engine()
    await setup_demo_schema()
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP TABLE IF EXISTS {DEMO_SCHEMA}.docs"))
        await conn.execute(
            text(
                f"""
                CREATE TABLE {DEMO_SCHEMA}.docs (
                    id        integer PRIMARY KEY,
                    author_id integer NOT NULL,
                    payload   jsonb NOT NULL
                )
                """
            )
        )
        # author 1 owns docs 1-2, author 2 owns doc 3, author 3 owns none (empty hasMany).
        # nested arrays / empty objects / bools exercise the structured json round-trip.
        await conn.execute(
            text(
                f"INSERT INTO {DEMO_SCHEMA}.docs (id, author_id, payload)"
                " VALUES (:id, :author_id, CAST(:payload AS jsonb))"
            ),
            [
                {"id": 1, "author_id": 1, "payload": '{"k": [1, 2, 3]}'},
                {"id": 2, "author_id": 1, "payload": '{"k": []}'},
                {"id": 3, "author_id": 2, "payload": '{"z": true, "n": null}'},
            ],
        )

    registry = PgRegistry()
    authors = PgResource(
        "authors", DEMO_SCHEMA, "authors", AUTHORS_COLS, registry=registry
    )
    docs = PgResource(
        "docs",
        DEMO_SCHEMA,
        "docs",
        # sql_type=JSONB() makes the DEFAULT derivation unsafe; json_stable=True is the host's
        # proven override that makes this non-native column foldable.
        [
            PgColumn("id", sql_type=Integer()),
            PgColumn("author_id", sql_type=Integer()),
            PgColumn("payload", codec=PgCodec(sql_type=JSONB(), json_stable=True)),
        ],
        registry=registry,
    )
    authors.has_many("docs", docs, local_column="id", remote_column="author_id")

    from grafast_py.pg.steps import PgSelectAllStep

    sdl = """
    type Query { authors: [Author!]! }
    type Author { id: Int! docs: [Doc!]! }
    type Doc { id: Int! payload: JSON! }
    scalar JSON
    """

    def leaf(key):
        def plan(parent, args, info):
            return access(parent, (key,))

        return plan

    schema = make_grafast_schema(
        sdl,
        {
            "Query": {
                "authors": lambda p, a, i: PgSelectAllStep(
                    authors, order_by=["id"]
                ).for_parent(p)
            },
            "Author": {
                "id": leaf("id"),
                "docs": lambda p, a, i: authors.related_many(p, "docs"),
            },
            "Doc": {"id": leaf("id"), "payload": leaf("payload")},
        },
    )
    yield schema
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP TABLE IF EXISTS {DEMO_SCHEMA}.docs"))
    await dispose_engine()


@pytest.mark.asyncio
async def test_json_stable_override_non_native_column_folds(jsonb_docs_schema):
    """A non-native jsonb column the host PROVED json-stable folds, byte-identically: 2 -> 1.

    The override path: ``sql_type=JSONB()`` defaults UNSAFE, but ``json_stable=True`` lets the
    fold fire — and the structured json values (nested arrays, empty objects, bools, nulls)
    round-trip the LATERAL identically to the batched decode. Author 3's empty docs also
    scatter ``[]`` identically. The fold genuinely saves the docs statement (2 -> 1).
    """
    query = "{ authors { id docs { id payload } } }"
    batched, _bc, _ic = await assert_inlined_equivalent(
        jsonb_docs_schema, query, expected_saving=1
    )
    by_id = {a["id"]: a for a in batched.data["authors"]}
    # the structured json decoded identically on both paths.
    assert by_id[1]["docs"][0]["payload"] == {"k": [1, 2, 3]}
    assert by_id[2]["docs"][0]["payload"] == {"z": True, "n": None}
    # author 3 owns no docs -> [] under the codec column, exactly as batched.
    assert by_id[3]["docs"] == []


# ============================================================ SKIP: self-referential relation


@pytest_asyncio.fixture
async def self_relation_schema():
    """A SELF-referential ``employees`` table: reports (hasMany) + manager (hasOne) on itself.

    ``employees(id, manager_id, name)`` with a self-FK ``manager_id -> id``. A self-relation
    folds the SAME table into itself, which the flat ``build_lateral`` cannot alias apart from
    the outer parent (the inner child would carry the SAME unaliased table name, collapsing the
    correlation so every parent silently gets ``[]`` / ``null``). So the predicate must SKIP it
    and keep the batched ``= ANY($1)`` path — this fixture proves the SKIP returns CORRECT,
    NON-EMPTY children byte-identically under both flag states, with NO statement saving.
    """
    await dispose_engine()
    await setup_demo_schema()
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP TABLE IF EXISTS {DEMO_SCHEMA}.employees"))
        await conn.execute(
            text(
                f"""
                CREATE TABLE {DEMO_SCHEMA}.employees (
                    id         integer PRIMARY KEY,
                    manager_id integer NULL REFERENCES {DEMO_SCHEMA}.employees (id),
                    name       text NOT NULL
                )
                """
            )
        )
        # employee 1 is the top manager; 2 and 3 report to 1; 4 reports to 2. So manager 1 has
        # two reports, manager 2 has one, employees 3 and 4 have none (the empty-hasMany case),
        # and employee 1's manager FK is NULL (the null-hasOne case).
        await conn.execute(
            text(
                f"INSERT INTO {DEMO_SCHEMA}.employees (id, manager_id, name) VALUES"
                " (1, NULL, 'Root'), (2, 1, 'Mid A'), (3, 1, 'Mid B'), (4, 2, 'Leaf')"
            )
        )

    registry = PgRegistry()
    employees = PgResource(
        "employees", DEMO_SCHEMA, "employees",
        [
            PgColumn("id", sql_type=Integer()),
            PgColumn("manager_id", sql_type=Integer()),
            PgColumn("name", sql_type=Text()),
        ],
        registry=registry,
    )
    # the self-FK both ways: a manager's reports (hasMany) and a report's manager (hasOne).
    employees.has_many(
        "reports", employees, local_column="id", remote_column="manager_id"
    )
    employees.has_one(
        "manager", employees, local_column="manager_id", remote_column="id"
    )

    from grafast_py.pg.steps import PgSelectAllStep

    sdl = """
    type Query { employees: [Employee!]! }
    type Employee {
      id: Int!
      name: String!
      reports: [Employee!]!
      manager: Employee
    }
    """

    def leaf(key):
        def plan(parent, args, info):
            return access(parent, (key,))

        return plan

    schema = make_grafast_schema(
        sdl,
        {
            "Query": {
                "employees": lambda p, a, i: PgSelectAllStep(
                    employees, order_by=["id"]
                ).for_parent(p)
            },
            "Employee": {
                "id": leaf("id"),
                "name": leaf("name"),
                "reports": lambda p, a, i: employees.related_many(p, "reports"),
                "manager": lambda p, a, i: employees.related_single(p, "manager"),
            },
        },
    )
    yield schema
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP TABLE IF EXISTS {DEMO_SCHEMA}.employees"))
    await dispose_engine()


@pytest.mark.asyncio
async def test_self_relation_skips_but_identical(self_relation_schema):
    """A self-referential relation is SKIPPED (no fold); children are CORRECT, byte-identical.

    The regression for the self-relation bug: without the same-table guard the fold collapses
    the correlation so EVERY manager gets ``reports: []`` (and every employee ``manager: null``)
    — silently wrong. With the SKIP, both the hasMany ``reports`` and the hasOne ``manager``
    fall back to the batched path, returning the IDENTICAL non-empty children under both flag
    states, and the count is UNCHANGED (no statement saved — the fold did not fire).
    """
    query = "{ employees { id name reports { id name } manager { id name } } }"
    batched, _bc, _ic = await assert_inlined_equivalent(
        self_relation_schema, query, expected_saving=0
    )
    by_id = {e["id"]: e for e in batched.data["employees"]}
    # the children are genuinely present (a collapsed correlation would give [] / null) —
    # manager 1 has reports 2 and 3; employee 2's manager is employee 1.
    assert [r["id"] for r in by_id[1]["reports"]] == [2, 3]
    assert by_id[1]["manager"] is None
    assert by_id[2]["manager"] == {"id": 1, "name": "Root"}
    assert [r["id"] for r in by_id[2]["reports"]] == [4]
    assert by_id[3]["reports"] == []  # a leaf with no reports


# ============================================================ SKIP: composite key


@pytest_asyncio.fixture
async def composite_schema():
    """A region -> stores hasMany over a COMPOSITE (org_id, region_id) FK; SKIPPED this wave."""
    from examples.seed import setup_composite_tables

    await dispose_engine()
    await setup_demo_schema()
    await setup_composite_tables()

    registry = PgRegistry()
    regions = PgResource(
        "regions", DEMO_SCHEMA, "regions",
        ["org_id", "region_id", "label"], primary_key="org_id", registry=registry,
    )
    stores = PgResource(
        "stores", DEMO_SCHEMA, "stores",
        ["id", "org_id", "region_id", "name"], registry=registry,
    )
    regions.has_many(
        "stores", stores,
        local_columns=("org_id", "region_id"), remote_columns=("org_id", "region_id"),
    )

    from grafast_py.pg.steps import PgSelectAllStep

    sdl = """
    type Query { regions: [Region!]! }
    type Region { label: String! stores: [Store!]! }
    type Store { id: Int! name: String! }
    """

    def leaf(key):
        def plan(parent, args, info):
            return access(parent, (key,))

        return plan

    schema = make_grafast_schema(
        sdl,
        {
            "Query": {
                "regions": lambda p, a, i: PgSelectAllStep(
                    regions, order_by=["org_id", "region_id"]
                ).for_parent(p)
            },
            "Region": {
                "label": leaf("label"),
                "stores": lambda p, a, i: regions.related_many(p, "stores"),
            },
            "Store": {"id": leaf("id"), "name": leaf("name")},
        },
    )
    yield schema
    await dispose_engine()


@pytest.mark.asyncio
async def test_composite_key_relation_skips_but_identical(composite_schema):
    """A composite-FK hasMany is SKIPPED this wave; count unchanged, data byte-identical."""
    query = "{ regions { label stores { id name } } }"
    await assert_inlined_equivalent(composite_schema, query, expected_saving=0)


# ============================================================ SKIP: filtered child


@pytest_asyncio.fixture
async def filtered_schema():
    """Author.posts with a per-plan ``.where(title LIKE ...)`` filter; SKIPPED this wave."""
    await dispose_engine()
    await setup_demo_schema()

    registry = PgRegistry()
    authors = PgResource(
        "authors", DEMO_SCHEMA, "authors", AUTHORS_COLS, registry=registry
    )
    posts = PgResource(
        "posts", DEMO_SCHEMA, "posts", POSTS_COLS, registry=registry
    )
    authors.has_many("posts", posts, local_column="id", remote_column="author_id")

    from grafast_py.pg.steps import PgSelectAllStep

    sdl = """
    type Query { authors: [Author!]! }
    type Author { id: Int! posts: [Post!]! }
    type Post { id: Int! title: String! }
    """

    def leaf(key):
        def plan(parent, args, info):
            return access(parent, (key,))

        return plan

    def posts_plan(parent, args, info):
        step = authors.related_many(parent, "posts")
        # a host filter the LATERAL would have to reproduce — deferred, so it SKIPS.
        step.add_where(column("title").like("%author 1%"))
        return step

    schema = make_grafast_schema(
        sdl,
        {
            "Query": {
                "authors": lambda p, a, i: PgSelectAllStep(
                    authors, order_by=["id"]
                ).for_parent(p)
            },
            "Author": {"id": leaf("id"), "posts": posts_plan},
            "Post": {"id": leaf("id"), "title": leaf("title")},
        },
    )
    yield schema
    await dispose_engine()


@pytest.mark.asyncio
async def test_filtered_child_skips_but_identical(filtered_schema):
    """A filtered hasMany is SKIPPED this wave; count unchanged, data byte-identical."""
    query = "{ authors { id posts { id title } } }"
    batched, _bc, _ic = await assert_inlined_equivalent(
        filtered_schema, query, expected_saving=0
    )
    # the filter genuinely narrowed the rows (only author-1 posts survive the LIKE).
    assert all(
        "author 1" in p["title"]
        for a in batched.data["authors"]
        for p in a["posts"]
    )


# ============================================================ SKIP: paginated connection


@pytest.mark.asyncio
async def test_paginated_connection_skips_but_identical(demo_schema):
    """postsConnection(first:2) is a PgConnectionStep — always SKIPPED; data byte-identical."""
    query = """
    {
      authors {
        id
        postsConnection(first: 2) {
          totalCount
          edges { cursor node { id } }
          pageInfo { hasNextPage endCursor }
        }
      }
    }
    """
    await assert_inlined_equivalent(demo_schema, query, expected_saving=0)


@pytest.mark.asyncio
async def test_per_parent_paginated_hasmany_skips_but_identical():
    """A per-parent window-sliced hasMany (first:2) is SKIPPED this wave; data identical."""
    await dispose_engine()
    await setup_demo_schema()

    registry = PgRegistry()
    authors = PgResource(
        "authors", DEMO_SCHEMA, "authors", AUTHORS_COLS, registry=registry
    )
    posts = PgResource(
        "posts", DEMO_SCHEMA, "posts", POSTS_COLS, registry=registry
    )
    authors.has_many("posts", posts, local_column="id", remote_column="author_id")

    from grafast_py.pg.steps import PgSelectAllStep

    sdl = """
    type Query { authors: [Author!]! }
    type Author { id: Int! posts: [Post!]! }
    type Post { id: Int! }
    """

    def leaf(key):
        def plan(parent, args, info):
            return access(parent, (key,))

        return plan

    schema = make_grafast_schema(
        sdl,
        {
            "Query": {
                "authors": lambda p, a, i: PgSelectAllStep(
                    authors, order_by=["id"]
                ).for_parent(p)
            },
            "Author": {
                "id": leaf("id"),
                # a per-parent page slice (first:2) — the window-sliced path, not foldable.
                "posts": lambda p, a, i: authors.related_many(p, "posts", first=2),
            },
            "Post": {"id": leaf("id")},
        },
    )
    try:
        query = "{ authors { id posts { id } } }"
        await assert_inlined_equivalent(schema, query, expected_saving=0)
    finally:
        await dispose_engine()
