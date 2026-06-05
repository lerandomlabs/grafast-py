"""DB-backed foundation tests for single-table Postgres polymorphism.

The single-table shape (the ``foundation`` feature of the polymorphism wave) is one
``grafast_demo.media`` table carrying a ``kind`` discriminator (``'image'`` / ``'video'``):
every row is some ``Media`` (interface) — an ``Image`` or a ``Video`` — and a
``resolve_type_from_discriminator`` bridge keyed on ``kind`` resolves each row's concrete
type at COMPLETION time. This composes with the EXISTING completion-time abstract dispatch
(``completion.dispatch_abstract`` groups values by concrete type and plans + executes each
group's sub-selection like a normal object field) with ZERO core-engine change — there is
no plan-time polymorphism bucket system.

These tests drive the real engine against the live ``grafast_py_test`` scratch DB
(``grafast_demo`` schema only) and prove the three contracts of the single-table shape:

1. an interface resolves each ROW to the right concrete type — for the LIST shape
   (``media`` / ``mediaByOwner``) AND the single (nullable ``get_single``) shape
   (``mediaById``);
2. type-conditioned inline fragments select PER-TYPE columns — an ``... on Image`` reads
   width/height, an ``... on Video`` reads durationSeconds, and (because the seed leaves
   the off-type columns NULL) a misgroup would surface as a wrong/NULL value, so correct
   per-type projection is observable, not merely structural;
3. a nested pg relation under a concrete type (``Image.tags`` / ``Video.tags``) batches
   PER concrete-type group — O(type-groups × depth) statements via ``count_sql``, never
   O(rows): adding more rows of the SAME types must NOT add statements.

The no-DB bridge/wiring unit layer and the broader example-schema parity tests live in
``test_poly_resolve_type.py`` / ``test_poly_schema.py``; this module is the single-table
DB foundation suite. There is no new pg step here (the shape rides the existing
``PgSelectAllStep`` / ``get_single`` / ``find``), so the DEDUP CORRECTNESS INVARIANT has
nothing SQL-affecting to fold at this layer — the cross-table ``PgUnionAllStep`` ships its
own dedup-correctness test in the pgUnionAll feature.
"""

import pytest
import pytest_asyncio
from graphql import graphql

from grafast_py.context import GrafastExecutionContext
from grafast_py.pg.engine import count_sql, dispose_engine, get_engine
from grafast_py.pg.executor import SQLAlchemyExecutor, pg_request_context
from examples.poly_schema import build_poly_schema
from examples.seed import setup_demo_schema, setup_media_table, setup_media_tags_table

pytestmark = [pytest.mark.pg, pytest.mark.asyncio]


@pytest_asyncio.fixture
async def poly_schema():
    """Build the single-table polymorphism schema after (re)seeding media + media_tags.

    Function-scoped so each test runs on its own event loop with a fresh engine (the async
    pool is bound to the creating loop). ``setup_demo_schema`` (re)creates the schema; the
    media fixtures add their tables independently of the authors/posts/comments fixtures.
    """
    await dispose_engine()
    await setup_demo_schema()
    await setup_media_table()
    await setup_media_tags_table()
    schema = build_poly_schema()
    yield schema
    await dispose_engine()


async def run(schema, query, variables=None):
    """Run a query through GrafastExecutionContext with a request-scoped pg executor."""
    with pg_request_context(SQLAlchemyExecutor(get_engine())):
        return await graphql(
            schema,
            query,
            variable_values=variables,
            execution_context_class=GrafastExecutionContext,
        )


# ------------------------------------------------- 1. per-row concrete type (list + single)


async def test_interface_list_resolves_concrete_type_per_row(poly_schema):
    """A ``[Media!]!`` list resolves EACH row's concrete type off the ``kind`` discriminator.

    The single ``media`` select returns mixed image/video rows; completion-time dispatch
    groups them by ``kind`` and ``__typename`` comes from the bridge — id-ordered, the
    seed's interleaving (image, video, image, image, video) is preserved.
    """
    result = await run(
        poly_schema,
        "{ media { __typename id title } }",
    )
    assert result.errors is None
    rows = result.data["media"]
    assert [r["__typename"] for r in rows] == [
        "Image",
        "Video",
        "Image",
        "Image",
        "Video",
    ]
    # the shared interface field (title) projects identically regardless of concrete type,
    # in id order, so the per-row dispatch did not reorder or drop any row.
    assert [r["id"] for r in rows] == [1, 2, 3, 4, 5]
    assert [r["title"] for r in rows] == [
        "sunrise",
        "timelapse",
        "portrait",
        "logo",
        "promo",
    ]


async def test_single_interface_field_resolves_concrete_type(poly_schema):
    """A single (nullable ``get_single``) ``Media`` field resolves the row's concrete type.

    The single shape feeds the SAME completion-time dispatch as the list: ``mediaById(2)`` is
    a video, so its ``__typename`` is ``Video``; ``mediaById(1)`` is an image.
    """
    video = await run(poly_schema, "{ mediaById(id: 2) { __typename id title } }")
    assert video.errors is None
    assert video.data["mediaById"] == {
        "__typename": "Video",
        "id": 2,
        "title": "timelapse",
    }

    image = await run(poly_schema, "{ mediaById(id: 1) { __typename id title } }")
    assert image.errors is None
    assert image.data["mediaById"] == {
        "__typename": "Image",
        "id": 1,
        "title": "sunrise",
    }


async def test_single_interface_field_missing_id_is_null(poly_schema):
    """A missing id completes the nullable ``Media`` field to null without hitting the bridge.

    The ``None`` abstract value short-circuits in completion (set to ``None`` before
    ``resolve_type`` is ever called), so a non-existent id is a clean null, not a bridge
    error — the bridge never sees a row that is not there.
    """
    result = await run(poly_schema, "{ mediaById(id: 999) { __typename id } }")
    assert result.errors is None
    assert result.data["mediaById"] is None


async def test_key_matched_root_resolves_concrete_type_per_row(poly_schema):
    """A key-matched root list (``mediaByOwner``, a batched ``find``) dispatches per row.

    Owner 2 owns image 4 (logo) and video 5 (promo); the key-matched select feeds the same
    completion-time dispatch as the all-rows list, resolving each row independently.
    """
    result = await run(
        poly_schema,
        "{ mediaByOwner(ownerId: 2) { __typename id title } }",
    )
    assert result.errors is None
    rows = result.data["mediaByOwner"]
    assert [r["__typename"] for r in rows] == ["Image", "Video"]
    assert [r["title"] for r in rows] == ["logo", "promo"]


# --------------------------------------- 2. type-conditioned inline fragments per-type columns


async def test_inline_fragments_select_per_type_columns(poly_schema):
    """Inline fragments read PER-TYPE columns; the off-type columns never leak in.

    The seed populates width/height only on images (NULL on videos) and duration_seconds
    only on videos (NULL on images). So if a video row were misgrouped as an Image, its
    ``width`` would read NULL — observable. Each concrete group reading only its own columns
    is the correctness contract here, asserted both ways (present on-type, absent off-type).
    """
    result = await run(
        poly_schema,
        """
        {
          media {
            __typename
            id
            title
            ... on Image { width height }
            ... on Video { durationSeconds }
          }
        }
        """,
    )
    assert result.errors is None
    rows = result.data["media"]
    by_id = {r["id"]: r for r in rows}

    # image rows expose width/height (NON-NULL — proving the right group's columns) and the
    # video-only field is absent from the response shape entirely.
    sunrise = by_id[1]
    assert sunrise["__typename"] == "Image"
    assert sunrise["width"] == 1920
    assert sunrise["height"] == 1080
    assert "durationSeconds" not in sunrise

    # video rows expose durationSeconds and carry NEITHER image column in the shape.
    timelapse = by_id[2]
    assert timelapse["__typename"] == "Video"
    assert timelapse["durationSeconds"] == 42
    assert "width" not in timelapse
    assert "height" not in timelapse

    # a second image (different owner) confirms the projection is per-row, not positional.
    logo = by_id[4]
    assert logo["__typename"] == "Image"
    assert logo["width"] == 512 and logo["height"] == 512


async def test_inline_fragment_columns_are_one_statement(poly_schema):
    """Pure per-type column projection over one source select issues exactly ONE statement.

    The two concrete-type groups share the single ``media`` select; their sub-selections are
    leaf column reads off the already-fetched row (no nested relation), so the whole
    polymorphic list — both groups — costs a SINGLE SQL statement, independent of how the
    rows split across the two types.
    """
    with count_sql(get_engine()) as counter:
        result = await run(
            poly_schema,
            """
            {
              media {
                __typename
                ... on Image { width height }
                ... on Video { durationSeconds }
              }
            }
            """,
        )
    assert result.errors is None
    assert counter.count == 1


async def test_single_field_inline_fragment_selects_per_type_column(poly_schema):
    """The single (``get_single``) shape also projects only the resolved type's columns.

    ``mediaById(2)`` is a video: ``durationSeconds`` reads and the ``... on Image`` width is
    not in scope, so the off-type column does not appear under the single field either.
    """
    result = await run(
        poly_schema,
        """
        {
          mediaById(id: 2) {
            __typename
            title
            ... on Image { width }
            ... on Video { durationSeconds }
          }
        }
        """,
    )
    assert result.errors is None
    assert result.data["mediaById"] == {
        "__typename": "Video",
        "title": "timelapse",
        "durationSeconds": 42,
    }


# ---------------------------------- 3. nested relation batches per concrete-type group (O(depth))


async def test_nested_relation_batches_per_type_group(poly_schema):
    """``Image.tags`` / ``Video.tags`` batch PER concrete-type group — O(type-groups × depth).

    ``tags`` is a hasMany pg relation selected on BOTH concrete types. Completion-time
    dispatch plans each concrete-type group like a normal object field, so each group issues
    ONE batched ``tags`` statement across all of its rows: media (1) + Image.tags (1, across
    every image) + Video.tags (1, across every video) = 3 statements. That is O(depth) per
    type-group, never O(rows) — the per-group batching that the whole feature rests on.
    """
    query = """
    {
      media {
        __typename
        ... on Image { id tags { label } }
        ... on Video { id tags { label } }
      }
    }
    """
    with count_sql(get_engine()) as counter:
        result = await run(poly_schema, query)

    assert result.errors is None
    assert counter.count == 3

    rows = result.data["media"]
    by_id = {r["id"]: r for r in rows}
    # image 1 -> nature+morning, image 3 -> people, image 4 -> none; video 2 -> motion.
    assert sorted(t["label"] for t in by_id[1]["tags"]) == ["morning", "nature"]
    assert [t["label"] for t in by_id[3]["tags"]] == ["people"]
    assert by_id[4]["tags"] == []
    assert [t["label"] for t in by_id[2]["tags"]] == ["motion"]


async def test_nested_relation_on_one_type_is_two_statements(poly_schema):
    """Selecting ``tags`` on a SINGLE concrete type issues exactly ONE relation statement.

    With ``tags`` selected only under ``... on Image``, the videos contribute no relation
    statement at all: media (1) + Image.tags (1) = 2. So the statement count tracks the
    number of TYPE-GROUPS that select the relation (× depth), not the number of concrete
    types in the result, and certainly not the number of rows.
    """
    query = """
    {
      media {
        __typename
        ... on Image { id tags { label } }
        ... on Video { id }
      }
    }
    """
    with count_sql(get_engine()) as counter:
        result = await run(poly_schema, query)

    assert result.errors is None
    assert counter.count == 2

    rows = result.data["media"]
    by_id = {r["id"]: r for r in rows}
    assert sorted(t["label"] for t in by_id[1]["tags"]) == ["morning", "nature"]
    # the video rows carry no tags key (the relation was not selected on Video)
    assert "tags" not in by_id[2]


async def test_relation_statement_count_is_independent_of_row_count(poly_schema):
    """The relation statement count is O(type-groups × depth) — adding same-type rows adds none.

    Owner 1 owns MORE media (2 images + 1 video) than owner 2 (1 image + 1 video). Both the
    larger and smaller key-matched lists resolve the ``tags`` relation in the SAME statement
    count (source select + Image.tags + Video.tags = 3), proving the batching is per
    concrete-type GROUP, not per row: more rows of the same types cost no extra statements.
    """
    query = """
    query ($owner: Int!) {
      mediaByOwner(ownerId: $owner) {
        __typename
        ... on Image { id tags { label } }
        ... on Video { id tags { label } }
      }
    }
    """
    with count_sql(get_engine()) as big:
        big_result = await run(poly_schema, query, {"owner": 1})
    with count_sql(get_engine()) as small:
        small_result = await run(poly_schema, query, {"owner": 2})

    assert big_result.errors is None
    assert small_result.errors is None
    # owner 1: 3 media (2 images, 1 video); owner 2: 2 media (1 image, 1 video). Both groups
    # are non-empty in each list, so each costs source + Image.tags + Video.tags = 3 — the
    # count is identical despite owner 1 having an extra image row.
    assert big.count == 3
    assert small.count == 3
    assert len(big_result.data["mediaByOwner"]) == 3
    assert len(small_result.data["mediaByOwner"]) == 2
