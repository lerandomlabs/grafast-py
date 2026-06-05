"""Postgres polymorphism: the resolve_type bridge + completion-time abstract dispatch.

The foundation feature is host wiring, NOT a new core engine path: a Postgres-backed
interface/union resolves each row's concrete type via a ``resolve_type`` bridge
(``resolve_type_from_discriminator`` / ``resolve_type_from_tag``) attached to the abstract
type. The EXISTING completion-time abstract dispatch (``completion.dispatch_abstract``)
then groups rows by concrete type and plans + executes each group's sub-selection like a
normal object field — so a concrete type's nested pg relation batches per concrete-type
group automatically, with NO plan-time polymorphism and NO core-engine change.

Two test layers:

- no-DB unit tests for the bridge builders + ``attach_type_resolvers`` wiring + the
  ``make_grafast_schema(type_resolvers=...)`` param + the top-level exports. These need no
  database and prove the fail-loud contracts.
- DB-backed (``pg``) end-to-end tests over the single ``grafast_demo.media`` table (a
  ``kind`` discriminator) proving the interface query resolves concrete types per row,
  type-specific fields read the right columns, the nested ``tags`` relation batches per
  concrete-type group (O(depth) statement count), and a bogus discriminator fails loud
  through the real engine (graphql-core conformance preserved).
"""

import pytest
import pytest_asyncio
from graphql import GraphQLInterfaceType, graphql

import grafast_py
from grafast_py.schema import (
    attach_type_resolvers,
    make_grafast_schema,
    resolve_type_from_discriminator,
    resolve_type_from_tag,
)
from examples.poly_schema import SDL, build_poly_schema


# ----------------------------------------------------------------- no-DB unit tests


def test_discriminator_bridge_maps_column_value_to_typename():
    bridge = resolve_type_from_discriminator("kind", {"image": "Image", "video": "Video"})
    assert bridge({"kind": "image"}, None, None) == "Image"
    assert bridge({"kind": "video"}, None, None) == "Video"


def test_discriminator_bridge_unmapped_value_fails_loud():
    """An unenumerated discriminator value is a wiring bug: raise, do not return None."""
    bridge = resolve_type_from_discriminator("kind", {"image": "Image"})
    with pytest.raises(KeyError) as exc:
        bridge({"kind": "audio"}, None, None)
    # the message names the offending value and the column so the wiring gap is obvious
    assert "audio" in str(exc.value)
    assert "kind" in str(exc.value)


def test_tag_bridge_reads_typename_directly_off_column():
    bridge = resolve_type_from_tag("__typename")
    assert bridge({"__typename": "Image"}, None, None) == "Image"
    assert bridge({"__typename": "Video"}, None, None) == "Video"


def test_attach_type_resolvers_wires_onto_abstract_resolve_type():
    schema = make_grafast_schema(SDL)
    media = schema.type_map["Media"]
    assert isinstance(media, GraphQLInterfaceType)
    assert media.resolve_type is None

    bridge = resolve_type_from_discriminator("kind", {"image": "Image", "video": "Video"})
    attach_type_resolvers(schema, {"Media": bridge})
    assert media.resolve_type is bridge
    assert media.resolve_type({"kind": "video"}, None, media) == "Video"


def test_attach_type_resolvers_unknown_type_fails_loud():
    schema = make_grafast_schema(SDL)
    with pytest.raises(KeyError) as exc:
        attach_type_resolvers(schema, {"Nope": resolve_type_from_tag("x")})
    assert "not an interface or union" in str(exc.value)


def test_attach_type_resolvers_object_type_fails_loud():
    """An OBJECT type name is not abstract — wiring it is a typo; fail loud like attach_plans."""
    schema = make_grafast_schema(SDL)
    with pytest.raises(KeyError) as exc:
        attach_type_resolvers(schema, {"Image": resolve_type_from_tag("x")})
    assert "not an interface or union" in str(exc.value)


def test_make_grafast_schema_type_resolvers_param_attaches_bridge():
    bridge = resolve_type_from_tag("__typename")
    schema = make_grafast_schema(SDL, type_resolvers={"Media": bridge})
    assert schema.type_map["Media"].resolve_type is bridge


def test_make_grafast_schema_without_type_resolvers_is_unchanged():
    """The new param is additive: omitting it leaves resolve_type unset (existing callers)."""
    schema = make_grafast_schema(SDL)
    assert schema.type_map["Media"].resolve_type is None


def test_bridges_exported_from_top_level():
    """The foundation symbols are importable from the package root (and in __all__)."""
    assert grafast_py.resolve_type_from_discriminator is resolve_type_from_discriminator
    assert grafast_py.resolve_type_from_tag is resolve_type_from_tag
    assert grafast_py.attach_type_resolvers is attach_type_resolvers
    for name in (
        "TypeResolver",
        "resolve_type_from_discriminator",
        "resolve_type_from_tag",
        "attach_type_resolvers",
    ):
        assert name in grafast_py.__all__


def test_build_poly_schema_attaches_discriminator_bridge():
    schema = build_poly_schema()
    media = schema.type_map["Media"]
    assert media.resolve_type is not None
    # the bridge resolves both concrete types off the kind column
    assert media.resolve_type({"kind": "image"}, None, media) == "Image"
    assert media.resolve_type({"kind": "video"}, None, media) == "Video"


def test_abstract_child_subtree_deduplicates():
    """A concrete-type sub-selection's step DAG is deduplicated, both directions (no DB).

    The completion-time abstract dispatch plans each concrete-type group as a self-contained
    subtree with its own ``Plan`` + ``RootStep`` and runs ``Plan.deduplicate()`` on it. Two
    fields reading the SAME column off the row must collapse to one step; two reading
    DISTINCT columns must stay separate. We exercise that exact path via the public
    plan/dedup machinery (no Postgres): build a tiny DAG over a RootStep the way
    ``abstract_child_plan`` does and assert through ``dag.Plan.deduplicate()``.
    """
    from grafast_py.core_steps import RootStep, access
    from grafast_py.dag import Plan

    plan = Plan()
    root = RootStep()
    plan.add_step(root)
    # two reads of the SAME column (e.g. an Image's `title` selected under two aliases) and
    # one of a DISTINCT column — the shape an abstract child sub-selection produces.
    title_a = plan.add_step(access(root, ("title",)))
    title_b = plan.add_step(access(root, ("title",)))
    width = plan.add_step(access(root, ("width",)))

    remap = plan.deduplicate()
    # identical column reads merge to one survivor; the distinct one stays separate.
    assert remap[title_a.id] is remap[title_b.id]
    assert remap[title_a.id] is not remap[width.id]


# ----------------------------------------------------------------- DB-backed tests

pg = pytest.mark.pg


@pytest_asyncio.fixture
async def poly_schema():
    """Build the polymorphism schema after (re)seeding ``media`` + ``media_tags``.

    Function-scoped so each test runs on its own event loop with a fresh engine (the async
    pool is bound to the creating loop). ``setup_demo_schema`` creates the schema; the media
    fixtures add their tables independently of the authors/posts/comments parity fixtures.
    """
    from grafast_py.pg.engine import dispose_engine
    from examples.seed import (
        setup_demo_schema,
        setup_media_table,
        setup_media_tags_table,
    )

    await dispose_engine()
    await setup_demo_schema()
    await setup_media_table()
    await setup_media_tags_table()
    schema = build_poly_schema()
    yield schema
    await dispose_engine()


async def run(schema, query, variables=None):
    """Run a query through GrafastExecutionContext with a request-scoped pg executor."""
    from grafast_py.context import GrafastExecutionContext
    from grafast_py.pg.engine import get_engine
    from grafast_py.pg.executor import SQLAlchemyExecutor, pg_request_context

    with pg_request_context(SQLAlchemyExecutor(get_engine())):
        return await graphql(
            schema,
            query,
            variable_values=variables,
            execution_context_class=GrafastExecutionContext,
        )


@pg
@pytest.mark.asyncio
async def test_interface_resolves_concrete_type_per_row(poly_schema):
    """A [Media!]! list with inline fragments resolves each row's concrete type.

    The single ``media`` select returns mixed image/video rows; completion-time dispatch
    groups them by the ``kind`` discriminator and reads each group's own columns —
    width/height for images, durationSeconds for videos. __typename comes from the bridge.
    """
    from grafast_py.pg.engine import count_sql, get_engine

    query = """
    {
      media {
        __typename
        id
        title
        ... on Image { width height }
        ... on Video { durationSeconds }
      }
    }
    """
    with count_sql(get_engine()) as counter:
        result = await run(poly_schema, query)

    assert result.errors is None
    # ONE statement for the whole media list (the two concrete-type groups share the source
    # select; the per-group sub-selection here is pure column projection, no extra SQL).
    assert counter.count == 1

    rows = result.data["media"]
    assert [r["__typename"] for r in rows] == [
        "Image",
        "Video",
        "Image",
        "Image",
        "Video",
    ]
    # image rows expose width/height and NOT durationSeconds
    sunrise = rows[0]
    assert sunrise["title"] == "sunrise"
    assert sunrise["width"] == 1920 and sunrise["height"] == 1080
    assert "durationSeconds" not in sunrise
    # video rows expose durationSeconds and NOT width/height
    timelapse = rows[1]
    assert timelapse["title"] == "timelapse"
    assert timelapse["durationSeconds"] == 42
    assert "width" not in timelapse


@pg
@pytest.mark.asyncio
async def test_nested_relation_under_concrete_type_batches_per_group(poly_schema):
    """An Image.tags hasMany relation chains off the completion-time-resolved concrete type.

    The nested ``tags`` pg relation issues ONE statement across ALL images in the bucket
    (and one across the videos), proving the per-concrete-type sub-selection batches like a
    normal object field — O(depth), not O(rows).
    """
    from grafast_py.pg.engine import count_sql, get_engine

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
    # media (1) + Image.tags (1, batched across all images) + Video.tags (1, across videos)
    # = 3 statements. The tags relation is selected on BOTH concrete types, so each
    # concrete-type group issues its own batched tags statement — still O(depth) per group,
    # never O(rows).
    assert counter.count == 3

    rows = result.data["media"]
    by_id = {r["id"]: r for r in rows}
    # image 1 has tags nature+morning, image 3 has people, image 4 has none
    assert sorted(t["label"] for t in by_id[1]["tags"]) == ["morning", "nature"]
    assert [t["label"] for t in by_id[3]["tags"]] == ["people"]
    assert by_id[4]["tags"] == []
    # video 2 has the motion tag
    assert [t["label"] for t in by_id[2]["tags"]] == ["motion"]


@pg
@pytest.mark.asyncio
async def test_single_interface_field_resolves_concrete_type(poly_schema):
    """A single (nullable) ``Media`` field (``get_single``) resolves the row's concrete type.

    The single shape feeds the same completion-time dispatch as the list: ``mediaById(2)`` is
    a video, so its ``__typename`` is ``Video`` and ``durationSeconds`` reads (width/height
    are not in scope under the ``... on Video`` fragment).
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


@pg
@pytest.mark.asyncio
async def test_single_interface_field_missing_id_is_null(poly_schema):
    """A missing id completes the nullable ``Media`` field to null without hitting the bridge.

    The ``None`` abstract value short-circuits in completion (it is set to ``None`` before
    ``resolve_type`` is ever called), so a non-existent id is a clean null, not a bridge error.
    """
    result = await run(poly_schema, "{ mediaById(id: 999) { __typename id } }")
    assert result.errors is None
    assert result.data["mediaById"] is None


@pg
@pytest.mark.asyncio
async def test_key_matched_root_resolves_polymorphic_rows(poly_schema):
    """A key-matched root select (mediaByOwner) feeds the same completion-time dispatch."""
    result = await run(
        poly_schema,
        """
        {
          mediaByOwner(ownerId: 2) {
            __typename
            title
            ... on Image { width }
            ... on Video { durationSeconds }
          }
        }
        """,
    )
    assert result.errors is None
    rows = result.data["mediaByOwner"]
    # owner 2 owns image 4 (logo) and video 5 (promo), id-ordered
    assert [r["__typename"] for r in rows] == ["Image", "Video"]
    assert rows[0] == {"__typename": "Image", "title": "logo", "width": 512}
    assert rows[1] == {
        "__typename": "Video",
        "title": "promo",
        "durationSeconds": 90,
    }


@pg
@pytest.mark.asyncio
async def test_bogus_discriminator_fails_loud_through_engine(poly_schema):
    """A row whose discriminator is outside the bridge mapping surfaces an error, not silent skip.

    Wiring the bridge with an incomplete mapping (no ``'video'`` entry) makes the video rows'
    resolve_type raise; the completion engine locates that as a field error (graphql-core's
    own abstract-resolution conformance), it does NOT silently drop the row.
    """
    from examples.poly_schema import build_registry

    # rebuild the schema with a deliberately incomplete discriminator map
    from grafast_py.schema import make_grafast_schema

    _registry, media, _tags = build_registry()
    schema = make_grafast_schema(
        SDL,
        plans=_poly_plans(media),
        type_resolvers={
            "Media": resolve_type_from_discriminator("kind", {"image": "Image"})
        },
    )
    result = await run(schema, "{ media { __typename id } }")

    # the video rows error out (their kind is unmapped); the engine reports it rather than
    # silently dropping them — at least one error mentioning the bad value.
    assert result.errors is not None
    assert any("video" in str(e) for e in result.errors)


def _poly_plans(media):
    """The Query+Image+Video+Tag plan map, factored so the fail-loud test can reuse it."""
    from grafast_py.core_steps import access
    from grafast_py.pg.steps import PgSelectAllStep

    def leaf(key):
        def plan(parent_step, args, info):
            return access(parent_step, (key,))

        return plan

    def plan_all(parent_step, args, info):
        return PgSelectAllStep(media, order_by=[media.primary_key]).for_parent(parent_step)

    def plan_single(parent_step, args, info):
        from grafast_py.core_steps import constant

        return media.get_single(constant(args.get("id")), media.primary_key)

    def plan_tags(parent_step, args, info):
        return media.related_many(parent_step, "tags")

    return {
        "Query": {
            "media": plan_all,
            "mediaById": plan_single,
            "mediaByOwner": plan_all,
        },
        "Image": {
            "id": leaf("id"),
            "title": leaf("title"),
            "width": leaf("width"),
            "height": leaf("height"),
            "tags": plan_tags,
        },
        "Video": {
            "id": leaf("id"),
            "title": leaf("title"),
            "durationSeconds": leaf("duration_seconds"),
            "tags": plan_tags,
        },
        "Tag": {"id": leaf("id"), "label": leaf("label")},
    }
