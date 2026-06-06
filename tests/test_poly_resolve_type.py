"""Foundation (no-DB) tests for Postgres polymorphism: the resolve_type bridge.

Postgres polymorphism in grafast-py is HOST WIRING, not a new core-engine path. A
Postgres-backed interface/union resolves each value's concrete type via a
``resolve_type`` bridge — :func:`resolve_type_from_discriminator` for the single-table
``{discriminator_column: typename}`` shape, :func:`resolve_type_from_tag` for the
pgUnionAll ``{__type tag column}`` shape — attached to the abstract type by
:func:`attach_type_resolvers`. The EXISTING completion-time abstract dispatch
(``completion.resolve_abstract_bucket`` reads ``abstract_type.resolve_type or
context.type_resolver``, then ``dispatch_abstract`` groups values by concrete type and
plans/executes each group's sub-selection like a normal object field) does the rest with
ZERO core-engine change.

These tests prove that contract WITHOUT a database:

- the two bridge builders map/raise correctly and serve both shapes (single-table
  discriminator and union tag);
- :func:`attach_type_resolvers` wires onto INTERFACE and UNION abstract types and fails
  loud (``KeyError``) on unknown / non-abstract type names, mirroring ``attach_plans``;
- the ``make_grafast_schema(type_resolvers=...)`` param is additive (omitting it leaves
  existing callers unchanged);
- the bridge actually drives completion-time dispatch end-to-end over a plain in-memory
  list (constant-step plans, no SQL) for an interface AND a union, and an unenumerated
  discriminator surfaces as a located field error rather than a silent drop — confirming
  the composition holds through the real engine.

The DB-backed end-to-end tests (over ``grafast_demo.media``) and the example-schema
parity tests live in ``test_poly_schema.py``; this module is the bridge/wiring unit layer.
"""

import pytest
from graphql import (
    GraphQLInterfaceType,
    GraphQLObjectType,
    GraphQLScalarType,
    GraphQLUnionType,
    graphql,
)

import grafast_py
from grafast_py import GrafastExecutionContext
from grafast_py.core_steps import access, constant
from grafast_py.schema import (
    attach_type_resolvers,
    make_grafast_schema,
    resolve_type_from_discriminator,
    resolve_type_from_tag,
)

# A self-contained SDL carrying BOTH abstract kinds over the SAME concrete object types:
# ``Media`` (interface, single-table discriminator shape) and ``SearchResult`` (union,
# pgUnionAll tag shape). Object types Image/Video back both, so one plan map serves both.
SDL = """
type Query {
  media: [Media!]!
  results: [SearchResult!]!
}

interface Media {
  id: Int!
}

type Image implements Media {
  id: Int!
  width: Int!
}

type Video implements Media {
  id: Int!
  durationSeconds: Int!
}

union SearchResult = Image | Video
"""


# ---------------------------------------------------- bridge builders: map + fail loud


def test_discriminator_bridge_maps_column_value_to_typename():
    """The single-table shape: row[discriminator] -> the mapped concrete type name."""
    bridge = resolve_type_from_discriminator(
        "kind", {"image": "Image", "video": "Video"}
    )
    assert bridge({"kind": "image"}, None, None) == "Image"
    assert bridge({"kind": "video"}, None, None) == "Video"


def test_discriminator_bridge_unmapped_value_fails_loud():
    """An unenumerated discriminator value is a wiring bug: raise, do not return None.

    Returning ``None`` would bubble a generic "must resolve to an Object type" error far
    from the cause; the bridge instead names the offending value AND the column.
    """
    bridge = resolve_type_from_discriminator("kind", {"image": "Image"})
    with pytest.raises(KeyError) as exc:
        bridge({"kind": "audio"}, None, None)
    assert "audio" in str(exc.value)
    assert "kind" in str(exc.value)


def test_discriminator_bridge_missing_column_raises_keyerror():
    """A row lacking the discriminator column is also a loud failure (KeyError on the key)."""
    bridge = resolve_type_from_discriminator("kind", {"image": "Image"})
    with pytest.raises(KeyError):
        bridge({"id": 1}, None, None)


def test_tag_bridge_reads_typename_directly_off_column():
    """The union shape: the member branch already tagged the row with its own typename."""
    bridge = resolve_type_from_tag("__typename")
    assert bridge({"__typename": "Image"}, None, None) == "Image"
    assert bridge({"__typename": "Video"}, None, None) == "Video"


def test_tag_bridge_honours_a_custom_column_name():
    """The tag column is configurable — the union SQL may label it anything."""
    bridge = resolve_type_from_tag("__t")
    assert bridge({"__t": "Video"}, None, None) == "Video"


def test_tag_bridge_missing_column_raises_keyerror():
    bridge = resolve_type_from_tag("__typename")
    with pytest.raises(KeyError):
        bridge({"id": 1}, None, None)


# ---------------------------------------- attach_type_resolvers: interface AND union


def test_attach_type_resolvers_wires_onto_interface_resolve_type():
    schema = make_grafast_schema(SDL)
    media = schema.type_map["Media"]
    assert isinstance(media, GraphQLInterfaceType)
    assert media.resolve_type is None

    bridge = resolve_type_from_discriminator("kind", {"image": "Image", "video": "Video"})
    attach_type_resolvers(schema, {"Media": bridge})
    assert media.resolve_type is bridge
    # the wired callable resolves through the interface's possible types
    assert media.resolve_type({"kind": "video"}, None, media) == "Video"


def test_attach_type_resolvers_wires_onto_union_resolve_type():
    """The union abstract kind is wired identically — the tag bridge reads the typename."""
    schema = make_grafast_schema(SDL)
    results = schema.type_map["SearchResult"]
    assert isinstance(results, GraphQLUnionType)
    assert results.resolve_type is None

    bridge = resolve_type_from_tag("__typename")
    attach_type_resolvers(schema, {"SearchResult": bridge})
    assert results.resolve_type is bridge
    assert results.resolve_type({"__typename": "Image"}, None, results) == "Image"


def test_attach_type_resolvers_wires_both_kinds_in_one_pass():
    schema = make_grafast_schema(SDL)
    attach_type_resolvers(
        schema,
        {
            "Media": resolve_type_from_discriminator(
                "kind", {"image": "Image", "video": "Video"}
            ),
            "SearchResult": resolve_type_from_tag("__typename"),
        },
    )
    assert schema.type_map["Media"].resolve_type is not None
    assert schema.type_map["SearchResult"].resolve_type is not None


def test_attach_type_resolvers_unknown_type_fails_loud():
    """A name absent from the schema is a typo: raise, mirroring attach_plans."""
    schema = make_grafast_schema(SDL)
    with pytest.raises(KeyError) as exc:
        attach_type_resolvers(schema, {"Nope": resolve_type_from_tag("x")})
    assert "not an interface or union" in str(exc.value)


def test_attach_type_resolvers_object_type_fails_loud():
    """An OBJECT type name is concrete, not abstract — wiring it is a typo; fail loud."""
    schema = make_grafast_schema(SDL)
    assert isinstance(schema.type_map["Image"], GraphQLObjectType)
    with pytest.raises(KeyError) as exc:
        attach_type_resolvers(schema, {"Image": resolve_type_from_tag("x")})
    assert "not an interface or union" in str(exc.value)


def test_attach_type_resolvers_scalar_type_fails_loud():
    """A scalar (non-composite) type name is equally a wiring error."""
    schema = make_grafast_schema(SDL)
    assert isinstance(schema.type_map["Int"], GraphQLScalarType)
    with pytest.raises(KeyError):
        attach_type_resolvers(schema, {"Int": resolve_type_from_tag("x")})


# ------------------------------------- make_grafast_schema(type_resolvers=...) param


def test_make_grafast_schema_type_resolvers_param_attaches_interface_bridge():
    bridge = resolve_type_from_discriminator(
        "kind", {"image": "Image", "video": "Video"}
    )
    schema = make_grafast_schema(SDL, type_resolvers={"Media": bridge})
    assert schema.type_map["Media"].resolve_type is bridge


def test_make_grafast_schema_type_resolvers_param_attaches_union_bridge():
    bridge = resolve_type_from_tag("__typename")
    schema = make_grafast_schema(SDL, type_resolvers={"SearchResult": bridge})
    assert schema.type_map["SearchResult"].resolve_type is bridge


def test_make_grafast_schema_without_type_resolvers_is_unchanged():
    """The new param is additive: omitting it leaves resolve_type unset (existing callers)."""
    schema = make_grafast_schema(SDL)
    assert schema.type_map["Media"].resolve_type is None
    assert schema.type_map["SearchResult"].resolve_type is None


def test_make_grafast_schema_attaches_plans_and_resolvers_together():
    """Plans and type_resolvers compose in one build call without interfering."""

    def leaf(key):
        def plan(parent_step, args, info):
            return access(parent_step, (key,))

        return plan

    schema = make_grafast_schema(
        SDL,
        plans={"Image": {"id": leaf("id"), "width": leaf("width")}},
        type_resolvers={"Media": resolve_type_from_tag("__typename")},
    )
    image = schema.type_map["Image"]
    assert image.fields["id"].extensions["grafast"]["plan"] is not None
    assert schema.type_map["Media"].resolve_type is not None


# ------------------------------------------------------------- top-level exports


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


# ----------------------------------- end-to-end composition through the real engine

# A plain in-memory list (no SQL): each row carries BOTH a ``kind`` discriminator (for the
# Media interface) and a ``__t`` tag (for the SearchResult union) plus each concrete type's
# own columns. The plans are pure constant/access steps, so this proves the bridge drives
# completion-time abstract dispatch with NO Postgres and NO core-engine change.
ROWS = [
    {"kind": "image", "__t": "Image", "id": 1, "width": 1920},
    {"kind": "video", "__t": "Video", "id": 2, "duration_seconds": 42},
    {"kind": "image", "__t": "Image", "id": 3, "width": 512},
]


def build_inmemory_poly_schema(type_resolvers):
    """A no-DB poly schema: constant-list roots + access-projection concrete plans.

    ``type_resolvers`` is supplied per test so the same schema shape exercises the
    discriminator bridge (Media) and/or the tag bridge (SearchResult), including the
    fail-loud case with a deliberately incomplete mapping.
    """

    def leaf(key):
        def plan(parent_step, args, info):
            return access(parent_step, (key,))

        return plan

    def list_root(parent_step, args, info):
        return constant(ROWS)

    plans = {
        "Query": {"media": list_root, "results": list_root},
        "Image": {"id": leaf("id"), "width": leaf("width")},
        "Video": {"id": leaf("id"), "durationSeconds": leaf("duration_seconds")},
    }
    return make_grafast_schema(SDL, plans, type_resolvers)


async def run(schema, query):
    return await graphql(
        schema, query, execution_context_class=GrafastExecutionContext
    )


@pytest.mark.asyncio
async def test_discriminator_bridge_drives_interface_dispatch_no_db():
    """The discriminator bridge groups in-memory rows by concrete type through the engine."""
    schema = build_inmemory_poly_schema(
        {
            "Media": resolve_type_from_discriminator(
                "kind", {"image": "Image", "video": "Video"}
            )
        }
    )
    result = await run(
        schema,
        """
        {
          media {
            __typename
            id
            ... on Image { width }
            ... on Video { durationSeconds }
          }
        }
        """,
    )
    assert result.errors is None
    rows = result.data["media"]
    assert [r["__typename"] for r in rows] == ["Image", "Video", "Image"]
    # each concrete-type group reads ONLY its own columns
    assert rows[0] == {"__typename": "Image", "id": 1, "width": 1920}
    assert rows[1] == {"__typename": "Video", "id": 2, "durationSeconds": 42}
    assert "durationSeconds" not in rows[0]
    assert "width" not in rows[1]


@pytest.mark.asyncio
async def test_tag_bridge_drives_union_dispatch_no_db():
    """The tag bridge resolves a UNION's members directly off the row tag through the engine."""
    schema = build_inmemory_poly_schema(
        {"SearchResult": resolve_type_from_tag("__t")}
    )
    result = await run(
        schema,
        """
        {
          results {
            __typename
            ... on Image { id width }
            ... on Video { id durationSeconds }
          }
        }
        """,
    )
    assert result.errors is None
    rows = result.data["results"]
    assert [r["__typename"] for r in rows] == ["Image", "Video", "Image"]
    assert rows[0] == {"__typename": "Image", "id": 1, "width": 1920}
    assert rows[1] == {"__typename": "Video", "id": 2, "durationSeconds": 42}


@pytest.mark.asyncio
async def test_unmapped_discriminator_fails_loud_through_engine_no_db():
    """A row whose discriminator is outside the mapping surfaces a located error, not a drop.

    Wiring an incomplete map (no ``'video'``) makes the video row's resolve_type raise; the
    completion engine records it as a per-value field error (graphql-core's abstract-
    resolution conformance) rather than silently dropping the row — the CDC "never silently
    skip" rule expressed through the engine.
    """
    schema = build_inmemory_poly_schema(
        {"Media": resolve_type_from_discriminator("kind", {"image": "Image"})}
    )
    result = await run(schema, "{ media { __typename id } }")
    assert result.errors is not None
    # the error names the offending discriminator value and is located on the bad item
    assert any("video" in str(e) for e in result.errors)
    assert any(e.path == ["media", 1] for e in result.errors)


@pytest.mark.asyncio
async def test_context_type_resolver_fallback_drives_dispatch_no_db():
    """With no per-type bridge, completion falls back to ``context.type_resolver``.

    ``resolve_abstract_bucket`` reads ``abstract_type.resolve_type OR context.type_resolver``;
    passing the bridge as graphql-core's ``type_resolver`` (and attaching NONE on the type)
    must dispatch identically — confirming the bridge is a plain ``(value, info, abstract)``
    callable usable on either surface.
    """
    schema = build_inmemory_poly_schema(type_resolvers=None)
    assert schema.type_map["Media"].resolve_type is None

    bridge = resolve_type_from_discriminator(
        "kind", {"image": "Image", "video": "Video"}
    )
    result = await graphql(
        schema,
        "{ media { __typename id ... on Image { width } ... on Video { durationSeconds } } }",
        type_resolver=bridge,
        execution_context_class=GrafastExecutionContext,
    )
    assert result.errors is None
    assert [r["__typename"] for r in result.data["media"]] == [
        "Image",
        "Video",
        "Image",
    ]


# ----------------------------------------------------------- dedup-correctness note
#
# The DEDUP CORRECTNESS INVARIANT (a new SQL-affecting pg-step input must fold into
# peer_key + dedup_params, with a no-DB test proving both directions) applies PER NEW STEP.
# This foundation feature introduces NO new pg step: the resolve_type bridge is pure host
# wiring that reads dict keys off an already-produced row and carries ZERO SQL — the single-
# table shape rides the existing PgSelectAllStep (whose dedup over table/order/customization
# is covered in test_pg_datasource / test_dag_dedup). The cross-table PgUnionAllStep and its
# branch-set/per-branch-where/discriminator dedup keying belong to the pgUnionAll feature,
# which ships that step's dedup-correctness test alongside it. There is nothing SQL-affecting
# here to fold, so there is no foundation-level dedup test to write — recorded explicitly so
# the invariant is satisfied by reasoning, not silently skipped.
