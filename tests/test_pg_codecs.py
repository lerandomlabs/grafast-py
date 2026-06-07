"""The codec registry: PgCodec lookup by Postgres type + recursive containers.

The codec registry (:mod:`grafast_py.pg.codecs`) names the
:class:`~grafast_py.pg.resource.PgCodec` for a Postgres type by NAME, recursing into the
three container kinds (ARRAY / RANGE / COMPOSITE) and validating ENUM labels. These tests
cover three concerns:

- REGISTRY (no DB): the scalar codecs, the recursive array/range/composite/enum
  constructors (including a composite of an array of a composite — the recursion goes all
  the way down), and the fail-loud paths (unknown type, enum label out of set, composite
  field-count mismatch).
- KEYSET CAST WIRING (no DB): a codec carrying an ``sql_type`` (a non-native scalar, or an
  array/range) seeds the resource ``column_types``, so the keyset comparator casts a
  text-origin cursor value back to that type — proving codecs keep the keyset path working
  for a non-native order column without a separate ``column_types`` declaration.
- DEDUP NEUTRALITY (no DB): a codec rides the POST-grouping decode and an ``sql_type`` is
  resource-static (keyed by ``qualified_table``), so a codec NEVER changes a step's
  ``peer_key`` / ``dedup_params`` — a resource WITH a codec and an identical one WITHOUT
  produce byte-identical dedup keys (decode is dedup-neutral).
- DECODE THROUGH EVERY PATH (DB, marked ``pg``): the array/range/enum/composite codecs
  decode correctly through a plain ``= ANY`` select, a window slice, and a Relay connection
  node over the dedicated ``codec_rows`` fixture.

DB tests are marked ``pg`` (deselectable ``-m 'not pg'``); they touch ONLY the
``grafast_demo`` schema of ``grafast_py_test`` and do NOT alter authors/posts/comments — the
container cases use the dedicated ``codec_rows`` fixture table (plus its schema-local ENUM /
composite types).
"""

from decimal import Decimal

import pytest
import pytest_asyncio
from sqlalchemy.dialects import postgresql

from grafast_py.core_steps import constant
from grafast_py.pg.codecs import (
    apply_to_py,
    array_codec,
    codec_for,
    composite_codec,
    enum_codec,
    range_codec,
    scalar_codecs,
)
from grafast_py.pg.connection import PgConnectionStep
from grafast_py.pg.cursor import keyset_where
from grafast_py.pg.engine import count_sql, dispose_engine, get_engine
from grafast_py.pg.executor import SQLAlchemyExecutor, pg_request_context
from grafast_py.pg.ordering import OrderTerm
from grafast_py.pg.resource import PgCodec, PgColumn, PgRegistry, PgResource
from grafast_py.pg.steps import PgSelectStep
from examples.seed import setup_codec_rows_table, setup_demo_schema


# ----------------------------------------------------------- registry: scalars (no DB)


def test_native_scalar_codec_is_pass_through():
    """A native scalar (``int4`` / ``text``) carries no ``to_py`` and no cast ``sql_type``.

    asyncpg already returns the right Python type and it binds directly, so the codec is a
    pure pass-through — only present in the registry as a building block for the container
    codecs.
    """
    for name in ("int4", "integer", "text", "varchar", "bool"):
        codec = codec_for(name)
        assert codec.to_py is None
        assert codec.sql_type is None


def test_non_native_scalar_codec_carries_cast_type():
    """A non-native scalar (``numeric`` / ``timestamptz``) carries the keyset cast type."""
    assert _render(codec_for("numeric").sql_type) == "NUMERIC"
    assert _render(codec_for("timestamptz").sql_type) == "TIMESTAMP WITH TIME ZONE"
    assert _render(codec_for("date").sql_type) == "DATE"


def test_unknown_scalar_type_fails_loud():
    """An unregistered Postgres type raises ``KeyError`` — never a silent pass-through.

    An unregistered type is a declaration gap to surface, not to paper over with an
    identity codec that would silently let an unknown value through.
    """
    with pytest.raises(KeyError, match="no codec registered"):
        codec_for("mystery_type")


def test_type_name_is_case_insensitive():
    """A type name resolves regardless of case (``TIMESTAMPTZ`` == ``timestamptz``)."""
    assert _render(codec_for("TIMESTAMPTZ").sql_type) == "TIMESTAMP WITH TIME ZONE"


def test_scalar_codecs_view_is_a_copy():
    """``scalar_codecs`` returns a copy — mutating it does not corrupt the registry."""
    view = dict(scalar_codecs())
    view.pop("int4")
    assert "int4" in scalar_codecs()


# ----------------------------------------------------------- registry: arrays (no DB)


def test_array_codec_maps_element_to_py():
    """An ARRAY codec maps the element ``to_py`` over each element (NULLs pass through)."""
    codec = array_codec(PgCodec(to_py=str.upper))
    assert codec.to_py(["a", "b", None, "c"]) == ["A", "B", None, "C"]


def test_array_codec_passes_null_container_through():
    """A NULL array column (a nullable text[] returning SQL NULL) decodes to None, not a crash.

    The element-transforming array codec must guard the whole container like its range/composite
    siblings; otherwise ``decode_row`` (which calls ``to_py`` directly, not via ``apply_to_py``)
    would iterate ``None`` and raise ``TypeError`` on every NULL array row.
    """
    codec = array_codec(PgCodec(to_py=str.upper))
    assert codec.to_py(None) is None


def test_array_of_native_element_is_pass_through_with_array_cast_type():
    """``numeric[]`` has no per-element transform but carries the ``NUMERIC[]`` cast type.

    asyncpg returns ``Decimal``s already, so there is no element ``to_py``; the array codec's
    only job is to carry ``ARRAY(NUMERIC)`` for the keyset cursor cast.
    """
    codec = codec_for("numeric[]")
    assert codec.to_py is None
    assert _render(codec.sql_type) == "NUMERIC[]"


def test_array_codec_recurses_through_suffix_and_oid_forms():
    """Both the ``name[]`` suffix and the ``_name`` array-OID form resolve to an array codec."""
    assert _render(codec_for("timestamptz[]").sql_type) == "TIMESTAMP WITH TIME ZONE[]"
    assert _render(codec_for("_timestamptz").sql_type) == "TIMESTAMP WITH TIME ZONE[]"


def test_nested_array_recurses():
    """A nested array (``int4[][]``) recurses to an array-of-array codec."""
    codec = codec_for("text[][]")
    # text has no to_py, so the nested array is a pass-through; the recursion still resolves.
    assert codec.to_py is None


# ----------------------------------------------------------- registry: ranges (no DB)


def test_range_codec_decodes_asyncpg_range_to_dict():
    """A RANGE codec flattens an ``asyncpg.Range`` into a plain bounds dict."""
    from asyncpg import Range

    codec = codec_for("int4range")
    decoded = codec.to_py(Range(2, 9))
    assert decoded == {
        "lower": 2,
        "upper": 9,
        "lower_inc": True,
        "upper_inc": False,
        "empty": False,
    }


def test_range_codec_applies_bound_to_py_and_carries_range_type():
    """A RANGE codec applies the bound codec's ``to_py`` and carries the range SQL type."""
    codec = range_codec(PgCodec(to_py=lambda v: v * 10))

    class FakeRange:
        lower, upper, lower_inc, upper_inc, isempty = 1, 2, True, False, False

    assert codec.to_py(FakeRange()) == {
        "lower": 10,
        "upper": 20,
        "lower_inc": True,
        "upper_inc": False,
        "empty": False,
    }
    assert _render(codec_for("tstzrange").sql_type) == "TSTZRANGE"


# ----------------------------------------------------------- registry: composite (no DB)


def test_composite_codec_zips_record_into_named_dict():
    """A COMPOSITE codec zips a positional record against named field codecs into a dict."""
    codec = codec_for("point", composite_fields=[("x", "int4"), ("y", "int4")])
    assert codec.to_py((3, 4)) == {"x": 3, "y": 4}


def test_composite_codec_recurses_all_the_way_down():
    """A composite OF an array OF a composite decodes recursively (the deep recursion case).

    ``outer`` has fields ``label`` (uppercasing scalar) and ``points`` (an array of a
    ``(x, y)`` composite); decoding must descend the whole tree.
    """
    inner = composite_codec([("x", PgCodec(to_py=lambda v: v + 1)), ("y", codec_for("int4"))])
    outer = composite_codec(
        [
            ("label", PgCodec(to_py=str.upper)),
            ("points", array_codec(inner)),
        ]
    )
    decoded = outer.to_py(("hello", [(1, 2), (3, 4)]))
    assert decoded == {
        "label": "HELLO",
        "points": [{"x": 2, "y": 2}, {"x": 4, "y": 4}],
    }


def test_composite_codec_field_count_mismatch_fails_loud():
    """A record with the wrong column count fails loud (it would otherwise misalign fields)."""
    codec = composite_codec([("x", codec_for("int4")), ("y", codec_for("int4"))])
    with pytest.raises(ValueError, match="expects 2 fields"):
        codec.to_py((1, 2, 3))


# ----------------------------------------------------------- registry: enum (no DB)


def test_enum_codec_passes_known_label_and_rejects_unknown():
    """An ENUM codec passes a declared label and fails loud on an undeclared one."""
    codec = codec_for("mood", enum_labels=["happy", "sad"])
    assert codec.to_py("happy") == "happy"
    with pytest.raises(ValueError, match="not in the declared label set"):
        codec.to_py("furious")


def test_enum_codec_passes_null_through():
    """A NULL enum column decodes to None — the absence of a value is not a drifted label.

    ``decode_row`` calls ``to_py`` directly (no ``apply_to_py`` short-circuit), so the None
    guard must live in the codec; otherwise every NULL enum row would fail loud with a
    misleading 'not in the declared label set' message.
    """
    codec = codec_for("mood", enum_labels=["happy", "sad"])
    assert codec.to_py(None) is None


def test_enum_codec_without_labels_is_pass_through():
    """An ENUM codec with no declared labels is a pure pass-through (text flows through)."""
    codec = enum_codec()
    assert codec.to_py is None


def test_apply_to_py_handles_none_and_missing_hook():
    """``apply_to_py`` is a no-op for a NULL value or a hook-free codec."""
    assert apply_to_py(PgCodec(to_py=str.upper), None) is None
    assert apply_to_py(PgCodec(), "x") == "x"
    assert apply_to_py(None, "x") == "x"


# ------------------------------------------------ NULL container through decode_row (no DB)


def test_decode_row_passes_null_container_columns_through():
    """``decode_row`` survives NULL array/enum container columns (the direct-call path).

    ``decode_row`` calls ``codec.to_py`` directly (it guards only ``name in out``, not the
    value), so a nullable array column WITH an element transform — or a nullable enum — must
    decode its NULL to None rather than crash the whole batched row materialisation. Mirrors
    the ``codec_rows`` fixture's ``tags``/``mood`` codec shapes with NULL values.
    """
    res = PgResource(
        "codec_rows",
        "grafast_demo",
        "codec_rows",
        [
            "id",
            PgColumn("tags", codec=array_codec(PgCodec(to_py=str.upper))),
            PgColumn("mood", codec=codec_for("mood", enum_labels=["happy", "sad"])),
        ],
        registry=PgRegistry(),
    )
    decoded = res.decode_row({"id": 1, "tags": None, "mood": None})
    assert decoded == {"id": 1, "tags": None, "mood": None}


# ------------------------------------------------ keyset CAST wiring (no DB)


def test_codec_sql_type_seeds_resource_column_types():
    """A codec carrying an ``sql_type`` seeds the resource ``column_types`` for the keyset CAST."""
    res = PgResource(
        "events",
        "grafast_demo",
        "events",
        [
            "id",
            PgColumn("created", codec=codec_for("timestamptz")),
            PgColumn("price", codec=codec_for("numeric")),
        ],
        registry=PgRegistry(),
    )
    assert _render(res.column_types["created"]) == "TIMESTAMP WITH TIME ZONE"
    assert _render(res.column_types["price"]) == "NUMERIC"


def test_explicit_column_types_override_codec_derived():
    """An EXPLICIT ``column_types`` entry overrides the codec-derived one (host's deliberate type)."""
    from sqlalchemy.types import Text

    res = PgResource(
        "events",
        "grafast_demo",
        "events",
        ["id", PgColumn("created", codec=codec_for("timestamptz"))],
        registry=PgRegistry(),
        column_types={"created": Text()},
    )
    assert _render(res.column_types["created"]) == "TEXT"


def test_codec_column_type_drives_keyset_cast():
    """The codec-derived ``column_types`` makes the keyset comparator CAST the cursor value.

    A text-origin cursor value over a codec-typed (timestamptz) column must be wrapped in a
    ``CAST(... AS TIMESTAMP WITH TIME ZONE)`` so Postgres coerces it — exactly the behaviour
    an explicit ``column_types`` gives, now derived from the attribute's codec.
    """
    res = PgResource(
        "events",
        "grafast_demo",
        "events",
        ["id", PgColumn("created", codec=codec_for("timestamptz"))],
        registry=PgRegistry(),
    )
    terms = [OrderTerm("created"), OrderTerm("id")]
    pred = keyset_where(
        terms, ["2024-06-01T12:00:00+00:00", 5], after=True, column_types=res.column_types
    )
    sql = str(
        pred.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
    )
    assert "CAST('2024-06-01T12:00:00+00:00' AS TIMESTAMP WITH TIME ZONE)" in sql


# ------------------------------------------------ dedup neutrality (no DB)
# A codec rides the POST-grouping decode and its sql_type is resource-static (keyed by
# qualified_table), so it NEVER enters a step's dedup key — a resource WITH a codec and an
# identical one WITHOUT must produce byte-identical dedup keys. This is the codec half of the
# dedup-correctness invariant: decode is dedup-neutral (it changes no emitted SQL skeleton), so
# folding a codec into the key would only over-discriminate. The DB-side keyset CAST is
# resource-static (the same qualified_table => the same column_types => the same SQL), so it is
# already covered by qualified_table and needs no extra discriminator.


def plain_and_coded_resources():
    """A pair of ``events`` resources over the SAME table — one plain, one codec-bearing."""
    plain = PgResource(
        "events", "grafast_demo", "events", ["id", "owner_id", "created"],
        registry=PgRegistry(),
    )
    coded = PgResource(
        "events", "grafast_demo", "events",
        ["id", "owner_id", PgColumn("created", codec=codec_for("timestamptz"))],
        registry=PgRegistry(),
    )
    return plain, coded


def test_codec_with_sql_type_is_dedup_neutral_for_select():
    """A codec carrying an ``sql_type`` does NOT change a ``PgSelectStep`` dedup key.

    The codec only adds a ``column_types`` entry (a resource-static keyset concern) and a
    post-grouping decode; the emitted plain-select SQL is byte-identical, so the two steps
    MUST stay peers.
    """
    plain, coded = plain_and_coded_resources()
    a = PgSelectStep(plain, constant(None), "owner_id", order_by=["id"])
    b = PgSelectStep(coded, constant(None), "owner_id", order_by=["id"])
    assert a.peer_key == b.peer_key
    assert a.dedup_params() == b.dedup_params()


def test_to_py_codec_is_dedup_neutral_for_select():
    """A value-transforming ``to_py`` codec is ALSO dedup-neutral (decode is post-grouping).

    The codec changes ``has_decoders`` (so it is genuinely present) but not the emitted SQL,
    so the dedup key must be unchanged.
    """
    plain = PgResource(
        "events", "grafast_demo", "events", ["id", "owner_id", "label"],
        registry=PgRegistry(),
    )
    coded = PgResource(
        "events", "grafast_demo", "events",
        ["id", "owner_id", PgColumn("label", codec=PgCodec(to_py=str.upper))],
        registry=PgRegistry(),
    )
    assert plain.has_decoders is False and coded.has_decoders is True
    a = PgSelectStep(plain, constant(None), "owner_id", order_by=["id"])
    b = PgSelectStep(coded, constant(None), "owner_id", order_by=["id"])
    assert a.dedup_params() == b.dedup_params()


def test_codec_is_dedup_neutral_for_connection():
    """A codec does NOT change a ``PgConnectionStep`` dedup key either.

    The connection's keyset CAST is resource-static (driven by ``qualified_table`` ->
    ``column_types``), so a codec-typed connection and a plain one over the same table, order
    and page bounds stay peers — the cursor VALUES (which DO discriminate) are unchanged.
    """
    plain, coded = plain_and_coded_resources()
    a = PgConnectionStep(plain, constant(None), "owner_id", order_by=["created", "id"], first=2)
    b = PgConnectionStep(coded, constant(None), "owner_id", order_by=["created", "id"], first=2)
    assert a.peer_key == b.peer_key
    assert a.dedup_params() == b.dedup_params()


def test_different_filter_value_still_discriminates_with_codec_present():
    """The codec is neutral, but a different host WHERE VALUE still discriminates (both directions).

    Proves the codec does not SWALLOW the existing value-discriminated dedup: with a codec on
    the resource, two selects that differ only by a filter value must NOT be peers, while two
    with the identical filter MUST be peers.
    """
    from sqlalchemy import column

    _plain, coded = plain_and_coded_resources()
    published = PgSelectStep(coded, constant(None), "owner_id", order_by=["id"])
    published.add_where(column("owner_id") == 1)
    draft = PgSelectStep(coded, constant(None), "owner_id", order_by=["id"])
    draft.add_where(column("owner_id") == 2)
    same = PgSelectStep(coded, constant(None), "owner_id", order_by=["id"])
    same.add_where(column("owner_id") == 1)

    # different VALUE => NOT peers
    assert published.dedup_params() != draft.dedup_params()
    # identical VALUE => peers
    assert published.dedup_params() == same.dedup_params()


# ------------------------------------------------------- DB: decode through every path


CODEC_COLUMNS = ["id", "owner_id", "tags", "scores", "span", "period", "mood", "point"]


def codec_rows_resource() -> PgResource:
    """A ``codec_rows`` resource wiring a codec onto each container column via the registry.

    Each column derives its codec by Postgres type name: ``tags`` (``text[]`` with an
    uppercasing element codec), ``scores`` (``numeric[]``), ``span`` (``int4range``),
    ``period`` (``tstzrange``), ``mood`` (an enum), ``point`` (a composite). The array/range
    codecs seed ``column_types`` for the keyset path.
    """
    return PgResource(
        "codec_rows",
        "grafast_demo",
        "codec_rows",
        [
            "id",
            "owner_id",
            PgColumn("tags", codec=array_codec(PgCodec(to_py=str.upper))),
            PgColumn("scores", codec=codec_for("numeric[]")),
            PgColumn("span", codec=codec_for("int4range")),
            PgColumn("period", codec=codec_for("tstzrange")),
            PgColumn("mood", codec=codec_for("mood", enum_labels=["happy", "sad", "meh"])),
            PgColumn(
                "point",
                codec=codec_for("point", composite_fields=[("x", "int4"), ("y", "int4")]),
            ),
        ],
        registry=PgRegistry(),
    )


@pytest_asyncio.fixture
async def codec_seeded():
    """(Re)seed ``grafast_demo`` + the ``codec_rows`` fixture (fresh engine per test)."""
    await dispose_engine()
    await setup_demo_schema()
    await setup_codec_rows_table()
    yield
    await dispose_engine()


pytestmark_db = pytest.mark.pg


@pytest.mark.pg
@pytest.mark.asyncio
async def test_container_codecs_decode_plain_select(codec_seeded):
    """Every container codec decodes through a plain ``= ANY`` select in ONE statement."""
    res = codec_rows_resource()
    step = PgSelectStep(res, constant(None), "owner_id", order_by=["id"])
    engine = get_engine()
    with count_sql(engine) as counter:
        with pg_request_context(SQLAlchemyExecutor(engine)):
            out = await step.execute(1, [[1]])
    assert counter.count == 1
    rows = {r["id"]: r for r in out[0]}
    # ARRAY (text[]) — element codec uppercased each tag.
    assert rows[1]["tags"] == ["ALPHA", "BETA"]
    assert rows[2]["tags"] == ["GAMMA"]
    # ARRAY (numeric[]) — pass-through Decimals (no element transform, only the cast type).
    assert rows[1]["scores"] == [Decimal("1.50"), Decimal("2.25")]
    # RANGE (int4range) — decoded to a bounds dict.
    assert rows[1]["span"] == {
        "lower": 1, "upper": 5, "lower_inc": True, "upper_inc": False, "empty": False,
    }
    # ENUM — validated label passes through.
    assert rows[1]["mood"] == "happy"
    # COMPOSITE — record zipped into a named dict.
    assert rows[1]["point"] == {"x": 3, "y": 4}
    assert rows[3]["point"] == {"x": 7, "y": 8}


@pytest.mark.pg
@pytest.mark.asyncio
async def test_container_codecs_decode_window_slice(codec_seeded):
    """Every container codec survives a per-parent window slice (``first=2``) in ONE statement."""
    res = codec_rows_resource()
    step = PgSelectStep(res, constant(None), "owner_id", order_by=["id"], first=2)
    assert step.is_limited is True
    engine = get_engine()
    with count_sql(engine) as counter:
        with pg_request_context(SQLAlchemyExecutor(engine)):
            out = await step.execute(1, [[1]])
    assert counter.count == 1
    rows = out[0]
    # owner 1, first=2 -> ids 1,2 with decoded containers.
    assert [r["id"] for r in rows] == [1, 2]
    assert rows[0]["tags"] == ["ALPHA", "BETA"]
    assert rows[1]["span"] == {
        "lower": 10, "upper": 20, "lower_inc": True, "upper_inc": False, "empty": False,
    }


@pytest.mark.pg
@pytest.mark.asyncio
async def test_container_codecs_decode_connection_node(codec_seeded):
    """Every container codec decodes inside a Relay connection node (one page statement)."""
    res = codec_rows_resource()
    step = PgConnectionStep(res, constant(None), "owner_id", order_by=["id"], first=2)
    engine = get_engine()
    with count_sql(engine) as counter:
        with pg_request_context(SQLAlchemyExecutor(engine)):
            out = await step.execute(1, [[1]])
    assert counter.count == 1  # no totalCount selected -> just the page query
    nodes = out[0]["nodes"]
    assert [n["id"] for n in nodes] == [1, 2]
    assert nodes[0]["tags"] == ["ALPHA", "BETA"]
    assert nodes[0]["mood"] == "happy"
    assert nodes[1]["point"] == {"x": 5, "y": 6}


@pytest.mark.pg
@pytest.mark.asyncio
async def test_enum_codec_rejects_drifted_label_through_select(codec_seeded):
    """An ENUM codec whose declared set MISSES a stored label fails loud on read.

    A label present in the DB but absent from the declared set is schema/migration drift; the
    codec must surface it (a ``ValueError`` out of the decode) rather than pass an unknown
    string downstream.
    """
    res = PgResource(
        "codec_rows",
        "grafast_demo",
        "codec_rows",
        # 'meh' is a real stored label (row 3) deliberately OMITTED from the declared set.
        ["id", "owner_id", PgColumn("mood", codec=enum_codec(["happy", "sad"]))],
        registry=PgRegistry(),
    )
    step = PgSelectStep(res, constant(None), "owner_id", order_by=["id"])
    with pytest.raises(ValueError, match="not in the declared label set"):
        with pg_request_context(SQLAlchemyExecutor(get_engine())):
            await step.execute(1, [[1]])


@pytest.mark.pg
@pytest.mark.asyncio
async def test_codec_typed_column_keyset_paging_round_trips(codec_seeded):
    """A keyset connection ordered by a codec-typed (tstzrange/period) row still pages cleanly.

    The connection orders by a NATIVE column (``id``) but the resource carries codec-derived
    ``column_types`` for ``period`` / ``scores`` / ``span``; the page walk must be unaffected
    (the codec column_types only matter when that column is an order key, and they must not
    perturb a native-ordered walk). Two ``first:2`` pages over owner 1 (ids 1,2,3) are
    [1,2] then [3].
    """
    res = codec_rows_resource()
    engine = get_engine()
    with pg_request_context(SQLAlchemyExecutor(engine)):
        page1 = PgConnectionStep(res, constant(None), "owner_id", order_by=["id"], first=2)
        out1 = await page1.execute(1, [[1]])
        cursor = out1[0]["edges"][-1]["cursor"]

        page2 = PgConnectionStep(
            res, constant(None), "owner_id", order_by=["id"], first=2, after=cursor
        )
        out2 = await page2.execute(1, [[1]])
    assert [n["id"] for n in out1[0]["nodes"]] == [1, 2]
    assert [n["id"] for n in out2[0]["nodes"]] == [3]


# ----------------------------------------------------------------------- helper


def _render(sql_type) -> str:
    """Render a SQLAlchemy type to its Postgres DDL string (for type-identity assertions)."""
    return str(sql_type.compile(dialect=postgresql.dialect()))
