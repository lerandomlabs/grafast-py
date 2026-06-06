"""Attribute descriptors, computed columns (pgClassExpression), and codecs.

These prove the enriched resource model:

- BACKWARD COMPAT: a ``PgResource`` built from BARE STRINGS keeps its ``columns`` name
  list unchanged (the from_sqlalchemy parity gate + every existing select path rely on
  this), and the descriptor map mirrors it.
- COMPUTED COLUMNS: a host-authored Core SQL expression (``lower(title)`` / ``upper(code)``
  over the TABLE columns — never request data) is projected as an extra labelled column in
  the SAME batched SELECT. Each layer stays ONE statement (computed columns are extra
  projected columns, never extra statements), and the value is correct through a plain
  select, a window slice (``first:N``), AND a connection node.
- CODEC DECODE: an attribute with a ``to_py`` hook decodes its value in EVERY
  row-materialisation path — plain select, windowed select, connection node.

DB tests are marked ``pg`` (deselectable ``-m 'not pg'``); they touch ONLY the
``grafast_demo`` schema of ``grafast_py_test`` and do NOT alter authors/posts/comments —
the codec/computed cases use the dedicated ``labels`` fixture table (computed-on-posts is
exercised over a fresh ``posts`` resource without changing the demo resource).
"""

import pytest
import pytest_asyncio
from graphql import graphql
from sqlalchemy import column, func

from grafast_py.context import GrafastExecutionContext
from grafast_py.core_steps import access, constant
from grafast_py.schema import make_grafast_schema
from grafast_py.pg.connection import PgConnectionStep
from grafast_py.pg.engine import count_sql, dispose_engine, get_engine
from grafast_py.pg.executor import SQLAlchemyExecutor, pg_request_context
from grafast_py.pg.resource import PgCodec, PgColumn, PgRegistry, PgResource
from grafast_py.pg.steps import PgSelectAllStep, PgSelectSingleStep, PgSelectStep
from examples.seed import setup_demo_schema, setup_labels_table

pytestmark = pytest.mark.pg


async def run(schema, query, variables=None):
    """Run a query through our engine over the convenience engine for the request."""
    with pg_request_context(SQLAlchemyExecutor(get_engine())):
        return await graphql(
            schema,
            query,
            variable_values=variables,
            execution_context_class=GrafastExecutionContext,
        )


# ----------------------------------------------------------------- backward compat


def test_bare_string_columns_unchanged():
    """A bare-string ``columns`` list still yields the SAME ``columns`` name list.

    The descriptor map mirrors it (one PgColumn per name, none computed), so existing
    build_query paths and the from_sqlalchemy parity test are untouched.
    """
    posts = PgResource(
        "posts", "grafast_demo", "posts", ["id", "author_id", "title"],
        registry=PgRegistry(),
    )
    assert posts.columns == ["id", "author_id", "title"]
    assert posts.computed == []
    assert list(posts.attributes) == ["id", "author_id", "title"]
    assert all(not a.is_computed for a in posts.attributes.values())


def test_duplicate_attribute_name_raises_loudly():
    """A duplicate attribute name is rejected loudly (the dict would silently collapse it).

    The attributes map is keyed by name, so a repeated column would overwrite the first
    and drop it; that is a declaration bug, so it must fail with a clear error.
    """
    with pytest.raises(ValueError, match="more than once"):
        PgResource(
            "posts", "grafast_demo", "posts", ["id", "title", "id"],
            registry=PgRegistry(),
        )
    # a duplicate that mixes a bare string and a PgColumn of the same name is still a dup.
    with pytest.raises(ValueError, match="more than once"):
        PgResource(
            "posts", "grafast_demo", "posts",
            ["id", PgColumn("title"), "title"],
            registry=PgRegistry(),
        )


def test_computed_attribute_excluded_from_columns():
    """A computed attribute lands in ``attributes``/``computed`` but NOT in ``columns``."""
    slug = PgColumn("slug", expression=func.lower(column("title")))
    posts = PgResource(
        "posts", "grafast_demo", "posts", ["id", "author_id", "title", slug],
        registry=PgRegistry(),
    )
    # the STORED (selectable) columns are exactly the table columns — slug is NOT here.
    assert posts.columns == ["id", "author_id", "title"]
    assert posts.computed == ["slug"]
    assert "slug" in posts.attributes


def test_order_by_computed_column_raises_plain_select():
    """Ordering a plain select by a COMPUTED column raises a clear error, not raw SQL.

    A computed attribute is only a SELECT-list alias, so Postgres cannot reference it in
    ``row_number() OVER (ORDER BY ...)`` or a keyset comparator — the guard converts that
    raw DB error into a clear, column-named ValueError at order-set time.
    """
    posts = posts_with_slug()
    with pytest.raises(ValueError, match="cannot order by computed column 'slug'"):
        PgSelectStep(posts, constant(None), "author_id", order_by=["slug"])
    # the same guard fires when the order term is appended after construction.
    step = PgSelectStep(posts, constant(None), "author_id", order_by=["id"])
    with pytest.raises(ValueError, match="cannot order by computed column 'slug'"):
        step.add_order_term("slug")


def test_order_by_computed_column_raises_connection():
    """Ordering a Relay connection by a COMPUTED column raises the same clear error."""
    posts = posts_with_slug()
    with pytest.raises(ValueError, match="cannot order by computed column 'slug'"):
        PgConnectionStep(posts, constant(None), "author_id", order_by=["slug"], first=2)


def test_order_by_stored_column_with_computed_present_is_fine():
    """A resource WITH a computed column still orders fine by a STORED column."""
    posts = posts_with_slug()
    step = PgSelectStep(posts, constant(None), "author_id", order_by=["title"])
    assert step.order_by[0].column == "title"
    # the computed slug is still projected (build the SQL to confirm).
    assert "lower(title)" in str(step.build_query()).lower()


def test_computed_expression_is_host_authored_core_not_args():
    """The computed expression composes via the RESOURCE (Core SQL), never request args.

    It is built at resource-declaration time from a Core ``ColumnElement`` over identifier
    columns; there is no plan/args input to the projection, so it is no injection surface.
    """
    slug = PgColumn("slug", expression=func.lower(column("title")))
    posts = PgResource(
        "posts", "grafast_demo", "posts", ["id", "title", slug], registry=PgRegistry()
    )
    projections = posts.computed_projections()
    assert len(projections) == 1
    assert projections[0].name == "slug"
    assert str(projections[0]) == "lower(title)"


# ------------------------------------------------------------------- fixtures


@pytest_asyncio.fixture
async def labels_seeded():
    """(Re)seed ``grafast_demo`` + the ``labels`` fixture (fresh engine per test)."""
    await dispose_engine()
    await setup_demo_schema()
    await setup_labels_table()
    yield
    await dispose_engine()


def posts_with_slug() -> PgResource:
    """A fresh ``posts`` resource with a computed ``slug = lower(title)`` (own registry)."""
    return PgResource(
        "posts", "grafast_demo", "posts",
        ["id", "author_id", "title", PgColumn("slug", expression=func.lower(column("title")))],
        registry=PgRegistry(),
    )


# uppercasing decode hook for the `code` text column; a computed `upper(code)` mirrors it
# IN SQL so the codec path and the computed path can be compared.
UPPER_CODEC = PgCodec(to_py=str.upper)


def labels_resource() -> PgResource:
    """A ``labels`` resource: ``code`` carries an uppercasing codec + a computed ``loud``."""
    return PgResource(
        "labels", "grafast_demo", "labels",
        [
            "id",
            "owner_id",
            PgColumn("code", codec=UPPER_CODEC),
            PgColumn("loud", expression=func.upper(column("code"))),
        ],
        registry=PgRegistry(),
    )


# ----------------------------------------------------- computed: plain select (one stmt)


@pytest.mark.asyncio
async def test_computed_column_plain_select_one_statement(labels_seeded):
    """A computed ``slug`` returns correctly via a plain ``= ANY`` select in ONE statement."""
    posts = posts_with_slug()
    step = PgSelectStep(posts, constant(None), "author_id", order_by=["id"])
    engine = get_engine()
    with count_sql(engine) as counter:
        with pg_request_context(SQLAlchemyExecutor(engine)):
            out = await step.execute(1, [[1]])
    assert counter.count == 1  # computed col is an extra projection, NOT an extra statement
    # author 1 owns posts 1,2; slug is lower(title) computed in SQL.
    assert [r["id"] for r in out[0]] == [1, 2]
    for row in out[0]:
        assert row["slug"] == row["title"].lower()


@pytest.mark.asyncio
async def test_computed_column_select_all_one_statement(labels_seeded):
    """A computed ``slug`` returns via the root ``PgSelectAllStep`` in ONE statement."""
    posts = posts_with_slug()
    engine = get_engine()
    with count_sql(engine) as counter:
        with pg_request_context(SQLAlchemyExecutor(engine)):
            step = PgSelectAllStep(posts, order_by=["id"]).for_parent(constant(None))
            out = await step.execute(1, [[None]])
    assert counter.count == 1
    rows = out[0]
    assert rows[0]["slug"] == rows[0]["title"].lower()


# ------------------------------------------- computed: window slice (first:N), one stmt


@pytest.mark.asyncio
async def test_computed_column_window_slice_one_statement(labels_seeded):
    """A computed ``slug`` survives the per-parent window slice (``first=1``) in ONE stmt.

    The computed expression references TABLE columns, so it is evaluated in the INNER
    (table-scope) select and projected through the subquery — the row still carries it.
    """
    posts = posts_with_slug()
    step = PgSelectStep(
        posts, constant(None), "author_id", order_by=["id"], first=1
    )
    assert step.is_limited is True
    # the slice is IN SQL and the computed label rides through the subquery.
    sql = str(step.build_query())
    assert "row_number" in sql.lower()
    assert "lower(title)" in sql.lower()

    engine = get_engine()
    with count_sql(engine) as counter:
        with pg_request_context(SQLAlchemyExecutor(engine)):
            out = await step.execute(2, [[1, 2]])
    assert counter.count == 1
    # first=1 caps each parent at one row; slug still correct.
    assert len(out[0]) == 1 and len(out[1]) == 1
    assert out[0][0]["slug"] == out[0][0]["title"].lower()
    assert out[1][0]["slug"] == out[1][0]["title"].lower()


# ---------------------------------------------------- computed: connection node, one stmt


@pytest.mark.asyncio
async def test_computed_column_in_connection_node(labels_seeded):
    """A computed ``slug`` rides into a Relay connection node (one page statement)."""
    posts = posts_with_slug()
    step = PgConnectionStep(
        posts, constant(None), "author_id", order_by=["id"], first=2
    )
    sql = str(step.build_page_query())
    assert "lower(title)" in sql.lower()

    engine = get_engine()
    with count_sql(engine) as counter:
        with pg_request_context(SQLAlchemyExecutor(engine)):
            out = await step.execute(1, [[3]])
    # no totalCount selected -> just the page query = 1 statement.
    assert counter.count == 1
    nodes = out[0]["nodes"]
    assert len(nodes) == 2
    for node in nodes:
        assert node["slug"] == node["title"].lower()


# --------------------------------------------------------- codec decode: every path


@pytest.mark.asyncio
async def test_codec_decode_plain_select(labels_seeded):
    """A ``to_py`` codec uppercases ``code`` through a plain ``= ANY`` select."""
    labels = labels_resource()
    step = PgSelectStep(labels, constant(None), "owner_id", order_by=["id"])
    with pg_request_context(SQLAlchemyExecutor(get_engine())):
        out = await step.execute(1, [[1]])
    rows = out[0]
    # owner 1 owns codes alpha..delta; the codec uppercases each on read.
    assert [r["code"] for r in rows] == ["ALPHA", "BRAVO", "CHARLIE", "DELTA"]
    # the computed `loud` (upper(code) IN SQL) matches the decoded value.
    assert [r["loud"] for r in rows] == [r["code"] for r in rows]


@pytest.mark.asyncio
async def test_codec_decode_select_all(labels_seeded):
    """A ``to_py`` codec uppercases ``code`` through the root ``PgSelectAllStep``."""
    labels = labels_resource()
    with pg_request_context(SQLAlchemyExecutor(get_engine())):
        step = PgSelectAllStep(labels, order_by=["id"]).for_parent(constant(None))
        out = await step.execute(1, [[None]])
    assert {r["code"] for r in out[0]} == {
        "ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO"
    }


@pytest.mark.asyncio
async def test_codec_decode_window_slice(labels_seeded):
    """A ``to_py`` codec uppercases ``code`` through a window slice (``first=2``)."""
    labels = labels_resource()
    step = PgSelectStep(labels, constant(None), "owner_id", order_by=["id"], first=2)
    with pg_request_context(SQLAlchemyExecutor(get_engine())):
        out = await step.execute(1, [[1]])
    # owner 1, first=2 -> alpha, bravo, uppercased.
    assert [r["code"] for r in out[0]] == ["ALPHA", "BRAVO"]


@pytest.mark.asyncio
async def test_codec_decode_connection_node(labels_seeded):
    """A ``to_py`` codec uppercases ``code`` inside a Relay connection node."""
    labels = labels_resource()
    step = PgConnectionStep(
        labels, constant(None), "owner_id", order_by=["id"], first=2
    )
    with pg_request_context(SQLAlchemyExecutor(get_engine())):
        out = await step.execute(1, [[1]])
    nodes = out[0]["nodes"]
    assert [n["code"] for n in nodes] == ["ALPHA", "BRAVO"]
    # the computed `loud` and the decoded `code` agree inside the node.
    assert [n["loud"] for n in nodes] == [n["code"] for n in nodes]


# ------------------------------------------------------- end-to-end GraphQL (node path)


def build_labels_schema() -> "object":
    """A GraphQL schema exposing ``labels`` with the codec ``code`` + computed ``loud``.

    ``Query.labels`` is a root collection; ``Query.labelsConnection(first)`` a Relay
    connection — both project ``code`` (decoded) and ``loud`` (computed in SQL), so a real
    GraphQL query observes the codec + computed value end to end.
    """
    sdl = """
    type Query {
      labels: [Label!]!
      labelsConnection(first: Int): LabelConnection!
    }
    type Label { id: Int!  code: String!  loud: String! }
    type LabelConnection {
      nodes: [Label!]!
      edges: [LabelEdge!]!
    }
    type LabelEdge { cursor: String!  node: Label! }
    """
    labels = labels_resource()

    def plan_labels(parent_step, args, info):
        return PgSelectAllStep(labels, order_by=["id"]).for_parent(parent_step)

    def plan_labels_connection(parent_step, args, info):
        return PgConnectionStep(
            labels, constant(1), "owner_id", order_by=["id"],
            first=args.get("first"),
        )

    def leaf(key):
        def plan(parent_step, args, info):
            return access(parent_step, (key,))

        return plan

    return make_grafast_schema(
        sdl,
        {
            "Query": {
                "labels": plan_labels,
                "labelsConnection": plan_labels_connection,
            },
            "Label": {"id": leaf("id"), "code": leaf("code"), "loud": leaf("loud")},
            "LabelConnection": {"nodes": leaf("nodes"), "edges": leaf("edges")},
            "LabelEdge": {"cursor": leaf("cursor"), "node": leaf("node")},
        },
    )


@pytest.mark.asyncio
async def test_graphql_codec_and_computed_end_to_end(labels_seeded):
    """An end-to-end GraphQL query reflects the decoded codec AND the computed column."""
    schema = build_labels_schema()
    result = await run(schema, "{ labels { id code loud } }")
    assert result.errors is None
    by_id = {r["id"]: r for r in result.data["labels"]}
    assert by_id[1]["code"] == "ALPHA"  # decoded on read
    assert by_id[1]["loud"] == "ALPHA"  # computed upper(code) in SQL
    assert by_id[5]["code"] == "ECHO"


@pytest.mark.asyncio
async def test_graphql_connection_codec_and_computed(labels_seeded):
    """The connection node reflects codec + computed through a real GraphQL query."""
    schema = build_labels_schema()
    query = """
    { labelsConnection(first: 2) { nodes { id code loud } edges { node { code } } } }
    """
    result = await run(schema, query)
    assert result.errors is None
    nodes = result.data["labelsConnection"]["nodes"]
    assert [n["code"] for n in nodes] == ["ALPHA", "BRAVO"]
    assert [n["loud"] for n in nodes] == ["ALPHA", "BRAVO"]


# ----------------------------- codec ON THE MATCH COLUMN (group raw, decode output)
# Regression for the silent-skip fix: a codec on the lookup/match column must NOT
# misgroup rows. The key step supplies the RAW match value, so rows are grouped on the
# raw `owner_id` BEFORE the codec runs; if grouping happened AFTER decode the rows would
# group under the decoded value and scatter to nobody (an empty result). See
# steps.group_and_decode / run_query.

# a codec that CHANGES the match value (x -> x*1000) so a group-after-decode bug would be
# unmistakable: the raw key 1 would never match the decoded owner_id 1000.
MATCH_CODEC = PgCodec(to_py=lambda v: v * 1000)


def labels_match_codec_resource() -> PgResource:
    """A ``labels`` resource whose MATCH column ``owner_id`` carries a value-changing codec."""
    return PgResource(
        "labels", "grafast_demo", "labels",
        ["id", PgColumn("owner_id", codec=MATCH_CODEC), "code"],
        registry=PgRegistry(),
    )


@pytest.mark.asyncio
async def test_codec_on_match_column_groups_raw_decodes_output_select(labels_seeded):
    """PgSelectStep: a codec on the MATCH column groups on the RAW value, decodes output.

    owner 1 owns 4 labels (ids 1..4). The key step supplies the RAW owner_id (1); rows
    must group under that raw value (non-empty group) and the OUTPUT owner_id is decoded
    (1 * 1000 = 1000). A regression to group-after-decode would scatter to nobody -> [].
    """
    labels = labels_match_codec_resource()
    step = PgSelectStep(labels, constant(None), "owner_id", order_by=["id"])
    with pg_request_context(SQLAlchemyExecutor(get_engine())):
        out = await step.execute(1, [[1]])
    rows = out[0]
    # the group is NON-EMPTY — the raw key 1 matched (would be [] under a group-after-
    # decode regression, since rows would group under decoded 1000).
    assert [r["id"] for r in rows] == [1, 2, 3, 4]
    # every output owner_id is the DECODED value.
    assert {r["owner_id"] for r in rows} == {1000}


@pytest.mark.asyncio
async def test_codec_on_match_column_groups_raw_decodes_output_single(labels_seeded):
    """PgSelectSingleStep: a codec on the MATCH column groups RAW, decodes the single row.

    Looking up the single label by its PK ``id`` would not exercise the match codec, so
    here the match column IS the codec-bearing ``owner_id`` and owner 2 owns exactly one
    label (id 5). The raw key 2 must match (non-empty), the returned owner_id decoded
    (2000). A group-after-decode regression returns None.
    """
    labels = labels_match_codec_resource()
    step = PgSelectSingleStep(labels, constant(None), "owner_id")
    with pg_request_context(SQLAlchemyExecutor(get_engine())):
        out = await step.execute(1, [[2]])
    row = out[0]
    assert row is not None  # raw key 2 matched (would be None under the regression)
    assert row["id"] == 5
    assert row["owner_id"] == 2000  # decoded on read
