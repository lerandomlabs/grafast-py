"""KEYSET (seek) cursors, the keyset WHERE comparator, and the SQL-sliced
Relay connection with a SEPARATE batched totalCount aggregate.

The connection now slices each parent's page IN SQL (a keyset ``after``/``before``
predicate in the inner WHERE plus a ``row_number()`` window, fetched one-extra for
``hasNextPage``) rather than fetch-all-and-slice-in-Python, and computes ``totalCount``
as its own batched ``GROUP BY`` aggregate issued ONLY when the field selects it. These
tests assert:

- forward ``first``/``after`` walks pages with no overlap/gap and the slice lives IN SQL
  (compiled SQL has ``row_number`` + the keyset predicate, not a Python ``__rn`` filter);
- the keyset comparator is correct over DESC and over a NULLABLE column (NULLS FIRST and
  LAST) and multi-key, across the null boundary;
- reverse ``last``/``before`` returns the correct tail in the requested order;
- a cursor minted for a DIFFERENT ordering — or a garbage cursor — is REJECTED LOUDLY
  (digest mismatch / parse error), never decoded-to-0;
- ``totalCount`` is a separate batched count, correct on a full AND an empty terminal
  page, and NOT issued when the field does not select it;
- a connection layer is at most 2 statements (page + optional count) across ALL parents.

DB tests are marked ``pg`` (deselectable ``-m 'not pg'``); they touch ONLY the
``grafast_demo`` schema of ``grafast_py_test`` and do not alter authors/posts/comments —
the nullable/multi-key/desc cases use the dedicated ``keyset_rows`` fixture table.
"""

import functools

import pytest
import pytest_asyncio
from graphql import graphql
from sqlalchemy import NUMERIC, TIMESTAMP, column, select, table

from grafast_py.context import GrafastExecutionContext
from grafast_py.core_steps import access, constant
from grafast_py.pg.connection import PgConnectionStep
from grafast_py.pg.cursor import (
    decode_keyset_cursor,
    effective_nulls,
    encode_keyset_cursor,
    keyset_where,
    order_digest,
)
from grafast_py.pg.engine import count_sql, dispose_engine, get_engine
from grafast_py.pg.executor import SQLAlchemyExecutor, pg_request_context
from grafast_py.pg.ordering import OrderTerm, normalize_order, order_clauses
from grafast_py.pg.resource import PgRegistry, PgResource
from examples.demo_schema import build_demo_schema
from examples.seed import setup_demo_schema, setup_keyset_table

pytestmark = pytest.mark.pg

KEYSET_COLUMNS = ["id", "owner_id", "rank", "name", "created", "price"]
# the non-native columns whose text-origin cursor value must be cast back to type.
KEYSET_TYPES = {"created": TIMESTAMP(timezone=True), "price": NUMERIC}


async def run(schema, query, variables=None):
    """Run a query through our engine over the convenience engine for the request."""
    with pg_request_context(SQLAlchemyExecutor(get_engine())):
        return await graphql(
            schema,
            query,
            variable_values=variables,
            execution_context_class=GrafastExecutionContext,
        )


@pytest_asyncio.fixture
async def demo_schema():
    """(Re)seed ``grafast_demo`` and build the demo schema (fresh engine per test)."""
    await dispose_engine()
    await setup_demo_schema()
    schema = build_demo_schema()
    yield schema
    await dispose_engine()


@pytest_asyncio.fixture
async def keyset_seeded():
    """(Re)seed ``grafast_demo`` + the ``keyset_rows`` fixture (fresh engine per test)."""
    await dispose_engine()
    await setup_demo_schema()
    await setup_keyset_table()
    yield
    await dispose_engine()


def keyset_resource():
    """A ``keyset_rows`` resource carrying the non-native column SQL types."""
    return PgResource(
        "keyset_rows",
        "grafast_demo",
        "keyset_rows",
        KEYSET_COLUMNS,
        registry=PgRegistry(),
        column_types=KEYSET_TYPES,
    )


def keyset_table():
    return table(
        "keyset_rows", *[column(c) for c in KEYSET_COLUMNS], schema="grafast_demo"
    )


async def fetch_keyset_rows():
    """All ``keyset_rows`` as column-keyed dicts (for the brute-force oracle)."""
    async with get_engine().connect() as conn:
        result = await conn.execute(
            select(*[column(c) for c in KEYSET_COLUMNS]).select_from(keyset_table())
        )
        return [dict(row) for row in result.mappings().all()]


def brute_force_sorted(rows, terms):
    """Sort ``rows`` exactly as Postgres ORDER BY would (direction + NULLS placement)."""

    def cmp(a, b):
        for term in terms:
            va, vb = a[term.column], b[term.column]
            if va is None or vb is None:
                if va is None and vb is None:
                    continue
                null_first = effective_nulls(term) == "first"
                a_is_null = va is None
                return -1 if (a_is_null == null_first) else 1
            if va < vb:
                return 1 if term.descending else -1
            if va > vb:
                return -1 if term.descending else 1
        return 0

    return sorted(rows, key=functools.cmp_to_key(cmp))


# ----------------------------------------------------------------- forward walk


@pytest.mark.asyncio
async def test_forward_first_after_walks_pages_no_overlap_no_gap(demo_schema):
    """``first:2`` then ``first:2, after:<endCursor>`` walks pages with no overlap/gap.

    author 3 owns posts 6,7,8,9; the two pages must be exactly [6,7] then [8,9] — every
    post once, in order, no repeats.
    """
    page1 = await run(
        demo_schema,
        "{ author(id: 3) { postsConnection(first: 2) {"
        " edges { node { id } } pageInfo { endCursor hasNextPage } } } }",
    )
    assert page1.errors is None
    conn1 = page1.data["author"]["postsConnection"]
    assert [e["node"]["id"] for e in conn1["edges"]] == [6, 7]
    assert conn1["pageInfo"]["hasNextPage"] is True

    page2 = await run(
        demo_schema,
        """
        query P($after: String!) {
          author(id: 3) { postsConnection(first: 2, after: $after) {
            edges { node { id } } pageInfo { hasNextPage hasPreviousPage } } }
        }
        """,
        {"after": conn1["pageInfo"]["endCursor"]},
    )
    assert page2.errors is None
    conn2 = page2.data["author"]["postsConnection"]
    assert [e["node"]["id"] for e in conn2["edges"]] == [8, 9]
    assert conn2["pageInfo"]["hasNextPage"] is False
    assert conn2["pageInfo"]["hasPreviousPage"] is True


@pytest.mark.asyncio
async def test_page_slice_is_in_sql_not_python():
    """The compiled page SQL carries ``row_number`` AND the keyset predicate.

    Proves the slice is IN SQL (a window + a parameterised keyset WHERE), NOT a Python
    ``__rn`` post-filter over all rows.
    """
    posts = PgResource(
        "posts", "grafast_demo", "posts", ["id", "author_id", "title"],
        registry=PgRegistry(),
    )
    after = encode_keyset_cursor((OrderTerm("id"),), {"id": 2})
    step = PgConnectionStep(
        posts, constant(None), "author_id", order_by=["id"], first=2, after=after
    )
    sql = str(step.build_page_query())
    assert "row_number" in sql.lower()
    # the keyset predicate over the id key (bound param, never inlined).
    assert "id > :__ks_0" in sql
    assert ":page_limit" in sql  # per-partition fetch-one-extra bound


# --------------------------------------------------------- keyset over DESC / NULLs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "spec",
    [
        [OrderTerm("rank"), OrderTerm("id")],  # ASC NULLS LAST (default)
        [OrderTerm("rank", descending=True), OrderTerm("id")],  # DESC NULLS FIRST
        [OrderTerm("rank", nulls="first"), OrderTerm("id")],  # ASC NULLS FIRST
        [OrderTerm("rank", descending=True, nulls="last"), OrderTerm("id")],  # DESC LAST
        [OrderTerm("rank"), OrderTerm("name", descending=True), OrderTerm("id")],  # 3-key
        [OrderTerm("created"), OrderTerm("id")],  # timestamptz round-trip
        [OrderTerm("price", descending=True), OrderTerm("id")],  # numeric round-trip
    ],
)
async def test_keyset_after_matches_brute_force_over_every_cursor(keyset_seeded, spec):
    """``keyset_where(after=True)`` selects exactly the rows AFTER each cursor.

    Exhaustive: for EVERY row as the cursor, the SQL ``WHERE keyset_after ORDER BY same``
    must equal the brute-force "sorted, take everything after this row". Covers DESC, the
    NULL boundary under both placements, multi-key with a DESC middle key, and the
    text-origin (timestamptz / numeric) cursor round-trip.
    """
    terms = normalize_order(spec, primary_key="id")
    rows = await fetch_keyset_rows()
    ordered = brute_force_sorted(rows, terms)

    for index, cursor_row in enumerate(ordered):
        values = [cursor_row[t.column] for t in terms]
        predicate = keyset_where(terms, values, after=True, column_types=KEYSET_TYPES)
        stmt = (
            select(column("id"))
            .select_from(keyset_table())
            .where(predicate)
            .order_by(*order_clauses(terms))
        )
        async with get_engine().connect() as conn:
            result = await conn.execute(stmt)
            got = [row[0] for row in result]
        expected = [r["id"] for r in ordered[index + 1 :]]
        assert got == expected, f"{spec} after id={cursor_row['id']}"


@pytest.mark.asyncio
async def test_keyset_before_matches_brute_force_over_nullable(keyset_seeded):
    """``keyset_where(after=False)`` selects exactly the rows BEFORE each cursor.

    Over the nullable ``rank`` (ASC NULLS LAST), for every cursor row the BEFORE predicate
    must equal "sorted, take everything before this row" — the reverse-paging building
    block, correct across the NULL boundary.
    """
    terms = normalize_order([OrderTerm("rank"), OrderTerm("id")], primary_key="id")
    rows = await fetch_keyset_rows()
    ordered = brute_force_sorted(rows, terms)

    for index, cursor_row in enumerate(ordered):
        values = [cursor_row[t.column] for t in terms]
        predicate = keyset_where(terms, values, after=False, column_types=KEYSET_TYPES)
        stmt = (
            select(column("id"))
            .select_from(keyset_table())
            .where(predicate)
            .order_by(*order_clauses(terms))
        )
        async with get_engine().connect() as conn:
            result = await conn.execute(stmt)
            got = [row[0] for row in result]
        expected = [r["id"] for r in ordered[:index]]
        assert got == expected, f"before id={cursor_row['id']}"


@pytest.mark.asyncio
async def test_connection_over_nullable_desc_pages_across_null_boundary(keyset_seeded):
    """A connection over ``rank DESC NULLS FIRST`` pages correctly across the NULLs.

    rank DESC NULLS FIRST orders the NULL ranks (ids 2,5,9) first, then 40,30,25,20,20,
    10,10,10,5 with id ascending as the tie-break. Walking the connection in pages of 4
    must reproduce that exact total order with no overlap/gap.
    """
    resource = keyset_resource()
    terms = normalize_order(
        [OrderTerm("rank", descending=True), OrderTerm("id")], primary_key="id"
    )
    rows = await fetch_keyset_rows()
    expected = [r["id"] for r in brute_force_sorted(rows, terms)]

    walked = []
    after = None
    with pg_request_context(SQLAlchemyExecutor(get_engine())):
        while True:
            step = PgConnectionStep(
                resource, constant(None), "owner_id",
                order_by=[OrderTerm("rank", descending=True)],
                first=4, after=after,
            )
            out = await step.execute(1, [[1]])
            conn = out[0]
            walked.extend(e["node"]["id"] for e in conn["edges"])
            if not conn["pageInfo"]["hasNextPage"]:
                break
            after = conn["pageInfo"]["endCursor"]

    assert walked == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "order_term",
    [
        OrderTerm("created"),  # timestamptz: cursor value via isoformat + cast(text AS timestamptz)
        OrderTerm("price", descending=True),  # numeric: cursor value via str + cast(text AS numeric)
    ],
)
async def test_forward_connection_pages_across_non_native_column_cast_path(
    keyset_seeded, order_term
):
    """Forward keyset paging walks a full connection over a NON-NATIVE order column.

    Locks the Phase-6 CAST-path probe end to end: paging by ``created`` (timestamptz) or
    ``price`` (numeric) means the endCursor value is a STRING (isoformat / str(Decimal)),
    and the next page's keyset predicate binds it back via ``cast(text AS <type>)``. If the
    round-trip were lossy the pages would overlap or skip; walking pages of 5 across all 12
    owner-1 rows must reproduce the exact brute-force order with no overlap/gap.
    """
    resource = keyset_resource()
    terms = normalize_order([order_term], primary_key="id")
    rows = await fetch_keyset_rows()
    expected = [r["id"] for r in brute_force_sorted(rows, terms)]

    walked = []
    after = None
    with pg_request_context(SQLAlchemyExecutor(get_engine())):
        while True:
            step = PgConnectionStep(
                resource, constant(None), "owner_id",
                order_by=[order_term], first=5, after=after,
            )
            out = await step.execute(1, [[1]])
            conn = out[0]
            # the cursor over a non-native column is an opaque string carrying the
            # text-origin value the next page casts back to the column type.
            for edge in conn["edges"]:
                assert isinstance(edge["cursor"], str)
            walked.extend(e["node"]["id"] for e in conn["edges"])
            if not conn["pageInfo"]["hasNextPage"]:
                break
            after = conn["pageInfo"]["endCursor"]

    assert walked == expected
    assert len(walked) == len(set(walked)) == 12  # every row once: no overlap, no gap


# ---------------------------------------------------------------------- reverse


@pytest.mark.asyncio
async def test_reverse_last_returns_tail_in_requested_order(demo_schema):
    """``last:2`` returns the tail of the connection in the REQUESTED (forward) order."""
    result = await run(
        demo_schema,
        "{ author(id: 3) { postsConnection(last: 2) {"
        " edges { node { id } } pageInfo { hasNextPage hasPreviousPage } } } }",
    )
    assert result.errors is None
    conn = result.data["author"]["postsConnection"]
    # author 3 owns 6,7,8,9; the last two in forward order are 8,9.
    assert [e["node"]["id"] for e in conn["edges"]] == [8, 9]
    assert conn["pageInfo"]["hasPreviousPage"] is True
    assert conn["pageInfo"]["hasNextPage"] is False


@pytest.mark.asyncio
async def test_reverse_last_before_returns_correct_window(demo_schema):
    """``last:2, before:<startCursor of tail>`` returns the window just before the tail."""
    tail = await run(
        demo_schema,
        "{ author(id: 3) { postsConnection(last: 2) {"
        " pageInfo { startCursor } } } }",
    )
    before = tail.data["author"]["postsConnection"]["pageInfo"]["startCursor"]

    result = await run(
        demo_schema,
        """
        query P($before: String!) {
          author(id: 3) { postsConnection(last: 2, before: $before) {
            edges { node { id } } pageInfo { hasNextPage hasPreviousPage } } }
        }
        """,
        {"before": before},
    )
    assert result.errors is None
    conn = result.data["author"]["postsConnection"]
    # before the tail [8,9] -> the preceding two are 6,7; later rows exist (hasNextPage).
    assert [e["node"]["id"] for e in conn["edges"]] == [6, 7]
    assert conn["pageInfo"]["hasNextPage"] is True
    assert conn["pageInfo"]["hasPreviousPage"] is False


# --------------------------------------------------------------- cursor validation


def test_cursor_for_different_order_is_rejected_loudly():
    """A cursor minted under order A is REJECTED (digest mismatch) under order B."""
    order_a = normalize_order([OrderTerm("id")], primary_key="id")
    order_b = normalize_order(
        [OrderTerm("rank", descending=True), OrderTerm("id")], primary_key="id"
    )
    cursor = encode_keyset_cursor(order_a, {"id": 5, "rank": 10})
    with pytest.raises(ValueError, match="different ordering"):
        decode_keyset_cursor(cursor, order_b)


def test_garbage_cursor_raises_loudly_not_decode_to_zero():
    """A truncated/garbage cursor raises LOUDLY rather than decoding to a 0/empty page."""
    terms = normalize_order([OrderTerm("id")], primary_key="id")
    with pytest.raises(ValueError, match="malformed keyset cursor"):
        decode_keyset_cursor("not-valid-base64!!", terms)
    with pytest.raises(ValueError, match="malformed keyset cursor"):
        decode_keyset_cursor("", terms)


def test_cursor_wrong_value_count_raises_loudly():
    """A cursor whose value count does not match the order is rejected loudly."""
    import base64
    import json

    terms = normalize_order([OrderTerm("rank"), OrderTerm("id")], primary_key="id")
    # right digest, but only one value where two are expected.
    payload = [order_digest(terms), 10]
    cursor = base64.b64encode(json.dumps(payload).encode()).decode()
    with pytest.raises(ValueError, match="values but the order has"):
        decode_keyset_cursor(cursor, terms)


# ------------------------------------------------ fail-loud guards: caller misuse


def test_connection_rejects_ambiguous_forward_and_reverse_pagination():
    """Supplying forward (first/after) AND reverse (last/before) args raises loudly.

    A connection pages forward XOR reverse; combining the two is undefined under the Relay
    spec. Constructing a step with both must fail with a clear error rather than silently
    derive a direction (and ignore the other args).
    """
    posts = PgResource(
        "posts", "grafast_demo", "posts", ["id", "author_id", "title"],
        registry=PgRegistry(),
    )
    with pytest.raises(ValueError, match="ambiguous Relay pagination"):
        PgConnectionStep(
            posts, constant(None), "author_id", order_by=["id"], first=2, last=2
        )
    # first + before (mixed forward/reverse) is equally ambiguous.
    with pytest.raises(ValueError, match="ambiguous Relay pagination"):
        PgConnectionStep(
            posts, constant(None), "author_id", order_by=["id"], first=2, before="x"
        )


def test_encode_keyset_cursor_missing_order_column_raises_loudly():
    """Ordering by a column NOT projected into the row raises a clear named error.

    The cursor is built from the row's order-column values; if an order column is absent
    from the row (it was not selected/stored) the encoder names the missing column rather
    than letting a bare KeyError surface deep in execute.
    """
    terms = normalize_order([OrderTerm("missing"), OrderTerm("id")], primary_key="id")
    with pytest.raises(ValueError, match="order column 'missing' is not present"):
        encode_keyset_cursor(terms, {"id": 1})


@pytest.mark.asyncio
async def test_connection_rejects_foreign_cursor_at_plan_time(demo_schema):
    """A foreign cursor surfaces as a loud error through the GraphQL plan resolver."""
    # a cursor minted under a 2-key order, used by postsConnection's single-key id order.
    foreign_order = normalize_order(
        [OrderTerm("title"), OrderTerm("id")], primary_key="id"
    )
    foreign = encode_keyset_cursor(foreign_order, {"title": "x", "id": 1})
    query = """
    query P($after: String!) {
      author(id: 3) { postsConnection(first: 2, after: $after) { edges { node { id } } } }
    }
    """
    with pytest.raises(ValueError, match="different ordering"):
        await run(demo_schema, query, {"after": foreign})


# --------------------------------------------------------------------- totalCount


@pytest.mark.asyncio
async def test_totalcount_full_page_is_separate_batched_count(demo_schema):
    """totalCount on a FULL page is correct and comes from a separate batched aggregate."""
    query = """
    {
      authors {
        postsConnection(first: 2) {
          totalCount
          edges { node { id } }
        }
      }
    }
    """
    with count_sql(get_engine()) as counter:
        result = await run(demo_schema, query)
    assert result.errors is None
    # authors (1) + connection page (1) + the separate count aggregate (1) = 3.
    assert counter.count == 3
    totals = [a["postsConnection"]["totalCount"] for a in result.data["authors"]]
    assert totals == [2, 3, 4]  # author i has i+1 posts — the FULL counts, not the page


@pytest.mark.asyncio
async def test_totalcount_correct_on_empty_terminal_page(demo_schema):
    """totalCount is correct (NOT 0) on an empty terminal page — no Phase-3 regression."""
    walk = await run(
        demo_schema,
        "{ author(id: 1) { postsConnection(first: 2) {"
        " pageInfo { endCursor } } } }",
    )
    end_cursor = walk.data["author"]["postsConnection"]["pageInfo"]["endCursor"]

    result = await run(
        demo_schema,
        """
        query P($after: String!) {
          author(id: 1) { postsConnection(first: 2, after: $after) {
            totalCount edges { node { id } } pageInfo { hasPreviousPage } } }
        }
        """,
        {"after": end_cursor},
    )
    assert result.errors is None
    conn = result.data["author"]["postsConnection"]
    assert conn["edges"] == []  # past the last row -> empty page
    assert conn["totalCount"] == 2  # the separate aggregate keeps the real count
    assert conn["pageInfo"]["hasPreviousPage"] is True


@pytest.mark.asyncio
async def test_totalcount_not_selected_skips_count_query(demo_schema):
    """When totalCount is NOT selected, the count query is NOT issued (assert via count_sql)."""
    query = """
    {
      authors {
        postsConnection(first: 2) {
          edges { node { id } }
          pageInfo { hasNextPage }
        }
      }
    }
    """
    with count_sql(get_engine()) as counter:
        result = await run(demo_schema, query)
    assert result.errors is None
    # authors (1) + connection page (1) ONLY — no count aggregate, since totalCount is
    # not in the selection set.
    assert counter.count == 2
    # the page is still correct.
    grace = result.data["authors"][2]
    assert [e["node"]["id"] for e in grace["postsConnection"]["edges"]] == [6, 7]


# --------------------------------------------------------------------- batching


@pytest.mark.asyncio
async def test_connection_layer_is_at_most_two_statements_across_parents(demo_schema):
    """A connection layer is at most 2 statements (page + optional count) for ALL parents.

    Three authors, each with a connection: still page + count = 2 connection statements
    (plus the authors root = 3 total) — O(depth), independent of the number of parents.
    """
    query = """
    {
      authors {
        postsConnection(first: 2) {
          totalCount
          edges { node { id } }
        }
      }
    }
    """
    with count_sql(get_engine()) as counter:
        result = await run(demo_schema, query)
    assert result.errors is None
    assert counter.count == 3  # 1 root + (page + count) for the whole bucket


# ----------------------------------------------------------------- cursor round-trip


@pytest.mark.asyncio
async def test_cursor_roundtrip_for_datetime_and_decimal(keyset_seeded):
    """A keyset cursor over (created, price) round-trips text-origin values losslessly.

    The encoder stringifies datetime via isoformat and Decimal via str; decode returns the
    text as-is and the keyset predicate casts it back to the column type — selecting
    exactly the brute-force rows after the cursor.
    """
    terms = normalize_order(
        [OrderTerm("created"), OrderTerm("price"), OrderTerm("id")], primary_key="id"
    )
    rows = await fetch_keyset_rows()
    ordered = brute_force_sorted(rows, terms)
    cursor_row = ordered[4]
    cursor = encode_keyset_cursor(terms, cursor_row)
    values = decode_keyset_cursor(cursor, terms)

    predicate = keyset_where(terms, values, after=True, column_types=KEYSET_TYPES)
    stmt = (
        select(column("id"))
        .select_from(keyset_table())
        .where(predicate)
        .order_by(*order_clauses(terms))
    )
    async with get_engine().connect() as conn:
        result = await conn.execute(stmt)
        got = [row[0] for row in result]
    assert got == [r["id"] for r in ordered[5:]]


# ----------------------------------------------------------------- dedup correctness


def dedup_key(step):
    """The class + peer_key + dedup_params slice of the planner's structural key."""
    return (type(step), step.peer_key, step.dedup_params())


def make_conn(**kwargs):
    posts = PgResource(
        "posts", "grafast_demo", "posts", ["id", "author_id", "title"],
        registry=PgRegistry(),
    )
    return PgConnectionStep(
        posts, constant(None), "author_id", order_by=["id"], **kwargs
    )


def test_different_after_cursor_values_do_not_dedup():
    """Two connections differing only by their ``after`` cursor VALUES are NOT peers."""
    c1 = encode_keyset_cursor((OrderTerm("id"),), {"id": 2})
    c2 = encode_keyset_cursor((OrderTerm("id"),), {"id": 5})
    a = make_conn(first=2, after=c1)
    b = make_conn(first=2, after=c2)
    assert a.peer_key != b.peer_key
    assert a.dedup_params() != b.dedup_params()
    assert dedup_key(a) != dedup_key(b)


def test_identical_after_cursor_values_dedup():
    """Two connections with the SAME first/after (and order) ARE peers."""
    c = encode_keyset_cursor((OrderTerm("id"),), {"id": 2})
    a = make_conn(first=2, after=c)
    b = make_conn(first=2, after=c)
    assert a.peer_key == b.peer_key
    assert a.dedup_params() == b.dedup_params()
    assert dedup_key(a) == dedup_key(b)


def test_needs_total_discriminates_dedup_key():
    """A totalCount-selecting connection does NOT merge with a totalCount-less one."""
    a = make_conn(first=2, needs_total=True)
    b = make_conn(first=2, needs_total=False)
    assert a.peer_key != b.peer_key
    assert a.dedup_params() != b.dedup_params()


def test_forward_and_reverse_do_not_dedup():
    """Forward (first) and reverse (last) connections are NOT peers."""
    a = make_conn(first=2)
    b = make_conn(last=2)
    assert a.peer_key != b.peer_key
    assert a.dedup_params() != b.dedup_params()
