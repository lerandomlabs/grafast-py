"""pgUnionAll: a keyset-paged Relay connection over N member tables via ``UNION ALL``.

The CROSS-TABLE polymorphism shape (the ``pgunionall`` feature of the polymorphism wave):
a GraphQL union (``SearchResult = Article | Snippet``) whose concrete types live in SEPARATE
tables. :class:`grafast_py.pg.union.PgUnionAllStep` fetches every member in ONE ``UNION ALL``
statement whose branches project a SHARED, NULL-PADDED column shape plus a ``__typename``
literal tag, keyset-slices the merged result over the shared order columns, and (per-parent
mode) partitions by the match key. The ``resolve_type_from_tag('__typename')`` bridge then
resolves each row at COMPLETION time and the EXISTING completion-time abstract dispatch
groups rows by concrete type and plans each group's sub-selection like a normal object field
— so a member's type-specific fields batch per concrete-type group with NO plan-time
polymorphism and NO core-engine change (the SAME machinery the single-table discriminator
shape rides).

These tests assert:

- the SQL shape: ROOT mode unions the member legs (NULL-padded shared projection + literal
  ``__typename``) under a plain ``LIMIT``; PER-PARENT mode adds ``= ANY(:keys)`` to each leg
  and a ``row_number() OVER (PARTITION BY match)`` window; the count is a SEPARATE union
  without the cursor predicate;
- end-to-end over the live ``grafast_demo`` scratch DB (the ``articles`` + ``snippets``
  fixture): a ROOT union resolves each row's concrete type off the tag, keyset-orders the
  interleaved members into one page, forward-pages with no overlap/gap across the union,
  reports ``totalCount`` (gated), and is at most TWO statements (page + count) regardless of
  member count; a PER-PARENT union keyed on ``owner_id`` scatters each parent's page back;
- the fail-loud guards fire — an order column that is not shared, a per-parent/root match
  mismatch, ambiguous forward+reverse paging, an empty member set, and a cursor minted under
  a different ordering are all rejected at construction/plan time;
- DEDUP CORRECTNESS (no DB), BOTH directions: two union steps differing only in the member
  set (a table / a tagged type name / the per-member match / a per-branch filter VALUE), the
  shared projection, the order, the cursor VALUES, ``first``/``last`` or ``needs_total`` get
  DIFFERENT keys and do NOT merge through ``dag.Plan.deduplicate()``; structurally-identical
  steps DO merge.

DB tests are marked ``pg`` (deselectable ``-m 'not pg'``); they use the dedicated
``articles`` / ``snippets`` fixture and touch ONLY the ``grafast_demo`` schema, perturbing
nothing else.
"""

import pytest
import pytest_asyncio
from graphql import graphql
from sqlalchemy import TIMESTAMP, column
from sqlalchemy.dialects import postgresql

from grafast_py.context import GrafastExecutionContext
from grafast_py.core_steps import constant
from grafast_py.dag import Plan
from grafast_py.pg.cursor import encode_keyset_cursor
from grafast_py.pg.engine import count_sql, dispose_engine, get_engine
from grafast_py.pg.executor import SQLAlchemyExecutor, pg_request_context
from grafast_py.pg.ordering import OrderTerm
from grafast_py.pg.resource import PgRegistry, PgResource
from grafast_py.pg.union import (
    TYPENAME_COLUMN,
    PgUnionAllStep,
    PgUnionMember,
    pg_union_all,
    union_all_connection,
)
from examples.poly_schema import build_poly_schema
from examples.seed import (
    setup_demo_schema,
    setup_media_table,
    setup_media_tags_table,
    setup_union_collision_tables,
    setup_union_member_tables,
)

# the non-native keyset order column (created is timestamptz) declares its SQL type so the
# text-origin cursor value casts back to type — mirrors the keyset fixture's KEYSET_TYPES.
CREATED_TYPE = {"created": TIMESTAMP(timezone=True)}


def make_articles() -> PgResource:
    return PgResource(
        "articles",
        "grafast_demo",
        "articles",
        ["id", "owner_id", "created", "headline", "word_count"],
        registry=PgRegistry(),
        column_types=CREATED_TYPE,
    )


def make_snippets() -> PgResource:
    return PgResource(
        "snippets",
        "grafast_demo",
        "snippets",
        ["id", "owner_id", "created", "body"],
        registry=PgRegistry(),
        column_types=CREATED_TYPE,
    )


def make_union(*, key_step=None, members=None, **kwargs) -> PgUnionAllStep:
    """An ``articles`` + ``snippets`` union keyed on the shared (id/owner_id/created) shape.

    ``members`` overrides the default pair (used to vary the member set in dedup tests);
    ``key_step`` switches on per-parent mode (members then carry a ``match``).
    """
    if members is None:
        match = "owner_id" if key_step is not None else None
        members = [
            PgUnionMember(make_articles(), "Article", match=match),
            PgUnionMember(make_snippets(), "Snippet", match=match),
        ]
    return PgUnionAllStep(
        members,
        shared_columns=["id", "owner_id", "created"],
        order_by=["created"],
        key_step=key_step,
        **kwargs,
    )


def compile_sql(stmt) -> str:
    return str(stmt.compile(dialect=postgresql.dialect()))


# ----------------------------------------------------------------- SQL shape (no DB)


def test_root_page_query_unions_legs_with_null_padded_projection():
    """ROOT page: each leg projects the shared cols + NULL-padded member cols + the tag."""
    step = make_union(first=3)
    sql = compile_sql(step.build_page_query())
    # both member tables are unioned
    assert "grafast_demo.articles" in sql
    assert "grafast_demo.snippets" in sql
    assert "UNION ALL" in sql
    # the typename tag column is projected as a literal in every leg
    assert "AS __typename" in sql
    # the NULL-padded member columns appear (articles lacks body, snippets lacks headline)
    assert "NULL AS body" in sql
    assert "NULL AS headline" in sql
    # plain LIMIT on the root page (no per-parent window)
    assert "LIMIT" in sql
    assert "row_number" not in sql.lower()


def test_per_parent_page_query_adds_match_and_window():
    """PER-PARENT page: each leg matches ``= ANY(:keys)`` under a partitioned window."""
    step = make_union(key_step=constant(None), first=2)
    sql = compile_sql(step.build_page_query())
    assert "= ANY (" in sql
    assert "row_number()" in sql.lower()
    assert "PARTITION BY" in sql
    # the per-partition page cap rather than a bucket-wide LIMIT
    assert "__rn <=" in sql


def test_count_query_omits_cursor_and_groups_per_parent():
    """The count union is SEPARATE, cursor-free, and grouped by the match key (per-parent)."""
    cur = encode_keyset_cursor(
        (OrderTerm("created"), OrderTerm("id"), OrderTerm("__typename")),
        {"created": "2024-03-01T09:00:00+00:00", "id": 1, "__typename": "Article"},
    )
    step = make_union(key_step=constant(None), first=2, after=cur, needs_total=True)
    count_sql_text = compile_sql(step.build_count_query())
    # grouped count over the union of the key-matched legs
    assert "count(*)" in count_sql_text.lower()
    assert "GROUP BY" in count_sql_text
    # the cursor predicate is on the PAGE query, never the count (full per-parent set)
    assert "IS NOT DISTINCT FROM" not in count_sql_text


def test_root_count_query_is_single_count_over_union():
    """ROOT count: a single ``count(*)`` over the whole union (no GROUP BY)."""
    step = make_union(first=3, needs_total=True)
    count_sql_text = compile_sql(step.build_count_query())
    assert "count(*)" in count_sql_text.lower()
    assert "GROUP BY" not in count_sql_text


def test_typename_column_name_is_the_meta_field_spelling():
    """The tag column is ``__typename`` (the resolve_type_from_tag bridge reads it)."""
    assert TYPENAME_COLUMN == "__typename"


# ----------------------------------------------------------------- fail-loud guards (no DB)


def test_empty_member_set_fails_loud():
    with pytest.raises(ValueError, match="at least one member"):
        PgUnionAllStep([], shared_columns=["id"], order_by=["id"])


def test_order_column_not_shared_fails_loud():
    """Ordering by a member-specific (non-shared) column is rejected with its name."""
    with pytest.raises(ValueError, match="not a shared column"):
        PgUnionAllStep(
            [PgUnionMember(make_articles(), "Article")],
            shared_columns=["id", "owner_id", "created"],
            order_by=["headline"],  # member-specific, not shared
        )


def test_per_parent_without_member_match_fails_loud():
    """A key_step (per-parent) with a member that declares no match is a wiring bug."""
    with pytest.raises(ValueError, match="needs each member to declare a match"):
        PgUnionAllStep(
            [PgUnionMember(make_articles(), "Article")],  # no match
            shared_columns=["id", "owner_id", "created"],
            order_by=["created"],
            key_step=constant(None),
        )


def test_root_with_member_match_fails_loud():
    """A root union (no key_step) whose member declares a match is a wiring bug."""
    with pytest.raises(ValueError, match="must not declare a member match"):
        PgUnionAllStep(
            [PgUnionMember(make_articles(), "Article", match="owner_id")],
            shared_columns=["id", "owner_id", "created"],
            order_by=["created"],
        )


def test_members_disagree_on_match_arity_fails_loud():
    """Per-parent members must match the SAME number of key columns."""
    with pytest.raises(ValueError, match="disagree on match arity"):
        PgUnionAllStep(
            [
                PgUnionMember(make_articles(), "Article", match="owner_id"),
                PgUnionMember(make_snippets(), "Snippet", match=("owner_id", "id")),
            ],
            shared_columns=["id", "owner_id", "created"],
            order_by=["created"],
            key_step=constant(None),
        )


def test_forward_and_reverse_paging_both_fails_loud():
    with pytest.raises(ValueError, match="forward XOR reverse"):
        make_union(first=3, last=3)


def test_cursor_for_different_ordering_rejected():
    """A cursor minted under a different order is rejected loudly (digest mismatch)."""
    # cursor minted against order (rank) — not this union's (created, id) order
    bad = encode_keyset_cursor((OrderTerm("rank"),), {"rank": 5})
    with pytest.raises(ValueError, match="different ordering"):
        make_union(first=3, after=bad)


def test_member_where_raw_string_fails_loud():
    """A per-branch WHERE must be a Core predicate, never a raw string (injection seam)."""
    with pytest.raises(TypeError, match="never a raw string"):
        PgUnionMember(make_articles(), "Article", where=["owner_id = 1"])


# ----------------------------------------------------------------- dedup correctness (no DB)


def dedup_key(step):
    """The class + peer_key + dedup_params slice of the planner's structural key."""
    return (type(step), step.peer_key, step.dedup_params())


def test_identical_unions_merge():
    """Two structurally-identical unions ARE peers and merge through deduplicate()."""
    a = make_union(first=3, needs_total=True)
    b = make_union(first=3, needs_total=True)
    assert a.peer_key == b.peer_key
    assert a.dedup_params() == b.dedup_params()
    assert dedup_key(a) == dedup_key(b)

    plan = Plan()
    plan.add_step(a)
    plan.add_step(b)
    remap = plan.deduplicate()
    assert remap[a.id] is remap[b.id]


def test_different_member_table_does_not_dedup():
    """Two unions over a DIFFERENT member table set never merge (different SQL)."""
    a = make_union(first=3)
    # swap one member for a different table (the demo posts table)
    other = PgResource(
        "posts", "grafast_demo", "posts", ["id", "owner_id", "created", "title"],
        registry=PgRegistry(), column_types=CREATED_TYPE,
    )
    b = make_union(
        first=3,
        members=[
            PgUnionMember(make_articles(), "Article"),
            PgUnionMember(other, "Post"),
        ],
    )
    assert a.peer_key != b.peer_key
    assert a.dedup_params() != b.dedup_params()

    plan = Plan()
    plan.add_step(a)
    plan.add_step(b)
    remap = plan.deduplicate()
    assert remap[a.id] is a
    assert remap[b.id] is b


def test_different_member_typename_tag_does_not_dedup():
    """Members tagged with DIFFERENT type names emit different ``__typename`` literals."""
    a = make_union(first=3)
    b = make_union(
        first=3,
        members=[
            PgUnionMember(make_articles(), "Story"),  # different tag
            PgUnionMember(make_snippets(), "Snippet"),
        ],
    )
    assert a.peer_key != b.peer_key
    assert a.dedup_params() != b.dedup_params()


def test_per_branch_where_value_discriminates_dedup():
    """Two unions whose branch filters differ only by VALUE never merge; same value merges."""
    a = make_union(
        first=3,
        members=[
            PgUnionMember(make_articles(), "Article", where=[column("owner_id") == 1]),
            PgUnionMember(make_snippets(), "Snippet"),
        ],
    )
    b = make_union(
        first=3,
        members=[
            PgUnionMember(make_articles(), "Article", where=[column("owner_id") == 2]),
            PgUnionMember(make_snippets(), "Snippet"),
        ],
    )
    # value-included signature -> distinct keys
    assert a.peer_key != b.peer_key
    assert a.dedup_params() != b.dedup_params()

    c = make_union(
        first=3,
        members=[
            PgUnionMember(make_articles(), "Article", where=[column("owner_id") == 1]),
            PgUnionMember(make_snippets(), "Snippet"),
        ],
    )
    # same filter value -> peers
    assert a.peer_key == c.peer_key
    assert a.dedup_params() == c.dedup_params()

    plan = Plan()
    plan.add_step(a)
    plan.add_step(b)
    plan.add_step(c)
    remap = plan.deduplicate()
    assert remap[a.id] is remap[c.id]  # a and c merge
    assert remap[b.id] is b  # b stays distinct


def test_cursor_values_discriminate_dedup():
    """Two unions differing only by their ``after`` cursor VALUES are NOT peers."""
    c1 = encode_keyset_cursor(
        (OrderTerm("created"), OrderTerm("id"), OrderTerm("__typename")),
        {"created": "2024-03-01T08:00:00+00:00", "id": 1, "__typename": "Article"},
    )
    c2 = encode_keyset_cursor(
        (OrderTerm("created"), OrderTerm("id"), OrderTerm("__typename")),
        {"created": "2024-03-01T11:00:00+00:00", "id": 2, "__typename": "Snippet"},
    )
    a = make_union(first=2, after=c1)
    b = make_union(first=2, after=c2)
    assert a.peer_key != b.peer_key
    assert a.dedup_params() != b.dedup_params()


def test_first_needs_total_and_mode_discriminate_dedup():
    """``first`` / ``needs_total`` / per-parent-vs-root each discriminate the dedup key."""
    base = make_union(first=2)
    assert base.peer_key != make_union(first=3).peer_key
    assert base.peer_key != make_union(first=2, needs_total=True).peer_key
    # per-parent vs root (the same logical members) are different SQL -> distinct keys
    per_parent = make_union(key_step=constant(None), first=2)
    assert base.peer_key != per_parent.peer_key


def test_shared_projection_discriminates_dedup():
    """A different shared-column projection is different SQL -> distinct dedup key."""
    a = PgUnionAllStep(
        [PgUnionMember(make_articles(), "Article"), PgUnionMember(make_snippets(), "Snippet")],
        shared_columns=["id", "owner_id", "created"],
        order_by=["created"],
        first=2,
    )
    b = PgUnionAllStep(
        [PgUnionMember(make_articles(), "Article"), PgUnionMember(make_snippets(), "Snippet")],
        shared_columns=["id", "created"],  # narrower shared set
        order_by=["created"],
        first=2,
    )
    assert a.peer_key != b.peer_key
    assert a.dedup_params() != b.dedup_params()


# ----------------------------------------------------------------- DB-backed end-to-end

pytestmark_db = [pytest.mark.pg, pytest.mark.asyncio]


@pytest_asyncio.fixture
async def union_schema():
    """Build the poly schema after (re)seeding the union member tables (fresh engine)."""
    await dispose_engine()
    await setup_demo_schema()
    await setup_media_table()
    await setup_media_tags_table()
    await setup_union_member_tables()
    schema = build_poly_schema()
    yield schema
    await dispose_engine()


async def run(schema, query, variables=None):
    with pg_request_context(SQLAlchemyExecutor(get_engine())):
        return await graphql(
            schema,
            query,
            variable_values=variables,
            execution_context_class=GrafastExecutionContext,
        )


@pytest.mark.pg
@pytest.mark.asyncio
async def test_root_union_resolves_concrete_types_keyset_ordered(union_schema):
    """A ROOT cross-table union tags + resolves each row and keyset-orders the merged result.

    articles/snippets interleave by ``created``; the union pages them into one ordered slice,
    each row's concrete type resolved off the ``__typename`` tag at completion time. The seed
    orders a1(08:00) < s1(09:00) < a2(10:00) < s2(11:00) < a3(14:00) < s3(15:00).
    """
    query = """
    {
      search(first: 4) {
        totalCount
        nodes {
          __typename
          ... on Article { id ownerId headline wordCount }
          ... on Snippet { id ownerId body }
        }
        pageInfo { hasNextPage }
      }
    }
    """
    with count_sql(get_engine()) as counter:
        result = await run(union_schema, query)

    assert result.errors is None
    # page (1) + count (1) across the whole union — two statements regardless of member count.
    assert counter.count == 2

    conn = result.data["search"]
    assert conn["totalCount"] == 6
    assert [n["__typename"] for n in conn["nodes"]] == [
        "Article",
        "Snippet",
        "Article",
        "Snippet",
    ]
    # the type-specific columns read from the right member (NULL-padded peers never leak in)
    first_article = conn["nodes"][0]
    assert first_article["headline"] == "alpha headline"
    assert first_article["wordCount"] == 120
    assert "body" not in first_article
    first_snippet = conn["nodes"][1]
    assert first_snippet["body"] == "first snippet"
    assert "headline" not in first_snippet
    assert conn["pageInfo"]["hasNextPage"] is True


@pytest.mark.pg
@pytest.mark.asyncio
async def test_root_union_forward_pages_with_no_overlap(union_schema):
    """Forward ``first``/``after`` walks the union with no overlap/gap across both members."""
    page1 = await run(
        union_schema,
        """
        {
          search(first: 4) {
            edges { cursor node { __typename ... on Article { id } ... on Snippet { id } } }
            pageInfo { endCursor }
          }
        }
        """,
    )
    assert page1.errors is None
    e1 = page1.data["search"]["edges"]
    assert [(n["node"]["__typename"], n["node"]["id"]) for n in e1] == [
        ("Article", 1),
        ("Snippet", 1),
        ("Article", 2),
        ("Snippet", 2),
    ]
    cursor = page1.data["search"]["pageInfo"]["endCursor"]

    page2 = await run(
        union_schema,
        """
        query ($after: String!) {
          search(first: 4, after: $after) {
            nodes { __typename ... on Article { id } ... on Snippet { id } }
            pageInfo { hasNextPage }
          }
        }
        """,
        {"after": cursor},
    )
    assert page2.errors is None
    nodes2 = page2.data["search"]["nodes"]
    # the remaining two rows, in order, no overlap with page 1
    assert [(n["__typename"], n["id"]) for n in nodes2] == [
        ("Article", 3),
        ("Snippet", 3),
    ]
    assert page2.data["search"]["pageInfo"]["hasNextPage"] is False


@pytest.mark.pg
@pytest.mark.asyncio
async def test_total_count_not_selected_issues_one_statement(union_schema):
    """Without ``totalCount`` the union is page-only (the count is selection-gated)."""
    with count_sql(get_engine()) as counter:
        result = await run(
            union_schema,
            "{ search(first: 4) { nodes { __typename } } }",
        )
    assert result.errors is None
    # the page query ONLY — no separate count.
    assert counter.count == 1


@pytest.mark.pg
@pytest.mark.asyncio
async def test_per_parent_union_scatters_each_owner(union_schema):
    """A PER-PARENT union keyed on ``owner_id`` pages + scatters each owner's merged rows.

    owner 1 owns 2 articles + 2 snippets (4 total), owner 2 owns 1 article + 1 snippet (2).
    The window partitions by owner, so each parent's page slices independently in one batched
    statement.
    """
    query = """
    query ($owner: Int!) {
      activity(ownerId: $owner, first: 3) {
        totalCount
        nodes { __typename ... on Article { id headline } ... on Snippet { id body } }
        pageInfo { hasNextPage }
      }
    }
    """
    with count_sql(get_engine()) as counter:
        owner1 = await run(union_schema, query, {"owner": 1})

    assert owner1.errors is None
    # page (1) + count (1) across the bucket — O(depth), independent of parent/member count.
    assert counter.count == 2
    conn1 = owner1.data["activity"]
    assert conn1["totalCount"] == 4
    # owner 1's first 3 by created: a1(08), s1(09), a2(10)
    assert [(n["__typename"], n["id"]) for n in conn1["nodes"]] == [
        ("Article", 1),
        ("Snippet", 1),
        ("Article", 2),
    ]
    assert conn1["pageInfo"]["hasNextPage"] is True

    owner2 = await run(union_schema, query, {"owner": 2})
    assert owner2.errors is None
    conn2 = owner2.data["activity"]
    assert conn2["totalCount"] == 2
    # owner 2 owns article 3 (gamma) and snippet 3, created 14:00 < 15:00
    assert [(n["__typename"], n["id"]) for n in conn2["nodes"]] == [
        ("Article", 3),
        ("Snippet", 3),
    ]
    assert conn2["pageInfo"]["hasNextPage"] is False


@pytest.mark.pg
@pytest.mark.asyncio
async def test_per_parent_union_unknown_owner_is_empty(union_schema):
    """An owner with no rows yields an empty connection (totalCount 0, no nodes)."""
    result = await run(
        union_schema,
        "{ activity(ownerId: 999, first: 3) { totalCount nodes { __typename } pageInfo { hasNextPage } } }",
    )
    assert result.errors is None
    conn = result.data["activity"]
    assert conn["totalCount"] == 0
    assert conn["nodes"] == []
    assert conn["pageInfo"]["hasNextPage"] is False


@pytest.mark.pg
@pytest.mark.asyncio
async def test_plan_helpers_build_the_step():
    """``pg_union_all`` / ``union_all_connection`` construct a :class:`PgUnionAllStep`."""
    members = [PgUnionMember(make_articles(), "Article"), PgUnionMember(make_snippets(), "Snippet")]
    step = pg_union_all(members, ["id", "owner_id", "created"], ["created"], first=2)
    assert isinstance(step, PgUnionAllStep)
    assert step.needs_total is False

    class FakeInfo:
        field_nodes = []
        fragments = {}

    gated = union_all_connection(
        members, ["id", "owner_id", "created"], ["created"], FakeInfo(), first=2
    )
    assert isinstance(gated, PgUnionAllStep)
    # no totalCount in the (empty) selection set -> the count is not requested
    assert gated.needs_total is False


# ------------------------------------------------ cross-branch keyset totality (DB, the blocker)


def make_coll_articles() -> PgResource:
    return PgResource(
        "coll_articles", "grafast_demo", "coll_articles",
        ["id", "owner_id", "created", "headline"],
        registry=PgRegistry(), column_types=CREATED_TYPE,
    )


def make_coll_snippets() -> PgResource:
    return PgResource(
        "coll_snippets", "grafast_demo", "coll_snippets",
        ["id", "owner_id", "created", "body"],
        registry=PgRegistry(), column_types=CREATED_TYPE,
    )


def make_collision_union(*, first=None, after=None, last=None, before=None) -> PgUnionAllStep:
    """A ROOT union over the collision tables (coll_articles + coll_snippets).

    The two tables deliberately share ``(created, id)`` across branches, so the union's
    ``__typename`` tie-break is the ONLY thing that totally orders them — a (created, id)-only
    seek would drop or duplicate a tied row at the page boundary.
    """
    return PgUnionAllStep(
        [
            PgUnionMember(make_coll_articles(), "CollArticle"),
            PgUnionMember(make_coll_snippets(), "CollSnippet"),
        ],
        shared_columns=["id", "owner_id", "created"],
        order_by=["created"],
        first=first, after=after, last=last, before=before,
    )


@pytest_asyncio.fixture
async def collision_seeded():
    """(Re)seed the cross-branch (created, id) collision tables on a fresh engine."""
    await dispose_engine()
    await setup_demo_schema()
    await setup_union_collision_tables()
    yield
    await dispose_engine()


@pytest.mark.pg
@pytest.mark.asyncio
async def test_cross_branch_keyset_tie_pages_without_dropping_a_row(collision_seeded):
    """Paging across a deliberate cross-branch ``(created, id)`` tie loses/duplicates NO row.

    Both members hold id=1 @09:00 and id=2 @10:00, so ``(created, id)`` collides across the
    branches: WITHOUT the ``__typename`` tie-break, a page boundary on one tied row drops or
    duplicates its peer (the silent-data-loss class the blocker reports). Walking the union in
    pages of ONE must surface all FOUR distinct (typename, id) rows exactly once, in the total
    order ``(created, id, __typename)`` — coll_article sorts before coll_snippet on the tag.
    """
    expected = [
        ("CollArticle", 1),
        ("CollSnippet", 1),
        ("CollArticle", 2),
        ("CollSnippet", 2),
    ]

    walked = []
    after = None
    with pg_request_context(SQLAlchemyExecutor(get_engine())):
        while True:
            step = make_collision_union(first=1, after=after)
            out = await step.execute(1, [])
            conn = out[0]
            for edge in conn["edges"]:
                node = edge["node"]
                walked.append((node[TYPENAME_COLUMN], node["id"]))
            if not conn["pageInfo"]["hasNextPage"]:
                break
            after = conn["pageInfo"]["endCursor"]

    # every distinct row seen exactly once, in total order — no drop (the blocker) and no dup.
    assert walked == expected
    assert len(walked) == len(set(walked)) == 4


@pytest.mark.pg
@pytest.mark.asyncio
async def test_cross_branch_keyset_tie_reverse_pages_without_dropping_a_row(collision_seeded):
    """Reverse paging (``last``/``before``) over the same tie is gap/dup-free and symmetric.

    Walking the union backward in pages of ONE must surface the same four distinct rows once,
    reproducing the forward total order reversed — proving the ``__typename`` tie-break makes
    the BEFORE seek total too (not just the forward AFTER seek).
    """
    expected_forward = [
        ("CollArticle", 1),
        ("CollSnippet", 1),
        ("CollArticle", 2),
        ("CollSnippet", 2),
    ]

    walked = []
    before = None
    with pg_request_context(SQLAlchemyExecutor(get_engine())):
        while True:
            step = make_collision_union(last=1, before=before)
            out = await step.execute(1, [])
            conn = out[0]
            for edge in conn["edges"]:
                node = edge["node"]
                walked.append((node[TYPENAME_COLUMN], node["id"]))
            if not conn["pageInfo"]["hasPreviousPage"]:
                break
            before = conn["pageInfo"]["startCursor"]

    # reverse walk yields the rows from the tail; reversing it reproduces the forward order.
    assert list(reversed(walked)) == expected_forward
    assert len(walked) == len(set(walked)) == 4
