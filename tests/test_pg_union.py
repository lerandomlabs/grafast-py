"""pgUnionAll end-to-end over Postgres: the cross-table polymorphism DB suite.

The cross-table shape (the ``pgUnionAll`` DB tests): a GraphQL union
(``SearchResult = Article | Snippet``) whose concrete types live in SEPARATE tables.
:class:`grafast_py.pg.union.PgUnionAllStep` fetches every member in ONE ``UNION ALL``
statement whose branches project a SHARED, NULL-padded column shape plus a ``__typename``
literal tag, keyset-slices the merged result over the shared order columns, and (per-parent
mode) partitions by the match key. The ``resolve_type_from_tag('__typename')`` bridge resolves
each row at COMPLETION time and the EXISTING completion-time abstract dispatch groups rows by
concrete type and plans each group's sub-selection like a normal object field — so a member's
nested pg relation batches per concrete-type group with NO plan-time polymorphism and NO
core-engine change.

Where :mod:`tests.test_pg_union_all` asserts the SQL SHAPE and the construction-time fail-loud
guards, and :mod:`tests.test_pg_union_dedup` proves the dedup correctness invariant through the
planner, THIS file drives the assembled steps through the real engine against the live
``grafast_demo`` scratch DB and proves the runtime contracts named for the feature:

- a ROOT union across two tables pages them TOGETHER in keyset order — ONE page statement plus
  ONE optional count (gated off the selection set), measured via ``count_sql``;
- ``totalCount`` across the union is the whole-union count, independent of the page;
- a cursor minted under a FOREIGN ordering is rejected loudly by the cursor digest;
- a PER-PARENT union NESTED under a LIST of parents stays O(depth): ONE batched ``UNION ALL``
  page statement across every parent, not one per parent;
- a nested pg relation UNDER a union member (``Article.comments`` / ``Snippet.reactions``)
  batches PER concrete-type group — one statement across every Article, one across every
  Snippet — never per row and never per parent.

DB tests are marked ``pg`` (deselectable ``-m 'not pg'``); they touch ONLY the ``grafast_demo``
schema via the dedicated union-member + child-table fixtures, perturbing nothing else.
"""

import pytest
import pytest_asyncio
from graphql import graphql

from grafast_py.context import GrafastExecutionContext
from grafast_py.pg.cursor import encode_keyset_cursor
from grafast_py.pg.engine import count_sql, dispose_engine, get_engine
from grafast_py.pg.executor import SQLAlchemyExecutor, pg_request_context
from grafast_py.pg.ordering import OrderTerm
from examples.poly_schema import build_poly_schema
from examples.seed import (
    setup_demo_schema,
    setup_media_table,
    setup_media_tags_table,
    setup_union_member_child_tables,
    setup_union_member_tables,
)

pytestmark = [pytest.mark.pg, pytest.mark.asyncio]


@pytest_asyncio.fixture
async def union_schema():
    """Build the poly schema after (re)seeding the union member + child tables (fresh engine).

    Function-scoped so each test runs on its own event loop with a fresh engine (the async pool
    is bound to the creating loop). ``setup_demo_schema`` (re)creates the schema and the
    ``authors`` table the per-parent ``Author.activity`` union keys on; the union-member and
    child fixtures add their tables independently of the authors/posts/comments fixtures.
    """
    await dispose_engine()
    await setup_demo_schema()
    await setup_media_table()
    await setup_media_tags_table()
    await setup_union_member_tables()
    await setup_union_member_child_tables()
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


# ----------------------------------------- root union: paged together, ONE page + optional count


async def test_root_union_pages_both_tables_together_in_keyset_order(union_schema):
    """A ROOT union interleaves both member tables into ONE keyset-ordered page (1 statement).

    articles/snippets interleave by ``created`` — a1(08:00) < s1(09:00) < a2(10:00) < s2(11:00)
    < a3(14:00) < s3(15:00). The union pages them into one ordered slice (each row's concrete
    type resolved off the ``__typename`` tag at completion), and WITHOUT ``totalCount`` selected
    the whole union costs exactly ONE statement — the page query, no separate count.
    """
    query = """
    {
      search(first: 4) {
        nodes {
          __typename
          ... on Article { id headline wordCount }
          ... on Snippet { id body }
        }
        pageInfo { hasNextPage }
      }
    }
    """
    with count_sql(get_engine()) as counter:
        result = await run(union_schema, query)

    assert result.errors is None
    # the page query ONLY — the count is selection-gated and totalCount was not asked for.
    assert counter.count == 1

    nodes = result.data["search"]["nodes"]
    assert [n["__typename"] for n in nodes] == ["Article", "Snippet", "Article", "Snippet"]
    # the type-specific columns read from the right member (NULL-padded peers never leak in).
    assert nodes[0]["headline"] == "alpha headline"
    assert nodes[0]["wordCount"] == 120
    assert "body" not in nodes[0]
    assert nodes[1]["body"] == "first snippet"
    assert "headline" not in nodes[1]
    # 6 rows total, page of 4 -> a next page remains.
    assert result.data["search"]["pageInfo"]["hasNextPage"] is True


async def test_root_union_total_count_is_one_extra_statement(union_schema):
    """Selecting ``totalCount`` adds exactly ONE statement (the page + the count = 2)."""
    query = """
    {
      search(first: 4) {
        totalCount
        nodes { __typename }
      }
    }
    """
    with count_sql(get_engine()) as counter:
        result = await run(union_schema, query)

    assert result.errors is None
    # page (1) + count (1) across the whole union — two statements regardless of member count.
    assert counter.count == 2
    # the count is the whole-union total, independent of the page size.
    assert result.data["search"]["totalCount"] == 6
    assert len(result.data["search"]["nodes"]) == 4


async def test_root_union_forward_paging_walks_the_union_without_overlap(union_schema):
    """``after`` seeks past the first page with no overlap/gap across both members."""
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
    # the remaining two rows, in order, no overlap with page 1.
    assert [(n["__typename"], n["id"]) for n in nodes2] == [("Article", 3), ("Snippet", 3)]
    assert page2.data["search"]["pageInfo"]["hasNextPage"] is False


async def test_foreign_order_cursor_is_rejected_by_the_digest(union_schema):
    """A cursor minted under a DIFFERENT ordering is rejected LOUDLY by the cursor digest.

    The union orders by (created, id); a cursor minted against a ``rank`` order carries a
    different order digest, so decoding it against this union's order raises at PLAN time
    (during the union step's construction, the earliest point) rather than seeking to a
    meaningless position. The engine plans before executing, so the digest mismatch propagates
    as a loud ``ValueError`` — the union never issues a misapplied page.
    """
    foreign = encode_keyset_cursor((OrderTerm("rank"),), {"rank": 5})
    with pytest.raises(ValueError, match="different ordering"):
        await run(
            union_schema,
            """
            query ($after: String!) {
              search(first: 2, after: $after) { nodes { __typename } }
            }
            """,
            {"after": foreign},
        )


# --------------------------------- per-parent union NESTED under a list of parents: O(depth)


async def test_per_parent_union_nested_under_authors_is_one_statement(union_schema):
    """A per-parent union under a LIST of authors stays ONE batched statement (O(depth)).

    ``authors`` returns 3 rows; each author's ``activity`` is a per-parent union keyed on the
    author id (matched against each member's ``owner_id``). The window partitions by owner, so
    EVERY author's page slices independently in the ONE batched ``UNION ALL`` statement — the
    cost is O(depth) (authors select + the one union page), NOT one union per author.
    """
    query = """
    {
      authors {
        id
        activity(first: 3) {
          nodes { __typename ... on Article { id } ... on Snippet { id } }
        }
      }
    }
    """
    with count_sql(get_engine()) as counter:
        result = await run(union_schema, query)

    assert result.errors is None
    # authors (1) + the single batched per-parent union page (1) = 2 statements across all 3
    # authors — totalCount is not selected, so no count statement, and crucially NOT one union
    # statement per author.
    assert counter.count == 2

    authors = result.data["authors"]
    by_id = {a["id"]: a for a in authors}
    # author 1 owns 2 articles + 2 snippets; first 3 by created: a1(08), s1(09), a2(10).
    assert [(n["__typename"], n["id"]) for n in by_id[1]["activity"]["nodes"]] == [
        ("Article", 1),
        ("Snippet", 1),
        ("Article", 2),
    ]
    # author 2 owns 1 article + 1 snippet (created 14:00 < 15:00).
    assert [(n["__typename"], n["id"]) for n in by_id[2]["activity"]["nodes"]] == [
        ("Article", 3),
        ("Snippet", 3),
    ]
    # author 3 owns nothing -> an empty connection (the empty-parent case in the same batch).
    assert by_id[3]["activity"]["nodes"] == []


async def test_per_parent_union_total_count_scatters_per_author(union_schema):
    """``totalCount`` under a nested per-parent union is each author's own whole-union total.

    One grouped count statement across the bucket scatters each author's total back, so the
    per-parent count is correct even on an author's terminal page or empty set. The whole query
    is still O(depth): authors (1) + union page (1) + grouped count (1) = 3.
    """
    query = """
    {
      authors {
        id
        activity(first: 1) {
          totalCount
          nodes { __typename }
        }
      }
    }
    """
    with count_sql(get_engine()) as counter:
        result = await run(union_schema, query)

    assert result.errors is None
    assert counter.count == 3

    totals = {a["id"]: a["activity"]["totalCount"] for a in result.data["authors"]}
    # author 1: 2 articles + 2 snippets; author 2: 1 + 1; author 3: none.
    assert totals == {1: 4, 2: 2, 3: 0}
    # each author still only got its first node despite the larger total.
    counts = {a["id"]: len(a["activity"]["nodes"]) for a in result.data["authors"]}
    assert counts == {1: 1, 2: 1, 3: 0}


# ------------------------------- nested relation under a union member batches per type-group


async def test_nested_relation_under_union_member_batches_per_type_group(union_schema):
    """``Article.comments`` / ``Snippet.reactions`` batch PER concrete-type group (O(depth)).

    Both relations are hasMany pg relations selected on their respective union member. Each
    chains off a CONCRETE type resolved at completion time, so the Article group issues ONE
    batched ``comments`` statement across every Article in the page and the Snippet group ONE
    batched ``reactions`` statement across every Snippet: search page (1) + Article.comments (1)
    + Snippet.reactions (1) = 3. That is O(depth) per type-group, never O(rows).
    """
    query = """
    {
      search(first: 6) {
        nodes {
          __typename
          ... on Article { id comments { body } }
          ... on Snippet { id reactions { emoji } }
        }
      }
    }
    """
    with count_sql(get_engine()) as counter:
        result = await run(union_schema, query)

    assert result.errors is None
    assert counter.count == 3

    nodes = result.data["search"]["nodes"]
    articles = {n["id"]: n for n in nodes if n["__typename"] == "Article"}
    snippets = {n["id"]: n for n in nodes if n["__typename"] == "Snippet"}
    # article 1 -> 2 comments, article 2 -> 1, article 3 -> none (the empty-parent case).
    assert sorted(c["body"] for c in articles[1]["comments"]) == ["agreed", "great read"]
    assert [c["body"] for c in articles[2]["comments"]] == ["needs sources"]
    assert articles[3]["comments"] == []
    # snippet 1 -> 1 reaction, snippet 2 -> none, snippet 3 -> 2 reactions.
    assert [r["emoji"] for r in snippets[1]["reactions"]] == ["thumbsup"]
    assert snippets[2]["reactions"] == []
    assert sorted(r["emoji"] for r in snippets[3]["reactions"]) == ["fire", "heart"]


async def test_relation_under_one_member_skips_the_other_group(union_schema):
    """Selecting a relation on a SINGLE member adds exactly one statement (that group only).

    With ``comments`` selected only under ``... on Article``, the Snippet group contributes no
    relation statement: search page (1) + Article.comments (1) = 2. So the relation statement
    count tracks the number of TYPE-GROUPS that select a relation, not the number of members in
    the result and certainly not the number of rows.
    """
    query = """
    {
      search(first: 6) {
        nodes {
          __typename
          ... on Article { id comments { body } }
          ... on Snippet { id }
        }
      }
    }
    """
    with count_sql(get_engine()) as counter:
        result = await run(union_schema, query)

    assert result.errors is None
    assert counter.count == 2

    nodes = result.data["search"]["nodes"]
    articles = {n["id"]: n for n in nodes if n["__typename"] == "Article"}
    snippets = [n for n in nodes if n["__typename"] == "Snippet"]
    assert sorted(c["body"] for c in articles[1]["comments"]) == ["agreed", "great read"]
    # the snippet rows carry no reactions key (the relation was not selected on Snippet).
    assert all("reactions" not in s for s in snippets)
