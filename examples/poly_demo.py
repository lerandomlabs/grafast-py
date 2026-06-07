"""End-to-end Postgres polymorphism demo for grafast-py.

Mirrors :mod:`examples.pg_blog` but for INTERFACES and UNIONS backed by Postgres. It
seeds the ``grafast_demo`` polymorphism fixtures, builds the schema from
:mod:`examples.poly_schema`, and runs GraphQL operations through
:class:`grafast_py.GrafastExecutionContext`, printing both the result and the number of
SQL statements each operation issued via :func:`grafast_py.pg.engine.count_sql`.

The point this demo makes (the same one ``pg_blog`` makes for plain relations): GraphQL
polymorphism over Postgres adds NO plan-time machinery and NO per-row fan-out. A
``resolve_type`` bridge tags each row with its concrete type, and the engine's EXISTING
completion-time abstract dispatch groups rows by concrete type and plans each group's
sub-selection like a normal object field. So:

- a single-table interface (``media`` with a ``kind`` discriminator) resolving mixed
  Image/Video rows costs ONE statement, and a nested relation under a concrete type
  (``Image.tags``) batches per concrete-type group — O(depth), not O(rows);
- a cross-table union (``SearchResult = Article | Snippet`` via ``pgUnionAll``) pages both
  member tables together in ONE keyset-ordered ``UNION ALL`` statement, and a per-parent
  union nested under a LIST of authors stays ONE batched statement across every parent.

The ``count_sql`` banners are an actual ``before_cursor_execute`` statement counter, so
each printed count is the real number of round-trips — it tracks DEPTH, not ROWS.

SAFETY: the only database touched is ``grafast_py_test`` and the only schema is
``grafast_demo`` (dropped + recreated idempotently by the seed fixtures).

Run:  uv run python examples/poly_demo.py
"""

import asyncio
import sys
from pathlib import Path

# run as a script: put the repo root on sys.path so the sibling `examples` package
# (poly_schema/seed) is importable as `examples.*`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphql import graphql

from grafast_py import GrafastExecutionContext
from grafast_py.pg.engine import count_sql, dispose_engine, get_engine
from grafast_py.pg.executor import SQLAlchemyExecutor, pg_request_context
from examples.poly_schema import build_poly_schema
from examples.seed import (
    setup_demo_schema,
    setup_media_table,
    setup_media_tags_table,
    setup_union_member_child_tables,
    setup_union_member_tables,
)


async def run(schema, query, variables=None):
    """Execute one operation through our plan-then-execute engine, counting SQL.

    Binds a request-scoped :class:`SQLAlchemyExecutor` so the pg steps run their statements
    via the request executor; ``count_sql`` instruments that same convenience engine.
    """
    engine = get_engine()
    with count_sql(engine) as counter:
        with pg_request_context(SQLAlchemyExecutor(engine)):
            result = await graphql(
                schema,
                query,
                variable_values=variables,
                execution_context_class=GrafastExecutionContext,
            )
    if result.errors:
        raise AssertionError(f"query errored: {result.errors}")
    return result.data, counter


def banner(text: str) -> None:
    print("=" * 72)
    print(text)
    print("=" * 72)


async def seed() -> None:
    """(Re)seed every polymorphism fixture inside ``grafast_demo`` (drop-first, idempotent).

    The demo entrypoint that registers all of the polymorphism fixtures: the single-table
    ``media`` (+ its ``media_tags`` child) interface fixture and the cross-table union member
    tables ``articles`` / ``snippets`` (+ their ``article_comments`` / ``snippet_reactions``
    children). ``setup_demo_schema`` creates the schema and the ``authors`` table the
    per-parent ``Author.activity`` union keys on.
    """
    await setup_demo_schema()
    await setup_media_table()
    await setup_media_tags_table()
    await setup_union_member_tables()
    await setup_union_member_child_tables()


async def main() -> None:
    banner("0. SEED grafast_demo (media interface + articles/snippets union fixtures)")
    await dispose_engine()
    await seed()
    print("   single-table interface: 5 media rows (3 images, 2 videos) + 4 media_tags.")
    print("   cross-table union: 3 articles + 3 snippets (+ comments / reactions),")
    print("   keyed on owner_id; authors 1/2 own activity, author 3 owns none.\n")

    schema = build_poly_schema()

    # ============================================================ single-table interface
    banner("1. INTERFACE — [Media!]! resolves each row's concrete type (ONE SELECT)")
    media_query = """
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
    data, counter = await run(schema, media_query)
    kinds = [m["__typename"] for m in data["media"]]
    print(f"   concrete types in id order: {kinds}")
    print(f"   image[0]: {data['media'][0]}")
    print(f"   video[1]: {data['media'][1]}")
    print(f"   SQL statements issued: {counter.count}  (expected 1 — one media select;")
    print("     completion-time dispatch groups the rows by `kind`, pure projection)\n")

    # ----------------------------------------------------- nested relation per type-group
    banner("2. INTERFACE + nested relation — Image.tags / Video.tags batch per type-group")
    tags_query = """
    {
      media {
        __typename
        ... on Image { id tags { label } }
        ... on Video { id tags { label } }
      }
    }
    """
    data, counter = await run(schema, tags_query)
    by_id = {m["id"]: m for m in data["media"]}
    print(f"   image 1 tags: {[t['label'] for t in by_id[1]['tags']]}")
    print(f"   image 3 tags: {[t['label'] for t in by_id[3]['tags']]}")
    print(f"   video 2 tags: {[t['label'] for t in by_id[2]['tags']]}")
    print(f"   SQL statements issued: {counter.count}  (expected 3 — media select +")
    print("     ONE batched Image.tags across all images + ONE Video.tags across videos)\n")

    # ----------------------------------------------------------- single nullable interface
    banner("3. INTERFACE single — mediaById resolves one row's concrete type")
    data, counter = await run(
        schema,
        """
        {
          present: mediaById(id: 2) {
            __typename title ... on Video { durationSeconds }
          }
          missing: mediaById(id: 999) { __typename id }
        }
        """,
    )
    print(f"   mediaById(2):   {data['present']}")
    print(f"   mediaById(999): {data['missing']}  (a missing id is a clean null,")
    print("     short-circuited before the resolve_type bridge is ever called)")
    print(f"   SQL statements issued: {counter.count}\n")

    # =============================================================== cross-table union
    banner("4. UNION — SearchResult = Article | Snippet pages both tables (ONE UNION ALL)")
    search_query = """
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
    data, counter = await run(schema, search_query)
    nodes = data["search"]["nodes"]
    print("   keyset-merged page (interleaved by `created` across BOTH tables):")
    for n in nodes:
        label = n.get("headline", n.get("body"))
        print(f"     {n['__typename']:<8} id={n['id']}  {label!r}")
    print(f"   hasNextPage: {data['search']['pageInfo']['hasNextPage']}")
    print(f"   SQL statements issued: {counter.count}  (expected 1 — ONE UNION ALL page;")
    print("     totalCount not selected, so no separate count statement)\n")

    # ------------------------------------------------------------ union totalCount gated
    banner("5. UNION totalCount — selection-gated whole-union count (+1 statement)")
    data, counter = await run(
        schema,
        "{ search(first: 4) { totalCount nodes { __typename } } }",
    )
    print(f"   totalCount across the whole union: {data['search']['totalCount']}  (6)")
    print(f"   nodes on this page: {len(data['search']['nodes'])}")
    print(f"   SQL statements issued: {counter.count}  (expected 2 — page + the count,")
    print("     two statements regardless of how many member tables the union spans)\n")

    # --------------------------------------------- per-parent union NESTED under a list
    banner("6. UNION per-parent — Author.activity over a LIST of authors stays O(depth)")
    activity_query = """
    {
      authors {
        id
        activity(first: 3) {
          totalCount
          nodes { __typename ... on Article { id } ... on Snippet { id } }
        }
      }
    }
    """
    data, counter = await run(schema, activity_query)
    for a in data["authors"]:
        act = a["activity"]
        seq = [(n["__typename"], n["id"]) for n in act["nodes"]]
        print(f"   author {a['id']}: totalCount={act['totalCount']} activity={seq}")
    print(f"   SQL statements issued: {counter.count}  (expected 3 — authors select +")
    print("     ONE batched UNION ALL page across ALL authors + ONE grouped count;")
    print("     NOT one union per author — the count tracks DEPTH, not parents)\n")

    # --------------------------------- nested relation under a union member per type-group
    banner("7. UNION + nested relation — Article.comments / Snippet.reactions per group")
    member_rel_query = """
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
    data, counter = await run(schema, member_rel_query)
    nodes = data["search"]["nodes"]
    articles = {n["id"]: n for n in nodes if n["__typename"] == "Article"}
    snippets = {n["id"]: n for n in nodes if n["__typename"] == "Snippet"}
    print(f"   article 1 comments:  {[c['body'] for c in articles[1]['comments']]}")
    print(f"   article 3 comments:  {articles[3]['comments']}  (empty-parent case)")
    print(f"   snippet 3 reactions: {[r['emoji'] for r in snippets[3]['reactions']]}")
    print(f"   SQL statements issued: {counter.count}  (expected 3 — union page +")
    print("     ONE Article.comments across every Article + ONE Snippet.reactions across")
    print("     every Snippet; the relation count tracks TYPE-GROUPS, never rows)\n")

    banner("DONE — Postgres polymorphism composes with completion-time dispatch (O(depth)).")

    await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
