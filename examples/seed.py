"""Idempotent demo schema + seed data for the Postgres data source.

``setup_demo_schema`` drops and recreates the ``grafast_demo`` schema with three
related tables — ``authors``, ``posts``, ``comments`` — wired by foreign keys, then
inserts deterministic seed rows. Every statement is confined to ``grafast_demo`` in
``grafast_py_test``; nothing else on the server is touched.

The shape (authors -> posts -> comments, plus comment.author back to authors) gives
a genuine multi-layer relation graph so a nested query exercises hasOne and hasMany
chaining and the O(depth) batching gate.
"""

import os

from sqlalchemy import text

from grafast_py.pg.engine import get_engine

# Demo/test fixtures only — NOT part of the shipped library. The demo data lives in
# this schema of a local scratch database; importing this module points the pg engine
# at that scratch DB by default (setdefault, so an explicit GRAFAST_PG_URL wins) so the
# examples/benchmarks/tests touch ONLY the scratch DB and never another database on the
# server. A real consumer of grafast_py sets their own GRAFAST_PG_URL / configure_engine.
DEMO_SCHEMA = "grafast_demo"
os.environ.setdefault("GRAFAST_PG_URL", "postgresql+asyncpg:///grafast_py_test")

# deterministic seed: 3 authors; author i has (i+1) posts; each post has 2 comments.
_AUTHORS = [
    (1, "Ada Lovelace"),
    (2, "Alan Turing"),
    (3, "Grace Hopper"),
]


def _seed_posts() -> list[tuple[int, int, str]]:
    """Return (id, author_id, title) rows: author i has i+1 posts."""
    rows: list[tuple[int, int, str]] = []
    post_id = 1
    for author_id, _ in _AUTHORS:
        for n in range(author_id + 1):
            rows.append((post_id, author_id, f"Post {post_id} by author {author_id}"))
            post_id += 1
    return rows


def _seed_comments(posts: list[tuple[int, int, str]]) -> list[tuple[int, int, int, str]]:
    """Return (id, post_id, author_id, body) rows: 2 comments per post."""
    rows: list[tuple[int, int, int, str]] = []
    comment_id = 1
    for post_id, _author_id, _title in posts:
        for k in range(2):
            # commenter cycles through the authors so comment.author resolves too
            commenter = (comment_id % len(_AUTHORS)) + 1
            rows.append((comment_id, post_id, commenter, f"Comment {comment_id} on post {post_id}"))
            comment_id += 1
    return rows


async def setup_demo_schema() -> None:
    """Drop + recreate ``grafast_demo`` and load seed data (idempotent)."""
    posts = _seed_posts()
    comments = _seed_comments(posts)

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP SCHEMA IF EXISTS {DEMO_SCHEMA} CASCADE"))
        await conn.execute(text(f"CREATE SCHEMA {DEMO_SCHEMA}"))

        await conn.execute(
            text(
                f"""
                CREATE TABLE {DEMO_SCHEMA}.authors (
                    id   integer PRIMARY KEY,
                    name text NOT NULL
                )
                """
            )
        )
        await conn.execute(
            text(
                f"""
                CREATE TABLE {DEMO_SCHEMA}.posts (
                    id        integer PRIMARY KEY,
                    author_id integer NOT NULL REFERENCES {DEMO_SCHEMA}.authors (id),
                    title     text NOT NULL
                )
                """
            )
        )
        await conn.execute(
            text(
                f"""
                CREATE TABLE {DEMO_SCHEMA}.comments (
                    id        integer PRIMARY KEY,
                    post_id   integer NOT NULL REFERENCES {DEMO_SCHEMA}.posts (id),
                    author_id integer NOT NULL REFERENCES {DEMO_SCHEMA}.authors (id),
                    body      text NOT NULL
                )
                """
            )
        )

        await conn.execute(
            text(f"INSERT INTO {DEMO_SCHEMA}.authors (id, name) VALUES (:id, :name)"),
            [{"id": a, "name": n} for a, n in _AUTHORS],
        )
        await conn.execute(
            text(
                f"INSERT INTO {DEMO_SCHEMA}.posts (id, author_id, title)"
                " VALUES (:id, :author_id, :title)"
            ),
            [{"id": p, "author_id": a, "title": t} for p, a, t in posts],
        )
        await conn.execute(
            text(
                f"INSERT INTO {DEMO_SCHEMA}.comments (id, post_id, author_id, body)"
                " VALUES (:id, :post_id, :author_id, :body)"
            ),
            [
                {"id": c, "post_id": p, "author_id": a, "body": b}
                for c, p, a, b in comments
            ],
        )


__all__ = ["setup_demo_schema"]
