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
from datetime import datetime, timezone

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


async def setup_things_table() -> None:
    """Create + seed ``grafast_demo.things`` (a NULLABLE orderable column) idempotently.

    A dedicated fixture for NULLS FIRST/LAST ordering: ``rank`` is nullable so the
    placement of NULL rows is observable. Lives in ``grafast_demo`` alongside the demo
    tables but is independent of them, so it does not perturb the authors/posts/comments
    parity fixtures. Run AFTER :func:`setup_demo_schema` (which creates the schema).
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP TABLE IF EXISTS {DEMO_SCHEMA}.things"))
        await conn.execute(
            text(
                f"""
                CREATE TABLE {DEMO_SCHEMA}.things (
                    id   integer PRIMARY KEY,
                    rank integer NULL
                )
                """
            )
        )
        await conn.execute(
            text(f"INSERT INTO {DEMO_SCHEMA}.things (id, rank) VALUES (:id, :rank)"),
            [
                {"id": 1, "rank": 10},
                {"id": 2, "rank": None},
                {"id": 3, "rank": 20},
                {"id": 4, "rank": None},
                {"id": 5, "rank": 30},
            ],
        )


async def setup_keyset_table() -> None:
    """Create + seed ``grafast_demo.keyset_rows`` for keyset-cursor tests idempotently.

    A dedicated fixture for Phase-6 keyset paging over NULLABLE / DESC / multi-key / and
    non-native (timestamptz, numeric) columns, modelled on the keyset probe. ``rank`` is
    nullable (3 NULLs) with duplicate values (so duplicate-rank cursors and the NULL
    boundary are exercised); ``created`` (timestamptz) and ``price`` (numeric) exercise the
    text-origin cursor round-trip; ``owner_id`` groups rows so the same table serves a
    hasMany-style connection lookup. Lives in ``grafast_demo`` alongside the demo tables but
    is independent of them, so it does not perturb the authors/posts/comments fixtures. Run
    AFTER :func:`setup_demo_schema` (which creates the schema).
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP TABLE IF EXISTS {DEMO_SCHEMA}.keyset_rows"))
        await conn.execute(
            text(
                f"""
                CREATE TABLE {DEMO_SCHEMA}.keyset_rows (
                    id       integer PRIMARY KEY,
                    owner_id integer NOT NULL,
                    rank     integer NULL,
                    name     text NOT NULL,
                    created  timestamptz NOT NULL,
                    price    numeric(12,2) NOT NULL
                )
                """
            )
        )
        # 12 rows, ALL owner 1: rank has 3 NULLs (id 2,5,9), dup 10 (id 1,4,7), dup 20
        # (id 3,8), plus 5,25,30,40 — covering the NULL boundary, duplicate-rank cursors,
        # and both ends. created/price are monotone in id so a text-origin keyset is checkable.
        base = datetime(2024, 1, 1, 5, 0, 0, tzinfo=timezone.utc)
        ranks = {1: 10, 2: None, 3: 20, 4: 10, 5: None, 6: 5, 7: 10, 8: 20, 9: None, 10: 25, 11: 30, 12: 40}
        rows = [
            {
                "id": i,
                "owner_id": 1,
                "rank": ranks[i],
                "name": f"name-{i:02d}",
                "created": base.replace(day=i),
                "price": f"{i * 1.25:.2f}",
            }
            for i in range(1, 13)
        ]
        await conn.execute(
            text(
                f"INSERT INTO {DEMO_SCHEMA}.keyset_rows"
                " (id, owner_id, rank, name, created, price)"
                " VALUES (:id, :owner_id, :rank, :name, :created, :price)"
            ),
            rows,
        )


async def setup_widgets_table() -> None:
    """Create + seed ``grafast_demo.widgets`` (a soft-delete / status fixture) idempotently.

    A dedicated fixture for Phase 4 WHERE-customization: ``deleted_at`` is nullable (a
    soft-delete flag) and ``status`` is a categorical column (tenant/visibility scoping),
    so resource-customizer and per-plan ``.where()`` predicates have observable effects.
    ``owner_id`` groups rows so a hasMany-style ``match_column`` lookup is exercisable.
    Lives in ``grafast_demo`` alongside the demo tables but is independent of them, so it
    does not perturb the authors/posts/comments parity fixtures. Run AFTER
    :func:`setup_demo_schema` (which creates the schema).
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP TABLE IF EXISTS {DEMO_SCHEMA}.widgets"))
        await conn.execute(
            text(
                f"""
                CREATE TABLE {DEMO_SCHEMA}.widgets (
                    id         integer PRIMARY KEY,
                    owner_id   integer NOT NULL,
                    title      text NOT NULL,
                    status     text NOT NULL,
                    deleted_at timestamptz NULL
                )
                """
            )
        )
        # owners 1 and 2; each owns 3 widgets. Per owner: one deleted, one draft, one
        # published — so soft-delete and status filters each remove a known subset and
        # paging over the FILTERED set is observable.
        gone = datetime(2020, 1, 1, tzinfo=timezone.utc)
        rows = [
            (1, 1, "w1", "published", None),
            (2, 1, "w2", "draft", None),
            (3, 1, "w3", "published", gone),
            (4, 2, "w4", "published", None),
            (5, 2, "w5", "draft", None),
            (6, 2, "w6", "published", gone),
        ]
        await conn.execute(
            text(
                f"INSERT INTO {DEMO_SCHEMA}.widgets (id, owner_id, title, status, deleted_at)"
                " VALUES (:id, :owner_id, :title, :status, :deleted_at)"
            ),
            [
                {
                    "id": i,
                    "owner_id": o,
                    "title": t,
                    "status": s,
                    "deleted_at": d,
                }
                for i, o, t, s, d in rows
            ],
        )


async def setup_settings_probe_view() -> None:
    """Create ``grafast_demo.setting_probe``: a view exposing a per-request GUC as a column.

    A Phase-5 pgSettings fixture independent of RLS: each row carries
    ``current_setting('app.demo', true)`` in its ``demo`` column, so any step type that
    selects it OBSERVES the GUC the executor set for the request — proving the
    ``set_config`` is applied in the SAME transaction as the query, regardless of the
    connecting role (so it works even though the scratch-DB superuser bypasses RLS).
    ``owner_id``/``id`` (1 and 2) give a key column for the hasMany/connection step shapes.
    Idempotent; lives in ``grafast_demo`` and does not touch authors/posts/comments. Run
    AFTER :func:`setup_demo_schema` (which creates the schema).
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP VIEW IF EXISTS {DEMO_SCHEMA}.setting_probe"))
        await conn.execute(
            text(
                f"""
                CREATE VIEW {DEMO_SCHEMA}.setting_probe AS
                SELECT g AS id,
                       g AS owner_id,
                       current_setting('app.demo', true) AS demo
                FROM generate_series(1, 2) AS g
                """
            )
        )


async def setup_rls_table() -> None:
    """Create + seed ``grafast_demo.secret_notes`` with ROW LEVEL SECURITY idempotently.

    A Phase-5 RLS fixture: rows are owned by ``owner`` (1 or 2) and the table ENABLEs +
    FORCEs row level security with a policy that admits only rows whose ``owner`` equals
    the per-request GUC ``current_setting('app.owner', true)``. So a query run with
    ``settings={'app.owner': '1'}`` should see only owner-1 rows — IF the connecting role
    is subject to RLS (a superuser / BYPASSRLS role bypasses it unconditionally). Lives in
    ``grafast_demo`` alongside the demo tables but is independent of them, so it does not
    perturb the authors/posts/comments parity fixtures. Run AFTER :func:`setup_demo_schema`
    (which creates the schema).
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP TABLE IF EXISTS {DEMO_SCHEMA}.secret_notes"))
        await conn.execute(
            text(
                f"""
                CREATE TABLE {DEMO_SCHEMA}.secret_notes (
                    id    integer PRIMARY KEY,
                    owner integer NOT NULL,
                    body  text NOT NULL
                )
                """
            )
        )
        await conn.execute(
            text(f"ALTER TABLE {DEMO_SCHEMA}.secret_notes ENABLE ROW LEVEL SECURITY")
        )
        # FORCE subjects the table OWNER to RLS too (otherwise the owner is exempt); it
        # does NOT subject a superuser/BYPASSRLS role — that bypass is unconditional.
        await conn.execute(
            text(f"ALTER TABLE {DEMO_SCHEMA}.secret_notes FORCE ROW LEVEL SECURITY")
        )
        await conn.execute(
            text(
                f"""
                CREATE POLICY owner_isolation ON {DEMO_SCHEMA}.secret_notes
                USING (owner = current_setting('app.owner', true)::int)
                """
            )
        )
        # owner 1 and owner 2 each own two notes.
        await conn.execute(
            text(
                f"INSERT INTO {DEMO_SCHEMA}.secret_notes (id, owner, body)"
                " VALUES (:id, :owner, :body)"
            ),
            [
                {"id": 1, "owner": 1, "body": "owner-1 note a"},
                {"id": 2, "owner": 1, "body": "owner-1 note b"},
                {"id": 3, "owner": 2, "body": "owner-2 note a"},
                {"id": 4, "owner": 2, "body": "owner-2 note b"},
            ],
        )


async def setup_labels_table() -> None:
    """Create + seed ``grafast_demo.labels`` (a codec / computed-column fixture) idempotently.

    A Phase-7 fixture for the minimal per-attribute codec and computed columns: ``code`` is
    a lowercase text value a ``to_py`` hook can uppercase, and a computed attribute can
    derive e.g. ``upper(code)`` over it. ``owner_id`` groups rows so the same table serves a
    hasMany-style relation AND a Relay connection lookup (so the codec/computed projection is
    observable through a plain select, a window slice, and a connection node). Lives in
    ``grafast_demo`` alongside the demo tables but is independent of them, so it does not
    perturb the authors/posts/comments parity fixtures. Run AFTER :func:`setup_demo_schema`
    (which creates the schema).
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP TABLE IF EXISTS {DEMO_SCHEMA}.labels"))
        await conn.execute(
            text(
                f"""
                CREATE TABLE {DEMO_SCHEMA}.labels (
                    id       integer PRIMARY KEY,
                    owner_id integer NOT NULL,
                    code     text NOT NULL
                )
                """
            )
        )
        # owner 1 owns 4 labels (so first:2 leaves a second page for the connection probe);
        # owner 2 owns 1. code is lowercase so an uppercasing to_py / upper() computed col
        # is observable.
        rows = [
            (1, 1, "alpha"),
            (2, 1, "bravo"),
            (3, 1, "charlie"),
            (4, 1, "delta"),
            (5, 2, "echo"),
        ]
        await conn.execute(
            text(
                f"INSERT INTO {DEMO_SCHEMA}.labels (id, owner_id, code)"
                " VALUES (:id, :owner_id, :code)"
            ),
            [{"id": i, "owner_id": o, "code": c} for i, o, c in rows],
        )


__all__ = [
    "setup_demo_schema",
    "setup_things_table",
    "setup_keyset_table",
    "setup_widgets_table",
    "setup_settings_probe_view",
    "setup_rls_table",
    "setup_labels_table",
]
