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
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from grafast_py.pg.engine import URL_ENV_VAR, get_engine

# Demo/test fixtures only — NOT part of the shipped library. The demo data lives in
# this schema of a local scratch database; importing this module points the pg engine
# at that scratch DB by default (setdefault, so an explicit GRAFAST_PG_URL wins) so the
# examples/benchmarks/tests touch ONLY the scratch DB and never another database on the
# server. A real consumer of grafast_py sets their own GRAFAST_PG_URL / configure_engine.
DEMO_SCHEMA = "grafast_demo"
os.environ.setdefault("GRAFAST_PG_URL", "postgresql+asyncpg:///grafast_py_test")

# A fixed Postgres advisory-lock key serializing every reseed of the shared
# ``grafast_demo`` schema ACROSS PROCESSES. Two pytest processes share the one local
# ``grafast_py_test`` database, so two reseeders dropping+recreating ``grafast_demo``
# concurrently would corrupt each other's snapshot mid-run (the soak seeds 50 authors and
# asserts exactly that; a regular fixture reseeds 3). Every reseeder takes this lock (the
# regular fixtures transaction-scoped, the soak session-scoped for its whole run), so the
# reseed+read windows can never interleave. The key is arbitrary but stable — any reseeder
# of this schema, in any process, must use THIS value.
RESEED_LOCK_KEY = 0x6772_6166  # 'graf'


@asynccontextmanager
async def reseed_lock():
    """Hold the cross-process ``grafast_demo`` reseed lock for the duration of the block.

    Acquires a SESSION-level Postgres advisory lock on :data:`RESEED_LOCK_KEY` and releases
    it on exit. Session-level (not transaction-level) so a caller can hold it across MANY
    transactions — the soak holds it across its reseed AND its whole concurrent-read run, so
    no other process can reseed the schema mid-soak. A regular fixture's transaction-level
    ``pg_advisory_xact_lock`` on the same key contends with this, so it blocks until the soak
    releases (and vice versa).

    The lock is taken over a DEDICATED single-connection engine (same ``grafast_py_test`` URL
    as the shared engine, never another database) rather than the shared engine's pool: the
    soak measures that shared pool's checkout/leak counts, and a held lock connection would
    register as a phantom checkout and a non-empty pool at quiescence, breaking the soak's
    no-leak assertions. An isolated engine keeps the lock entirely off the measured pool.
    """
    url = os.environ.get(URL_ENV_VAR)
    if not url:
        raise RuntimeError(
            f"reseed_lock: {URL_ENV_VAR} is not set — refusing to open a lock "
            "connection without an explicit grafast_py_test URL"
        )
    lock_engine = create_async_engine(url, pool_size=1, max_overflow=0)
    try:
        async with lock_engine.connect() as conn:
            await conn.execute(
                text("SELECT pg_advisory_lock(:key)"), {"key": RESEED_LOCK_KEY}
            )
            try:
                yield
            finally:
                await conn.execute(
                    text("SELECT pg_advisory_unlock(:key)"), {"key": RESEED_LOCK_KEY}
                )
    finally:
        await lock_engine.dispose()


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
        # serialize against any concurrent reseeder (another pytest process, the soak):
        # a transaction-level advisory lock auto-released at commit, contending with the
        # soak's session-level lock on the same key so the two reseeds never interleave.
        await conn.execute(
            text("SELECT pg_advisory_xact_lock(:key)"), {"key": RESEED_LOCK_KEY}
        )
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


async def setup_codec_rows_table() -> None:
    """Create + seed ``grafast_demo.codec_rows`` (the array/range/enum/composite codec fixture).

    A dedicated fixture for the codec registry's RECURSIVE container codecs, exercising every
    kind in one table so a plain select observes each decode path:

    - ``tags`` (``text[]``) — an ARRAY codec maps an element ``to_py`` over each element;
    - ``scores`` (``numeric[]``) — an ARRAY over a non-native element, so the array carries
      the ``numeric[]`` cast type for the keyset path while asyncpg returns ``Decimal``s;
    - ``span`` (``int4range``) / ``period`` (``tstzrange``) — RANGE codecs decode the
      ``asyncpg.Range`` into a plain ``{lower, upper, ...}`` dict;
    - ``mood`` (a schema-local ENUM) — an ENUM codec validates the label set;
    - ``point`` (a schema-local COMPOSITE ``(x int, y int)``) — a COMPOSITE codec zips the
      record into a ``{x, y}`` dict.

    The ENUM + composite are CREATEd inside ``grafast_demo`` (schema-scoped, dropped first so
    re-seeding is idempotent) — never a server-global type. ``owner_id`` groups rows so the
    same table serves a hasMany-style ``match_column`` lookup. Lives in ``grafast_demo``
    alongside the demo tables but is independent of them, so it does not perturb the
    authors/posts/comments parity fixtures. Run AFTER :func:`setup_demo_schema` (which creates
    the schema).
    """
    engine = get_engine()
    async with engine.begin() as conn:
        # drop the table before its dependent types (a column still referencing the type
        # would block the DROP TYPE); recreate the types fresh so re-seeding is idempotent.
        await conn.execute(text(f"DROP TABLE IF EXISTS {DEMO_SCHEMA}.codec_rows"))
        await conn.execute(text(f"DROP TYPE IF EXISTS {DEMO_SCHEMA}.codec_mood"))
        await conn.execute(text(f"DROP TYPE IF EXISTS {DEMO_SCHEMA}.codec_point"))
        await conn.execute(
            text(f"CREATE TYPE {DEMO_SCHEMA}.codec_mood AS ENUM ('happy', 'sad', 'meh')")
        )
        await conn.execute(
            text(f"CREATE TYPE {DEMO_SCHEMA}.codec_point AS (x integer, y integer)")
        )
        await conn.execute(
            text(
                f"""
                CREATE TABLE {DEMO_SCHEMA}.codec_rows (
                    id       integer PRIMARY KEY,
                    owner_id integer NOT NULL,
                    tags     text[] NOT NULL,
                    scores   numeric(8,2)[] NOT NULL,
                    span     int4range NOT NULL,
                    period   tstzrange NOT NULL,
                    mood     {DEMO_SCHEMA}.codec_mood NOT NULL,
                    point    {DEMO_SCHEMA}.codec_point NOT NULL
                )
                """
            )
        )
        # owner 1 owns rows 1..3, owner 2 owns row 4. tags are lowercase so an uppercasing
        # element codec is observable; the ranges/enum/composite values are deterministic so
        # the decoded dicts are checkable.
        base = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        rows = [
            (1, 1, ["alpha", "beta"], ["1.50", "2.25"], (1, 5), "happy", (3, 4)),
            (2, 1, ["gamma"], ["9.99"], (10, 20), "sad", (5, 6)),
            (3, 1, ["delta", "epsilon"], ["0.00"], (0, 1), "meh", (7, 8)),
            (4, 2, ["zeta"], ["3.33", "4.44"], (100, 200), "happy", (9, 10)),
        ]
        # the range columns are built SERVER-SIDE (int4range(...) / tstzrange(...)) from
        # plain int / timestamp binds: asyncpg pre-binds a range-typed parameter as a Range
        # object, so a text-literal CAST would error — building the range in SQL keeps the
        # binds plain scalars.
        await conn.execute(
            text(
                f"""
                INSERT INTO {DEMO_SCHEMA}.codec_rows
                  (id, owner_id, tags, scores, span, period, mood, point)
                VALUES
                  (:id, :owner_id, :tags, CAST(:scores AS numeric(8,2)[]),
                   int4range(:span_lower, :span_upper, '[)'),
                   tstzrange(:period_lower, :period_upper, '[)'),
                   CAST(:mood AS {DEMO_SCHEMA}.codec_mood),
                   ROW(:px, :py)::{DEMO_SCHEMA}.codec_point)
                """
            ),
            [
                {
                    "id": i,
                    "owner_id": o,
                    "tags": tags,
                    "scores": scores,
                    "span_lower": span[0],
                    "span_upper": span[1],
                    "period_lower": base.replace(day=i),
                    "period_upper": base.replace(day=i + 1),
                    "mood": mood,
                    "px": pt[0],
                    "py": pt[1],
                }
                for i, o, tags, scores, span, mood, pt in rows
            ],
        )


async def setup_line_items_table() -> None:
    """Create + seed ``grafast_demo.line_items`` (the connection-aggregate fixture) idempotently.

    A dedicated fixture for the SEPARATE batched connection aggregate
    (``sum``/``avg``/``min``/``max``/``count``, optionally GROUPed). ``order_id`` groups the
    rows so an ``order -> line_items`` hasMany connection aggregates a known per-parent set:
    ``quantity`` (integer) and ``price`` (numeric) give checkable sum/avg/min/max, ``category``
    (a categorical column) gives an extra GROUP BY key for the grouped-aggregate path, and
    ``status`` gives a WHERE-filterable column so the aggregate-under-filter case is
    observable. Lives in ``grafast_demo`` alongside the demo tables but is independent of
    them, so it does not perturb the authors/posts/comments parity fixtures. Run AFTER
    :func:`setup_demo_schema` (which creates the schema).
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP TABLE IF EXISTS {DEMO_SCHEMA}.line_items"))
        await conn.execute(
            text(
                f"""
                CREATE TABLE {DEMO_SCHEMA}.line_items (
                    id       integer PRIMARY KEY,
                    order_id integer NOT NULL,
                    category text NOT NULL,
                    status   text NOT NULL,
                    quantity integer NOT NULL,
                    price    numeric(10,2) NOT NULL
                )
                """
            )
        )
        # order 1 owns 4 items across 2 categories, order 2 owns 2 items, order 3 owns NONE
        # (so the empty-aggregate path is observable). quantity/price are deterministic so
        # the per-order sum/avg/min/max and per-category grouped sub-totals are checkable.
        # One item is 'void' so a status filter removes a known row from the aggregate.
        rows = [
            (1, 1, "book", "ok", 2, "10.00"),
            (2, 1, "book", "ok", 3, "20.00"),
            (3, 1, "media", "ok", 1, "5.00"),
            (4, 1, "media", "void", 10, "99.00"),
            (5, 2, "book", "ok", 4, "8.00"),
            (6, 2, "media", "ok", 5, "12.50"),
        ]
        await conn.execute(
            text(
                f"INSERT INTO {DEMO_SCHEMA}.line_items"
                " (id, order_id, category, status, quantity, price)"
                " VALUES (:id, :order_id, :category, :status, :quantity,"
                " CAST(:price AS numeric(10,2)))"
            ),
            [
                {
                    "id": i,
                    "order_id": o,
                    "category": c,
                    "status": s,
                    "quantity": q,
                    "price": p,
                }
                for i, o, c, s, q, p in rows
            ],
        )


async def setup_composite_tables() -> None:
    """Create + seed ``grafast_demo.regions`` and ``grafast_demo.stores`` idempotently.

    A dedicated fixture for the COMPOSITE-key match path: ``regions`` has a two-column
    primary key ``(org_id, region_id)`` and ``stores`` carries a two-column foreign key
    ``(org_id, region_id)`` back to it. So a region -> stores hasMany and a store -> region
    hasOne both match on the column TUPLE (the tuple-IN skeleton), and a row whose
    ``(org, region)`` pair appears under several single-column values proves the match is
    over the whole tuple — never a single column. ``label`` / ``name`` make the decoded rows
    checkable. Lives in ``grafast_demo`` alongside the demo tables but is independent of
    them, so it does not perturb the authors/posts/comments parity fixtures. Run AFTER
    :func:`setup_demo_schema` (which creates the schema).
    """
    engine = get_engine()
    async with engine.begin() as conn:
        # drop the child first (its FK references the parent), then the parent.
        await conn.execute(text(f"DROP TABLE IF EXISTS {DEMO_SCHEMA}.stores"))
        await conn.execute(text(f"DROP TABLE IF EXISTS {DEMO_SCHEMA}.regions"))
        await conn.execute(
            text(
                f"""
                CREATE TABLE {DEMO_SCHEMA}.regions (
                    org_id    integer NOT NULL,
                    region_id integer NOT NULL,
                    label     text NOT NULL,
                    PRIMARY KEY (org_id, region_id)
                )
                """
            )
        )
        await conn.execute(
            text(
                f"""
                CREATE TABLE {DEMO_SCHEMA}.stores (
                    id        integer PRIMARY KEY,
                    org_id    integer NOT NULL,
                    region_id integer NOT NULL,
                    name      text NOT NULL,
                    FOREIGN KEY (org_id, region_id)
                        REFERENCES {DEMO_SCHEMA}.regions (org_id, region_id)
                )
                """
            )
        )
        # two orgs, each with two regions. The (org_id, region_id) pairs deliberately reuse
        # the same single-column values across orgs — org 1/region 1 and org 2/region 1 share
        # region_id 1, and both orgs use org-local region_id 1,2 — so a match on region_id
        # ALONE (or org_id alone) would cross-link rows; only the tuple match is correct.
        regions = [
            (1, 1, "org1-north"),
            (1, 2, "org1-south"),
            (2, 1, "org2-north"),
            (2, 2, "org2-south"),
        ]
        # region (1,1) owns 2 stores, (1,2) owns 1, (2,1) owns 3, (2,2) owns 0 — so per-parent
        # counts differ and the (2,2) empty case is observable.
        stores = [
            (1, 1, 1, "store-a"),
            (2, 1, 1, "store-b"),
            (3, 1, 2, "store-c"),
            (4, 2, 1, "store-d"),
            (5, 2, 1, "store-e"),
            (6, 2, 1, "store-f"),
        ]
        await conn.execute(
            text(
                f"INSERT INTO {DEMO_SCHEMA}.regions (org_id, region_id, label)"
                " VALUES (:org_id, :region_id, :label)"
            ),
            [{"org_id": o, "region_id": r, "label": lb} for o, r, lb in regions],
        )
        await conn.execute(
            text(
                f"INSERT INTO {DEMO_SCHEMA}.stores (id, org_id, region_id, name)"
                " VALUES (:id, :org_id, :region_id, :name)"
            ),
            [
                {"id": i, "org_id": o, "region_id": r, "name": n}
                for i, o, r, n in stores
            ],
        )


async def setup_media_table() -> None:
    """Create + seed ``grafast_demo.media`` (the single-table polymorphism fixture) idempotently.

    A dedicated fixture for Postgres polymorphism over ONE table with a ``kind``
    discriminator: every row is some ``Media`` (interface) — an ``Image`` or a ``Video`` —
    and ``kind`` (``'image'`` / ``'video'``) tells which. ``title`` is the shared interface
    field; ``width``/``height`` are populated only on images (NULL on videos) and
    ``duration_seconds`` only on videos (NULL on images), so a resolve_type bridge keyed on
    ``kind`` must group the rows by concrete type and each concrete type's sub-selection
    reads only its own columns. ``owner_id`` groups rows so the same table serves a
    hasMany-style ``match_column`` lookup AND a root collection. Lives in ``grafast_demo``
    alongside the demo tables but is independent of them, so it does not perturb the
    authors/posts/comments parity fixtures. Run AFTER :func:`setup_demo_schema` (which
    creates the schema).
    """
    engine = get_engine()
    async with engine.begin() as conn:
        # drop the child media_tags first: its FK references media, so a non-cascade DROP of
        # media alone would fail if a previous seed left media_tags behind. The normal test
        # path resets the whole schema first, but dropping child-before-parent here keeps this
        # fixture independently drop-first idempotent (the setup_composite_tables idiom).
        await conn.execute(text(f"DROP TABLE IF EXISTS {DEMO_SCHEMA}.media_tags"))
        await conn.execute(text(f"DROP TABLE IF EXISTS {DEMO_SCHEMA}.media"))
        await conn.execute(
            text(
                f"""
                CREATE TABLE {DEMO_SCHEMA}.media (
                    id               integer PRIMARY KEY,
                    owner_id         integer NOT NULL,
                    kind             text NOT NULL,
                    title            text NOT NULL,
                    width            integer NULL,
                    height           integer NULL,
                    duration_seconds integer NULL
                )
                """
            )
        )
        # owner 1 owns 3 media (2 images, 1 video), owner 2 owns 2 (1 image, 1 video) — so
        # both concrete-type groups are non-empty under several owners and a root list mixes
        # the two types in id order. image rows carry width/height (duration NULL); video rows
        # carry duration_seconds (width/height NULL), so a concrete sub-selection over the
        # wrong columns would read NULL — a misgrouping is observable.
        rows = [
            (1, 1, "image", "sunrise", 1920, 1080, None),
            (2, 1, "video", "timelapse", None, None, 42),
            (3, 1, "image", "portrait", 800, 1200, None),
            (4, 2, "image", "logo", 512, 512, None),
            (5, 2, "video", "promo", None, None, 90),
        ]
        await conn.execute(
            text(
                f"INSERT INTO {DEMO_SCHEMA}.media"
                " (id, owner_id, kind, title, width, height, duration_seconds)"
                " VALUES (:id, :owner_id, :kind, :title, :width, :height, :duration_seconds)"
            ),
            [
                {
                    "id": i,
                    "owner_id": o,
                    "kind": k,
                    "title": t,
                    "width": w,
                    "height": h,
                    "duration_seconds": d,
                }
                for i, o, k, t, w, h, d in rows
            ],
        )


async def setup_media_tags_table() -> None:
    """Create + seed ``grafast_demo.media_tags`` (a nested-relation-under-polymorphism fixture).

    A child table for :func:`setup_media_table`: each tag belongs to one ``media`` row via
    ``media_id``, so an ``Image.tags`` (hasMany) relation chains OFF a concrete type resolved
    at completion time — proving a nested pg relation batches per concrete-type group (one
    statement across all images in the bucket), not per row. ``label`` makes the decoded tag
    rows checkable. Lives in ``grafast_demo``; run AFTER :func:`setup_media_table` (its FK
    references ``media``).
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP TABLE IF EXISTS {DEMO_SCHEMA}.media_tags"))
        await conn.execute(
            text(
                f"""
                CREATE TABLE {DEMO_SCHEMA}.media_tags (
                    id       integer PRIMARY KEY,
                    media_id integer NOT NULL REFERENCES {DEMO_SCHEMA}.media (id),
                    label    text NOT NULL
                )
                """
            )
        )
        # image 1 has 2 tags, image 3 has 1, image 4 has none, video 2 has 1 tag (so a video's
        # tags relation is also exercisable). Per-parent counts differ so the batched scatter
        # back to each parent is observable, and image 4's empty case is covered.
        rows = [
            (1, 1, "nature"),
            (2, 1, "morning"),
            (3, 3, "people"),
            (4, 2, "motion"),
        ]
        await conn.execute(
            text(
                f"INSERT INTO {DEMO_SCHEMA}.media_tags (id, media_id, label)"
                " VALUES (:id, :media_id, :label)"
            ),
            [{"id": i, "media_id": m, "label": lb} for i, m, lb in rows],
        )


async def setup_union_member_tables() -> None:
    """Create + seed ``grafast_demo.articles`` and ``grafast_demo.snippets`` idempotently.

    The CROSS-TABLE polymorphism fixture: a ``SearchResult = Article | Snippet`` union whose
    concrete types live in SEPARATE tables, exercising the pgUnionAll step's ``UNION ALL`` of
    N member tables. Both members share ``id`` / ``owner_id`` / ``created`` (the shared
    projection + keyset order columns) and each carries a type-specific column —
    ``articles.headline`` (with a ``word_count``), ``snippets.body`` — so a NULL-padded shared
    shape is observable and a misgrouped row would read the wrong member's column. ``owner_id``
    groups rows so the same tables serve a ROOT collection (``Query.search``) AND a per-parent
    union (``Author.activity``, keyed on ``owner_id``). The interleaving of ``created`` across
    the two tables exercises the keyset-over-union ordering (rows from both members sort into
    one page). Lives in ``grafast_demo`` alongside the demo tables but is independent of them,
    so it does not perturb the authors/posts/comments parity fixtures. Run AFTER
    :func:`setup_demo_schema` (which creates the schema).
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP TABLE IF EXISTS {DEMO_SCHEMA}.articles"))
        await conn.execute(text(f"DROP TABLE IF EXISTS {DEMO_SCHEMA}.snippets"))
        await conn.execute(
            text(
                f"""
                CREATE TABLE {DEMO_SCHEMA}.articles (
                    id         integer PRIMARY KEY,
                    owner_id   integer NOT NULL,
                    created    timestamptz NOT NULL,
                    headline   text NOT NULL,
                    word_count integer NOT NULL
                )
                """
            )
        )
        await conn.execute(
            text(
                f"""
                CREATE TABLE {DEMO_SCHEMA}.snippets (
                    id       integer PRIMARY KEY,
                    owner_id integer NOT NULL,
                    created  timestamptz NOT NULL,
                    body     text NOT NULL
                )
                """
            )
        )
        # owner 1 owns 2 articles + 2 snippets, owner 2 owns 1 article + 1 snippet. created
        # interleaves the two tables (a1 < s1 < a2 < s2 ...) so a keyset page over the union
        # mixes members in one ordered slice. id is unique PER table; the union's tie-break is
        # the (created, id) keyset — created is distinct here so the order is unambiguous.
        base = datetime(2024, 3, 1, 8, 0, 0, tzinfo=timezone.utc)
        articles = [
            (1, 1, base.replace(hour=8), "alpha headline", 120),
            (2, 1, base.replace(hour=10), "beta headline", 80),
            (3, 2, base.replace(hour=14), "gamma headline", 200),
        ]
        snippets = [
            (1, 1, base.replace(hour=9), "first snippet"),
            (2, 1, base.replace(hour=11), "second snippet"),
            (3, 2, base.replace(hour=15), "third snippet"),
        ]
        await conn.execute(
            text(
                f"INSERT INTO {DEMO_SCHEMA}.articles"
                " (id, owner_id, created, headline, word_count)"
                " VALUES (:id, :owner_id, :created, :headline, :word_count)"
            ),
            [
                {
                    "id": i,
                    "owner_id": o,
                    "created": c,
                    "headline": h,
                    "word_count": w,
                }
                for i, o, c, h, w in articles
            ],
        )
        await conn.execute(
            text(
                f"INSERT INTO {DEMO_SCHEMA}.snippets (id, owner_id, created, body)"
                " VALUES (:id, :owner_id, :created, :body)"
            ),
            [
                {"id": i, "owner_id": o, "created": c, "body": b}
                for i, o, c, b in snippets
            ],
        )


async def setup_union_collision_tables() -> None:
    """Create + seed ``grafast_demo.coll_articles`` / ``coll_snippets`` with a CROSS-BRANCH tie.

    The page-boundary HAZARD fixture for the keyset-over-UNION-ALL total order: two SEPARATE
    member tables whose rows deliberately COLLIDE on the ``(created, id)`` keyset across the
    branches — both tables hold an ``id=1`` and an ``id=2`` row sharing the SAME ``created``
    timestamp. The per-table ``id`` is NOT unique across the union, so ``(created, id)`` alone
    is a non-total order: a page boundary that lands on one tied row would, under a (created,
    id)-only seek, silently DROP the other tied row (it fails ``id > cursor_id``) or duplicate
    it. The union's ``__typename`` final tie-break is what restores a total order, so paging
    through this fixture in pages of 1 must surface EVERY distinct row exactly once with no gap
    or duplicate. Independent of the ``articles`` / ``snippets`` fixture (own tables) so it
    perturbs no other assertion. Lives in ``grafast_demo``; run AFTER
    :func:`setup_demo_schema`.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP TABLE IF EXISTS {DEMO_SCHEMA}.coll_articles"))
        await conn.execute(text(f"DROP TABLE IF EXISTS {DEMO_SCHEMA}.coll_snippets"))
        await conn.execute(
            text(
                f"""
                CREATE TABLE {DEMO_SCHEMA}.coll_articles (
                    id       integer PRIMARY KEY,
                    owner_id integer NOT NULL,
                    created  timestamptz NOT NULL,
                    headline text NOT NULL
                )
                """
            )
        )
        await conn.execute(
            text(
                f"""
                CREATE TABLE {DEMO_SCHEMA}.coll_snippets (
                    id       integer PRIMARY KEY,
                    owner_id integer NOT NULL,
                    created  timestamptz NOT NULL,
                    body     text NOT NULL
                )
                """
            )
        )
        # both tables carry id=1 @09:00 and id=2 @10:00 — a cross-branch (created, id) tie on
        # EACH pair. A (created, id)-only seek would drop or duplicate the colliding peer at the
        # page boundary; the __typename tie-break (article < snippet by tag) totally orders them.
        base = datetime(2024, 5, 1, 9, 0, 0, tzinfo=timezone.utc)
        coll_articles = [
            (1, 1, base.replace(hour=9), "coll article one"),
            (2, 1, base.replace(hour=10), "coll article two"),
        ]
        coll_snippets = [
            (1, 1, base.replace(hour=9), "coll snippet one"),
            (2, 1, base.replace(hour=10), "coll snippet two"),
        ]
        await conn.execute(
            text(
                f"INSERT INTO {DEMO_SCHEMA}.coll_articles"
                " (id, owner_id, created, headline)"
                " VALUES (:id, :owner_id, :created, :headline)"
            ),
            [
                {"id": i, "owner_id": o, "created": c, "headline": h}
                for i, o, c, h in coll_articles
            ],
        )
        await conn.execute(
            text(
                f"INSERT INTO {DEMO_SCHEMA}.coll_snippets"
                " (id, owner_id, created, body)"
                " VALUES (:id, :owner_id, :created, :body)"
            ),
            [
                {"id": i, "owner_id": o, "created": c, "body": b}
                for i, o, c, b in coll_snippets
            ],
        )


async def setup_union_member_child_tables() -> None:
    """Create + seed ``grafast_demo.article_comments`` / ``snippet_reactions`` idempotently.

    Child tables for the cross-table union members (:func:`setup_union_member_tables`): a
    ``comments`` hasMany off ``articles`` and a ``reactions`` hasMany off ``snippets``. Each
    chains off a CONCRETE union member type resolved at completion time, so a nested pg
    relation under a union member batches PER concrete-type group — ONE statement across every
    Article in the bucket for ``comments``, ONE across every Snippet for ``reactions`` — never
    per row and never per parent. The two distinct child tables let a single query select a
    relation on EACH member and observe the two type-groups batching independently. ``body`` /
    ``emoji`` make the decoded child rows checkable. Lives in ``grafast_demo``; run AFTER
    :func:`setup_union_member_tables` (the FKs reference ``articles`` / ``snippets``).
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(
            text(f"DROP TABLE IF EXISTS {DEMO_SCHEMA}.article_comments")
        )
        await conn.execute(
            text(f"DROP TABLE IF EXISTS {DEMO_SCHEMA}.snippet_reactions")
        )
        await conn.execute(
            text(
                f"""
                CREATE TABLE {DEMO_SCHEMA}.article_comments (
                    id         integer PRIMARY KEY,
                    article_id integer NOT NULL REFERENCES {DEMO_SCHEMA}.articles (id),
                    body       text NOT NULL
                )
                """
            )
        )
        await conn.execute(
            text(
                f"""
                CREATE TABLE {DEMO_SCHEMA}.snippet_reactions (
                    id         integer PRIMARY KEY,
                    snippet_id integer NOT NULL REFERENCES {DEMO_SCHEMA}.snippets (id),
                    emoji      text NOT NULL
                )
                """
            )
        )
        # article 1 has 2 comments, article 2 has 1, article 3 has none (the empty-parent case);
        # snippet 1 has 1 reaction, snippet 3 has 2, snippet 2 has none. Per-parent counts differ
        # so the batched scatter back to each parent is observable, and the empty cases are covered.
        comments = [
            (1, 1, "great read"),
            (2, 1, "agreed"),
            (3, 2, "needs sources"),
        ]
        reactions = [
            (1, 1, "thumbsup"),
            (2, 3, "fire"),
            (3, 3, "heart"),
        ]
        await conn.execute(
            text(
                f"INSERT INTO {DEMO_SCHEMA}.article_comments (id, article_id, body)"
                " VALUES (:id, :article_id, :body)"
            ),
            [{"id": i, "article_id": a, "body": b} for i, a, b in comments],
        )
        await conn.execute(
            text(
                f"INSERT INTO {DEMO_SCHEMA}.snippet_reactions (id, snippet_id, emoji)"
                " VALUES (:id, :snippet_id, :emoji)"
            ),
            [{"id": i, "snippet_id": s, "emoji": e} for i, s, e in reactions],
        )


__all__ = [
    "setup_demo_schema",
    "setup_things_table",
    "setup_keyset_table",
    "setup_widgets_table",
    "setup_settings_probe_view",
    "setup_rls_table",
    "setup_labels_table",
    "setup_codec_rows_table",
    "setup_line_items_table",
    "setup_composite_tables",
    "setup_media_table",
    "setup_media_tags_table",
    "setup_union_member_tables",
    "setup_union_collision_tables",
    "setup_union_member_child_tables",
]
