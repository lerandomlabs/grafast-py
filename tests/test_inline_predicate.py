"""The inlining SAFETY PREDICATE: ``inline_candidates`` / ``find_inline_candidates``.

The LATERAL SQL (``build_lateral`` / ``apply_laterals``) a parent emits when it carries an
:class:`InlineSpec` is paired with this PURE PREDICATE that DECIDES which child relations are
provably safe to fold. ``parent.inline_candidates(plan)`` (delegating to
:func:`grafast_py.pg.inline.find_inline_candidates`) returns ``[(child_step, InlineSpec), ...]``
for every child that passes EVERY enumerated condition, and ``[]`` when inlining is off — with
NO DAG mutation and NO SQL (the predicate only inspects; the optimize wiring consumes the list
separately).

The defining property is CONSERVATISM: inlining must be byte-identical to the batched
``= ANY($1)`` path, so any unmet/uncertain condition SKIPS the child (keeping the batched
child — a redundant statement, never wrong data). These tests pin BOTH halves:

- the PASS cases: a plain hasOne (``Post.author``) and an unpaginated hasMany
  (``Author.posts``) fold, producing an :class:`InlineSpec` with the right kind, FK columns,
  alias and reproduced order (incl. the PK tie-break);
- each SKIP branch returns NO candidate: inlining OFF, a parent/child ``opt_out_inline``, a
  mutation (``dedupable=False``) child, a window-sliced (limited) hasMany, a composite FK, a
  paginated :class:`PgConnectionStep`, a non-json-safe codec column, a filtered (``.where``)
  child, and a cross-schema child.

All non-DB (no engine, no request context): the predicate is a pure plan-DAG inspection.
"""

import pytest
import pytest_asyncio
from sqlalchemy import column
from sqlalchemy.types import DateTime, Integer, Text

from grafast_py.core_steps import access, constant
from grafast_py.dag import Plan
from grafast_py.pg.connection import PgConnectionStep
from grafast_py.pg.inline import (
    KIND_HAS_MANY,
    KIND_HAS_ONE,
    NestedExtractStep,
    access_column_path,
    nested_alias_for,
)
from grafast_py.pg.resource import PgCodec, PgColumn, PgRegistry, PgResource
from grafast_py.pg.steps import PgSelectAllStep, PgSelectStep
from grafast_py.step_model import Step


# ---------------------------------------------------------------- the blog registry
# Native column types so the inlining safety predicate can PROVE each column json-stable
# native and fold it (a bare untyped column is UNKNOWN-typed and the predicate fails safe —
# see PgResource.is_inline_json_safe). These mirror the types the ORM bridge records.
AUTHOR_COLUMNS = [PgColumn("id", sql_type=Integer()), PgColumn("name", sql_type=Text())]
POST_COLUMNS = [
    PgColumn("id", sql_type=Integer()),
    PgColumn("author_id", sql_type=Integer()),
    PgColumn("title", sql_type=Text()),
]


def make_blog_registry():
    """authors(id, name) <-hasMany posts-> posts(id, author_id, title) <-hasOne author-."""
    registry = PgRegistry()
    authors = PgResource(
        "authors", "grafast_demo", "authors", AUTHOR_COLUMNS, registry=registry
    )
    posts = PgResource(
        "posts", "grafast_demo", "posts", POST_COLUMNS, registry=registry
    )
    authors.has_many("posts", posts, local_column="id", remote_column="author_id")
    posts.has_one("author", authors, local_column="author_id", remote_column="id")
    return registry, authors, posts


def root_authors_plan(authors: PgResource):
    """A `Query.authors` root collection step in a plan with inlining ON, returns (plan, parent)."""
    plan = Plan()
    plan.inline_relations = True
    parent = PgSelectAllStep(authors, order_by=["id"])
    plan.add_step(parent)
    return plan, parent


# ============================================================ PASS: the foldable shapes


def test_has_many_unpaginated_folds_with_faithful_spec():
    """`Author.posts` (unpaginated hasMany) folds: kind, FK columns, alias, order are right."""
    _registry, authors, posts = make_blog_registry()
    plan, parent = root_authors_plan(authors)
    child = authors.related_many(parent, "posts")
    plan.add_step(child)

    candidates = parent.inline_candidates(plan)
    assert len(candidates) == 1
    folded_child, spec = candidates[0]
    assert folded_child is child
    assert spec.kind == KIND_HAS_MANY
    # the textbook correlation: parent.id == child.author_id.
    assert spec.local_columns == ("id",)
    assert spec.remote_columns == ("author_id",)
    assert spec.resource is posts
    # the order is reproduced verbatim — the default PK order (id asc), the PK tie-break baked.
    assert [t.column for t in spec.order_by] == ["id"]
    # an unfiltered fold carries no LATERAL predicates.
    assert spec.where_predicates == ()
    # the alias is the deterministic derived nested column the LATERAL projects.
    assert spec.nested_alias == nested_alias_for(posts, ("author_id",))


def test_has_one_folds_as_single_kind():
    """`Post.author` (hasOne) folds: a single-row kind with FK author_id -> id."""
    _registry, authors, posts = make_blog_registry()
    plan = Plan()
    plan.inline_relations = True
    parent = PgSelectStep(posts, constant(None), "author_id", order_by=["id"])
    plan.add_step(parent)
    child = posts.related_single(parent, "author")
    plan.add_step(child)

    candidates = parent.inline_candidates(plan)
    assert len(candidates) == 1
    _child, spec = candidates[0]
    assert spec.kind == KIND_HAS_ONE
    assert spec.local_columns == ("author_id",)
    assert spec.remote_columns == ("id",)
    assert spec.resource is authors


def test_non_pk_descending_order_is_reproduced_with_pk_tiebreak():
    """A relation ordered by a non-PK column folds, reproducing the order + PK tie-break."""
    from grafast_py.pg.ordering import OrderTerm

    _registry, authors, posts = make_blog_registry()
    plan, parent = root_authors_plan(authors)
    child = authors.related_many(
        parent, "posts", order_by=[OrderTerm("title", descending=True)]
    )
    plan.add_step(child)

    _child, spec = parent.inline_candidates(plan)[0]
    # title DESC then the id PK tie-break (what normalize_order baked onto the child).
    assert [(t.column, t.descending) for t in spec.order_by] == [
        ("title", True),
        ("id", False),
    ]
    # the spec order is byte-identical to the standalone child's normalized order.
    assert spec.order_by == tuple(child.order_by)


def test_two_distinct_relations_both_fold_under_distinct_aliases():
    """A parent with two foldable children yields two specs under distinct nested columns."""
    registry = PgRegistry()
    authors = PgResource(
        "authors", "grafast_demo", "authors", AUTHOR_COLUMNS, registry=registry
    )
    posts = PgResource(
        "posts", "grafast_demo", "posts", POST_COLUMNS, registry=registry
    )
    comments = PgResource(
        "comments", "grafast_demo", "comments",
        [
            PgColumn("id", sql_type=Integer()),
            PgColumn("author_id", sql_type=Integer()),
            PgColumn("body", sql_type=Text()),
        ],
        registry=registry,
    )
    authors.has_many("posts", posts, local_column="id", remote_column="author_id")
    authors.has_many("comments", comments, local_column="id", remote_column="author_id")

    plan, parent = root_authors_plan(authors)
    plan.add_step(authors.related_many(parent, "posts"))
    plan.add_step(authors.related_many(parent, "comments"))

    candidates = parent.inline_candidates(plan)
    aliases = {spec.nested_alias for _c, spec in candidates}
    assert len(candidates) == 2
    assert len(aliases) == 2  # distinct columns so the two LATERALs do not collide


# ============================================================ SKIP: the predicate branches


def test_skip_when_inlining_off():
    """The toggle gates everything: inline_relations False -> no candidates (the no-op path)."""
    _registry, authors, posts = make_blog_registry()
    plan, parent = root_authors_plan(authors)
    plan.inline_relations = False  # flip it back off
    plan.add_step(authors.related_many(parent, "posts"))
    assert parent.inline_candidates(plan) == []


def test_skip_when_parent_resource_opts_out():
    """A parent table with opt_out_inline never absorbs a child, even with the flag on."""
    registry = PgRegistry()
    authors = PgResource(
        "authors", "grafast_demo", "authors", ["id", "name"],
        registry=registry, opt_out_inline=True,
    )
    posts = PgResource(
        "posts", "grafast_demo", "posts", ["id", "author_id", "title"], registry=registry
    )
    authors.has_many("posts", posts, local_column="id", remote_column="author_id")
    plan, parent = root_authors_plan(authors)
    plan.add_step(authors.related_many(parent, "posts"))
    assert parent.inline_candidates(plan) == []


def test_skip_when_child_resource_opts_out():
    """A child table with opt_out_inline is never folded (the table-level escape hatch)."""
    registry = PgRegistry()
    authors = PgResource(
        "authors", "grafast_demo", "authors", ["id", "name"], registry=registry
    )
    posts = PgResource(
        "posts", "grafast_demo", "posts", ["id", "author_id", "title"],
        registry=registry, opt_out_inline=True,
    )
    authors.has_many("posts", posts, local_column="id", remote_column="author_id")
    plan, parent = root_authors_plan(authors)
    plan.add_step(authors.related_many(parent, "posts"))
    assert parent.inline_candidates(plan) == []


def test_skip_limited_has_many():
    """A per-parent paginated hasMany (first/offset) is NOT foldable — SKIP."""
    _registry, authors, posts = make_blog_registry()
    plan, parent = root_authors_plan(authors)
    plan.add_step(authors.related_many(parent, "posts", first=2))
    assert parent.inline_candidates(plan) == []


def test_skip_composite_fk_relation():
    """A composite FK (a ListStep key) is not the single-column correlation — SKIP."""
    registry = PgRegistry()
    lines = PgResource(
        "lines", "grafast_demo", "lines", ["org_id", "item_id"],
        primary_key="org_id", registry=registry,
    )
    prices = PgResource(
        "prices", "grafast_demo", "prices", ["org_id", "item_id", "amount"],
        primary_key="org_id", registry=registry,
    )
    lines.has_many(
        "prices", prices,
        local_columns=("org_id", "item_id"), remote_columns=("org_id", "item_id"),
    )
    plan = Plan()
    plan.inline_relations = True
    parent = PgSelectAllStep(lines, order_by=["org_id"])
    plan.add_step(parent)
    plan.add_step(lines.related_many(parent, "prices"))
    assert parent.inline_candidates(plan) == []


def test_skip_paginated_connection():
    """A PgConnectionStep (paginated / aggregate / keyset) is always SKIPPED."""
    _registry, authors, posts = make_blog_registry()
    plan, parent = root_authors_plan(authors)
    key = access(parent, ("id",))
    plan.add_step(key)
    conn = PgConnectionStep(posts, key, "author_id", order_by=["id"], first=2)
    plan.add_step(conn)
    assert parent.inline_candidates(plan) == []


def test_skip_non_json_safe_codec_column():
    """A child with a non-native codec column (timestamptz) is not json-stable — SKIP."""
    registry = PgRegistry()
    authors = PgResource(
        "authors", "grafast_demo", "authors", ["id", "name"], registry=registry
    )
    events = PgResource(
        "events", "grafast_demo", "events",
        ["id", "author_id", PgColumn("at", codec=PgCodec(sql_type=DateTime(timezone=True)))],
        registry=registry,
    )
    authors.has_many("events", events, local_column="id", remote_column="author_id")
    plan, parent = root_authors_plan(authors)
    plan.add_step(authors.related_many(parent, "events"))
    assert events.is_inline_json_safe is False
    assert parent.inline_candidates(plan) == []


def test_json_stable_codec_column_still_folds():
    """A NATIVE-typed codec column (str.upper over text) IS json-stable — it still folds.

    The codec runs on the NestedExtractStep's decode, and the raw text round-trips through the
    json column identically, so the decoded value matches the batched path — safe to inline.
    """
    registry = PgRegistry()
    authors = PgResource(
        "authors", "grafast_demo", "authors", AUTHOR_COLUMNS, registry=registry
    )
    labels = PgResource(
        "labels", "grafast_demo", "labels",
        [
            PgColumn("id", sql_type=Integer()),
            PgColumn("author_id", sql_type=Integer()),
            PgColumn("code", codec=PgCodec(to_py=str.upper)),
        ],
        registry=registry,
    )
    authors.has_many("labels", labels, local_column="id", remote_column="author_id")
    plan, parent = root_authors_plan(authors)
    plan.add_step(authors.related_many(parent, "labels"))
    assert labels.is_inline_json_safe is True
    candidates = parent.inline_candidates(plan)
    assert len(candidates) == 1
    assert candidates[0][1].resource is labels


def test_skip_bare_column_with_no_codec_and_no_type():
    """A codec-LESS column with NO declared type is UNKNOWN — cannot prove json-stable, SKIP.

    The json-stability condition of the inlining safety predicate: a bare string column (no
    codec, no ``sql_type``) MIGHT be a non-native scalar whose ``to_jsonb`` -> JSON form
    differs from the asyncpg row value (numeric precision loss, timestamptz tz-shift,
    bytea-as-string). The predicate refuses to ASSUME native; it fails safe and keeps the
    batched child. Declaring the column's type (or a codec) is what makes a genuinely-native
    column foldable (see the other branch tests).
    """
    registry = PgRegistry()
    authors = PgResource(
        "authors", "grafast_demo", "authors", AUTHOR_COLUMNS, registry=registry
    )
    # `amount`/`ts` are BARE: no codec, no sql_type — the predicate cannot prove them native.
    events = PgResource(
        "events", "grafast_demo", "events",
        ["id", "author_id", "amount", "ts"],
        registry=registry,
    )
    authors.has_many("events", events, local_column="id", remote_column="author_id")
    plan, parent = root_authors_plan(authors)
    plan.add_step(authors.related_many(parent, "events"))
    assert events.is_inline_json_safe is False
    assert parent.inline_candidates(plan) == []


def test_skip_bare_non_native_typed_column():
    """A codec-LESS column with a KNOWN NON-native ``sql_type`` (numeric) is NOT json-stable.

    The type IS known (e.g. the ORM bridge recorded it), and it is non-native — its JSON form
    drops trailing numeric precision (``12.5000`` -> ``12.5``) — so the fold SKIPS even with no
    codec. This is the type-aware half of the json-stability condition.
    """
    from sqlalchemy.types import Numeric

    registry = PgRegistry()
    authors = PgResource(
        "authors", "grafast_demo", "authors", AUTHOR_COLUMNS, registry=registry
    )
    invoices = PgResource(
        "invoices", "grafast_demo", "invoices",
        [
            PgColumn("id", sql_type=Integer()),
            PgColumn("author_id", sql_type=Integer()),
            PgColumn("total", sql_type=Numeric(12, 2)),  # known non-native -> not json-stable
        ],
        registry=registry,
    )
    authors.has_many("invoices", invoices, local_column="id", remote_column="author_id")
    plan, parent = root_authors_plan(authors)
    plan.add_step(authors.related_many(parent, "invoices"))
    assert invoices.is_inline_json_safe is False
    assert parent.inline_candidates(plan) == []


def test_typed_native_columns_fold():
    """Columns with KNOWN native ``sql_type``s (int / text) are provably json-stable — fold."""
    registry = PgRegistry()
    authors = PgResource(
        "authors", "grafast_demo", "authors", AUTHOR_COLUMNS, registry=registry
    )
    posts = PgResource(
        "posts", "grafast_demo", "posts", POST_COLUMNS, registry=registry
    )
    authors.has_many("posts", posts, local_column="id", remote_column="author_id")
    plan, parent = root_authors_plan(authors)
    plan.add_step(authors.related_many(parent, "posts"))
    assert posts.is_inline_json_safe is True
    assert len(parent.inline_candidates(plan)) == 1


def test_skip_filtered_child():
    """A child carrying a host .where() (a filter) is SKIPPED — not currently inlined."""
    _registry, authors, posts = make_blog_registry()
    plan, parent = root_authors_plan(authors)
    child = authors.related_many(parent, "posts")
    child.add_where(column("title") == "published")  # a per-plan filter the LATERAL would carry
    plan.add_step(child)
    assert parent.inline_candidates(plan) == []


def test_skip_cross_schema_child():
    """A child in a DIFFERENT schema is a different executor binding — not foldable."""
    registry = PgRegistry()
    authors = PgResource(
        "authors", "grafast_demo", "authors", ["id", "name"], registry=registry
    )
    # a posts table in another schema; the FK shape is identical but the binding differs.
    posts = PgResource(
        "posts", "other_schema", "posts", ["id", "author_id", "title"], registry=registry
    )
    authors.has_many("posts", posts, local_column="id", remote_column="author_id")
    plan, parent = root_authors_plan(authors)
    plan.add_step(authors.related_many(parent, "posts"))
    assert parent.inline_candidates(plan) == []


def test_skip_self_referential_relation():
    """A SELF-referential relation (child resource is the SAME table as the parent) — SKIP.

    employees.has_many("reports", employees, ...) folds the SAME table into itself: the flat
    build_lateral emits the inner child as a bare table with the SAME unaliased name as the
    outer parent, so the `parent.local = child.remote` correlation would resolve to the INNER
    table and collapse into a within-row comparison (every parent silently gets []). Without a
    per-select alias subsystem the predicate must NOT fold a self-relation — it keeps the
    correct batched `= ANY($1)` path. Covers BOTH the hasMany and hasOne self-relations.
    """
    registry = PgRegistry()
    employees = PgResource(
        "employees", "grafast_demo", "employees",
        [
            PgColumn("id", sql_type=Integer()),
            PgColumn("manager_id", sql_type=Integer()),
            PgColumn("name", sql_type=Text()),
        ],
        registry=registry,
    )
    # the self-FK both ways: a manager's reports (hasMany) and a report's manager (hasOne).
    employees.has_many(
        "reports", employees, local_column="id", remote_column="manager_id"
    )
    employees.has_one(
        "manager", employees, local_column="manager_id", remote_column="id"
    )

    # hasMany self-relation off a root collection.
    plan, parent = root_authors_plan(employees)
    plan.add_step(employees.related_many(parent, "reports"))
    assert parent.inline_candidates(plan) == []

    # hasOne self-relation off a keyed select.
    from grafast_py.core_steps import constant

    plan2 = Plan()
    plan2.inline_relations = True
    parent2 = PgSelectStep(employees, constant(None), "id", order_by=["id"])
    plan2.add_step(parent2)
    plan2.add_step(employees.related_single(parent2, "manager"))
    assert parent2.inline_candidates(plan2) == []


def test_skip_mutation_child():
    """A side-effecting (dedupable=False) child is NEVER inlined.

    A real mutation is not wired as a relation, so this models the invariant directly: a
    PgSelectStep subclass marked dedupable=False keyed on the parent access must SKIP.
    """
    _registry, authors, posts = make_blog_registry()

    class WriteyChild(PgSelectStep):
        dedupable = False

    plan, parent = root_authors_plan(authors)
    key = access(parent, ("id",))
    plan.add_step(key)
    child = WriteyChild(posts, key, "author_id", order_by=["id"])
    plan.add_step(child)
    assert parent.inline_candidates(plan) == []


def test_skip_when_parent_is_side_effecting():
    """A side-effecting parent is never a fold ROOT — fail safe (no candidates)."""
    _registry, authors, posts = make_blog_registry()

    class WriteyParent(PgSelectAllStep):
        dedupable = False

    plan = Plan()
    plan.inline_relations = True
    parent = WriteyParent(authors, order_by=["id"])
    plan.add_step(parent)
    plan.add_step(authors.related_many(parent, "posts"))
    assert parent.inline_candidates(plan) == []


def test_limited_parent_select_hosts_no_lateral():
    """A window-sliced (limited) PgSelectStep parent cannot host a LATERAL — no candidates.

    `build_query` asserts against specs on a limited select, so the predicate must not even
    offer one; a paginated parent returns no folds regardless of its children.
    """
    _registry, authors, posts = make_blog_registry()
    # a nested hasMany: posts (limited, per-parent paginated) is itself a parent of comments.
    comments = PgResource(
        "comments", "grafast_demo", "comments", ["id", "post_id", "body"],
        registry=_registry,
    )
    posts.has_many("comments", comments, local_column="id", remote_column="post_id")
    plan = Plan()
    plan.inline_relations = True
    limited_posts = PgSelectStep(posts, constant(None), "author_id", order_by=["id"], first=2)
    plan.add_step(limited_posts)
    plan.add_step(posts.related_many(limited_posts, "comments"))
    assert limited_posts.inline_candidates(plan) == []


# ============================================================ helper: access_column_path


def test_access_column_path_recognises_single_column_key():
    """A single-column access off the parent yields its 1-segment path (the FK key shape)."""
    plan, parent = root_authors_plan(make_blog_registry()[1])
    key = access(parent, ("id",))
    assert access_column_path(key) == ("id",)


def test_access_column_path_rejects_multi_segment_and_fallback_and_non_access():
    """A deep path, a fallback access, or a non-access is NOT a plain FK key — None."""
    plan, parent = root_authors_plan(make_blog_registry()[1])
    assert access_column_path(access(parent, ("a", "b"))) is None  # nested path
    assert access_column_path(access(parent, ("id",), fallback=0)) is None  # has a fallback
    assert access_column_path(constant(5)) is None  # not an access at all

    class Other(Step):
        def execute(self, count, values):
            return [None] * count

    assert access_column_path(Other()) is None


# ============================================== DB-backed equivalence of the PREDICATE spec


@pytest_asyncio.fixture
async def reseeded_engine():
    """(Re)seed grafast_demo and yield a fresh engine; dispose after."""
    from grafast_py.pg.engine import dispose_engine, get_engine
    from examples.seed import setup_demo_schema

    await dispose_engine()
    await setup_demo_schema()
    yield get_engine()
    await dispose_engine()


@pytest.mark.pg
@pytest.mark.asyncio
async def test_predicate_spec_inlines_equivalently_in_one_fewer_statement(reseeded_engine):
    """The PREDICATE-produced spec, when applied, is byte-identical to the batched child.

    The predicate produces the :class:`InlineSpec` without wiring the rewrite itself. This is
    the honest equivalence proof for the PREDICATE: take the spec ``inline_candidates``
    returns for ``Author.posts``, APPLY it (build the inlined parent + the
    :class:`NestedExtractStep` the fold would create), and assert the scattered child lists
    are BYTE-IDENTICAL to the standalone batched ``= ANY($1)`` child — in ONE FEWER statement.
    A wrong/over-eager spec (bad order, wrong FK, dropped codec) would diverge here; equal
    data in fewer statements is the inlining invariant the predicate must uphold.
    """
    from grafast_py.pg.engine import count_sql
    from grafast_py.pg.executor import SQLAlchemyExecutor, pg_request_context

    _registry, authors, posts = make_blog_registry()
    engine = reseeded_engine

    # --- batched baseline: authors (1) + posts WHERE author_id = ANY (1) = 2 statements.
    with count_sql(engine) as batched_counter:
        with pg_request_context(SQLAlchemyExecutor(engine)):
            root = PgSelectAllStep(authors, order_by=["id"])
            author_rows = (await root.execute(1, [[None]]))[0]
            author_ids = [r["id"] for r in author_rows]
            child = PgSelectStep(posts, constant(None), "author_id", order_by=["id"])
            batched_lists = await child.execute(len(author_ids), [author_ids])
    assert batched_counter.count == 2

    # --- derive the fold from the PREDICATE, then apply it: 1 statement.
    with count_sql(engine) as inlined_counter:
        with pg_request_context(SQLAlchemyExecutor(engine)):
            plan, parent = root_authors_plan(authors)
            relation_child = authors.related_many(parent, "posts")
            plan.add_step(relation_child)
            candidates = parent.inline_candidates(plan)
            assert len(candidates) == 1  # the predicate chose to fold Author.posts
            _folded, spec = candidates[0]

            # apply the spec exactly as the optimize pass would: a parent carrying the
            # spec + a NestedExtractStep on its derived alias.
            inlined_root = PgSelectAllStep(
                authors, order_by=["id"], inline_specs=[spec]
            )
            parent_rows = (await inlined_root.execute(1, [[None]]))[0]
            extract = NestedExtractStep(
                inlined_root, spec.nested_alias, posts, spec.kind
            )
            inlined_lists = extract.execute(len(parent_rows), [parent_rows])
    assert inlined_counter.count == 1

    # the PREDICATE's spec scatters the IDENTICAL per-author posts lists — byte-identical
    # data, one fewer statement.
    assert inlined_lists == batched_lists
    assert sum(len(group) for group in batched_lists) == 9  # 2 + 3 + 4 posts in the seed
