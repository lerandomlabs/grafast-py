"""``finalize_plan`` COMPOSES the inlining rewrite end-to-end.

The inlining pieces are exercised in isolation elsewhere: the safety predicate
(``test_inline_predicate``), the ``optimize`` hook + ``tree_shake`` over a hand-wired
``Plan`` (``test_inline_optimize``), the LATERAL SQL (``test_inline_lateral``), the
``NestedExtractStep`` scatter (``test_inline_extract``). This file is the COMPOSITION
oracle for the planner's single finalize path: it drives the genuine
``plan_operation`` -> ``finalize_plan`` pipeline (optimize -> deduplicate -> tree_shake ->
``remap_object_plan``) over a REAL ObjectPlan tree built by the planner, and asserts that
the four passes hand off to each other correctly:

  * optimize REPLACES the parent select (it now carries the :class:`InlineSpec`) and
    REWRITES each folded child relation step into a :class:`NestedExtractStep`;
  * ``remap_object_plan`` repoints the child object bucket's ``parent_step`` (which WAS the
    folded child step) to that NestedExtractStep, so the executor seeds the child bucket
    from the parent's nested column;
  * deduplicate MERGES two parent buckets that folded the SAME child into one survivor
    (the inlined parent's dedup key folds the specs, so identical folds collapse);
  * tree_shake DROPS the now-orphaned key :class:`AccessStep` (the child's old dep 0) AND
    the orphaned standalone child select — both unreachable once the child reads off the
    parent row;
  * ``collect_consumption_root_steps`` over the finalized tree REACHES the NestedExtractStep
    (a child ``FieldPlan.step`` reads off it) AND the rewritten child bucket ``parent_step``
    (the same NestedExtractStep), so tree_shake measures reachability against exactly the
    surface the executor consumes.

These run PURELY at plan time — ``plan_operation`` / ``finalize_plan`` is a step-DAG
transform that issues no SQL — so the resources here carry no ``select_customizer`` and the
test needs no database. The BYTE-IDENTICAL data equivalence (inlined vs batched) is the
DB-backed battery in ``test_pg_inlining.py``; here we pin the plan-DAG COMPOSITION.
"""

from graphql import parse
from graphql.execution.collect_fields import collect_fields

from grafast_py import GrafastExecutionContext
from grafast_py.config import GrafastConfig
from grafast_py.core_steps import AccessStep, access
from grafast_py.pg.inline import KIND_HAS_MANY, KIND_HAS_ONE, NestedExtractStep
from grafast_py.pg.resource import PgColumn, PgRegistry, PgResource
from grafast_py.pg.steps import PgSelectAllStep, PgSelectStep
from grafast_py.plan import (
    collect_consumption_root_steps,
    plan_operation,
)
from grafast_py.schema import make_grafast_schema
from sqlalchemy.types import Integer, Text


def make_blog_registry():
    """authors(id, name) <-hasMany posts-> posts(id, author_id, title); posts.author hasOne.

    Native-scalar columns carry their SQL type (``Integer`` / ``Text``) so
    ``is_inline_json_safe`` PROVES them json-stable and every relation here is foldable (a
    bare untyped column would not fold — see ``PgResource.is_inline_json_safe``). No codecs
    and no ``select_customizer``, so building the plan needs no bound pg request.
    """
    registry = PgRegistry()
    authors = PgResource(
        "authors", "grafast_demo", "authors",
        [PgColumn("id", sql_type=Integer()), PgColumn("name", sql_type=Text())],
        registry=registry,
    )
    posts = PgResource(
        "posts", "grafast_demo", "posts",
        [
            PgColumn("id", sql_type=Integer()),
            PgColumn("author_id", sql_type=Integer()),
            PgColumn("title", sql_type=Text()),
        ],
        registry=registry,
    )
    authors.has_many("posts", posts, local_column="id", remote_column="author_id")
    posts.has_one("author", authors, local_column="author_id", remote_column="id")
    return registry, authors, posts


def leaf(key):
    """A plan resolver projecting ``key`` off the bucket's parent row."""

    def plan(parent, args, info):
        return access(parent, (key,))

    return plan


def build_hasmany_schema(authors: PgResource):
    """``{ authors { id posts { id title } } }`` — a hasMany fold (Author.posts)."""
    sdl = """
    type Query { authors: [Author!]! }
    type Author { id: Int! name: String! posts: [Post!]! }
    type Post { id: Int! title: String! }
    """
    return make_grafast_schema(
        sdl,
        {
            "Query": {
                "authors": lambda p, a, i: PgSelectAllStep(
                    authors, order_by=["id"]
                ).for_parent(p)
            },
            "Author": {
                "id": leaf("id"),
                "name": leaf("name"),
                "posts": lambda p, a, i: authors.related_many(p, "posts"),
            },
            "Post": {"id": leaf("id"), "title": leaf("title")},
        },
    )


def build_hasone_schema(posts: PgResource):
    """``{ posts { id author { id name } } }`` — a hasOne fold (Post.author)."""
    sdl = """
    type Query { posts: [Post!]! }
    type Post { id: Int! author: Author }
    type Author { id: Int! name: String! }
    """
    return make_grafast_schema(
        sdl,
        {
            "Query": {
                "posts": lambda p, a, i: PgSelectAllStep(
                    posts, order_by=["id"]
                ).for_parent(p)
            },
            "Post": {
                "id": leaf("id"),
                "author": lambda p, a, i: posts.related_single(p, "author"),
            },
            "Author": {"id": leaf("id"), "name": leaf("name")},
        },
    )


def context_class(inline: bool):
    """A throwaway context subclass carrying ``GrafastConfig(inline_relations=inline)``."""

    class _Ctx(GrafastExecutionContext):
        grafast_config = GrafastConfig(inline_relations=inline)

    return _Ctx


def build_finalized(schema, query: str, *, inline: bool):
    """Drive the genuine ``plan_operation`` -> ``finalize_plan`` path; return (ctx, plan).

    ``plan_operation`` reads ``inline_relations`` off the context's ``grafast_config`` and
    threads it onto the ``Plan``, runs ``finalize_plan`` (optimize -> dedup -> tree_shake ->
    remap), and stashes the finalized plan on the context. No SQL is issued — planning is a
    pure step-DAG transform — so this needs no database.
    """
    document = parse(query)
    operation = document.definitions[0]
    # plan_operation reads `type(context).grafast_config.inline_relations`, so building on
    # the inline-configured subclass threads the toggle through finalize_plan.
    ctx = context_class(inline).build(schema, document)
    root_fields = collect_fields(
        ctx.schema,
        ctx.fragments,
        ctx.variable_values,
        schema.query_type,
        operation.selection_set,
    )
    object_plan = plan_operation(ctx, operation, schema.query_type, root_fields)
    return ctx, object_plan


def find_child_plan(object_plan, response_name: str):
    """The (FieldPlan, child ObjectPlan) for ``response_name``, searched depth-first.

    Walks the whole (transitively nested) ObjectPlan tree — the field may sit under an
    enclosing object bucket (``posts`` lives under ``authors``'s child plan) — and returns
    the field plus the child plan its object completer carries.
    """
    from grafast_py.completion import find_object_completer

    def visit(op):
        for fp in op.fields:
            child_completer = find_object_completer(fp.completer)
            child_plan = child_completer.child_plan if child_completer else None
            if fp.response_name == response_name:
                assert child_plan is not None
                return fp, child_plan
            if child_plan is not None:
                found = visit(child_plan)
                if found is not None:
                    return found
        return None

    found = visit(object_plan)
    assert found is not None, f"no nested field {response_name!r} in the plan tree"
    return found


def pg_selects(plan):
    """The surviving root/relation pg SELECT steps (each = one batched statement)."""
    return [s for s in plan.steps if isinstance(s, (PgSelectStep, PgSelectAllStep))]


# ============================================================ the hasMany composition


def test_finalize_folds_hasmany_replaces_parent_and_rewrites_child():
    """authors -> posts folds through finalize: parent carries the spec, child is an extract.

    The end-to-end COMPOSITION of optimize's two rewrites THROUGH ``finalize_plan``: the
    ``Query.authors`` root is REPLACED by an inlined ``PgSelectAllStep`` carrying ONE
    :class:`InlineSpec` (the posts fold), and the ``Author.posts`` child relation step is
    REWRITTEN into a :class:`NestedExtractStep` whose dep 0 is that replacement parent.
    """
    _registry, authors, _posts = make_blog_registry()
    schema = build_hasmany_schema(authors)
    _ctx, object_plan = build_finalized(
        schema, "{ authors { id name posts { id title } } }", inline=True
    )

    # the root authors select was REPLACED by an inlined one carrying the posts spec.
    root_select = object_plan.fields[0].step
    assert isinstance(root_select, PgSelectAllStep)
    assert len(root_select.inline_specs) == 1
    spec = root_select.inline_specs[0]
    assert spec.kind == KIND_HAS_MANY

    # the Author.posts child object bucket's parent_step is the NestedExtractStep the fold
    # rewrote the child relation step into (the remap repointed the bucket source).
    _posts_fp, posts_plan = find_child_plan(object_plan, "posts")
    extract = posts_plan.layer.parent_step
    assert isinstance(extract, NestedExtractStep)
    assert extract.kind == KIND_HAS_MANY
    assert extract.alias == spec.nested_alias
    # the extract reads off the REPLACEMENT parent row dicts (the same column the child
    # bucket would have been seeded from), so the bucket seeds identically.
    assert extract.dependencies[0] is root_select


def test_finalize_rewrites_child_field_steps_to_read_off_the_extract():
    """The child posts fields (id, title) read off the NestedExtractStep, not a pg select.

    ``remap_object_plan`` repoints not just the child bucket ``parent_step`` but every child
    ``FieldPlan.step`` reading the folded child's rows — each ``access(child, col)`` now
    reads off the NestedExtractStep (the child bucket's new source).
    """
    _registry, authors, _posts = make_blog_registry()
    schema = build_hasmany_schema(authors)
    _ctx, object_plan = build_finalized(
        schema, "{ authors { posts { id title } } }", inline=True
    )

    _posts_fp, posts_plan = find_child_plan(object_plan, "posts")
    extract = posts_plan.layer.parent_step
    assert isinstance(extract, NestedExtractStep)
    # each leaf field in the posts bucket projects off the extract step.
    for fp in posts_plan.fields:
        assert isinstance(fp.step, AccessStep)
        assert fp.step.dependencies[0] is extract


def test_finalize_tree_shakes_orphaned_key_access_and_standalone_child():
    """tree_shake drops the orphaned key AccessStep AND the orphaned standalone child select.

    Once the posts child reads off the parent's nested column, its old per-entry key
    ``AccessStep`` (authors.id off the parent row) and its standalone batched
    ``PgSelectStep`` are both unconsumed — finalize's tree_shake (last pass) removes them.
    Exactly ONE pg statement survives: the inlined authors root.
    """
    _registry, authors, _posts = make_blog_registry()
    schema = build_hasmany_schema(authors)
    ctx, object_plan = build_finalized(
        schema, "{ authors { id posts { id title } } }", inline=True
    )
    plan = ctx._grafast_plan

    # the standalone batched posts PgSelectStep is gone (folded into the root's LATERAL).
    assert not any(isinstance(s, PgSelectStep) for s in plan.steps)
    # exactly one pg statement remains: the inlined authors root.
    selects = pg_selects(plan)
    assert len(selects) == 1
    assert isinstance(selects[0], PgSelectAllStep)
    assert len(selects[0].inline_specs) == 1

    # the orphaned key access (the child's old FK key step) was shaken out: no surviving
    # AccessStep reads the authors.id key off the (now-replaced) parent for a relation match.
    # All surviving AccessSteps read off the NestedExtractStep (the child leaf fields) or are
    # the root's own leaf fields — none is an orphaned relation key feeding a dropped select.
    extract = next(s for s in plan.steps if isinstance(s, NestedExtractStep))
    surviving_accesses = [s for s in plan.steps if isinstance(s, AccessStep)]
    for acc in surviving_accesses:
        # every surviving access reads off either the root select or the extract — never a
        # dropped standalone child select.
        assert acc.dependencies[0] in (selects[0], extract)


def test_finalize_consumption_roots_reach_extract_and_child_bucket_parent():
    """collect_consumption_root_steps reaches the NestedExtractStep + child bucket parent_step.

    The consumption surface tree_shake measures against must include the NestedExtractStep
    (a child ``FieldPlan.step`` reads off it) AND the rewritten child bucket ``parent_step``
    (the same extract — the bucket boundary the executor seeds). Were either missing,
    tree_shake would shake the extract out and the child bucket would have no source.
    """
    _registry, authors, _posts = make_blog_registry()
    schema = build_hasmany_schema(authors)
    _ctx, object_plan = build_finalized(
        schema, "{ authors { id posts { id title } } }", inline=True
    )

    roots = collect_consumption_root_steps(object_plan)
    root_ids = {s.id for s in roots}

    _posts_fp, posts_plan = find_child_plan(object_plan, "posts")
    extract = posts_plan.layer.parent_step
    assert isinstance(extract, NestedExtractStep)

    # the child bucket's parent_step (the extract) is a consumption root (the seeded boundary).
    assert extract.id in root_ids
    # ... and the extract is reached AS a consumption surface member (the same step the child
    # fields read off), so tree_shake keeps it.
    assert extract in roots
    # the child posts field steps are consumption roots too (they read off the extract).
    for fp in posts_plan.fields:
        assert fp.step.id in root_ids


# ============================================================ the hasOne composition


def test_finalize_folds_hasone_through_the_pipeline():
    """Post.author (hasOne) folds through finalize: child bucket reads a single-row extract.

    The hasOne counterpart: the ``Query.posts`` root absorbs ``Post.author`` into a
    ``to_jsonb ... LIMIT 1`` LATERAL, and the author child bucket's ``parent_step`` becomes a
    hasOne :class:`NestedExtractStep` (scatters the single dict / None).
    """
    _registry, _authors, posts = make_blog_registry()
    schema = build_hasone_schema(posts)
    _ctx, object_plan = build_finalized(
        schema, "{ posts { id author { id name } } }", inline=True
    )

    root_select = object_plan.fields[0].step
    assert isinstance(root_select, PgSelectAllStep)
    assert len(root_select.inline_specs) == 1
    assert root_select.inline_specs[0].kind == KIND_HAS_ONE

    _author_fp, author_plan = find_child_plan(object_plan, "author")
    extract = author_plan.layer.parent_step
    assert isinstance(extract, NestedExtractStep)
    assert extract.kind == KIND_HAS_ONE
    assert extract.dependencies[0] is root_select
    # the consumption surface reaches the hasOne extract too.
    assert extract.id in {s.id for s in collect_consumption_root_steps(object_plan)}


# ============================================================ dedup merges identical folds


def test_finalize_dedup_merges_two_identical_folded_parents():
    """Two aliases of the SAME inlined root collapse to one survivor through finalize's dedup.

    The same ``authors`` root selected twice under two aliases (each folding the SAME posts
    relation identically) produces two structurally-identical inlined parents; finalize's
    deduplicate (which runs AFTER optimize, over the rewritten DAG) merges them into ONE
    survivor — the inlined parent's dedup key folds the specs, so identical folds DO merge.
    Both aliases' field steps then point at the single survivor.
    """
    _registry, authors, _posts = make_blog_registry()
    sdl = """
    type Query { a: [Author!]! b: [Author!]! }
    type Author { id: Int! posts: [Post!]! }
    type Post { id: Int! }
    """
    schema = make_grafast_schema(
        sdl,
        {
            "Query": {
                "a": lambda p, ar, i: PgSelectAllStep(
                    authors, order_by=["id"]
                ).for_parent(p),
                "b": lambda p, ar, i: PgSelectAllStep(
                    authors, order_by=["id"]
                ).for_parent(p),
            },
            "Author": {
                "id": leaf("id"),
                "posts": lambda p, ar, i: authors.related_many(p, "posts"),
            },
            "Post": {"id": leaf("id")},
        },
    )
    ctx, object_plan = build_finalized(
        schema, "{ a { id posts { id } } b { id posts { id } } }", inline=True
    )
    plan = ctx._grafast_plan

    a_step = next(f.step for f in object_plan.fields if f.response_name == "a")
    b_step = next(f.step for f in object_plan.fields if f.response_name == "b")
    # both inlined roots merged into the SAME survivor (dedup folded the identical specs).
    assert a_step is b_step
    assert isinstance(a_step, PgSelectAllStep)
    assert len(a_step.inline_specs) == 1
    # exactly one pg statement survives across BOTH aliases (the merged inlined root); the two
    # standalone posts children both folded away.
    assert len(pg_selects(plan)) == 1


# ============================================================ the OFF baseline (no-op)


def test_finalize_default_off_keeps_batched_child():
    """Inlining OFF: finalize is a no-op — the standalone batched child select survives.

    The no-op invariant THROUGH the full pipeline: with ``inline_relations`` False
    every pg step's ``optimize`` returns ``self``, the parent is NOT replaced, the child stays
    a standalone ``PgSelectStep`` (its own ``WHERE author_id = ANY($1)``), and TWO pg
    statements survive — the correctness baseline the inlined run must match byte-for-byte.
    """
    _registry, authors, _posts = make_blog_registry()
    schema = build_hasmany_schema(authors)
    ctx, object_plan = build_finalized(
        schema, "{ authors { id posts { id title } } }", inline=False
    )
    plan = ctx._grafast_plan

    # no fold: no inline specs, no extract, the child stays a standalone batched select.
    root_select = object_plan.fields[0].step
    assert isinstance(root_select, PgSelectAllStep)
    assert root_select.inline_specs == ()
    assert not any(isinstance(s, NestedExtractStep) for s in plan.steps)

    _posts_fp, posts_plan = find_child_plan(object_plan, "posts")
    assert isinstance(
        posts_plan.layer.parent_step, PgSelectStep
    )  # the batched child survives
    # TWO pg statements: authors root + posts batched child — the baseline statement count.
    assert len(pg_selects(plan)) == 2


def test_finalize_inline_drops_exactly_one_statement_vs_off():
    """The fold reduces the surviving pg statement count by exactly one (2 -> 1).

    The plan-level analogue of the DB battery's ``count_sql`` assertion: inlining ON folds
    the posts child into the authors root, so ONE pg statement survives where the OFF baseline
    keeps TWO. (The byte-identical DATA equivalence is the DB-backed ``test_pg_inlining.py``;
    here we pin that the fold structurally fired — fewer statements — at plan time.)
    """
    _registry, authors, _posts = make_blog_registry()
    schema = build_hasmany_schema(authors)
    query = "{ authors { id name posts { id title } } }"

    off_ctx, _off_plan = build_finalized(schema, query, inline=False)
    on_ctx, _on_plan = build_finalized(schema, query, inline=True)

    off_count = len(pg_selects(off_ctx._grafast_plan))
    on_count = len(pg_selects(on_ctx._grafast_plan))
    assert off_count == 2
    assert on_count == 1
    assert on_count == off_count - 1
