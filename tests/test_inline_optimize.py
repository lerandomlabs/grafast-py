"""The inlining OPTIMIZE wiring: ``PgSelect*Step.optimize``.

The ``optimize`` hook ACTS on the pure safety PREDICATE (``inline_candidates``) — building
the replacement parent carrying the :class:`InlineSpec`\\ s and rewriting each folded child
relation step into a :class:`NestedExtractStep` that reads the parent's nested column. These
tests pin the DAG transform in ISOLATION (no DB):

- a parent with foldable children returns a REPLACEMENT carrying the specs, and records
  ``child -> NestedExtractStep`` so the optimize pass repoints every reference to the child;
- the NestedExtractStep's dep 0 is the REPLACEMENT parent (the same row dicts the child
  bucket is seeded from), so the child bucket is seeded identically;
- the orphaned key :class:`AccessStep` (the child's old dep 0) is tree-shaken once nothing
  reads it;
- the no-op invariant: a parent that folds NOTHING (inlining off, every child skipped)
  returns ``self`` — ``Plan.optimize`` then records no replacement and rewires nothing;
- NESTING composes via the optimize fixpoint (authors -> posts -> comments folds to one
  parent), and the inlined parent's dedup key (peer_key / dedup_params) folds in the specs.

The byte-identical-vs-batched equivalence proof is the DB-backed battery
(``test_pg_inlining.py``); here we only prove the plan-DAG rewrite is correct.
"""

from grafast_py.core_steps import AccessStep, RootStep, constant
from grafast_py.dag import Plan
from grafast_py.pg.inline import (
    KIND_HAS_MANY,
    KIND_HAS_ONE,
    NestedExtractStep,
    nested_alias_for,
)
from grafast_py.pg.resource import PgColumn, PgRegistry, PgResource
from grafast_py.pg.steps import PgSelectAllStep, PgSelectStep
from sqlalchemy.types import Integer, Text


def make_blog_registry():
    """authors(id, name) <-hasMany posts-> posts(id, author_id, title) <-hasMany comments-.

    Native columns carry their SQL type so the inlining safety predicate proves them
    json-stable native and folds (a bare untyped column would not — see
    ``PgResource.is_inline_json_safe``).
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
    comments = PgResource(
        "comments", "grafast_demo", "comments",
        [
            PgColumn("id", sql_type=Integer()),
            PgColumn("post_id", sql_type=Integer()),
            PgColumn("body", sql_type=Text()),
        ],
        registry=registry,
    )
    authors.has_many("posts", posts, local_column="id", remote_column="author_id")
    posts.has_one("author", authors, local_column="author_id", remote_column="id")
    posts.has_many("comments", comments, local_column="id", remote_column="post_id")
    comments.has_one("author", authors, local_column="author_id", remote_column="id")
    return registry, authors, posts, comments


def root_authors_plan(authors: PgResource):
    """A `Query.authors` root (over a RootStep) with inlining ON; returns (plan, parent).

    Mirrors the demo schema's ``_plan_all_rows``: a ``PgSelectAllStep`` wired to the
    operation ``RootStep`` via ``for_parent`` (so it has a bucket-sizing dep 0, as the real
    planner always builds it).
    """
    plan = Plan()
    plan.inline_relations = True
    root = RootStep()
    plan.add_step(root)
    parent = PgSelectAllStep(authors, order_by=["id"]).for_parent(root)
    plan.add_step(parent)
    return plan, parent


# ============================================================ the fold transform


def test_optimize_returns_replacement_carrying_the_spec():
    """A foldable `Author.posts` makes the root's optimize return a replacement + spec."""
    _registry, authors, posts, _comments = make_blog_registry()
    plan, parent = root_authors_plan(authors)
    child = authors.related_many(parent, "posts")
    plan.add_step(child)

    replacement = parent.optimize(plan)

    assert replacement is not parent  # a genuine rewrite, not identity
    assert isinstance(replacement, PgSelectAllStep)
    assert len(replacement.inline_specs) == 1
    spec = replacement.inline_specs[0]
    assert spec.resource is posts
    assert spec.kind == KIND_HAS_MANY
    assert spec.nested_alias == nested_alias_for(posts, ("author_id",))
    # the replacement keeps the SAME bucket-sizing parent (dep 0) so it seeds identically.
    assert replacement.dependencies[0] is parent.dependencies[0]


def test_optimize_records_child_to_nested_extract():
    """The folded child is recorded `child -> NestedExtractStep(replacement, alias, ...)`."""
    _registry, authors, posts, _comments = make_blog_registry()
    plan, parent = root_authors_plan(authors)
    child = authors.related_many(parent, "posts")
    plan.add_step(child)

    replacement = parent.optimize(plan)

    side = plan._optimize_side_replacements
    assert len(side) == 1
    old_child, extract = side[0]
    assert old_child is child
    assert isinstance(extract, NestedExtractStep)
    # dep 0 is the REPLACEMENT parent — the same parent row dicts the child bucket seeds from.
    assert extract.dependencies[0] is replacement
    assert extract.alias == replacement.inline_specs[0].nested_alias
    assert extract.kind == KIND_HAS_MANY
    # the extract was registered in the plan (carries an id for the rewire).
    assert extract in plan.steps


def test_optimize_identity_when_no_candidates():
    """Inlining off -> optimize is identity (no replacement, no side rewrites): the no-op."""
    _registry, authors, posts, _comments = make_blog_registry()
    plan, parent = root_authors_plan(authors)
    plan.inline_relations = False  # flip off
    plan.add_step(authors.related_many(parent, "posts"))

    assert parent.optimize(plan) is parent
    assert plan._optimize_side_replacements == []


# ============================================================ end-to-end Plan.optimize


def test_plan_optimize_rewires_child_and_its_access_to_the_extract():
    """`Plan.optimize` repoints the child AND a leaf AccessStep reading it to the extract."""
    _registry, authors, posts, _comments = make_blog_registry()
    plan, parent = root_authors_plan(authors)
    child = authors.related_many(parent, "posts")
    plan.add_step(child)
    # a leaf access reading a column off the child posts rows (e.g. `posts { title }`).
    title = AccessStep(child, ["title"])
    plan.add_step(title)

    remap = plan.optimize()

    # the child folded into a NestedExtractStep survivor.
    child_survivor = remap[child.id]
    assert isinstance(child_survivor, NestedExtractStep)
    # the title access was rewired to read off the extract (the child bucket's new source).
    assert title.dependencies[0] is child_survivor
    # the parent was replaced by the inlined root carrying the spec.
    parent_survivor = remap[parent.id]
    assert isinstance(parent_survivor, PgSelectAllStep)
    assert len(parent_survivor.inline_specs) == 1


def test_optimize_then_tree_shake_drops_orphaned_key_access():
    """The child's old key AccessStep, now unconsumed, is tree-shaken; the extract survives."""
    _registry, authors, posts, _comments = make_blog_registry()
    plan, parent = root_authors_plan(authors)
    child = authors.related_many(parent, "posts")
    plan.add_step(child)
    # the child's dep 0 is the per-entry key access (authors.id off the parent row).
    key_access = child.dependencies[0]
    assert isinstance(key_access, AccessStep)

    remap = plan.optimize()
    child_survivor = remap[child.id]
    parent_survivor = remap[parent.id]
    # consumption roots after the fold: the inlined root + the extract that reads its rows.
    plan.tree_shake([parent_survivor, child_survivor])

    surviving = {s.id for s in plan.steps}
    assert key_access.id not in surviving  # orphaned key access dropped
    assert child.id not in surviving  # the old batched child dropped
    assert child_survivor.id in surviving  # the extract kept (a consumption root)
    assert parent_survivor.id in surviving  # the inlined parent kept


# ============================================================ nesting: SINGLE-level fold


def test_nested_fold_is_single_level_and_comments_falls_back_batched():
    """authors -> posts -> comments folds ONE level: posts into authors; comments batched.

    Nested LATERAL (comments INSIDE posts' LATERAL inside authors) is NOT supported by the
    flat ``build_lateral`` — an :class:`InlineSpec` carries no nested child spec. The
    fold is therefore SINGLE-LEVEL: the outer relation (posts) folds into the root, and the
    deeper relation (comments) FALLS BACK to its batched ``WHERE post_id = ANY($1)`` path,
    reading the post ids off the extracted post rows (its key access is rewired to the posts
    extract). The result is still byte-identical to fully-batched; only the statement count
    drops by one (3 -> 2), not two. Proven here at the DAG level: posts folded to an extract,
    comments left a live batched PgSelectStep whose key reads off that extract.
    """
    _registry, authors, posts, comments = make_blog_registry()
    plan, parent = root_authors_plan(authors)
    posts_child = authors.related_many(parent, "posts")
    plan.add_step(posts_child)
    comments_child = posts.related_many(posts_child, "comments")
    plan.add_step(comments_child)

    remap = plan.optimize()

    # posts folded into the inlined authors root (a NestedExtractStep survivor).
    assert isinstance(remap[parent.id], PgSelectAllStep)
    assert len(remap[parent.id].inline_specs) == 1  # the posts fold
    posts_extract = remap[posts_child.id]
    assert isinstance(posts_extract, NestedExtractStep)
    # comments was NOT folded (its parent posts became an extract, not a select); it stays a
    # live batched PgSelectStep whose key access now reads off the posts extract.
    assert comments_child.id not in remap  # never replaced — still the standalone child
    assert comments_child.dependencies[0].dependencies[0] is posts_extract
    # after a tree-shake from the consumption roots TWO pg selects survive: the inlined
    # authors root and the still-batched comments select.
    roots = [remap[parent.id], posts_extract, comments_child]
    plan.tree_shake(roots)
    pg_selects = [s for s in plan.steps if isinstance(s, (PgSelectStep, PgSelectAllStep))]
    assert set(pg_selects) == {remap[parent.id], comments_child}


# ============================================================ dedup key folds the spec


def test_inlined_parent_dedup_key_folds_the_specs():
    """The inlined parent's peer_key / dedup_params discriminate on the folded specs.

    Two roots inlining DIFFERENT children must never dedup-merge (the SQL text differs); an
    inlined root must differ from the same root WITHOUT the fold.
    """
    _registry, authors, posts, _comments = make_blog_registry()
    plan, parent = root_authors_plan(authors)
    plan.add_step(authors.related_many(parent, "posts"))
    inlined = parent.optimize(plan)

    bare = PgSelectAllStep(authors, order_by=["id"])
    # the inlined parent carries a spec; the bare one does not — so their keys differ.
    assert inlined.peer_key != bare.peer_key
    assert inlined.dedup_params() != bare.dedup_params()
    # two inlined roots folding the SAME child DO share a key (so dedup can still merge them).
    plan2, parent2 = root_authors_plan(authors)
    plan2.add_step(authors.related_many(parent2, "posts"))
    inlined2 = parent2.optimize(plan2)
    assert inlined.peer_key == inlined2.peer_key
    assert inlined.dedup_params() == inlined2.dedup_params()


def test_has_one_parent_select_folds_child_has_one():
    """A PgSelectStep parent folds a hasOne child (Post.author) into a single-row extract."""
    _registry, authors, posts, _comments = make_blog_registry()
    plan = Plan()
    plan.inline_relations = True
    parent = PgSelectStep(posts, constant(None), "author_id", order_by=["id"])
    plan.add_step(parent)
    child = posts.related_single(parent, "author")
    plan.add_step(child)

    remap = plan.optimize()
    child_survivor = remap[child.id]
    assert isinstance(child_survivor, NestedExtractStep)
    assert child_survivor.kind == KIND_HAS_ONE
    parent_survivor = remap[parent.id]
    assert isinstance(parent_survivor, PgSelectStep)
    assert parent_survivor.inline_specs[0].kind == KIND_HAS_ONE
