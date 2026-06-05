"""pgUnionAll DEDUP CORRECTNESS (no DB), both directions, through ``dag.Plan.deduplicate()``.

The dedup invariant for the cross-table polymorphism step (:class:`PgUnionAllStep`): every
SQL-affecting input — the UNION ALL member set, each branch's per-member WHERE, the shared
ORDER BY, and the keyset ``after``/``before`` cursor VALUES — MUST fold into the step's
structural dedup key (``peer_key`` + ``dedup_params``, value-included via ``literal_binds``).
So two union steps differing ONLY in one such input emit DIFFERENT SQL and must NOT collapse
into one; two that are structurally identical describe the SAME query and MUST merge so it is
issued once.

Where :mod:`tests.test_pg_union_all` asserts the dedup KEYS differ/match in isolation, this
file drives every case THROUGH the real planner pass — :meth:`grafast_py.dag.Plan.deduplicate`
— and asserts the survivor remap, so the structural key actually feeds the merge it claims to.
Both directions are proven for each of the four inputs: the distinct variant keeps two
survivors, the identical variant collapses to one. The per-parent cases additionally exercise
the dependency half of the structural key (the union depends on a key step), proving two
per-parent unions merge only when their key steps merge too.

NO DB: every assertion is over the constructed step's plan-time structure and the dedup pass —
nothing connects, so this file carries no ``pg`` mark.
"""

from typing import Optional, Sequence

from sqlalchemy import TIMESTAMP, column

from grafast_py.core_steps import constant
from grafast_py.dag import Plan
from grafast_py.pg.cursor import encode_keyset_cursor
from grafast_py.pg.ordering import OrderTerm
from grafast_py.pg.resource import PgRegistry, PgResource
from grafast_py.pg.union import PgUnionAllStep, PgUnionMember

# the keyset order column (timestamptz) declares its SQL type so a text-origin cursor value
# casts back — mirrors the keyset fixture's KEYSET_TYPES and the union_all suite's CREATED_TYPE.
CREATED_TYPE = {"created": TIMESTAMP(timezone=True)}

# order_by=["created"] normalises to (created, id, __typename): the first member's primary
# key (id) is appended as the per-table tie-break, then the __typename discriminator as the
# FINAL union-wide tie-break (id alone is not unique across the UNION ALL — two members can
# share an id, so __typename is what makes the order total). A cursor must be minted under
# this triple to match the order digest the step decodes against; the DB-backed suite mints
# cursors the same way.
ORDER_TRIPLE = (OrderTerm("created"), OrderTerm("id"), OrderTerm("__typename"))


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


def make_union(
    *,
    key_step=None,
    members: Optional[Sequence[PgUnionMember]] = None,
    order_by: Sequence = ("created",),
    **kwargs,
) -> PgUnionAllStep:
    """An ``articles`` + ``snippets`` union over the shared (id, owner_id, created) shape.

    ``members`` overrides the default pair to vary the member set; ``key_step`` switches on
    per-parent mode (the default members then carry an ``owner_id`` match); ``order_by`` lets a
    case vary the ORDER BY. Each call builds FRESH resources/members so two unions never share a
    member object by identity — the dedup pass must merge on STRUCTURE, not object identity.
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
        order_by=order_by,
        key_step=key_step,
        **kwargs,
    )


def deduplicate(*steps: PgUnionAllStep):
    """Register ``steps`` into a fresh plan and run the real dedup pass; return the remap.

    Mirrors the planner: each step (and its transitive deps — the key step for a per-parent
    union) is added to a :class:`Plan`, then :meth:`Plan.deduplicate` collapses structurally
    identical steps and returns ``old id -> survivor``.
    """
    plan = Plan()
    for step in steps:
        plan.add_step(step)
    return plan.deduplicate()


# --------------------------------------------------------------- member set (both directions)


def test_identical_member_set_merges():
    """Two unions over the SAME member set (same tables/tags/order/page) collapse to one."""
    a = make_union(first=3, needs_total=True)
    b = make_union(first=3, needs_total=True)
    # same structural key even though the member/resource objects are distinct instances.
    assert a.peer_key == b.peer_key
    assert a.dedup_params() == b.dedup_params()

    remap = deduplicate(a, b)
    assert remap[a.id] is remap[b.id]


def test_different_member_table_does_not_merge():
    """Swapping one member for a DIFFERENT table is different SQL — two survivors."""
    posts = PgResource(
        "posts", "grafast_demo", "posts", ["id", "owner_id", "created", "title"],
        registry=PgRegistry(), column_types=CREATED_TYPE,
    )
    a = make_union(first=3)
    b = make_union(
        first=3,
        members=[
            PgUnionMember(make_articles(), "Article"),
            PgUnionMember(posts, "Post"),
        ],
    )
    assert a.peer_key != b.peer_key

    remap = deduplicate(a, b)
    assert remap[a.id] is a
    assert remap[b.id] is b


def test_different_member_typename_tag_does_not_merge():
    """A member tagged with a different concrete type emits a different ``__typename`` literal."""
    a = make_union(first=3)
    b = make_union(
        first=3,
        members=[
            PgUnionMember(make_articles(), "Story"),  # Article -> Story
            PgUnionMember(make_snippets(), "Snippet"),
        ],
    )
    assert a.peer_key != b.peer_key

    remap = deduplicate(a, b)
    assert remap[a.id] is a
    assert remap[b.id] is b


def test_member_order_does_not_merge():
    """Member ORDER is UNION ALL leg order (the SQL), so a reordered member set is distinct."""
    a = make_union(first=3)
    b = make_union(
        first=3,
        members=[
            PgUnionMember(make_snippets(), "Snippet"),  # legs swapped vs. a
            PgUnionMember(make_articles(), "Article"),
        ],
    )
    assert a.peer_key != b.peer_key

    remap = deduplicate(a, b)
    assert remap[a.id] is a
    assert remap[b.id] is b


# ----------------------------------------------------------- per-branch where (both directions)


def test_per_branch_where_same_value_merges():
    """Two unions whose branch filters carry the SAME value are peers and merge."""
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
            PgUnionMember(make_articles(), "Article", where=[column("owner_id") == 1]),
            PgUnionMember(make_snippets(), "Snippet"),
        ],
    )
    assert a.peer_key == b.peer_key
    assert a.dedup_params() == b.dedup_params()

    remap = deduplicate(a, b)
    assert remap[a.id] is remap[b.id]


def test_per_branch_where_value_does_not_merge():
    """Branch filters differing ONLY by VALUE (owner_id == 1 vs == 2) are distinct SQL."""
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
    assert a.peer_key != b.peer_key

    remap = deduplicate(a, b)
    assert remap[a.id] is a
    assert remap[b.id] is b


def test_per_branch_where_on_different_branch_does_not_merge():
    """The SAME predicate on a DIFFERENT member is different SQL (which leg is scoped matters)."""
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
            PgUnionMember(make_articles(), "Article"),
            PgUnionMember(make_snippets(), "Snippet", where=[column("owner_id") == 1]),
        ],
    )
    assert a.peer_key != b.peer_key

    remap = deduplicate(a, b)
    assert remap[a.id] is a
    assert remap[b.id] is b


# ----------------------------------------------------------------- order (both directions)


def test_same_order_merges():
    """Two unions with the SAME explicit order collapse to one."""
    a = make_union(first=3, order_by=[OrderTerm("created", descending=True)])
    b = make_union(first=3, order_by=[OrderTerm("created", descending=True)])
    assert a.peer_key == b.peer_key
    assert a.dedup_params() == b.dedup_params()

    remap = deduplicate(a, b)
    assert remap[a.id] is remap[b.id]


def test_order_direction_does_not_merge():
    """Same order COLUMN, opposite direction is a different keyset window — two survivors."""
    asc = make_union(first=3, order_by=[OrderTerm("created")])
    desc = make_union(first=3, order_by=[OrderTerm("created", descending=True)])
    assert asc.peer_key != desc.peer_key

    remap = deduplicate(asc, desc)
    assert remap[asc.id] is asc
    assert remap[desc.id] is desc


def test_order_nulls_placement_does_not_merge():
    """A differing NULLS placement on the order column changes the window — distinct keys."""
    nulls_first = make_union(first=3, order_by=[OrderTerm("created", nulls="first")])
    nulls_last = make_union(first=3, order_by=[OrderTerm("created", nulls="last")])
    assert nulls_first.peer_key != nulls_last.peer_key

    remap = deduplicate(nulls_first, nulls_last)
    assert remap[nulls_first.id] is nulls_first
    assert remap[nulls_last.id] is nulls_last


# ------------------------------------------------------------- after-cursor (both directions)


def test_same_after_cursor_merges():
    """Two unions paged from the SAME cursor are the same seek — they merge."""
    cur = encode_keyset_cursor(
        ORDER_TRIPLE,
        {"created": "2024-03-01T09:00:00+00:00", "id": 1, "__typename": "Article"},
    )
    a = make_union(first=2, after=cur)
    b = make_union(first=2, after=cur)
    assert a.peer_key == b.peer_key
    assert a.dedup_params() == b.dedup_params()

    remap = deduplicate(a, b)
    assert remap[a.id] is remap[b.id]


def test_after_cursor_value_does_not_merge():
    """Two unions differing ONLY by the ``after`` cursor VALUES seek different pages."""
    c1 = encode_keyset_cursor(
        ORDER_TRIPLE,
        {"created": "2024-03-01T08:00:00+00:00", "id": 1, "__typename": "Article"},
    )
    c2 = encode_keyset_cursor(
        ORDER_TRIPLE,
        {"created": "2024-03-01T11:00:00+00:00", "id": 2, "__typename": "Snippet"},
    )
    a = make_union(first=2, after=c1)
    b = make_union(first=2, after=c2)
    assert a.peer_key != b.peer_key

    remap = deduplicate(a, b)
    assert remap[a.id] is a
    assert remap[b.id] is b


def test_unpaged_and_after_cursor_do_not_merge():
    """An unpaged union (no cursor) and the same union seeking from an ``after`` are distinct."""
    cur = encode_keyset_cursor(
        ORDER_TRIPLE,
        {"created": "2024-03-01T09:00:00+00:00", "id": 1, "__typename": "Article"},
    )
    unpaged = make_union(first=2)
    paged = make_union(first=2, after=cur)
    assert unpaged.peer_key != paged.peer_key

    remap = deduplicate(unpaged, paged)
    assert remap[unpaged.id] is unpaged
    assert remap[paged.id] is paged


# ----------------------------------------- per-parent: the key-step dependency also discriminates


def test_per_parent_unions_merge_when_key_steps_merge():
    """Two per-parent unions over a SHARED key step collapse — deps + structure both match.

    Per-parent mode wires the key step as dependency 0, so the dedup key folds in the key
    step's survivor id. A single shared key step keeps both unions' dep id equal, so the
    identical-structure unions merge into one.
    """
    key = constant(("owners",))
    a = make_union(key_step=key, first=2)
    b = make_union(key_step=key, first=2)
    assert a.peer_key == b.peer_key

    remap = deduplicate(a, b)
    assert remap[a.id] is remap[b.id]


def test_per_parent_unions_with_distinct_key_steps_do_not_merge():
    """Identical-structure per-parent unions over DIFFERENT (non-merging) key steps stay split.

    The key steps carry different constants, so they do not merge; the unions then depend on
    different survivor ids and — despite identical peer_key/params — are not peers. This proves
    the union's dependency participates in its structural identity (not just its own params).
    """
    a = make_union(key_step=constant(("owners-a",)), first=2)
    b = make_union(key_step=constant(("owners-b",)), first=2)
    # same own structural key: the difference is entirely in the (distinct) key-step dep.
    assert a.peer_key == b.peer_key
    assert a.dedup_params() == b.dedup_params()

    remap = deduplicate(a, b)
    assert remap[a.id] is a
    assert remap[b.id] is b


def test_per_parent_and_root_do_not_merge():
    """The same logical members per-parent vs root are different SQL — never peers."""
    root = make_union(first=2)
    per_parent = make_union(key_step=constant(None), first=2)
    assert root.peer_key != per_parent.peer_key

    remap = deduplicate(root, per_parent)
    assert remap[root.id] is root
    assert remap[per_parent.id] is per_parent
