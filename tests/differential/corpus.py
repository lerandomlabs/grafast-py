"""Shared differential corpus (Python / grafast-py side).

This is the grafast-py twin of ``node/corpus.mjs``. The two files MUST encode
the same fixtures: same ``name``, same SDL text, same seed values, matched plan
wiring, byte-identical ``query`` and ``variables``. The differ asserts the
fixture-name SET is identical across the two result files, so any coverage drift is
itself a hard failure.

A fixture is ``{name, sdl, plans, query, variables}``. ``plans`` is the
``make_grafast_schema`` ``{"Type": {"field": plan}}`` map. Batch-load callbacks are
referenced by stable names so the harness can wrap each one with a fetch counter;
loaders a fixture never triggers are absent from its fetchCounts (both sides agree on
absence).

Argument-shape note (mirrors notes/phaseC-reference-api.md): grafast (JS) passes
field args as STEPS — ``args.getRaw("id")`` is an input step fed straight into the
plan (e.g. ``loadOne(args.getRaw("id"), loadAuthors)``), and value computation goes
through ``lambda(args.getRaw(name), fn)``. grafast-py's ``FieldArgs.get(name)``
returns the already-coerced PLAIN value, so the mirror lifts it with ``constant(...)``
where a step is required and computes directly otherwise. Both sides feed the SAME
final value into the SAME loader/step, so data and fetch counts are identical.
"""

from typing import Any, Dict, List

from grafast_py import (
    access,
    constant,
    each,
    lambda_step,
    load_many,
    load_one,
    resolve_type_from_discriminator,
    resolve_type_from_tag,
)

# --------------------------------------------------------------------- seed data
# Plain JSON-able values, written identically in node/corpus.mjs.

AUTHORS: List[Dict[str, Any]] = [
    {"id": 1, "name": "Ada", "bio": None, "tags": ["math", "engine"]},
    {"id": 2, "name": "Babbage", "bio": "engines", "tags": []},
    {"id": 3, "name": "Curie", "bio": "radioactivity", "tags": ["physics"]},
]

AUTHOR_BY_ID: Dict[int, Dict[str, Any]] = {a["id"]: a for a in AUTHORS}

# posts per author id; author 3 has no posts (empty-list fixture).
POSTS_BY_AUTHOR: Dict[int, List[Dict[str, Any]]] = {
    1: [
        {"id": 11, "title": "A1", "authorId": 1},
        {"id": 12, "title": "A2", "authorId": 1},
    ],
    2: [{"id": 21, "title": "B1", "authorId": 2}],
    3: [],
}

ALL_POSTS: List[Dict[str, Any]] = [
    {"id": 11, "title": "A1", "authorId": 1},
    {"id": 12, "title": "A2", "authorId": 1},
    {"id": 21, "title": "B1", "authorId": 2},
]

COMMENTS_BY_POST: Dict[int, List[Dict[str, Any]]] = {
    11: [{"id": 111, "body": "c-a"}, {"id": 112, "body": "c-b"}],
    12: [{"id": 121, "body": "c-c"}],
    21: [{"id": 211, "body": "c-d"}],
}

COAUTHOR_IDS: Dict[int, List[int]] = {1: [2, 3], 2: [1], 3: []}


# --------------------------------------------------------- named batch callbacks
# Each takes the list of ALL lookup keys in the bucket and returns an index-aligned
# list (loadOne: one record per key; loadMany: one sub-list per key). The harness
# wraps these with a per-fixture counter keyed by the same names as the JS side.

def load_authors(ids: List[int]) -> List[Any]:
    return [AUTHOR_BY_ID.get(i) for i in ids]


def load_posts_by_author(ids: List[int]) -> List[Any]:
    return [POSTS_BY_AUTHOR.get(i, []) for i in ids]


def load_comments_by_post(ids: List[int]) -> List[Any]:
    return [COMMENTS_BY_POST.get(i, []) for i in ids]


LOADERS: Dict[str, Any] = {
    "loadAuthors": load_authors,
    "loadPostsByAuthor": load_posts_by_author,
    "loadCommentsByPost": load_comments_by_post,
}

# ------------------------------------------------------------------------- SDL
# IDENTICAL text to node/corpus.mjs SDL_BLOG.
SDL_BLOG = """
  type Query {
    hello: String
    answer: Int
    flag: Boolean
    me: Author
    authors: [Author!]!
    author(id: Int!): Author
    posts: [Post!]!
    echo(n: Int): Int
    color(pick: Hue!): String
    greet(who: Name!): String
    sum(xs: [Int!]!): Int
    boom: String
  }
  type Author {
    id: Int!
    name: String!
    bio: String
    tags: [String!]!
    posts: [Post!]!
    coauthors: [Author!]!
    title: String!
    nullTags: [String!]!
  }
  type Post {
    id: Int!
    title: String!
    author: Author
    comments: [Comment!]!
  }
  type Comment {
    id: Int!
    body: String!
  }
  enum Hue { RED GREEN BLUE }
  input Name { first: String! last: String! }
"""


# ------------------------------------------------------------ shared plan pieces
def leaf(key: str):
    """A leaf plan reading ``key`` off the parent step (mirror of JS ``access($p,key)``)."""
    return lambda p, args, info: access(p, [key])


def objects_for(query_plans: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build the plan map for a fixture from a partial set of Query plans.

    Re-uses the shared Author/Post/Comment plans so every fixture has the same
    nested wiring as the JS ``objectsFor``. ``LOADERS[...]`` is read at plan-build
    time so the harness's per-fixture wrapping is observed.
    """
    return {
        "Query": query_plans,
        "Author": {
            "id": leaf("id"),
            "name": leaf("name"),
            "bio": leaf("bio"),
            "tags": leaf("tags"),
            # NON-NULL String field always null in the data -> bubbles.
            "title": lambda p, args, info: constant(None),
            # [String!]! whose 2nd element is null -> element bubbles, list NonNull.
            "nullTags": lambda p, args, info: constant(["ok", None]),
            "posts": lambda p, args, info: load_many(
                access(p, ["id"]), LOADERS["loadPostsByAuthor"]
            ),
            "coauthors": lambda p, args, info: each(
                lambda_step(access(p, ["id"]), lambda aid: COAUTHOR_IDS[aid]),
                lambda item: load_one(item, LOADERS["loadAuthors"]),
            ),
        },
        "Post": {
            "id": leaf("id"),
            "title": leaf("title"),
            "author": lambda p, args, info: load_one(
                access(p, ["authorId"]), LOADERS["loadAuthors"]
            ),
            "comments": lambda p, args, info: load_many(
                access(p, ["id"]), LOADERS["loadCommentsByPost"]
            ),
        },
        "Comment": {"id": leaf("id"), "body": leaf("body")},
    }


# ----------------------------------------------------- polymorphism (abstract types)
# Interface + union over a list of MIXED concrete types — the batching profile the
# blog corpus never exercised. The decisive case: a relation selected on two concrete
# types in one list must fire each loader ONCE (per concrete-type group), which the
# differ cross-checks against reference Node Grafast. Mirrors node/corpus.mjs SDL_POLY.
SDL_POLY = """
  type Query {
    feed: [Content!]!
    item(id: Int!): Content
    search: [Hit!]!
  }
  interface Content { id: Int! }
  type Article implements Content { id: Int! headline: String! author: Author }
  type Photo implements Content { id: Int! caption: String! tags: [Tag!]! }
  type Author { id: Int! name: String! }
  type Tag { id: Int! label: String! }
  union Hit = Article | Photo
"""

# Each row carries BOTH a `kind` discriminator (interface bridge) and a `__typename`
# tag (union bridge). Articles point at AUTHOR_BY_ID; photos at TAGS_BY_PHOTO.
FEED: List[Dict[str, Any]] = [
    {"id": 1, "kind": "article", "__typename": "Article", "headline": "H1", "authorId": 1},
    {"id": 2, "kind": "photo", "__typename": "Photo", "caption": "C2"},
    {"id": 3, "kind": "article", "__typename": "Article", "headline": "H3", "authorId": 2},
    {"id": 4, "kind": "photo", "__typename": "Photo", "caption": "C4"},
    {"id": 5, "kind": "article", "__typename": "Article", "headline": "H5", "authorId": 1},
]
FEED_BY_ID: Dict[int, Dict[str, Any]] = {r["id"]: r for r in FEED}

TAGS_BY_PHOTO: Dict[int, List[Dict[str, Any]]] = {
    2: [{"id": 201, "label": "sky"}],
    4: [{"id": 202, "label": "sea"}, {"id": 203, "label": "sun"}],
}


def load_tags_by_photo(ids: List[int]) -> List[Any]:
    return [TAGS_BY_PHOTO.get(i, []) for i in ids]


LOADERS["loadTagsByPhoto"] = load_tags_by_photo


def poly_objects_for(query_plans: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Plan map for the abstract schema: Article/Photo each load a DIFFERENT relation.

    Article.author re-uses the shared ``loadAuthors`` (loadOne by id); Photo.tags uses
    ``loadTagsByPhoto`` (loadMany by photo id). A query selecting both on a Content list
    runs each in its own concrete-type group bucket — one batched call apiece.
    """
    return {
        "Query": query_plans,
        "Article": {
            "id": leaf("id"),
            "headline": leaf("headline"),
            "author": lambda p, args, info: load_one(
                access(p, ["authorId"]), LOADERS["loadAuthors"]
            ),
        },
        "Photo": {
            "id": leaf("id"),
            "caption": leaf("caption"),
            "tags": lambda p, args, info: load_many(
                access(p, ["id"]), LOADERS["loadTagsByPhoto"]
            ),
        },
        "Author": {"id": leaf("id"), "name": leaf("name")},
        "Tag": {"id": leaf("id"), "label": leaf("label")},
    }


# The resolve_type bridges: interface by discriminator column, union by typename tag.
POLY_TYPE_RESOLVERS = {
    "Content": resolve_type_from_discriminator(
        "kind", {"article": "Article", "photo": "Photo"}
    ),
    "Hit": resolve_type_from_tag("__typename"),
}


# ----------------------------------------------------------------- the fixtures
# The list ORDER and the `name` strings are the single source of truth shared with
# node/corpus.mjs.
FIXTURES = [
    {
        "name": "flat_scalars",
        "sdl": SDL_BLOG,
        "plans": objects_for(
            {
                "hello": lambda p, a, i: constant("world"),
                "answer": lambda p, a, i: constant(42),
                "flag": lambda p, a, i: constant(True),
            }
        ),
        "query": "{ hello answer flag }",
        "variables": {},
    },
    {
        "name": "nested_object",
        "sdl": SDL_BLOG,
        "plans": objects_for({"me": lambda p, a, i: constant(AUTHORS[0])}),
        "query": "{ me { id name } }",
        "variables": {},
    },
    {
        "name": "list_of_objects",
        "sdl": SDL_BLOG,
        "plans": objects_for({"authors": lambda p, a, i: constant(AUTHORS)}),
        "query": "{ authors { id name } }",
        "variables": {},
    },
    {
        "name": "list_of_leaves",
        "sdl": SDL_BLOG,
        "plans": objects_for(
            {"author": lambda p, a, i: load_one(constant(a.get("id")), LOADERS["loadAuthors"])}
        ),
        "query": "{ author(id: 1) { tags } }",
        "variables": {},
    },
    {
        "name": "deep_nesting",
        "sdl": SDL_BLOG,
        "plans": objects_for({"authors": lambda p, a, i: constant(AUTHORS)}),
        "query": "{ authors { posts { comments { body } } } }",
        "variables": {},
    },
    {
        "name": "null_leaf_nullable",
        "sdl": SDL_BLOG,
        "plans": objects_for(
            {"author": lambda p, a, i: load_one(constant(a.get("id")), LOADERS["loadAuthors"])}
        ),
        "query": "{ author(id: 1) { bio } }",
        "variables": {},
    },
    {
        "name": "null_in_nonnull_list",
        "sdl": SDL_BLOG,
        "plans": objects_for(
            {"author": lambda p, a, i: load_one(constant(a.get("id")), LOADERS["loadAuthors"])}
        ),
        "query": "{ author(id: 1) { nullTags } }",
        "variables": {},
    },
    {
        "name": "nonnull_field_returns_null",
        "sdl": SDL_BLOG,
        "plans": objects_for(
            {"author": lambda p, a, i: load_one(constant(a.get("id")), LOADERS["loadAuthors"])}
        ),
        "query": "{ author(id: 1) { title } }",
        "variables": {},
    },
    {
        "name": "nonnull_bubbles_to_root",
        "sdl": SDL_BLOG,
        "plans": objects_for(
            {"author": lambda p, a, i: load_one(constant(a.get("id")), LOADERS["loadAuthors"])}
        ),
        # authors is [Author!]!; every Author.title is String! returning null, so the
        # element bubbles to the NonNull list, which bubbles to the NonNull root field
        # `authors`, making data null with one error at ["authors"].
        "query": "{ authors { title } }",
        "variables": {},
    },
    {
        "name": "aliases",
        "sdl": SDL_BLOG,
        "plans": objects_for(
            {
                "hello": lambda p, a, i: constant("world"),
                "answer": lambda p, a, i: constant(42),
                "me": lambda p, a, i: constant(AUTHORS[0]),
            }
        ),
        "query": "{ a: hello b: hello x: answer who: me { ident: id label: name } }",
        "variables": {},
    },
    {
        "name": "arg_scalar",
        "sdl": SDL_BLOG,
        "plans": objects_for({"echo": lambda p, a, i: constant(a.get("n"))}),
        "query": "{ echo(n: 7) }",
        "variables": {},
    },
    {
        "name": "arg_enum",
        "sdl": SDL_BLOG,
        "plans": objects_for(
            {"color": lambda p, a, i: constant(f"picked:{a.get('pick')}")}
        ),
        "query": "{ color(pick: GREEN) }",
        "variables": {},
    },
    {
        "name": "arg_input_object",
        "sdl": SDL_BLOG,
        "plans": objects_for(
            {
                "greet": lambda p, a, i: constant(
                    f"{a.get('who')['first']} {a.get('who')['last']}"
                )
            }
        ),
        "query": '{ greet(who: { first: "Ada", last: "Lovelace" }) }',
        "variables": {},
    },
    {
        "name": "arg_list",
        "sdl": SDL_BLOG,
        "plans": objects_for({"sum": lambda p, a, i: constant(sum(a.get("xs")))}),
        "query": "{ sum(xs: [1, 2, 3]) }",
        "variables": {},
    },
    {
        "name": "var_required",
        "sdl": SDL_BLOG,
        "plans": objects_for(
            {"author": lambda p, a, i: load_one(constant(a.get("id")), LOADERS["loadAuthors"])}
        ),
        "query": "query Q($id: Int!) { author(id: $id) { name } }",
        "variables": {"id": 2},
    },
    {
        "name": "var_default",
        "sdl": SDL_BLOG,
        "plans": objects_for({"echo": lambda p, a, i: constant(a.get("n"))}),
        "query": "query Q($n: Int = 5) { echo(n: $n) }",
        "variables": {},
    },
    {
        "name": "fragment_spread",
        "sdl": SDL_BLOG,
        "plans": objects_for({"me": lambda p, a, i: constant(AUTHORS[1])}),
        "query": "{ me { ...F } } fragment F on Author { id name bio }",
        "variables": {},
    },
    {
        "name": "inline_fragment",
        "sdl": SDL_BLOG,
        "plans": objects_for({"me": lambda p, a, i: constant(AUTHORS[1])}),
        "query": "{ me { id ... on Author { bio } } }",
        "variables": {},
    },
    {
        "name": "skip_true",
        "sdl": SDL_BLOG,
        "plans": objects_for({"me": lambda p, a, i: constant(AUTHORS[0])}),
        "query": "query Q($s: Boolean!) { me { id name @skip(if: $s) } }",
        "variables": {"s": True},
    },
    {
        "name": "include_false",
        "sdl": SDL_BLOG,
        "plans": objects_for({"me": lambda p, a, i: constant(AUTHORS[0])}),
        "query": "query Q($i: Boolean!) { me { id name @include(if: $i) } }",
        "variables": {"i": False},
    },
    {
        "name": "loadmany_n_plus_1",
        "sdl": SDL_BLOG,
        "plans": objects_for({"authors": lambda p, a, i: constant(AUTHORS)}),
        "query": "{ authors { posts { id } } }",
        "variables": {},
    },
    {
        "name": "loadone_n_plus_1",
        "sdl": SDL_BLOG,
        "plans": objects_for({"posts": lambda p, a, i: constant(ALL_POSTS)}),
        "query": "{ posts { author { name } } }",
        "variables": {},
    },
    {
        "name": "each_loadone",
        "sdl": SDL_BLOG,
        "plans": objects_for({"authors": lambda p, a, i: constant(AUTHORS)}),
        "query": "{ authors { coauthors { name } } }",
        "variables": {},
    },
    {
        "name": "empty_list",
        "sdl": SDL_BLOG,
        "plans": objects_for(
            {"author": lambda p, a, i: load_one(constant(a.get("id")), LOADERS["loadAuthors"])}
        ),
        "query": "{ author(id: 3) { posts { id } } }",
        "variables": {},
    },
    {
        "name": "explicit_error",
        "sdl": SDL_BLOG,
        "plans": objects_for({"boom": lambda p, a, i: _boom_step()}),
        "query": "{ boom }",
        "variables": {},
    },
    # ---- polymorphism: interface/union batching over mixed concrete-type lists ----
    {
        "name": "iface_list_typename",
        "sdl": SDL_POLY,
        "plans": poly_objects_for({"feed": lambda p, a, i: constant(FEED)}),
        "type_resolvers": POLY_TYPE_RESOLVERS,
        # pure projection (__typename + interface id): no loader fires.
        "query": "{ feed { __typename id } }",
        "variables": {},
    },
    {
        "name": "iface_list_relation_one_type",
        "sdl": SDL_POLY,
        "plans": poly_objects_for({"feed": lambda p, a, i: constant(FEED)}),
        "type_resolvers": POLY_TYPE_RESOLVERS,
        # only the Photo group loads tags -> loadTagsByPhoto once over [2, 4].
        "query": "{ feed { ... on Photo { tags { label } } } }",
        "variables": {},
    },
    {
        "name": "iface_list_relation_both_types",
        "sdl": SDL_POLY,
        "plans": poly_objects_for({"feed": lambda p, a, i: constant(FEED)}),
        "type_resolvers": POLY_TYPE_RESOLVERS,
        # Article group -> loadAuthors once over [1,2,1]; Photo group -> loadTagsByPhoto
        # once over [2,4]. The headline multi-concrete-type-over-a-list batching case.
        "query": (
            "{ feed { id ... on Article { author { name } } "
            "... on Photo { tags { label } } } }"
        ),
        "variables": {},
    },
    {
        "name": "iface_single_relation",
        "sdl": SDL_POLY,
        "plans": poly_objects_for(
            {"item": lambda p, a, i: constant(FEED_BY_ID.get(a.get("id")))}
        ),
        "type_resolvers": POLY_TYPE_RESOLVERS,
        # a single nullable interface value (an Article) -> loadAuthors once (1 key).
        "query": "{ item(id: 1) { ... on Article { author { name } } } }",
        "variables": {},
    },
    {
        "name": "union_list_relation_both",
        "sdl": SDL_POLY,
        "plans": poly_objects_for({"search": lambda p, a, i: constant(FEED)}),
        "type_resolvers": POLY_TYPE_RESOLVERS,
        # same shape as iface_list_relation_both_types but via the union tag bridge.
        "query": (
            "{ search { ... on Article { author { name } } "
            "... on Photo { tags { label } } } }"
        ),
        "variables": {},
    },
]


def _raise_boom(_value: Any) -> Any:
    raise RuntimeError("boom")


def _boom_step():
    """A step whose execution raises ``Exception("boom")`` (mirror of JS throw)."""
    return lambda_step(constant(0), _raise_boom)
