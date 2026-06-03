"""Plan-resolver conformance parity.

The graphql-core execution suite proves the resolver-adapter path is correct, but
it never exercises the plan-resolver path (it uses plain `resolve=` functions). This
file closes that gap: it re-expresses a representative subset of the conformance
schema shapes — objects, lists (incl. lists of objects and lists of leaves), nested
selection, nullability + null-bubbling through NonNull, field arguments (scalar +
boolean + default), and aliases — TWICE over the SAME static data:

  * a PLAIN schema with ordinary `resolve=` functions, executed through the engine's
    resolver-adapter path (and cross-checked against stock graphql-core), and
  * a PLAN schema where every field carries a plan resolver
    (`extensions["grafast"]["plan"]`) building a step DAG (access / constant /
    lambda / loadOne / loadMany / each), executed through the genuine
    plan-then-execute path.

For each query in a battery we assert the plan path's `data` AND `errors` (message +
path) are IDENTICAL to the plain path's — i.e. the plan-resolver engine is a faithful
re-expression, not merely "doesn't crash". Static dicts only; no database.
"""

from typing import Any, Dict, List, Optional

from graphql import (
    GraphQLArgument,
    GraphQLBoolean,
    GraphQLField,
    GraphQLInt,
    GraphQLList,
    GraphQLNonNull,
    GraphQLObjectType,
    GraphQLSchema,
    GraphQLString,
    execute,
    parse,
)
from graphql.execution.execute import ExecutionContext

from grafast_py import (
    GrafastExecutionContext,
    access,
    constant,
    each,
    lambda_step,
    load_many,
    load_one,
    make_grafast_schema,
)

# --------------------------------------------------------------------- shared data
# A tiny blog graph. `bio` is a NULLABLE field that is None for some authors;
# `name` is NON-NULL. Author 99 does not exist (a nullable `author(id:)` lookup that
# returns null). Post 3 has a NON-NULL `title` that is None in the data, so selecting
# it must bubble the null up to the nullable post and produce an error — the same on
# both paths.
AUTHORS: Dict[int, Dict[str, Any]] = {
    1: {"id": 1, "name": "Ada", "bio": None, "post_ids": [1, 2]},
    2: {"id": 2, "name": "Linus", "bio": "kernel", "post_ids": [3]},
    3: {"id": 3, "name": "Grace", "bio": "compiler", "post_ids": []},
}
POSTS: Dict[int, Dict[str, Any]] = {
    1: {"id": 1, "title": "Hello", "tags": ["a", "b"]},
    2: {"id": 2, "title": "World", "tags": []},
    3: {"id": 3, "title": None, "tags": ["x"]},  # NON-NULL title is None -> bubbles
}


def load_authors(keys: List[int]) -> List[Optional[Dict[str, Any]]]:
    return [AUTHORS.get(k) for k in keys]


def load_posts_for_author(keys: List[int]) -> List[List[Dict[str, Any]]]:
    out: List[List[Dict[str, Any]]] = []
    for author_id in keys:
        author = AUTHORS.get(author_id)
        ids = author["post_ids"] if author else []
        out.append([POSTS[pid] for pid in ids])
    return out


# ------------------------------------------------------------------- plain schema
def build_plain_schema() -> GraphQLSchema:
    """The reference: ordinary resolvers over the shared data."""

    def r_id(o, info):
        return o["id"]

    def r_name(o, info):
        return o["name"]

    def r_bio(o, info):
        return o["bio"]

    def r_title(o, info):
        return o["title"]

    def r_tags(o, info):
        return o["tags"]

    def r_shout_title(o, info, upper=False):
        title = o["title"]
        if upper and title is not None:
            return title.upper()
        return title

    post_type = GraphQLObjectType(
        "Post",
        lambda: {
            "id": GraphQLField(GraphQLNonNull(GraphQLInt), resolve=r_id),
            "title": GraphQLField(GraphQLNonNull(GraphQLString), resolve=r_title),
            "shoutTitle": GraphQLField(
                GraphQLString,
                args={"upper": GraphQLArgument(GraphQLBoolean, default_value=False)},
                resolve=r_shout_title,
            ),
            "tags": GraphQLField(
                GraphQLNonNull(GraphQLList(GraphQLNonNull(GraphQLString))),
                resolve=r_tags,
            ),
        },
    )
    author_type = GraphQLObjectType(
        "Author",
        lambda: {
            "id": GraphQLField(GraphQLNonNull(GraphQLInt), resolve=r_id),
            "name": GraphQLField(GraphQLNonNull(GraphQLString), resolve=r_name),
            "bio": GraphQLField(GraphQLString, resolve=r_bio),
            "posts": GraphQLField(
                GraphQLNonNull(GraphQLList(GraphQLNonNull(post_type))),
                resolve=lambda o, info: [POSTS[pid] for pid in o["post_ids"]],
            ),
        },
    )

    def r_authors(root, info):
        return list(AUTHORS.values())

    def r_author(root, info, id):
        return AUTHORS.get(id)

    query_type = GraphQLObjectType(
        "Query",
        {
            "authors": GraphQLField(
                GraphQLNonNull(GraphQLList(GraphQLNonNull(author_type))),
                resolve=r_authors,
            ),
            "author": GraphQLField(
                author_type,
                args={"id": GraphQLArgument(GraphQLNonNull(GraphQLInt))},
                resolve=r_author,
            ),
        },
    )
    return GraphQLSchema(query=query_type)


# -------------------------------------------------------------------- plan schema
PLAN_SDL = """
type Query {
  authors: [Author!]!
  author(id: Int!): Author
}

type Author {
  id: Int!
  name: String!
  bio: String
  posts: [Post!]!
}

type Post {
  id: Int!
  title: String!
  shoutTitle(upper: Boolean = false): String
  tags: [String!]!
}
"""


def build_plan_schema() -> GraphQLSchema:
    """The same shapes, every field a plan resolver building a step DAG."""

    def plan_authors(parent, args, info):
        # one batched load returning all authors (keyed on a sentinel constant)
        return load_many(constant("__all__"), lambda keys: [list(AUTHORS.values())])

    def plan_author(parent, args, info):
        return load_one(constant(args["id"]), load_authors)

    def plan_id(parent, args, info):
        return access(parent, ["id"])

    def plan_name(parent, args, info):
        return access(parent, ["name"])

    def plan_bio(parent, args, info):
        return access(parent, ["bio"])

    def plan_title(parent, args, info):
        return access(parent, ["title"])

    def plan_tags(parent, args, info):
        return access(parent, ["tags"])

    def plan_shout_title(parent, args, info):
        title = access(parent, ["title"])
        if args.get("upper"):
            return lambda_step(title, lambda v: v.upper() if v is not None else v)
        return title

    def plan_posts(parent, args, info):
        # batched loadMany over every author's id: one call for the whole bucket
        return load_many(access(parent, ["id"]), load_posts_for_author)

    return make_grafast_schema(
        PLAN_SDL,
        {
            "Query": {"authors": plan_authors, "author": plan_author},
            "Author": {
                "id": plan_id,
                "name": plan_name,
                "bio": plan_bio,
                "posts": plan_posts,
            },
            "Post": {
                "id": plan_id,
                "title": plan_title,
                "shoutTitle": plan_shout_title,
                "tags": plan_tags,
            },
        },
    )


# ----------------------------------------------------------------------- helpers
PLAIN_SCHEMA = build_plain_schema()
PLAN_SCHEMA = build_plan_schema()


def errors_as_tuples(result):
    """Normalise errors to (message, path) tuples for order-insensitive comparison."""
    if not result.errors:
        return []
    return sorted((e.message, tuple(e.path) if e.path else None) for e in result.errors)


def run_plain(query, variables=None):
    return execute(
        PLAIN_SCHEMA,
        parse(query),
        execution_context_class=ExecutionContext,
        variable_values=variables,
    )


def run_plain_adapter(query, variables=None):
    """The plain schema through OUR engine's resolver-adapter path."""
    return execute(
        PLAIN_SCHEMA,
        parse(query),
        execution_context_class=GrafastExecutionContext,
        variable_values=variables,
    )


def run_plan(query, variables=None):
    return execute(
        PLAN_SCHEMA,
        parse(query),
        execution_context_class=GrafastExecutionContext,
        variable_values=variables,
    )


# ------------------------------------------------------------------- the battery
# Each entry is a query (some with variables) that, executed against either schema,
# must yield identical data + errors. Covers: simple object, aliases, nested
# objects, lists of objects, lists of leaves, arguments (NonNull Int + Boolean +
# default), nullable null result, and null-bubbling through a NonNull field.
QUERIES = [
    # simple object + leaves
    "{ authors { id name } }",
    # nullable leaf that is None for author 1, present for author 2
    "{ authors { name bio } }",
    # aliases on fields and on a selection
    "{ people: authors { ident: id label: name } }",
    # nested object list (authors -> posts), list of leaves (tags)
    "{ authors { name posts { id title tags } } }",
    # argument: NonNull Int lookup that hits
    "{ author(id: 2) { id name bio } }",
    # argument: NonNull Int lookup that misses -> nullable null
    "{ author(id: 99) { id } }",
    # boolean argument with explicit value
    "{ author(id: 1) { posts { id shoutTitle(upper: true) } } }",
    # boolean argument default (false) -> title passthrough (post 1 has a title)
    "{ author(id: 1) { posts { shoutTitle } } }",
    # empty list (author 3 has no posts)
    "{ author(id: 3) { name posts { id } } }",
    # NULL-BUBBLING: post 3's NonNull title is None -> bubbles to nullable post item;
    # the list item is non-null though, so it bubbles to the NonNull list, to posts
    # (NonNull) up to the nullable author. Identical on both paths.
    "{ author(id: 2) { name posts { id title } } }",
    # the same null-bubbling but title not selected -> no error
    "{ author(id: 2) { name posts { id tags } } }",
]


def test_plain_adapter_matches_stock_graphql_core():
    """Sanity: the plain schema through OUR adapter == stock graphql-core."""
    for query in QUERIES:
        stock = run_plain(query)
        ours = run_plain_adapter(query)
        assert ours.data == stock.data, query
        assert errors_as_tuples(ours) == errors_as_tuples(stock), query


def test_plan_resolver_results_match_plain_resolvers():
    """THE GATE: plan-resolver path data+errors == plain-resolver path, every query."""
    for query in QUERIES:
        plain = run_plain(query)
        plan = run_plan(query)
        assert plan.data == plain.data, f"data mismatch for {query!r}"
        assert errors_as_tuples(plan) == errors_as_tuples(plain), (
            f"errors mismatch for {query!r}: "
            f"{errors_as_tuples(plan)} != {errors_as_tuples(plain)}"
        )


def test_plan_resolver_argument_via_variable_matches_plain():
    """Arguments fed through a query VARIABLE coerce + plan identically to plain."""
    query = "query Q($id: Int!) { author(id: $id) { id name } }"
    for author_id in (1, 2, 3, 99):
        plain = run_plain(query, {"id": author_id})
        plan = run_plan(query, {"id": author_id})
        assert plan.data == plain.data, author_id
        assert errors_as_tuples(plan) == errors_as_tuples(plain), author_id


def test_each_loadone_relation_matches_plain():
    """An `each(loadOne)` relation re-expressed plan-side matches a plain resolver.

    `author.coauthors` resolves a list of author ids to author objects: plain does it
    inline, plan does it as `each($ids, loadOne)` — one batched author load for the
    whole bucket — and the visible result must be identical.
    """
    coauthor_ids = {1: [2, 3], 2: [1], 3: []}

    def r_coauthors(o, info):
        return [AUTHORS[i] for i in coauthor_ids[o["id"]]]

    plain_post = GraphQLObjectType(
        "P", {"id": GraphQLField(GraphQLNonNull(GraphQLInt), resolve=lambda o, i: o["id"])}
    )
    plain_author = GraphQLObjectType(
        "A",
        lambda: {
            "id": GraphQLField(GraphQLNonNull(GraphQLInt), resolve=lambda o, i: o["id"]),
            "name": GraphQLField(GraphQLNonNull(GraphQLString), resolve=lambda o, i: o["name"]),
            "coauthors": GraphQLField(
                GraphQLNonNull(GraphQLList(GraphQLNonNull(plain_author))),
                resolve=r_coauthors,
            ),
        },
    )
    plain_schema = GraphQLSchema(
        query=GraphQLObjectType(
            "Query",
            {
                "authors": GraphQLField(
                    GraphQLNonNull(GraphQLList(GraphQLNonNull(plain_author))),
                    resolve=lambda root, info: list(AUTHORS.values()),
                )
            },
        )
    )

    sdl = """
    type Query { authors: [A!]! }
    type A { id: Int! name: String! coauthors: [A!]! }
    """

    def plan_authors(p, a, i):
        return load_many(constant("__all__"), lambda keys: [list(AUTHORS.values())])

    def plan_coauthors(p, a, i):
        ids = access(p, ["id"])
        # resolve this author's coauthor id list, then loadOne each id (batched)
        id_list = lambda_step(ids, lambda author_id: coauthor_ids[author_id])
        return each(id_list, lambda item: load_one(item, load_authors))

    plan_schema = make_grafast_schema(
        sdl,
        {
            "Query": {"authors": plan_authors},
            "A": {
                "id": lambda p, a, i: access(p, ["id"]),
                "name": lambda p, a, i: access(p, ["name"]),
                "coauthors": plan_coauthors,
            },
        },
    )

    query = "{ authors { name coauthors { id name } } }"
    plain = execute(plain_schema, parse(query), execution_context_class=ExecutionContext)
    plan = execute(plan_schema, parse(query), execution_context_class=GrafastExecutionContext)
    assert plan.data == plain.data
    assert errors_as_tuples(plan) == errors_as_tuples(plain)
