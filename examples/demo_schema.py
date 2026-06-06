"""An end-to-end GraphQL schema fully backed by Postgres via plan resolvers.

Builds (via :func:`grafast_py.schema.make_grafast_schema`) a schema over the
``grafast_demo`` tables where EVERY field is a plan resolver returning a pg step:

- ``Query.authors`` / ``Query.author(id)`` / ``Query.posts`` — root collections.
- ``Author.posts`` (hasMany), ``Author.postsConnection(first, after)`` (Relay).
- ``Post.author`` (hasOne), ``Post.comments`` (hasMany).
- ``Comment.author`` (hasOne), ``Comment.post`` (hasOne).

Runnable through :class:`grafast_py.GrafastExecutionContext`. The registry +
resources are built once and shared; root plans seed their key step from a constant
so a root list fetches all rows in ONE statement and the relation layers chain off
the returned row steps (one batched statement per resource-layer).
"""

from typing import Optional, Tuple

from graphql import GraphQLSchema

from grafast_py.core_steps import access, constant
from grafast_py.schema import make_grafast_schema
from grafast_py.pg.connection import connection, connection_needs_total
from grafast_py.pg.mutations import (
    pg_delete_single,
    pg_insert_single,
    pg_update_single,
)
from grafast_py.pg.resource import PgRegistry, PgResource
from grafast_py.pg.steps import pg_select, pg_select_single

SDL = """
type Query {
  authors: [Author!]!
  author(id: Int!): Author
  posts: [Post!]!
}

type Mutation {
  createPost(input: CreatePostInput!): Post
  updatePost(id: Int!, input: UpdatePostInput!): Post
  deletePost(id: Int!): Post
}

input CreatePostInput {
  id: Int!
  authorId: Int!
  title: String!
}

input UpdatePostInput {
  title: String
}

type Author {
  id: Int!
  name: String!
  posts: [Post!]!
  postsConnection(
    first: Int, after: String, last: Int, before: String
  ): PostConnection!
}

type Post {
  id: Int!
  title: String!
  author: Author!
  comments: [Comment!]!
}

type Comment {
  id: Int!
  body: String!
  author: Author!
  post: Post!
}

type PostConnection {
  totalCount: Int!
  edges: [PostEdge!]!
  nodes: [Post!]!
  pageInfo: PageInfo!
}

type PostEdge {
  cursor: String!
  node: Post!
}

type PageInfo {
  hasNextPage: Boolean!
  hasPreviousPage: Boolean!
  startCursor: String
  endCursor: String
}
"""


def build_registry() -> Tuple[PgRegistry, PgResource, PgResource, PgResource]:
    """Build the demo registry + resources with their relations wired."""
    registry = PgRegistry()
    authors = PgResource(
        "authors", "grafast_demo", "authors", ["id", "name"], registry=registry
    )
    posts = PgResource(
        "posts", "grafast_demo", "posts", ["id", "author_id", "title"], registry=registry
    )
    comments = PgResource(
        "comments",
        "grafast_demo",
        "comments",
        ["id", "post_id", "author_id", "body"],
        registry=registry,
    )

    authors.has_many("posts", target=posts, local_column="id", remote_column="author_id")
    posts.has_one("author", target=authors, local_column="author_id", remote_column="id")
    posts.has_many(
        "comments", target=comments, local_column="id", remote_column="post_id"
    )
    comments.has_one(
        "author", target=authors, local_column="author_id", remote_column="id"
    )
    comments.has_one("post", target=posts, local_column="post_id", remote_column="id")

    return registry, authors, posts, comments


def build_demo_schema(registry: Optional[PgRegistry] = None) -> GraphQLSchema:
    """Build the Postgres-backed demo GraphQL schema with plan resolvers.

    With no ``registry`` the demo builds its own (hand-declared) resources via
    :func:`build_registry`. When a ``registry`` is passed — e.g. one derived from
    SQLAlchemy models by :func:`grafast_py.pg.resources_from_models` — the three
    resources are fetched from it by name and the SAME plans are wired, so a
    model-derived registry drives the identical schema.
    """
    if registry is None:
        _registry, authors, posts, comments = build_registry()
    else:
        authors = registry["authors"]
        posts = registry["posts"]
        comments = registry["comments"]

    plans = {
        "Query": {
            "authors": _plan_all_rows(authors),
            "posts": _plan_all_rows(posts),
            "author": _plan_author_by_id(authors),
        },
        "Mutation": {
            "createPost": _plan_create_post(posts),
            "updatePost": _plan_update_post(posts),
            "deletePost": _plan_delete_post(posts),
        },
        "Author": {
            "id": _leaf("id"),
            "name": _leaf("name"),
            "posts": _plan_related_many(authors, "posts"),
            "postsConnection": _plan_posts_connection(authors),
        },
        "Post": {
            "id": _leaf("id"),
            "title": _leaf("title"),
            "author": _plan_related_single(posts, "author"),
            "comments": _plan_related_many(posts, "comments"),
        },
        "Comment": {
            "id": _leaf("id"),
            "body": _leaf("body"),
            "author": _plan_related_single(comments, "author"),
            "post": _plan_related_single(comments, "post"),
        },
        "PostConnection": {
            "totalCount": _leaf("totalCount"),
            "edges": _leaf("edges"),
            "nodes": _leaf("nodes"),
            "pageInfo": _leaf("pageInfo"),
        },
        "PostEdge": {
            "cursor": _leaf("cursor"),
            "node": _leaf("node"),
        },
        "PageInfo": {
            "hasNextPage": _leaf("hasNextPage"),
            "hasPreviousPage": _leaf("hasPreviousPage"),
            "startCursor": _leaf("startCursor"),
            "endCursor": _leaf("endCursor"),
        },
    }
    return make_grafast_schema(SDL, plans)


def _plan_all_rows(resource: PgResource):
    """A root-collection plan returning ALL rows of ``resource`` (one statement)."""

    def plan(parent_step, args, info):
        from grafast_py.pg.steps import PgSelectAllStep

        return PgSelectAllStep(
            resource, order_by=[resource.primary_key]
        ).for_parent(parent_step)

    return plan


def _leaf(key: str):
    """A plan resolver projecting one column/key out of the parent row dict."""

    def plan(parent_step, args, info):
        return access(parent_step, (key,))

    return plan


def _plan_author_by_id(authors: PgResource):
    def plan(parent_step, args, info):
        return authors.get_single(constant(args.get("id")), authors.primary_key)

    return plan


def _plan_related_single(resource: PgResource, relation_name: str):
    def plan(parent_step, args, info):
        return resource.related_single(parent_step, relation_name)

    return plan


def _plan_related_many(resource: PgResource, relation_name: str):
    def plan(parent_step, args, info):
        return resource.related_many(parent_step, relation_name)

    return plan


def _plan_create_post(posts: PgResource):
    """A mutation plan: INSERT one post from the input, RETURNING the row.

    The input's ``authorId`` maps to the ``author_id`` column; the returned step is the
    inserted row dict, so the Post sub-selection (id/title, and relations) reads it like
    a normal row.
    """

    def plan(parent_step, args, info):
        data = args.get("input") or {}
        return pg_insert_single(
            posts,
            {
                "id": data["id"],
                "author_id": data["authorId"],
                "title": data["title"],
            },
        )

    return plan


def _plan_update_post(posts: PgResource):
    """A mutation plan: UPDATE the PK-keyed post's supplied columns, RETURNING the row."""

    def plan(parent_step, args, info):
        data = args.get("input") or {}
        values = {}
        if "title" in data:
            values["title"] = data["title"]
        return pg_update_single(posts, args.get("id"), values)

    return plan


def _plan_delete_post(posts: PgResource):
    """A mutation plan: DELETE the PK-keyed post, RETURNING the deleted row."""

    def plan(parent_step, args, info):
        return pg_delete_single(posts, args.get("id"))

    return plan


def _plan_posts_connection(authors: PgResource):
    def plan(parent_step, args, info):
        relation = authors.get_relation("posts")
        key = access(parent_step, (relation.local_column,))
        # the count aggregate is issued ONLY when the selection set asks for totalCount.
        return connection(
            relation.target,
            key,
            relation.remote_column,
            order_by=[relation.target.primary_key],
            first=args.get("first"),
            after=args.get("after"),
            last=args.get("last"),
            before=args.get("before"),
            needs_total=connection_needs_total(info),
        )

    return plan


__all__ = ["SDL", "build_registry", "build_demo_schema"]
