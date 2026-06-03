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
from grafast_py.pg.connection import connection
from grafast_py.pg.resource import PgRegistry, PgResource
from grafast_py.pg.steps import pg_select, pg_select_single

SDL = """
type Query {
  authors: [Author!]!
  author(id: Int!): Author
  posts: [Post!]!
}

type Author {
  id: Int!
  name: String!
  posts: [Post!]!
  postsConnection(first: Int, after: String): PostConnection!
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


def build_demo_schema() -> GraphQLSchema:
    """Build the Postgres-backed demo GraphQL schema with plan resolvers."""
    _registry, authors, posts, comments = build_registry()

    plans = {
        "Query": {
            "authors": _plan_all_rows(authors),
            "posts": _plan_all_rows(posts),
            "author": _plan_author_by_id(authors),
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


def _plan_posts_connection(authors: PgResource):
    def plan(parent_step, args, info):
        relation = authors.get_relation("posts")
        key = access(parent_step, (relation.local_column,))
        return connection(
            relation.target,
            key,
            relation.remote_column,
            order_by=[relation.target.primary_key],
            first=args.get("first"),
            after=args.get("after"),
        )

    return plan


__all__ = ["SDL", "build_registry", "build_demo_schema"]
