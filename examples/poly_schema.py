"""A Postgres-backed GraphQL schema exercising interface/union polymorphism.

Builds (via :func:`grafast_py.schema.make_grafast_schema`) a schema over a SINGLE
``grafast_demo.media`` table that carries a ``kind`` discriminator (``'image'`` /
``'video'``). Every row is a ``Media`` (interface) — an ``Image`` or a ``Video`` — and
a ``resolve_type`` bridge keyed on ``kind`` resolves each row's concrete type at
completion time. This composes with the existing completion-time abstract dispatch
(``completion.dispatch_abstract`` groups values by concrete type and plans each group's
sub-selection like a normal object field) — there is NO plan-time polymorphism bucket
system and NO core-engine change.

The plan for ``Query.media`` returns a plain :class:`PgSelectAllStep` producing row
dicts; the engine resolves each row's concrete type and batches each type's
sub-selection (including the nested ``tags`` pg relation) per concrete-type group.

Structure (so the integration step can extend this file without restructuring):

- ``SDL`` — the SDL constant; integration appends the union type + its query field.
- ``build_registry`` — the media + media_tags resources with relations wired.
- ``build_poly_schema`` — assembles plans + the ``type_resolvers`` bridge map and calls
  :func:`make_grafast_schema`; integration adds the union's plan + its bridge here.
- small ``_plan_*`` / ``_leaf`` helpers mirroring ``demo_schema``.
"""

from typing import Optional, Tuple

from graphql import GraphQLSchema
from sqlalchemy import TIMESTAMP

from grafast_py.core_steps import access, constant
from grafast_py.schema import (
    make_grafast_schema,
    resolve_type_from_discriminator,
    resolve_type_from_tag,
)
from grafast_py.pg.resource import PgRegistry, PgResource
from grafast_py.pg.steps import PgSelectAllStep
from grafast_py.pg.union import PgUnionMember, union_all_connection

SDL = """
type Query {
  media: [Media!]!
  mediaById(id: Int!): Media
  mediaByOwner(ownerId: Int!): [Media!]!
  search(first: Int, after: String): SearchConnection!
  activity(ownerId: Int!, first: Int): SearchConnection!
  authors: [Author!]!
}

interface Media {
  id: Int!
  title: String!
}

type Image implements Media {
  id: Int!
  title: String!
  width: Int!
  height: Int!
  tags: [Tag!]!
}

type Video implements Media {
  id: Int!
  title: String!
  durationSeconds: Int!
  tags: [Tag!]!
}

type Tag {
  id: Int!
  label: String!
}

union SearchResult = Article | Snippet

type Article {
  id: Int!
  ownerId: Int!
  headline: String!
  wordCount: Int!
  comments: [Comment!]!
}

type Snippet {
  id: Int!
  ownerId: Int!
  body: String!
  reactions: [Reaction!]!
}

type Comment {
  id: Int!
  body: String!
}

type Reaction {
  id: Int!
  emoji: String!
}

type Author {
  id: Int!
  activity(first: Int): SearchConnection!
}

type SearchConnection {
  totalCount: Int!
  nodes: [SearchResult!]!
  edges: [SearchEdge!]!
  pageInfo: PageInfo!
}

type SearchEdge {
  cursor: String!
  node: SearchResult!
}

type PageInfo {
  hasNextPage: Boolean!
  hasPreviousPage: Boolean!
  startCursor: String
  endCursor: String
}
"""

# the discriminator column (``kind``) -> concrete GraphQL type name. The bridge reads
# ``row['kind']`` and returns the mapped typename; graphql-core's completion validates it
# against Media's possible types, so a row whose kind is outside this map fails loud.
KIND_TO_TYPE = {"image": "Image", "video": "Video"}


def build_registry() -> Tuple[PgRegistry, PgResource, PgResource]:
    """Build the polymorphism registry: ``media`` + ``media_tags`` with relations wired.

    The ONE ``media`` table backs both concrete types (Image / Video); ``media_tags`` is
    its child so a concrete type's ``tags`` hasMany chains off the row resolved at
    completion time. ``media`` carries every column either concrete type needs (the
    type-specific ones are nullable in the table); each concrete sub-selection reads only
    its own columns.

    The cross-table union members (``articles`` / ``snippets``) are registered too, so the
    SAME registry backs both the single-table interface and the cross-table ``SearchResult``
    union (reached by the ``search`` / ``activity`` plans below).
    """
    registry = PgRegistry()
    media = PgResource(
        "media",
        "grafast_demo",
        "media",
        ["id", "owner_id", "kind", "title", "width", "height", "duration_seconds"],
        registry=registry,
    )
    media_tags = PgResource(
        "media_tags",
        "grafast_demo",
        "media_tags",
        ["id", "media_id", "label"],
        registry=registry,
    )
    media.has_many(
        "tags", target=media_tags, local_column="id", remote_column="media_id"
    )
    # the cross-table union members: separate tables, each a concrete type of SearchResult.
    # ``created`` is timestamptz — a non-native keyset order column — so each member declares
    # its SQL type for the keyset CAST (the text-origin cursor value is cast back to
    # timestamptz; see grafast_py.pg.cursor), exactly as the keyset fixture does.
    created_type = {"created": TIMESTAMP(timezone=True)}
    articles = PgResource(
        "articles",
        "grafast_demo",
        "articles",
        ["id", "owner_id", "created", "headline", "word_count"],
        registry=registry,
        column_types=created_type,
    )
    snippets = PgResource(
        "snippets",
        "grafast_demo",
        "snippets",
        ["id", "owner_id", "created", "body"],
        registry=registry,
        column_types=created_type,
    )
    # child tables of the union members: a nested pg relation under a CONCRETE union member
    # chains off the row resolved at completion time, so it batches per concrete-type group
    # (one statement across every Article for ``comments``, one across every Snippet for
    # ``reactions``) — the cross-table analogue of the single-table ``Image.tags`` relation.
    article_comments = PgResource(
        "article_comments",
        "grafast_demo",
        "article_comments",
        ["id", "article_id", "body"],
        registry=registry,
    )
    snippet_reactions = PgResource(
        "snippet_reactions",
        "grafast_demo",
        "snippet_reactions",
        ["id", "snippet_id", "emoji"],
        registry=registry,
    )
    articles.has_many(
        "comments", target=article_comments, local_column="id", remote_column="article_id"
    )
    snippets.has_many(
        "reactions", target=snippet_reactions, local_column="id", remote_column="snippet_id"
    )
    # the authors root table (seeded by setup_demo_schema) backs the ``Author`` type whose
    # per-parent ``activity`` union is keyed on the author's id matching each member's owner_id.
    PgResource(
        "authors",
        "grafast_demo",
        "authors",
        ["id", "name"],
        registry=registry,
    )
    return registry, media, media_tags


def build_poly_schema(registry: Optional[PgRegistry] = None) -> GraphQLSchema:
    """Build the polymorphism demo schema with plan resolvers + the resolve_type bridge.

    With no ``registry`` the demo builds its own resources via :func:`build_registry`;
    passing one (e.g. a model-derived registry) fetches ``media`` / ``media_tags`` by name
    and wires the SAME plans. The ``Media`` interface's concrete type is resolved per row
    by a discriminator bridge over the ``kind`` column.
    """
    if registry is None:
        registry, media, _media_tags = build_registry()
    else:
        # media_tags / the union members' child tables are reached via relations, so only the
        # roots are bound by name here.
        media = registry["media"]
    articles = registry["articles"]
    snippets = registry["snippets"]
    authors = registry["authors"]

    plans = {
        "Query": {
            "media": _plan_all_media(media),
            "mediaById": _plan_media_by_id(media),
            "mediaByOwner": _plan_media_by_owner(media),
            "search": _plan_search(articles, snippets),
            "activity": _plan_activity(articles, snippets),
            "authors": _plan_all_authors(authors),
        },
        "Image": {
            "id": _leaf("id"),
            "title": _leaf("title"),
            "width": _leaf("width"),
            "height": _leaf("height"),
            "tags": _plan_related_many(media, "tags"),
        },
        "Video": {
            "id": _leaf("id"),
            "title": _leaf("title"),
            "durationSeconds": _leaf("duration_seconds"),
            "tags": _plan_related_many(media, "tags"),
        },
        "Tag": {
            "id": _leaf("id"),
            "label": _leaf("label"),
        },
        # the cross-table union's connection wrapper + its two concrete members. Each member's
        # leaf reads its own (NULL-padded-in-the-union) columns off the tagged row dict.
        "SearchConnection": {
            "totalCount": _leaf("totalCount"),
            "nodes": _leaf("nodes"),
            "edges": _leaf("edges"),
            "pageInfo": _leaf("pageInfo"),
        },
        "SearchEdge": {
            "cursor": _leaf("cursor"),
            "node": _leaf("node"),
        },
        "PageInfo": {
            "hasNextPage": _leaf("hasNextPage"),
            "hasPreviousPage": _leaf("hasPreviousPage"),
            "startCursor": _leaf("startCursor"),
            "endCursor": _leaf("endCursor"),
        },
        "Article": {
            "id": _leaf("id"),
            "ownerId": _leaf("owner_id"),
            "headline": _leaf("headline"),
            "wordCount": _leaf("word_count"),
            # a nested hasMany OFF the Article concrete type — batches per type-group.
            "comments": _plan_related_many(articles, "comments"),
        },
        "Snippet": {
            "id": _leaf("id"),
            "ownerId": _leaf("owner_id"),
            "body": _leaf("body"),
            "reactions": _plan_related_many(snippets, "reactions"),
        },
        "Comment": {
            "id": _leaf("id"),
            "body": _leaf("body"),
        },
        "Reaction": {
            "id": _leaf("id"),
            "emoji": _leaf("emoji"),
        },
        # the Author root row whose per-parent ``activity`` union is keyed on the author id;
        # nesting the union under a list of authors proves it stays ONE batched statement
        # across every parent (O(depth), not per-parent).
        "Author": {
            "id": _leaf("id"),
            "activity": _plan_author_activity(articles, snippets),
        },
    }
    # the bridges wired onto each abstract type's resolve_type; completion-time dispatch
    # reads them per value to group rows by concrete type. Keyed by abstract type name. The
    # single-table interface uses a discriminator column; the cross-table union reads the
    # ``__typename`` tag the UNION ALL legs project (resolve_type_from_tag).
    type_resolvers = {
        "Media": resolve_type_from_discriminator("kind", KIND_TO_TYPE),
        "SearchResult": resolve_type_from_tag("__typename"),
    }
    return make_grafast_schema(SDL, plans, type_resolvers)


def _plan_all_media(media: PgResource):
    """A root-collection plan returning ALL media rows (one statement), id-ordered.

    The returned step is a list of row dicts each carrying ``kind``; the engine resolves
    each row's concrete type at completion and batches per type group.
    """

    def plan(parent_step, args, info):
        return PgSelectAllStep(media, order_by=[media.primary_key]).for_parent(parent_step)

    return plan


def _plan_media_by_id(media: PgResource):
    """A root plan returning ONE media row by primary key, or null when absent.

    The single (``get_single``) shape behind a NULLABLE ``Media`` interface field: the row
    (or ``None``) feeds the SAME completion-time dispatch — a present row resolves its
    concrete type via the ``kind`` bridge, a missing id completes to ``null`` without ever
    reaching the bridge (the engine short-circuits a ``None`` abstract value).
    """

    def plan(parent_step, args, info):
        return media.get_single(constant(args.get("id")), media.primary_key)

    return plan


def _plan_media_by_owner(media: PgResource):
    """A root plan returning one owner's media rows (a batched key-matched select)."""

    def plan(parent_step, args, info):
        return media.find(
            constant(args.get("ownerId")), "owner_id", order_by=[media.primary_key]
        )

    return plan


def _leaf(key: str):
    """A plan resolver projecting one column/key out of the parent row dict."""

    def plan(parent_step, args, info):
        return access(parent_step, (key,))

    return plan


def _plan_related_many(resource: PgResource, relation_name: str):
    def plan(parent_step, args, info):
        return resource.related_many(parent_step, relation_name)

    return plan


# the shared projection + keyset order for the cross-table SearchResult union: every member
# carries id/owner_id/created, ordered by created (the keyset appends the PK tie-break).
_SEARCH_SHARED = ["id", "owner_id", "created"]
_SEARCH_ORDER = ["created"]


def _plan_search(articles: PgResource, snippets: PgResource):
    """A ROOT cross-table union connection over ``articles`` + ``snippets`` (no key match).

    One ``UNION ALL`` statement tags each row with its concrete type (``Article`` / ``Snippet``)
    and keyset-pages the merged, ``created``-ordered result; completion-time dispatch resolves
    each row via the ``__typename`` tag bridge. ``totalCount`` is gated off the selection set.
    """

    def plan(parent_step, args, info):
        return union_all_connection(
            [
                PgUnionMember(articles, "Article"),
                PgUnionMember(snippets, "Snippet"),
            ],
            _SEARCH_SHARED,
            _SEARCH_ORDER,
            info,
            first=args.get("first"),
            after=args.get("after"),
        )

    return plan


def _plan_activity(articles: PgResource, snippets: PgResource):
    """A PER-PARENT cross-table union keyed on ``owner_id`` (each member matches the owner).

    The per-parent shape: each member matches ``owner_id = ANY(:keys)`` against the ``ownerId``
    arg, partitioned by owner so the page slices per parent in the one batched statement. Feeds
    the SAME completion-time dispatch as the root union.
    """

    def plan(parent_step, args, info):
        return union_all_connection(
            [
                PgUnionMember(articles, "Article", match="owner_id"),
                PgUnionMember(snippets, "Snippet", match="owner_id"),
            ],
            _SEARCH_SHARED,
            _SEARCH_ORDER,
            info,
            key_step=constant(args.get("ownerId")),
            first=args.get("first"),
        )

    return plan


def _plan_all_authors(authors: PgResource):
    """A root-collection plan returning ALL authors (one statement), id-ordered.

    Each row feeds the per-parent ``Author.activity`` union below as a parent, so a list of
    authors fans out to ONE batched union statement across every author (not one per author).
    """

    def plan(parent_step, args, info):
        return PgSelectAllStep(authors, order_by=[authors.primary_key]).for_parent(
            parent_step
        )

    return plan


def _plan_author_activity(articles: PgResource, snippets: PgResource):
    """A PER-PARENT cross-table union NESTED under an ``Author`` row, keyed on the author id.

    The per-parent union as a RELATION field: the key step accesses the author's ``id`` off the
    parent row (matching each member's ``owner_id``), so every author's activity slices
    independently in the ONE batched ``UNION ALL`` statement across the whole author bucket —
    O(depth), not per-parent. Feeds the SAME completion-time dispatch as the root union.
    """

    def plan(parent_step, args, info):
        return union_all_connection(
            [
                PgUnionMember(articles, "Article", match="owner_id"),
                PgUnionMember(snippets, "Snippet", match="owner_id"),
            ],
            _SEARCH_SHARED,
            _SEARCH_ORDER,
            info,
            key_step=access(parent_step, ("id",)),
            first=args.get("first"),
        )

    return plan


__all__ = ["SDL", "KIND_TO_TYPE", "build_registry", "build_poly_schema"]
