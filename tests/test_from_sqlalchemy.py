"""Tests for deriving :class:`PgResource` descriptors from SQLAlchemy ORM models.

Two layers:

- NO-DB: pure introspection. ``resources_from_models`` over the demo models
  (:mod:`examples.models`) must reproduce, descriptor-for-descriptor, the
  hand-declared registry from :mod:`examples.demo_schema`; plus the edge cases the
  derivation must handle loudly (composite PK, many-to-many, one-to-one, out-of-batch
  relation target).
- DB (``pg`` marker): the end-to-end parity gate. Build the demo GraphQL schema from
  the MODEL-derived registry and assert it returns the SAME data AND issues the SAME
  number of SQL statements (the O(depth) batching profile) as the hand-declared path
  in ``tests/test_pg_datasource.py``.
"""

import pytest
import pytest_asyncio
from graphql import graphql
from sqlalchemy import ForeignKey, Integer, Table, Column, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from grafast_py.context import GrafastExecutionContext
from grafast_py.pg.engine import count_sql, dispose_engine, get_engine
from grafast_py.pg.executor import SQLAlchemyExecutor, pg_request_context
from grafast_py.pg.from_sqlalchemy import resource_from_model, resources_from_models
from examples.demo_schema import build_demo_schema, build_registry
from examples.models import Author, Comment, Post
from examples.seed import setup_demo_schema


# --------------------------------------------------------------------- edge-case
# Local SQLAlchemy fixtures for the introspection edge cases. These are test
# fixtures (not shipped library code), so importing/declaring SQLAlchemy is fine.
# Distinct tablenames keep their metadata clear of the demo models.


class EdgeBase(DeclarativeBase):
    pass


class CompositePK(EdgeBase):
    """A model whose primary key spans two columns (no single-column PK)."""

    __tablename__ = "composite_pk"

    org_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    item_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)


tag_assoc = Table(
    "tag_assoc",
    EdgeBase.metadata,
    Column("article_id", ForeignKey("article.id"), primary_key=True),
    Column("tag_id", ForeignKey("tag.id"), primary_key=True),
)


class Article(EdgeBase):
    """One half of a many-to-many (via the ``tag_assoc`` association table)."""

    __tablename__ = "article"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tags: Mapped[list["Tag"]] = relationship(
        "Tag", secondary=tag_assoc, back_populates="articles"
    )


class Tag(EdgeBase):
    __tablename__ = "tag"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    articles: Mapped[list["Article"]] = relationship(
        "Article", secondary=tag_assoc, back_populates="tags"
    )


class User(EdgeBase):
    """Owning side of a one-to-one (``uselist=False`` over a ONETOMANY direction)."""

    __tablename__ = "user_acct"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    profile: Mapped["Profile"] = relationship(
        "Profile", back_populates="user", uselist=False
    )


class Profile(EdgeBase):
    __tablename__ = "profile"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("user_acct.id"), nullable=False
    )
    user: Mapped["User"] = relationship("User", back_populates="profile")


# ------------------------------------------------------------------- NO-DB tests


def test_model_derived_registry_matches_hand_declared():
    """The model-derived resources equal the hand-declared ones, descriptor-by-descriptor.

    Compares schema / table / ORDERED columns / primary_key, and for each relation its
    kind, local_column, remote_column and target name — for all three demo resources.
    """
    derived = resources_from_models([Author, Post, Comment])
    hand = build_registry()[0]

    for name in ("authors", "posts", "comments"):
        d = derived[name]
        h = hand[name]
        assert d.schema == h.schema
        assert d.table == h.table
        assert d.columns == h.columns  # ordered list equality
        assert d.primary_key == h.primary_key
        assert set(d.relations) == set(h.relations)
        for rel_name, d_rel in d.relations.items():
            h_rel = h.relations[rel_name]
            assert d_rel.kind == h_rel.kind
            assert d_rel.local_column == h_rel.local_column
            assert d_rel.remote_column == h_rel.remote_column
            assert d_rel.target.name == h_rel.target.name


def test_non_mapped_object_raises():
    """A plain (non-mapped) class fails loudly rather than yielding a bad resource."""
    with pytest.raises(ValueError, match="not a mapped declarative model"):
        resource_from_model(object)


def test_composite_pk_raises_without_override():
    """A composite-PK model fails loudly; a ``primary_key`` override makes it work."""
    with pytest.raises(ValueError, match="composite"):
        resource_from_model(CompositePK)

    resource = resource_from_model(CompositePK, primary_key="org_id")
    assert resource.primary_key == "org_id"
    assert resource.columns == ["org_id", "item_id", "name"]


def test_many_to_many_relation_skipped_by_default_and_strict_raises():
    """A many-to-many relation is dropped by default; strict turns the skip into a raise."""
    registry = resources_from_models([Article, Tag])
    assert "tags" not in registry["article"].relations
    assert "articles" not in registry["tag"].relations

    with pytest.raises(ValueError, match="many-to-many"):
        resources_from_models([Article, Tag], strict=True)


def test_one_to_one_relation_is_has_one():
    """``uselist=False`` over a ONETOMANY direction derives a ``has_one`` relation.

    The join columns stay oriented relative to the relationship's own parent: on
    ``User.profile`` local is the owner key ``id`` and remote is the FK ``user_id``.
    """
    registry = resources_from_models([User, Profile])
    rel = registry["user_acct"].relations["profile"]
    assert rel.kind == "has_one"
    assert rel.local_column == "id"
    assert rel.remote_column == "user_id"
    assert rel.target.name == "profile"

    # the mirror MANYTOONE side (also uselist=False) derives has_one with the join
    # columns oriented relative to ITS parent: local is the FK, remote the owner key.
    back = registry["profile"].relations["user"]
    assert back.kind == "has_one"
    assert back.local_column == "user_id"
    assert back.remote_column == "id"
    assert back.target.name == "user_acct"


def test_out_of_batch_relation_target_skipped_and_strict_raises():
    """A relation whose target model is not in the batch is skipped (strict raises)."""
    registry = resources_from_models([Post])
    assert registry["posts"].relations == {}

    with pytest.raises(ValueError, match="not in the batch"):
        resources_from_models([Post], strict=True)


# ------------------------------------------------------------- DB end-to-end parity
# (the DB tests below carry their own @pytest.mark.pg; the module mixes no-DB and DB
# tests, so there is deliberately no module-level `pytestmark`.)


async def run(schema, query, variables=None):
    """Run a query through our engine (not the stock executor).

    Binds a :class:`SQLAlchemyExecutor` over the convenience engine for the request so
    the pg steps execute their statements via the request-scoped executor.
    """
    with pg_request_context(SQLAlchemyExecutor(get_engine())):
        return await graphql(
            schema,
            query,
            variable_values=variables,
            execution_context_class=GrafastExecutionContext,
        )


@pytest_asyncio.fixture
async def model_demo_schema():
    """Build the demo schema from the MODEL-derived registry after reseeding.

    Mirrors ``tests/test_pg_datasource.py``'s ``demo_schema`` fixture (function-scoped,
    fresh engine per test) but feeds ``build_demo_schema`` a registry derived from the
    SQLAlchemy models — the path under test.
    """
    await dispose_engine()
    await setup_demo_schema()
    registry = resources_from_models([Author, Post, Comment])
    schema = build_demo_schema(registry=registry)
    yield schema
    await dispose_engine()


@pytest.mark.pg
@pytest.mark.asyncio
async def test_model_derived_schema_nested_query_matches_o_depth(model_demo_schema):
    """The deep nested query returns the SAME data and SAME statement count (5).

    This is the end-to-end parity gate: a schema built from model-derived resources
    executes identically (same data, same O(depth) batching) to the hand-declared one.
    """
    query = """
    {
      authors {
        id
        name
        posts {
          id
          title
          author { id name }
          comments { id body author { name } }
        }
      }
    }
    """
    with count_sql(get_engine()) as counter:
        result = await run(model_demo_schema, query)

    assert result.errors is None
    assert counter.count == 5

    authors = result.data["authors"]
    assert [a["name"] for a in authors] == [
        "Ada Lovelace",
        "Alan Turing",
        "Grace Hopper",
    ]
    assert [len(a["posts"]) for a in authors] == [2, 3, 4]
    first_post = authors[0]["posts"][0]
    assert first_post["author"]["name"] == "Ada Lovelace"
    assert len(first_post["comments"]) == 2
    assert first_post["comments"][0]["author"]["name"]


@pytest.mark.pg
@pytest.mark.asyncio
async def test_model_derived_schema_connection_matches(model_demo_schema):
    """The connection query returns the SAME data and SAME statement count (3)."""
    query = """
    {
      authors {
        name
        postsConnection(first: 2) {
          totalCount
          edges { cursor node { id } }
          pageInfo { hasNextPage hasPreviousPage startCursor endCursor }
        }
      }
    }
    """
    with count_sql(get_engine()) as counter:
        result = await run(model_demo_schema, query)
    assert result.errors is None
    # authors (1) + connection page + separate totalCount aggregate = 3 (O(depth)).
    assert counter.count == 3

    grace = result.data["authors"][2]
    conn = grace["postsConnection"]
    assert conn["totalCount"] == 4
    assert len(conn["edges"]) == 2
    assert conn["pageInfo"]["hasNextPage"] is True
    assert conn["pageInfo"]["hasPreviousPage"] is False
    assert conn["pageInfo"]["endCursor"] == conn["edges"][-1]["cursor"]

    ada = result.data["authors"][0]
    assert ada["postsConnection"]["totalCount"] == 2
    assert ada["postsConnection"]["pageInfo"]["hasNextPage"] is False
