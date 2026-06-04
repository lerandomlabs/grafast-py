"""DB-backed tests for the single-row CRUD mutations.

Exercises ``createPost`` / ``updatePost`` / ``deletePost`` end-to-end through
:class:`GrafastExecutionContext` on the serial mutation seam: a mutation root field's
plan resolver returns an insert/update/delete step, the engine runs the mutation root
fields SERIALLY, and the RETURNING row decodes/projects like a read row.

These tests WRITE rows — but ONLY ever to ``grafast_demo`` tables in ``grafast_py_test``.
The ``demo_schema`` fixture re-seeds ``grafast_demo`` before AND after each test, so the
shared authors/posts/comments seed is always restored for the other pg suites.

Marked ``pg`` so a no-DB run can deselect them (``-m 'not pg'``).
"""

import pytest
import pytest_asyncio
from graphql import graphql

from grafast_py.context import GrafastExecutionContext
from examples.demo_schema import build_demo_schema
from grafast_py.pg.engine import dispose_engine, get_engine
from grafast_py.pg.executor import SQLAlchemyExecutor, pg_request_context
from grafast_py.pg.mutations import pg_insert_single
from grafast_py.pg.resource import PgColumn, PgRegistry, PgResource
from examples.seed import setup_demo_schema

pytestmark = pytest.mark.pg


async def run(schema, query, variables=None):
    """Run an operation through our engine with a request-scoped executor."""
    with pg_request_context(SQLAlchemyExecutor(get_engine())):
        return await graphql(
            schema,
            query,
            variable_values=variables,
            execution_context_class=GrafastExecutionContext,
        )


@pytest_asyncio.fixture
async def demo_schema():
    """Re-seed ``grafast_demo`` before AND after each mutation test.

    Function-scoped (fresh engine per test loop, as in the datasource suite). Reseeding
    on teardown restores the canonical authors/posts/comments seed for the other pg
    suites no matter what a mutation test wrote.
    """
    await dispose_engine()
    await setup_demo_schema()
    schema = build_demo_schema()
    yield schema
    await dispose_engine()
    await setup_demo_schema()
    await dispose_engine()


@pytest.mark.asyncio
async def test_create_post_inserts_and_returns_row(demo_schema):
    """createPost inserts a row; RETURNING surfaces it and a follow-up read confirms it."""
    result = await run(
        demo_schema,
        """
        mutation {
          createPost(input: {id: 1000, authorId: 1, title: "phase8 insert"}) {
            id
            title
            author { id name }
          }
        }
        """,
    )
    assert result.errors is None
    created = result.data["createPost"]
    # the returned dict's fields are accessible (AccessStep) and decoded like a read row,
    # and a relation off the returned row resolves (Post.author hasOne).
    assert created == {
        "id": 1000,
        "title": "phase8 insert",
        "author": {"id": 1, "name": "Ada Lovelace"},
    }

    # follow-up read proves the row actually exists (the write committed).
    read = await run(demo_schema, "{ posts { id title } }")
    assert read.errors is None
    titles = {p["id"]: p["title"] for p in read.data["posts"]}
    assert titles[1000] == "phase8 insert"


@pytest.mark.asyncio
async def test_update_post_changes_row_and_missing_pk_is_none(demo_schema):
    """updatePost changes a row (RETURNING reflects it); a missing PK returns None."""
    updated = await run(
        demo_schema,
        'mutation { updatePost(id: 1, input: {title: "renamed"}) { id title } }',
    )
    assert updated.errors is None
    assert updated.data["updatePost"] == {"id": 1, "title": "renamed"}

    # the change is persisted (visible to a fresh read).
    read = await run(demo_schema, "{ posts { id title } }")
    assert {p["id"]: p["title"] for p in read.data["posts"]}[1] == "renamed"

    missing = await run(
        demo_schema,
        'mutation { updatePost(id: 999999, input: {title: "nope"}) { id title } }',
    )
    assert missing.errors is None
    assert missing.data["updatePost"] is None


@pytest.mark.asyncio
async def test_delete_post_removes_row_and_missing_pk_is_none(demo_schema):
    """deletePost removes a row (RETURNING the deleted row); a missing PK returns None.

    Deletes a post we INSERT (the seed posts are referenced by comments via FK, so
    deleting one would violate the constraint — a separate concern from this v1 scope).
    """
    insert = await run(
        demo_schema,
        'mutation { createPost(input: {id: 1500, authorId: 1, title: "to delete"}) { id } }',
    )
    assert insert.errors is None

    deleted = await run(demo_schema, "mutation { deletePost(id: 1500) { id title } }")
    assert deleted.errors is None
    assert deleted.data["deletePost"] == {"id": 1500, "title": "to delete"}

    # a follow-up read confirms it is gone.
    read = await run(demo_schema, "{ posts { id } }")
    assert read.errors is None
    assert 1500 not in {p["id"] for p in read.data["posts"]}

    missing = await run(demo_schema, "mutation { deletePost(id: 999999) { id } }")
    assert missing.errors is None
    assert missing.data["deletePost"] is None


@pytest.mark.asyncio
async def test_mutations_run_serially(demo_schema):
    """Two mutation fields in one operation observe each other's effects, in order.

    createPost (a) then deletePost (b) of the SAME id: b can only return the row a just
    inserted if a's write committed before b ran — proving serial execution. After the
    op the row is gone (b deleted it).
    """
    result = await run(
        demo_schema,
        """
        mutation {
          a: createPost(input: {id: 3000, authorId: 2, title: "serial seam"}) { id title }
          b: deletePost(id: 3000) { id title }
        }
        """,
    )
    assert result.errors is None
    assert result.data["a"] == {"id": 3000, "title": "serial seam"}
    # b observed a's insert (else it would have returned None on a missing row).
    assert result.data["b"] == {"id": 3000, "title": "serial seam"}

    read = await run(demo_schema, "{ posts { id } }")
    assert 3000 not in {p["id"] for p in read.data["posts"]}


@pytest.mark.asyncio
async def test_mutation_value_is_param_bound_not_executed(demo_schema):
    """A title with SQL metacharacters is bound as a param and stored verbatim.

    Injection safety: a value like ``x'); DROP TABLE ...`` must be a bind value, never
    interpolated into SQL — so it is stored literally and the table is untouched.
    """
    payload = "x'); DROP TABLE grafast_demo.posts;--"
    result = await run(
        demo_schema,
        "mutation($t: String!) {"
        "  createPost(input: {id: 4000, authorId: 1, title: $t}) { id title }"
        "}",
        {"t": payload},
    )
    assert result.errors is None
    # the literal string round-trips verbatim (it was NOT executed as SQL).
    assert result.data["createPost"]["title"] == payload

    # the table still exists and the DROP did not run: the seed rows are all present.
    read = await run(demo_schema, "{ posts { id title } }")
    assert read.errors is None
    titles = {p["id"]: p["title"] for p in read.data["posts"]}
    assert titles[4000] == payload
    # the 9 seed posts plus the one we inserted.
    assert len(read.data["posts"]) == 10


@pytest.mark.asyncio
async def test_insert_not_null_no_default_omitted_fails_loud():
    """A NOT NULL no-default column omitted fails loud with a clear message.

    Not a raw DB not-null violation: the step surfaces a named, actionable error before
    building the statement. Uses a resource carrying the not_null/has_default metadata.
    """
    await dispose_engine()
    await setup_demo_schema()
    registry = PgRegistry()
    posts = PgResource(
        "posts",
        "grafast_demo",
        "posts",
        [
            PgColumn("id", not_null=True),
            PgColumn("author_id", not_null=True),
            PgColumn("title", not_null=True),
        ],
        registry=registry,
    )
    # omit the NOT NULL no-default `title`.
    step = pg_insert_single(posts, {"id": 5000, "author_id": 1})
    with pg_request_context(SQLAlchemyExecutor(get_engine())):
        with pytest.raises(ValueError, match=r"posts\.title is NOT NULL"):
            await step.execute(1, [])
    # nothing was written (the error fires before the statement runs).
    await dispose_engine()
    await setup_demo_schema()
    await dispose_engine()


@pytest.mark.asyncio
async def test_read_path_untouched_by_mutations(demo_schema):
    """The batched read path stays O(depth) after the mutation work — a sanity gate.

    Mutations are a SEPARATE serial path; a nested read still issues one batched
    statement per resource-layer (here authors + posts = 2), unchanged by Phase 8.
    """
    from grafast_py.pg.engine import count_sql

    with count_sql(get_engine()) as counter:
        result = await run(demo_schema, "{ authors { id posts { id } } }")
    assert result.errors is None
    assert counter.count == 2
