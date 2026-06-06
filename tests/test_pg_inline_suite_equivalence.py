"""Suite-wide inlining oracle: the whole pg suite, run with inlining ON, is byte-identical.

Relation inlining is an OPPORTUNISTIC, EQUIVALENCE-PRESERVING optimization: for ANY
operation the inlined result must be BYTE-IDENTICAL to the batched ``= ANY($1)`` baseline.
The broadest possible oracle for that claim is the WHOLE pg suite — every one of those tests
already asserts EXACT result data — re-run with the flag forced on. Setting
``GRAFAST_INLINE_RELATIONS=1`` does exactly that: the autouse ``inline_relations_suite_toggle``
fixture in ``tests/conftest.py`` honours it by flipping the base
:class:`GrafastExecutionContext`'s config to ``inline_relations=True`` for the run.

This module is the dedicated, always-on companion to that suite-wide oracle. It does two
things:

1. META — it proves the suite-wide switch MECHANISM works: the autouse fixture is a no-op by
   default (so a plain ``uv run pytest`` and the conformance run are untouched), and flips the
   base config on under the env var. This is what lets a single suite run turn the entire
   suite into an inlining oracle.

2. DATA ORACLE — it runs a matrix of operations drawn from across the pg suite (including the
   model-DERIVED resource path and the deep O(depth) datasource query that the
   count-characterization tests pin to the batched baseline) TWICE, once batched and once
   inlined, asserting (a) ``result.data`` / ``result.errors`` are byte-identical and (b) the
   inlined run issues STRICTLY FEWER statements where a fold is expected — so the DATA of the
   count tests pinned to inlining-off is still proven equivalent under inlining here, even
   though those tests themselves stay on the batched path to keep their count assertion exact.

Marked ``pg`` (DB-backed) and touches ONLY the ``grafast_demo`` schema of
``grafast_py_test`` via the idempotent, drop-first seed fixtures.
"""

import pytest
import pytest_asyncio
from graphql import graphql

from grafast_py.config import DEFAULT_CONFIG, GrafastConfig
from grafast_py.context import GrafastExecutionContext
from grafast_py.pg.engine import count_sql, dispose_engine, get_engine
from grafast_py.pg.executor import SQLAlchemyExecutor, pg_request_context
from grafast_py.pg.from_sqlalchemy import resources_from_models
from examples.demo_schema import build_demo_schema
from examples.models import Author, Comment, Post
from examples.seed import setup_demo_schema

from .conftest import inline_relations_enabled

pytestmark = pytest.mark.pg


def context_class(inline: bool):
    """A throwaway context subclass carrying ``GrafastConfig(inline_relations=inline)``.

    A SUBCLASS (not a mutation of the base) so this module's explicit batched-vs-inlined
    comparison is independent of the suite-wide ``GRAFAST_INLINE_RELATIONS`` switch: the
    subclass attribute shadows whatever the autouse fixture did to the base class, so each
    case here always runs BOTH flag states regardless of how the suite was invoked.
    """

    class _Ctx(GrafastExecutionContext):
        grafast_config = GrafastConfig(inline_relations=inline)

    return _Ctx


async def run_counted(schema, query, *, inline: bool, variables=None):
    """Run ``query`` under the inline toggle; return ``(result, statement_count)``."""
    engine = get_engine()
    with count_sql(engine) as counter:
        with pg_request_context(SQLAlchemyExecutor(engine)):
            result = await graphql(
                schema,
                query,
                variable_values=variables,
                execution_context_class=context_class(inline),
            )
    return result, counter.count


async def assert_inlined_equivalent(schema, query, *, min_saving: int, variables=None):
    """Run ``query`` inlined vs batched; assert byte-identical data + a count drop.

    ``min_saving`` is the MINIMUM number of statements the inlined run must save: ``0`` for a
    SKIP shape (data identical, count equal — unsafe-but-skipped is always correct), a positive
    value proves at least that many folds fired. We assert ``>=`` rather than ``==`` here (the
    dedicated ``test_pg_inlining`` battery pins the exact savings) because this oracle's job is
    the DATA identity across a broad matrix; the count is the secondary "the fold fired" check.
    """
    batched, batched_count = await run_counted(
        schema, query, inline=False, variables=variables
    )
    inlined, inlined_count = await run_counted(
        schema, query, inline=True, variables=variables
    )
    # the inlining invariant: BYTE-IDENTICAL data and errors under both flag states.
    assert batched.errors == inlined.errors
    assert batched.data == inlined.data
    assert batched_count - inlined_count >= min_saving, (
        f"expected >= {min_saving} fewer statements inlined; "
        f"batched={batched_count} inlined={inlined_count}"
    )
    return batched, batched_count, inlined_count


# ============================================================ META: the suite-toggle mechanism

# the default config is OFF — inlining is off by default, so a plain run and the conformance
# run (which never sets the env var) are untouched. This is a property of the default config,
# independent of whatever the autouse suite-toggle did to the base class under an inline-on
# run, so it holds in both run modes.
def test_default_config_ships_inlining_off():
    """The default ``DEFAULT_CONFIG`` has inlining OFF — the off-by-default guarantee."""
    assert DEFAULT_CONFIG.inline_relations is False


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, False),  # unset -> off (so a plain run and conformance never inline)
        ("", False),
        ("0", False),
        ("false", False),
        ("False", False),
        ("1", True),  # any truthy value arms the suite-wide inline-on switch (off by default)
        ("true", True),
        ("True", True),
        ("yes", True),
    ],
)
def test_inline_switch_reads_env(monkeypatch, value, expected):
    """``inline_relations_enabled`` (the suite-toggle predicate) is driven purely by the env var.

    This is the gate the autouse ``inline_relations_suite_toggle`` reads: unset/falsey leaves
    the suite on the batched baseline (off by default), any truthy value flips the whole suite
    to inline-on. Testing the predicate directly keeps the META check independent of the ambient
    base-config the autouse fixture legitimately mutates under an inline-on run.
    """
    if value is None:
        monkeypatch.delenv("GRAFAST_INLINE_RELATIONS", raising=False)
    else:
        monkeypatch.setenv("GRAFAST_INLINE_RELATIONS", value)
    assert inline_relations_enabled() is expected


@pytest_asyncio.fixture
async def demo_schema():
    """(Re)seed ``grafast_demo`` and yield the hand-declared demo schema; dispose after."""
    await dispose_engine()
    await setup_demo_schema()
    schema = build_demo_schema()
    yield schema
    await dispose_engine()


@pytest_asyncio.fixture
async def model_demo_schema():
    """The demo schema built from the MODEL-DERIVED registry — the second construction path.

    Mirrors ``tests/test_from_sqlalchemy.py``: a registry derived from the SQLAlchemy models
    feeds ``build_demo_schema``, so inlining is exercised over model-derived resources, not
    only hand-declared ones. The count-characterization test for this path is pinned to
    inlining-off; this fixture lets us prove its DATA is byte-identical under inlining here.
    """
    await dispose_engine()
    await setup_demo_schema()
    registry = resources_from_models([Author, Post, Comment])
    schema = build_demo_schema(registry=registry)
    yield schema
    await dispose_engine()


# the broad oracle matrix: operations drawn from across the pg suite that the count tests
# pinned to inlining-off assert, plus the standard fold/skip shapes. ``min_saving`` is the
# minimum number of folds that must fire (0 = a pure-SKIP shape that must stay identical).
SUITE_MATRIX = [
    # from test_pg_datasource / test_pg_settings / test_bench_nplus1 — the deep O(depth) query
    # those tests pin to the batched baseline; here we prove its DATA is identical inlined.
    (
        "deep_o_depth",
        "{ authors { id name posts { id title author { id name } "
        "comments { id body author { name } } } } }",
        1,
    ),
    # from test_pg_datasource::test_deep_query_statement_count_independent_of_rows
    ("authors_posts", "{ authors { id posts { id } } }", 1),
    ("authors_posts_comments", "{ authors { id posts { id comments { id } } } }", 1),
    # from test_pg_datasource::test_hasone_relation_resolves_single_parent
    ("post_author_hasone", "{ posts { id author { id name } } }", 1),
    # null/empty shapes that must stay byte-identical
    ("missing_author", "{ author(id: 999) { id name } }", 0),
    ("root_only", "{ posts { id title } }", 0),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query,min_saving",
    [(q, s) for _name, q, s in SUITE_MATRIX],
    ids=[name for name, _q, _s in SUITE_MATRIX],
)
async def test_suite_queries_byte_identical_under_toggle(demo_schema, query, min_saving):
    """Every suite shape is byte-identical inlined vs batched (the broad oracle).

    These are the exact operations the inlining-off-pinned count tests assert. They pin
    OFF to keep their count assertion exact; this oracle proves their DATA is unchanged when
    inlining is on — closing the loop so nothing the suite asserts is left unproven inlined.
    """
    await assert_inlined_equivalent(demo_schema, query, min_saving=min_saving)


@pytest.mark.asyncio
async def test_model_derived_deep_query_byte_identical_under_toggle(model_demo_schema):
    """The model-DERIVED schema's deep query is byte-identical inlined vs batched.

    ``test_from_sqlalchemy``'s O(depth) parity gate is pinned to inlining-off (it asserts the
    exact batched count). Inlining operates on the same ``PgSelectStep`` / ``PgSelectSingleStep``
    regardless of how the resource was built, so this proves the model-derived construction
    path folds to byte-identical data — the only construction path the demo matrix above did
    not already cover.
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
    await assert_inlined_equivalent(model_demo_schema, query, min_saving=1)
