"""Structured filter Conditions folded onto the batched WHERE.

A host expresses a GraphQL ``filter:`` arg as a :class:`Condition` tree
(``And``/``Or``/``Not``/``Compare``) instead of a hand-built SQLAlchemy expression. Each
node COMPILES to a Core boolean predicate, which folds onto the EXISTING
``where_predicates`` list via ``builder.where_tree(...)`` (or a step's ``.where_tree(...)``)
exactly like a hand-built ``.where()`` — so a compiled condition is JUST another uniform
WHERE peer. It changes no skeleton, adds no new dedup discriminator, and inherits the
value-discriminated dedup of ``predicate_key`` for free.

These tests assert:

- each AST node compiles to the expected Core predicate (the leaf ops, the n-ary AND/OR,
  NOT, the empty-AND/OR identities, value inlining);
- a ``where_tree`` filter folds onto the batched WHERE and filters rows UNIFORMLY across
  parents in ONE statement (the O(depth) statement count unchanged), and combines with a
  resource customizer and a hand-built ``.where()`` (all flat-AND peers);
- the fail-loud guards still fire — a Condition compiling to a reserved-bind / unbound /
  raw predicate is rejected by the SAME ``check_predicate`` path as a raw ``.where()``;
- DEDUP CORRECTNESS (no DB), proven BOTH directions: two filters differing only by a leaf
  VALUE (``status == 'published'`` vs ``== 'draft'``) get DIFFERENT keys and do NOT merge
  through ``dag.Plan.deduplicate()``; structurally-identical filters DO merge. A
  ``where_tree`` filter and the EQUIVALENT hand-built ``.where()`` dedup-merge (the
  compiled condition is indistinguishable downstream).

DB tests are marked ``pg`` (deselectable ``-m 'not pg'``); they reuse the ``widgets``
fixture and touch ONLY the ``grafast_demo`` schema, perturbing nothing else.
"""

import pytest
import pytest_asyncio
from sqlalchemy import bindparam, column
from sqlalchemy.dialects import postgresql

from grafast_py.core_steps import constant
from grafast_py.dag import Plan
from grafast_py.pg.conditions import (
    And,
    Compare,
    Condition,
    Or,
    compile_condition,
)
from grafast_py.pg.connection import PgConnectionStep
from grafast_py.pg.customize import predicate_key
from grafast_py.pg.engine import count_sql, dispose_engine, get_engine
from grafast_py.pg.executor import SQLAlchemyExecutor, pg_request_context
from grafast_py.pg.resource import PgRegistry, PgResource
from grafast_py.pg.steps import PgSelectAllStep, PgSelectStep
from examples.seed import setup_demo_schema, setup_widgets_table


def make_widgets(*, select_customizer=None) -> PgResource:
    """A fresh ``widgets`` resource (its own registry) for the filter tests."""
    registry = PgRegistry()
    return PgResource(
        "widgets",
        "grafast_demo",
        "widgets",
        ["id", "owner_id", "title", "status", "deleted_at"],
        registry=registry,
        select_customizer=select_customizer,
    )


def render(condition: Condition) -> str:
    """Compile a condition and render it value-inlined (the dedup-key view of it)."""
    return str(
        compile_condition(condition).compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )


@pytest_asyncio.fixture
async def seeded():
    """(Re)seed ``grafast_demo`` + the ``widgets`` fixture (fresh engine per test)."""
    await dispose_engine()
    await setup_demo_schema()
    await setup_widgets_table()
    yield
    await dispose_engine()


# ------------------------------------------------------------- AST -> Core compile


def test_compare_leaf_operators_compile_to_core_predicates():
    """Each leaf op compiles to the expected value-inlined Core predicate."""
    assert render(Compare("status", "eq", "published")) == "status = 'published'"
    assert render(Compare("status", "ne", "draft")) == "status != 'draft'"
    assert render(Compare("id", "lt", 5)) == "id < 5"
    assert render(Compare("id", "le", 5)) == "id <= 5"
    assert render(Compare("id", "gt", 5)) == "id > 5"
    assert render(Compare("id", "ge", 5)) == "id >= 5"
    assert render(Compare("id", "in", [1, 2, 3])) == "id IN (1, 2, 3)"
    # the pg dialect doubles a literal `%` (its paramstyle marker) when value-inlined; the
    # emitted predicate is an ordinary LIKE / ILIKE over the pattern.
    assert render(Compare("title", "like", "%x%")) == "title LIKE '%%x%%'"
    assert render(Compare("title", "ilike", "%x%")) == "title ILIKE '%%x%%'"


def test_is_null_truthiness_selects_is_null_or_is_not_null():
    """``is_null`` emits IS NULL when truthy, IS NOT NULL when falsy (no value inlined)."""
    assert render(Compare("deleted_at", "is_null", True)) == "deleted_at IS NULL"
    assert render(Compare("deleted_at", "is_null", False)) == "deleted_at IS NOT NULL"


def test_and_or_not_combinators_compile():
    """AND/OR combine children; NOT negates; nesting composes."""
    both = And([Compare("status", "eq", "published"), Compare("deleted_at", "is_null", True)])
    assert render(both) == "status = 'published' AND deleted_at IS NULL"

    either = Or([Compare("status", "eq", "published"), Compare("status", "eq", "draft")])
    assert render(either) == "status = 'published' OR status = 'draft'"

    nested = And([either, Compare("deleted_at", "is_null", True)])
    assert render(nested) == "(status = 'published' OR status = 'draft') AND deleted_at IS NULL"


def test_empty_and_or_are_the_identity_constants():
    """An empty AND is the always-true identity; an empty OR the always-false identity."""
    assert render(And([])) == "true"
    assert render(Or([])) == "false"


def test_unsupported_operator_fails_loud():
    """An unknown leaf operator is a declaration bug — rejected at construction."""
    with pytest.raises(ValueError, match="unsupported filter operator"):
        Compare("status", "matches", "%x%")


# ----------------------------------------------------- where_tree folds onto WHERE


@pytest.mark.pg
@pytest.mark.asyncio
async def test_where_tree_filters_uniformly_across_parents_one_statement(seeded):
    """A ``where_tree`` filter folds onto the batched WHERE — one statement, every parent.

    ``status == 'published' AND deleted_at IS NULL`` drops the draft and the soft-deleted
    widget for BOTH owners in the single ``= ANY(:keys)`` select.
    """
    widgets = make_widgets()
    engine = get_engine()
    with pg_request_context(SQLAlchemyExecutor(engine)):
        step = PgSelectStep(widgets, constant(None), "owner_id", order_by=["id"])
        step.builder().where_tree(
            And([Compare("status", "eq", "published"), Compare("deleted_at", "is_null", True)])
        )
        with count_sql(engine) as counter:
            out = await step.execute(2, [[1, 2]])

    assert counter.count == 1  # one batched statement; the filter is just a WHERE clause
    # owner 1's only published-and-live widget is 1 (2 is draft, 3 is deleted); owner 2's is 4.
    assert [r["id"] for r in out[0]] == [1]
    assert [r["id"] for r in out[1]] == [4]
    # the compiled filter is IN the batched WHERE, alongside the skeleton's ANY(:keys); the
    # value is a bound :param (re-parameterised at execute), not inlined.
    sql = str(step.build_query())
    assert "status = :" in sql
    assert "deleted_at IS NULL" in sql
    assert "ANY (:keys)" in sql


@pytest.mark.pg
@pytest.mark.asyncio
async def test_where_tree_or_filter_unions_rows(seeded):
    """An OR filter keeps rows matching EITHER branch (one statement)."""
    widgets = make_widgets()
    engine = get_engine()
    with pg_request_context(SQLAlchemyExecutor(engine)):
        step = PgSelectStep(widgets, constant(None), "owner_id", order_by=["id"])
        step.builder().where_tree(
            Or([Compare("id", "eq", 1), Compare("id", "eq", 2)])
        )
        out = await step.execute(2, [[1, 2]])

    # owner 1 keeps widgets 1 and 2; owner 2 keeps neither (ids 4,5,6).
    assert [r["id"] for r in out[0]] == [1, 2]
    assert [r["id"] for r in out[1]] == []


@pytest.mark.pg
@pytest.mark.asyncio
async def test_where_tree_combines_with_customizer_and_where(seeded):
    """A customizer, a ``where_tree`` filter, and a hand-built ``.where()`` all AND (flat)."""

    def only_owner_one(ctx):
        return [column("owner_id") == 1]

    widgets = make_widgets(select_customizer=only_owner_one)
    engine = get_engine()
    with pg_request_context(SQLAlchemyExecutor(engine), context={}):
        step = PgSelectAllStep(widgets, order_by=["id"])
        step.for_parent(constant(None))
        step.builder().where_tree(Compare("status", "eq", "published"))
        step.builder().where(column("deleted_at").is_(None))
        out = await step.execute(1, [[None]])

    # owner 1 AND published AND live -> just widget 1 (2 draft, 3 deleted).
    assert [r["id"] for r in out[0]] == [1]
    # three flat-AND peers: the customizer, the where_tree filter, the hand-built where.
    assert len(step.where_predicates) == 3


# ------------------------------------------------------------- fail-loud guards


def test_where_tree_reserved_bind_in_value_fails_loud():
    """A Condition whose value is a reserved-name bind is rejected by check_predicate."""
    widgets = make_widgets()
    step = PgSelectStep(widgets, constant(None), "owner_id", order_by=["id"])
    with pytest.raises(ValueError, match="reserved skeleton bind"):
        step.where_tree(Compare("id", "eq", bindparam("keys", value=1)))


def test_where_tree_unbound_bind_in_value_fails_loud():
    """A Condition whose value is an UNBOUND bind (no plan-time value) is rejected."""
    widgets = make_widgets()
    step = PgSelectStep(widgets, constant(None), "owner_id", order_by=["id"])
    with pytest.raises(ValueError, match="unbound bindparam"):
        step.where_tree(Compare("status", "eq", bindparam("p_status")))


# --------------------------------------------------------------- dedup correctness


def dedup_key(step):
    """The class + peer_key + dedup_params slice of the planner's structural key.

    A ``where_tree`` filter adds no dependency, so its discriminator lives entirely in
    ``peer_key`` / ``dedup_params`` (the customization signature), exactly as a ``.where()``.
    """
    return (type(step), step.peer_key, step.dedup_params())


def make_step(condition=None):
    """A widgets select, optionally with one ``where_tree(condition)``."""
    widgets = make_widgets()
    step = PgSelectStep(widgets, constant(None), "owner_id", order_by=["id"])
    if condition is not None:
        step.builder().where_tree(condition)
    return step


def test_filters_differing_by_value_differ_and_do_not_merge():
    """REGRESSION GATE: filters differing only by a leaf VALUE never dedup-merge.

    The compiled condition rides the value-included ``predicate_key`` path, so
    ``status == 'published'`` and ``status == 'draft'`` render DIFFERENT key strings — the
    steps are not peers and ``dag.Plan.deduplicate()`` keeps them as distinct survivors.
    """
    published = make_step(Compare("status", "eq", "published"))
    draft = make_step(Compare("status", "eq", "draft"))
    assert published.peer_key != draft.peer_key
    assert published.dedup_params() != draft.dedup_params()
    assert dedup_key(published) != dedup_key(draft)

    plan = Plan()
    plan.add_step(published)
    plan.add_step(draft)
    remap = plan.deduplicate()
    assert remap[published.id] is published
    assert remap[draft.id] is draft


def test_identical_filters_merge():
    """Two selects with the STRUCTURALLY-identical filter DO merge (same survivor)."""
    a = make_step(And([Compare("status", "eq", "published"), Compare("deleted_at", "is_null", True)]))
    b = make_step(And([Compare("status", "eq", "published"), Compare("deleted_at", "is_null", True)]))
    assert a.peer_key == b.peer_key
    assert a.dedup_params() == b.dedup_params()
    assert dedup_key(a) == dedup_key(b)

    plan = Plan()
    plan.add_step(a)
    plan.add_step(b)
    remap = plan.deduplicate()
    # the lower-id step wins; both references resolve to the same survivor.
    assert remap[a.id] is remap[b.id]


def test_where_tree_filter_merges_with_equivalent_hand_built_where():
    """A ``where_tree`` filter and the EQUIVALENT ``.where()`` are peers (indistinguishable).

    The compiled condition is just a Core predicate, so it produces the same content key as
    the hand-built equivalent — proving filters fold onto the SAME where path and inherit its
    dedup with no new discriminator.
    """
    widgets = make_widgets()
    via_tree = PgSelectStep(widgets, constant(None), "owner_id", order_by=["id"])
    via_tree.builder().where_tree(Compare("status", "eq", "published"))
    via_where = PgSelectStep(make_widgets(), constant(None), "owner_id", order_by=["id"])
    via_where.builder().where(column("status") == "published")

    assert via_tree.customization_signature() == via_where.customization_signature()
    assert via_tree.peer_key == via_where.peer_key
    assert via_tree.dedup_params() == via_where.dedup_params()
    assert dedup_key(via_tree) == dedup_key(via_where)


def test_like_ilike_values_discriminate_both_directions():
    """REGRESSION GATE: like/ilike filters inline their pattern, so they discriminate by VALUE.

    The new string operators ride the same value-included ``predicate_key`` path as every
    other op: two ``like`` filters with different patterns must NOT be peers; identical ones
    must be; and a ``like`` never merges with an ``ilike`` of the same pattern (different SQL).
    """
    foo = make_step(Compare("title", "like", "%foo%"))
    bar = make_step(Compare("title", "like", "%bar%"))
    same = make_step(Compare("title", "like", "%foo%"))
    ci = make_step(Compare("title", "ilike", "%foo%"))
    # different pattern => not peers; identical => peers.
    assert foo.dedup_params() != bar.dedup_params()
    assert foo.dedup_params() == same.dedup_params()
    # like vs ilike over the same pattern emit different SQL => not peers.
    assert foo.dedup_params() != ci.dedup_params()


def test_filter_vs_no_filter_do_not_dedup():
    """A filtered select never merges with an un-filtered one over the same skeleton."""
    plain = make_step()
    filtered = make_step(Compare("deleted_at", "is_null", True))
    assert plain.peer_key != filtered.peer_key
    assert dedup_key(plain) != dedup_key(filtered)


def test_filter_signature_is_the_compiled_predicate_key():
    """The customization signature is the compiled condition's value-included key."""
    cond = And([Compare("status", "eq", "published"), Compare("deleted_at", "is_null", True)])
    step = make_step(cond)
    assert step.customization_signature() == (predicate_key(compile_condition(cond)),)


def test_connection_where_tree_participates_in_key():
    """A ``where_tree`` filter folds into the connection step's dedup key too."""
    a = PgConnectionStep(make_widgets(), constant(None), "owner_id", order_by=["id"])
    a.builder().where_tree(Compare("deleted_at", "is_null", True))
    b = PgConnectionStep(make_widgets(), constant(None), "owner_id", order_by=["id"])
    assert a.peer_key != b.peer_key
    assert a.dedup_params() != b.dedup_params()
