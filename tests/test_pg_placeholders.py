"""Value-agnostic WHERE placeholders: dedup-by-source and cross-request re-use.

A host inlines a plan-time LITERAL arg value (``column("status") == args["status"]``) and
gets a VALUE-discriminated dedup key (two values -> two keys -> never merge). When the value
came from a GraphQL ``$variable``, the host instead wraps it as ``pg_placeholder(
field_args.source("status"), args["status"])`` — a source-tagged bindparam that carries the
request's value (so it still rides ``compiled.params`` at execute) but is dedup-keyed by its
STABLE source tag, never by its runtime value.

This module gates the CRUX of value-agnostic placeholders: the four placeholder dedup
behaviours, proven both directions through ``dag.Plan.deduplicate()`` with NO DB:

  1. two steps over the SAME placeholder source MERGE;
  2. two steps over DIFFERENT placeholder sources do NOT merge;
  3. a placeholder step and a literal step of a COINCIDENTALLY equal value do NOT merge
     (``$1`` vs an inlined ``'published'`` are different SQL);
  4. an existing literal-only step keeps its value-included key UNCHANGED (no regression).

Plus the NO-REGRESSION oracle (same query via the old literal path vs the new placeholder
path -> byte-identical result AND statement count) and a cross-request data-correctness test
(the same value-agnostic statement run with TWO different variable values returns each its
OWN correct rows — the anti-corruption gate the plan cache relies on).

DB tests are marked ``pg`` (deselectable ``-m 'not pg'``); they touch ONLY the
``grafast_demo`` ``widgets`` fixture and alter nothing else.
"""

import pytest
import pytest_asyncio
from sqlalchemy import and_, column

from grafast_py.core_steps import constant
from grafast_py.dag import Plan
from grafast_py.pg.connection import PgConnectionStep
from grafast_py.pg.customize import placeholder_binds_in, predicate_key
from grafast_py.pg.engine import count_sql, dispose_engine, get_engine
from grafast_py.pg.executor import SQLAlchemyExecutor, pg_request_context
from grafast_py.pg.placeholders import (
    PLACEHOLDER_SOURCE_ATTR,
    pg_placeholder,
    placeholder_source,
)
from grafast_py.pg.resource import PgRegistry, PgResource
from grafast_py.pg.steps import PgSelectStep
from examples.seed import setup_demo_schema, setup_widgets_table


def make_widgets() -> PgResource:
    """A fresh ``widgets`` resource (its own registry) for the placeholder tests."""
    registry = PgRegistry()
    return PgResource(
        "widgets",
        "grafast_demo",
        "widgets",
        ["id", "owner_id", "title", "status", "deleted_at"],
        registry=registry,
    )


@pytest_asyncio.fixture
async def seeded():
    """(Re)seed ``grafast_demo`` + the ``widgets`` fixture (fresh engine per test)."""
    await dispose_engine()
    await setup_demo_schema()
    await setup_widgets_table()
    yield
    await dispose_engine()


def make_step(predicate=None) -> PgSelectStep:
    """A widgets select, optionally with one per-plan ``.where(predicate)``."""
    step = PgSelectStep(make_widgets(), constant(None), "owner_id", order_by=["id"])
    if predicate is not None:
        step.builder().where(predicate)
    return step


def dedup_key(step):
    """The class + peer_key + dedup_params slice of the planner's structural key.

    A ``.where()`` adds no dependency, so the placeholder discriminator lives entirely in
    ``peer_key`` / ``dedup_params`` (the dependency survivor ids are identical here).
    """
    return (type(step), step.peer_key, step.dedup_params())


# --------------------------------------------------------- pg_placeholder construction


def test_pg_placeholder_carries_value_and_source_tag():
    """A placeholder is a bound bindparam stamped with its stable source tag."""
    bind = pg_placeholder("var:status", "published")
    # it carries THIS request's value, so it rides compiled.params at execute (nothing new
    # is needed at execute time).
    assert bind.value == "published"
    assert bind.required is False
    # and the stable source tag, on the side attribute add_where / predicate_key read.
    assert placeholder_source(bind) == "var:status"
    assert getattr(bind, PLACEHOLDER_SOURCE_ATTR) == "var:status"


def test_pg_placeholder_names_are_unique_per_call():
    """Each placeholder gets a UNIQUE bind NAME so two never collide in one statement."""
    a = pg_placeholder("var:status", "published")
    b = pg_placeholder("var:status", "draft")
    assert a.key != b.key  # unique names...
    # ...but the SAME source tag (request-stable), so they key identically (see crux #1).
    assert placeholder_source(a) == placeholder_source(b) == "var:status"


def test_placeholder_source_is_none_for_a_plain_literal_bind():
    """A plain SQLAlchemy literal bind carries no source tag (it stays on the literal path)."""
    literal = column("status") == "published"
    binds = placeholder_binds_in(literal)
    assert binds == {}  # nothing tagged -> the literal key path


# --------------------------------------------------------- add_where registers placeholders


def test_add_where_registers_placeholder_bind_in_registry():
    """``add_where`` populates the per-step ``placeholder_binds`` registry from a placeholder."""
    ph = pg_placeholder("var:status", "published")
    step = make_step(column("status") == ph)
    assert step.placeholder_binds == {ph.key: "var:status"}


def test_add_where_literal_leaves_registry_empty():
    """A literal ``.where()`` adds NOTHING to the registry (the byte-identical literal path)."""
    step = make_step(column("status") == "published")
    assert step.placeholder_binds == {}


# ---------------------------------------------------- predicate_key: the four crux behaviours


def test_predicate_key_same_source_is_identical():
    """CRUX 1: two placeholder predicates over the SAME source produce the IDENTICAL key.

    Even with DIFFERENT runtime values and different unique bind names — the key is
    value-agnostic and source-tagged, so it merges (a cache hit across requests).
    """
    p1 = column("status") == pg_placeholder("var:status", "published")
    p2 = column("status") == pg_placeholder("var:status", "draft")
    k1 = predicate_key(p1, placeholder_binds_in(p1))
    k2 = predicate_key(p2, placeholder_binds_in(p2))
    assert k1 == k2
    # and it carries the source tag, NOT the value.
    assert "var:status" in k1
    assert "published" not in k1 and "draft" not in k1


def test_predicate_key_different_source_differs():
    """CRUX 2: two placeholder predicates over DIFFERENT sources produce DIFFERENT keys."""
    p1 = column("status") == pg_placeholder("var:status", "published")
    p2 = column("status") == pg_placeholder("var:other", "published")
    k1 = predicate_key(p1, placeholder_binds_in(p1))
    k2 = predicate_key(p2, placeholder_binds_in(p2))
    assert k1 != k2


def test_predicate_key_placeholder_never_equals_literal_of_same_value():
    """CRUX 3: a placeholder key never equals a literal key of a coincidentally equal value.

    ``status = %(ph)s | ph=[var:status]`` (placeholder) vs ``status = 'published'`` (inlined
    literal) are different SQL, so the steps must not merge — re-binding the placeholder would
    otherwise serve a literal-pinned plan a runtime value it never planned for.
    """
    ph_pred = column("status") == pg_placeholder("var:status", "published")
    lit_pred = column("status") == "published"
    ph_key = predicate_key(ph_pred, placeholder_binds_in(ph_pred))
    lit_key = predicate_key(lit_pred, placeholder_binds_in(lit_pred) or None)
    assert ph_key != lit_key


def test_predicate_key_source_to_column_mapping_is_positional():
    """CRUX (subtle): swapping which source binds which column yields a DIFFERENT key.

    ``a == $X AND b == $Y`` and ``a == $Y AND b == $X`` are different statements once re-bound
    per request, so their keys MUST differ — a flat sorted-source suffix alone would collide
    them (same SQL shape, same source SET) and wrongly merge two semantically distinct plans.
    The source is pinned POSITIONALLY in the value-agnostic SQL to keep them apart.
    """
    p1 = and_(
        column("a") == pg_placeholder("var:x", 1),
        column("b") == pg_placeholder("var:y", 2),
    )
    p2 = and_(
        column("a") == pg_placeholder("var:y", 1),
        column("b") == pg_placeholder("var:x", 2),
    )
    k1 = predicate_key(p1, placeholder_binds_in(p1))
    k2 = predicate_key(p2, placeholder_binds_in(p2))
    assert k1 != k2
    # the SAME mapping still converges (only the unique bind names / values differ).
    p1_again = and_(
        column("a") == pg_placeholder("var:x", 7),
        column("b") == pg_placeholder("var:y", 8),
    )
    assert predicate_key(p1_again, placeholder_binds_in(p1_again)) == k1


def test_predicate_key_literal_path_unchanged():
    """CRUX 4: a literal-only predicate keeps its value-included ``literal_binds`` key.

    Passing the (empty) placeholder map of a literal predicate is byte-identical to passing
    none — every existing merge/count is preserved.
    """
    pred = column("status") == "published"
    with_empty_map = predicate_key(pred, placeholder_binds_in(pred) or None)
    no_map = predicate_key(pred)
    assert with_empty_map == no_map == "status = 'published'"


# ----------------------------------------------- the crux, both directions through deduplicate()


def test_same_source_placeholder_steps_merge():
    """CRUX 1 end-to-end: two selects over the SAME source MERGE through ``deduplicate()``.

    A cached/shared plan for one request is correct for the other (same value-agnostic SQL,
    re-bound per request), so the lowest-id step wins and both references resolve to it.
    """
    a = make_step(column("status") == pg_placeholder("var:status", "published"))
    b = make_step(column("status") == pg_placeholder("var:status", "draft"))
    assert dedup_key(a) == dedup_key(b)

    plan = Plan()
    plan.add_step(a)
    plan.add_step(b)
    remap = plan.deduplicate()
    assert remap[a.id] is remap[b.id]  # one survivor


def test_different_source_placeholder_steps_do_not_merge():
    """CRUX 2 end-to-end: two selects over DIFFERENT sources do NOT merge."""
    a = make_step(column("status") == pg_placeholder("var:status", "published"))
    b = make_step(column("owner_id") == pg_placeholder("var:owner", 1))
    assert dedup_key(a) != dedup_key(b)

    plan = Plan()
    plan.add_step(a)
    plan.add_step(b)
    remap = plan.deduplicate()
    assert remap[a.id] is a
    assert remap[b.id] is b


def test_placeholder_step_and_literal_step_do_not_merge():
    """CRUX 3 end-to-end: a placeholder step never merges with a literal step of same value.

    The anti-corruption gate: a value-agnostic placeholder plan must never be served where a
    value-pinned literal plan was planned (and vice versa).
    """
    ph = make_step(column("status") == pg_placeholder("var:status", "published"))
    lit = make_step(column("status") == "published")
    assert dedup_key(ph) != dedup_key(lit)

    plan = Plan()
    plan.add_step(ph)
    plan.add_step(lit)
    remap = plan.deduplicate()
    assert remap[ph.id] is ph
    assert remap[lit.id] is lit


def test_literal_only_steps_dedup_exactly_as_before():
    """CRUX 4 end-to-end: literal-only steps keep their value-discriminated merge/no-merge.

    The literal-path behaviour holds alongside the placeholder-aware code: identical literals
    merge; different literals do not.
    """
    a = make_step(column("status") == "published")
    b = make_step(column("status") == "published")
    c = make_step(column("status") == "draft")
    assert dedup_key(a) == dedup_key(b)
    assert dedup_key(a) != dedup_key(c)

    plan = Plan()
    for step in (a, b, c):
        plan.add_step(step)
    remap = plan.deduplicate()
    assert remap[a.id] is remap[b.id]  # identical literals merge
    assert remap[c.id] is c  # the different literal stays its own survivor
    assert remap[c.id] is not remap[a.id]


def test_mixed_literal_and_placeholder_keys_per_predicate():
    """CRUX 1 x CRUX 4 on ONE step: a placeholder and a literal predicate key INDEPENDENTLY.

    A step may carry BOTH a placeholder predicate (value-agnostic, source-keyed) and a plain
    literal predicate (value-included) at once. Each predicate must take its OWN key path —
    the placeholder source tag must NOT leak into the literal's key, and the literal value
    must NOT leak into the placeholder's. So two such steps over the SAME placeholder source
    but DIFFERENT literal values do NOT merge (the literal still discriminates), while the
    SAME source AND the SAME literal DO merge — proving :meth:`_placeholder_binds_for` scopes
    each key to its own predicate's binds, the subtlety a coarse step-wide registry would break.
    """
    # same placeholder source, but different literal title values -> must NOT merge.
    a = make_step(column("status") == pg_placeholder("var:status", "published"))
    a.builder().where(column("title") == "alpha")
    b = make_step(column("status") == pg_placeholder("var:status", "draft"))
    b.builder().where(column("title") == "beta")
    assert dedup_key(a) != dedup_key(b)

    # the placeholder predicate is value-agnostic (source tag, no runtime value); the literal
    # predicate is value-included — proving each predicate keyed independently in the signature.
    a_sig = a.customization_signature()
    assert a_sig[0] == "status = <<ph:var:status>>|ph=['var:status']"
    assert a_sig[1] == "title = 'alpha'"
    assert "published" not in a_sig[0]  # the placeholder value never enters its key

    plan = Plan()
    plan.add_step(a)
    plan.add_step(b)
    remap = plan.deduplicate()
    assert remap[a.id] is a
    assert remap[b.id] is b

    # SAME source AND same literal (different placeholder runtime value) -> DO merge.
    c = make_step(column("status") == pg_placeholder("var:status", "anything"))
    c.builder().where(column("title") == "alpha")
    assert dedup_key(a) == dedup_key(c)
    plan2 = Plan()
    plan2.add_step(a)
    plan2.add_step(c)
    remap2 = plan2.deduplicate()
    assert remap2[a.id] is remap2[c.id]


def test_combined_placeholder_and_literal_in_one_predicate_keeps_literal_value():
    """REGRESSION: a placeholder AND a literal in ONE ``and_(...)`` keep the literal VALUE-discriminated.

    The subtle cross-value-corruption path the per-predicate test above does NOT cover: when a
    SINGLE predicate combines a ``pg_placeholder`` bind with an ordinary literal bind (the
    natural ``where(and_($var, literal))`` / ``where_tree(And([...]))`` host shape), the
    placeholder must render value-agnostically while the co-located literal still renders
    INLINE by value. So two steps differing ONLY in that co-located literal (``title='alpha'``
    vs ``'beta'``) must NOT merge — the literal is not a placeholder, so a wrong merge would
    silently serve the survivor's literal to both requests (the worst-case bleed). The SAME
    source AND the SAME literal still merge.
    """
    a = make_step(
        and_(column("status") == pg_placeholder("var:status", "published"), column("title") == "alpha")
    )
    b = make_step(
        and_(column("status") == pg_placeholder("var:status", "draft"), column("title") == "beta")
    )
    # the placeholder source is shared, but the co-located LITERAL differs -> must NOT merge.
    assert dedup_key(a) != dedup_key(b)
    plan = Plan()
    plan.add_step(a)
    plan.add_step(b)
    remap = plan.deduplicate()
    assert remap[a.id] is not remap[b.id]
    # the literal value is in the key (value-discriminated); the placeholder value is NOT.
    a_key = a.customization_signature()[0]
    assert "'alpha'" in a_key
    assert "<<ph:var:status>>" in a_key
    assert "published" not in a_key

    # SAME source AND SAME co-located literal -> DO merge (only the placeholder value differs).
    c = make_step(
        and_(column("status") == pg_placeholder("var:status", "anything"), column("title") == "alpha")
    )
    assert dedup_key(a) == dedup_key(c)
    plan2 = Plan()
    plan2.add_step(a)
    plan2.add_step(c)
    remap2 = plan2.deduplicate()
    assert remap2[a.id] is remap2[c.id]


def test_expanding_in_placeholder_same_source_merges():
    """REGRESSION: two SAME-source ``IN`` (expanding) placeholders MERGE despite distinct bind names.

    ``column.in_(pg_placeholder(...))`` compiles to an expanding ``IN
    (__[POSTCOMPILE_grafast_ph_N])`` whose per-call counter is NOT in the scalar ``%(name)s``
    form a string rewrite would catch — so two same-source IN placeholders that SHOULD share a
    cache entry must still produce the IDENTICAL key (the AST sentinel erases the counter). A
    DIFFERENT source still keys distinctly.
    """
    a = make_step(column("id").in_(pg_placeholder("var:ids", [1, 2])))
    b = make_step(column("id").in_(pg_placeholder("var:ids", [3, 4])))
    assert dedup_key(a) == dedup_key(b)
    plan = Plan()
    plan.add_step(a)
    plan.add_step(b)
    remap = plan.deduplicate()
    assert remap[a.id] is remap[b.id]

    # a different source still keys distinctly (no spurious merge).
    c = make_step(column("id").in_(pg_placeholder("var:other", [1, 2])))
    assert dedup_key(a) != dedup_key(c)


# ------------------------------------------------------------ connection step placeholder path


def test_connection_placeholder_participates_in_key_by_source():
    """A placeholder folds into the connection step's dedup key BY SOURCE (not by value)."""
    a = PgConnectionStep(make_widgets(), constant(None), "owner_id", order_by=["id"])
    a.builder().where(column("status") == pg_placeholder("var:status", "published"))
    b = PgConnectionStep(make_widgets(), constant(None), "owner_id", order_by=["id"])
    b.builder().where(column("status") == pg_placeholder("var:status", "draft"))
    # same source, different value -> same key (cache hit across requests).
    assert a.peer_key == b.peer_key
    assert a.dedup_params() == b.dedup_params()

    c = PgConnectionStep(make_widgets(), constant(None), "owner_id", order_by=["id"])
    c.builder().where(column("owner_id") == pg_placeholder("var:owner", 1))
    # different source -> different key.
    assert a.peer_key != c.peer_key


# ---------------------------------------------------- no-regression + cross-request DB oracle


@pytest.mark.pg
@pytest.mark.asyncio
async def test_placeholder_result_byte_identical_to_inlined_literal(seeded):
    """NO-REGRESSION: a placeholder filter returns the SAME rows + SAME count as inlining.

    The literal path inlines ``column("status") == "draft"``; the placeholder path wraps the
    value as a placeholder. Same one batched statement, same rows — the placeholder changes
    only the dedup key, never the executed result.
    """
    engine = get_engine()
    with pg_request_context(SQLAlchemyExecutor(engine)):
        literal = PgSelectStep(make_widgets(), constant(None), "owner_id", order_by=["id"])
        literal.builder().where(column("status") == "draft")
        with count_sql(engine) as lit_counter:
            lit_out = await literal.execute(2, [[1, 2]])

    with pg_request_context(SQLAlchemyExecutor(engine)):
        placeheld = PgSelectStep(make_widgets(), constant(None), "owner_id", order_by=["id"])
        placeheld.builder().where(
            column("status") == pg_placeholder("var:status", "draft")
        )
        with count_sql(engine) as ph_counter:
            ph_out = await placeheld.execute(2, [[1, 2]])

    # byte-identical rows AND statement count.
    assert ph_counter.count == lit_counter.count == 1
    assert [[r["id"] for r in bucket] for bucket in ph_out] == [[2], [5]]
    assert ph_out == lit_out
    # the value is NOT inlined in the placeholder statement — it is a bound :param.
    sql = str(placeheld.build_query())
    assert "'draft'" not in sql
    assert "status = :" in sql


@pytest.mark.pg
@pytest.mark.asyncio
async def test_same_placeholder_statement_serves_two_values_correctly(seeded):
    """ANTI-CORRUPTION: one value-agnostic statement, run with TWO values, gives each its rows.

    A placeholder-bearing select is value-agnostic SQL re-bound per request. Running the same
    statement shape with ``status='draft'`` then ``status='published'`` must return each its
    OWN correct rows — the property a shared/cached plan depends on (a cache hit must never
    serve one request the value of another).
    """
    engine = get_engine()
    with pg_request_context(SQLAlchemyExecutor(engine)):
        draft = PgSelectStep(make_widgets(), constant(None), "owner_id", order_by=["id"])
        draft.builder().where(column("status") == pg_placeholder("var:status", "draft"))
        draft_out = await draft.execute(2, [[1, 2]])

    with pg_request_context(SQLAlchemyExecutor(engine)):
        published = PgSelectStep(make_widgets(), constant(None), "owner_id", order_by=["id"])
        published.builder().where(
            column("status") == pg_placeholder("var:status", "published")
        )
        pub_out = await published.execute(2, [[1, 2]])

    # the two requests share one value-agnostic statement SHAPE (the bind NAMES are unique
    # per call — execution plumbing the dedup key normalises out — so compare the SQL with
    # the placeholder names erased) but get distinct rows.
    draft_sql = _erase_placeholder_names(str(draft.build_query()), draft)
    pub_sql = _erase_placeholder_names(str(published.build_query()), published)
    assert draft_sql == pub_sql
    assert [[r["id"] for r in bucket] for bucket in draft_out] == [[2], [5]]
    assert [[r["id"] for r in bucket] for bucket in pub_out] == [[1, 3], [4, 6]]


def _erase_placeholder_names(sql: str, step) -> str:
    """Erase a step's unique placeholder bind names from its SQL, leaving the value-agnostic
    shape — the per-call ``grafast_ph_N`` name is plumbing the dedup key already normalises."""
    for bind_name in step.placeholder_binds:
        sql = sql.replace(f":{bind_name}", ":ph")
    return sql
