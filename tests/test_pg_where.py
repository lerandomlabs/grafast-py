"""Batch-uniform WHERE predicates and the customization seam.

A host may NARROW our batched skeleton but never touch it. The skeleton —
``match = ANY(:keys)`` (plain) or the ``row_number() OVER (PARTITION BY match)`` window
slice — stays ours; hosts add only UNIFORM additions, applied identically to every
parent's rows in the ONE batched statement: WHERE predicates (Core expressions, never
raw strings), a resource-level ``select_customizer`` (the selectAuth analogue, driven by
the per-request context), plus the structured order / first / offset surfaces. A host
filters by a GraphQL arg value by INLINING it (``.where(column("status") ==
args["status"])``): plan-time-known, parameterised at execute, and value-discriminated in
the dedup key. (Value-agnostic placeholders are deferred to the plan-caching phase.)

These tests assert:

- a per-plan ``.where()`` filters rows UNIFORMLY across parents in one statement, with
  the O(depth) statement count unchanged (a predicate adds a WHERE clause, never a query);
- a resource ``select_customizer`` is ANDed onto every select and is driven by the
  per-request context;
- fail-loud guards: a raw string ``.where()``, an UNBOUND bindparam, and a RESERVED bind
  name all raise — via ``.where()`` AND via a resource customizer;
- the builder's ``set_first``/``set_offset`` work on ``PgSelectAllStep``, and connection
  ``set_offset`` raises a CLEAR (non-AttributeError) message;
- dedup correctness (no DB), THE REGRESSION GATE: two selects with literal predicates
  ``status == 'published'`` vs ``== 'draft'`` get DIFFERENT keys and do NOT merge through
  ``dag.Plan.deduplicate()``; identical literals DO merge. A customizer-derived predicate
  participates in the key.

DB tests are marked ``pg`` (deselectable ``-m 'not pg'``); they touch ONLY the
``grafast_demo`` schema of ``grafast_py_test`` via the dedicated ``widgets`` fixture and
do not alter authors/posts/comments.
"""

import pytest
import pytest_asyncio
from sqlalchemy import String, bindparam, column

from grafast_py.core_steps import constant
from grafast_py.dag import Plan
from grafast_py.pg.connection import PgConnectionStep
from grafast_py.pg.customize import predicate_key
from grafast_py.pg.engine import count_sql, dispose_engine, get_engine
from grafast_py.pg.executor import SQLAlchemyExecutor, pg_request_context
from grafast_py.pg.resource import PgRegistry, PgResource
from grafast_py.pg.steps import PgSelectAllStep, PgSelectStep
from examples.seed import setup_demo_schema, setup_widgets_table


def make_widgets(*, select_customizer=None) -> PgResource:
    """A fresh ``widgets`` resource (its own registry) for the WHERE tests."""
    registry = PgRegistry()
    return PgResource(
        "widgets",
        "grafast_demo",
        "widgets",
        ["id", "owner_id", "title", "status", "deleted_at"],
        registry=registry,
        select_customizer=select_customizer,
    )


@pytest_asyncio.fixture
async def seeded():
    """(Re)seed ``grafast_demo`` + the ``widgets`` fixture (fresh engine per test)."""
    await dispose_engine()
    await setup_demo_schema()
    await setup_widgets_table()
    yield
    await dispose_engine()


# ------------------------------------------------------- per-plan .where() uniform


@pytest.mark.pg
@pytest.mark.asyncio
async def test_where_filters_uniformly_across_parents_one_statement(seeded):
    """A per-plan ``.where()`` filters EVERY parent's rows in ONE statement.

    Each owner (1, 2) has one soft-deleted widget; ``deleted_at IS NULL`` drops it for
    both owners in the single batched ``= ANY(:keys)`` select — the count stays 1.
    """
    widgets = make_widgets()
    engine = get_engine()
    with pg_request_context(SQLAlchemyExecutor(engine)):
        step = PgSelectStep(widgets, constant(None), "owner_id", order_by=["id"])
        step.builder().where(column("deleted_at").is_(None))
        with count_sql(engine) as counter:
            out = await step.execute(2, [[1, 2]])

    assert counter.count == 1  # one batched statement; the predicate is just a WHERE
    # owner 1 keeps widgets 1,2 (3 soft-deleted); owner 2 keeps 4,5 (6 soft-deleted).
    assert [r["id"] for r in out[0]] == [1, 2]
    assert [r["id"] for r in out[1]] == [4, 5]
    # the predicate is IN the batched WHERE, alongside the skeleton's ANY(:keys).
    sql = str(step.build_query())
    assert "deleted_at IS NULL" in sql
    assert "ANY (:keys)" in sql


@pytest.mark.pg
@pytest.mark.asyncio
async def test_where_filters_before_window_paging(seeded):
    """A ``.where()`` filters BEFORE per-parent ``row_number()`` paging.

    With ``first=1`` over the soft-delete-filtered set, owner 1's first remaining widget
    is 1 (not the deleted 3) — the page is numbered over the FILTERED rows.
    """
    widgets = make_widgets()
    engine = get_engine()
    with pg_request_context(SQLAlchemyExecutor(engine)):
        step = PgSelectStep(
            widgets, constant(None), "owner_id", order_by=["id"], first=1
        )
        step.builder().where(column("deleted_at").is_(None))
        with count_sql(engine) as counter:
            out = await step.execute(2, [[1, 2]])

    assert counter.count == 1
    assert [r["id"] for r in out[0]] == [1]
    assert [r["id"] for r in out[1]] == [4]
    # the predicate sits in the INNER select (before row_number), not the outer slice.
    sql = str(step.build_query())
    inner = sql.split(") AS anon_1", 1)[0]
    assert "deleted_at IS NULL" in inner


@pytest.mark.pg
@pytest.mark.asyncio
async def test_where_inlined_arg_value_filters_one_statement(seeded):
    """A host filters by an arg value by INLINING it — one value, one batched statement.

    This is the supported replacement for a value-agnostic placeholder: the value is
    plan-time-known, parameterised again at execute (SQLAlchemy binds the literal when the
    statement runs), and value-discriminated in the dedup key.
    """
    widgets = make_widgets()
    engine = get_engine()
    arg = {"status": "draft"}
    with pg_request_context(SQLAlchemyExecutor(engine)):
        step = PgSelectStep(widgets, constant(None), "owner_id", order_by=["id"])
        step.builder().where(column("status") == arg["status"])
        with count_sql(engine) as counter:
            out = await step.execute(2, [[1, 2]])

    assert counter.count == 1
    # only the draft widget per owner: owner 1 -> 2, owner 2 -> 5.
    assert [r["id"] for r in out[0]] == [2]
    assert [r["id"] for r in out[1]] == [5]
    # the value is NOT inlined in the EXECUTED statement — it is a bound :param.
    sql = str(step.build_query())
    assert "'draft'" not in sql
    assert "status = :" in sql


# --------------------------------------------------- resource-level select_customizer


@pytest.mark.pg
@pytest.mark.asyncio
async def test_select_customizer_anded_onto_every_select_from_context(seeded):
    """A resource ``select_customizer`` is ANDed onto every select, driven by the context.

    The customizer keeps only the context's ``status``; with ``status='published'`` only
    the published widgets survive, for every owner, in the one batched statement.
    """

    def only_status(ctx):
        # the value comes from the per-request context, inlined into the predicate; it is
        # plan-time-known here (planning runs inside pg_request_context) and rendered as a
        # bound :param at execute, value-discriminated in the dedup key.
        return [column("status") == ctx["status"]]

    widgets = make_widgets(select_customizer=only_status)
    engine = get_engine()
    with pg_request_context(
        SQLAlchemyExecutor(engine), context={"status": "published"}
    ):
        step = PgSelectStep(widgets, constant(None), "owner_id", order_by=["id"])
        with count_sql(engine) as counter:
            out = await step.execute(2, [[1, 2]])

    assert counter.count == 1
    # owner 1 published: widgets 1,3; owner 2 published: 4,6 (the drafts 2,5 are dropped).
    assert [r["id"] for r in out[0]] == [1, 3]
    assert [r["id"] for r in out[1]] == [4, 6]


@pytest.mark.pg
@pytest.mark.asyncio
async def test_select_customizer_combines_with_per_plan_where(seeded):
    """The resource customizer AND the per-plan ``.where()`` BOTH apply (flat AND)."""

    def only_published(ctx):
        return [column("status") == "published"]

    widgets = make_widgets(select_customizer=only_published)
    engine = get_engine()
    with pg_request_context(SQLAlchemyExecutor(engine), context={}):
        step = PgSelectStep(widgets, constant(None), "owner_id", order_by=["id"])
        # additionally drop soft-deleted rows.
        step.builder().where(column("deleted_at").is_(None))
        out = await step.execute(2, [[1, 2]])

    # published AND not-deleted: owner 1 -> just 1 (3 is deleted); owner 2 -> just 4.
    assert [r["id"] for r in out[0]] == [1]
    assert [r["id"] for r in out[1]] == [4]


# ------------------------------------------------------------- fail-loud guards


def test_raw_string_where_fails_loud():
    """A raw string ``where()`` is rejected (injection seam), not interpolated."""
    widgets = make_widgets()
    step = PgSelectStep(widgets, constant(None), "owner_id", order_by=["id"])
    with pytest.raises(TypeError, match="never a raw string"):
        step.builder().where("status = 'published'")


def test_unbound_bindparam_where_fails_loud():
    """An UNBOUND bindparam (no plan-time value) is rejected — pass the value inline."""
    widgets = make_widgets()
    step = PgSelectStep(widgets, constant(None), "owner_id", order_by=["id"])
    with pytest.raises(ValueError, match="unbound bindparam"):
        step.builder().where(column("status") == bindparam("p_status"))


def test_reserved_bind_name_where_fails_loud():
    """A predicate reusing a reserved skeleton bind (keys/first/offset) is rejected."""
    widgets = make_widgets()
    step = PgSelectStep(widgets, constant(None), "owner_id", order_by=["id"])
    for reserved in ("keys", "first", "offset"):
        with pytest.raises(ValueError, match="reserved skeleton bind"):
            step.builder().where(column("x") == bindparam(reserved, value=1))


def test_customizer_unbound_bindparam_fails_loud():
    """A customizer predicate with an unbound bindparam fails loud at construction."""

    def bad(ctx):
        return [column("status") == bindparam("p_status")]

    widgets = make_widgets(select_customizer=bad)
    with pg_request_context(SQLAlchemyExecutor(get_engine()), context={}):
        with pytest.raises(ValueError, match="unbound bindparam"):
            PgSelectStep(widgets, constant(None), "owner_id", order_by=["id"])


def test_customizer_reserved_bind_name_fails_loud():
    """A customizer predicate reusing a reserved skeleton bind fails loud."""

    def bad(ctx):
        return [column("x") == bindparam("keys", value=1)]

    widgets = make_widgets(select_customizer=bad)
    with pg_request_context(SQLAlchemyExecutor(get_engine()), context={}):
        with pytest.raises(ValueError, match="reserved skeleton bind"):
            PgSelectStep(widgets, constant(None), "owner_id", order_by=["id"])


def test_customizer_raw_string_fails_loud():
    """A customizer returning a raw string is rejected (injection seam)."""

    def bad(ctx):
        return ["status = 'published'"]

    widgets = make_widgets(select_customizer=bad)
    with pg_request_context(SQLAlchemyExecutor(get_engine()), context={}):
        with pytest.raises(TypeError, match="never a raw string"):
            PgSelectStep(widgets, constant(None), "owner_id", order_by=["id"])


# ------------------------------------------------------------- builder capability


def test_builder_set_first_offset_on_select_all():
    """``set_first``/``set_offset`` work on ``PgSelectAllStep`` (root page slice)."""
    widgets = make_widgets()
    step = PgSelectAllStep(widgets, order_by=["id"])
    step.for_parent(constant(None))
    step.builder().set_first(3).set_offset(2)
    assert step.first == 3
    assert step.offset == 2
    sql = str(step.build_query())
    assert "LIMIT" in sql
    assert "OFFSET" in sql


def test_builder_connection_set_offset_raises_clear_not_attributeerror():
    """Connection ``set_offset`` raises a CLEAR (non-AttributeError) message."""
    conn = PgConnectionStep(make_widgets(), constant(None), "owner_id", order_by=["id"])
    builder = conn.builder()
    with pytest.raises(TypeError) as excinfo:
        builder.set_offset(5)
    assert not isinstance(excinfo.value, AttributeError)
    assert "PgConnectionStep does not support set_offset" in str(excinfo.value)
    assert "after-cursor" in str(excinfo.value)


# --------------------------------------------------------------- dedup correctness


def dedup_key(step):
    """The class + peer_key + dedup_params slice of the planner's structural key.

    The full key in :func:`grafast_py.dag._structural_key` also folds the dependency
    survivor ids; a ``.where()`` adds no dependency, so the customization discriminator
    lives entirely in ``peer_key`` / ``dedup_params`` here.
    """
    return (type(step), step.peer_key, step.dedup_params())


def make_step(predicate=None):
    """A widgets select, optionally with one per-plan ``.where(predicate)``."""
    widgets = make_widgets()
    step = PgSelectStep(widgets, constant(None), "owner_id", order_by=["id"])
    if predicate is not None:
        step.builder().where(predicate)
    return step


def test_literal_predicates_differ_and_do_not_merge():
    """THE REGRESSION GATE: literal ``status=='published'`` vs ``=='draft'`` never merge.

    A value-free compile collapsed both to ``status = %(status_1)s`` and wrongly merged
    them; the value-included key renders ``status = 'published'`` vs ``status = 'draft'``,
    so the keys differ AND ``dag.Plan.deduplicate()`` keeps them as distinct survivors.
    """
    published = make_step(column("status") == "published")
    draft = make_step(column("status") == "draft")
    # the literal values are what discriminate the keys.
    assert published.peer_key != draft.peer_key
    assert published.dedup_params() != draft.dedup_params()
    assert dedup_key(published) != dedup_key(draft)

    # and they do NOT merge through the real deduplicate() pass.
    plan = Plan()
    plan.add_step(published)
    plan.add_step(draft)
    remap = plan.deduplicate()
    assert remap[published.id] is published
    assert remap[draft.id] is draft


def test_identical_literal_predicates_merge():
    """Two selects with the IDENTICAL literal predicate DO merge (same survivor)."""
    a = make_step(column("status") == "published")
    b = make_step(column("status") == "published")
    assert a.peer_key == b.peer_key
    assert a.dedup_params() == b.dedup_params()
    assert dedup_key(a) == dedup_key(b)

    plan = Plan()
    plan.add_step(a)
    plan.add_step(b)
    remap = plan.deduplicate()
    # the lower-id step wins; both references resolve to the same survivor.
    assert remap[a.id] is remap[b.id]


def test_different_bound_predicates_do_not_dedup():
    """Two selects differing only by a bound-value predicate are NOT peers (no merge)."""
    a = make_step(column("status") == bindparam("p", value="published"))
    b = make_step(column("status") == bindparam("p", value="draft"))
    assert a.peer_key != b.peer_key
    assert a.dedup_params() != b.dedup_params()
    assert dedup_key(a) != dedup_key(b)


def test_identical_predicates_dedup():
    """Two selects with the identical host predicate ARE peers (same dedup key)."""
    a = make_step(column("deleted_at").is_(None))
    b = make_step(column("deleted_at").is_(None))
    assert a.peer_key == b.peer_key
    assert a.dedup_params() == b.dedup_params()
    assert dedup_key(a) == dedup_key(b)


def test_predicate_vs_no_predicate_do_not_dedup():
    """A customized select never merges with an un-customized one over the same skeleton."""
    plain = make_step()
    filtered = make_step(column("deleted_at").is_(None))
    assert plain.peer_key != filtered.peer_key
    assert dedup_key(plain) != dedup_key(filtered)


def test_customizer_derived_predicate_participates_in_key():
    """A resource ``select_customizer`` predicate changes the dedup key.

    Two resources identical but for the customizer yield steps that must NOT merge — the
    customizer-derived predicate is folded into peer_key/dedup_params via its content key.
    """

    def scope(ctx):
        return [column("status") == "scoped"]

    with pg_request_context(SQLAlchemyExecutor(get_engine()), context={}):
        scoped_res = make_widgets(select_customizer=scope)
        scoped = PgSelectStep(scoped_res, constant(None), "owner_id", order_by=["id"])
        plain = PgSelectStep(make_widgets(), constant(None), "owner_id", order_by=["id"])
    assert scoped.peer_key != plain.peer_key
    assert scoped.dedup_params() != plain.dedup_params()
    assert dedup_key(scoped) != dedup_key(plain)
    # the customizer predicate's content key is present in the signature.
    assert scoped.customization_signature() == (
        predicate_key(column("status") == "scoped"),
    )


def test_predicate_key_is_value_included_and_discriminating():
    """The content key renders VALUES inline (value-included) and discriminates on value."""
    published = predicate_key(column("status") == "published")
    draft = predicate_key(column("status") == "draft")
    assert "'published'" in published
    assert published != draft
    # rebuilding the same logical predicate yields the identical key (so it dedups).
    again = predicate_key(column("status") == "published")
    assert published == again
    # the keys are hashable and tuple-composable (slot into dedup_params()).
    assert isinstance(hash((published, draft)), int)


def test_predicate_key_falls_back_for_unrenderable_literal():
    """An exotic literal that literal_binds cannot render must NOT crash planning.

    A non-UTF8 ``bytes`` value against a ``bytea`` column raises ``CompileError`` under
    ``literal_binds``; predicate_key falls back to a structural key (value-free SQL +
    bound-value repr) that still distinguishes two different exotic predicates.
    """
    from sqlalchemy import LargeBinary

    blob = column("data", LargeBinary)
    k1 = predicate_key(blob == b"\xff\xfe\x00")
    k2 = predicate_key(blob == b"\xff\xfe\x01")
    # planning did not crash, and two different exotic literals get different keys.
    assert k1 != k2
    # the same exotic predicate rebuilds to the identical key (so it dedups), and the
    # key is hashable/tuple-composable for dedup_params().
    assert predicate_key(blob == b"\xff\xfe\x00") == k1
    assert isinstance(hash((k1, k2)), int)


def test_connection_predicate_participates_in_key():
    """Customization folds into the connection step's dedup key too."""
    a = PgConnectionStep(
        make_widgets(), constant(None), "owner_id", order_by=["id"]
    )
    a.builder().where(column("deleted_at").is_(None))
    b = PgConnectionStep(
        make_widgets(), constant(None), "owner_id", order_by=["id"]
    )
    assert a.peer_key != b.peer_key
    assert a.dedup_params() != b.dedup_params()


# ----------------------------------------------------------- all-rows + apply seam


@pytest.mark.pg
@pytest.mark.asyncio
async def test_select_all_where_and_apply_seam(seeded):
    """``PgSelectAllStep`` honours ``.where()`` and the ``.apply(callback)`` host seam."""
    widgets = make_widgets()
    engine = get_engine()
    with pg_request_context(SQLAlchemyExecutor(engine)):
        step = PgSelectAllStep(widgets, order_by=["id"])
        step.for_parent(constant(None))

        def host(builder):
            builder.where(column("deleted_at").is_(None)).where(
                column("status") == "published"
            )

        step.builder().apply(host)
        with count_sql(engine) as counter:
            out = await step.execute(1, [[None]])

    assert counter.count == 1
    # published AND not deleted across all owners: widgets 1 and 4.
    assert [r["id"] for r in out[0]] == [1, 4]


@pytest.mark.pg
@pytest.mark.asyncio
async def test_select_all_first_offset_pages_root(seeded):
    """``PgSelectAllStep`` ``first``/``offset`` page the single root result (LIMIT/OFFSET)."""
    widgets = make_widgets()
    engine = get_engine()
    with pg_request_context(SQLAlchemyExecutor(engine)):
        step = PgSelectAllStep(widgets, order_by=["id"])
        step.for_parent(constant(None))
        step.builder().set_offset(1).set_first(2)
        with count_sql(engine) as counter:
            out = await step.execute(1, [[None]])

    assert counter.count == 1
    # all widgets ordered by id are 1..6; offset 1 + first 2 -> ids 2,3.
    assert [r["id"] for r in out[0]] == [2, 3]


# ------------------------- 2-arg placeholder customizer (cacheable, value-per-request)
#
# The convergence to upstream selectAuth: a 2-arg ``customizer(context, sources)`` uses
# ``sources.placeholder(key)`` to emit a value-LESS ``ctx:`` bind whose value is read from
# the request context PER request at execute (in ``where_params``). The predicate STRUCTURE is
# fixed at plan time; the VALUE is supplied per request — so the plan stays value-independent
# (cacheable) instead of baking one request's context value as a literal.


@pytest.mark.pg
@pytest.mark.asyncio
async def test_two_arg_placeholder_customizer_filters_from_context(seeded):
    """A 2-arg customizer filters by the context status via a value-AGNOSTIC bind."""

    def only_status(ctx, sources):
        return [column("status") == sources.placeholder("status", type_=String)]

    widgets = make_widgets(select_customizer=only_status)
    engine = get_engine()
    with pg_request_context(SQLAlchemyExecutor(engine), context={"status": "published"}):
        step = PgSelectStep(widgets, constant(None), "owner_id", order_by=["id"])
        out = await step.execute(2, [[1, 2]])
        sql = str(step.build_query())

    # owner 1 published: 1,3; owner 2 published: 4,6 — exactly the 1-arg customizer's result.
    assert [r["id"] for r in out[0]] == [1, 3]
    assert [r["id"] for r in out[1]] == [4, 6]
    # value-agnostic: the status is a bound :param, NOT inlined into the SQL text (so the plan
    # is shareable across contexts — the cacheable form).
    assert "'published'" not in sql
    assert "status = :" in sql


def test_two_arg_placeholder_customizer_is_value_less_and_resolves_per_request():
    """The ctx placeholder is value-LESS; ``where_params`` re-reads the context PER request."""

    def only_status(ctx, sources):
        return [column("status") == sources.placeholder("status")]

    widgets = make_widgets(select_customizer=only_status)
    with pg_request_context(SQLAlchemyExecutor(get_engine()), context={"status": "published"}):
        step = PgSelectStep(widgets, constant(None), "owner_id", order_by=["id"])
        ph_bind = step.where_predicates[0].right
        assert ph_bind.value is None  # value-LESS — no request value lives on the shared bind
        assert step.customizer_bakes_literal is False  # a placeholder customizer stays cacheable
        name = ph_bind.key
        # the value is read from THIS request's context at render.
        assert step.where_params({}) == {name: "published"}
    # a DIFFERENT request re-reads ITS OWN context off the SAME step — no baked value, no bleed.
    with pg_request_context(SQLAlchemyExecutor(get_engine()), context={"status": "draft"}):
        assert step.where_params({}) == {name: "draft"}


def test_one_arg_literal_customizer_marks_bakes_literal():
    """A 1-arg (literal) customizer marks the step ``customizer_bakes_literal`` (the floor signal)."""

    def only_status(ctx):
        return [column("status") == ctx["status"]]

    widgets = make_widgets(select_customizer=only_status)
    with pg_request_context(SQLAlchemyExecutor(get_engine()), context={"status": "published"}):
        step = PgSelectStep(widgets, constant(None), "owner_id", order_by=["id"])
    assert step.customizer_bakes_literal is True
    assert step.has_literal_customization() is True


def test_two_arg_customizer_ignoring_sources_still_bakes_literal():
    """A 2-arg customizer that inlines a literal (ignoring ``sources``) is STILL non-cacheable.

    Guards a host mistake: using the 2-arg form but reading ``ctx[...]`` into the predicate
    instead of ``sources.placeholder(...)`` must not silently re-enable caching of a per-request
    value (which would leak across contexts).
    """

    def only_status(ctx, sources):
        return [column("status") == ctx["status"]]  # a literal — ignores sources

    widgets = make_widgets(select_customizer=only_status)
    with pg_request_context(SQLAlchemyExecutor(get_engine()), context={"status": "published"}):
        step = PgSelectStep(widgets, constant(None), "owner_id", order_by=["id"])
    assert step.customizer_bakes_literal is True


def test_static_customizer_with_no_value_stays_cacheable():
    """A customizer with NO per-request value (a static ``IS NULL``) is value-independent."""

    def hide_deleted(ctx):
        return [column("deleted_at").is_(None)]

    widgets = make_widgets(select_customizer=hide_deleted)
    with pg_request_context(SQLAlchemyExecutor(get_engine()), context={}):
        step = PgSelectStep(widgets, constant(None), "owner_id", order_by=["id"])
    # no bound value at all => nothing per-request is baked => the plan stays cacheable.
    assert step.customizer_bakes_literal is False


def test_ctx_placeholder_missing_key_fails_loud():
    """A ctx placeholder for a key ABSENT from the request context fails LOUD at render.

    Not a silent ``None`` (which would render ``owner_id = NULL`` and silently widen the scope):
    a missing scoping value is a wiring bug, so the ``KeyError`` propagates.
    """

    def only_tenant(ctx, sources):
        return [column("owner_id") == sources.placeholder("tenant")]

    widgets = make_widgets(select_customizer=only_tenant)
    with pg_request_context(SQLAlchemyExecutor(get_engine()), context={}):
        step = PgSelectStep(widgets, constant(None), "owner_id", order_by=["id"])
        with pytest.raises(KeyError):
            step.where_params({})


def test_customizer_with_too_many_args_fails_loud():
    """A select_customizer taking >2 args is rejected at resource construction (clear TypeError)."""
    with pytest.raises(TypeError):
        make_widgets(select_customizer=lambda a, b, c: [])


def test_inline_clone_preserves_bakes_literal_flag():
    """An inlining clone (copy_customization_from) keeps the safety-floor flag + predicate count.

    A clone of a literal-baking customizer step must stay non-cacheable; dropping the flag on the
    clone would silently re-enable caching of a per-request value (a cross-context leak).
    """

    def only_status(ctx):
        return [column("status") == ctx["status"]]

    widgets = make_widgets(select_customizer=only_status)
    with pg_request_context(SQLAlchemyExecutor(get_engine()), context={"status": "published"}):
        step = PgSelectStep(widgets, constant(None), "owner_id", order_by=["id"])
        clone = step.clone_with_inline_specs([])
    assert step.customizer_bakes_literal is True
    assert clone.customizer_bakes_literal is True
    assert clone._customizer_predicate_count == step._customizer_predicate_count
