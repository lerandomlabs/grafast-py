"""Cross-context plan-cache LEAK: the request-CONTEXT a step bakes into a SHARED cached plan.

The plan cache key (:func:`grafast_py.cache.compute_cache_key`) carries NO request context — it
is ``(id(schema), document-text hash, operation name, variable fingerprint, config
fingerprint)``. Two requests of the SAME document under DIFFERENT per-request CONTEXT (a tenant
id, a viewer) therefore collide on ONE cache entry. On a HIT the only re-validation
(:func:`grafast_py.plan.lookup_cached_plan`) is the surviving customizer steps'
``customizer_structure_matches`` — so a plan that baked a CONTEXT value somewhere OTHER than a
resource ``select_customizer`` is served verbatim to the next request, leaking the FIRST
request's scope.

These tests target the two genuinely UNGUARDED context-baking paths (the dedup/customizer paths
are guarded — ``id(customizer)`` leads ``customization_signature`` and the cacheability floor +
structural guard cover ``select_customizer`` — so they are NOT exercised here):

  (a) a ``pgUnionAll`` member ``where=`` that bakes a per-request CONTEXT literal.
      :class:`grafast_py.pg.union.PgUnionAllStep` is NOT a ``PgCustomizable`` (it subclasses
      ``Step`` directly), so it carries neither ``customizer_bakes_literal`` (the cacheability
      floor never trips — the plan is wrongly stored) NOR ``customizer_structure_matches`` (the
      cache-hit guard never fires). The baked literal renders straight into the union's SQL.

  (b) a raw ``PgSelectQueryBuilder.where()`` predicate that bakes a per-request CONTEXT literal.
      ``add_where`` does NOT set ``customizer_bakes_literal``, and the cacheability floor scans
      ONLY the customizer prefix (``has_literal_customization`` reads
      ``where_predicates[:_customizer_predicate_count]``), so a raw ``.where()`` appended after
      that prefix leaves the plan wrongly cacheable and the structural guard trivially matches
      (no customizer present).

Each test asserts the CORRECT (tenant-2) end state and is ``xfail(strict=True)`` today, so it is
RED now and turns into a loud CI failure (forcing the marker's removal) once the ConstraintCache
machine lands. The desired upstream end-state — a divergent context is a structural cache MISS,
because context is a runtime unary dependency (never baked) and constraint lists are validated on
lookup — is in grafast-crystal ``establishOperationPlan.ts`` (constraint lists validated on
lookup) and ``dataplan-pg/src/steps/pgSelect.ts`` (context as a runtime unary dependency).

No-DB throughout: we drive the REAL :func:`grafast_py.plan.lookup_cached_plan` path (proving the
guards do not fire) and assert on the compiled statement TEXT of the plan the cache served the
second request — never on fetched rows. A genuinely DB-needing variant is marked ``pg``.
"""

import types

import pytest
from graphql import parse
from sqlalchemy import column
from sqlalchemy.dialects import postgresql

from grafast_py.cache import CachedPlan, PlanCache, compute_cache_key
from grafast_py.config import GrafastConfig
from grafast_py.core_steps import constant
from grafast_py.dag import Plan
from grafast_py.plan import lookup_cached_plan
from grafast_py.pg.executor import pg_request_context
from grafast_py.pg.resource import PgRegistry, PgResource
from grafast_py.pg.steps import PgSelectAllStep
from grafast_py.pg.union import PgUnionAllStep, PgUnionMember


# --------------------------------------------------------------------------- helpers (no DB)


def make_articles() -> PgResource:
    return PgResource(
        "articles", "grafast_demo", "articles",
        ["id", "owner_id", "created", "headline", "word_count"],
        registry=PgRegistry(),
    )


def make_snippets() -> PgResource:
    return PgResource(
        "snippets", "grafast_demo", "snippets",
        ["id", "owner_id", "created", "body"],
        registry=PgRegistry(),
    )


def make_widgets() -> PgResource:
    return PgResource(
        "widgets", "grafast_demo", "widgets",
        ["id", "owner_id", "title", "status", "deleted_at"],
        registry=PgRegistry(),
    )


def inlined_sql(stmt) -> str:
    """Compile a Core statement with literal_binds so a BAKED value renders INLINE in the text.

    The baked context literal is what we are detecting: a value-LESS placeholder would render as
    a ``%(name)s`` param (no inline value), so inlining the binds surfaces exactly the leaked
    literal (``owner_id = 1``) vs the correct one (``owner_id = 2``).
    """
    return str(
        stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
    )


def stand_in_context(schema):
    """A minimal stand-in for a graphql-core ExecutionContext that lookup_cached_plan reads.

    ``lookup_cached_plan`` touches only ``schema`` / ``fragments`` / ``variable_values`` on the
    context and writes the resolved plan back onto it — a SimpleNamespace covers all of that
    without standing up a real operation, keeping the construction no-DB and focused on the cache
    lookup itself.
    """
    return types.SimpleNamespace(schema=schema, fragments=None, variable_values={})


def serve_second_request(plan: Plan, *, context):
    """Store ``plan`` (built under request 1) and run request 2's REAL cache lookup; return the
    served plan.

    Both requests share the SAME document + config, so request 2 collides on request 1's cache
    entry. We drive the genuine :func:`grafast_py.plan.lookup_cached_plan` (not a hand-rolled
    get) so the assertion proves the cache-hit structural guard does NOT reject the divergent
    context — the leak is in the production lookup path, not in the test harness.
    """
    schema = object()
    config = GrafastConfig(cache_plans=True, placeholders=True, plan_cache=PlanCache())
    operation = parse("query Q { rows { id } }").definitions[0]
    config.plan_cache.put(
        compute_cache_key(schema, operation, None, config),
        CachedPlan(object_plan="request-1-plan", root_step=None, plan=plan, schema=schema),
    )
    served_context = stand_in_context(schema)
    with pg_request_context(object(), context=context):
        served_object_plan = lookup_cached_plan(served_context, operation, config)
    assert served_object_plan == "request-1-plan", "request 2 must hit request 1's cache entry"
    return served_context._grafast_plan


# ----------------------------------------------------- (a) pgUnionAll member where= context leak


@pytest.mark.xfail(
    reason="pgUnionAll member where= baking a per-request CONTEXT literal leaks across "
    "contexts under cache_plans: PgUnionAllStep is not a PgCustomizable, so neither the "
    "cacheability floor nor the cache-hit structural guard covers it — fixed by ConstraintCache",
    strict=True,
)
def test_union_member_where_context_literal_does_not_leak_across_contexts():
    """A union member ``where=`` baking the request CONTEXT must not bleed across requests.

    Request 1 (tenant 1) builds a root ``UNION ALL`` whose every member ``where=`` bakes
    ``owner_id = 1`` from its context. Request 2 (tenant 2) runs the SAME document and HITS the
    cache. The CORRECT end-state is request 2's served plan scoping to ``owner_id = 2``; today it
    inherits the baked ``owner_id = 1`` (the leak), because the union step is invisible to both
    cache-safety guards.
    """
    tenant_1, tenant_2 = 1, 2

    def union_for(tenant: int) -> PgUnionAllStep:
        return PgUnionAllStep(
            [
                PgUnionMember(make_articles(), "Article", where=[column("owner_id") == tenant]),
                PgUnionMember(make_snippets(), "Snippet", where=[column("owner_id") == tenant]),
            ],
            shared_columns=["id", "owner_id", "created"],
            order_by=["created"],
            first=3,
        )

    plan = Plan()
    plan.add_step(union_for(tenant_1))  # request 1 baked tenant 1 into the union members

    served = serve_second_request(plan, context={"owner_id": tenant_2})
    union = next(s for s in served.steps if isinstance(s, PgUnionAllStep))
    sql = inlined_sql(union.build_page_query(source_values=union.member_where_params({})))

    # request 2 must be scoped to ITS tenant, never tenant 1's baked predicate.
    assert "owner_id = 2" in sql, f"request 2 should scope to tenant 2; got:\n{sql}"
    assert "owner_id = 1" not in sql, f"request 2 leaked tenant 1's predicate:\n{sql}"


@pytest.mark.xfail(
    reason="pgUnionAll member where= context literal: member_where_params re-binds nothing "
    "(the baked literal is not a placeholder), so a cache HIT cannot rescope to this request's "
    "context — fixed by ConstraintCache",
    strict=True,
)
def test_union_member_context_value_rebinds_per_request():
    """The served union must re-bind THIS request's context value, not carry a frozen literal.

    A cache-safe context-scoped step exposes its per-request value as a render-time PARAM (like a
    ``pg_placeholder``), so a HIT re-binds the new context. A baked literal exposes NOTHING to
    re-bind: ``member_where_params`` is empty and the value is frozen in the SQL. The desired
    end-state surfaces the request-2 tenant value at render — asserted here on the served step.
    """
    plan = Plan()
    plan.add_step(
        PgUnionAllStep(
            [PgUnionMember(make_articles(), "Article", where=[column("owner_id") == 1])],
            shared_columns=["id", "owner_id", "created"],
            order_by=["created"],
            first=3,
        )
    )

    served = serve_second_request(plan, context={"owner_id": 2})
    union = next(s for s in served.steps if isinstance(s, PgUnionAllStep))

    # the served step must expose request 2's context value (2) as a re-bindable param; today the
    # value is baked into the SQL and member_where_params is empty, so this is RED.
    assert 2 in union.member_where_params({}).values(), (
        "request 2's context value must re-bind per request, not be frozen in the cached SQL"
    )


# ----------------------------------------------------- (b) raw .where() context literal leak


@pytest.mark.xfail(
    reason="raw PgSelectQueryBuilder.where() baking a per-request CONTEXT literal leaks across "
    "contexts under cache_plans: add_where leaves customizer_bakes_literal False and the "
    "cacheability floor scans only the customizer prefix — fixed by ConstraintCache",
    strict=True,
)
def test_raw_where_context_literal_does_not_leak_across_contexts():
    """A raw ``.where()`` baking the request CONTEXT must not bleed across requests.

    Request 1 (tenant 1) adds ``.where(column('tenant_id') == 1)`` from its context onto a root
    select. Request 2 (tenant 2) runs the SAME document and HITS the cache. The CORRECT end-state
    is request 2 scoping to ``tenant_id = 2``; today the served plan still bakes ``tenant_id = 1``
    (the leak), because a raw ``.where()`` is past the customizer prefix the cacheability floor
    scans and carries no structural guard.
    """
    step = PgSelectAllStep(make_widgets(), order_by=["id"]).for_parent(constant(None))
    step.builder().where(column("tenant_id") == 1)  # request 1 baked its context tenant

    # the cacheability floor (plan.py) refuses to cache ONLY steps that set
    # customizer_bakes_literal; a raw .where() does not, so this plan is wrongly cacheable.
    plan = Plan()
    plan.add_step(step)
    assert not any(getattr(s, "customizer_bakes_literal", False) for s in plan.steps)

    served = serve_second_request(plan, context={"tenant_id": 2})
    select_step = next(s for s in served.steps if isinstance(s, PgSelectAllStep))
    sql = inlined_sql(select_step.build_query())

    assert "tenant_id = 2" in sql, f"request 2 should scope to tenant 2; got:\n{sql}"
    assert "tenant_id = 1" not in sql, f"request 2 leaked tenant 1's predicate:\n{sql}"


@pytest.mark.xfail(
    reason="raw .where() context literal evades the cacheability floor: has_literal_customization "
    "scans only the customizer prefix (where_predicates[:_customizer_predicate_count]), so a "
    "context-baking raw .where() is wrongly cacheable — fixed by ConstraintCache",
    strict=True,
)
def test_raw_where_context_predicate_forces_non_cacheable():
    """A raw ``.where()`` carrying a plan-time CONTEXT literal must force the plan NON-cacheable.

    The safety-floor analogue of a literal ``select_customizer``: a value baked from the request
    context cannot be shared across requests, so the plan must not be cached. The floor today
    inspects only the resource-customizer prefix, so it misses a raw ``.where()`` — the desired
    end-state is that any context-derived baked predicate (wherever it sits) trips the floor.
    """
    step = PgSelectAllStep(make_widgets(), order_by=["id"]).for_parent(constant(None))
    step.builder().where(column("tenant_id") == 1)

    # The desired end-state: a context-baking predicate makes the step non-cacheable. Today
    # add_where leaves customizer_bakes_literal False, so the floor (which reads exactly this
    # duck-typed flag in plan.py) lets the value-pinned plan be cached — the leak's root.
    assert step.customizer_bakes_literal is True, (
        "a raw .where() baking a context literal must mark the step non-cacheable, "
        "like a literal select_customizer does"
    )
