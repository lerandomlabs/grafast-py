"""Cross-context plan-cache ISOLATION: a context-scoped step served from a SHARED cached plan
must render THIS request's context, never the request that built the plan.

The plan cache key (:func:`grafast_py.cache.compute_cache_key`) carries NO request context — it
is ``(id(schema), document-text hash, operation name, variable fingerprint, config
fingerprint)``. Two requests of the SAME document under DIFFERENT per-request CONTEXT (a tenant
id, a viewer) therefore collide on ONE cache entry. The MR-B end-state makes that collision SAFE:
a context-scoping predicate is built with a value-LESS ``ctx:`` placeholder (the cache-safe way to
scope by request context — see :class:`grafast_py.pg.customize.ContextSources`), so the SHARED
cached SQL bakes NO per-request value. On a cache HIT the served step is re-rendered against the
NEW request's context via the render seam (:meth:`PgCustomizable.where_params` /
:meth:`PgUnionAllStep.member_where_params`), which resolves each ``ctx:`` bind from THIS request's
context — so request 2 (tenant 2) gets tenant-2 params off the very plan request 1 built.

These tests cover the two paths that were the original UNGUARDED leak surfaces and prove they are
now cacheable-AND-correct:

  (a) a ``pgUnionAll`` member ``where=`` that scopes by request context.
      :class:`grafast_py.pg.union.PgUnionAllStep` is NOT a ``PgCustomizable``; it gathers its own
      per-request placeholder params in :meth:`member_where_params`, which resolves a member's
      ``ctx:`` bind from THIS request's context. The SHARED union SQL is value-LESS; the per-request
      value rides ``params``.

  (b) a context-scoping predicate added through the query builder.
      The cache-safe way to scope by request context is a ``ctx:`` placeholder (NOT a raw context
      literal, which the builder docs forbid — it would bleed across requests). Such a step stays
      cacheable (``customizer_bakes_literal`` False) and re-binds per request through
      :meth:`PgCustomizable.where_params`.

We assert on the RENDER SEAM (the per-request ``where_params`` / ``member_where_params`` the
subclass hands ``executor.run``), NOT on ``literal_binds`` of the shared statement: a value-LESS
``ctx:`` bind renders as a value-less ``%(name)s`` param (it would inline as ``NULL`` under
``literal_binds`` — the flaw of the earlier baked-literal model), while the actual per-request
value lives in the render-seam params. The leaked-literal failure mode (a per-request value baked
into the SHARED cached SQL) is asserted ABSENT: the compiled statement text carries no tenant
literal at all. The desired upstream end-state — context as a runtime unary dependency, never
baked, with constraint lists validated on lookup — is in grafast-crystal ``pgSelect.ts`` (context
as a runtime unary dependency) and ``establishOperationPlan.ts`` (constraints validated on lookup);
the optimization-independent constraint guard is exercised in :mod:`tests.test_constraint_keyed_cache`.

No-DB throughout: we drive the REAL :func:`grafast_py.plan.lookup_cached_plan` path (proving the
cache HIT serves the divergent context the shared plan) and assert on the compiled statement TEXT
and the render-seam params of the served plan — never on fetched rows.
"""

import types

from graphql import parse
from sqlalchemy import String, column
from sqlalchemy.dialects import postgresql

from grafast_py.cache import CachedPlan, PlanCache, compute_cache_key
from grafast_py.config import GrafastConfig
from grafast_py.core_steps import constant
from grafast_py.dag import Plan
from grafast_py.plan import lookup_cached_plan
from grafast_py.pg.executor import pg_request_context
from grafast_py.pg.placeholders import pg_placeholder
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


def compiled_sql(stmt) -> str:
    """Compile a Core statement to its parameterized text (NO ``literal_binds``).

    We use this to assert a per-request value is NOT baked into the SHARED cached SQL: a value-LESS
    ``ctx:`` placeholder renders as a bound param (``WHERE owner_id = %(grafast_ph_0)s``), so a
    tenant integer (``owner_id = 1``) would appear in the text ONLY if a per-request value leaked
    inline onto the shared predicate. (Compiling with ``literal_binds`` would render the value-less
    bind as ``NULL`` — the misleading artefact of the earlier baked-literal model; the correct
    per-request value rides the render seam ``where_params`` / ``member_where_params``, asserted
    separately.)
    """
    return str(stmt.compile(dialect=postgresql.dialect()))


def stand_in_context(schema):
    """A minimal stand-in for a graphql-core ExecutionContext that lookup_cached_plan reads.

    ``lookup_cached_plan`` touches only ``schema`` / ``fragments`` / ``variable_values`` /
    ``context_value`` on the context and writes the resolved plan back onto it — a
    SimpleNamespace covers all of that without standing up a real operation, keeping the
    construction no-DB and focused on the cache lookup itself.
    """
    return types.SimpleNamespace(
        schema=schema, fragments=None, variable_values={}, context_value=None
    )


def serve_second_request(plan: Plan, *, context):
    """Store ``plan`` (built under request 1) and run request 2's REAL cache lookup; return the
    served plan.

    Both requests share the SAME document + config, so request 2 collides on request 1's cache
    entry. We drive the genuine :func:`grafast_py.plan.lookup_cached_plan` (not a hand-rolled get)
    so the assertion proves the cache HIT serves the divergent context the very plan request 1
    built — the isolation must come from the value-LESS render seam, in the production lookup path.
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


# ----------------------------------------------------- (a) pgUnionAll member where= context scope


def union_scoped_by_context() -> PgUnionAllStep:
    """A root ``UNION ALL`` whose every member scopes by a value-LESS ``ctx:owner_id`` placeholder.

    The cache-safe way to scope a union member by request context: a ``pg_placeholder`` tagged
    ``ctx:owner_id`` (resolved per request in :meth:`member_where_params`), NOT a hand-baked
    ``column('owner_id') == <tenant>`` literal (which would bleed the building request's value into
    the SHARED cached SQL).
    """
    return PgUnionAllStep(
        [
            PgUnionMember(
                make_articles(), "Article",
                where=[column("owner_id") == pg_placeholder("ctx:owner_id", type_=String)],
            ),
            PgUnionMember(
                make_snippets(), "Snippet",
                where=[column("owner_id") == pg_placeholder("ctx:owner_id", type_=String)],
            ),
        ],
        shared_columns=["id", "owner_id", "created"],
        order_by=["created"],
        first=3,
    )


def test_union_member_where_does_not_leak_across_contexts():
    """A context-scoped union member served from request 1's cache must render TENANT 2, never 1.

    Request 1 (tenant 1) builds a root ``UNION ALL`` whose members scope by a ``ctx:owner_id``
    placeholder. Request 2 (tenant 2) runs the SAME document and HITS request 1's cache entry. The
    served plan IS request 1's plan (deepcopy-free), so isolation must come from the render seam:
    rendered under tenant-2 context, ``member_where_params`` resolves the ``ctx:`` bind to 2 for
    EVERY member, never tenant 1's value. The SHARED union SQL is value-LESS — it bakes no tenant
    literal at all, so there is nothing to leak.
    """
    plan = Plan()
    plan.add_step(union_scoped_by_context())  # request 1 (tenant 1) builds the shared plan

    served = serve_second_request(plan, context={"owner_id": 2})
    union = next(s for s in served.steps if isinstance(s, PgUnionAllStep))

    # the served union renders request 2's tenant for every member branch, off the SHARED step.
    with pg_request_context(object(), context={"owner_id": 2}):
        params = union.member_where_params({})
    assert list(params.values()) == [2, 2], (
        f"request 2 must scope every member to tenant 2; got {params!r}"
    )
    assert 1 not in params.values(), f"request 2 leaked tenant 1's scope: {params!r}"

    # the SHARED cached SQL scopes via a value-LESS bound param (no per-request tenant baked in), so
    # a tenant literal can never bleed across requests through the shared statement.
    sql = compiled_sql(union.build_page_query())
    assert "owner_id = %(" in sql, f"the union scope must ride a value-less bound param:\n{sql}"
    assert "owner_id = 1" not in sql, f"a per-request tenant leaked into the shared SQL:\n{sql}"
    assert "owner_id = 2" not in sql, f"a per-request tenant leaked into the shared SQL:\n{sql}"


def test_union_member_value_rebinds_per_request():
    """The served union RE-BINDS each request's context value off the ONE shared step.

    Isolated from the no-leak data assertion above: this pins the rebind MECHANISM. The served step
    (request 1's plan) exposes its scope as a render-time PARAM, so rendering it under context after
    context yields each request's OWN value — tenant 2, then tenant 7 — never a frozen literal. One
    shared plan, the value supplied per request.
    """
    plan = Plan()
    plan.add_step(union_scoped_by_context())

    served = serve_second_request(plan, context={"owner_id": 2})
    union = next(s for s in served.steps if isinstance(s, PgUnionAllStep))

    with pg_request_context(object(), context={"owner_id": 2}):
        params_2 = union.member_where_params({})
    with pg_request_context(object(), context={"owner_id": 7}):
        params_7 = union.member_where_params({})

    assert set(params_2.values()) == {2}, f"render under tenant 2 must bind 2; got {params_2!r}"
    assert set(params_7.values()) == {7}, f"render under tenant 7 must bind 7; got {params_7!r}"
    # same bind names across requests (one shared step), only the resolved value differs.
    assert set(params_2) == set(params_7)


# ----------------------------------------------------- (b) builder context-scoped predicate


def test_raw_where_context_scope_does_not_leak_across_contexts():
    """A context-scoped builder predicate served from request 1's cache renders TENANT 2, never 1.

    Request 1 (tenant 1) scopes a root select by a ``ctx:tenant_id`` placeholder through the query
    builder. Request 2 (tenant 2) runs the SAME document and HITS the cache. Because the predicate
    is a value-LESS ``ctx:`` bind, the plan stays cacheable (``customizer_bakes_literal`` False) and
    the served step re-binds through ``where_params``: under tenant-2 context it resolves to 2,
    never tenant 1. The SHARED cached SQL bakes no tenant literal, so nothing can bleed across
    requests through the statement.
    """
    step = PgSelectAllStep(make_widgets(), order_by=["id"]).for_parent(constant(None))
    # the cache-safe scope: a value-LESS ctx: placeholder, resolved per request (NOT a raw context
    # literal, which the builder docstring forbids precisely because it would bleed across requests).
    step.builder().where(column("tenant_id") == pg_placeholder("ctx:tenant_id", type_=String))

    plan = Plan()
    plan.add_step(step)
    # a ctx: placeholder is value-agnostic, so the step stays cacheable (the floor does not trip).
    assert not any(getattr(s, "customizer_bakes_literal", False) for s in plan.steps)

    served = serve_second_request(plan, context={"tenant_id": 2})
    select_step = next(s for s in served.steps if isinstance(s, PgSelectAllStep))

    with pg_request_context(object(), context={"tenant_id": 2}):
        params = select_step.where_params({})
    assert list(params.values()) == [2], f"request 2 must scope to tenant 2; got {params!r}"
    assert 1 not in params.values(), f"request 2 leaked tenant 1's scope: {params!r}"

    # the SHARED cached SQL scopes via a value-LESS bound param — no tenant literal baked in.
    sql = compiled_sql(select_step.build_query())
    assert "tenant_id = %(" in sql, f"the scope must ride a value-less bound param:\n{sql}"
    assert "tenant_id = 1" not in sql, f"a per-request tenant leaked into the shared SQL:\n{sql}"
    assert "tenant_id = 2" not in sql, f"a per-request tenant leaked into the shared SQL:\n{sql}"


def test_raw_where_context_scope_rebinds_per_request():
    """The served builder-scoped step RE-BINDS each request's context value off the ONE shared step.

    The rebind MECHANISM for the builder path, isolated from the no-leak data assertion: the served
    step exposes its scope as a render-time PARAM, so rendering it under successive contexts yields
    each request's OWN value (tenant 2, then tenant 5), never a frozen literal — one cacheable plan
    serving every context correctly.
    """
    step = PgSelectAllStep(make_widgets(), order_by=["id"]).for_parent(constant(None))
    step.builder().where(column("tenant_id") == pg_placeholder("ctx:tenant_id", type_=String))

    plan = Plan()
    plan.add_step(step)

    served = serve_second_request(plan, context={"tenant_id": 2})
    select_step = next(s for s in served.steps if isinstance(s, PgSelectAllStep))

    with pg_request_context(object(), context={"tenant_id": 2}):
        params_2 = select_step.where_params({})
    with pg_request_context(object(), context={"tenant_id": 5}):
        params_5 = select_step.where_params({})

    assert list(params_2.values()) == [2], f"render under tenant 2 must bind 2; got {params_2!r}"
    assert list(params_5.values()) == [5], f"render under tenant 5 must bind 5; got {params_5!r}"
    assert set(params_2) == set(params_5)
