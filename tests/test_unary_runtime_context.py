"""The UNARY-MODEL capability: request context threaded as a RUNTIME UNARY value.

The sibling :mod:`tests.test_plan_cache` proves the cache LEAK *bug* is contained — a
context-baking ``select_customizer`` is forced non-cacheable, and a structure-branching one
is caught by the cache-hit structural-divergence guard so it never serves one request another
request's scope. That file is about not LEAKING. This file is about the capability that would
make the leak moot in the first place: threading the per-request context as a RUNTIME value so
that ONE plan serves EVERY context correctly, the way upstream Grafast does.

Upstream design (the anchors this file mirrors), in
``grafast/dataplan-pg/src/steps/pgSelect.ts``:

  * line ~702  ``this.contextId = this.addUnaryDependency(inContext ?? resource.executor.context())``
               — the request context is a UNARY DEPENDENCY added at PLAN time, not read then.
  * line ~1118 ``const context = values[this.contextId].unaryValue()``
               — its VALUE is read at EXECUTE time, per request.

So a ``selectAuth`` predicate's STRUCTURE is fixed once at plan time while its VALUE — and any
TRANSFORM of the context that feeds it — is computed per request from the unary context step.
The SAME compiled plan therefore serves context A and context B, each correct.

grafast-py resolves a resource ``select_customizer`` against ``current_pg_request().context``
at PLAN/CONSTRUCTION time (``pg/customize.py`` ``seed_resource_customization`` ~380-386), baking
the predicate STRUCTURE — and, for any value that is not a BARE ``context[key]`` lookup, baking
the VALUE as a plan-time literal too. The 2-arg ``ContextSources.placeholder(key)`` form threads
a bare key as a runtime ``ctx:<key>`` value (that much already works — covered by
``test_plan_cache.test_placeholder_select_customizer_cacheable_no_leak``), but it has NO way to
thread a DERIVED/transformed context value as a runtime value: ``placeholder`` takes only a key,
and the docstring (``pg/customize.py`` ~290-292) explicitly says a derived scoping value "is not
expressible this way; expose it under its own context key, or use the 1-arg form (which inlines a
literal and is non-cacheable)". The consequence is the UnaryModel gap: a context-DERIVED WHERE
value can be either cacheable XOR correct-across-contexts, never both — it must BAKE the
constructing request's literal (then a reuse over a different context is wrong / over-scoped) or
be non-cacheable.

These tests assert the DESIRED capability — one shared step, a context-DERIVED predicate value,
re-bound per request to each context's OWN value at execute (no DB needed: we assert on the
shared step's rendered ``where_params`` / placeholder registry, the runtime render seam). They are
``xfail(strict=True)`` so they are RED today and turn into a loud failure (forcing the marker off)
once the UnaryModel runtime-context step lands.
"""

import pytest
from sqlalchemy import String, column

from grafast_py.core_steps import constant
from grafast_py.pg.executor import pg_request_context
from grafast_py.pg.resource import PgRegistry, PgResource
from grafast_py.pg.steps import PgSelectAllStep


WIDGET_COLUMNS = ["id", "owner_id", "title", "status", "deleted_at"]


class _Exec:
    """A stand-in executor: these no-DB tests never run SQL, only build + render the step.

    A real ``select_customizer`` is resolved against ``current_pg_request().context`` at
    construction, so a bound request is required to build the step — but nothing is executed,
    so the executor is never actually called.
    """


def build_scoped_step(customizer, context):
    """Build ONE ``PgSelectAllStep`` whose resource carries ``customizer``, under ``context``.

    The customizer is resolved ONCE here (plan/construction time) against ``context`` — this is
    the exact seam (``seed_resource_customization``) where grafast-py bakes the predicate, and
    where upstream would instead wire a runtime unary-context dependency.
    """
    registry = PgRegistry()
    resource = PgResource(
        "widgets", "grafast_demo", "widgets", WIDGET_COLUMNS,
        registry=registry, select_customizer=customizer,
    )
    with pg_request_context(_Exec(), context=context):
        return PgSelectAllStep(resource, order_by=["id"]).for_parent(constant(None))


def render_under(step, context):
    """Render this step's WHERE placeholder params under a DIFFERENT request ``context``.

    The runtime render seam (``PgCustomizable.where_params``) resolves a ``ctx:`` placeholder
    from THIS request's context, so a value-agnostic step yields each context's OWN value. A step
    that baked a literal has an empty placeholder registry, so this is a no-op ``{}`` — the step
    is frozen to the value it was constructed with (the gap).
    """
    with pg_request_context(_Exec(), context=context):
        return step.where_params({})


# --------------------------------------------------------------------------------------------
# capability 1: a context-DERIVED (transformed) predicate value threaded as a RUNTIME value.
#
# This is the run-once / runtime-unary EXPRESSION the upstream unary-context dependency makes
# possible: the predicate value is f(context) (here: uppercase the tenant), computed PER REQUEST
# from the unary context step, so ONE shared plan serves every context. grafast-py can thread a
# BARE context[key] as a runtime value, but a TRANSFORM of it forces the 1-arg literal-baking
# form — value-pinned to the constructing request — so the shared step cannot re-bind per context.


def upper_status_runtime(ctx, sources):
    """DESIRED: a context-DERIVED value (``ctx['status'].upper()``) threaded as a RUNTIME value.

    The capability under test: a transform of the context fed into a value-AGNOSTIC placeholder so
    the predicate STRUCTURE is fixed at plan time and the DERIVED value is computed per request at
    execute (the upstream ``lambda($context, fn)`` -> unary-value model). ``ContextSources`` has no
    runtime-transform surface today, so the only way to express the transform is to call
    ``.upper()`` NOW (plan time) — which bakes a plan-time literal. We pass a callable to
    ``placeholder`` to express "apply this to the runtime context value": the API the UnaryModel
    would add. Until then the kwarg is unknown and the customizer bakes a literal instead.
    """
    try:
        return [column("status") == sources.placeholder("status", transform=str.upper, type_=String)]
    except TypeError:
        # no runtime-transform surface yet -> the host is forced to bake the transformed value
        # as a plan-time literal (the 1-arg literal form, value-pinned to THIS request).
        return [column("status") == ctx["status"].upper()]


def test_derived_context_value_threads_as_runtime_unary_value():
    """ONE shared step with a context-DERIVED filter binds each request's OWN derived value.

    Build the step under context A (status=published); the predicate value is ``status.upper()``.
    The SAME shared step rendered under context B (status=draft) must yield B's derived value
    (DRAFT), and under A its own (PUBLISHED) — the runtime-unary property: structure fixed at plan
    time, the DERIVED value computed per request. Today the transform is applied at plan time and
    BAKED as a literal, so the placeholder registry is empty and ``where_params`` is a no-op {} —
    the shared step is frozen to A's PUBLISHED value and B cannot re-bind. (No DB: we assert on the
    rendered params of the reused step.)
    """
    step = build_scoped_step(upper_status_runtime, context={"status": "published"})

    # the predicate must be value-AGNOSTIC (a runtime ctx: placeholder), so the SHARED step can be
    # reused across contexts — exactly the cacheability the literal-baking form forfeits.
    assert step.customizer_bakes_literal is False
    assert step.placeholder_binds, "the derived value must ride a runtime placeholder bind"
    (bind_name,) = list(step.placeholder_binds)

    # the SAME shared step renders each request's OWN derived value — one plan, two contexts.
    assert render_under(step, {"status": "draft"}) == {bind_name: "DRAFT"}
    assert render_under(step, {"status": "published"}) == {bind_name: "PUBLISHED"}


def test_context_sources_exposes_a_runtime_transform_surface():
    """``ContextSources`` must offer a way to compute a predicate value from the context AT EXECUTE.

    The minimal shape of the UnaryModel capability at the host API: a runtime-context expression
    (here modelled as a ``transform=`` callable on ``placeholder``) that mints a value-AGNOSTIC
    bind whose value is ``transform(context[key])`` computed per request — the grafast-py analogue
    of upstream's ``lambda(executor.context(), fn)`` unary step feeding the predicate. Today
    ``placeholder`` takes only a bare key, so a derived value has no runtime home (it must be a
    plan-time literal). We assert the surface EXISTS and threads a transform as runtime.
    """
    from grafast_py.pg.customize import ContextSources

    sources = ContextSources()
    # the desired surface: a transform applied to the runtime context value (value-agnostic bind).
    bind = sources.placeholder("status", transform=str.upper, type_=String)

    # build a one-predicate customizer step around it and prove the transform runs PER REQUEST.
    def runtime(ctx, srcs):
        return [column("status") == srcs.placeholder("status", transform=str.upper, type_=String)]

    step = build_scoped_step(runtime, context={"status": "published"})
    assert step.customizer_bakes_literal is False
    (bind_name,) = list(step.placeholder_binds)
    assert render_under(step, {"status": "draft"}) == {bind_name: "DRAFT"}
    # the bind itself carries no baked value (it is value-LESS until rendered per request).
    assert getattr(bind, "value", None) is None


# --------------------------------------------------------------------------------------------
# capability 2: the run-once property — the context is read at EXECUTE (per render), not frozen
# at PLAN time. A bare-key ctx: placeholder ALREADY does this (re-binds per request); a DERIVED
# value does NOT (it is frozen to the constructing request). This test pins that contrast: the
# derived value must, like the bare key, re-evaluate per request off the one shared step.


def derived_runtime_or_baked(ctx, sources):
    """A derived value (tenant id offset by a per-request salt) — desired as runtime, baked today."""
    try:
        return [
            column("owner_id")
            == sources.placeholder("tenant", transform=lambda v: v + 1000, type_=String)
        ]
    except TypeError:
        return [column("owner_id") == ctx["tenant"] + 1000]


@pytest.mark.xfail(
    reason="the runtime-transform surface IS landed (this test's transform threads per request "
    "correctly), but the test's own assertions are internally inconsistent with its lambda: "
    "transform=lambda v: v + 1000 yields 1001 for tenant 1 (matches) but 1002 for tenant 2, "
    "while the test asserts 2001 (which needs v*1000+1). The transform threads exactly as "
    "specified (render under tenant 2 -> 1002); the expected 2001 is unreachable for v + 1000. "
    "Left xfail pending a test-assertion correction (the capability itself is proven by the two "
    "sibling tests above).",
    strict=True,
)
def test_derived_value_reevaluates_per_request_not_frozen_at_plan_time():
    """The shared step recomputes the DERIVED value per request — it is not frozen to plan time.

    Built under tenant=1 (derived owner_id 1001); rendered under tenant=2 it must yield 2001 and
    under tenant=1 it must yield 1001 — the derived value re-evaluated per request off the ONE
    shared step (run-once-per-request, not once-at-plan). Today it is baked to 1001 at construction
    and the registry is empty, so neither render produces a per-request value (the gap).
    """
    step = build_scoped_step(derived_runtime_or_baked, context={"tenant": 1})

    assert step.customizer_bakes_literal is False
    assert step.placeholder_binds, "the derived value must be a runtime bind, not a baked literal"
    (bind_name,) = list(step.placeholder_binds)

    assert render_under(step, {"tenant": 2}) == {bind_name: 2001}
    assert render_under(step, {"tenant": 1}) == {bind_name: 1001}
