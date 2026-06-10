"""The CONSTRAINT-KEYED machine: the two production-safe pillars that DID land.

The sibling :mod:`tests.test_cache_context_isolation` proves the end-to-end cache-HIT path: a
context-scoped step built with a value-LESS ``ctx:`` placeholder is served from request 1's cache
to request 2 and re-binds to request 2's context via the render seam (``where_params`` /
``member_where_params``), so it is cacheable AND correct across contexts. This file pins the two
under-the-hood pillars that path relies on, exercised in isolation:

  * the union runtime ctx: rebind — :meth:`PgUnionAllStep.member_where_params` resolves a
    ``ctx:`` placeholder (and any ``transform=``) per request from the request context, so a
    union member that scopes the cache-SAFE way (a ``sources.placeholder('owner_id')`` bind, not
    a hand-baked literal) re-binds to each request's OWN context value off the SHARED step; and
  * the optimization-INDEPENDENT constraint guard — a customizer's value-agnostic predicate-shape
    signature, captured at store time over the PRE-optimization step set and re-validated on every
    hit (:func:`grafast_py.plan.constraints_match`), so a structural divergence is a MISS even for
    a customizer-bearing step that dedup-merged or tree-shook out of ``plan.steps``.
"""

from sqlalchemy import String, column

from grafast_py.pg.customize import (
    CustomizerConstraint,
    placeholder_binds_in,
    predicate_key,
    resolve_customizer_predicates,
)
from grafast_py.pg.executor import pg_request_context
from grafast_py.pg.placeholders import pg_placeholder
from grafast_py.pg.resource import PgRegistry, PgResource
from grafast_py.pg.union import PgUnionAllStep, PgUnionMember
from grafast_py.plan import constraints_match


class _Exec:
    """A stand-in executor — these no-DB tests build + render steps, never run SQL."""


def make_articles() -> PgResource:
    return PgResource(
        "articles", "grafast_demo", "articles",
        ["id", "owner_id", "created"], registry=PgRegistry(),
    )


# ---------------------------------------------------- union runtime ctx: rebind (the landed half)


def test_union_member_ctx_placeholder_rebinds_per_request():
    """One SHARED union, a ctx: member predicate, rebinds to each request's OWN context value.

    The cache-SAFE way to scope a union member: a value-LESS ``ctx:owner_id`` placeholder, NOT a
    baked literal. ``member_where_params`` resolves it per request from the request context, so the
    SAME shared step serves owner_id 2 and owner_id 7 — the runtime-unary property the deepcopy-free
    cache relies on (structure fixed at plan time, value per request).
    """
    member = PgUnionMember(
        make_articles(), "Article",
        where=[column("owner_id") == pg_placeholder("ctx:owner_id", type_=String)],
    )
    union = PgUnionAllStep(
        [member], shared_columns=["id", "owner_id", "created"],
        order_by=["created"], first=3,
    )

    with pg_request_context(_Exec(), context={"owner_id": 2}):
        params_a = union.member_where_params({})
    with pg_request_context(_Exec(), context={"owner_id": 7}):
        params_b = union.member_where_params({})

    assert list(params_a.values()) == [2]
    assert list(params_b.values()) == [7]
    # same bind name across requests (one shared step), only the resolved value differs.
    assert set(params_a) == set(params_b)


def test_union_member_ctx_transform_computes_derived_value_per_request():
    """A context-DERIVED union member value (``transform=``) is computed per request at render.

    The runtime-transform surface threaded through the union: a member predicate whose value is
    ``transform(context['owner_id'])`` rides a value-AGNOSTIC bind, computed per request, so the
    shared union serves each context its OWN derived value (no plan-time-baked literal).
    """
    member = PgUnionMember(
        make_articles(), "Article",
        where=[
            column("owner_id")
            == pg_placeholder("ctx:owner_id", type_=String, transform=lambda v: v + 100)
        ],
    )
    union = PgUnionAllStep(
        [member], shared_columns=["id", "owner_id", "created"],
        order_by=["created"], first=3,
    )

    with pg_request_context(_Exec(), context={"owner_id": 2}):
        assert list(union.member_where_params({}).values()) == [102]
    with pg_request_context(_Exec(), context={"owner_id": 5}):
        assert list(union.member_where_params({}).values()) == [105]


def test_union_member_var_transform_is_applied_at_render():
    """A ``var:`` union member placeholder with ``transform=`` binds the TRANSFORMED variable
    value at render — not only ``ctx:`` sources.

    The union render seam (:meth:`PgUnionAllStep.member_where_params`) must apply ``transform=``
    for a ``var:`` source too; otherwise a transformed variable placeholder binds the RAW variable
    value, and (now that the dedup key discriminates by transform) a transformed vs untransformed
    bind would carry the same value despite distinct keys. A var:-only union needs no request
    context. Regression for the var:-source transform render gap.
    """
    member = PgUnionMember(
        make_articles(), "Article",
        where=[
            column("status")
            == pg_placeholder("var:status", "x", type_=String, transform=str.upper)
        ],
    )
    union = PgUnionAllStep(
        [member], shared_columns=["id", "owner_id", "created"],
        order_by=["created"], first=3,
    )

    assert list(union.member_where_params({"var:status": "published"}).values()) == [
        "PUBLISHED"
    ]


# ----------------------------------------------- optimization-independent constraint guard


def context_branching_customizer(ctx, sources):
    """A customizer whose predicate STRUCTURE branches on context (admin: none; user: scoped)."""
    if ctx.get("admin"):
        return []
    return [column("owner_id") == sources.placeholder("owner_id", type_=String)]


def build_constraint(context) -> CustomizerConstraint:
    """Capture the structural constraint the customizer resolves to under ``context``."""
    predicates, _ = resolve_customizer_predicates(
        context_branching_customizer, context, 2
    )
    keys = tuple(
        predicate_key(p, placeholder_binds_in(p) or None) for p in predicates
    )
    return CustomizerConstraint(context_branching_customizer, 2, keys)


def test_constraint_value_only_change_still_matches():
    """A value-only context change keeps the SAME predicate shape, so the constraint MATCHES.

    Captured under one user context; re-validated under a DIFFERENT user context (a different
    owner_id) — the placeholder rides the value, so the value-agnostic shape is identical and the
    cache HIT is correct (one shared plan serves both users).
    """
    constraint = build_constraint({"owner_id": 1, "admin": False})
    with pg_request_context(_Exec(), context={"owner_id": 9, "admin": False}):
        assert constraint.matches() is True
        assert constraints_match((constraint,)) is True


def test_constraint_structural_divergence_is_a_miss():
    """A STRUCTURAL context change (user-scoped -> admin no-filter) makes the constraint a MISS.

    The leak class upstream closes: a request whose customizer resolves to a DIFFERENT predicate
    shape must not inherit the cached structure. The constraint re-resolves the customizer under
    the admin context (no filter) and the shape diverges, so the hit is rejected (re-plan).
    """
    constraint = build_constraint({"owner_id": 1, "admin": False})
    with pg_request_context(_Exec(), context={"owner_id": 9, "admin": True}):
        assert constraint.matches() is False
        assert constraints_match((constraint,)) is False


def transform_customizer(ctx, sources):
    """A value-only customizer scoping by a DERIVED (``transform=``) context value."""
    return [
        column("owner_id")
        == sources.placeholder("owner_id", type_=String, transform=lambda v: v + 100)
    ]


def test_constraint_with_transform_customizer_still_matches_on_reinvocation():
    """A customizer using a ``transform=`` placeholder must still MATCH on a cache-hit re-validation.

    ``matches()`` re-invokes the customizer, minting a FRESH transform lambda each time. The dedup
    key must be cache-STABLE (code-object based, not ``id``) so the recomputed key equals the
    stored one — otherwise every request using a transform customizer is a false MISS and
    ``cache_plans`` silently degrades to a re-plan. Regression for the id()-instability cache
    degradation (this is the exact ``CustomizerConstraint.matches()`` path that re-resolves the
    customizer on a hit).
    """
    predicates, _ = resolve_customizer_predicates(transform_customizer, {"owner_id": 1}, 2)
    keys = tuple(
        predicate_key(p, placeholder_binds_in(p) or None) for p in predicates
    )
    constraint = CustomizerConstraint(transform_customizer, 2, keys)
    with pg_request_context(_Exec(), context={"owner_id": 9}):
        assert constraint.matches() is True
        assert constraints_match((constraint,)) is True


def test_empty_constraint_list_matches_trivially():
    """A plan with no context-scoping customizer carries no constraints and always matches."""
    assert constraints_match(()) is True
