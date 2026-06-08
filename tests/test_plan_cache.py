"""Cross-request plan cache: the bounded-LRU core module + its no-regression / anti-corruption
gates.

The reuse layer builds on the opt-in flags, the ``FieldArgs`` variable provenance, the
plan-level flag threading, and the pg placeholder surfaces (WHERE + pagination):
:mod:`grafast_py.cache` — a bounded-LRU process cache keyed by ``(id(schema),
document-text hash, operation name, variable-arg fingerprint)`` storing the finalized
ObjectPlan + RootStep + Plan, served as the SHARED entry on a hit (DEEPCOPY-FREE) with a
per-request SOURCE MAP rendered into the compiled statement's params — wired into
``plan_operation`` behind ``GrafastConfig.cache_plans`` (default off).

This module gates:

  * the cache module itself (LRU store/get/eviction, key stability/discrimination, fingerprint);
  * the CACHEABILITY guard (a plan that inlined a variable LITERAL is NOT cached — reusing it
    would serve the wrong value);
  * the RENDER-INJECTION correctness (a cached step's value-less placeholders resolve to THIS
    request's value from the per-request source map at SQL-render time);
  * the SHARED-ENTRY IDENTITY proof (a cache HIT reuses the cached plan object by identity — no
    deepcopy is taken);
  * the NO-REGRESSION oracle (same query cached-off vs cached-on -> byte-identical result), plus
    the plan-build-SKIP proof (the second request of a document does NOT re-plan);
  * the ANTI-CORRUPTION gate end-to-end over the DB: the SAME cached value-agnostic plan, run
    with TWO different variable values, returns each its OWN correct rows (a cache hit must
    never serve one request the value of another — the worst failure mode in the project),
    INCLUDING an aggressive BARRIER-SYNCHRONIZED concurrent no-bleed backstop (the deepcopy-free
    cache shares one step DAG across concurrent requests, so a missed relocation would BLEED).

DB tests are marked ``pg`` (deselectable ``-m 'not pg'``); they touch ONLY the ``grafast_demo``
``widgets`` fixture and alter nothing else.
"""

import asyncio

import pytest
import pytest_asyncio
from graphql import graphql, graphql_sync, parse
from sqlalchemy import Integer, String, Text, column

import grafast_py.plan as plan_module
from grafast_py.cache import (
    CachedPlan,
    PlanCache,
    compute_cache_key,
    config_fingerprint,
    default_cache,
    document_text,
    values_by_source,
    variable_arg_fingerprint,
)
from grafast_py.config import GrafastConfig
from grafast_py.context import GrafastExecutionContext
from grafast_py.core_steps import access, constant
from grafast_py.dag import Plan
from grafast_py.pg.engine import count_sql, dispose_engine, get_engine
from grafast_py.pg.executor import SQLAlchemyExecutor, pg_request_context
from grafast_py.pg.placeholders import Placeholder, pg_placeholder
from grafast_py.pg.resource import PgRegistry, PgResource
from grafast_py.pg.steps import PgSelectAllStep, PgSelectStep
from grafast_py.steps import BucketExtra
from examples.seed import setup_demo_schema, setup_widgets_table


# --------------------------------------------------------------- the PlanCache (no DB)


def make_cached(tag: str) -> CachedPlan:
    """A throwaway CachedPlan whose ``object_plan`` is a sentinel string (identity only)."""
    return CachedPlan(object_plan=tag, root_step=None, plan=Plan(), schema=None)


def test_cache_store_and_get_roundtrips():
    """A stored entry is returned by its key; an absent key returns None."""
    cache = PlanCache()
    entry = make_cached("a")
    cache.put(("k",), entry)
    assert cache.get(("k",)) is entry
    assert cache.get(("missing",)) is None
    assert cache.hits == 1 and cache.misses == 1


def test_cache_lru_evicts_least_recently_used():
    """Past the cap the LEAST-recently-used entry is evicted; a GET refreshes recency."""
    cache = PlanCache(max_entries=2)
    cache.put(("a",), make_cached("a"))
    cache.put(("b",), make_cached("b"))
    # touch "a" so "b" becomes the LRU
    assert cache.get(("a",)) is not None
    cache.put(("c",), make_cached("c"))  # over cap -> evict the LRU ("b")
    assert cache.get(("b",)) is None
    assert cache.get(("a",)) is not None
    assert cache.get(("c",)) is not None
    assert cache.evictions == 1
    assert len(cache) == 2


def test_cache_rejects_zero_cap():
    """A cache must hold at least one entry; a non-positive cap fails loud."""
    with pytest.raises(ValueError):
        PlanCache(max_entries=0)


def test_cache_clear_empties():
    cache = PlanCache()
    cache.put(("a",), make_cached("a"))
    cache.clear()
    assert len(cache) == 0
    assert cache.get(("a",)) is None


# --------------------------------------------------------------- the cache KEY (no DB)


SDL = """
type Query {
  things(status: String, limit: Int): [Thing!]!
}
type Thing {
  id: Int!
}
"""


def first_operation(query: str):
    return parse(query).definitions[0]


def test_cache_key_same_document_is_stable():
    """Two re-parses of the SAME query text produce the SAME key (id() is not used)."""
    schema = object()
    op1 = first_operation("query Q($s: String) { things(status: $s) { id } }")
    op2 = first_operation("query Q($s: String) { things(status: $s) { id } }")
    assert compute_cache_key(schema, op1) == compute_cache_key(schema, op2)
    # ...and the operation nodes are genuinely distinct objects (so id() would have differed).
    assert op1 is not op2


def test_cache_key_differs_by_schema_identity():
    """Two schema instances key differently (a host serving several schemas)."""
    op = first_operation("{ things(status: \"x\") { id } }")
    assert compute_cache_key(object(), op) != compute_cache_key(object(), op)


def test_cache_key_differs_by_operation_name():
    """A different operation name selects a different entry (multi-operation document)."""
    schema = object()
    a = first_operation("query A { things(status: \"x\") { id } }")
    b = first_operation("query B { things(status: \"x\") { id } }")
    assert compute_cache_key(schema, a) != compute_cache_key(schema, b)


def test_cache_key_differs_by_literal_vs_variable_structure():
    """A literal arg and a $variable arg of the same field key DIFFERENTLY.

    The variable-arg fingerprint captures the literal-vs-$var structure, so a value-pinned
    literal plan can never collide with a value-agnostic placeholder plan in the cache.
    """
    schema = object()
    literal = first_operation("{ things(status: \"x\") { id } }")
    variable = first_operation("query Q($s: String) { things(status: $s) { id } }")
    assert compute_cache_key(schema, literal) != compute_cache_key(schema, variable)


def test_variable_arg_fingerprint_walks_the_whole_operation():
    """The fingerprint is the sorted (field-path, arg-name) pairs of every $variable arg."""
    op = first_operation(
        "query Q($s: String, $n: Int) { things(status: $s, limit: $n) { id } }"
    )
    fp = variable_arg_fingerprint(op)
    assert fp == (("things", "limit"), ("things", "status"))
    # an all-literal operation has an empty fingerprint
    lit = first_operation("{ things(status: \"x\", limit: 3) { id } }")
    assert variable_arg_fingerprint(lit) == ()


def test_document_text_folds_in_fragments():
    """The document text includes referenced fragment bodies, so a fragment change re-keys."""
    op = first_operation("query Q { things(status: \"x\") { ...F } }")
    frag_a = parse("fragment F on Thing { id }").definitions[0]
    text_a = document_text(op, {"F": frag_a})
    assert "fragment F" in text_a and "things" in text_a


def test_cache_key_differs_by_plan_affecting_config():
    """A plan-affecting config field re-keys, so two configs sharing the default cache never collide.

    ``placeholders`` / ``cache_plans`` / ``inline_relations`` change the SHAPE of the planned
    DAG (or whether a resolver placeholders vs inlines), so a plan built under one combination
    must not be served to a request under another (the cross-config bleed). The limit/tracing
    knobs do NOT change the plan, so they SHARE an entry.
    """
    schema = object()
    op = first_operation("query Q($s: String) { things(status: $s) { id } }")
    a = GrafastConfig(placeholders=True, cache_plans=True)
    b = GrafastConfig(placeholders=False, cache_plans=True)
    c = GrafastConfig(placeholders=True, cache_plans=True, inline_relations=True)
    # a config-affecting field re-keys (no cross-config collision on the shared default cache).
    assert compute_cache_key(schema, op, None, a) != compute_cache_key(schema, op, None, b)
    assert compute_cache_key(schema, op, None, a) != compute_cache_key(schema, op, None, c)
    # the SAME config keys identically (so same-config repeats still HIT).
    a2 = GrafastConfig(placeholders=True, cache_plans=True)
    assert compute_cache_key(schema, op, None, a) == compute_cache_key(schema, op, None, a2)
    # a NON-plan-affecting knob (timeout) does NOT re-key — those configs share an entry.
    d = GrafastConfig(placeholders=True, cache_plans=True, execution_timeout_s=5.0)
    assert compute_cache_key(schema, op, None, a) == compute_cache_key(schema, op, None, d)
    # hoist changes the finalized LayerPlan run_steps/boundary, so it re-keys too — else a
    # hoist=True request could be served a non-hoisted cached plan (or vice-versa).
    e = GrafastConfig(placeholders=True, cache_plans=True, hoist=True)
    assert compute_cache_key(schema, op, None, a) != compute_cache_key(schema, op, None, e)
    # the fingerprint is exactly the four plan-affecting flags, in (inline, placeholders, cache, hoist) order.
    assert config_fingerprint(a) == (False, True, True, False)
    assert config_fingerprint(None) == (False, False, False, False)
    assert config_fingerprint(e) == (False, True, True, True)


def test_two_configs_sharing_default_cache_do_not_bleed():
    """END-TO-END: two configs on ONE schema (both default cache) never serve each other's plan.

    The cross-config bleed this guards: if ``compute_cache_key`` carried no config component, a
    plan built under config A and a config-B request of the SAME document would collide on one
    default-cache entry — B would get a HIT and be served A's plan, its own ``is_variable`` plan
    resolver bypassed. The config fingerprint is folded into the key, so B's lookup is a MISS (it
    plans its own).
    """
    from grafast_py.schema import make_grafast_schema

    schema = make_grafast_schema(
        SDL, {"Query": {"things": things_placeholder_plan}, "Thing": {"id": id_plan}}
    )
    default_cache().clear()
    query = "query Q($s: String) { things(status: $s) { id } }"
    a = plan_query(schema, query, GrafastConfig(placeholders=True, cache_plans=True), {"s": "x"})
    hits_after_a = default_cache().hits
    # a DIFFERENT config of the same schema/document must MISS (not be served A's plan).
    plan_query(schema, query, GrafastConfig(placeholders=False, cache_plans=True), {"s": "x"})
    assert default_cache().hits == hits_after_a  # config B was a miss, no cross-config hit
    # the SAME config of the same document IS a hit (caching still works within a config).
    plan_query(schema, query, GrafastConfig(placeholders=True, cache_plans=True), {"s": "x"})
    assert default_cache().hits == hits_after_a + 1
    default_cache().clear()
    assert a is not None


# --------------------------------------------------------------- values_by_source (no DB)


def test_values_by_source_prefixes_var_tags():
    """A request's {variable: value} becomes {source-tag: value} for the rebind."""
    assert values_by_source({"s": "published", "n": 5}) == {
        "var:s": "published",
        "var:n": 5,
    }
    assert values_by_source(None) == {}
    assert values_by_source({}) == {}


def test_values_by_source_covers_omitted_variables_as_none():
    """An OMITTED (no-default) declared variable maps to None, not absent.

    So a cache HIT re-points it to None rather than inheriting the PRIOR request's value (the
    omitted-no-default correctness gap). A defaulted-but-omitted variable is already folded into
    ``variable_values`` by graphql-core, so it carries its default here.
    """
    op = first_operation("query Q($s: String, $n: Int) { things(status: $s, limit: $n) { id } }")
    # only `s` supplied; `n` omitted with no default -> None
    mapping = values_by_source({"s": "published"}, op)
    assert mapping == {"var:s": "published", "var:n": None}


# --------------------------------------- render-injection correctness (deepcopy-free, no DB)


def make_widgets() -> PgResource:
    registry = PgRegistry()
    return PgResource(
        "widgets",
        "grafast_demo",
        "widgets",
        ["id", "owner_id", "title", "status", "deleted_at"],
        registry=registry,
    )


def test_where_placeholder_is_value_less_and_resolves_from_source_map():
    """A cached WHERE placeholder bind is value-LESS; its value is rendered from the source map.

    The deepcopy-free model: the shared step holds NO request value (the ``pg_placeholder`` bind
    is value-less). Two different source maps render two different param values from the SAME
    shared step, and the dedup key (value-agnostic, source-keyed) is unchanged by either render.
    """
    step = PgSelectStep(make_widgets(), constant(None), "owner_id", order_by=["id"])
    step.builder().where(column("status") == pg_placeholder("var:status", "draft"))
    key = (step.peer_key, step.dedup_params())

    ph_bind = step.where_predicates[0].right
    assert ph_bind.value is None  # value-LESS — no request value lives on the shared bind
    name = ph_bind.key

    # request A and request B render their OWN value from the SHARED step — no copy, no bleed.
    assert step.where_params({"var:status": "draft"}) == {name: "draft"}
    assert step.where_params({"var:status": "published"}) == {name: "published"}
    # the dedup key never depended on the value, so neither render invalidates it.
    assert (step.peer_key, step.dedup_params()) == key


def test_pagination_placeholder_is_value_less_and_resolves_from_source_map():
    """A cached ``first`` Placeholder is value-LESS; its value resolves from the source map."""
    step = PgSelectStep(
        make_widgets(), constant(None), "owner_id", order_by=["id"],
        first=Placeholder("var:n"),
    )
    key = (step.peer_key, step.dedup_params())

    assert isinstance(step.first, Placeholder)
    assert step.first.source == "var:n"
    assert not hasattr(step.first, "value")  # value-LESS sentinel (only the source tag lives on)
    # the root LIMIT placeholder renders its value per request from the source map.
    root = PgSelectAllStep(
        make_widgets(), order_by=["id"], first=Placeholder("var:n"),
    ).for_parent(constant(None))
    assert root.run_params({"var:n": 9}) == {"root_first": 9}
    assert root.run_params({"var:n": 2}) == {"root_first": 2}
    # the dedup key is unchanged by the value-less sentinel.
    assert (step.peer_key, step.dedup_params()) == key


def test_where_params_is_empty_without_placeholders():
    """A literal-only step has no placeholder binds, so ``where_params`` is a byte-identical no-op."""
    step = PgSelectStep(make_widgets(), constant(None), "owner_id", order_by=["id"])
    step.builder().where(column("status") == "published")  # a literal, not a placeholder
    assert step.where_params({"var:status": "draft"}) == {}
    # the literal value is baked on its own auto-named bind, untouched by the source map.
    assert step.where_predicates[0].right.value == "published"


# -------------------------------- shared-entry identity (deepcopy-free, gate 9, no DB)


def test_cache_hit_returns_the_shared_plan_object(monkeypatch):
    """A cache HIT reuses the cached plan object BY IDENTITY — proving the deepcopy is gone.

    The deepcopy-free property (gate 9): ``lookup_cached_plan`` stashes the SHARED cached triple
    on the context (``context._grafast_plan IS cached.plan``), not a per-request copy. We capture
    what ``store_cached_plan`` put in the cache on the MISS, then capture the second request's
    ``context._grafast_plan`` on the HIT and assert they are the IDENTICAL objects (an ``is``
    check no deepcopy could pass). The per-request SOURCE MAP differs, but the plan is shared.
    """
    from grafast_py.schema import make_grafast_schema

    schema = make_grafast_schema(
        SDL, {"Query": {"things": things_placeholder_plan}, "Thing": {"id": id_plan}}
    )
    cache = PlanCache()
    config = GrafastConfig(placeholders=True, cache_plans=True, plan_cache=cache)

    stored = {}
    real_store = plan_module.store_cached_plan

    def spy_store(context, operation, cfg, object_plan, root_step, plan):
        stored["object_plan"] = object_plan
        stored["plan"] = plan
        stored["root_step"] = root_step
        return real_store(context, operation, cfg, object_plan, root_step, plan)

    monkeypatch.setattr(plan_module, "store_cached_plan", spy_store)

    captured = {}
    real_lookup = plan_module.lookup_cached_plan

    def spy_lookup(context, operation, cfg):
        result = real_lookup(context, operation, cfg)
        if result is not None:  # a HIT — capture what was stashed on the context
            captured["object_plan"] = result
            captured["plan"] = context._grafast_plan
            captured["root_step"] = context._grafast_root_step
        return result

    monkeypatch.setattr(plan_module, "lookup_cached_plan", spy_lookup)

    query = "query Q($s: String) { things(status: $s) { id } }"
    # request 1 MISSes and stores the shared triple; request 2 HITs and reuses it by identity.
    graphql_sync(schema, query, variable_values={"s": "x"},
                 execution_context_class=context_class_with(config))
    graphql_sync(schema, query, variable_values={"s": "y"},
                 execution_context_class=context_class_with(config))

    assert cache.hits == 1 and cache.misses == 1
    # the HIT reused the SHARED objects — no deepcopy was taken (the gate-9 identity proof).
    assert captured["object_plan"] is stored["object_plan"]
    assert captured["plan"] is stored["plan"]
    assert captured["root_step"] is stored["root_step"]


# --------------------------------------------------------------- cacheability guard


def things_inlining_plan(parent, args, info):
    """A plan that INLINES the variable arg value as a literal (the non-cacheable path)."""
    rows = [{"id": 1, "status": args["status"]}]  # reading args["status"] -> a literal read
    return constant(rows)


def things_placeholder_plan(parent, args, info):
    """A plan that wraps the variable arg as a value-agnostic placeholder (the cacheable path)."""
    # it does NOT read the raw value; it only asks for the source tag, so the plan stays
    # value-independent. (A no-DB constant stands in for a real pg select here.)
    if args.is_variable("status"):
        _ = args.source("status")
    return constant([{"id": 1}])


def id_plan(parent, args, info):
    return access(parent, ("id",))


def context_class_with(config: GrafastConfig):
    class _Ctx(GrafastExecutionContext):
        grafast_config = config

    return _Ctx


def plan_query(schema, query, config, variables):
    """Plan one operation under ``config`` and return its Plan (the freshly-built one)."""
    from graphql.execution.collect_fields import collect_fields
    from grafast_py.plan import plan_operation

    document = parse(query)
    operation = document.definitions[0]
    ctx = context_class_with(config).build(
        schema, document, raw_variable_values=variables or {}
    )
    root_fields = collect_fields(
        ctx.schema, ctx.fragments, ctx.variable_values, schema.query_type,
        operation.selection_set,
    )
    plan_operation(ctx, operation, schema.query_type, root_fields)
    return ctx._grafast_plan


def test_plan_inlining_a_variable_literal_is_not_cacheable():
    """A plan that INLINED a $variable value as a literal is marked NON-cacheable.

    Reusing it across requests would serve a later request the earlier value, so it must not
    be cached. ``FieldArgs`` records the raw read; the planner flips ``plan.cacheable`` False.
    """
    from grafast_py.schema import make_grafast_schema

    schema = make_grafast_schema(
        SDL, {"Query": {"things": things_inlining_plan}, "Thing": {"id": id_plan}}
    )
    plan = plan_query(
        schema,
        "query Q($s: String) { things(status: $s) { id } }",
        GrafastConfig(placeholders=True, cache_plans=True),
        {"s": "published"},
    )
    assert plan.cacheable is False


def test_plan_using_a_placeholder_stays_cacheable():
    """A plan that wrapped the $variable as a placeholder (read source(), not the value) is cacheable."""
    from grafast_py.schema import make_grafast_schema

    schema = make_grafast_schema(
        SDL, {"Query": {"things": things_placeholder_plan}, "Thing": {"id": id_plan}}
    )
    plan = plan_query(
        schema,
        "query Q($s: String) { things(status: $s) { id } }",
        GrafastConfig(placeholders=True, cache_plans=True),
        {"s": "published"},
    )
    assert plan.cacheable is True


def test_inlining_a_variable_is_not_cacheable_even_with_placeholders_off():
    """cache_plans WITHOUT placeholders still detects an inlined $variable -> non-cacheable.

    With placeholders off, provenance would not otherwise be computed, so an inlined variable
    would go unrecorded and the value-pinned plan would be cached — a later request bleeding the
    first value. Caching forces provenance (placeholders OR cache_plans), so the inline is
    detected and the plan re-plans per request.
    """
    from grafast_py.schema import make_grafast_schema

    schema = make_grafast_schema(
        SDL, {"Query": {"things": things_inlining_plan}, "Thing": {"id": id_plan}}
    )
    plan = plan_query(
        schema,
        "query Q($s: String) { things(status: $s) { id } }",
        GrafastConfig(placeholders=False, cache_plans=True),
        {"s": "published"},
    )
    assert plan.cacheable is False


def test_legacy_resolver_field_with_variable_arg_is_not_cacheable():
    """A legacy (no-plan-resolver) field with a $variable arg is non-cacheable under cache_plans.

    The coerced args are frozen on FieldPlan.args and a cache HIT replays them, so a legacy
    resolver reading the $variable would serve a later request the first value — and a legacy
    resolver has no step to re-point. The plan must therefore not be cached (re-plan instead).
    """
    from grafast_py.schema import make_grafast_schema

    # `things` has NO plan resolver -> the legacy graphql-core resolver path.
    schema = make_grafast_schema(SDL, {})
    plan = plan_query(
        schema,
        "query Q($s: String) { things(status: $s) { id } }",
        GrafastConfig(cache_plans=True),
        {"s": "published"},
    )
    assert plan.cacheable is False


def test_all_literal_plan_is_cacheable():
    """A plan with NO variable args is cacheable (reading a literal arg is always value-stable)."""
    from grafast_py.schema import make_grafast_schema

    schema = make_grafast_schema(
        SDL, {"Query": {"things": things_inlining_plan}, "Thing": {"id": id_plan}}
    )
    plan = plan_query(
        schema,
        '{ things(status: "published") { id } }',
        GrafastConfig(placeholders=True, cache_plans=True),
        {},
    )
    assert plan.cacheable is True


# -------------------------------------- abstract-field cacheability + provenance


ABSTRACT_SDL = """
type Query { item: Node }
interface Node { id: Int! }
type Widget implements Node { id: Int! echo(v: String): String }
type Gadget implements Node { id: Int! }
"""


def build_abstract_schema(echo_plan):
    """A schema whose interface field resolves to Widget, with a $variable-taking echo field."""
    from grafast_py.schema import make_grafast_schema, resolve_type_from_tag

    def item_plan(parent, args, info):
        return constant({"id": 1, "__typename": "Widget"})

    return make_grafast_schema(
        ABSTRACT_SDL,
        {
            "Query": {"item": item_plan},
            "Widget": {"id": id_plan, "echo": echo_plan},
            "Gadget": {"id": id_plan},
        },
        type_resolvers={"Node": resolve_type_from_tag("__typename")},
    )


def test_operation_owning_an_abstract_field_is_not_cacheable():
    """An operation with an interface/union field is NON-cacheable (its subtrees plan lazily).

    An abstract field's per-concrete-type subtree is planned at EXECUTE time and its steps live
    on the completer, beyond the operation-level placeholder rebind's reach — so a cached
    operation would serve a later request the first request's inlined value. The operation
    conservatively refuses to cache when it owns ANY abstract field.
    """
    def echo_plan(parent, args, info):
        return constant(args["v"])

    schema = build_abstract_schema(echo_plan)
    plan = plan_query(
        schema,
        "query Q($v: String) { item { ... on Widget { echo(v: $v) } } }",
        GrafastConfig(placeholders=True, cache_plans=True),
        {"v": "x"},
    )
    assert plan.cacheable is False


def test_abstract_subtree_threads_placeholder_provenance():
    """A $variable arg UNDER a concrete type sees its provenance (is_variable True).

    Without threading ``plan.placeholders`` onto the abstract subtree's own Plan, a field under a
    concrete type would see empty provenance and inline the variable as a literal even with
    placeholders enabled. The flag is threaded so the subtree plans like the operation root.
    """
    seen = {}

    def echo_plan(parent, args, info):
        seen["is_variable"] = args.is_variable("v")
        if args.is_variable("v"):
            _ = args.source("v")
        return constant("ok")

    schema = build_abstract_schema(echo_plan)
    graphql_sync(
        schema,
        "query Q($v: String) { item { ... on Widget { echo(v: $v) } } }",
        variable_values={"v": "x"},
        execution_context_class=context_class_with(
            GrafastConfig(placeholders=True, cache_plans=True)
        ),
    )
    assert seen.get("is_variable") is True


def test_abstract_field_no_cross_request_value_bleed_under_cache():
    """END-TO-END: two requests of an abstract-field op get THEIR OWN value (no bleed).

    The failure mode this guards against: request 1 inlines its $variable under a concrete type,
    the operation is cached as cacheable, and request 2 hits the cache and is served request 1's
    value. With the operation marked non-cacheable, each request re-plans and gets its own value.
    """
    def echo_plan(parent, args, info):
        return constant(args["v"])  # inlines the variable value (raw read)

    schema = build_abstract_schema(echo_plan)
    default_cache().clear()
    hits_before = default_cache().hits  # the counter is cumulative; compare the DELTA
    ctx = context_class_with(GrafastConfig(placeholders=True, cache_plans=True))
    query = "query Q($v: String) { item { ... on Widget { echo(v: $v) } } }"

    r1 = graphql_sync(schema, query, variable_values={"v": "FIRST"}, execution_context_class=ctx)
    r2 = graphql_sync(schema, query, variable_values={"v": "SECOND"}, execution_context_class=ctx)
    assert r1.errors is None and r2.errors is None
    assert r1.data == {"item": {"echo": "FIRST"}}
    assert r2.data == {"item": {"echo": "SECOND"}}  # NOT "FIRST" — no cross-request bleed
    # the operation was never cached (it owns an abstract field), so it was never STORED and
    # nothing was served from the cache — len stays 0 and no NEW hit was recorded.
    assert len(default_cache()) == 0
    assert default_cache().hits == hits_before
    default_cache().clear()


# --------------------------------------------------------------- no-regression (no DB)


def test_cache_off_vs_on_byte_identical_result():
    """NO-REGRESSION: a query's result is identical with caching off vs on (no host placeholder)."""
    from grafast_py.schema import make_grafast_schema

    schema = make_grafast_schema(
        SDL, {"Query": {"things": things_placeholder_plan}, "Thing": {"id": id_plan}}
    )
    query = "query Q($s: String) { things(status: $s) { id } }"
    variables = {"s": "published"}

    off = graphql_sync(
        schema, query, variable_values=variables,
        execution_context_class=context_class_with(GrafastConfig()),
    )
    on = graphql_sync(
        schema, query, variable_values=variables,
        execution_context_class=context_class_with(
            GrafastConfig(placeholders=True, cache_plans=True, plan_cache=PlanCache())
        ),
    )
    assert off.errors is None and on.errors is None
    assert off.data == on.data == {"things": [{"id": 1}]}


def test_second_request_of_a_document_skips_planning(monkeypatch):
    """The second request of a cacheable document is a HIT — finalize_plan does NOT re-run.

    A cache hit changes only WHETHER planning re-runs: we spy on ``finalize_plan`` and assert
    it ran exactly ONCE across two identical requests (the first plans + stores, the second
    hits the cache and skips the build).
    """
    from grafast_py.schema import make_grafast_schema

    schema = make_grafast_schema(
        SDL, {"Query": {"things": things_placeholder_plan}, "Thing": {"id": id_plan}}
    )
    cache = PlanCache()
    config = GrafastConfig(placeholders=True, cache_plans=True, plan_cache=cache)

    calls = []
    real_finalize = plan_module.finalize_plan

    def spy(plan, object_plan):
        calls.append(1)
        return real_finalize(plan, object_plan)

    monkeypatch.setattr(plan_module, "finalize_plan", spy)

    query = "query Q($s: String) { things(status: $s) { id } }"
    for _ in range(2):
        result = graphql_sync(
            schema, query, variable_values={"s": "x"},
            execution_context_class=context_class_with(config),
        )
        assert result.errors is None
        assert result.data == {"things": [{"id": 1}]}

    assert len(calls) == 1  # planned once; the second request hit the cache
    assert cache.hits == 1 and cache.misses == 1


# --------------------------------------------------------------- anti-corruption (DB)


@pytest_asyncio.fixture
async def seeded():
    await dispose_engine()
    await setup_demo_schema()
    await setup_widgets_table()
    yield
    await dispose_engine()


WIDGETS_SDL = """
type Query {
  widgets(status: String!): [Widget!]!
}
type Widget {
  id: Int!
  status: String!
}
"""


def build_widgets_schema():
    """A schema whose ``widgets`` plan builds a value-agnostic placeholder filter on $status.

    When ``status`` came from a variable it wraps the value as a ``pg_placeholder`` (so the
    plan is cacheable and value-agnostic); otherwise it inlines the literal. This is the
    host-declared opt-in surface the cache relies on.
    """
    from grafast_py.schema import make_grafast_schema

    registry = PgRegistry()
    widgets = PgResource(
        "widgets", "grafast_demo", "widgets",
        ["id", "owner_id", "title", "status", "deleted_at"], registry=registry,
    )

    def widgets_plan(parent, args, info):
        # a root collection over ALL widgets, with a uniform WHERE on status as the host's
        # filter. With provenance on AND the arg variable-derived, build a value-agnostic
        # placeholder (cacheable); else inline the literal (value-pinned, not cached).
        step = PgSelectAllStep(widgets, order_by=["id"]).for_parent(parent)
        if args.is_variable("status"):
            step.builder().where(
                column("status") == pg_placeholder(args.source("status"), args["status"])
            )
        else:
            step.builder().where(column("status") == args["status"])
        return step

    def widget_id_plan(parent, args, info):
        return access(parent, ("id",))

    def widget_status_plan(parent, args, info):
        return access(parent, ("status",))

    return make_grafast_schema(
        WIDGETS_SDL,
        {
            "Query": {"widgets": widgets_plan},
            "Widget": {"id": widget_id_plan, "status": widget_status_plan},
        },
    )


@pytest.mark.pg
@pytest.mark.asyncio
@pytest.mark.cache_off
async def test_cached_plan_serves_two_variable_values_correctly(seeded):
    """ANTI-CORRUPTION (the crux end-to-end): one cached value-agnostic plan, TWO values.

    The SAME document is run with ``status="draft"`` then ``status="published"``. The second
    request is a cache HIT that re-binds the placeholder to ITS value — each request must get
    ITS OWN correct rows. Getting this wrong would serve one request the cached/merged plan of
    another (cross-request data corruption — the worst failure mode in the project).

    Marked ``cache_off`` so the suite-wide cache-on oracle does not also drive it (it owns its
    own isolated ``PlanCache`` to assert the hit precisely).
    """
    cache = PlanCache()
    config = GrafastConfig(placeholders=True, cache_plans=True, plan_cache=cache)

    class _Ctx(GrafastExecutionContext):
        grafast_config = config

    schema = build_widgets_schema()
    query = "query Q($s: String!) { widgets(status: $s) { id status } }"

    with pg_request_context(SQLAlchemyExecutor(get_engine())):
        draft = await graphql(
            schema, query, variable_values={"s": "draft"},
            execution_context_class=_Ctx,
        )
    with pg_request_context(SQLAlchemyExecutor(get_engine())):
        published = await graphql(
            schema, query, variable_values={"s": "published"},
            execution_context_class=_Ctx,
        )

    assert draft.errors is None and published.errors is None
    # the second request was a cache HIT (one plan built, reused once).
    assert cache.hits == 1 and cache.misses == 1
    # each request got ITS OWN rows — the anti-corruption property.
    draft_ids = sorted(w["id"] for w in draft.data["widgets"])
    pub_ids = sorted(w["id"] for w in published.data["widgets"])
    assert draft_ids == [2, 5]
    assert pub_ids == [1, 3, 4, 6]
    assert {w["status"] for w in draft.data["widgets"]} == {"draft"}
    assert {w["status"] for w in published.data["widgets"]} == {"published"}


@pytest.mark.pg
@pytest.mark.asyncio
@pytest.mark.cache_off
async def test_cached_plan_has_no_cross_request_bleed_under_concurrency(seeded):
    """ANTI-CORRUPTION under CONCURRENCY: many interleaved cache hits, different values, no bleed.

    The shared-rebind race this guards against: a cache HIT must isolate a per-request copy
    BEFORE re-binding its placeholder, or two concurrent requests of the SAME cached plan with
    DIFFERENT variables overwrite each other's bound value mid-flight.
    Warm the cache once, then fire 16 interleaved requests (alternating status) through
    ``asyncio.gather``; each must return ITS OWN correct rows. A shared in-place rebind would
    surface here as some request seeing another's status rows.
    """
    cache = PlanCache()
    config = GrafastConfig(placeholders=True, cache_plans=True, plan_cache=cache)

    class _Ctx(GrafastExecutionContext):
        grafast_config = config

    schema = build_widgets_schema()
    query = "query Q($s: String!) { widgets(status: $s) { id status } }"
    expected = {"draft": [2, 5], "published": [1, 3, 4, 6]}

    async def one(status: str):
        with pg_request_context(SQLAlchemyExecutor(get_engine())):
            return status, await graphql(
                schema, query, variable_values={"s": status},
                execution_context_class=_Ctx,
            )

    # warm the cache so every concurrent request below is a HIT that must isolate its own copy.
    await one("published")

    statuses = ["draft", "published"] * 8  # 16 interleaved concurrent requests
    results = await asyncio.gather(*(one(s) for s in statuses))

    for status, result in results:
        assert result.errors is None
        ids = sorted(w["id"] for w in result.data["widgets"])
        assert ids == expected[status], f"{status} request bled: got {ids}"
        assert {w["status"] for w in result.data["widgets"]} == {status}
    # only the warm-up missed; all 16 concurrent requests were cache hits over the SHARED plan.
    assert cache.misses == 1
    assert cache.hits == len(statuses)


@pytest.mark.pg
@pytest.mark.asyncio
@pytest.mark.cache_off
async def test_deepcopy_free_cache_no_bleed_under_barrier_concurrency(seeded):
    """BACKSTOP (gate 8): aggressive BARRIER-SYNCHRONIZED concurrent no-bleed over the SHARED plan.

    The deepcopy is GONE — a cache HIT serves the SHARED step DAG, and each request renders its
    OWN variable value into the compiled statement's params via its OWN ``BucketExtra.source_values``.
    A single missed per-request relocation would BLEED across concurrent requests and pass a
    single-threaded run. This is the load-bearing guard against that: warm the cache once, then run
    MANY rounds of N requests with DIFFERENT variable values, each coroutine waiting on an
    ``asyncio.Barrier`` IMMEDIATELY before ``graphql(...)`` so all N requests release into execution
    TOGETHER (a genuine overlap, not a lucky schedule). Every request must get ITS OWN correct rows.
    """
    cache = PlanCache()
    config = GrafastConfig(placeholders=True, cache_plans=True, plan_cache=cache)

    class _Ctx(GrafastExecutionContext):
        grafast_config = config

    schema = build_widgets_schema()
    query = "query Q($s: String!) { widgets(status: $s) { id status } }"
    expected = {"draft": [2, 5], "published": [1, 3, 4, 6]}

    # warm the cache so EVERY request in the rounds below is a deepcopy-free HIT over the shared plan.
    with pg_request_context(SQLAlchemyExecutor(get_engine())):
        await graphql(schema, query, variable_values={"s": "published"},
                      execution_context_class=_Ctx)

    rounds = 20
    width = 8  # 20 * 8 = 160 barrier-overlapped concurrent requests

    async def one(barrier, status):
        with pg_request_context(SQLAlchemyExecutor(get_engine())):
            # release all `width` requests into execution together so they genuinely overlap.
            await barrier.wait()
            return status, await graphql(
                schema, query, variable_values={"s": status},
                execution_context_class=_Ctx,
            )

    total = 0
    for _ in range(rounds):
        barrier = asyncio.Barrier(width)
        # interleave draft/published so adjacent overlapping requests carry DIFFERENT values.
        statuses = [("draft" if i % 2 == 0 else "published") for i in range(width)]
        results = await asyncio.gather(*(one(barrier, s) for s in statuses))
        for status, result in results:
            assert result.errors is None
            ids = sorted(w["id"] for w in result.data["widgets"])
            assert ids == expected[status], f"{status} request bled: got {ids}"
            assert {w["status"] for w in result.data["widgets"]} == {status}
        total += width

    # exactly the warm-up missed; every barrier-overlapped request was a deepcopy-free HIT.
    assert cache.misses == 1
    assert cache.hits == total


@pytest.mark.pg
@pytest.mark.asyncio
@pytest.mark.cache_off
async def test_cached_result_byte_identical_to_uncached(seeded):
    """NO-REGRESSION over the DB: the cached path returns the SAME rows + SAME count as uncached.

    The same query run uncached (caching off) and cached (a cache hit on the second call) must
    produce byte-identical data and the SAME statement count — a hit changes WHETHER planning
    runs, never the SQL.
    """
    schema = build_widgets_schema()
    query = "query Q($s: String!) { widgets(status: $s) { id status } }"

    uncached_ctx = context_class_with(GrafastConfig())  # caching off
    cache = PlanCache()
    cached_ctx = context_class_with(
        GrafastConfig(placeholders=True, cache_plans=True, plan_cache=cache)
    )

    engine = get_engine()
    with pg_request_context(SQLAlchemyExecutor(engine)):
        with count_sql(engine) as uncached_count:
            uncached = await graphql(
                schema, query, variable_values={"s": "published"},
                execution_context_class=uncached_ctx,
            )
    # prime the cache (miss), then the measured run is a HIT.
    with pg_request_context(SQLAlchemyExecutor(engine)):
        await graphql(
            schema, query, variable_values={"s": "published"},
            execution_context_class=cached_ctx,
        )
    with pg_request_context(SQLAlchemyExecutor(engine)):
        with count_sql(engine) as cached_count:
            cached = await graphql(
                schema, query, variable_values={"s": "published"},
                execution_context_class=cached_ctx,
            )

    assert uncached.errors is None and cached.errors is None
    assert uncached.data == cached.data
    assert cached_count.count == uncached_count.count
    # the cached context ran twice: the first primed the cache (miss), the measured run hit it.
    assert cache.hits == 1 and cache.misses == 1


# ----------------------------------- select_customizer convergence to selectAuth (DB)
#
# A resource select_customizer (the selectAuth analogue) scopes reads by the per-request
# context. The cross-request plan cache shares a finalized plan across requests; a 1-arg
# customizer that bakes the context value as a LITERAL into that shared plan leaks one
# request's scope to the next. The convergence: a literal customizer is forced NON-cacheable
# (the safety floor) so it can never leak, while a value-agnostic placeholder customizer
# (the 2-arg form) stays cacheable and re-binds its value per request — structure fixed at
# plan time, value supplied per request at execute time, exactly as upstream selectAuth.


CUSTOMIZER_WIDGETS_SDL = """
type Query { widgets: [Widget!]! }
type Widget { id: Int! status: String! }
"""


def build_customizer_widgets_schema(select_customizer):
    """A widgets schema whose root list applies a resource ``select_customizer`` (no GraphQL arg).

    The filter comes entirely from the per-request context via the customizer, so two requests
    of the SAME document (no variables) differ only by their bound context — the setup that
    exposes a cache leak if the context value is baked into the shared cached plan.
    """
    from grafast_py.schema import make_grafast_schema

    registry = PgRegistry()
    widgets = PgResource(
        "widgets", "grafast_demo", "widgets",
        ["id", "owner_id", "title", "status", "deleted_at"],
        registry=registry, select_customizer=select_customizer,
    )

    def widgets_plan(parent, args, info):
        return PgSelectAllStep(widgets, order_by=["id"]).for_parent(parent)

    return make_grafast_schema(
        CUSTOMIZER_WIDGETS_SDL,
        {
            "Query": {"widgets": widgets_plan},
            "Widget": {
                "id": lambda p, a, i: access(p, ("id",)),
                "status": lambda p, a, i: access(p, ("status",)),
            },
        },
    )


@pytest.mark.pg
@pytest.mark.asyncio
@pytest.mark.cache_off
async def test_literal_select_customizer_leaks_across_contexts_under_cache(seeded):
    """A 1-arg (literal) customizer must NOT be cached (else it leaks across contexts).

    Request A (context status=published) plans a select with ``status = 'published'`` BAKED in.
    Without the safety floor that plan is wrongly cacheable, so request B (context status=draft,
    SAME document) hits the cache and is served request A's published rows — a cross-context
    leak. The fix forces a literal customizer NON-cacheable, so request B re-plans with ITS
    context and gets ITS own rows.
    """

    def only_status(ctx):
        return [column("status") == ctx["status"]]

    cache = PlanCache()
    config = GrafastConfig(cache_plans=True, plan_cache=cache)

    class _Ctx(GrafastExecutionContext):
        grafast_config = config

    schema = build_customizer_widgets_schema(only_status)
    query = "{ widgets { id status } }"

    with pg_request_context(SQLAlchemyExecutor(get_engine()), context={"status": "published"}):
        a = await graphql(schema, query, execution_context_class=_Ctx)
    with pg_request_context(SQLAlchemyExecutor(get_engine()), context={"status": "draft"}):
        b = await graphql(schema, query, execution_context_class=_Ctx)

    assert a.errors is None and b.errors is None
    # the literal customizer is value-pinned, so NEITHER request may reuse the other's plan.
    assert cache.misses == 2 and cache.hits == 0
    # each request gets ITS OWN context's rows — no leak.
    assert sorted(w["id"] for w in a.data["widgets"]) == [1, 3, 4, 6]
    assert sorted(w["id"] for w in b.data["widgets"]) == [2, 5]
    assert {w["status"] for w in b.data["widgets"]} == {"draft"}


@pytest.mark.pg
@pytest.mark.asyncio
@pytest.mark.cache_off
async def test_placeholder_select_customizer_cacheable_no_leak(seeded):
    """A 2-arg (placeholder) customizer stays CACHEABLE and re-binds its value per request.

    The convergence to upstream selectAuth: the predicate STRUCTURE (``status = :ctx``) is fixed
    at plan time and the VALUE is read per request from the context at execute. Request A
    (published) caches the value-independent plan; request B (draft, SAME document) is a cache HIT
    that re-binds ITS context — so the plan is SHARED (hits==1) AND each request gets ITS OWN rows
    (no leak across the shared plan). This is what the 1-arg literal form cannot do (it is forced
    non-cacheable; see test_literal_select_customizer_leaks_across_contexts_under_cache).
    """

    def only_status(ctx, sources):
        return [column("status") == sources.placeholder("status", type_=String)]

    cache = PlanCache()
    config = GrafastConfig(cache_plans=True, plan_cache=cache)

    class _Ctx(GrafastExecutionContext):
        grafast_config = config

    schema = build_customizer_widgets_schema(only_status)
    query = "{ widgets { id status } }"

    with pg_request_context(SQLAlchemyExecutor(get_engine()), context={"status": "published"}):
        a = await graphql(schema, query, execution_context_class=_Ctx)
    with pg_request_context(SQLAlchemyExecutor(get_engine()), context={"status": "draft"}):
        b = await graphql(schema, query, execution_context_class=_Ctx)

    assert a.errors is None and b.errors is None
    # the placeholder customizer is value-independent: the plan is SHARED across the two requests.
    assert cache.misses == 1 and cache.hits == 1
    # yet each request bound ITS OWN context value off the shared plan — no leak.
    assert sorted(w["id"] for w in a.data["widgets"]) == [1, 3, 4, 6]
    assert sorted(w["id"] for w in b.data["widgets"]) == [2, 5]
    assert {w["status"] for w in a.data["widgets"]} == {"published"}
    assert {w["status"] for w in b.data["widgets"]} == {"draft"}


@pytest.mark.pg
@pytest.mark.asyncio
@pytest.mark.cache_off
async def test_structure_branching_customizer_does_not_leak_across_contexts(seeded):
    """A customizer that BRANCHES its predicate STRUCTURE on context must not leak across requests.

    An admin context yields NO filter (all rows); a user context yields a scoped filter. Both
    branches are value-independent (no baked literal), so the literal safety floor does NOT trip.
    Without a structural-divergence guard, request A (admin) caches the UNFILTERED plan and request
    B (user, SAME document) hits it and sees ALL rows — a privilege-escalation leak. The guard
    re-resolves the customizer per request and re-plans when the STRUCTURE differs, so the user
    gets only their scoped rows.
    """

    def scope(ctx, sources):
        if ctx.get("role") == "admin":
            return []  # admin sees everything (no filter)
        return [column("status") == sources.placeholder("status", type_=String)]

    cache = PlanCache()
    config = GrafastConfig(cache_plans=True, plan_cache=cache)

    class _Ctx(GrafastExecutionContext):
        grafast_config = config

    schema = build_customizer_widgets_schema(scope)
    query = "{ widgets { id status } }"

    with pg_request_context(SQLAlchemyExecutor(get_engine()), context={"role": "admin"}):
        admin = await graphql(schema, query, execution_context_class=_Ctx)
    with pg_request_context(
        SQLAlchemyExecutor(get_engine()), context={"role": "user", "status": "draft"}
    ):
        user = await graphql(schema, query, execution_context_class=_Ctx)

    assert admin.errors is None and user.errors is None
    # the two requests share a cache KEY (same document/config), so the user's lookup matches the
    # admin's entry — exactly the collision that leaked before the guard.
    assert cache.hits == 1
    # ...yet the user still sees ONLY their scoped (draft) rows, never admin's full set: the
    # structural-divergence guard rejected the shape mismatch and re-planned.
    assert sorted(w["id"] for w in admin.data["widgets"]) == [1, 2, 3, 4, 5, 6]
    assert sorted(w["id"] for w in user.data["widgets"]) == [2, 5]
    assert {w["status"] for w in user.data["widgets"]} == {"draft"}


AUTHORS_POSTS_SDL = """
type Query { authors: [Author!]! }
type Author { id: Int! posts: [Post!]! }
type Post { id: Int! title: String! }
"""


def build_authors_posts_schema(posts_customizer):
    """authors -> posts, where the posts RELATION child carries a ``select_customizer``.

    The posts relation is an inline-fold candidate (an unfiltered hasMany), so this exercises the
    path where a customizer-bearing CHILD could be folded out of ``plan.steps`` under
    ``inline_relations`` — the place the structural-divergence guard would otherwise never see.
    """
    from grafast_py.schema import make_grafast_schema
    from grafast_py.pg.resource import PgColumn

    registry = PgRegistry()
    authors = PgResource(
        "authors", "grafast_demo", "authors",
        [PgColumn("id", sql_type=Integer()), PgColumn("name", sql_type=Text())],
        registry=registry,
    )
    posts = PgResource(
        "posts", "grafast_demo", "posts",
        [
            PgColumn("id", sql_type=Integer()),
            PgColumn("author_id", sql_type=Integer()),
            PgColumn("title", sql_type=Text()),
        ],
        registry=registry, select_customizer=posts_customizer,
    )
    authors.has_many("posts", target=posts, local_column="id", remote_column="author_id")

    return make_grafast_schema(
        AUTHORS_POSTS_SDL,
        {
            "Query": {
                "authors": lambda p, a, i: PgSelectAllStep(authors, order_by=["id"]).for_parent(p)
            },
            "Author": {
                "id": lambda p, a, i: access(p, ("id",)),
                "posts": lambda p, a, i: authors.related_many(p, "posts"),
            },
            "Post": {
                "id": lambda p, a, i: access(p, ("id",)),
                "title": lambda p, a, i: access(p, ("title",)),
            },
        },
    )


@pytest.mark.pg
@pytest.mark.asyncio
@pytest.mark.cache_off
async def test_structure_branching_customizer_on_inlined_relation_does_not_leak(seeded):
    """A structure-branching customizer on an INLINE-FOLDED relation child must not leak.

    Under ``cache_plans`` + ``inline_relations`` the admin branch returns ``[]`` (all posts), and
    folding that unfiltered child would drop it from ``plan.steps`` so the structural-divergence
    guard could not re-check it — a later user request would then inherit the admin's unfiltered
    posts (privilege escalation). The fold-skip for customizer-bearing children keeps the child on
    the batched path, where the guard runs and re-plans the user.
    """

    def posts_scope(ctx, sources):
        if ctx.get("role") == "admin":
            return []  # admin sees ALL posts
        return [column("title") == sources.placeholder("title")]

    cache = PlanCache()
    config = GrafastConfig(cache_plans=True, inline_relations=True, plan_cache=cache)

    class _Ctx(GrafastExecutionContext):
        grafast_config = config

    schema = build_authors_posts_schema(posts_scope)
    query = "{ authors { id posts { id } } }"

    with pg_request_context(SQLAlchemyExecutor(get_engine()), context={"role": "admin"}):
        admin = await graphql(schema, query, execution_context_class=_Ctx)
    with pg_request_context(
        SQLAlchemyExecutor(get_engine()), context={"role": "user", "title": "no-such-title"}
    ):
        user = await graphql(schema, query, execution_context_class=_Ctx)

    assert admin.errors is None and user.errors is None
    # admin sees every author's posts (9 total across 3 authors); the user, scoped to a title that
    # matches nothing, must see NONE — never the admin's posts.
    assert sum(len(a["posts"]) for a in admin.data["authors"]) == 9
    assert sum(len(a["posts"]) for a in user.data["authors"]) == 0
    # the two requests collide on the cache key (a hit), yet the user stays isolated: the
    # customizer-bearing child is not folded, so the structural guard re-planned for the user.
    assert cache.hits == 1


def build_same_table_two_resource_schema(scoped_customizer):
    """Two resources over the SAME widgets table (same columns) — one plain, one customizer-scoped.

    Selected as two root fields in one operation. When the scoped customizer returns [] for the
    planning request, the scoped select looks IDENTICAL to the plain one (same table, columns,
    empty customization signature), so dedup could merge the scoped step into the plain peer.
    """
    from grafast_py.schema import make_grafast_schema

    cols = ["id", "owner_id", "title", "status", "deleted_at"]
    registry = PgRegistry()
    plain = PgResource("widgets_plain", "grafast_demo", "widgets", cols, registry=registry)
    scoped = PgResource(
        "widgets_scoped", "grafast_demo", "widgets", cols, registry=registry,
        select_customizer=scoped_customizer,
    )
    return make_grafast_schema(
        "type Query { plain: [W!]! scoped: [W!]! }\ntype W { id: Int! status: String! }",
        {
            "Query": {
                "plain": lambda p, a, i: PgSelectAllStep(plain, order_by=["id"]).for_parent(p),
                "scoped": lambda p, a, i: PgSelectAllStep(scoped, order_by=["id"]).for_parent(p),
            },
            "W": {
                "id": lambda p, a, i: access(p, ("id",)),
                "status": lambda p, a, i: access(p, ("status",)),
            },
        },
    )


@pytest.mark.pg
@pytest.mark.asyncio
@pytest.mark.cache_off
async def test_dedup_merging_customizer_step_into_uncustomized_peer_does_not_leak(seeded):
    """A scoped customizer step must not dedup-merge into an uncustomized same-table peer and leak.

    Request A (admin) -> the scoped customizer returns [] -> the scoped select looks identical to
    the plain select and could merge into it; if cached, request B (user) would read the plain
    (unfiltered) result for the scoped field. The scoped step must stay distinct/visible to the
    cache-hit structural guard so the user gets only their scoped rows.
    """

    def scope(ctx, sources):
        if ctx.get("role") == "admin":
            return []
        return [column("status") == sources.placeholder("status")]

    cache = PlanCache()
    config = GrafastConfig(cache_plans=True, plan_cache=cache)

    class _Ctx(GrafastExecutionContext):
        grafast_config = config

    schema = build_same_table_two_resource_schema(scope)
    query = "{ plain { id } scoped { id status } }"

    with pg_request_context(SQLAlchemyExecutor(get_engine()), context={"role": "admin"}):
        admin = await graphql(schema, query, execution_context_class=_Ctx)
    with pg_request_context(
        SQLAlchemyExecutor(get_engine()), context={"role": "user", "status": "draft"}
    ):
        user = await graphql(schema, query, execution_context_class=_Ctx)

    assert admin.errors is None and user.errors is None, (admin.errors, user.errors)
    # admin: both fields see all 6 widgets. user: plain still all 6, but scoped MUST be only [2,5].
    assert sorted(w["id"] for w in admin.data["scoped"]) == [1, 2, 3, 4, 5, 6]
    assert sorted(w["id"] for w in user.data["plain"]) == [1, 2, 3, 4, 5, 6]
    assert sorted(w["id"] for w in user.data["scoped"]) == [2, 5]
