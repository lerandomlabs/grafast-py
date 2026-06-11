"""Request-input constraints: the general plan-cache correctness substrate.

A cached plan may be reused only when it is provably correct for the requesting
request. Every way a per-request input can influence a plan's SHAPE or FIELD
SELECTION must either record a re-checkable CONSTRAINT (a note: "this plan assumed
input X looked like Y", validated on every hit) or refuse the hit. This module
gates the whole substrate:

  * the live @skip/@include/fragment-if BUG: a directive arg that is a $variable
    changes the resolved field selection BEFORE planning (graphql-core's
    collect_fields), so the cache must constrain those variable values — a different
    value is a different plan variant, not a stale hit;
  * the MULTI-VARIANT cache: one document holds several plan variants (hide=true /
    hide=false; admin / user), each guarded by its constraint set — no thrash,
    correct variant per request;
  * the VARIABLES RULE: a variable value observed raw by planning (FieldArgs reads,
    ``raw``, ``__contains__``, ``info.variable_values``, a variable NESTED inside an
    input-object literal) records a value constraint instead of silently baking;
  * the CONTEXT GATE: plan resolvers see ``info.context`` as a gate — bare reads
    yield blank tokens (no value, no constraint; the placeholder channel), and
    ``eval`` / ``eval_is`` / ``eval_has`` are the explicit look-now reads that record
    constraints (the upstream __TrackedValueStep.eval* model, public here);
  * the HIT-RATE guarantee: the placeholder path records ZERO constraints.
"""

import pytest
from graphql import graphql_sync, parse

from grafast_py.cache import PlanCache, compute_cache_key, default_cache
from grafast_py.config import GrafastConfig
from grafast_py.context import GrafastExecutionContext
from grafast_py.core_steps import access, constant
from grafast_py.constraints import (
    ABSENT,
    ContextToken,
    EqualityConstraint,
    ExistsConstraint,
    RequestFacts,
    ValueConstraint,
    constraints_match,
    directive_variable_constraints,
)
from grafast_py.schema import make_grafast_schema


def context_class_with(config: GrafastConfig):
    class _Ctx(GrafastExecutionContext):
        grafast_config = config

    return _Ctx


def cached_config() -> GrafastConfig:
    """A fresh caching config with its OWN PlanCache (isolated counters per test)."""
    return GrafastConfig(placeholders=True, cache_plans=True, plan_cache=PlanCache())


def run(schema, query, config, variables=None, context_value=None):
    result = graphql_sync(
        schema,
        query,
        variable_values=variables or {},
        context_value=context_value,
        execution_context_class=context_class_with(config),
    )
    assert result.errors is None, result.errors
    return result.data


# ------------------------------------------------------------------ constraint unit layer


def test_value_constraint_matches_by_equality():
    facts = RequestFacts(variable_values={"hide": True}, context_value=None)
    assert ValueConstraint("variables", ("hide",), True).matches(facts)
    assert not ValueConstraint("variables", ("hide",), False).matches(facts)
    # a missing variable only matches a constraint recorded as ABSENT
    facts_missing = RequestFacts(variable_values={}, context_value=None)
    assert ValueConstraint("variables", ("hide",), ABSENT).matches(facts_missing)
    assert not ValueConstraint("variables", ("hide",), True).matches(facts_missing)


def test_constraint_equality_is_type_pinned():
    """Python's cross-type aliasing (True == 1, 0 == False) must not validate a hit —
    upstream's === semantics, approximated by a top-level type check (a mismatch only
    ever re-plans, the safe direction)."""
    assert not ValueConstraint("context", ("flag",), 1).matches(
        RequestFacts({}, {"flag": True})
    )
    assert not ValueConstraint("context", ("flag",), False).matches(
        RequestFacts({}, {"flag": 0})
    )
    assert ValueConstraint("context", ("flag",), {"a": [1]}).matches(
        RequestFacts({}, {"flag": {"a": [1]}})
    )


def test_equality_constraint_matches_the_outcome_not_the_value():
    """eval_is records the comparison OUTCOME: every non-matching value shares one plan."""
    recorded = EqualityConstraint("context", ("role",), "admin", False)
    assert recorded.matches(RequestFacts({}, {"role": "user"}))
    assert recorded.matches(RequestFacts({}, {"role": "editor"}))  # also non-admin -> same plan
    assert not recorded.matches(RequestFacts({}, {"role": "admin"}))


def test_exists_constraint_matches_presence():
    c = ExistsConstraint("variables", ("s",), True)
    assert c.matches(RequestFacts({"s": None}, None))  # present-but-null IS present
    assert not c.matches(RequestFacts({}, None))


def test_context_scope_walks_mapping_and_attributes():
    class Ctx:
        tenant = "t1"

    assert ValueConstraint("context", ("tenant",), "t1").matches(RequestFacts({}, Ctx()))
    assert ValueConstraint("context", ("tenant",), "t1").matches(
        RequestFacts({}, {"tenant": "t1"})
    )
    assert not ValueConstraint("context", ("tenant",), "t1").matches(RequestFacts({}, {}))


def test_directive_variable_scan_finds_skip_include_and_fragment_spreads():
    doc = parse(
        """
        query Q($hide: Boolean!, $show: Boolean!, $deep: Boolean!) {
          user {
            email @skip(if: $hide)
            name @include(if: $show)
            always @include(if: true)
            ...F
          }
        }
        fragment F on User { phone @skip(if: $deep) }
        """
    )
    operation = doc.definitions[0]
    fragments = {"F": doc.definitions[1]}
    constraints = directive_variable_constraints(
        operation, fragments, {"hide": True, "show": False, "deep": False}
    )
    by_path = {c.path: c.value for c in constraints}
    # one value constraint per directive VARIABLE; the literal if: records nothing.
    assert by_path == {("hide",): True, ("show",): False, ("deep",): False}
    assert all(c.scope == "variables" for c in constraints)


def test_directive_variable_scan_records_nothing_for_literals():
    doc = parse("query { user { email @skip(if: false) name } }")
    assert directive_variable_constraints(doc.definitions[0], {}, {}) == []


def test_directive_variable_scan_covers_defer_and_stream():
    """@defer/@stream variables are plan-time reads in THIS engine (the planner bakes
    deferred partitions and FieldPlan.stream), so the scan constrains them too — unlike
    upstream, which resolves @stream args at runtime."""
    doc = parse(
        """
        query Q($d: Boolean!, $n: Int) {
          user { ... @defer(if: $d) { email } posts @stream(initialCount: $n) }
        }
        """
    )
    constraints = directive_variable_constraints(
        doc.definitions[0], {}, {"d": True, "n": 3}
    )
    assert {c.path: c.value for c in constraints} == {("d",): True, ("n",): 3}


def test_constraints_match_is_an_and_over_the_list():
    facts = RequestFacts({"a": 1, "b": 2}, None)
    ok = ValueConstraint("variables", ("a",), 1)
    bad = ValueConstraint("variables", ("b",), 99)
    assert constraints_match((ok,), facts)
    assert not constraints_match((ok, bad), facts)
    assert constraints_match((), facts)


def test_cache_key_distinguishes_incremental_planning():
    """An incremental-built plan (deferred partitions) must not collide with a normal plan."""
    schema = object()
    op = parse("query Q { user { name } }").definitions[0]
    assert compute_cache_key(schema, op, None, None, incremental=True) != compute_cache_key(
        schema, op, None, None, incremental=False
    )


# ------------------------------------------------------- R4: the live directive-variable bug


USER_SDL = """
type Query { user: User }
type User { name: String! email: String! phone: String! }
"""


def build_user_schema():
    def user_plan(parent, args, info):
        return constant({"name": "ada", "email": "a@x", "phone": "123"})

    def field_plan(key):
        return lambda parent, args, info: access(parent, (key,))

    return make_grafast_schema(
        USER_SDL,
        {
            "Query": {"user": user_plan},
            "User": {
                "name": field_plan("name"),
                "email": field_plan("email"),
                "phone": field_plan("phone"),
            },
        },
    )


SKIP_QUERY = "query Q($hide: Boolean!) { user { name email @skip(if: $hide) } }"


@pytest.mark.parametrize("first_hide", [False, True])
def test_skip_directive_variable_gets_correct_field_set_per_value(first_hide):
    """THE LIVE BUG (spec R4): @skip(if: $hide) must honour EACH request's value.

    Both orderings: whichever value plans first, the other value must NOT be served
    the first request's frozen field selection.
    """
    schema = build_user_schema()
    config = cached_config()

    first = run(schema, SKIP_QUERY, config, {"hide": first_hide})
    second = run(schema, SKIP_QUERY, config, {"hide": not first_hide})

    with_email = {"user": {"name": "ada", "email": "a@x"}}
    without_email = {"user": {"name": "ada"}}
    assert first == (without_email if first_hide else with_email)
    assert second == (with_email if first_hide else without_email)


def test_include_directive_variable_gets_correct_field_set_per_value():
    schema = build_user_schema()
    config = cached_config()
    query = "query Q($show: Boolean!) { user { name email @include(if: $show) } }"

    shown = run(schema, query, config, {"show": True})
    hidden = run(schema, query, config, {"show": False})
    assert shown == {"user": {"name": "ada", "email": "a@x"}}
    assert hidden == {"user": {"name": "ada"}}


def test_fragment_spread_skip_variable_gets_correct_field_set_per_value():
    schema = build_user_schema()
    config = cached_config()
    query = (
        "query Q($v: Boolean!) { user { name ...F @skip(if: $v) } }"
        " fragment F on User { email }"
    )

    kept = run(schema, query, config, {"v": False})
    skipped = run(schema, query, config, {"v": True})
    assert kept == {"user": {"name": "ada", "email": "a@x"}}
    assert skipped == {"user": {"name": "ada"}}


def test_directive_variants_coexist_in_the_cache_no_thrash():
    """The MULTI-VARIANT payoff: hide=true and hide=false plans coexist under one key.

    Four requests alternating the value: the first two plan (one variant each), the
    second two are HITS on their matching variant — alternating values must not evict
    each other (the single-slot thrash the old cache had).
    """
    schema = build_user_schema()
    config = cached_config()
    cache = config.plan_cache

    for hide in (True, False, True, False):
        data = run(schema, SKIP_QUERY, config, {"hide": hide})
        expected = {"user": {"name": "ada"}} if hide else {"user": {"name": "ada", "email": "a@x"}}
        assert data == expected

    assert cache.misses == 2  # one plan build per variant
    assert cache.hits == 2  # each repeat value validated against ITS variant


def test_literal_directive_args_stay_fully_cached():
    """A literal @skip(if: false) records NO constraint — the second request is a plain hit."""
    schema = build_user_schema()
    config = cached_config()
    query = "{ user { name email @skip(if: false) } }"

    a = run(schema, query, config)
    b = run(schema, query, config)
    assert a == b == {"user": {"name": "ada", "email": "a@x"}}
    assert config.plan_cache.misses == 1 and config.plan_cache.hits == 1


# ------------------------------------------------- the variables rule (raw reads constrain)


THINGS_SDL = """
input Where { status: String }
type Query {
  things(status: String): [Thing!]!
  posts(where: Where): [Thing!]!
}
type Thing { id: Int! status: String! }
"""


def thing_field_plans():
    return {
        "id": lambda p, a, i: access(p, ("id",)),
        "status": lambda p, a, i: access(p, ("status",)),
    }


def test_nested_variable_in_input_literal_is_not_served_stale():
    """A $variable NESTED inside an input-object literal must not bake-and-cache.

    ``posts(where: {status: $s})``: the arg's value node is an ObjectValueNode containing a
    VariableNode — invisible to the old direct-only provenance, so the coerced value (with
    request 1's $s frozen inside) was cached and served to request 2.
    """

    def posts_plan(parent, args, info):
        where = args["where"]  # a coerced dict CONTAINING the variable's value
        return constant([{"id": 1, "status": where["status"]}])

    schema = make_grafast_schema(
        THINGS_SDL, {"Query": {"posts": posts_plan}, "Thing": thing_field_plans()}
    )
    config = cached_config()
    query = "query Q($s: String) { posts(where: {status: $s}) { id status } }"

    first = run(schema, query, config, {"s": "draft"})
    second = run(schema, query, config, {"s": "published"})
    assert first == {"posts": [{"id": 1, "status": "draft"}]}
    assert second == {"posts": [{"id": 1, "status": "published"}]}  # NOT request 1's "draft"


def test_fieldargs_raw_read_is_not_served_stale():
    """``args.raw`` hands out variable values too — it must do the same bookkeeping as []."""

    def things_plan(parent, args, info):
        status = args.raw["status"]  # the bypass: no __getitem__, no .get
        return constant([{"id": 1, "status": status}])

    schema = make_grafast_schema(
        THINGS_SDL, {"Query": {"things": things_plan}, "Thing": thing_field_plans()}
    )
    config = cached_config()
    query = "query Q($s: String) { things(status: $s) { id status } }"

    first = run(schema, query, config, {"s": "a"})
    second = run(schema, query, config, {"s": "b"})
    assert first == {"things": [{"id": 1, "status": "a"}]}
    assert second == {"things": [{"id": 1, "status": "b"}]}


def test_fieldargs_membership_branching_is_not_served_stale():
    """Branching plan shape on ``"x" in args`` is a per-request decision (provided vs omitted)."""

    def things_plan(parent, args, info):
        if "status" in args:  # presence depends on whether $s was provided
            return constant([{"id": 1, "status": "filtered"}])
        return constant([{"id": 1, "status": "all"}, {"id": 2, "status": "all"}])

    schema = make_grafast_schema(
        THINGS_SDL, {"Query": {"things": things_plan}, "Thing": thing_field_plans()}
    )
    config = cached_config()
    query = "query Q($s: String) { things(status: $s) { id } }"

    provided = run(schema, query, config, {"s": "x"})
    omitted = run(schema, query, config, {})
    assert provided == {"things": [{"id": 1}]}
    assert omitted == {"things": [{"id": 1}, {"id": 2}]}  # NOT the cached filtered shape


def test_info_variable_values_read_is_not_served_stale():
    """Reading ``info.variable_values`` at plan time is a tracked eval, not a silent bake."""

    def things_plan(parent, args, info):
        status = info.variable_values["s"]
        return constant([{"id": 1, "status": status}])

    schema = make_grafast_schema(
        THINGS_SDL, {"Query": {"things": things_plan}, "Thing": thing_field_plans()}
    )
    config = cached_config()
    query = "query Q($s: String) { things(status: $s) { id status } }"

    first = run(schema, query, config, {"s": "a"})
    second = run(schema, query, config, {"s": "b"})
    assert first == {"things": [{"id": 1, "status": "a"}]}
    assert second == {"things": [{"id": 1, "status": "b"}]}


def test_raw_variable_read_caches_per_value_not_uncacheable():
    """The upgrade over the old rule: an inlined variable SPLITS the cache, never re-plans
    the same value twice. (Old behaviour: plan.cacheable=False -> a re-plan EVERY request.)
    """

    def things_plan(parent, args, info):
        return constant([{"id": 1, "status": args["status"]}])

    schema = make_grafast_schema(
        THINGS_SDL, {"Query": {"things": things_plan}, "Thing": thing_field_plans()}
    )
    config = cached_config()
    query = "query Q($s: String) { things(status: $s) { id status } }"

    run(schema, query, config, {"s": "a"})  # miss: plans the s=="a" variant
    run(schema, query, config, {"s": "b"})  # miss: plans the s=="b" variant
    assert run(schema, query, config, {"s": "a"}) == {
        "things": [{"id": 1, "status": "a"}]
    }  # hit on the a-variant
    assert config.plan_cache.misses == 2 and config.plan_cache.hits == 1


# ----------------------------------------------------------- R3: the context gate


def test_context_bare_read_yields_a_token_that_cannot_branch():
    """``info.context[...]`` withholds the value: a blank token, and ``if token:`` raises."""
    seen = {}

    def things_plan(parent, args, info):
        token = info.context["tenant_id"]
        seen["token"] = token
        return constant([{"id": 1, "status": "x"}])

    schema = make_grafast_schema(
        THINGS_SDL, {"Query": {"things": things_plan}, "Thing": thing_field_plans()}
    )
    run(schema, "{ things { id } }", cached_config(), context_value={"tenant_id": "t1"})

    token = seen["token"]
    assert isinstance(token, ContextToken)
    assert token.source == "ctx:tenant_id"
    with pytest.raises(TypeError, match="eval"):
        bool(token)


def test_context_eval_branches_correctly_and_caches_per_value():
    """A plan resolver branching shape on context must get the right plan per context."""

    def things_plan(parent, args, info):
        if info.context.eval("is_admin"):
            return constant([{"id": 1, "status": "all"}, {"id": 2, "status": "all"}])
        return constant([{"id": 1, "status": "mine"}])

    schema = make_grafast_schema(
        THINGS_SDL, {"Query": {"things": things_plan}, "Thing": thing_field_plans()}
    )
    config = cached_config()
    query = "{ things { id } }"

    admin = run(schema, query, config, context_value={"is_admin": True})
    user = run(schema, query, config, context_value={"is_admin": False})
    admin_again = run(schema, query, config, context_value={"is_admin": True})

    assert admin == admin_again == {"things": [{"id": 1}, {"id": 2}]}
    assert user == {"things": [{"id": 1}]}
    # two variants planned; the third request validated against the admin variant.
    assert config.plan_cache.misses == 2 and config.plan_cache.hits == 1


def test_context_eval_is_shares_one_plan_across_non_matching_values():
    """eval_is constrains the OUTCOME: every non-admin tenant shares ONE cached plan."""

    def things_plan(parent, args, info):
        if info.context.eval_is("role", "admin"):
            return constant([{"id": 1, "status": "all"}, {"id": 2, "status": "all"}])
        return constant([{"id": 1, "status": "mine"}])

    schema = make_grafast_schema(
        THINGS_SDL, {"Query": {"things": things_plan}, "Thing": thing_field_plans()}
    )
    config = cached_config()
    query = "{ things { id } }"

    run(schema, query, config, context_value={"role": "user"})  # miss: the non-admin variant
    run(schema, query, config, context_value={"role": "editor"})  # HIT: outcome matches
    admin = run(schema, query, config, context_value={"role": "admin"})  # miss: admin variant

    assert admin == {"things": [{"id": 1}, {"id": 2}]}
    assert config.plan_cache.misses == 2 and config.plan_cache.hits == 1


def test_context_eval_has_constrains_presence():
    def things_plan(parent, args, info):
        if info.context.eval_has("filter"):
            return constant([{"id": 1, "status": "filtered"}])
        return constant([{"id": 1, "status": "all"}, {"id": 2, "status": "all"}])

    schema = make_grafast_schema(
        THINGS_SDL, {"Query": {"things": things_plan}, "Thing": thing_field_plans()}
    )
    config = cached_config()
    query = "{ things { id } }"

    with_filter = run(schema, query, config, context_value={"filter": "x"})
    without = run(schema, query, config, context_value={})
    assert with_filter == {"things": [{"id": 1}]}
    assert without == {"things": [{"id": 1}, {"id": 2}]}


def test_placeholder_path_records_zero_constraints():
    """The HIT-RATE guarantee: source()/token placeholders never split the cache."""
    from grafast_py import _compat
    from grafast_py.plan import plan_operation

    def things_plan(parent, args, info):
        if args.is_variable("status"):
            _ = args.source("status")  # the placeholder channel — no constraint
        _ = info.context["tenant_id"]  # a bare token — no constraint
        return constant([{"id": 1, "status": "x"}])

    schema = make_grafast_schema(
        THINGS_SDL, {"Query": {"things": things_plan}, "Thing": thing_field_plans()}
    )
    config = cached_config()
    document = parse("query Q($s: String) { things(status: $s) { id } }")
    operation = document.definitions[0]
    ctx = context_class_with(config).build(
        schema, document, context_value={"tenant_id": "t1"}, raw_variable_values={"s": "x"}
    )
    root_fields = _compat.collect_root_fields(ctx, schema.query_type, operation)
    plan_operation(ctx, operation, schema.query_type, root_fields)
    assert ctx._grafast_plan.request_constraints == []
    # ...and the STORED candidate carries an empty constraint set end to end (directive
    # scan included), so every later request of this document validates trivially.
    [bucket] = config.plan_cache._entries.values()
    assert bucket[0].constraints == ()


# ------------------------------------------------- review findings: eval_is edge cases


def test_eval_is_with_omitted_variable_and_arg_default_is_not_served_stale():
    """eval_is observes the COERCED arg, constraints validate the VARIABLE — the two
    diverge when the variable is omitted and the ARGUMENT default folds in. The recorded
    outcome must therefore be pinned to the variable's presence: an absent-variable
    request must never validate a present-built variant (or vice versa).
    """

    def things_plan(parent, args, info):
        if args.eval_is("status", "active"):
            return constant([{"id": 1, "status": "active-branch"}])
        return constant([{"id": 2, "status": "other-branch"}])

    sdl = """
    type Query { things(status: String = "active"): [Thing!]! }
    type Thing { id: Int! status: String! }
    """
    schema = make_grafast_schema(
        sdl, {"Query": {"things": things_plan}, "Thing": thing_field_plans()}
    )
    config = cached_config()
    query = "query Q($s: String) { things(status: $s) { id } }"

    # build with the variable PRESENT on the false branch...
    assert run(schema, query, config, {"s": "inactive"}) == {"things": [{"id": 2}]}
    # ...then OMIT it: the coerced arg is the schema default "active" -> the TRUE branch.
    # Serving the false-branch variant here was the stale-serve bug.
    assert run(schema, query, config, {}) == {"things": [{"id": 1}]}
    # and a present-"active" request takes the true branch via its own variant.
    assert run(schema, query, config, {"s": "active"}) == {"things": [{"id": 1}]}


def test_eval_is_constraint_holds_for_its_own_builder():
    """An eval_is variant must validate for the request that BUILT it (no perpetual-miss
    churn): the same request twice is one miss + one hit, for present and absent alike."""

    def things_plan(parent, args, info):
        args.eval_is("status", "active")
        return constant([{"id": 1, "status": "x"}])

    sdl = """
    type Query { things(status: String = "active"): [Thing!]! }
    type Thing { id: Int! status: String! }
    """
    schema = make_grafast_schema(
        sdl, {"Query": {"things": things_plan}, "Thing": thing_field_plans()}
    )
    query = "query Q($s: String) { things(status: $s) { id } }"

    for variables in ({"s": "inactive"}, {}):
        config = cached_config()
        run(schema, query, config, variables)
        run(schema, query, config, variables)
        assert (config.plan_cache.misses, config.plan_cache.hits) == (1, 1), variables


def test_eval_is_on_nested_variable_arg_is_not_served_stale():
    """eval_is on an arg whose variables are NESTED in a literal observes those values —
    it must constrain (as a full observation), never freeze the outcome unconstrained."""

    def posts_plan(parent, args, info):
        if args.eval_is("where", {"status": "draft"}):
            return constant([{"id": 1, "status": "draft-branch"}])
        return constant([{"id": 2, "status": "other-branch"}])

    schema = make_grafast_schema(
        THINGS_SDL, {"Query": {"posts": posts_plan}, "Thing": thing_field_plans()}
    )
    config = cached_config()
    query = "query Q($s: String) { posts(where: {status: $s}) { id } }"

    assert run(schema, query, config, {"s": "draft"}) == {"posts": [{"id": 1}]}
    assert run(schema, query, config, {"s": "published"}) == {"posts": [{"id": 2}]}


# ------------------------------------------- review findings: args_error + enumeration


def test_variable_dependent_coercion_error_is_not_served_across_values():
    """A plan-time argument-coercion error is a function of the variable values, so the
    frozen ``args_error`` must constrain on them: a null-variable request's error plan
    must not serve a valid request (which would wrongly error), nor the reverse (which
    would swallow the required-null error)."""
    schema = make_grafast_schema(
        "type Query { echo(n: Int!): Int }\n", {}
    )
    config = cached_config()
    query = "query Q($n: Int = 1) { echo(n: $n) }"

    def result_for(variables):
        return graphql_sync(
            schema, query, variable_values=variables,
            execution_context_class=context_class_with(config),
        )

    erroring = result_for({"n": None})  # null into Int! -> located coercion error
    assert erroring.errors is not None
    valid = result_for({"n": 5})  # must NOT inherit the frozen error
    assert valid.errors is None
    erroring_again = result_for({"n": None})  # and the error must not be swallowed
    assert erroring_again.errors is not None


def test_variable_values_enumeration_pins_the_key_set():
    """Iterating/len()-ing info.variable_values reveals WHICH variables were provided —
    a per-request fact that must constrain (the keys analogue of upstream evalKeys)."""

    def things_plan(parent, args, info):
        return constant([{"id": len(info.variable_values), "status": "x"}])

    schema = make_grafast_schema(
        "type Query { things(a: Int, b: Int): [Thing!]! }\ntype Thing { id: Int! status: String! }",
        {"Query": {"things": things_plan}, "Thing": thing_field_plans()},
    )
    config = cached_config()
    query = "query Q($a: Int, $b: Int) { things(a: $a, b: $b) { id } }"

    assert run(schema, query, config, {"a": 1}) == {"things": [{"id": 1}]}
    assert run(schema, query, config, {"a": 1, "b": 2}) == {"things": [{"id": 2}]}


# --------------------------------------------- review findings: gate + token hardening


def test_eval_is_missing_context_key_fails_loud_even_on_a_warm_cache():
    """eval_is pins the key's PRESENCE: a request missing the key must re-plan and hit
    the loud KeyError a fresh build raises — a warm cache must not mask the wiring bug."""

    def things_plan(parent, args, info):
        info.context.eval_is("flag", True)
        return constant([{"id": 1, "status": "x"}])

    schema = make_grafast_schema(
        THINGS_SDL, {"Query": {"things": things_plan}, "Thing": thing_field_plans()}
    )
    config = cached_config()
    query = "{ things { id } }"

    # warm the cache with the key present, on the False outcome...
    run(schema, query, config, context_value={"flag": False})
    # ...then drop the key: the False-outcome constraint would ALSO read False here, so
    # without the presence pin this request would silently hit the warm variant. Instead
    # it re-plans and the fresh eval_is raises the same loud KeyError a cold start would.
    with pytest.raises(KeyError, match="flag"):
        graphql_sync(
            schema, query, context_value={},
            execution_context_class=context_class_with(config),
        )


def test_gate_fail_loud_surfaces():
    """The gate's misuse surfaces all fail loud with guidance, never silently misbehave."""
    from grafast_py.constraints import ContextGate

    gate = ContextGate({"k": 1}, [])
    with pytest.raises(TypeError, match="eval"):
        iter(gate)
    with pytest.raises(TypeError, match="eval_has"):
        "k" in gate
    with pytest.raises(TypeError, match="eval_has"):
        gate.get("k", None)  # the dict default idiom signals value semantics
    with pytest.raises(KeyError):
        gate.eval("missing")


def test_context_token_equality_keys_off_the_source():
    tok_a = ContextToken("ctx:a")
    assert tok_a == ContextToken("ctx:a")
    assert tok_a != ContextToken("ctx:b")
    assert (tok_a == "a") is False  # never equal to a bare value
    assert hash(tok_a) == hash(ContextToken("ctx:a"))


def test_source_on_nested_variable_arg_fails_loud():
    from grafast_py.schema import FieldArgs

    args = FieldArgs(
        {"where": {"status": "draft"}},
        nested_variable_args={"where": frozenset({"s"})},
    )
    with pytest.raises(ValueError, match="nested"):
        args.source("where")


def test_pagination_placeholder_resolves_ctx_source_from_request_context():
    """``Placeholder(info.context["page_size"])`` resolves fresh from the pg request
    context (missing key fails LOUD — a silent None would drop the window bound)."""
    pytest.importorskip("sqlalchemy")  # the pg extra; absent on the core-only 3.3 leg
    from grafast_py.constraints import ContextGate
    from grafast_py.pg.executor import pg_request_context
    from grafast_py.pg.placeholders import Placeholder, resolve_placeholder

    gate = ContextGate({"page_size": 5}, [])
    placeholder = Placeholder(gate["page_size"])
    assert placeholder.source == "ctx:page_size"

    with pg_request_context(object(), context={"page_size": 5}):
        assert resolve_placeholder(placeholder, {}) == 5
    with pg_request_context(object(), context={"page_size": 9}):
        assert resolve_placeholder(placeholder, {}) == 9
    with pg_request_context(object(), context={}):
        with pytest.raises(KeyError):
            resolve_placeholder(placeholder, {})


# ----------------------------------------------- review findings: cache + off-mode


def test_bucket_eviction_past_variant_cap():
    from grafast_py.cache import MAX_VARIANTS_PER_KEY, CachedPlan, PlanCache
    from grafast_py.dag import Plan

    cache = PlanCache()
    # a realistic key SHAPE (the eviction log reads the operation-name component)
    key = (1, 2, "Q", (), (False, True, True, True), False)
    for i in range(MAX_VARIANTS_PER_KEY + 2):
        cache.put(key, CachedPlan(
            object_plan=i, root_step=None, plan=Plan(), schema=None,
            constraints=(ValueConstraint("variables", ("i",), i),),
        ))
    assert len(cache._entries[key]) == MAX_VARIANTS_PER_KEY
    assert cache.evictions == 2
    # the freshest variants survive (MRU at the front)
    assert cache._entries[key][0].object_plan == MAX_VARIANTS_PER_KEY + 1


def test_put_replaces_a_same_constraints_candidate():
    """Concurrent double-plans of ONE variant collapse to a single candidate."""
    from grafast_py.cache import CachedPlan, PlanCache
    from grafast_py.dag import Plan

    cache = PlanCache()
    key = ("k",)
    constraints = (ValueConstraint("variables", ("s",), "x"),)
    cache.put(key, CachedPlan(object_plan="first", root_step=None, plan=Plan(),
                              schema=None, constraints=constraints))
    cache.put(key, CachedPlan(object_plan="second", root_step=None, plan=Plan(),
                              schema=None, constraints=constraints))
    assert [c.object_plan for c in cache._entries[key]] == ["second"]


def test_omitted_defaulted_directive_variable_gets_its_own_variant():
    """An OMITTED directive variable with a default constrains on the folded default.

    (Validation forbids a plain-nullable variable in a ``Boolean!`` directive position,
    so in a valid operation a directive variable is always either provided or defaulted —
    graphql-core folds the default into ``variable_values``, and the constraint pins it.)
    A later request providing a different value gets its own variant, not the omitted
    request's field set.
    """
    schema = build_user_schema()
    config = cached_config()
    query = "query Q($hide: Boolean = false) { user { name email @skip(if: $hide) } }"

    omitted = run(schema, query, config, {})  # default false -> not skipped
    hidden = run(schema, query, config, {"hide": True})
    omitted_again = run(schema, query, config, {})
    assert omitted == omitted_again == {"user": {"name": "ada", "email": "a@x"}}
    assert hidden == {"user": {"name": "ada"}}
    # omitted/defaulted and provided-true coexist as two variants; the third request hit.
    assert config.plan_cache.misses == 2 and config.plan_cache.hits == 1


def test_off_mode_gate_present_and_output_unchanged():
    """The plan-resolver API is uniform across modes: caching OFF still hands plan
    resolvers the gate (constraints just go nowhere), and the output is unchanged."""
    from grafast_py.constraints import ContextGate

    seen = {}

    def things_plan(parent, args, info):
        seen["gate"] = isinstance(info.context, ContextGate)
        seen["eval"] = info.context.eval("tenant")
        return constant([{"id": 1, "status": "x"}])

    schema = make_grafast_schema(
        THINGS_SDL, {"Query": {"things": things_plan}, "Thing": thing_field_plans()}
    )
    data = run(schema, "{ things { id } }", GrafastConfig(), context_value={"tenant": "t1"})
    assert data == {"things": [{"id": 1}]}
    assert seen == {"gate": True, "eval": "t1"}
