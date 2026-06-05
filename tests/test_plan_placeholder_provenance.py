"""Threading the placeholder/caching flags onto the Plan and computing per-argument
variable provenance (Wave 4, step 3).

Step 1 added the two opt-in toggles (``GrafastConfig.placeholders`` /
``GrafastConfig.cache_plans``) and step 2 grew ``FieldArgs`` a provenance surface
(``variable_args`` / ``is_variable`` / ``source``) but nothing populated it. This step:

  * PLUMBS both flags from the execution context's config onto the operation's step DAG
    as ``Plan.placeholders`` / ``Plan.cache_plans`` (mirroring ``Plan.inline_relations``),
    set in ``plan_operation``.
  * Walks each field's argument AST in ``plan_object`` when ``Plan.placeholders`` is on,
    computing the SET of ``$variable``-derived argument names and the arg-name ->
    variable-name mapping, and threads them into the ``FieldArgs`` the plan resolver sees.

It is still a NO-OP on the default path: with ``placeholders`` off (the default) the
planner threads NO provenance, so ``FieldArgs.is_variable`` is always False and a host
inlines literals exactly as before. These tests pin the wiring (flags land on the right
Plan; provenance reaches the resolver only when the flag is on; literals are never marked
as variables) plus the byte-identical execution under both flag states.
"""

import grafast_py.plan as plan_module
from graphql import graphql_sync, parse
from graphql.execution.collect_fields import collect_fields
from graphql.language import VariableNode

from grafast_py import GrafastExecutionContext
from grafast_py.config import GrafastConfig
from grafast_py.core_steps import access, constant
from grafast_py.dag import Plan
from grafast_py.plan import plan_operation, variable_provenance
from grafast_py.schema import make_grafast_schema

SDL = """
type Query {
  things(status: String, limit: Int): [Thing!]!
}
type Thing {
  id: Int!
}
"""

ROWS = [{"id": 1}, {"id": 2}]


def context_class_with(config: GrafastConfig):
    """A throwaway context subclass carrying `config` (keeps the base class clean)."""

    class _Ctx(GrafastExecutionContext):
        grafast_config = config

    return _Ctx


def build_operation_plan(schema, query: str, config: GrafastConfig, variables=None):
    """Run `plan_operation` under a context carrying `config`; return its Plan.

    Mirrors `build_operation_plan` in test_inline_plan_flag but threads variable values
    so the field-argument AST carries real `$variable` nodes for the provenance walk.
    """
    document = parse(query)
    operation = document.definitions[0]
    ctx = context_class_with(config).build(
        schema, document, raw_variable_values=variables or {}
    )
    root_type = schema.query_type
    root_fields = collect_fields(
        ctx.schema, ctx.fragments, ctx.variable_values, root_type, operation.selection_set
    )
    plan_operation(ctx, operation, root_type, root_fields)
    return ctx._grafast_plan


def make_schema(seen_args=None):
    """A no-DB schema whose `things` plan records the FieldArgs it was handed.

    `seen_args` (a list) lets a test inspect the exact FieldArgs the planner threaded in,
    so we can assert provenance landed (or did not) without running any SQL.
    """

    def things_plan(parent, args, info):
        if seen_args is not None:
            seen_args.append(args)
        return constant(ROWS)

    def id_plan(parent, args, info):
        return access(parent, ("id",))

    return make_grafast_schema(
        SDL, {"Query": {"things": things_plan}, "Thing": {"id": id_plan}}
    )


# --------------------------------------------------------------- the Plan defaults


def test_plan_placeholder_flags_default_off():
    """A bare `Plan()` (e.g. a unit-test DAG) ships dark: both flags default False."""
    plan = Plan()
    assert plan.placeholders is False
    assert plan.cache_plans is False


# ---------------------------------------------------- plan_operation threads the flags


def test_plan_operation_threads_flags_off_by_default():
    """The default config ships dark: the operation DAG carries both flags False."""
    plan = build_operation_plan(make_schema(), "{ things { id } }", GrafastConfig())
    assert plan.placeholders is False
    assert plan.cache_plans is False


def test_plan_operation_threads_placeholders_when_host_opts_in():
    """A host opting in stamps placeholders onto the operation DAG."""
    plan = build_operation_plan(
        make_schema(), "{ things { id } }", GrafastConfig(placeholders=True)
    )
    assert plan.placeholders is True
    assert plan.cache_plans is False


def test_plan_operation_threads_cache_plans_when_host_opts_in():
    """A host opting in stamps cache_plans onto the operation DAG."""
    plan = build_operation_plan(
        make_schema(), "{ things { id } }", GrafastConfig(cache_plans=True)
    )
    assert plan.cache_plans is True
    assert plan.placeholders is False


# ------------------------------------------ variable_provenance walks the argument AST


def test_variable_provenance_splits_variables_from_literals():
    """The AST walk marks only `$variable`-valued args; literals are excluded."""
    doc = parse("query Q($s: String) { things(status: $s, limit: 10) { id } }")
    field = doc.definitions[0].selection_set.selections[0]
    variable_args, variable_sources = variable_provenance(field)
    assert variable_args == frozenset({"status"})
    assert variable_sources == {"status": "s"}


def test_variable_provenance_maps_arg_name_to_variable_name():
    """The mapping keys off the GraphQL VARIABLE name, not the argument name."""
    doc = parse("query Q($wantedStatus: String) { things(status: $wantedStatus) { id } }")
    field = doc.definitions[0].selection_set.selections[0]
    variable_args, variable_sources = variable_provenance(field)
    assert variable_args == frozenset({"status"})
    assert variable_sources == {"status": "wantedStatus"}
    # the value node really is a VariableNode (the seam the walk keys off)
    assert isinstance(field.arguments[0].value, VariableNode)


def test_variable_provenance_empty_when_all_literals():
    """A field whose args are all literals yields empty provenance."""
    doc = parse('{ things(status: "published", limit: 10) { id } }')
    field = doc.definitions[0].selection_set.selections[0]
    variable_args, variable_sources = variable_provenance(field)
    assert variable_args == frozenset()
    assert variable_sources == {}


# ----------------------------- plan_object threads provenance into FieldArgs (flag-gated)


def test_plan_object_threads_provenance_into_field_args_when_on():
    """With placeholders ON, the resolver's FieldArgs reports the variable provenance."""
    seen = []
    schema = make_schema(seen_args=seen)
    build_operation_plan(
        schema,
        "query Q($s: String) { things(status: $s, limit: 10) { id } }",
        GrafastConfig(placeholders=True),
        variables={"s": "published"},
    )
    assert len(seen) == 1
    args = seen[0]
    # the variable-derived arg is flagged; the literal arg is not
    assert args.is_variable("status") is True
    assert args.is_variable("limit") is False
    # the source tag keys off the variable name (request-stable)
    assert args.source("status") == "var:s"
    # the coerced value still rides through unchanged
    assert args["status"] == "published"
    assert args["limit"] == 10


def test_plan_object_threads_no_provenance_when_off():
    """With placeholders OFF (default), the resolver sees a literal-only FieldArgs.

    Even though `status` came from a variable, the planner threads NO provenance, so
    `is_variable` is always False and the host falls back to literal inlining.
    """
    seen = []
    schema = make_schema(seen_args=seen)
    build_operation_plan(
        schema,
        "query Q($s: String) { things(status: $s) { id } }",
        GrafastConfig(),
        variables={"s": "published"},
    )
    assert len(seen) == 1
    args = seen[0]
    assert args.variable_args == frozenset()
    assert args.is_variable("status") is False
    # the coerced value is identical to the on-path: provenance never touches the value
    assert args["status"] == "published"


def test_plan_object_does_not_walk_ast_when_flag_off(monkeypatch):
    """The AST walk is skipped entirely on the default path (no wasted work, no surprises).

    Provenance computation is purely additive; with the flag off the planner must not even
    call `variable_provenance`, so the default path is byte-identical to pre-Wave-4.
    """
    calls = []
    real = plan_module.variable_provenance

    def spy(field_node):
        calls.append(field_node)
        return real(field_node)

    monkeypatch.setattr(plan_module, "variable_provenance", spy)
    schema = make_schema()
    build_operation_plan(
        schema,
        "query Q($s: String) { things(status: $s) { id } }",
        GrafastConfig(),
        variables={"s": "published"},
    )
    assert calls == []


# --------------------------------- byte-identical execution under both flag states


def test_execution_byte_identical_under_both_flag_states():
    """No-op proof: result data is identical with placeholders/caching on vs off.

    Nothing CONSUMES the threaded provenance yet (no host builds a placeholder here), so
    flipping the flags must not change the result — the equivalence oracle this step keeps.
    """
    schema = make_schema()
    query = "query Q($s: String) { things(status: $s, limit: 10) { id } }"
    variables = {"s": "published"}

    off = graphql_sync(
        schema,
        query,
        variable_values=variables,
        execution_context_class=context_class_with(GrafastConfig()),
    )
    on = graphql_sync(
        schema,
        query,
        variable_values=variables,
        execution_context_class=context_class_with(
            GrafastConfig(placeholders=True, cache_plans=True)
        ),
    )
    assert off.errors is None and on.errors is None
    assert off.data == on.data == {"things": [{"id": 1}, {"id": 2}]}
