"""Threading the inlining flag onto the Plan.

The global ``GrafastConfig.inline_relations`` toggle (and the per-resource
``opt_out_inline``) is plumbed from the execution context's config onto the operation's
step DAG as ``Plan.inline_relations`` — set in BOTH finalize entry points, the operation
root (``plan_operation``) and every abstract concrete-type subtree
(``completion.abstract_child_plan``) — so a pg step's ``optimize(self, plan)`` can read one
plan-level constant instead of plumbing the whole context.

The flag is purely declarative when no step's ``optimize`` reads ``plan.inline_relations``:
the executed result is byte-identical with the flag on or off. These tests pin the wiring
(the flag lands on the right Plan, defaults off, and the abstract subtree inherits it)
without asserting any behaviour change — the byte-identical execution under both flag
states is the proof there is none.
"""

import grafast_py.dag as dag_module
from graphql import parse
from graphql.execution.collect_fields import collect_fields

from grafast_py import GrafastExecutionContext
from grafast_py.config import GrafastConfig
from grafast_py.core_steps import access, constant
from grafast_py.dag import Plan
from grafast_py.plan import plan_operation
from grafast_py.schema import (
    make_grafast_schema,
    resolve_type_from_tag,
)

SDL = """
type Query {
  config: Config
}
type Config {
  name: String
}
"""


def context_class_with(config: GrafastConfig):
    """A throwaway context subclass carrying `config` (keeps the base class clean)."""

    class _Ctx(GrafastExecutionContext):
        grafast_config = config

    return _Ctx


def build_operation_plan(schema, query: str, config: GrafastConfig):
    """Run `plan_operation` under a context carrying `config`; return its Plan.

    Mirrors the `build_plan` idiom in test_plan_finalize / test_optimize_substrate but
    threads an explicit config so the flag the context stamped onto the DAG can be read.
    """
    document = parse(query)
    operation = document.definitions[0]
    ctx = context_class_with(config).build(schema, document)
    root_type = schema.query_type
    root_fields = collect_fields(
        ctx.schema, ctx.fragments, ctx.variable_values, root_type, operation.selection_set
    )
    plan_operation(ctx, operation, root_type, root_fields)
    return ctx._grafast_plan


def make_schema():
    def config_plan(parent, args, info):
        return constant({"name": "grafast"})

    def name_plan(parent, args, info):
        return access(parent, ("name",))

    return make_grafast_schema(
        SDL, {"Query": {"config": config_plan}, "Config": {"name": name_plan}}
    )


# --------------------------------------------------------------- the Plan default


def test_plan_inline_relations_defaults_off():
    """A bare `Plan()` (e.g. a unit-test DAG) never inlines: the no-op invariant holds."""
    assert Plan().inline_relations is False


# ---------------------------------------------------- plan_operation threads the flag


def test_plan_operation_threads_flag_off_by_default():
    """The default config ships dark: the operation's DAG carries inline_relations False."""
    plan = build_operation_plan(make_schema(), "{ config { name } }", GrafastConfig())
    assert plan.inline_relations is False


def test_plan_operation_threads_flag_on_when_host_opts_in():
    """A host opting in (config flag True) stamps inline_relations onto the operation DAG."""
    plan = build_operation_plan(
        make_schema(), "{ config { name } }", GrafastConfig(inline_relations=True)
    )
    assert plan.inline_relations is True


# ------------------------------------------ abstract_child_plan inherits the flag

ABSTRACT_SDL = """
type Query {
  results: [SearchResult!]!
}
type Image {
  id: Int!
}
type Video {
  id: Int!
}
union SearchResult = Image | Video
"""

ROWS = [
    {"__t": "Image", "id": 1},
    {"__t": "Video", "id": 2},
]


def make_abstract_schema():
    """A no-DB union schema: a constant-list root, access-projection concrete plans.

    Selecting a field under the union forces completion-time abstract dispatch, which
    runs `abstract_child_plan` per concrete type — the second finalize entry point that
    must inherit the inlining flag off the same context config.
    """

    def results_plan(parent, args, info):
        return constant(ROWS)

    def id_plan(parent, args, info):
        return access(parent, ("id",))

    return make_grafast_schema(
        ABSTRACT_SDL,
        plans={
            "Query": {"results": results_plan},
            "Image": {"id": id_plan},
            "Video": {"id": id_plan},
        },
        type_resolvers={"SearchResult": resolve_type_from_tag("__t")},
    )


def test_abstract_child_plan_inherits_flag(monkeypatch):
    """The per-concrete-type subtree's own Plan inherits inline_relations off the config.

    `abstract_child_plan` builds a SELF-CONTAINED DAG (fresh RootStep + Plan) per concrete
    type; it must stamp the same plan-level inlining decision, else a relation under a
    polymorphic field could never be folded even with inlining enabled. The subtree's Plan
    is local to that function, so we capture every Plan constructed during execution and
    assert the abstract subtrees (built when the flag is on) all carry it.
    """
    built_plans = []
    real_plan_cls = dag_module.Plan

    class _CapturingPlan(real_plan_cls):
        def __init__(self) -> None:
            super().__init__()
            built_plans.append(self)

    monkeypatch.setattr(dag_module, "Plan", _CapturingPlan)

    from graphql import graphql_sync

    schema = make_abstract_schema()
    result = graphql_sync(
        schema,
        "{ results { ... on Image { id } ... on Video { id } } }",
        execution_context_class=context_class_with(GrafastConfig(inline_relations=True)),
    )
    assert result.errors is None
    # abstract_child_plan resolves `Plan` via a deferred `from .dag import Plan`, so the
    # patch captures exactly the per-concrete-type subtree Plans (Image + Video); the
    # operation root's Plan binds dag.Plan at plan.py import time and is covered separately
    # by test_plan_operation_threads_flag_on_when_host_opts_in.
    assert len(built_plans) == 2
    assert all(p.inline_relations is True for p in built_plans)


def test_abstract_dispatch_byte_identical_under_both_flag_states():
    """No-op proof: union execution is byte-identical with the flag on vs off.

    Nothing reads `plan.inline_relations` while the flag is purely declarative, so flipping
    it must not change the result — the equivalence oracle the flag wiring preserves.
    """
    from graphql import graphql_sync

    schema = make_abstract_schema()
    query = "{ results { ... on Image { id } ... on Video { id } } }"

    off = graphql_sync(
        schema, query, execution_context_class=context_class_with(GrafastConfig())
    )
    on = graphql_sync(
        schema,
        query,
        execution_context_class=context_class_with(GrafastConfig(inline_relations=True)),
    )
    assert off.errors is None and on.errors is None
    assert off.data == on.data == {"results": [{"id": 1}, {"id": 2}]}
