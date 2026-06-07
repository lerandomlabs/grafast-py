"""grafast-py: a plan-then-execute GraphQL executor for graphql-core / Ariadne.

This is an experimental Python re-implementation of the core ideas of Graphile's
Grafast (https://grafast.org): instead of resolving fields by walking the query
tree and invoking a resolver per (field, parent) pair, we first build a static
plan (a DAG of steps) and then execute each step once over a *batch* of inputs.

The public entry point is `GrafastExecutionContext`, a drop-in replacement for
graphql-core's `ExecutionContext`. The conformance harness swaps it in globally
so the upstream graphql-core execution test-suite runs against this engine.

Injection note
--------------
graphql-core's `graphql.execution` package re-exports a *function* named
``execute`` into its namespace, which shadows the sibling ``execute`` submodule.
Because of that name collision, ``import graphql.execution.execute as m`` binds
``m`` to the function, not the module — so assigning ``m.ExecutionContext`` (as the
conformance ``conftest.py`` does) lands on the function object and never reaches
the real module dict that ``execute()`` reads ``ExecutionContext`` from at call
time. To make the swap actually take effect we install ourselves into the *real*
modules (looked up via ``sys.modules``) when this package is imported. The harness
only imports this package under ``GRAFAST=1`` (the baseline run never touches it),
so the stock baseline stays on graphql-core's own executor.
"""

import sys

from .config import (
    GrafastConfig,
    GrafastTimeoutError,
)
from .context import GrafastExecutionContext
from .entry import grafast_execute, grafast_subscribe
from .core_steps import (
    AccessStep,
    ConstantStep,
    EachStep,
    FilterStep,
    FirstStep,
    LambdaStep,
    LastStep,
    ListStep,
    LoadManyStep,
    LoadOneStep,
    NodeStep,
    ObjectStep,
    ReverseStep,
    RootStep,
    access,
    constant,
    decode_global_id,
    each,
    encode_global_id,
    filter_step,
    first_step,
    get,
    lambda_step,
    last_step,
    list_step,
    load_many,
    load_one,
    node,
    object_step,
    reverse_step,
)
from .schema import (
    FieldArgs,
    GrafastSchemaBindable,
    PlanResolver,
    TypeResolver,
    attach_plans,
    attach_type_resolvers,
    get_field_plan,
    make_grafast_schema,
    resolve_type_from_discriminator,
    resolve_type_from_tag,
    set_field_plan,
)
from .step_model import Step

# Postgres data source (the optional `pg` extra). Deliberately NOT imported here:
# the core engine depends only on graphql-core, so `import grafast_py` must not pull
# in SQLAlchemy / asyncpg. The pg symbols (PgResource, pg_select, connection, …) are
# resolved lazily via `__getattr__` below — `from grafast_py import pg_select` works
# when the extra is installed, and raises a clear install hint when it is not. Or
# import the submodule directly: `import grafast_py.pg`.

__all__ = [
    "GrafastExecutionContext",
    # function-seam entry points (run the engine without an ExecutionContext subclass)
    "grafast_execute",
    "grafast_subscribe",
    "install",
    "uninstall",
    # hardening config + error class
    "GrafastConfig",
    "GrafastTimeoutError",
    # NOTE: the Postgres data-source symbols (PgResource, PgRegistry, PgRelation,
    # PgSelectStep, PgSelectSingleStep, PgSelectAllStep, PgConnectionStep,
    # pg_select, pg_select_single, connection) are intentionally NOT in __all__:
    # they require the `pg` extra and are resolved lazily via __getattr__. They are
    # still importable explicitly (`from grafast_py import pg_select`) or via the
    # submodule (`grafast_py.pg`); see grafast_py.pg.__all__.
    # plan-resolver API
    "make_grafast_schema",
    "GrafastSchemaBindable",
    "set_field_plan",
    "attach_plans",
    "get_field_plan",
    "PlanResolver",
    "FieldArgs",
    # resolve_type bridges for Postgres-backed interfaces/unions (completion-time dispatch)
    "TypeResolver",
    "resolve_type_from_discriminator",
    "resolve_type_from_tag",
    "attach_type_resolvers",
    # step base + core steps
    "Step",
    "ConstantStep",
    "AccessStep",
    "LambdaStep",
    "ListStep",
    "ObjectStep",
    "EachStep",
    "RootStep",
    "LoadOneStep",
    "LoadManyStep",
    "NodeStep",
    "FilterStep",
    "FirstStep",
    "LastStep",
    "ReverseStep",
    # plan-helper constructors
    "constant",
    "access",
    "get",
    "lambda_step",
    "list_step",
    "object_step",
    "each",
    "load_one",
    "load_many",
    # Relay global ids + node(id) resolution
    "encode_global_id",
    "decode_global_id",
    "node",
    # list-transform helpers
    "filter_step",
    "first_step",
    "last_step",
    "reverse_step",
]


# pg symbols resolved on first access so the core stays free of SQLAlchemy/asyncpg.
_PG_LAZY = {
    "PgConnectionStep": ".pg.connection",
    "PgAggregate": ".pg.connection",
    "AGGREGATE_FUNCTIONS": ".pg.connection",
    "connection": ".pg.connection",
    "connection_needs_total": ".pg.connection",
    "connection_aggregates": ".pg.connection",
    "encode_keyset_cursor": ".pg.cursor",
    "decode_keyset_cursor": ".pg.cursor",
    "keyset_where": ".pg.cursor",
    "order_digest": ".pg.cursor",
    "PgRegistry": ".pg.resource",
    "PgRelation": ".pg.resource",
    "PgResource": ".pg.resource",
    "PgColumn": ".pg.resource",
    "PgCodec": ".pg.resource",
    "as_column": ".pg.resource",
    "PgSelectAllStep": ".pg.steps",
    "PgSelectSingleStep": ".pg.steps",
    "PgSelectStep": ".pg.steps",
    "pg_select": ".pg.steps",
    "pg_select_single": ".pg.steps",
    "PgInsertSingleStep": ".pg.mutations",
    "PgUpdateSingleStep": ".pg.mutations",
    "PgDeleteSingleStep": ".pg.mutations",
    "pg_insert_single": ".pg.mutations",
    "pg_update_single": ".pg.mutations",
    "pg_delete_single": ".pg.mutations",
    "PgExecutor": ".pg.executor",
    "PgRequestContext": ".pg.executor",
    "SQLAlchemyExecutor": ".pg.executor",
    "RawExecutor": ".pg.executor",
    "pg_request_context": ".pg.executor",
    "pg_request_context_async": ".pg.executor",
    "current_pg_request": ".pg.executor",
    "resource_from_model": ".pg.from_sqlalchemy",
    "resources_from_models": ".pg.from_sqlalchemy",
    "columns_from_table": ".pg.from_sqlalchemy",
    "OrderTerm": ".pg.ordering",
    "normalize_order": ".pg.ordering",
    "order_clauses": ".pg.ordering",
    "PgSelectQueryBuilder": ".pg.customize",
    "PgCustomizable": ".pg.customize",
    "check_predicate": ".pg.customize",
    "predicate_key": ".pg.customize",
    "resolve_customizer_predicates": ".pg.customize",
    "pg_placeholder": ".pg.placeholders",
    "placeholder_source": ".pg.placeholders",
    "Placeholder": ".pg.placeholders",
    "resolve_placeholder": ".pg.placeholders",
    "placeholder_source_tag": ".pg.placeholders",
}


def __getattr__(name: str):
    """Lazily resolve the Postgres data-source symbols (PEP 562).

    Keeps `import grafast_py` free of a hard SQLAlchemy/asyncpg dependency: the pg
    symbols load on first access. A missing `pg` extra surfaces as a clear install
    hint; any unrelated ImportError from the pg code propagates untouched.
    """
    submodule = _PG_LAZY.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    try:
        module = import_module(submodule, __name__)
    except ImportError as error:
        missing = (getattr(error, "name", "") or "").split(".")[0]
        if missing in {"sqlalchemy", "asyncpg", "greenlet", "psycopg"}:
            raise ImportError(
                f"{name!r} is part of the Postgres data source (the 'pg' extra). "
                "Install it with:  pip install 'grafast-py[pg]'"
            ) from error
        raise
    value = getattr(module, name)
    globals()[name] = value  # cache so subsequent lookups skip __getattr__
    return value


# On 3.2 both `execute` and `subscribe` are submodules carrying their own `ExecutionContext`
# fallback; on 3.3 `subscribe` was folded into `execute` (there is no `subscribe` submodule),
# and both `execute()` and `subscribe()` read `ExecutionContext` from `execute`. So patch
# whichever of these modules actually exists — a missing one is skipped, not an error.
_GRAPHQL_MODULES = ("graphql.execution.execute", "graphql.execution.subscribe")
_saved_execution_contexts: dict = {}
# P7: on graphql-core 3.3 the conformance defer/stream/subscribe tests call the MODULE
# functions ``experimental_execute_incrementally`` / ``subscribe`` (and ``execute``) directly,
# bypassing the ExecutionContext swap. So under install() we additionally replace those
# functions on the real execute module (3.3 only); uninstall() restores them. The originals are
# saved here keyed by function name.
_saved_module_functions: dict = {}


def _graphql_module(name: str):
    """Return the real graphql-core submodule, or ``None`` if it does not exist.

    Resolved via ``sys.modules`` to dodge the package's shadowing of the ``execute``
    submodule by the ``execute`` function. Returns ``None`` for a submodule absent on this
    graphql-core version (the 3.3 line dropped the standalone ``subscribe`` submodule).
    """
    module = sys.modules.get(name)
    if module is None:
        try:
            __import__(name)
        except ModuleNotFoundError:
            return None
        module = sys.modules[name]
    return module


def install() -> None:
    """OPT-IN: globally route graphql-core's `execute()`/`subscribe()` through
    grafast-py for the whole process.

    Most callers should NOT use this — pass `execution_context_class=
    GrafastExecutionContext` to `graphql()`/`execute()` (or your Ariadne/FastAPI app)
    instead, which is explicit and process-local. `install()` exists for true drop-in
    replacement scenarios where you can't thread the class through. It is idempotent;
    pair it with `uninstall()` to restore graphql-core's original executor.
    """
    for name in _GRAPHQL_MODULES:
        module = _graphql_module(name)
        if module is None:
            # submodule absent on this graphql-core version (3.3 folded `subscribe` into
            # `execute`); the surviving module's ExecutionContext covers both entry points.
            continue
        _saved_execution_contexts.setdefault(name, module.ExecutionContext)
        module.ExecutionContext = GrafastExecutionContext

    # P7: route the module-level incremental entry points through grafast on 3.3 (the
    # conformance defer/stream/subscribe tests call them directly). 3.2 lacks these names, so
    # the leg is skipped and 3.2 install is unchanged.
    from . import _compat

    if not _compat.supports_incremental():
        return
    execute_module = _graphql_module("graphql.execution.execute")
    if execute_module is None:  # pragma: no cover
        return
    from . import entry

    replacements = {
        "experimental_execute_incrementally": entry.experimental_execute_incrementally,
        "execute": entry.grafast_execute_plain,
        "subscribe": entry.grafast_subscribe,
    }
    # graphql.execution.execute is the real module; graphql.execution (the package) and
    # graphql (top-level) RE-EXPORT the same function objects, and callers do
    # `from graphql.execution import experimental_execute_incrementally` — binding the package
    # attribute at import time. So replace the name in every namespace that re-exports it, or
    # the swap never reaches an already-imported caller (the same shadowing dodge install() uses
    # for ExecutionContext).
    namespaces = [execute_module]
    for ns_name in ("graphql.execution", "graphql"):
        ns = sys.modules.get(ns_name)
        if ns is not None:
            namespaces.append(ns)
    for fn_name, replacement in replacements.items():
        for index, ns in enumerate(namespaces):
            if not hasattr(ns, fn_name):
                continue
            key = (id(ns), fn_name)
            _saved_module_functions.setdefault(key, (ns, getattr(ns, fn_name)))
            setattr(ns, fn_name, replacement)


def uninstall() -> None:
    """Undo `install()`, restoring graphql-core's original `ExecutionContext` + functions."""
    for name, original in list(_saved_execution_contexts.items()):
        module = sys.modules.get(name)
        if module is not None:
            module.ExecutionContext = original
    _saved_execution_contexts.clear()

    for _key, (ns, original) in list(_saved_module_functions.items()):
        fn_name = _key[1]
        setattr(ns, fn_name, original)
    _saved_module_functions.clear()


# NOTE: install() is deliberately NOT called on import. Importing grafast_py has no
# global side effect on graphql-core; use execution_context_class=... (preferred) or
# call install() explicitly to opt into the process-wide patch.
