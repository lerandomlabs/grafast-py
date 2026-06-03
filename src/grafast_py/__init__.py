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
from .core_steps import (
    AccessStep,
    ConstantStep,
    EachStep,
    LambdaStep,
    ListStep,
    LoadManyStep,
    LoadOneStep,
    ObjectStep,
    RootStep,
    access,
    constant,
    each,
    get,
    lambda_step,
    list_step,
    load_many,
    load_one,
    object_step,
)
from .schema import (
    FieldArgs,
    GrafastSchemaBindable,
    PlanResolver,
    attach_plans,
    get_field_plan,
    make_grafast_schema,
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
]


# pg symbols resolved on first access so the core stays free of SQLAlchemy/asyncpg.
_PG_LAZY = {
    "PgConnectionStep": ".pg.connection",
    "connection": ".pg.connection",
    "PgRegistry": ".pg.resource",
    "PgRelation": ".pg.resource",
    "PgResource": ".pg.resource",
    "PgSelectAllStep": ".pg.steps",
    "PgSelectSingleStep": ".pg.steps",
    "PgSelectStep": ".pg.steps",
    "pg_select": ".pg.steps",
    "pg_select_single": ".pg.steps",
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


_GRAPHQL_MODULES = ("graphql.execution.execute", "graphql.execution.subscribe")
_saved_execution_contexts: dict = {}


def _graphql_module(name: str):
    """Return the real graphql-core submodule (via sys.modules to dodge the package's
    shadowing of the `execute` submodule by the `execute` function)."""
    module = sys.modules.get(name)
    if module is None:
        __import__(name)
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
        _saved_execution_contexts.setdefault(name, module.ExecutionContext)
        module.ExecutionContext = GrafastExecutionContext


def uninstall() -> None:
    """Undo `install()`, restoring graphql-core's original `ExecutionContext`."""
    for name, original in list(_saved_execution_contexts.items()):
        module = sys.modules.get(name)
        if module is not None:
            module.ExecutionContext = original
    _saved_execution_contexts.clear()


# NOTE: install() is deliberately NOT called on import. Importing grafast_py has no
# global side effect on graphql-core; use execution_context_class=... (preferred) or
# call install() explicitly to opt into the process-wide patch.
