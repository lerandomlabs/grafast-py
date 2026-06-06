# grafast-py

A **plan-then-execute** GraphQL execution engine for
[graphql-core](https://github.com/graphql-python/graphql-core) — a drop-in
`ExecutionContext` with real [Grafast](https://grafast.org)-style **automatic
batching** (the N+1 fix). Framework-agnostic: works with any graphql-core server
([Ariadne](https://ariadnegraphql.org/), FastAPI, Strawberry-over-core, …). An
experimental Python re-implementation of the core ideas of Graphile's Grafast.

## The Grafast idea in 3 lines

1. Build a static **plan** — a DAG of **steps** — from the operation, once.
2. Execute each step **once over a whole batch** of inputs (a "bucket"), not once per
   (field, parent) pair.
3. Batching — the DataLoader N+1 problem — becomes **automatic**: a `loadMany` over N
   parents sees all N keys in a single call and fires its loader exactly once.

## Install

```bash
uv add grafast-py          # or: pip install grafast-py   (core dep: graphql-core only)
# optional extras:
uv add 'grafast-py[pg]'        # Postgres data source (SQLAlchemy + asyncpg)
uv add 'grafast-py[structlog]' # structured kv logging (else a stdlib shim is used)
```

The **core depends only on `graphql-core`** — it is framework-agnostic (any
graphql-core server) and data-source-agnostic (the generic `load_one`/`load_many`
batch steps work against anything). The Postgres data source (`grafast_py.pg`) is the
optional `[pg]` extra; its symbols import lazily and raise a clear install hint if the
extra is missing. The package ships a `py.typed` marker for typing consumers.

Ariadne integration needs **no** extra — it works through graphql-core (see below).

## graphql-core / Ariadne integration

It is a drop-in `ExecutionContext`. With graphql-core:

```python
from graphql import execute, parse
from grafast_py import GrafastExecutionContext

result = execute(
    schema,
    parse(query),
    execution_context_class=GrafastExecutionContext,
)
```

With Ariadne, bind your plan resolvers via the `GrafastSchemaBindable` and pass the
context class through to `graphql`/`graphql_sync`:

```python
from ariadne import make_executable_schema
from grafast_py import GrafastSchemaBindable, GrafastExecutionContext

schema = make_executable_schema(type_defs, GrafastSchemaBindable(plans))
# then call ariadne's graphql(..., execution_context_class=GrafastExecutionContext)
```

Fields **without** a plan resolver keep the ordinary per-parent resolver path, so an
existing plain-resolver schema runs unchanged (this is exactly why the full
graphql-core 3.2.8 execution conformance suite passes against this engine).

## Plan-resolver example (in-memory, no DB)

A field's plan resolver `($parent_step, args, info) -> Step` returns a step; the
engine runs it once per bucket. Here `posts` is a `load_many` that collapses the
classic per-author N+1 into a single call:

```python
from grafast_py import access, load_many, make_grafast_schema

def plan_posts(parent, args, info):
    return load_many(access(parent, ["id"]), load_posts_by_author_ids)

schema = make_grafast_schema(SDL, {
    "Author": {"posts": plan_posts, ...},
    ...
})
```

Runnable: [`examples/plan_blog.py`](examples/plan_blog.py) — prints `load_posts called
once with author_ids=[1, 2, 3]`, proving the one-call payoff.

```bash
uv run python examples/plan_blog.py
```

## Postgres example (O(depth) SQL)

The `grafast_py.pg` data source mirrors Grafast's `@dataplan/pg`: a `PgResource`
(table / columns / pk / relations) with batched `pg_select` / `pg_select_single`
steps that emit one `WHERE col = ANY($1)` statement per resource layer, plus a
batched Relay `connection`. A depth-D nested query issues ~D batched SQL statements
total — **O(depth), not O(rows)**:

```
authors { posts { author, comments { author } } }   ->  exactly 5 SQL statements
```

That O(depth) batching property holds across the full feature set:

- **Structured `ORDER BY`** — per-term direction + `NULLS FIRST/LAST`, multi-column,
  with a primary-key tie-break appended for a stable non-unique order.
- **Filtering** — a resource `select_customizer(context)` (the selectAuth analogue for
  soft-delete / tenant scoping) AND per-plan `.where(<Core predicate>)`, AND-combined
  onto the batched `WHERE` before paging.
- **Per-parent paging** — `first` / `offset` slice **each parent's** rows in SQL via a
  `row_number()` window partitioned by the match column (never a bucket-wide `LIMIT`).
- **Keyset Relay connections** — forward (`first`/`after`) **and** reverse
  (`last`/`before`), sliced in SQL by a seek predicate (not an offset), with opaque
  digest-validated cursors (a cursor minted under a different ordering is rejected, not
  misapplied) and a **separate** batched `totalCount` aggregate issued only when selected.
- **Computed columns + minimal codecs** — a host-authored Core SQL expression projected
  as an extra labelled column in the same select, and a per-attribute `to_py` / `to_pg`
  codec applied uniformly on read / write. Computed columns are **projection-only** in v1:
  to order or filter, use a stored column or inline the SQL expression (ordering by a
  computed column raises a clear error; orderable/filterable computed columns are deferred).
- **CRUD mutations** — `pg_insert_single` / `pg_update_single` / `pg_delete_single` on
  the serial mutation seam, fully param-bound, `RETURNING` a projectable row.
- **Per-request `pgSettings` / RLS** and **bring-your-own-pool** (the production path) —
  see below.

```python
from grafast_py.pg import OrderTerm, connection

# a forward keyset Relay connection over Author.posts, newest first, with totalCount:
connection(
    posts, key_step=author_id, match_column="author_id",
    order_by=[OrderTerm("created", descending=True), OrderTerm("id")],
    first=20, after=cursor, needs_total=True,
)
```

Runnable: [`examples/pg_blog.py`](examples/pg_blog.py) (uses the scratch DB
`grafast_py_test`, schema `grafast_demo`):

```bash
uv run python examples/pg_blog.py
```

`grafast_py.pg` is the canonical import path for the data source; the most-used
symbols are also re-exported at the top level for convenience.

### Define resources from your SQLAlchemy models

If your tables are already mapped as SQLAlchemy declarative models, you can derive
the `PgResource` descriptors instead of re-typing table / column / relation metadata:

```python
from grafast_py import resources_from_models
from myapp.models import Author, Post, Comment

registry = resources_from_models([Author, Post, Comment])
schema = build_my_schema(registry)   # you still write the SDL + plan resolvers
```

This is **resources-only** — it derives table, columns, primary key and hasOne/hasMany
relations from the ORM. It does **not** generate the GraphQL SDL or the plan resolvers
(you still write those) and does not map column types/codecs. `resource_from_model(model, …)`
builds a single resource; `resources_from_models([...])` builds a `PgRegistry` and wires
relations between models that are both in the batch.

Limits (pass an override or the relation is skipped): a composite/absent primary key
needs `primary_key="…"`; many-to-many relations, composite-foreign-key relations, and
relations whose target model is not in the batch are skipped with a warning (set
`strict=True` to raise instead).

## Hardening config (opt-in)

All production controls are opt-in via a `GrafastConfig` on the context class; the
defaults reproduce the un-hardened behaviour exactly.

```python
from grafast_py import GrafastConfig, GrafastExecutionContext

class HardenedContext(GrafastExecutionContext):
    grafast_config = GrafastConfig(
        execution_timeout_s=5.0,    # async-path wall-clock budget
        max_step_concurrency=32,    # cap in-flight step fan-out (see pool note below)
        # tracing hooks (no-ops by default); each may return a context-manager span:
        on_operation=my_op_span,    # (context, operation)
        on_plan=my_plan_span,       # (context, operation)
        on_step_batch=my_batch_span # (step, count) — the batch boundary
    )
```

Point the Postgres data source at **your** database and size its pool — pass `url=`
or set the `GRAFAST_PG_URL` env var (the library bakes in no database):

```python
from grafast_py.pg import configure_engine
configure_engine(
    url="postgresql+asyncpg://user:pass@host/dbname",
    pool_size=32, max_overflow=8, pool_timeout=30,
)
```

**Concurrency / pool relationship:** the real ceiling on concurrent in-flight DB
connections is `pool_size + max_overflow` (SQLAlchemy enforces it; excess operations
queue on checkout, raising the latency tail). Size the pool to your target
concurrency. `max_step_concurrency` is a secondary throttle on the engine's own step
fan-out, not the DB bound — the pool is. For per-statement bounds, set a server-side
`statement_timeout` via `connect_args`.

### Bring your own pool (the production path)

`get_engine` / `configure_engine` are a **demo/test convenience** (one process-global
engine the examples and the test suite drive). In production you inject a **host
executor** per request — nothing on the hot path calls `get_engine`; the pg steps run
their statements through the request-scoped executor. Wrap **your** engine/pool and bind
it around `await graphql(...)`:

```python
from grafast_py.pg import SQLAlchemyExecutor, pg_request_context

with pg_request_context(SQLAlchemyExecutor(my_async_engine)):
    result = await graphql(schema, query, execution_context_class=GrafastExecutionContext)
```

`SQLAlchemyExecutor(engine)` runs the built Core statement on **your** `AsyncEngine`.
To run on a non-SQLAlchemy pool instead, supply a `RawExecutor(run_callable)`: it
compiles the statement to `$1` positional SQL and hands it to your callback to run on
your raw pool (the `= ANY(:keys)` batch stays a single `$1` array param). (A
single-shared-request-transaction executor mode — one txn spanning every step of a
request — is a future addition, not built yet.)

### Per-request `pgSettings` / RLS

Pass `settings={...}` to `pg_request_context` to apply Postgres GUCs for the request.
`SQLAlchemyExecutor` opens a transaction, applies all settings with
`set_config(key, value, true)` (transaction-local, so they auto-clear at commit and
never leak onto a pooled connection), and runs the query in that **same** transaction —
so a row-level-security policy referencing `current_setting('app.owner', true)` sees the
per-request value:

```python
with pg_request_context(SQLAlchemyExecutor(engine), settings={"app.owner": str(viewer_id)}):
    result = await graphql(schema, query, execution_context_class=GrafastExecutionContext)
```

grafast-py's responsibility ends at injecting the GUCs into the query's transaction;
enforcement is your `CREATE POLICY` + `ENABLE ROW LEVEL SECURITY` and a database role
that does **not** bypass RLS (a superuser/`BYPASSRLS` role is never subject to a policy).
With `settings=None` the executor takes the plain no-transaction path, so the O(depth)
batching profile is unchanged.

### Query cost / depth limiting — use your validation layer

grafast-py is a drop-in `ExecutionContext`, and **validation runs before execution**,
so query cost/depth limiting is the validation layer's job and composes with this
engine for free. It is deliberately **not** built into the engine (the executor isn't
the right layer, and a baked-in static guard isn't `first:`-aware). Add a validation
rule — e.g. **Ariadne's `cost_validator`** (`first:`-aware via cost-map multipliers) or
[`graphql-cost-analysis`](https://github.com/pa-bru/graphql-cost-analysis) — to your
server:

```python
from ariadne.asgi import GraphQL
from ariadne.validation import cost_validator
from grafast_py import GrafastExecutionContext

app = GraphQL(
    schema,
    execution_context_class=GrafastExecutionContext,   # batched execution
    validation_rules=[cost_validator(maximum_cost=1000, cost_map=COST_MAP)],  # cost limit
)
```

## Status / caveats

This engine **passes a rigorous internal gate set**: the full graphql-core 3.2.8
execution conformance suite (302 on the stock executor / 300 + 2 skipped through this
engine), differential parity vs the reference Node Grafast (`tests/differential/`), an
O(depth) N+1 benchmark, and a concurrent soak. **"Passes our gates" is not the same as
"validated against your production workload."** Before running with real money: pin
the version, run the differential harness against **your** schema and fixtures,
load-test with **your** pool size and concurrency, and commission a security review.

Not yet covered: `@defer`/`@stream` incremental delivery (graphql-core 3.3); multiple
databases in one process (one engine per URL — dispose+reconfigure to switch); and the
execution timeout bounds the caller but does not itself cancel in-flight SQL (pair it
with a server-side `statement_timeout`). Query cost/depth limiting is by design **not**
in this engine — do it in your validation layer (see above).

**Deferred in the `grafast_py.pg` data source** (single-column / single-statement paths
are built; these are not): multi-column relation keys (relations match on a single FK
column); runtime `from_step` placeholders (a relation's match value comes from the parent
row's column, not an arbitrary upstream step); a single-shared-request-transaction
executor mode (one txn spanning every step of a request); query inlining / `LATERAL`
(each resource layer is its own batched `= ANY($1)` round-trip, not folded into the
parent's query); `GROUP BY` / `HAVING` / aggregates beyond the connection's own
`totalCount`; and a full codec **types** table (codecs are minimal `to_py` / `to_pg`
hooks plus an optional cast `sql_type` for the keyset path, not a complete PG type registry).

## More

- Examples: [`examples/`](examples/)
- Running the test / conformance / differential / benchmark suites: [`AGENTS.md`](AGENTS.md)

## License

MIT — see [`LICENSE`](LICENSE).
