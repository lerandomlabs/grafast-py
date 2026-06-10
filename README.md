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

## The core design choice (and its trade-off)

grafast-py reuses graphql-core's **type system, parser, validation and leaf helpers**
(argument/variable coercion, scalar serialize, `GraphQLError`, `located_error`) but
**owns execution**: it is a genuine plan-then-execute engine, not a field-by-field walk
on top of graphql-core's `complete_value`. You adopt it either as a drop-in
`ExecutionContext` (`execution_context_class=GrafastExecutionContext`) or through the
function seam (`grafast_execute` / `grafast_subscribe`); both run the *same* pipeline.
That single decision defines what this library is and isn't — read it before adopting:

**What it buys you**

- **Adoption is a one-liner** (`execution_context_class=GrafastExecutionContext`, or call
  `grafast_execute`): no new server, no rewrite, and planned fields coexist with ordinary
  graphql-core resolvers in the *same* schema.
- **A small, legible core** (a fraction of upstream's ~26k lines): an explicit **two-tree
  model** — a `LayerPlan` (batched execution) de-fused from an `OutputPlan` (serialization),
  mirroring upstream Grafast's substrate at a fraction of its size — rather than a
  field-by-field walk fused to graphql-core's completion.
- **One execution path**: every field is a *step* — a plan step, or a resolver-adapter
  (`ResolveStep`) wrapping a plain resolver — so a plain-resolver schema and the full
  graphql-core conformance suite run through the same machinery, byte-identically to stock
  graphql-core. graphql-core's own execution conformance suite is the oracle.
- **The asymptotic win equals upstream**: O(depth) batched SQL, not O(rows) — the N+1 fix
  is fully here.
- **Incremental delivery + subscriptions** ship on the graphql-core 3.3 line: `@defer`,
  `@stream` and subscriptions run through the owned `OutputPlan` / incremental driver (see
  the 3.3 tier below).

**What it costs**

- The gap vs upstream Grafast is **constant-factor** (round-trips, a few redundant step
  runs), **not asymptotic** — and the optimizers (LATERAL inlining, cross-parent hoisting,
  plan caching) intentionally cover fewer cases and **fall back** to the batched path on any
  uncertain shape. Cross-parent hoisting is **on by default** (like upstream Grafast, proven
  byte-identical by its oracle); inlining and plan caching are off by default (opt-in
  `GrafastConfig` flags).
- We remain **coupled to graphql-core's type system, validation and execution semantics**
  (the deliberate reuse), and to its release lines: incremental delivery / subscriptions
  ride the **graphql-core 3.3 alpha** line (the 3.2 line has no incremental delivery).
- The 3.3 `@defer`/`@stream` tier carries **7 documented payload-grouping cases**: the
  response *data* is byte-identical to upstream, but the *chunk grouping/ordering* of the
  incremental payloads can differ (identical after client reassembly — see the tier below).

**In one line:** if you want Grafast-style batching for an *existing* graphql-core / Ariadne
app — on either the 3.2 or the 3.3 line — this is the right tool. If you need upstream-class
*optimization power* across every shape, that is upstream Grafast (≈ its size), which this
library deliberately is **not**.

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

Fields **without** a plan resolver are adapted into a resolver step (`ResolveStep`) on the
**same** unified execution path — there is no separate per-parent resolver path. Every
field is a step (a plan step or a `ResolveStep`), so an existing plain-resolver schema runs
byte-identically to stock graphql-core (this is exactly why the full graphql-core execution
conformance suite passes against this engine).

### The function seam (no `ExecutionContext` subclass)

`GrafastExecutionContext` is a thin **drop-in shim**: its `execute_operation` delegates
straight into the same core (`run_planned_operation`) that the function-level entry points
run, so the two share one pipeline. When you can't thread an execution-context class
through your host, call the engine directly — it returns a plain `graphql.ExecutionResult`
on every version:

```python
from graphql import parse
from grafast_py import grafast_execute, grafast_subscribe

result = grafast_execute(schema, parse(query), variable_values=vars)

# subscriptions (graphql-core 3.3): grafast_subscribe mirrors graphql-core's maybe-awaitable
# shape — it returns the mapped AsyncIterator directly when source-stream creation is sync,
# or a coroutine of it when async; await only if awaitable.
stream = grafast_subscribe(schema, parse(subscription))
```

`grafast_execute` accepts the same arguments as graphql-core's `execute` (document/string,
`root_value`, `context_value`, `variable_values`, `operation_name`, resolvers, `config`).

### graphql-core 3.2 and 3.3 (dual-version)

The engine runs on **both** the graphql-core 3.2 line and the 3.3-alpha line. Version-specific
differences (`get_field_def`, `collect_fields`, error-sort, the incremental classes) are isolated
behind a feature-probed seam in `grafast_py._compat`, so one engine core serves both. The pin is
`graphql-core>=3.2.8` (no upper cap), which admits either line. CI runs a two-leg matrix proving
the seam on both.

To run the 3.3 leg locally, build the sibling env and use it directly (the default `.venv`
keeps graphql-core 3.2):

```bash
bash scripts/venv33.sh                                   # builds .venv33 with graphql-core 3.3 alpha
.venv33/bin/python scripts/fetch_conformance.py 3.3.0a12 # fetch the matching conformance suite
.venv33/bin/python -m pytest conformance/_suite/execution -p anyio -q
```

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
- **Filtering** — a resource `select_customizer` (the selectAuth analogue for soft-delete /
  tenant scoping, applied to **reads only** — see note below), per-plan `.where(<Core
  predicate>)`, AND a structured **filter `Condition` tree** (`And` / `Or` / `Not` + leaf ops
  `eq` / `ne` / `lt` / `le` / `gt` / `ge` / `in` / `like` / `ilike` / `is_null`) that compiles
  to a Core predicate — all AND-combined onto the batched `WHERE` before paging. The customizer
  has two forms: `customizer(context)` inlines values as plan-time literals (value-specific, so
  **not** plan-cacheable under `cache_plans`); `customizer(context, sources)` uses
  `sources.placeholder(key)` to emit a value-less placeholder read from `context[key]` per
  request at execute, so the plan stays value-independent and **cacheable** (structure fixed at
  plan time, value supplied per request — the convergence to upstream `selectAuth`). A customizer
  that varies only its *values* across requests (the common tenant case) is shared from the cache
  and re-binds per request. A customizer that varies its predicate *structure* by context (e.g.
  `return [] if ctx["role"] == "admin" else [scope]`) is still **correct** — a structural-
  divergence guard re-resolves the customizer on a cache hit and re-plans whenever the shape
  differs from the cached one, so a request can never inherit another request's customizer-decided
  structure — it just doesn't share one cached plan across the differing shapes.
- **Per-parent paging** — `first` / `offset` slice **each parent's** rows in SQL via a
  `row_number()` window partitioned by the match column (never a bucket-wide `LIMIT`).
- **Keyset Relay connections** — forward (`first`/`after`) **and** reverse
  (`last`/`before`), sliced in SQL by a seek predicate (not an offset), with opaque
  digest-validated cursors (a cursor minted under a different ordering is rejected, not
  misapplied) and a **separate** batched `totalCount` aggregate issued only when selected.
- **Connection aggregates** — `sum` / `avg` / `min` / `max` / `count(distinct)` over the
  (filtered) per-parent set, plus optional `GROUP BY`, issued as a **separate** batched
  statement only when selected (the same batching shape as `totalCount`; `min`/`max` and
  group-by keys decode through the resource codec, consistently with node columns).
- **Composite (multi-column) relation keys** — a relation matches on a single FK column
  **or** a column tuple (`(a, b) IN (…)`), batched in one statement and grouped on the tuple.
- **Computed columns + a codec type library** — a host-authored Core SQL expression
  projected as an extra labelled column in the same select, and a per-attribute codec
  applied uniformly on read / write. Codecs cover the scalar PG types plus recursive
  **arrays / ranges / enums / composites** (and feed the keyset CAST). Computed columns are
  **projection-only** in v1: to order or filter, use a stored column or inline the SQL
  expression (ordering by a computed column raises a clear error; orderable/filterable
  computed columns are deferred).
- **Relay `node(id)` + list transforms** — global-id encode/decode and a typed `node`
  step that batches one load **per type**, plus `filter` / `first` / `last` / `reverse`
  list-transform steps (core, source-agnostic — no SQLAlchemy).
- **CRUD mutations** — `pg_insert_single` / `pg_update_single` / `pg_delete_single` on
  the serial mutation seam, fully param-bound, `RETURNING` a projectable row.
- **GraphQL interfaces / unions over Postgres** — two flavours, both resolved at
  **completion time** (see below): **single-table inheritance**, where one table's rows
  carry a discriminator column and a `resolve_type` bridge maps that column to the concrete
  type (`resolve_type_from_discriminator("kind", {...})`); and **cross-table `pgUnionAll`**,
  where the concrete types live in separate tables and one batched `UNION ALL` (a shared
  NULL-padded projection + a `__typename` tag per branch, `resolve_type_from_tag`) fetches
  all members, keyset-paged as a Relay connection (forward + reverse, separate `totalCount`)
  — at most **two** statements per union layer regardless of member count.
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

### Interfaces / unions: the completion-time-dispatch model

Polymorphism is **not** a plan-time bucket system. The engine already dispatches
interfaces and unions at **completion** time: for an abstract field it resolves each
value's concrete type (via the abstract type's `resolve_type`, falling back to a
context `type_resolver`), **groups** the values by concrete type, and for each group
plans + executes that concrete type's sub-selection exactly like a normal object field.
So a Postgres-backed interface/union is just **(a)** a pg step/field that produces
type-tagged **row values** (the discriminator column, or the `pgUnionAll` `__typename`
tag), and **(b)** a `resolve_type(row, info, abstract_type) -> typename` bridge wired
onto the abstract type. Each concrete-type group becomes its own batched bucket, so a
member's type-specific fields **and its nested pg relations batch per concrete-type
group** — with no plan-time polymorphism. The only engine support this needs is that the
per-concrete-type child plan is built as a **self-contained step subtree** (its own
`RootStep`, deduplicated and remapped, mirroring the operation root) so plan-resolver
subfields under an abstract type actually plan into steps; that one minimal change lives
in the completion dispatch and leaves graphql-core's interface/union conformance green.

Wire the bridges with `make_grafast_schema(SDL, plans, type_resolvers={...})` (or
`attach_type_resolvers`); a name that is not an abstract type fails loud, mirroring
`attach_plans`. Runnable: [`examples/poly_demo.py`](examples/poly_demo.py) (single-table
discriminator + cross-table `pgUnionAll`, on the `grafast_demo` scratch DB).

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

Limits: a composite/absent primary key needs `primary_key="…"` via the single-resource
`resource_from_model(model, primary_key=…)` — the batch `resources_from_models([...])` has
no per-model PK override yet, so a composite-**primary**-key model goes through the single
form. Many-to-many relations and relations whose target model is not in the batch are
skipped with a warning (set `strict=True` to raise instead). Composite **foreign**-key
relations **are** wired (matched on the column tuple).

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
your raw pool (the `= ANY(:keys)` batch stays a single `$1` array param).

For a **single-shared-request-transaction** mode — one `REPEATABLE READ` connection
spanning every statement of a request, so all batched reads observe one consistent
snapshot and the pgSettings apply once — use the async form with `shared_txn=True`:

```python
from grafast_py.pg import SQLAlchemyExecutor, pg_request_context_async

async with pg_request_context_async(SQLAlchemyExecutor(my_async_engine), shared_txn=True):
    result = await graphql(schema, query, execution_context_class=GrafastExecutionContext)
```

The trade-off: because every statement shares the one held connection, the request's read
fan-out runs **serially** (SQLAlchemy serialises statements on a single connection) — enable
it when snapshot consistency matters more than intra-request query concurrency. The sync
`pg_request_context` rejects `shared_txn` (it cannot await the connection teardown on exit).

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

RLS is also the **write** authorization boundary. A resource `select_customizer` scopes
**reads only** — `pg_insert_single` / `pg_update_single` / `pg_delete_single` do **not**
apply it (update/delete are primary-key keyed), matching upstream Grafast, where `selectAuth`
is a read-time concern. To scope writes, rely on RLS policies (`USING` for `UPDATE`/`DELETE`,
`WITH CHECK` for `INSERT`/`UPDATE`) via the same per-request `pgSettings`. Do not assume
`select_customizer` protects writes — it does not.

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

## LATERAL relation inlining (experimental, opt-in)

By default every resource layer is its own batched `WHERE col = ANY($1)` round-trip — the
O(depth) profile above and the correctness baseline. **Relation inlining** is an opt-in
optimization that folds a *provably-safe* child relation into its parent's statement via a
`LEFT JOIN LATERAL`, so a parent + its inlined children become **one** statement instead of
one-per-layer. It is **opportunistic and equivalence-preserving**: with inlining on, the
result is **byte-identical** to the batched path (same data, same list order, same `[]` /
`null` for empty children) — it changes only the *number* of SQL statements, never the data.

It is **experimental and off by default** (`inline_relations` defaults to `False`).
Turn it on globally on the context class:

```python
from grafast_py import GrafastConfig, GrafastExecutionContext

class InlinedContext(GrafastExecutionContext):
    grafast_config = GrafastConfig(inline_relations=True)   # default False
```

With the flag off (the default) the optimize pass is a **no-op** — every pg step's
`optimize` short-circuits to identity, so the executed plan is byte-identical to one
compiled without the feature at all.

### Per-resource opt-out

Even with the global flag on, a single table can be excluded — useful for a table whose
codecs or `LATERAL` semantics you'd rather not fold while keeping inlining everywhere else:

```python
from grafast_py.pg import PgResource

PgResource("events", schema="grafast_demo", table="events", columns=[...],
           opt_out_inline=True)   # never folded as a parent NOR inlined as a child
```

`opt_out_inline` has no effect when the global flag is off (nothing inlines anyway).

### What inlines — and what falls back (SKIP scope)

Inlining is **conservative by construction**: a child is folded **only** when *every*
safety condition is provably met; on *any* unmet or uncertain condition the child is
**SKIPPED** — it stays a standalone batched `= ANY($1)` statement, exactly as the planner
built it. Unsafe-but-skipped is always correct (a redundant extra statement, never wrong
data); the predicate never inlines when in doubt. Currently inlined:

- **hasOne** (e.g. `Post.author`) — folded as `to_jsonb(child) … LIMIT 1`, scattered as the
  single row or `null`.
- **unpaginated hasMany** (e.g. `Author.posts`) — folded as
  `coalesce(json_agg(to_jsonb(child) ORDER BY …), '[]')`, scattered as the list (or `[]` for
  a parent with no children), reproducing the child's exact `ORDER BY` including its
  primary-key tie-break.
- **multi-level nesting** composes — `authors → posts → comments` folds to **one**
  statement (the optimize pass runs to a fixpoint, so a parent re-folds after its own child
  folded).

**Not inlined** (kept on the batched path, still byte-identical via fallback):

- **paginated / keyset connections** (`PgConnectionStep`: `first` / `offset` / `after` /
  `before`, `totalCount`, aggregates) — always skipped.
- **per-parent limited hasMany** (a `first` / `offset` window slice on a relation) — skipped
  (the per-parent `LIMIT`-via-`row_number()` fold is deferred).
- **filtered children** — a child carrying a host `.where()` or a resource
  `select_customizer` scope is skipped (reproducing an arbitrary correlated predicate inside
  the `LATERAL` with a proven-identical customization signature is deferred); an *unfiltered*
  child folds.
- **composite (multi-column) FK relations** — skipped (only single-column FK correlation is
  currently folded).
- **non-JSON-stable codecs** — a child is folded only when every projected column survives
  the `to_jsonb → JSON → to_py` round-trip to the *same* Python value as the batched row
  decode; a column whose codec is not on the json-safe allowlist (e.g. some
  timestamptz / numeric / array / range / composite codecs) skips the fold.
- **cross-schema / cross-executor** resources, and any **mutation** (a write step is never
  inlined).

Because the fold is gated this strictly, you can flip the flag on for an existing suite as
the broadest possible equivalence oracle: `GRAFAST_INLINE_RELATIONS=1 uv run pytest tests -m
pg` re-runs the whole Postgres suite with inlining forced on and every existing exact-data
assertion still holds.

## Plan caching + runtime placeholders (experimental, opt-in)

By default every request **re-plans** from scratch (the selection set → step-DAG build runs each
time) and every `$variable` value is **inlined into the SQL as a literal** at plan time — so two
requests of the same document with different variable values produce *different* SQL text. That is
the correctness baseline: a literal still inlines and **dedups by value**, so two steps that differ
only by a filter value get different dedup keys and never merge.

Two opt-in flags (both default **OFF**) let a host reuse a plan across requests:

- **`placeholders`** turns on per-argument *variable provenance*. The planner walks each field's
  AST and records which arguments came from a `$variable`; a plan resolver can then ask
  `field_args.is_variable("status")` and, for a variable-derived value, build a **value-agnostic
  placeholder** instead of inlining the literal:

  ```python
  from grafast_py.pg import pg_placeholder

  def widgets_plan(parent, args, info):
      step = pg_select(widgets).for_parent(parent)
      if args.is_variable("status"):
          # value-agnostic: the SQL renders `status = $1`, tagged by the SOURCE "var:status"
          step.builder().where(column("status") == pg_placeholder(args.source("status"), args["status"]))
      else:
          step.builder().where(column("status") == args["status"])   # literal, inlines + dedups by value
      return step
  ```

  A placeholder **dedups by its SOURCE identity** (the variable name), *never* by the runtime value:
  two steps over the same source merge; two over different sources do not; a placeholder step and a
  coincidentally-equal *literal* step do **not** merge (their SQL differs — `$1` vs an inlined
  literal); and existing literal-only steps keep their value-included keys **unchanged**. The host
  owns the predicate, so it opts in **per value** — there is no auto-placeholdering.

- **`cache_plans`** reuses a finalized plan across requests of the same document, keyed by
  `(schema identity, document text, operation name, variable-arg fingerprint)`. **Only a
  value-independent plan is cached** — every SQL-affecting variable value must be either a
  same-every-request literal or a value-agnostic placeholder; a plan that inlined a `$variable` as a
  literal is value-specific and is *never* cached for reuse. On a cache **hit** the stored plan is
  re-used **without deepcopy** — per-request variable values are **render-injected** at execute
  time, never mutated into the shared plan, so the cached plan is safe to share across concurrent
  requests; each placeholder is **re-bound** to this request's variable values (the cached SQL is
  value-agnostic, so only the bound values move). It is a pure optimization: a hit changes only
  *whether* planning re-runs — never the SQL text or the result data.

  ```python
  from grafast_py import GrafastConfig, GrafastExecutionContext

  class CachedContext(GrafastExecutionContext):
      grafast_config = GrafastConfig(placeholders=True, cache_plans=True)   # both default False
  ```

  The cache is a bounded **LRU** (`max_entries`, default 1000; set your own via
  `GrafastConfig(plan_cache=PlanCache(max_entries=...))`), so an adversarial stream of unique
  documents cannot grow it without bound.

### ⚠️ Cache-safety: never bake a per-request value (the one contract you must honour)

`cache_plans` keys a plan by **document text** (plus schema / operation / variable-arg fingerprint) —
**not** by your request context. So any *per-request* value — a tenant id, a user id, an RLS scope
read from `info.context` — that you **bake as a literal** into the SQL is frozen into the *shared*
cached plan, and the next request for the same document (a **different tenant**) is served that first
request's value. That is a **cross-tenant data leak**. The rule is one line:

**Scope by a placeholder, never by a baked context literal.**

```python
from grafast_py.pg import pg_placeholder

# ❌ LEAKS under cache_plans: the literal 42 is frozen into the shared plan
step.builder().where(column("tenant_id") == info.context["tenant_id"])

# ✅ SAFE + cacheable: value-LESS, re-bound to THIS request's context[...] at execute (fails LOUD
#    if the key is missing — never a silent `col = NULL` that would widen the scope)
step.builder().where(column("tenant_id") == pg_placeholder("ctx:tenant_id"))
```

Two paths are **auto-protected** for you: a resource `customizer(context)` that bakes a literal, and
an inlined `$variable`, are both detected and forced **non-cacheable** (that plan just re-plans every
request). But a raw `.builder().where(...)` predicate and a `pgUnionAll` member `where=` are **not**
inspected — and that is deliberate. By the time the engine sees `column == 42`, the value has already
evaluated in Python, so a per-request `context["tenant_id"]` is **indistinguishable** from a safe
constant like `status == "published"`; auto-rejecting *every* literal would force you to wrap harmless
constants in placeholders too. So on those two paths the contract is **yours to honour**.

> The proper fix — *tracking* context reads at the source so the engine can tell a context value from
> a constant automatically (the way upstream Grafast's tracked-value steps do), and so keep constants
> cacheable while catching context-bakes for free — is a planned follow-up. Until then: use the
> placeholder.

This only bites under `cache_plans=True`; with caching off, every request re-plans and a baked literal
is correct (just not reused).

With both flags off (the default) `plan_operation` never reads or writes the cache and
`FieldArgs.is_variable` is always `False`, so every host falls back to literal inlining and the
executed plan is **byte-identical** to a build without these features at all.

Like inlining, you can flip these on for the whole suite as the broadest equivalence oracle:
`GRAFAST_CACHE_PLANS=1 uv run pytest tests` re-runs everything with caching (+ placeholders) forced
on, and `GRAFAST_PLACEHOLDERS=1 uv run pytest tests` exercises the placeholder dedup path *without*
caching — every existing exact-data **and** statement-count assertion still holds, because a cache
hit changes only whether planning re-runs and a placeholder changes only how a value-agnostic value
is dedup-keyed, never the SQL or the data.

## Cross-parent step hoisting (on by default)

`hoist` (a `GrafastConfig` field, default **ON** — like upstream Grafast) is a pure optimization that **lifts** a step
whose inputs are constant across a child bucket up into a shallower layer, so it runs
once-per-request (or once-per-few-parents) instead of once-per-child-bucket. It changes *where* a
step runs, never *whether* it runs — no step is replaced, no `FieldPlan.step` reference moves, only
the column's production layer shifts — so the data is **byte-identical** to the naive plan; the
hoisted column is produced in the parent bucket and threaded down to the child bucket as an extra
`parent_store` seed (the child reads it, never re-runs it). It is **disabled entirely under
mutations** (a serial mutation root must not be reordered). It defaults **on**: the `GRAFAST_HOIST`
byte-identity oracle proves hoist-on == hoist-off across the whole suite + conformance, and the
carry is a faithful port of upstream Grafast's bucket-boundary model.

```python
from grafast_py import GrafastConfig, GrafastExecutionContext

class NoHoistContext(GrafastExecutionContext):
    grafast_config = GrafastConfig(hoist=False)  # opt OUT (hoist defaults True)
```

You can run the suite with hoisting forced OFF as the byte-identity oracle (the default already runs it on):
`GRAFAST_HOIST=0 uv run pytest tests` re-runs everything with hoisting forced OFF (the default
already runs it on, so this is the byte-identity baseline; the existing corpus has no hoistable
shape, so it is byte-identical including fetchCounts). `tests/test_hoist.py` constructs a hoistable
shape and proves relocation + fire-once + the mutation guard.

## Incremental delivery: `@defer` / `@stream` / subscriptions (graphql-core 3.3 only)

On the graphql-core **3.3** line the engine implements incremental delivery and subscriptions
through the owned `OutputPlan` + an incremental driver (a driver-owned record graph + a publisher
emitting the 3.3 `pending` / `incremental` / `completed` wire protocol). Drive it via
`grafast_py.entry.experimental_execute_incrementally` (or, when `install()` is active, graphql-core's
own `experimental_execute_incrementally`); subscriptions ride the same path via `grafast_subscribe`.
It is a **3.3-only tier** — gated on `_compat.supports_incremental()`, absent on the 3.2 line, which
has no incremental delivery.

It covers single-fragment `@defer`, nested `@defer`, sync-list `@stream`, **all** async-iterator
`@stream` cases, and subscriptions **byte-identically** to upstream graphql-core (the 49
slow-stream / async-iterator cases pass).

**Honest limitation — 7 documented payload-grouping cases.** For 7 specific `@defer`/`@stream`
cases the response **data is byte-identical** to upstream, but the **payload chunk-grouping /
ordering differs**: items that upstream batches into one incremental payload may arrive as separate
payloads, or a defer-vs-stream payload pair may swap order. **After client reassembly the result is
identical.** The cases are: 4 same-tick stream-item-batching cases (a list of awaitables /
async-resolved items that all resolve in one event-loop tick must batch into one payload, plus 3
error-after-initial-count variants), 2 defer-vs-stream completion-order cases, and 1 nested-deferred
group resolver-start-timing case (see `conformance/conftest.py`, the `_GQL33_P7_PARTIAL` set).

**Root cause** (quantified by event-loop-turn instrumentation): the step engine's per-streamed-item /
per-promoted-defer-group completion costs ~5 event-loop turns vs graphql-core's ~1. Upstream's
single-payload batching is *emergent* from that cheap per-item cost — its producer runs ahead of its
consumer, so same-tick items drain into one payload. The only fix that reproduces it exactly is
reducing the engine's per-item/per-group await count to ~1 turn (in `execute.py` / `completion.py`),
which is high blast radius on the green 3.2 baseline and the 49 passing slow-stream cases; it is
deliberately out of scope this wave. Whether to advertise this 3.3-alpha tier as production-shippable
now, or gate it until graphql-core 3.3 stabilises, is a deployment decision (the engine's 3.2 path is
unaffected).

## Status / caveats

This engine **passes a rigorous internal gate set**: on the graphql-core **3.2** line the
full execution conformance suite (302 on the stock executor / 300 + 2 skipped through this
engine), differential parity vs the reference Node Grafast (`tests/differential/`), an
O(depth) N+1 benchmark, and a concurrent soak; on the graphql-core **3.3** line the full
execution conformance suite runs through the engine (396 passed, including `@defer`/`@stream`
modulo the 7 documented payload-grouping cases above). **"Passes our gates" is not the same as
"validated against your production workload."** Before running with real money: pin
the version, run the differential harness against **your** schema and fixtures,
load-test with **your** pool size and concurrency, and commission a security review.

Remaining caveats: the `@defer`/`@stream` tier is **3.3-only** and carries the **7 documented
payload-grouping cases** (data byte-identical, chunk-grouping differs; identical after client
reassembly — see above) plus a dependency on the **graphql-core 3.3 alpha**; multiple
databases in one process are not supported (one engine per URL — dispose+reconfigure to switch);
and the execution timeout bounds the caller but does not itself cancel in-flight SQL (pair it
with a server-side `statement_timeout`). Query cost/depth limiting is by design **not**
in this engine — do it in your validation layer (see above).

**Deferred in the `grafast_py.pg` data source**: `HAVING` on aggregates is not yet exposed.

**Optimizers**: opportunistic `LATERAL` relation inlining (`inline_relations`, opt-in, default
**OFF**), cross-request **plan caching** (`cache_plans`, opt-in, default **OFF**), runtime
**placeholders** (`placeholders`, opt-in, default **OFF**), and cross-parent step **hoisting**
(`hoist`, default **ON**, like upstream Grafast). With the three opt-in optimizers off — their
default — each resource layer is its own batched `= ANY($1)` round-trip and every operation plans
per-request, inlining each `$variable` value as a SQL literal. Turning inlining on folds a
*safe-to-prove* hasOne / unpaginated-hasMany child into the parent's statement to cut SQL
round-trips; turning placeholders on lets a host express a variable-derived filter / pagination
value as a *value-agnostic* bindparam tagged by its source variable, so the resulting plan is
value-independent; turning caching on reuses that value-independent plan across requests of the same
document (re-binding only the values, no deepcopy); hoisting (on by default) lifts a
child-bucket-constant step to a shallower layer so it runs once instead of once-per-child-bucket.
The three opt-in optimizers ship **dark** (default off); hoisting is **on by default** (proven
byte-identical by its `GRAFAST_HOIST` oracle). All are gated so the result is byte-identical
regardless of which are enabled (see
[LATERAL relation inlining](#lateral-relation-inlining-experimental-opt-in),
[Plan caching + runtime placeholders](#plan-caching--runtime-placeholders-experimental-opt-in), and
[Cross-parent step hoisting](#cross-parent-step-hoisting-experimental-opt-in) above).

The engine provides: the explicit two-tree
`LayerPlan`/`OutputPlan` model (execution de-fused from serialization); resolver-unification
(every field is a step, one execution path); the function-level `grafast_execute()` /
`grafast_subscribe()` seam + the `GrafastExecutionContext` drop-in shim; dual graphql-core 3.2
**and** 3.3 support; `@defer`/`@stream`/subscriptions as the 3.3-only tier (with the 7-case caveat
above); cross-parent hoisting (`hoist`, on by default); and the deepcopy-free plan cache. The
Postgres data source also provides: multi-column (composite) relation keys, the single-shared-request-transaction mode,
connection `GROUP BY` / aggregates, a codec type library (recursive arrays / ranges / enums /
composites), and GraphQL **interfaces / unions** backed by Postgres — column-discriminator
single-table inheritance (a `resolve_type` bridge on the discriminator column) and cross-table
`UNION ALL` (`pgUnionAll`: a keyset-paged Relay connection over N member tables, each tagged with its
concrete type). Both compose with the engine's completion-time abstract dispatch, so a concrete
type's nested relations batch per concrete-type group with no plan-time polymorphism.

## More

- Examples: [`examples/`](examples/)
- Running the test / conformance / differential / benchmark suites: [`AGENTS.md`](AGENTS.md)

## License

MIT — see [`LICENSE`](LICENSE).
