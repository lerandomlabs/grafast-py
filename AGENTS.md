# Working in grafast-py

Guidance for agents (and humans) working in this repository. grafast-py is a
**plan-then-execute GraphQL execution engine for graphql-core** — a drop-in
`ExecutionContext` that brings Grafast-style **automatic batching** (the N+1 fix) to
any graphql-core server (Ariadne, FastAPI, Strawberry-over-core, …). Plan resolvers
return *steps*; the planner builds a step DAG; each step runs **once over a bucket of
parents**, so `loadMany` / `pg_select` collapse N+1 into a single batched fetch.

## Layout

```
src/grafast_py/        the engine (the only thing the published wheel ships)
  entry.py             the function seam + shared pipeline: grafast_execute / grafast_subscribe /
                       experimental_execute_incrementally, all over run_planned_operation on a
                       _PlanRunContext (the engine runs WITHOUT subclassing graphql-core's
                       ExecutionContext); returns a plain graphql.ExecutionResult on both versions
  context.py           GrafastExecutionContext — a thin drop-in ExecutionContext shim whose
                       execute_operation delegates into entry.run_planned_operation (same core)
  _compat.py           the graphql-core 3.2/3.3 seam: feature-probed get_field_def, collect_fields,
                       error-sort, incremental classes, IS_32 / supports_incremental() — one engine
                       core serves both lines
  plan.py              planner: selection set → OutputPlan (serialization) de-fused from a
                       self-contained LayerPlan (batched execution); plan resolvers → step DAG;
                       a plain-resolver field is wrapped as a ResolveStep (one execution path)
  steps.py             ResolveStep: the resolver-adapter step that runs a plain field resolver on
                       the unified step path (resolver-unification — every field is a step)
  step_model.py        Step base class + per-bucket step execution
  core_steps.py        steps: constant, access/get, lambda_step, list/object_step, each,
                       load_one, load_many, RootStep
  dag.py               step-DAG ordering + cross-step deduplication
  execute.py           bucket executor (batched layer-by-layer; serial mutation path)
  incremental.py       the @defer/@stream incremental-delivery driver (3.3 only): a driver-owned
                       record graph + a publisher emitting the 3.3 pending/incremental/completed
                       wire protocol; subscriptions ride this path (P7)
  completion.py        wrapping-type completers (leaf/object/list/non-null/abstract) + null-bubbling;
                       abstract dispatch groups values by concrete type and plans each group as a
                       self-contained step subtree (its own RootStep), so pg interfaces/unions ride
                       the normal object-field machinery (no plan-time polymorphism)
  bubble.py            null-bubble sentinel
  schema.py            plan-resolver API: set_field_plan, make_grafast_schema, FieldArgs;
                       resolve_type bridges for pg interfaces/unions (resolve_type_from_discriminator /
                       resolve_type_from_tag, attach_type_resolvers) — host wiring, no pg import
  config.py            GrafastConfig (execution timeout, concurrency, logging, tracing; the opt-in
                       ship-dark optimizer flags inline_relations / cache_plans / placeholders /
                       hoist, all default OFF) + error class (query cost/depth limiting is a
                       validation-layer concern, not here)
  pg/                  Postgres data source — the optional `[pg]` extra (SQLAlchemy/asyncpg):
                       resource.py (PgResource: attributes/codecs/relations, select_customizer,
                         single- OR composite-column match keys; decode_row/decode_value),
                       steps.py (pg_select / pg_select_single / PgSelectAllStep, batched = ANY($1)
                         / composite tuple-IN),
                       union.py (pgUnionAll: keyset Relay connection over N member tables via
                         UNION ALL — NULL-padded shared projection + __typename tag, root + per-parent
                         modes; the cross-table polymorphism shape),
                       connection.py (keyset Relay connection: forward+reverse, separate totalCount,
                         connection aggregates sum/avg/min/max/count(distinct) + GROUP BY),
                       cursor.py (keyset/seek cursors + the NULL-aware keyset WHERE comparator),
                       ordering.py (OrderTerm → structured ORDER BY, direction/nulls/multi-column),
                       pagination.py (per-parent first/offset window slice),
                       customize.py (host WHERE: select_customizer + per-plan .where()/.apply()/.where_tree()),
                       conditions.py (filter Condition AST: And/Or/Not + Compare → Core predicate),
                       codecs.py (codec type library: scalars + recursive arrays/ranges/enums/composites),
                       mutations.py (pg_insert/update/delete_single on the serial seam),
                       engine.py (async engine, configure_engine, count_sql),
                       executor.py (request-scoped PgExecutor + pg_request_context[_async], pgSettings/RLS,
                         opt-in shared_txn REPEATABLE READ mode),
                       from_sqlalchemy.py (derive PgResource descriptors from ORM models),
                       inline.py (Wave 3b LATERAL relation inlining: InlineSpec + NestedExtractStep +
                         the equivalence-preserving safety predicate; opt-in via
                         GrafastConfig.inline_relations, default OFF, per-table opt_out_inline),
                       placeholders.py (Wave 4 runtime placeholders: pg_placeholder builds a
                         value-agnostic bindparam tagged with a SOURCE identity — "var:<name>" —
                         so a WHERE/pagination value dedups by source, NEVER by runtime value;
                         literals still inline + dedup by value unchanged).
                       Experimental/opt-in: LATERAL relation inlining + cross-request plan caching
                       + runtime placeholders (all default OFF; ship dark).
                       Deferred: HAVING on aggregates.

src/grafast_py/cache.py  Wave 4 cross-request plan cache (core, sqlalchemy-free): a bounded-LRU
                       process cache of finalized plans keyed by (id(schema), document-text hash,
                       operation name, variable-arg fingerprint); a HIT re-binds each placeholder
                       to THIS request's variables by source tag via render-injection — it NEVER
                       deepcopies or mutates the shared plan (P5: per-request values are injected at
                       execute time, so the cached plan is safe to share across concurrent requests).
                       Opt-in via GrafastConfig.cache_plans (default OFF); only a value-agnostic
                       (placeholder-bearing / all-literal) plan is cacheable across values, so a plan
                       that inlined a variable literal is never reused.

tests/                 our own pytest suite (fast, pure-Python; run in CI)
tests/differential/    parity vs reference Node Grafast (on-demand; needs Node) — see its README
benchmarks/            N+1 benchmark + concurrent soak (on-demand; need Postgres)
conformance/           harness for graphql-core's execution suite (fetched on demand)
examples/              runnable demos (plan_blog: in-memory; pg_blog: Postgres) + demo fixtures
                       (models.py: SQLAlchemy ORM mapping of the grafast_demo tables)
scripts/               dev tooling (fetch_conformance.py)
```

## Running things (use `uv`)

```bash
uv run pytest tests                                   # our suite (fast)
GRAFAST_INLINE_RELATIONS=1 uv run pytest tests -m pg  # CI inline-on oracle: whole pg suite
                                                      # re-run with LATERAL inlining forced on,
                                                      # proving byte-identical data everywhere
GRAFAST_CACHE_PLANS=1 uv run pytest tests             # CI cache-on oracle: whole suite re-run
                                                      # with cross-request plan caching (+ runtime
                                                      # placeholders) forced on — a hit changes only
                                                      # WHETHER planning re-runs, so byte-identical
GRAFAST_PLACEHOLDERS=1 uv run pytest tests            # CI placeholders-on oracle: the variable
                                                      # provenance + placeholder dedup path forced
                                                      # on WITHOUT caching (A/B'd independently)
GRAFAST_HOIST=1 uv run pytest tests                   # CI hoist-on oracle: cross-parent hoisting
                                                      # forced on suite-wide — changes only WHERE a
                                                      # step runs, so byte-identical (incl. counts)
uv run python examples/plan_blog.py                   # see batching with no DB

# conformance (graphql-core's own execution suite as an oracle)
uv run python scripts/fetch_conformance.py            # fetch into conformance/_suite/ (git-ignored)
uv run pytest conformance                             # baseline: stock graphql-core executor (302)
GRAFAST=1 uv run pytest conformance                   # routed through grafast-py (300, 2 skipped)

# graphql-core 3.3 leg (the second CI leg; the function-seam runs on BOTH 3.2 and 3.3)
bash scripts/venv33.sh                                # build .venv33 (graphql-core 3.3 alpha sibling env)
.venv33/bin/python scripts/fetch_conformance.py 3.3.0a12          # fetch the matching 3.3 suite
.venv33/bin/python -m pytest tests/test_function_seam.py tests/test_compat.py -p anyio -q
GRAFAST=1 .venv33/bin/python -m pytest conformance/_suite/execution -p anyio -q   # 396 passed
                                                      # (@defer/@stream: 7 documented payload-grouping
                                                      # cases skipped via conformance/conftest.py)

# differential parity vs reference Grafast (needs Node)
cd tests/differential/node && npm install             # once
uv run pytest tests/differential

# performance (need Postgres — see DB note)
uv run python benchmarks/bench_nplus1.py
uv run python benchmarks/soak.py
```

## Invariants — do not break these

- **Core depends on graphql-core only.** SQLAlchemy/asyncpg are the `[pg]` extra and
  are imported lazily via `__getattr__` in `__init__.py`. Never add a hard dependency
  to the core, and never eagerly import `grafast_py.pg` from the top-level package.
- **Genuinely plan-then-execute.** Build a step DAG and run steps over batches. Do
  **not** delegate to graphql-core's control flow (`execute_fields` / `execute_field`
  / `complete_value`). Reusing graphql-core *leaf* helpers (collect_fields, argument/
  variable coercion, scalar serialize, `GraphQLError`, `located_error`) is fine. The plan
  is a **two-tree** model — an `OutputPlan` (serialization) de-fused from a self-contained
  `LayerPlan` (execution); every field is a **step** (a plan step or a `ResolveStep`
  resolver-adapter), so there is ONE execution path.
- **One core, two entry shapes.** The function seam (`entry.run_planned_operation`, exposed as
  `grafast_execute` / `grafast_subscribe`) and the `GrafastExecutionContext` shim share one
  pipeline — the shim just delegates. Don't add engine logic to `context.py`; it stays a thin
  delegator.
- **Dual graphql-core (3.2 AND 3.3).** The engine runs on both lines; version differences are
  confined to `grafast_py._compat` (probe via `_compat.IS_32` / `_compat.supports_incremental()`),
  never sprinkled through the engine. The pin is `graphql-core>=3.2.8` (no upper cap). `@defer` /
  `@stream` / subscriptions are a **3.3-only tier** (gated on `supports_incremental()`); they carry
  7 documented payload-grouping cases (data byte-identical, chunk-grouping differs — see
  `conformance/conftest.py` `_GQL33_P7_PARTIAL`).
- **Conformance stays green.** On 3.2: `pytest conformance` = 302; `GRAFAST=1 pytest conformance`
  = 300 + 2 skipped. On 3.3 (`.venv33`): the full execution suite = 396 passed through the engine
  (the 7 defer/stream payload-grouping cases skipped via `conformance/conftest.py`). The suite under
  `conformance/_suite/` is fetched/generated — never edit it; if behaviour changes, fix the engine,
  not the oracle.
- **Batching is the point.** A relation/loader must fire its batch callback once per
  layer, not once per parent. `tests/differential` asserts our `fetchCounts` match
  reference Grafast exactly; `benchmarks/bench_nplus1.py` asserts SQL is O(depth).
- **The pg hot path runs through the request executor, not `get_engine`.** Production
  injects a host executor per request — `SQLAlchemyExecutor(host_engine)` (or a
  `RawExecutor` over a host pool) bound via `pg_request_context` around `await
  graphql(...)`. `get_engine`/`configure_engine` are a demo/test convenience (one
  process-global engine); the steps must read `current_pg_request().executor`, never call
  `get_engine` on the hot path. Per-request `pgSettings`/RLS: pass `settings={...}` to
  `pg_request_context`; the executor applies them with transaction-local `set_config` in
  the query's own transaction (auto-clearing at commit). Enforcement needs a DB role that
  does not bypass RLS — a superuser/`BYPASSRLS` role is never subject to a policy.

## Database (DB-backed tests + benchmarks)

DB work targets **only** the local Postgres database `grafast_py_test`, schema
`grafast_demo` (`postgresql+asyncpg:///grafast_py_test`). DDL/data is created and
dropped only within `grafast_demo`. **Never connect to any other database on the
server** — this checkout shares a Postgres instance with unrelated production
databases. DB-backed tests carry the `pg` marker (`pytest -m 'not pg'` to skip them).

## Code style

Specific exception classes (no bare `except Exception`); lower-case sentence-style
structured logging with short kv fields; verbs for side-effecting functions
(`get_…` implies no side effect, `register_…` implies one); global imports unless a
cycle forces otherwise; don't add tiny single-use helpers or one-off module constants.
Keep comments about the code/domain (a concise non-obvious *why*), not the diff.

## Tests

New engine tests go in `tests/`. Differential and benchmark suites are on-demand and
need extra toolchains (Node / Postgres); they skip or are excluded from the default
`pytest tests` run accordingly. Don't commit generated artifacts (`*/out/`, the
benchmark `*.md` reports, fetched `conformance/_suite/`, `node_modules/`).
