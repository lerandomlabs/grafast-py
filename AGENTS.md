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
                       wire protocol; subscriptions ride this path
  completion.py        wrapping-type completers (leaf/object/list/non-null/abstract) + null-bubbling;
                       abstract dispatch groups values by concrete type and plans each group as a
                       self-contained step subtree (its own RootStep), so pg interfaces/unions ride
                       the normal object-field machinery (no plan-time polymorphism)
  bubble.py            null-bubble sentinel
  schema.py            plan-resolver API: set_field_plan, make_grafast_schema, FieldArgs;
                       resolve_type bridges for pg interfaces/unions (resolve_type_from_discriminator /
                       resolve_type_from_tag, attach_type_resolvers) — host wiring, no pg import
  config.py            GrafastConfig (execution timeout, concurrency, logging, tracing; the
                       optimizer flags inline_relations / cache_plans / placeholders (opt-in,
                       default OFF) + hoist (default ON, like upstream) + error class (query
                       cost/depth limiting is a validation-layer concern, not here)
  constraints.py       request-input constraints (the plan-cache correctness substrate):
                       Value/Equality/Exists constraints + constraints_match replay (upstream
                       constraints.ts), directive_variable_constraints (the @skip/@include/
                       @defer/@stream document scan), and the plan-resolver accessors —
                       ContextGate/ContextToken (info.context: bare read = value-less token,
                       eval/eval_is/eval_has = look-now + constraint) and TrackedVariables
                       (info.variable_values: every read is an eval)
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
                       inline.py (LATERAL relation inlining: InlineSpec + NestedExtractStep +
                         the equivalence-preserving safety predicate; opt-in via
                         GrafastConfig.inline_relations, default OFF, per-table opt_out_inline),
                       placeholders.py (runtime placeholders: pg_placeholder builds a
                         value-agnostic bindparam tagged with a SOURCE identity — "var:<name>" —
                         so a WHERE/pagination value dedups by source, NEVER by runtime value;
                         literals still inline + dedup by value unchanged).
                       Experimental/opt-in: LATERAL relation inlining + cross-request plan caching
                       + runtime placeholders (all default OFF).
                       Deferred: HAVING on aggregates.

src/grafast_py/cache.py  cross-request plan cache (core, sqlalchemy-free): a bounded-LRU
                       process cache keyed by (id(schema), document-text hash, operation name,
                       variable-arg fingerprint, config fingerprint, incremental flag); each key
                       holds a small MRU bucket of plan VARIANTS (MAX_VARIANTS_PER_KEY), one per
                       constraint-distinguished build (upstream establishOperationPlan's
                       candidate list). A HIT = the first candidate whose constraint set
                       validates for THIS request (constraints.py replay + the customizer
                       guards); it re-binds each placeholder to THIS request's variables by
                       source tag via render-injection — it NEVER deepcopies or mutates the
                       shared plan (per-request values are injected at execute time, so the
                       cached plan is safe to share across concurrent requests). Opt-in via
                       GrafastConfig.cache_plans (default OFF); a plan that OBSERVED a request
                       input (inlined variable, directive variable, context eval) caches as a
                       per-value variant guarded by constraints.

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
GRAFAST_INLINE_RELATIONS=1 uv run pytest tests -m pg  # suite-wide inlining oracle: the whole pg
                                                      # suite re-run with LATERAL inlining forced on,
                                                      # asserting byte-identical data everywhere
GRAFAST_CACHE_PLANS=1 uv run pytest tests             # suite-wide caching oracle: the whole suite
                                                      # re-run with cross-request plan caching (+
                                                      # runtime placeholders) forced on — a hit
                                                      # changes only WHETHER planning re-runs, so
                                                      # the result is byte-identical
GRAFAST_PLACEHOLDERS=1 uv run pytest tests            # suite-wide placeholders oracle: the variable
                                                      # provenance + placeholder dedup path forced
                                                      # on WITHOUT caching (A/B'd independently)
GRAFAST_HOIST=0 uv run pytest tests                   # CI hoist OFF-baseline oracle: hoist is ON by
                                                      # default, so this forces it off suite-wide —
                                                      # must be byte-identical (incl. counts), since
                                                      # hoisting changes only WHERE a step runs
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

## Request-dependent planning & the plan cache — the invariant and its machinery

This is the area where we repeatedly missed subtle correctness holes before the general
substrate landed. The invariant: **no per-request input may influence a plan's shape or field
selection without the cache either re-validating that influence on a hit, or refusing to cache.**
Walk ALL the vectors below whenever you touch anything cache-related.

`cache_plans` (opt-in, default off) builds a plan ONCE and reuses it for the same document. The
cache key — `(schema id, document text, operation name, variable-arg fingerprint, config
fingerprint, incremental flag)` — deliberately does **NOT** include request **context** or
variable **values**. A cache HIT is correct because of exactly two channels (every caching bug is
a hole in one of them):

1. **Runtime VALUE channel** — a per-request value (tenant/RLS, a `$variable`) flows in at execute
   time, never baked into the shared plan, and records NO constraint (the hit-rate guarantee).
   Ours: value-less `ctx:` / `var:` placeholders (`pg/placeholders.py`, resolved per request in
   `where_params` / `member_where_params`); plan resolvers reach the ctx channel via the gate —
   `info.context["k"]` is a value-less `ContextToken` (`constraints.py`), `args.source("x")` is
   the var channel. ≈ upstream `__ValueStep` + bucket-store seed + `unaryValue`.
2. **Constraint channel (on-HIT re-validation)** — when planning OBSERVES an input (branches shape
   on it, or bakes its value), the observation is recorded as a re-checkable constraint and the
   plan caches as one VARIANT in its key's bucket; every hit replays the candidate's whole
   constraint set against THIS request (`constraints.py` `constraints_match` + the customizer
   guards in `lookup_cached_plan`). ≈ upstream `__TrackedValueStep.eval*` → `Constraint` →
   `matchesConstraints` over `establishOperationPlan`'s candidate list.

**The vectors, and what covers each** (all regression-tested in
`tests/test_cache_request_constraints.py`):

- **Directive variables** (`@skip`/`@include`/`@defer`/`@stream`/fragment `if:` = `$variable`):
  resolved by graphql-core's `collect_fields` against variable VALUES *before* planning — a read
  inside foreign code that no accessor can intercept. Covered by the document scan
  (`directive_variable_constraints`): one value constraint per directive variable, recorded on
  every miss. `@stream` IS constrained here (unlike upstream, our planner reads its args at plan
  time into `FieldPlan.stream`).
- **Plan-resolver context reads**: `info.context` is a `ContextGate` — bare reads yield tokens
  (channel 1), `eval`/`eval_is`/`eval_has` return the value AND record a constraint (channel 2),
  `bool(token)` raises. There is no raw plan-time context object to leak.
- **Variable values observed raw**: a `FieldArgs` value read of a variable-derived arg (direct
  `status: $s` or NESTED `where: {status: $s}`), `args.raw`, `"x" in args`, and
  `info.variable_values[...]` all record constraints (the eval semantics; see
  `record_field_args_constraints`). An "inlined" variable therefore caches per value — it is
  never stale and never blanket-non-cacheable.
- **Plain-resolver (ResolveStep) fields with variable args**: coerced args are frozen onto
  `FieldPlan.args`; value constraints on the deriving variables make the frozen args provably
  correct on a hit.
- **pg `select_customizer` structure-branching**: `CustomizerConstraint` (re-invokes the pure
  customizer per hit and compares value-agnostic predicate keys) — now one case of the same
  constraint protocol (`matches(facts)`, facts ignored: it reads the pg request context), riding
  the same stored list. Plus the per-surviving-step `customizer_structure_matches` walk.
- **Structurally unre-validatable shapes stay non-cacheable**: a literal-baking customizer and
  any operation owning an abstract (interface/union) field (its subtrees plan lazily at execute,
  outside `plan.steps`).
- **Side-channel bakes** (a closure/ContextVar/global evaluated into a literal) — the engine sees
  only the evaluated constant; this is the one DOCUMENTED CONTRACT left (README "Cache-safety").
  A placeholder `transform=` closing over request state is caught by content-keying (below).

**Constraint discipline.** Capture happens over the PRE-optimization step set
(`collect_customizer_constraints` runs before `finalize_plan`, so a dedup-merged step still
constrains); re-validation is PURE, cheap, and runs on EVERY hit (it executes under the cache
lock); record a constraint ONLY for reads that affect plan shape/selection — every constraint is
a split point, and the placeholder path must stay constraint-free
(`test_placeholder_path_records_zero_constraints` pins this). Prefer outcome constraints
(`eval_is`/`eval_has` → Equality/Exists) over full value constraints where the plan only needed a
comparison. Buckets are bounded (`MAX_VARIANTS_PER_KEY`); steady-state variant churn in the logs
means something records a high-cardinality value constraint.

**Transform identity** (`transform_key` + `sentinel_placeholders`, `pg/placeholders.py` /
`pg/customize.py`): `predicate_key` feeds BOTH within-plan dedup AND the cross-request constraint
re-validation, so any identity it embeds must be cache-STABLE (a customizer is re-invoked per hit
→ a FRESH callable each time; `id()` and source-location both break it) AND
behaviour-distinguishing — key on executable CONTENT (bytecode + consts + defaults + kwdefaults +
closure values; a callable instance by type + `__dict__`; a builtin by qualname). PR #24's review
history shows the ~6 ways this was gotten wrong.

**Drop-in nuance — don't re-derive the wrong conclusion.** Supporting classic graphql-core
resolvers is NOT in tension with the context gate. Classic resolvers read context at RUN time
(post-plan; `ResolveStep.execute` builds its own info with the REAL context) — like upstream's
`GraphQLResolverStep`. Only the PLAN-time info our planner hands to plan resolvers carries the
gate + `TrackedVariables`, and that info exists in all modes (uniform API; with caching off the
recorded constraints are simply never consumed).

**Search terms / where to look.** Ours: `constraints.py` (the substrate), `cache_plans`,
`compute_cache_key`, `MAX_VARIANTS_PER_KEY`, `lookup_cached_plan`, `record_field_args_constraints`,
`directive_variable_constraints`, `collect_customizer_constraints`, `constraints_match`,
`customizer_structure_matches`, `CustomizerConstraint`, `ContextGate`, `ContextToken`,
`pg_placeholder`, `where_params`, `transform_key`, `sentinel_placeholders`, `values_by_source`.
Upstream reference (cloned at `grafast-crystal/grafast/grafast/src`): `steps/__value.ts`,
`steps/__trackedValue.ts`, `constraints.ts`, `establishOperationPlan.ts`,
`graphqlCollectFields.ts`.

## Purposeful distinctions from upstream

Deliberate design divergences from upstream Grafast — do not "fix" these toward upstream without
revisiting the reasoning:

- **Context: withhold by default, public `eval` as an extension.** Upstream's public `context()`
  exposes ONLY the untracked runtime channel (plan authors cannot eval context at all; the eval
  machinery is engine-internal). We adopt the same withholding default (`info.context[...]` =
  value-less token) but expose `eval`/`eval_is`/`eval_has` publicly — same machinery upstream uses
  internally for `@skip`, deliberately offered to plan resolvers (spec R3). Our tokens also fail
  LOUD on `bool()` where upstream's steps are vacuously truthy.
- **Arguments are plain coerced VALUES, not steps.** Upstream hands plan resolvers input STEPS
  (`fieldArgs.getRaw` → `__TrackedValueStep`, tokens all the way down). We hand coerced Python
  values (the friendlier API) and recover provenance from the field AST
  (`variable_provenance`, incl. variables nested in input literals); a value read of a
  variable-derived arg is recorded as the eval it is. Same eval vocabulary as the gate, two
  defaults: bare context reads are always per-request (token), bare arg reads are usually
  document-literals (value). An all-tokens FieldArgs is upstream's shape if ever wanted.
- **Field collection stays graphql-core's; directive variables are constrained by a document
  scan.** Upstream reimplements CollectFields (`graphqlCollectFields.ts`) so directive-arg reads
  route through its tracked step. We keep calling graphql-core's `collect_fields` and instead
  pre-record the same constraints from the AST (`directive_variable_constraints`) — identical
  outcome, no engine fork. Consequence: our `@stream` args are read at PLAN time
  (`FieldPlan.stream`) and therefore constrained, where upstream resolves them at runtime
  (unconstrained).
- **Constraints may re-invoke host code; matching is `==`.** Upstream constraints are pure data
  compared with `===`. Ours are a duck-typed protocol (`matches(facts)`) so the pg
  `CustomizerConstraint` — which re-invokes the (contractually pure, cheap) customizer per hit —
  rides the same list; value matching uses Python `==` (GraphQL coercion pins types per
  operation, and `==` compares coerced dicts/lists sanely where reference identity would always
  miss). Only the three constraint shapes with consumers exist (value/equal/exists) — upstream's
  `evalKeys`/`evalLength`/`evalIsEmpty` surface has no consumer here (spec: out of scope).
- **Cache keyed by document TEXT, candidates in an LRU bucket.** Upstream keys by
  OperationDefinitionNode object identity (works because of its query-string→AST cache) and keeps
  up to 50 candidates per operation on the schema object. We key by printed document text (stable
  across re-parses) in a process `PlanCache`, `MAX_VARIANTS_PER_KEY`=8, and count hits as
  VALIDATED hits. Upstream also caches planning ERRORS per constraint set; we don't (no consumer).
- **"Placeholder" is our name for upstream's unary/value-step channel** (`pg_placeholder`,
  pagination `Placeholder`, `ContextToken`): a source-tagged value-less slot resolved per request
  at the render seam, instead of a `__ValueStep` wired through the DAG. DAG context-edges are
  deliberately not built (spec: out of scope until a consumer exists).
- **`info.root_value` is NOT tracked** (upstream tracks rootValue as its third entity). No
  consumer here; a plan resolver reading it at plan time is the same side-channel class as a
  closure bake (the documented contract) — note the subscription per-event path, where
  root_value is the EVENT payload and plans are cached across events. Wire it as a third
  constraint scope if a consumer appears.

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
