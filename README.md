# grafast-py

A **plan-then-execute** GraphQL engine for [graphql-core](https://github.com/graphql-python/graphql-core)
and [Ariadne](https://ariadnegraphql.org/) — a drop-in `ExecutionContext` with real
[Grafast](https://grafast.org)-style **automatic batching** (the N+1 fix). It is an
experimental Python re-implementation of the core ideas of Graphile's Grafast.

## The Grafast idea in 3 lines

1. Build a static **plan** — a DAG of **steps** — from the operation, once.
2. Execute each step **once over a whole batch** of inputs (a "bucket"), not once per
   (field, parent) pair.
3. Batching — the DataLoader N+1 problem — becomes **automatic**: a `loadMany` over N
   parents sees all N keys in a single call and fires its loader exactly once.

## Install

```bash
uv add grafast-py        # or: pip install grafast-py
# optional extras:
uv add 'grafast-py[ariadne]'    # Ariadne SchemaBindable integration
uv add 'grafast-py[structlog]'  # structured kv logging
```

Runtime dependencies are just `graphql-core`, `sqlalchemy`, `asyncpg`, and `greenlet`.
The package ships a `py.typed` marker, so type information is available to consumers.

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

Runnable: [`examples/pg_blog.py`](examples/pg_blog.py) (uses the scratch DB
`grafast_py_test`, schema `grafast_demo`):

```bash
uv run python examples/pg_blog.py
```

`grafast_py.pg` is the canonical import path for the data source; the most-used
symbols are also re-exported at the top level for convenience.

## Hardening config (opt-in)

All production controls are opt-in via a `GrafastConfig` on the context class; the
defaults reproduce the un-hardened behaviour exactly.

```python
from grafast_py import GrafastConfig, GrafastExecutionContext

class HardenedContext(GrafastExecutionContext):
    grafast_config = GrafastConfig(
        execution_timeout_s=5.0,    # async-path wall-clock budget
        max_depth=12,               # reject deeply nested queries (plan time)
        max_cost=10_000,            # basic static cost guard (plan time)
        max_step_concurrency=32,    # cap in-flight awaitables / DB round-trips
        # tracing hooks (no-ops by default); each may return a context-manager span:
        on_operation=my_op_span,    # (context, operation)
        on_plan=my_plan_span,       # (context, operation)
        on_step_batch=my_batch_span # (step, count) — the batch boundary
    )
```

Configure the Postgres pool (the URL is fixed to the scratch DB; the knobs tune the
pool/connection only):

```python
from grafast_py.pg import configure_engine
configure_engine(pool_size=32, max_overflow=8, pool_timeout=30)
```

**Concurrency / pool relationship:** the ceiling on concurrent in-flight SQL is
`pool_size + max_overflow`. If application concurrency exceeds that, the excess
queues on checkout and raises the latency tail. Size the pool to your target
concurrency, cap concurrency with `max_step_concurrency`, or both.

## Status / caveats

This engine **passes a rigorous internal gate set** — see [`SUMMARY.md`](SUMMARY.md)
for the evidence (conformance, differential parity vs the reference Node Grafast,
the O(depth) bench table, and the soak numbers) and an **honest** statement of what
is *not* yet battle-tested. "Passes our rigorous gates" is not the same as
"validated against your production workload." Before running with real money, pin
the version, run the differential harness against **your** schema and fixtures, and
load-test with **your** pool/concurrency. A security review is recommended.

## More

- Examples: [`examples/`](examples/)
- Production-readiness assessment: [`SUMMARY.md`](SUMMARY.md)

## License

MIT — see [`LICENSE`](LICENSE).
