# Differential parity vs. reference Grafast

This suite proves that **grafast-py behaves like the canonical TypeScript Grafast**
(`grafast` on npm) — not just that it's GraphQL-spec-correct (that's the
`conformance/` suite), but that for the same query it produces the same data, the
same errors, **and the same batching profile** (how many times each batch loader
fires).

## How it works

It's a *twin-corpus* differential. The same set of fixtures is encoded twice:

| | grafast-py (Python) | reference (Node) |
|---|---|---|
| corpus | [`corpus.py`](corpus.py) | [`node/corpus.mjs`](node/corpus.mjs) |
| harness | [`harness.py`](harness.py) | [`node/harness.mjs`](node/harness.mjs) |

Each fixture is `{name, sdl, plans, query, variables}` — identical SDL text, seed
data, query and variables on both sides, with **matched plan resolvers**. Batch
loaders are referenced by stable names (`loadAuthors`, `loadPostsByAuthor`, …).

For every fixture, each harness:
1. wraps every named loader with a **fetch counter** (so we record how many times
   each batch actually fired),
2. builds a schema (`make_grafast_schema` / `makeGrafastSchema`) and runs the query,
3. writes `{data, errors:[{message, path}], fetchCounts}` keyed by fixture name —
   Python → `out/python-results.json`, Node → `node/out/reference-results.json`.

[`test_differential.py`](test_differential.py) regenerates **both** files fresh
(so a stale file can't mask a regression) and asserts, per fixture:

- **`data`** — order-sensitive structural equality (dict key order + list order).
- **`fetchCounts`** — exact equality. *This is the batching-parity proof:* e.g.
  `loadPostsByAuthor` must fire **once** on both engines for a multi-author query,
  not N times.
- **`errors`** — error count and `path` multiset must match exactly; messages are
  compared only after normalizing cosmetic graphql-js vs graphql-core wording (a
  difference that survives normalization is a hard failure, never silently passed).
- **anti-drift** — the fixture-name sets in the two result files must be identical
  (≥ 20 fixtures), so the two corpora can't silently diverge in coverage.

## Running it

The Node side needs a one-time install:

```bash
cd tests/differential/node && npm install      # pulls `grafast`
```

Then from the repo root:

```bash
uv run pytest tests/differential                # regenerates both sides + diffs
```

`test_differential.py` **skips automatically** when Node or `node/node_modules`
aren't present, so `pytest` stays green on a clone without the Node toolchain.

## The one honest asymmetry

The two corpora are **hand-mirrored** — JS plan resolvers can't be auto-translated
to Python. The known difference: Grafast (JS) passes field args as *steps*
(`args.getRaw("id")`), while grafast-py's `FieldArgs.get(name)` returns the
already-coerced *value*, so the Python mirror lifts it with `constant(...)` where a
step is required. Both feed the **same final value** into the **same loader**, so
data and fetch counts stay identical. The anti-drift name-set check is what keeps the
two encodings honest.

## Generated / ignored

`node/node_modules/`, `node/out/`, and `out/` are generated and git-ignored. Only the
corpora, harnesses, `test_differential.py`, this README, and `node/package*.json` are
committed.
