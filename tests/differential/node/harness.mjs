// Phase C — reference Node harness.
//
// For each fixture in corpus.mjs: install a per-fixture fetch counter around every
// named batch loader, build a grafast schema via makeGrafastSchema, run
// grafast({schema, source, variableValues}), and record
// { data, errors:[{message,path}], fetchCounts } keyed by fixture name. Writes the
// whole map to out/reference-results.json.
//
// The counter must be installed for the duration of BOTH the schema build (so the
// plan closures capture the wrapped loader) AND the run, then restored — so the
// install/restore brackets the whole fixture run. Counters reset per fixture; a
// loader a fixture never invokes is simply absent from its fetchCounts (count > 0
// filter), matching the Python side's "absent when untriggered" rule.

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { makeGrafastSchema, grafast } from "grafast";
import { FIXTURES, LOADERS } from "./corpus.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));

async function runFixture(fixture) {
  const counts = {};
  const originals = {};
  for (const name of Object.keys(LOADERS)) {
    originals[name] = LOADERS[name];
    const fn = originals[name];
    LOADERS[name] = (...args) => {
      counts[name] = (counts[name] || 0) + 1;
      return fn(...args);
    };
  }
  try {
    const schema = makeGrafastSchema({
      typeDefs: fixture.sdl,
      objects: fixture.plans,
    });
    const result = await grafast({
      schema,
      source: fixture.query,
      variableValues: fixture.variables || {},
    });
    const errors = (result.errors ?? []).map((e) => ({
      message: e.message,
      path: e.path ?? null,
    }));
    const fetchCounts = {};
    for (const [k, v] of Object.entries(counts)) if (v > 0) fetchCounts[k] = v;
    return { data: result.data ?? null, errors, fetchCounts };
  } finally {
    for (const name of Object.keys(originals)) LOADERS[name] = originals[name];
  }
}

async function main() {
  const results = {};
  for (const fixture of FIXTURES) {
    results[fixture.name] = await runFixture(fixture);
  }
  const outDir = join(__dirname, "out");
  mkdirSync(outDir, { recursive: true });
  const outPath = join(outDir, "reference-results.json");
  writeFileSync(outPath, JSON.stringify(results, null, 2), "utf8");
  process.stderr.write(
    `wrote ${Object.keys(results).length} fixtures -> ${outPath}\n`
  );
}

await main();
