"""Phase C — grafast-py harness (Python side of the differential corpus).

For each fixture in ``corpus.py``: install a per-fixture fetch counter around
every named batch loader (by mutating ``LOADERS`` so the plan closures pick up the
wrapped fn at plan time, exactly as the JS harness mutates its ``LOADERS``), build a
schema via ``make_grafast_schema(sdl, plans)``, run the operation through
``GrafastExecutionContext``, and record ``{data, errors:[{message,path}],
fetchCounts}`` keyed by fixture name. Writes the whole map to
``tests/differential/out/python-results.json``.

Counters reset per fixture; a loader a fixture never invokes is absent from its
fetchCounts (count > 0 filter), matching the JS side. Run standalone
(``python tests/differential/harness.py``) or via :func:`generate_python_results` from the
differ's session fixture.
"""

import json
import os
from typing import Any, Dict, List

from graphql import ExecutionResult, execute, parse

from grafast_py import GrafastExecutionContext, make_grafast_schema

from corpus import FIXTURES, LOADERS

OUT_DIR = os.path.join(os.path.dirname(__file__), "out")
OUT_PATH = os.path.join(OUT_DIR, "python-results.json")


def run_fixture(fixture: Dict[str, Any]) -> Dict[str, Any]:
    """Run one fixture through the engine with a fresh per-loader fetch counter."""
    counts: Dict[str, int] = {}
    originals = dict(LOADERS)

    def make_wrapper(name: str, fn):
        def wrapped(keys):
            counts[name] = counts.get(name, 0) + 1
            return fn(keys)

        return wrapped

    for name, fn in originals.items():
        LOADERS[name] = make_wrapper(name, fn)
    try:
        schema = make_grafast_schema(fixture["sdl"], fixture["plans"])
        result = execute(
            schema,
            parse(fixture["query"]),
            variable_values=fixture["variables"],
            execution_context_class=GrafastExecutionContext,
        )
    finally:
        LOADERS.update(originals)

    if not isinstance(result, ExecutionResult):
        # an async result would be a coroutine/awaitable; the corpus is fully sync.
        raise TypeError(
            f"fixture {fixture['name']!r} produced a non-sync result {type(result)!r}"
        )

    errors: List[Dict[str, Any]] = []
    for err in result.errors or []:
        errors.append(
            {"message": err.message, "path": list(err.path) if err.path else None}
        )
    fetch_counts = {k: v for k, v in counts.items() if v > 0}
    return {"data": result.data, "errors": errors, "fetchCounts": fetch_counts}


def generate_python_results() -> Dict[str, Any]:
    """Run every fixture and return the result map (also written to OUT_PATH)."""
    results: Dict[str, Any] = {}
    for fixture in FIXTURES:
        results[fixture["name"]] = run_fixture(fixture)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False, sort_keys=False)
    return results


if __name__ == "__main__":
    out = generate_python_results()
    print(f"wrote {len(out)} fixtures -> {OUT_PATH}")
