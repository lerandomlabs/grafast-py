"""Differential parity test vs the reference Node Grafast.

Regenerates BOTH result files (so a stale file can never mask a regression), then
diffs them per fixture:

- ``data``: order-sensitive structural equality (selection/key order + list order
  preserved). HARD failure on any divergence.
- ``errors``: the multiset of error ``path``s and the error COUNT must be identical
  (HARD). Messages are compared only after normalizing known graphql-js vs
  graphql-core framework wording; a residual wording delta that survives
  normalization is recorded as KNOWN-ACCEPTABLE (not a failure); anything else fails.
- ``fetchCounts``: exact dict equality (HARD) — the batching profile must match.

Anti-drift: the set of fixture names in the two files must be identical, else the
corpus encodings drifted and the suite fails loudly.

The reference file is produced by ``node node/harness.mjs`` (run via subprocess with
cwd=node/); the Python file by the in-process ``harness`` module. See README.md for
setup (``cd node && npm install``).

Skipped automatically when Node or ``node/node_modules`` are absent, so a clone
without the Node toolchain still has a green ``pytest`` run.
"""

import json
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(__file__)
NODE_DIR = os.path.join(HERE, "node")
REF_RESULTS = os.path.join(NODE_DIR, "out", "reference-results.json")

# ensure `import corpus` / `import harness` resolve from tests/differential/
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# The reference side needs Node + an `npm install` in node/. Skip the whole module
# (rather than error) when either is missing, so `pytest` stays green on a clone that
# hasn't set up the Node toolchain.
pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not os.path.isdir(os.path.join(NODE_DIR, "node_modules")),
    reason="differential parity needs Node + `npm install` in tests/differential/node",
)


def normalize_message(message: str) -> str:
    """Collapse purely-cosmetic framework wording so equal conditions compare equal.

    graphql-js and graphql-core phrase the SAME condition with minor punctuation /
    quoting differences. We lower-case, collapse internal whitespace, strip a trailing
    period, and unify the quote/backtick characters used around coordinates. This is
    intentionally minimal: PATH and COUNT are already asserted exactly; this only
    decides whether a message DIFFERENCE is the known framework-wording kind. A
    difference that survives this is reported, never silently passed.
    """
    text = message.strip().lower()
    text = " ".join(text.split())
    text = text.rstrip(".")
    for ch in ("`", "'", '"'):
        text = text.replace(ch, "")
    return text


def load_json(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def structural_equal(a, b, trail=""):
    """Order-sensitive structural equality; returns (ok, first-divergence-path).

    dict: same keys in the SAME order, recursively equal values. list: same length,
    element-wise recursive equality in order. numbers: ``1`` and ``1.0`` compare
    equal by value (a documented narrow net for int/float JSON token shape); bool and
    None compare by identity-of-value.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        ka, kb = list(a.keys()), list(b.keys())
        if ka != kb:
            return False, f"{trail}: key order/set {ka!r} != {kb!r}"
        for k in ka:
            ok, where = structural_equal(a[k], b[k], f"{trail}.{k}")
            if not ok:
                return False, where
        return True, ""
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False, f"{trail}: list len {len(a)} != {len(b)}"
        for idx, (ea, eb) in enumerate(zip(a, b)):
            ok, where = structural_equal(ea, eb, f"{trail}[{idx}]")
            if not ok:
                return False, where
        return True, ""
    # bool must not be treated as a number (True == 1 in Python)
    if isinstance(a, bool) or isinstance(b, bool):
        return (a is b, f"{trail}: bool {a!r} != {b!r}")
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return (a == b, f"{trail}: number {a!r} != {b!r}")
    return (a == b, f"{trail}: {a!r} != {b!r}")


@pytest.fixture(scope="session")
def results():
    """Regenerate both result files and return (python_results, reference_results)."""
    # reference (Node) — subprocess in reference/; tolerate the GRAPHILE_ENV banner.
    proc = subprocess.run(
        ["node", "harness.mjs"],
        cwd=NODE_DIR,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"reference harness failed (rc={proc.returncode})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    reference = load_json(REF_RESULTS)

    # python (in-process) — import here so the engine is loaded once.
    import harness

    python = harness.generate_python_results()
    return python, reference


def test_fixture_name_sets_match(results):
    """Anti-drift: the two corpus encodings must cover the SAME fixtures."""
    python, reference = results
    assert set(python.keys()) == set(reference.keys()), (
        f"fixture coverage drifted: "
        f"py-only={set(python) - set(reference)} "
        f"ref-only={set(reference) - set(python)}"
    )
    assert len(python) >= 20, f"corpus must have >= 20 fixtures, got {len(python)}"


def fixture_names():
    """The shared fixture-name list (read from the Python corpus, the source of truth)."""
    import corpus

    return [f["name"] for f in corpus.FIXTURES]


@pytest.mark.parametrize("name", fixture_names())
def test_fixture_parity(results, name):
    python, reference = results
    py = python[name]
    ref = reference[name]

    # DATA — order-sensitive structural equality. HARD.
    ok, where = structural_equal(py["data"], ref["data"])
    assert ok, f"[{name}] DATA divergence at {where}\n py={py['data']!r}\n ref={ref['data']!r}"

    # FETCH COUNTS — exact dict equality. HARD.
    assert py["fetchCounts"] == ref["fetchCounts"], (
        f"[{name}] fetchCounts divergence: py={py['fetchCounts']!r} ref={ref['fetchCounts']!r}"
    )

    # ERRORS — count + path multiset HARD; message only after normalization.
    py_errs, ref_errs = py["errors"], ref["errors"]
    assert len(py_errs) == len(ref_errs), (
        f"[{name}] error COUNT divergence: py={len(py_errs)} ref={len(ref_errs)}\n"
        f" py={py_errs!r}\n ref={ref_errs!r}"
    )
    py_paths = sorted(tuple(e["path"]) if e["path"] else () for e in py_errs)
    ref_paths = sorted(tuple(e["path"]) if e["path"] else () for e in ref_errs)
    assert py_paths == ref_paths, (
        f"[{name}] error PATH divergence: py={py_paths!r} ref={ref_paths!r}"
    )

    # MESSAGE parity after normalization — record-not-fail for known framework wording.
    def by_path(errs):
        out = {}
        for e in errs:
            out.setdefault(tuple(e["path"]) if e["path"] else (), []).append(e["message"])
        return out

    py_by, ref_by = by_path(py_errs), by_path(ref_errs)
    for path, py_msgs in py_by.items():
        ref_msgs = ref_by[path]
        py_msgs_sorted = sorted(py_msgs)
        ref_msgs_sorted = sorted(ref_msgs)
        for pm, rm in zip(py_msgs_sorted, ref_msgs_sorted):
            if pm == rm:
                continue
            if normalize_message(pm) == normalize_message(rm):
                # KNOWN-ACCEPTABLE framework wording difference (path + count already
                # matched). Recorded, not failed.
                print(
                    f"[{name}] known-acceptable message wording at path {path}:\n"
                    f"  py:  {pm!r}\n  ref: {rm!r}"
                )
                continue
            pytest.fail(
                f"[{name}] UNEXPLAINED message divergence at path {path} "
                f"(survives normalization):\n  py:  {pm!r}\n  ref: {rm!r}"
            )
