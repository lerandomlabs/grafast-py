#!/usr/bin/env python3
"""Fetch the graphql-core execution conformance suite on demand.

grafast-py uses graphql-core's own GraphQL execution test-suite as a conformance
oracle: run it stock (the baseline) to prove the suite is intact, and with
``GRAFAST=1`` (see the root ``conftest.py``) to prove our engine reproduces
graphql-core's spec behaviour. Those tests are NOT shipped in the graphql-core pip
package, so rather than vendoring them we download them from the matching git tag.

This writes a ``conformance/`` package at the repo root (git-ignored — it is a
generated artifact) containing only the execution-exercising subset, with the
upstream ``tests`` package name rewritten to ``conformance`` (the sole edit;
otherwise byte-identical to upstream).

Usage:
    uv run python scripts/fetch_conformance.py          # match the installed graphql-core
    uv run python scripts/fetch_conformance.py 3.2.8    # a specific version

Then (conformance/conftest.py drives the routing):
    uv run pytest conformance              # baseline: stock graphql-core executor
    GRAFAST=1 uv run pytest conformance    # routed through grafast-py
"""

import io
import shutil
import sys
import tarfile
import urllib.request
from importlib.metadata import version as package_version
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# The fetched suite is a git-ignored subpackage; the committed conformance/ package
# (its __init__.py + conftest.py) drives it. _suite/ keeps the upstream conftest from
# colliding with our committed conformance/conftest.py.
DEST = REPO_ROOT / "conformance" / "_suite"

# The execution-exercising subset to keep; everything else upstream (language/,
# type/, validation/, utilities/, ...) does not route through an executor.
KEEP = {
    "execution",
    "utils",
    "fixtures",
    "__init__.py",
    "conftest.py",
    "star_wars_data.py",
    "star_wars_schema.py",
    "test_star_wars_query.py",
    "test_star_wars_introspection.py",
}


def resolve_version(argv: list[str]) -> str:
    """The graphql-core version to fetch: an explicit arg, else the installed one."""
    if len(argv) > 1:
        return argv[1].lstrip("v")
    return package_version("graphql-core")


def download_tests(version: str) -> bytes:
    """Download the graphql-core source tarball for ``version`` from its git tag."""
    url = (
        "https://github.com/graphql-python/graphql-core/archive/refs/tags/"
        f"v{version}.tar.gz"
    )
    print(f"fetching graphql-core v{version} tests from {url}")
    with urllib.request.urlopen(url) as response:  # pinned github release tag
        return response.read()


def extract_subset(tarball: bytes, version: str) -> None:
    """Extract the kept subset of ``tests/`` from the tarball into ``conformance/``."""
    prefix = f"graphql-core-{version}/tests/"
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(tarball), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.name.startswith(prefix):
                continue
            rel = member.name[len(prefix):]
            if not rel:
                continue
            if rel.split("/", 1)[0] not in KEEP:
                continue
            # Keep the utils/ helper MODULES (dedent, gen_fuzz_strings — imported by
            # the execution tests) but drop utils/ own test files: they exercise
            # graphql-core's string helpers, not our executor, and would inflate the
            # conformance count with tests irrelevant to the oracle.
            parts = rel.split("/")
            if parts[0] == "utils" and parts[-1].startswith("test_"):
                continue
            target = DEST / rel
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is not None:
                target.write_bytes(source.read())


def rewrite_package_name() -> int:
    """Rename the upstream ``tests`` package to ``conformance`` (the one edit).

    The vendored files import each other as the ``tests`` package (e.g.
    ``from tests.star_wars_data import ...``); since the directory is
    ``conformance/`` those references are rewritten. Returns the number of files
    changed (1 on graphql-core 3.2.8).
    """
    edited = 0
    for module in DEST.rglob("*.py"):
        text = module.read_text(encoding="utf-8")
        rewritten = (
            text.replace("from tests.", "from conformance._suite.")
            .replace("from tests import", "from conformance._suite import")
            .replace("import tests.", "import conformance._suite.")
        )
        if rewritten != text:
            module.write_text(rewritten, encoding="utf-8")
            edited += 1
    return edited


def write_notice(version: str) -> None:
    """Drop a NOTICE recording provenance + the single edit (MIT attribution)."""
    (DEST / "NOTICE.md").write_text(
        "# Fetched conformance suite (generated — do not edit, do not commit)\n\n"
        "Generated by `scripts/fetch_conformance.py`; this directory is git-ignored.\n\n"
        f"Contents: the execution test-suite of [graphql-core]"
        f"(https://github.com/graphql-python/graphql-core) **v{version}** (MIT "
        "licensed), unmodified except that the upstream `tests` package name is "
        "rewritten to `conformance`.\n\nRegenerate:\n\n"
        "    uv run python scripts/fetch_conformance.py\n",
        encoding="utf-8",
    )


def main(argv: list[str]) -> int:
    version = resolve_version(argv)
    tarball = download_tests(version)
    extract_subset(tarball, version)
    edited = rewrite_package_name()
    write_notice(version)
    print(
        f"wrote {DEST.relative_to(REPO_ROOT)}/ (graphql-core v{version}); "
        f"renamed tests->conformance in {edited} file(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
