#!/usr/bin/env bash
# Build the 3.3 sibling virtualenv: graphql-core 3.3 alpha installed alongside the
# editable grafast-py, in a SEPARATE .venv33 (the default .venv keeps graphql-core 3.2).
#
# This is the second CI leg's environment (the function-seam runs on BOTH 3.2 and 3.3).
# The package pin was widened to `graphql-core>=3.2.8` (it no longer caps at <3.3), so the
# verbatim command below resolves; on the old <3.3 cap uv refused the resolution.
#
# Verify after:
#   .venv33/bin/python -c "import graphql; print(graphql.__version__)"   # -> 3.3.0a12
#   .venv33/bin/python -c "import grafast_py"                             # -> (no output)
set -euo pipefail

GQL33_VERSION="${GQL33_VERSION:-3.3.0a12}"

uv venv .venv33 --python 3.12

# Primary path (works with the widened pin): one resolution covering the editable package
# and the pinned graphql-core, reinstalling grafast-py so a stale build is replaced.
if ! uv pip install --python .venv33 -e . "graphql-core==${GQL33_VERSION}" \
    --reinstall-package grafast-py; then
    # Fallback for an older uv that still enforces a cap, or any resolver refusal: install
    # the pinned graphql-core first, then the editable package WITHOUT its dependencies so
    # the 3.3 alpha is not contested. (uv 0.3.x has no `uv build`, so this is the fallback,
    # not a wheel build.)
    echo "primary resolution failed; falling back to --no-deps over pinned graphql-core" >&2
    uv pip install --python .venv33 "graphql-core==${GQL33_VERSION}"
    uv pip install --python .venv33 -e . --no-deps
fi

echo "graphql-core: $(.venv33/bin/python -c 'import graphql; print(graphql.__version__)')"
.venv33/bin/python -c "import grafast_py; print('grafast_py import OK')"
