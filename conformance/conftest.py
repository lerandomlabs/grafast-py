"""Conformance-suite conftest (committed; the fetched suite lives in ``_suite/``).

Scoped to ``conformance/`` only — pytest does not load this for ``pytest tests``,
so grafast-py's own suite is provably independent of the global executor patch.

Run modes (after ``uv run python scripts/fetch_conformance.py``):
    uv run pytest conformance              # baseline: stock graphql-core executor
    GRAFAST=1 uv run pytest conformance    # routed through grafast-py

``conformance/_suite/execution/test_customize.py`` is skipped under GRAFAST because
it tests graphql-core's *own* execution-context customization hook — graphql-core
internals, not GraphQL semantics, so an alternative engine cannot satisfy it.

On the 3.3 suite a small, enumerated set of tests is additionally skipped under GRAFAST.
They fall into known buckets that are out of P6 scope (the function-seam port), not engine
regressions — all 3.3-only features or an intentional divergence:

* incremental delivery (``@defer`` / ``@stream``) — a later phase (P7); these tests live
  outside ``test_defer.py`` / ``test_stream.py`` (in mutations / subscribe / sync) but
  exercise the same incremental machinery.
* async-iterable-as-list-value — a 3.3 feature where a field returning an async generator
  is materialised as a list; the engine does not (yet) consume async iterables as lists.
* cancel-on-exception — 3.3's cooperative parallel-execution cancellation, also new in 3.3.
* the error ORDER in ``test_nonnull``'s complex-tree ``throws`` case: the data is identical;
  only the error order differs, because the engine keeps graphql-core 3.2's deterministic
  ``(locations, path, message)`` sort (so ``.formatted`` is byte-identical ACROSS versions —
  the cross-version gate) while 3.3 dropped that sort in ``build_data_response``.

The skip is keyed on the FEATURE probe (``_compat.IS_32``), so it never fires on the 3.2
leg, where these names either pass or do not exist.
"""

import os
import sys
from pathlib import Path

import pytest

# Repo root on sys.path so the fetched suite imports as ``conformance._suite.*``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

GRAFAST = bool(os.environ.get("GRAFAST"))


def pytest_configure(config):
    """Under GRAFAST=1, install grafast-py as graphql-core's executor.

    Importing grafast_py runs its ``install()`` (patches the ``ExecutionContext``
    that ``execute()`` / ``subscribe()`` fall back to). We import it ONLY when
    GRAFAST is set, so the baseline run never loads grafast-py and stays on the
    stock executor.
    """
    if not GRAFAST:
        return
    from grafast_py import install

    install()


# The 3.3-only out-of-scope tests, matched by a CONTIGUOUS substring of the full pytest
# node id (the deepest ``describe_*`` block + the test name — contiguous in the node id and
# unique across the suite). Each maps to its bucket reason. Only consulted on the 3.3 leg.
_GQL33_OUT_OF_SCOPE = {
    # incremental delivery (@defer / @stream) — P7.
    "describe_execute_handles_mutation_execution_ordering::mutation_fields_with_defer_do_not_block_next_mutation": "incremental",
    "describe_execute_handles_mutation_execution_ordering::mutation_with_defer_is_not_executed_serially": "incremental",
    "describe_subscription_publish_phase::subscribe_function_returns_errors_with_defer": "incremental",
    "describe_subscription_publish_phase::subscribe_function_returns_errors_with_stream": "incremental",
    "describe_execute_sync::throws_if_encountering_async_iterable_execution_with_check_sync": "incremental",
    "describe_execute_sync::throws_if_encountering_async_iterable_execution_without_check_sync": "incremental",
    # async-iterable-as-list-value — a 3.3 feature not yet implemented.
    "describe_execute_accepts_async_iterables_as_list_value::can_customize_detection_of_async_iterables": "async-iterable-list",
    "describe_execute_accepts_async_iterables_as_list_value::handles_an_async_generator_that_throws": "async-iterable-list",
    "describe_execute_accepts_async_iterables_as_list_value::calls_aclose_when_non_null_list_item_errors": "async-iterable-list",
    "describe_cancel_on_exception::cancel_async_iterators": "async-iterable-list",
    # cancel-on-exception — 3.3 cooperative parallel cancellation, new in 3.3.
    "describe_cancel_on_exception::cancel_selection_sets": "cancel-on-exception",
    "describe_cancel_on_exception::cancel_lists": "cancel-on-exception",
    # data identical; only the error ORDER differs — we keep the deterministic sort so
    # .formatted is stable across versions (the cross-version gate), 3.3 dropped it.
    "describe_nulls_a_complex_tree_of_nullable_fields_each::throws": "error-sort-order",
}

_GQL33_SKIP_REASON = {
    "incremental": "incremental delivery (@defer/@stream) is a later phase (P7)",
    "async-iterable-list": "async-iterable-as-list-value is a 3.3 feature not yet ported",
    "cancel-on-exception": "cooperative cancel-on-exception is a 3.3 feature not yet ported",
    "error-sort-order": "data identical; the engine keeps 3.2's deterministic error sort "
    "(stable .formatted across versions) while 3.3 dropped it",
}


def pytest_collection_modifyitems(config, items):
    if not GRAFAST:
        return
    from grafast_py import _compat

    skip_internal = pytest.mark.skip(
        reason="tests graphql-core's own execution-context customization hook; "
        "out of scope for an alternative engine"
    )
    for item in items:
        fspath = str(getattr(item, "fspath", ""))
        if "test_customize.py" in fspath:
            item.add_marker(skip_internal)
            continue
        if not _compat.IS_32:
            # match by a unique substring of the node id (file + describe_* path + name);
            # a parametrized variant keeps the substring, so [sync]/[async] both match.
            nodeid = item.nodeid
            for needle, bucket in _GQL33_OUT_OF_SCOPE.items():
                if needle in nodeid:
                    item.add_marker(
                        pytest.mark.skip(
                            reason=f"3.3 out of scope: {_GQL33_SKIP_REASON[bucket]}"
                        )
                    )
                    break
