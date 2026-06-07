"""grafast-py test suite (a package).

This ``__init__`` makes ``tests`` a real package so test modules import as
``tests.test_*`` under EVERY pytest import mode — in particular pytest's ``prepend``
mode (used in CI), where a directory without ``__init__.py`` would otherwise import
each test file as a top-level module and break the ``from .conftest import ...``
relative helpers (e.g. ``test_cache_config`` / ``test_pg_inline_suite_equivalence``).
Mirrors ``conformance/__init__.py``, which exists for the same reason.
"""
