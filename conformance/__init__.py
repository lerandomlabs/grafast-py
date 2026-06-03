"""grafast-py conformance harness.

The graphql-core execution test-suite (our conformance oracle) is fetched on demand
into the git-ignored ``_suite/`` subpackage by ``scripts/fetch_conformance.py`` — it
is never committed. This package's ``conftest.py`` routes that suite through
grafast-py when ``GRAFAST=1``, otherwise it runs on the stock graphql-core executor
(the baseline). This ``__init__`` exists so the fetched files import as the
``conformance._suite`` package under pytest's prepend import mode.
"""
