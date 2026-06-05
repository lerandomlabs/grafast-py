"""Plan-caching / placeholder config knobs (Wave 4): the two opt-in toggles + their CI oracle
switches.

The config step added the toggles for cross-request plan caching and per-argument variable
provenance (the placeholder enabling surface): ``GrafastConfig.cache_plans`` and
``GrafastConfig.placeholders`` both default OFF. These tests pin the defaults (so the no-op
invariant is visible) and that an explicit override is honoured. The byte-identical behaviour
of the wider suite under the default-off flags is the real proof there is no behaviour change;
this file documents the surface.

The step-9 CI oracle switches (``GRAFAST_CACHE_PLANS`` / ``GRAFAST_PLACEHOLDERS``) are the
sibling of the Wave 3b ``GRAFAST_INLINE_RELATIONS`` switch: an autouse fixture in
``tests/conftest.py`` flips the whole suite's base config on when the env var is set, so the
existing result-asserting suite becomes the broadest byte-identical oracle. The META tests at
the bottom pin the predicate those fixtures read (driven purely by the env var, with caching
implying placeholders), independent of the ambient base-config the fixture legitimately mutates.
"""

import pytest

from grafast_py.config import DEFAULT_CONFIG, GrafastConfig

from .conftest import cache_plans_enabled, placeholders_enabled


def test_cache_plans_defaults_off():
    """The plan cache ships dark: default config has it False."""
    assert GrafastConfig().cache_plans is False
    assert DEFAULT_CONFIG.cache_plans is False


def test_cache_plans_can_be_enabled():
    """A host opts in by constructing a config with the flag set."""
    assert GrafastConfig(cache_plans=True).cache_plans is True


def test_placeholders_defaults_off():
    """The variable-provenance surface ships dark: default config has it False."""
    assert GrafastConfig().placeholders is False
    assert DEFAULT_CONFIG.placeholders is False


def test_placeholders_can_be_enabled():
    """A host opts in by constructing a config with the flag set."""
    assert GrafastConfig(placeholders=True).placeholders is True


def test_new_toggles_independent_of_other_knobs():
    """The Wave 4 toggles do not disturb the existing default knobs (no-op on defaults)."""
    config = GrafastConfig()
    assert config.cache_plans is False
    assert config.placeholders is False
    assert config.inline_relations is False
    assert config.execution_timeout_s is None
    assert config.max_step_concurrency is None


# ----------------------------------------------- the CI oracle-switch predicates (no DB)


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, False),  # unset
        ("", False),
        ("0", False),
        ("false", False),
        ("False", False),
        ("1", True),  # any truthy value arms the suite-wide cache-on switch
        ("true", True),
        ("True", True),
        ("yes", True),
    ],
)
def test_cache_switch_reads_env(monkeypatch, value, expected):
    """``cache_plans_enabled`` (the cache-on predicate) is driven purely by ``GRAFAST_CACHE_PLANS``.

    This is the gate the autouse ``cache_plans_suite_toggle`` reads: unset/falsey leaves the
    suite on per-request planning (dark ship), any truthy value flips the whole suite to
    cache-on. We clear the placeholders var so this check is isolated to the cache predicate.
    """
    monkeypatch.delenv("GRAFAST_PLACEHOLDERS", raising=False)
    if value is None:
        monkeypatch.delenv("GRAFAST_CACHE_PLANS", raising=False)
    else:
        monkeypatch.setenv("GRAFAST_CACHE_PLANS", value)
    assert cache_plans_enabled() is expected


def test_placeholders_switch_reads_env_independently(monkeypatch):
    """``placeholders_enabled`` is armed by GRAFAST_PLACEHOLDERS WITHOUT caching."""
    monkeypatch.delenv("GRAFAST_CACHE_PLANS", raising=False)
    monkeypatch.setenv("GRAFAST_PLACEHOLDERS", "1")
    assert placeholders_enabled() is True
    assert cache_plans_enabled() is False  # placeholders alone never arms caching


def test_caching_implies_placeholders(monkeypatch):
    """Arming ONLY caching also arms placeholders (a cacheable plan must be value-agnostic).

    The cache-on oracle forces ``cache_plans=True``; caching is only safe across values for a
    value-agnostic placeholder-bearing plan, so ``placeholders_enabled`` is True under the cache
    switch even when its own env var is unset.
    """
    monkeypatch.delenv("GRAFAST_PLACEHOLDERS", raising=False)
    monkeypatch.setenv("GRAFAST_CACHE_PLANS", "1")
    assert cache_plans_enabled() is True
    assert placeholders_enabled() is True


def test_both_switches_off_by_default(monkeypatch):
    """With neither env var set, both oracle predicates are off — the dark-ship default."""
    monkeypatch.delenv("GRAFAST_CACHE_PLANS", raising=False)
    monkeypatch.delenv("GRAFAST_PLACEHOLDERS", raising=False)
    assert cache_plans_enabled() is False
    assert placeholders_enabled() is False
