"""Inlining config knobs: the two opt-in toggles only.

These toggles gate opportunistic LATERAL relation inlining:
``GrafastConfig.inline_relations`` and the per-resource
``PgResource(opt_out_inline=...)`` escape hatch both default OFF. These tests pin
the defaults (so the no-op invariant is visible) and that an explicit override is
honoured. The byte-identical behaviour of the wider suite under the default-off
flag is the real proof there is no behaviour change; this file just documents the
surface.
"""

from grafast_py.config import DEFAULT_CONFIG, GrafastConfig
from grafast_py.pg.resource import PgResource, PgRegistry


def test_inline_relations_defaults_off():
    """The global inlining toggle is off by default: default config has it False."""
    assert GrafastConfig().inline_relations is False
    assert DEFAULT_CONFIG.inline_relations is False


def test_inline_relations_can_be_enabled():
    """A host opts in by constructing a config with the flag set."""
    assert GrafastConfig(inline_relations=True).inline_relations is True


def test_resource_opt_out_inline_defaults_off():
    """A resource is foldable by default; the per-table escape hatch is opt-in."""
    resource = PgResource(
        "author", "grafast_demo", "author", ["id", "name"], registry=PgRegistry()
    )
    assert resource.opt_out_inline is False


def test_resource_opt_out_inline_can_be_set():
    """A host disables inlining for one suspect table via the constructor kwarg."""
    resource = PgResource(
        "label",
        "grafast_demo",
        "label",
        ["id", "code"],
        registry=PgRegistry(),
        opt_out_inline=True,
    )
    assert resource.opt_out_inline is True
