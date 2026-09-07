"""
Pytest wrapper around audit.py's look-back audit — parametrized per
filter so a failure names exactly which filter is leaking future data,
rather than one opaque pass/fail. Runs against the real config/settings.yaml
so it audits the config that's actually shipped, not a synthetic stand-in.
"""

import yaml
import pytest

from audit import FILTER_REGISTRY, audit_filter


def _real_filters_config() -> dict:
    with open("config/settings.yaml") as f:
        config = yaml.safe_load(f)
    return config["filters"]


@pytest.mark.parametrize("filter_name", sorted(FILTER_REGISTRY))
def test_filter_is_point_in_time_safe(filter_name):
    filters_config = _real_filters_config()
    filter_config = filters_config.get(filter_name)
    assert filter_config is not None, (
        f"settings.yaml has no config block for registered filter '{filter_name}' — "
        f"the audit can't test it"
    )

    finding = audit_filter(filter_name, filter_config)
    assert finding.ok, finding.detail


def test_every_registered_filter_has_a_settings_yaml_block_to_audit():
    """A filter registered in FILTER_REGISTRY but missing from
    settings.yaml would silently skip the audit above — this catches that
    directly rather than relying on the per-filter test's own assert."""
    filters_config = _real_filters_config()
    missing = [name for name in FILTER_REGISTRY if name not in filters_config]
    assert missing == []
