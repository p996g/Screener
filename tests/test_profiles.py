"""
Tests for profile resolution (profiles.py).

These use small hand-built config dicts (not the real settings.yaml) so
they can pin down the exact merge semantics: a profile inherits everything
it doesn't mention, and can override a single nested field without
clobbering its siblings.
"""

import pytest

from profiles import list_profile_names, resolve_profile_config


def _config(**overrides):
    base = {
        "universe": {"source": "static_list", "tickers": ["AAPL", "MSFT"]},
        "data": {"provider": "yfinance", "lookback_days": 400},
        "filters": {
            "liquidity": {"enabled": True, "min_value": 10_000_000},
            "trend": {
                "enabled": False,
                "period": 50,
                "direction": "above",
                "require_slope": "none",
            },
        },
        "ranking": {"weights": {"trend": 0.3}, "top_n": 20, "min_score": None},
        "output": {"csv_path": "output/watchlist.csv", "formats": ["console"]},
    }
    base.update(overrides)
    return base


def test_list_profile_names_returns_file_order():
    config = _config(profiles={"b": {}, "a": {}})
    assert list_profile_names(config) == ["b", "a"]


def test_list_profile_names_empty_when_no_profiles_section():
    assert list_profile_names(_config()) == []


def test_unknown_profile_raises_with_helpful_message():
    config = _config(profiles={"momentum_screen": {}})
    with pytest.raises(ValueError, match="momentum_screen"):
        resolve_profile_config(config, "nonexistent")


def test_profile_with_no_overrides_inherits_everything():
    config = _config(profiles={"vanilla": {}})
    resolved = resolve_profile_config(config, "vanilla")
    assert resolved["universe"] == config["universe"]
    assert resolved["filters"] == config["filters"]
    assert resolved["ranking"] == config["ranking"]
    assert resolved["output"] == config["output"]


def test_profile_overrides_a_single_nested_field_and_keeps_siblings():
    """The core inheritance behavior: overriding `trend.period` alone must
    not disturb `trend.direction` or `trend.require_slope`, and must not
    touch the unrelated `liquidity` filter at all."""
    config = _config(
        profiles={
            "pullback": {
                "filters": {"trend": {"enabled": True, "period": 200}},
            }
        }
    )
    resolved = resolve_profile_config(config, "pullback")

    assert resolved["filters"]["trend"] == {
        "enabled": True,
        "period": 200,
        "direction": "above",  # inherited, untouched
        "require_slope": "none",  # inherited, untouched
    }
    assert resolved["filters"]["liquidity"] == config["filters"]["liquidity"]


def test_profile_can_add_a_filter_not_mentioned_in_defaults():
    config = _config(
        profiles={"custom": {"filters": {"momentum": {"enabled": True, "threshold": 0.05}}}}
    )
    resolved = resolve_profile_config(config, "custom")
    assert resolved["filters"]["momentum"] == {"enabled": True, "threshold": 0.05}
    # untouched defaults still present
    assert resolved["filters"]["trend"] == config["filters"]["trend"]


def test_profile_ranking_weight_override_merges_key_by_key():
    config = _config(
        profiles={
            "momentum_heavy": {
                "ranking": {"weights": {"momentum": 0.9}, "top_n": 5},
            }
        }
    )
    resolved = resolve_profile_config(config, "momentum_heavy")
    # momentum added, trend's inherited weight (0.3) still present
    assert resolved["ranking"]["weights"] == {"momentum": 0.9, "trend": 0.3}
    assert resolved["ranking"]["top_n"] == 5
    assert resolved["ranking"]["min_score"] is None  # inherited, untouched


def test_profile_can_override_output_csv_path_independently():
    config = _config(
        profiles={
            "screen_a": {"output": {"csv_path": "output/screen_a.csv"}},
            "screen_b": {"output": {"csv_path": "output/screen_b.csv"}},
        }
    )
    a = resolve_profile_config(config, "screen_a")
    b = resolve_profile_config(config, "screen_b")
    assert a["output"]["csv_path"] == "output/screen_a.csv"
    assert b["output"]["csv_path"] == "output/screen_b.csv"
    # both still inherit the shared formats list
    assert a["output"]["formats"] == ["console"]
    assert b["output"]["formats"] == ["console"]


def test_profile_can_disable_a_filter_that_defaults_to_enabled():
    config = _config(
        profiles={"no_liquidity_gate": {"filters": {"liquidity": {"enabled": False}}}}
    )
    resolved = resolve_profile_config(config, "no_liquidity_gate")
    assert resolved["filters"]["liquidity"]["enabled"] is False
    # min_value inherited even though only `enabled` was overridden
    assert resolved["filters"]["liquidity"]["min_value"] == 10_000_000


def test_profile_can_override_universe():
    config = _config(
        profiles={
            "small_caps": {"universe": {"tickers": ["ZZZZ"]}},
        }
    )
    resolved = resolve_profile_config(config, "small_caps")
    assert resolved["universe"]["tickers"] == ["ZZZZ"]
    assert resolved["universe"]["source"] == "static_list"  # inherited
