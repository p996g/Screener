"""
Tests for the abstract Filter base class (filters/base.py).
"""

from datetime import date

import pandas as pd
import pytest

from filters.base import Filter, FilterResult


def _make_ohlcv(n_days: int = 5, start_close: float = 100.0) -> pd.DataFrame:
    dates = pd.date_range(end=date.today(), periods=n_days, freq="D")
    closes = [start_close + i for i in range(n_days)]
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c + 1 for c in closes],
            "Low": [c - 1 for c in closes],
            "Close": closes,
            "Volume": [1_000_000] * n_days,
        },
        index=dates,
    )


class AlwaysPassFilter(Filter):
    name = "always_pass"

    def pass_fail(self, symbol_data: pd.DataFrame) -> FilterResult:
        return FilterResult(True, "always passes")


class ThresholdFilter(Filter):
    """A minimal concrete filter used to prove config values flow through
    to pass_fail without being hard-coded."""

    name = "threshold"

    def pass_fail(self, symbol_data: pd.DataFrame) -> FilterResult:
        threshold = self.config["threshold"]
        last_close = float(symbol_data["Close"].iloc[-1])
        if last_close >= threshold:
            return FilterResult(True, f"{last_close} >= {threshold}")
        return FilterResult(False, f"{last_close} < {threshold}")


def test_filter_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        Filter(config={})


def test_subclass_receives_its_config_section():
    filt = ThresholdFilter(config={"threshold": 50})
    assert filt.config == {"threshold": 50}


def test_pass_fail_returns_filter_result():
    filt = AlwaysPassFilter(config={})
    result = filt.pass_fail(_make_ohlcv())
    assert isinstance(result, FilterResult)
    assert result.passed is True
    assert result.reason


def test_threshold_filter_passes_above_threshold():
    filt = ThresholdFilter(config={"threshold": 50})
    data = _make_ohlcv(n_days=1, start_close=100.0)
    result = filt.pass_fail(data)
    assert result.passed is True
    assert "100" in result.reason


def test_threshold_filter_fails_below_threshold_with_reason():
    filt = ThresholdFilter(config={"threshold": 150})
    data = _make_ohlcv(n_days=1, start_close=100.0)
    result = filt.pass_fail(data)
    assert result.passed is False
    assert "100" in result.reason and "150" in result.reason


def test_changing_config_changes_behavior_without_code_changes():
    """Same filter class, different config -> different outcome. This is
    the property that lets settings.yaml fully control filter behavior."""
    data = _make_ohlcv(n_days=1, start_close=100.0)

    lenient = ThresholdFilter(config={"threshold": 10})
    strict = ThresholdFilter(config={"threshold": 1000})

    assert lenient.pass_fail(data).passed is True
    assert strict.pass_fail(data).passed is False
