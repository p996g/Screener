"""
Tests for the built-in filter library (filters/library.py).

Each filter gets a small synthetic OHLCV dataset with hand-computable
expected values, so these tests prove the actual arithmetic/logic of each
filter, not just that it runs without error.
"""

from datetime import date

import pandas as pd
import pytest

from filters.library import (
    GapFilter,
    LiquidityFilter,
    MomentumFilter,
    PriceRangeFilter,
    ProximityFilter,
    TrendFilter,
    VolatilityFilter,
    VolumeSurgeFilter,
)


def _bars(closes, highs=None, lows=None, volumes=None, start=date(2024, 1, 1)):
    n = len(closes)
    dates = pd.date_range(start=start, periods=n, freq="D")
    highs = highs if highs is not None else [c + 1 for c in closes]
    lows = lows if lows is not None else [c - 1 for c in closes]
    volumes = volumes if volumes is not None else [1_000_000] * n
    return pd.DataFrame(
        {"Open": closes, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=dates,
    )


# -----------------------------------------------------------------------------
# LiquidityFilter
# -----------------------------------------------------------------------------


def test_liquidity_dollar_volume_passes_at_or_above_floor():
    data = _bars([10, 10, 10], volumes=[40, 40, 40])  # dollar vol = 400/day
    filt = LiquidityFilter(config={"metric": "dollar_volume", "lookback_days": 3, "min_value": 300})
    result = filt.pass_fail(data)
    assert result.passed is True
    assert "400" in result.reason


def test_liquidity_dollar_volume_fails_below_floor():
    data = _bars([10, 10, 10], volumes=[10, 10, 10])  # dollar vol = 100/day
    filt = LiquidityFilter(config={"metric": "dollar_volume", "lookback_days": 3, "min_value": 300})
    result = filt.pass_fail(data)
    assert result.passed is False
    assert "failed LiquidityFilter" in result.reason


def test_liquidity_share_volume_metric():
    data = _bars([10, 10, 10], volumes=[60, 60, 60])
    filt = LiquidityFilter(config={"metric": "share_volume", "lookback_days": 3, "min_value": 50})
    result = filt.pass_fail(data)
    assert result.passed is True
    assert "share volume" in result.reason


def test_liquidity_fails_with_insufficient_history():
    data = _bars([10, 10])  # only 2 bars
    filt = LiquidityFilter(config={"metric": "dollar_volume", "lookback_days": 5, "min_value": 1})
    result = filt.pass_fail(data)
    assert result.passed is False
    assert "only 2 bars available" in result.reason


# -----------------------------------------------------------------------------
# PriceRangeFilter
# -----------------------------------------------------------------------------


def test_price_range_passes_within_bounds():
    data = _bars([10])
    filt = PriceRangeFilter(config={"min_price": 5, "max_price": 50})
    assert filt.pass_fail(data).passed is True


def test_price_range_fails_below_min():
    data = _bars([3])
    filt = PriceRangeFilter(config={"min_price": 5, "max_price": 50})
    result = filt.pass_fail(data)
    assert result.passed is False
    assert "below min" in result.reason


def test_price_range_fails_above_max():
    data = _bars([60])
    filt = PriceRangeFilter(config={"min_price": 5, "max_price": 50})
    result = filt.pass_fail(data)
    assert result.passed is False
    assert "above max" in result.reason


def test_price_range_unbounded_side_accepts_null():
    data = _bars([1000])
    filt = PriceRangeFilter(config={"min_price": 5, "max_price": None})
    assert filt.pass_fail(data).passed is True


# -----------------------------------------------------------------------------
# TrendFilter
# -----------------------------------------------------------------------------


def test_trend_passes_when_price_above_ma_no_slope_required():
    data = _bars([10, 11, 12, 13])  # tail(3) mean(11,12,13) = 12, last = 13
    filt = TrendFilter(config={"period": 3, "direction": "above", "require_slope": "none"})
    result = filt.pass_fail(data)
    assert result.passed is True


def test_trend_fails_when_price_below_ma():
    data = _bars([13, 12, 11, 10])  # tail(3) mean(12,11,10) = 11, last = 10
    filt = TrendFilter(config={"period": 3, "direction": "above", "require_slope": "none"})
    result = filt.pass_fail(data)
    assert result.passed is False
    assert "failed TrendFilter" in result.reason


def test_trend_requires_rising_ma_and_passes_when_it_is():
    data = _bars([10, 11, 12, 13, 14, 15])
    # ma_now = mean(13,14,15) = 14; ma_past = mean(11,12,13) = 12 -> rising
    filt = TrendFilter(
        config={"period": 3, "direction": "above", "require_slope": "rising", "slope_lookback_days": 2}
    )
    result = filt.pass_fail(data)
    assert result.passed is True
    assert "rising" in result.reason


def test_trend_fails_when_slope_requirement_not_met_even_if_price_condition_holds():
    data = _bars([15, 14, 13, 12, 11, 10])
    # ma_now = mean(12,11,10) = 11, last_close = 10 -> price_ok for direction="below"
    # ma_past = mean(14,13,12) = 13 -> ma_now(11) < ma_past(13) -> falling, but rising required
    filt = TrendFilter(
        config={"period": 3, "direction": "below", "require_slope": "rising", "slope_lookback_days": 2}
    )
    result = filt.pass_fail(data)
    assert result.passed is False
    assert "falling" in result.reason and "required rising" in result.reason


def test_trend_fails_with_insufficient_history():
    data = _bars([10, 11, 12])
    filt = TrendFilter(config={"period": 5, "direction": "above", "require_slope": "none"})
    result = filt.pass_fail(data)
    assert result.passed is False
    assert "only 3 bars available" in result.reason


# -----------------------------------------------------------------------------
# MomentumFilter
# -----------------------------------------------------------------------------


def test_momentum_absolute_passes_above_threshold():
    data = _bars([100, 100, 100, 110])  # 3-day return = 10%
    filt = MomentumFilter(config={"lookback_days": 3, "direction": "above", "threshold": 0.05})
    result = filt.pass_fail(data)
    assert result.passed is True
    assert "+10.00%" in result.reason


def test_momentum_absolute_fails_below_threshold():
    data = _bars([100, 100, 100, 102])  # 3-day return = 2%
    filt = MomentumFilter(config={"lookback_days": 3, "direction": "above", "threshold": 0.05})
    result = filt.pass_fail(data)
    assert result.passed is False


def test_momentum_relative_to_benchmark_uses_excess_return():
    symbol_data = _bars([100, 100, 100, 110])  # +10%
    benchmark_data = _bars([100, 100, 100, 103])  # +3%
    filt = MomentumFilter(
        config={
            "lookback_days": 3,
            "direction": "above",
            "threshold": 0.0,
            "benchmark_symbol": "SPY",
        }
    )
    assert filt.required_symbols() == ["SPY"]

    context = {"reference_data": {"SPY": benchmark_data}}
    result = filt.pass_fail(symbol_data, context)
    assert result.passed is True
    assert "vs SPY" in result.reason


def test_momentum_relative_fails_gracefully_when_benchmark_missing():
    symbol_data = _bars([100, 100, 100, 110])
    filt = MomentumFilter(
        config={
            "lookback_days": 3,
            "direction": "above",
            "threshold": 0.0,
            "benchmark_symbol": "SPY",
        }
    )
    result = filt.pass_fail(symbol_data, context={"reference_data": {}})
    assert result.passed is False
    assert "benchmark" in result.reason


def test_momentum_with_no_benchmark_configured_requires_no_extra_symbols():
    filt = MomentumFilter(config={"lookback_days": 3, "direction": "above", "threshold": 0.0})
    assert filt.required_symbols() == []


# -----------------------------------------------------------------------------
# ProximityFilter
# -----------------------------------------------------------------------------


def test_proximity_high_passes_within_band():
    data = _bars([100, 101, 103], highs=[100, 101, 105])  # 3d high = 105, close = 103
    filt = ProximityFilter(config={"reference": "high", "lookback_days": 3, "max_distance_pct": 2.0, "side": "any"})
    result = filt.pass_fail(data)
    assert result.passed is True  # (103-105)/105 = -1.90%


def test_proximity_high_fails_outside_band():
    data = _bars([100, 101, 95], highs=[100, 101, 105])
    filt = ProximityFilter(config={"reference": "high", "lookback_days": 3, "max_distance_pct": 2.0, "side": "any"})
    result = filt.pass_fail(data)
    assert result.passed is False


def test_proximity_side_at_or_above_rejects_price_below_reference():
    data = _bars([100, 101, 104], highs=[100, 101, 105])  # close below high but within band
    filt = ProximityFilter(
        config={"reference": "high", "lookback_days": 3, "max_distance_pct": 2.0, "side": "at_or_above"}
    )
    result = filt.pass_fail(data)
    assert result.passed is False  # within band, but on the wrong side


def test_proximity_side_at_or_above_accepts_breakout_above_reference():
    data = _bars([100, 101, 106], highs=[100, 101, 105])  # close above high, breakout
    filt = ProximityFilter(
        config={"reference": "high", "lookback_days": 3, "max_distance_pct": 2.0, "side": "at_or_above"}
    )
    result = filt.pass_fail(data)
    assert result.passed is True


def test_proximity_sma_reference():
    data = _bars([10, 10, 10])  # sma = 10, close = 10 -> 0% distance
    filt = ProximityFilter(config={"reference": "sma", "sma_period": 3, "max_distance_pct": 1.0, "side": "any"})
    result = filt.pass_fail(data)
    assert result.passed is True
    assert "SMA3" in result.reason


def test_proximity_fails_with_insufficient_history():
    data = _bars([10, 10])
    filt = ProximityFilter(config={"reference": "high", "lookback_days": 5, "max_distance_pct": 1.0, "side": "any"})
    result = filt.pass_fail(data)
    assert result.passed is False
    assert "only 2 bars available" in result.reason


# -----------------------------------------------------------------------------
# VolatilityFilter
# -----------------------------------------------------------------------------


def test_volatility_atr_within_band_passes():
    # Day1: H=10,L=8 (no prev close, TR = 2)
    # Day2: H=11,L=9,prevClose=9 -> TR = max(2, 2, 0) = 2
    # Day3: H=12,L=10,prevClose=10 -> TR = max(2, 2, 0) = 2
    # ATR(2) over last 2 days = mean(2,2) = 2; last close = 11 -> ATR% = 2/11*100 = 18.18
    data = pd.DataFrame(
        {
            "Open": [9, 10, 11],
            "High": [10, 11, 12],
            "Low": [8, 9, 10],
            "Close": [9, 10, 11],
            "Volume": [1_000_000] * 3,
        },
        index=pd.date_range(date(2024, 1, 1), periods=3, freq="D"),
    )
    filt = VolatilityFilter(config={"metric": "atr", "lookback_days": 2, "min_value": None, "max_value": 20})
    result = filt.pass_fail(data)
    assert result.passed is True
    assert "18.18" in result.reason


def test_volatility_atr_above_band_fails():
    data = pd.DataFrame(
        {
            "Open": [9, 10, 11],
            "High": [10, 11, 12],
            "Low": [8, 9, 10],
            "Close": [9, 10, 11],
            "Volume": [1_000_000] * 3,
        },
        index=pd.date_range(date(2024, 1, 1), periods=3, freq="D"),
    )
    filt = VolatilityFilter(config={"metric": "atr", "lookback_days": 2, "min_value": None, "max_value": 15})
    result = filt.pass_fail(data)
    assert result.passed is False
    assert "above max" in result.reason


def test_volatility_realized_vol_zero_when_returns_are_constant():
    # Exactly +1% every day -> stdev of returns is 0 -> annualized vol = 0
    closes = [100, 101, 102.01, 103.0301]
    data = _bars(closes)
    filt = VolatilityFilter(config={"metric": "realized_vol", "lookback_days": 3, "min_value": None, "max_value": 5})
    result = filt.pass_fail(data)
    assert result.passed is True
    assert "0.00%" in result.reason


def test_volatility_realized_vol_fails_when_clearly_too_high():
    closes = [100, 130, 70, 120]  # wild swings
    data = _bars(closes)
    filt = VolatilityFilter(config={"metric": "realized_vol", "lookback_days": 3, "min_value": None, "max_value": 1})
    result = filt.pass_fail(data)
    assert result.passed is False


def test_volatility_fails_with_insufficient_history():
    data = _bars([10, 10])
    filt = VolatilityFilter(config={"metric": "atr", "lookback_days": 5, "min_value": None, "max_value": 100})
    result = filt.pass_fail(data)
    assert result.passed is False
    assert "only 2 bars available" in result.reason


# -----------------------------------------------------------------------------
# VolumeSurgeFilter
# -----------------------------------------------------------------------------


def test_volume_surge_passes_above_multiple():
    data = _bars([10, 10, 10, 10], volumes=[100, 100, 100, 300])  # avg=100, last=300 -> 3.0x
    filt = VolumeSurgeFilter(config={"avg_lookback_days": 3, "direction": "above", "threshold_multiple": 2.0})
    result = filt.pass_fail(data)
    assert result.passed is True
    assert "3.00x" in result.reason


def test_volume_surge_fails_below_multiple():
    data = _bars([10, 10, 10, 10], volumes=[100, 100, 100, 150])  # avg=100, last=150 -> 1.5x
    filt = VolumeSurgeFilter(config={"avg_lookback_days": 3, "direction": "above", "threshold_multiple": 2.0})
    result = filt.pass_fail(data)
    assert result.passed is False


def test_volume_surge_direction_below_passes_when_volume_has_dried_up():
    data = _bars([10, 10, 10, 10], volumes=[100, 100, 100, 60])  # avg=100, last=60 -> 0.6x
    filt = VolumeSurgeFilter(config={"avg_lookback_days": 3, "direction": "below", "threshold_multiple": 0.8})
    result = filt.pass_fail(data)
    assert result.passed is True
    assert "0.60x" in result.reason


def test_volume_surge_direction_below_fails_when_volume_has_not_dried_up():
    data = _bars([10, 10, 10, 10], volumes=[100, 100, 100, 150])  # avg=100, last=150 -> 1.5x
    filt = VolumeSurgeFilter(config={"avg_lookback_days": 3, "direction": "below", "threshold_multiple": 0.8})
    result = filt.pass_fail(data)
    assert result.passed is False


def test_volume_surge_defaults_to_direction_above_when_unset():
    data = _bars([10, 10, 10, 10], volumes=[100, 100, 100, 300])
    filt = VolumeSurgeFilter(config={"avg_lookback_days": 3, "threshold_multiple": 2.0})
    assert filt.pass_fail(data).passed is True


def test_volume_surge_fails_with_insufficient_history():
    data = _bars([10, 10], volumes=[100, 100])
    filt = VolumeSurgeFilter(config={"avg_lookback_days": 5, "direction": "above", "threshold_multiple": 2.0})
    result = filt.pass_fail(data)
    assert result.passed is False
    assert "only 2 bars available" in result.reason


# -----------------------------------------------------------------------------
# GapFilter
# -----------------------------------------------------------------------------


def test_gap_filter_excludes_symbol_on_blacklist():
    data = _bars([100])
    filt = GapFilter(config={"exclusion_days": 3, "excluded_symbols": ["BADCO"], "event_dates": {}})
    context = {"symbol": "BADCO", "evaluation_date": date(2024, 1, 1)}
    result = filt.pass_fail(data, context)
    assert result.passed is False
    assert "excluded_symbols" in result.reason


def test_gap_filter_excludes_symbol_near_known_event():
    data = _bars([100])
    filt = GapFilter(
        config={
            "exclusion_days": 3,
            "excluded_symbols": [],
            "event_dates": {"AAPL": ["2024-01-10"]},
        }
    )
    context = {"symbol": "AAPL", "evaluation_date": date(2024, 1, 12)}  # 2 days away
    result = filt.pass_fail(data, context)
    assert result.passed is False
    assert "known event" in result.reason


def test_gap_filter_passes_when_event_is_outside_window():
    data = _bars([100])
    filt = GapFilter(
        config={
            "exclusion_days": 3,
            "excluded_symbols": [],
            "event_dates": {"AAPL": ["2024-01-10"]},
        }
    )
    context = {"symbol": "AAPL", "evaluation_date": date(2024, 1, 1)}  # 9 days away
    result = filt.pass_fail(data, context)
    assert result.passed is True


def test_gap_filter_passes_symbol_with_no_configured_events():
    data = _bars([100])
    filt = GapFilter(config={"exclusion_days": 3, "excluded_symbols": [], "event_dates": {}})
    context = {"symbol": "MSFT", "evaluation_date": date(2024, 1, 1)}
    result = filt.pass_fail(data, context)
    assert result.passed is True


# -----------------------------------------------------------------------------
# score_component — the continuous "strength" signal each filter feeds into
# ranking.py's composite score (see also tests/test_ranking.py).
# -----------------------------------------------------------------------------


def test_momentum_score_component_is_the_raw_return_when_direction_above():
    data = _bars([100, 100, 100, 110])  # 3-day return = +10%
    filt = MomentumFilter(config={"lookback_days": 3, "direction": "above", "threshold": 0.0})
    assert filt.score_component(data) == pytest.approx(0.10)


def test_momentum_score_component_flips_sign_when_direction_below():
    """Hunting for weakness (direction: below) should score the weakest
    momentum highest, not the strongest — so the raw return is negated."""
    data = _bars([100, 100, 100, 110])  # +10% raw return
    filt = MomentumFilter(config={"lookback_days": 3, "direction": "below", "threshold": 0.0})
    assert filt.score_component(data) == pytest.approx(-0.10)


def test_momentum_score_component_uses_benchmark_when_configured():
    symbol_data = _bars([100, 100, 100, 110])  # +10%
    benchmark_data = _bars([100, 100, 100, 103])  # +3%
    filt = MomentumFilter(
        config={"lookback_days": 3, "direction": "above", "threshold": 0.0, "benchmark_symbol": "SPY"}
    )
    score = filt.score_component(symbol_data, context={"reference_data": {"SPY": benchmark_data}})
    assert score == pytest.approx(0.07)


def test_momentum_score_component_is_none_with_insufficient_history():
    data = _bars([100, 100])
    filt = MomentumFilter(config={"lookback_days": 3, "direction": "above", "threshold": 0.0})
    assert filt.score_component(data) is None


def test_trend_score_component_positive_when_above_ma_and_direction_above():
    data = _bars([10, 11, 12, 13])  # tail(3) mean = 12, close = 13 -> +8.33%
    filt = TrendFilter(config={"period": 3, "direction": "above", "require_slope": "none"})
    score = filt.score_component(data)
    assert score == pytest.approx((13 - 12) / 12)


def test_trend_score_component_flips_sign_when_direction_below():
    data = _bars([10, 11, 12, 13])  # same data, opposite intent
    filt = TrendFilter(config={"period": 3, "direction": "below", "require_slope": "none"})
    score = filt.score_component(data)
    assert score == pytest.approx(-(13 - 12) / 12)


def test_volume_surge_score_component_is_the_multiple_when_direction_above():
    data = _bars([10, 10, 10, 10], volumes=[100, 100, 100, 300])
    filt = VolumeSurgeFilter(config={"avg_lookback_days": 3, "direction": "above", "threshold_multiple": 2.0})
    assert filt.score_component(data) == pytest.approx(3.0)


def test_volume_surge_score_component_flips_sign_when_direction_below():
    """Drying-up screens should score the most contracted volume highest."""
    data = _bars([10, 10, 10, 10], volumes=[100, 100, 100, 60])  # 0.6x
    filt = VolumeSurgeFilter(config={"avg_lookback_days": 3, "direction": "below", "threshold_multiple": 0.8})
    assert filt.score_component(data) == pytest.approx(-0.6)


def test_volume_surge_score_component_is_none_with_insufficient_history():
    data = _bars([10, 10], volumes=[100, 100])
    filt = VolumeSurgeFilter(config={"avg_lookback_days": 5, "direction": "above", "threshold_multiple": 2.0})
    assert filt.score_component(data) is None


def test_proximity_score_component_rewards_closeness_regardless_of_side():
    close_data = _bars([100, 101, 103], highs=[100, 101, 105])  # 1.90% below high
    far_data = _bars([100, 101, 95], highs=[100, 101, 105])  # 9.52% below high
    filt = ProximityFilter(config={"reference": "high", "lookback_days": 3, "max_distance_pct": 50.0, "side": "any"})

    close_score = filt.score_component(close_data)
    far_score = filt.score_component(far_data)

    assert close_score > far_score  # closer to the trigger scores higher
    assert close_score < 0  # oriented as negative distance


def test_proximity_score_component_is_none_with_insufficient_history():
    data = _bars([10, 10])
    filt = ProximityFilter(config={"reference": "high", "lookback_days": 5, "max_distance_pct": 1.0, "side": "any"})
    assert filt.score_component(data) is None


def test_filters_without_a_score_component_return_none_by_default():
    data = _bars([10])
    assert PriceRangeFilter(config={"min_price": 1, "max_price": None}).score_component(data) is None
    assert LiquidityFilter(
        config={"metric": "dollar_volume", "lookback_days": 1, "min_value": 0}
    ).score_component(data) is None
