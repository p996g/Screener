"""
Tests for data/quality.py's per-symbol data-quality checks.
"""

from datetime import date

import pandas as pd

from data.quality import check_symbol_data_quality

DEFAULT_CONFIG = {"max_staleness_days": 5, "max_gap_days": 5, "max_daily_move_pct": 50.0}


def _bars(closes, dates, highs=None, lows=None, volumes=None):
    n = len(closes)
    highs = highs if highs is not None else [c + 1 for c in closes]
    lows = lows if lows is not None else [c - 1 for c in closes]
    volumes = volumes if volumes is not None else [1_000_000] * n
    return pd.DataFrame(
        {"Open": closes, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=pd.DatetimeIndex(dates),
    )


def test_empty_dataframe_has_no_issues():
    assert check_symbol_data_quality(pd.DataFrame(), date(2024, 1, 10), DEFAULT_CONFIG) == []


def test_clean_data_has_no_issues():
    dates = pd.bdate_range("2024-01-02", "2024-01-10")
    closes = [100, 101, 100.5, 102, 101.5, 103, 102.5]
    data = _bars(closes, dates[: len(closes)])
    issues = check_symbol_data_quality(data, date(2024, 1, 9), DEFAULT_CONFIG)
    assert issues == []


def test_stale_data_is_flagged():
    dates = [date(2024, 1, 2), date(2024, 1, 3)]
    data = _bars([100, 101], dates)
    issues = check_symbol_data_quality(data, date(2024, 1, 15), DEFAULT_CONFIG)
    assert any("stale data" in issue for issue in issues)


def test_data_within_staleness_window_is_not_flagged():
    dates = [date(2024, 1, 8), date(2024, 1, 9)]
    data = _bars([100, 101], dates)
    # 2024-01-12 is 3 days after the last bar -- within the 5-day default
    issues = check_symbol_data_quality(data, date(2024, 1, 12), DEFAULT_CONFIG)
    assert not any("stale data" in issue for issue in issues)


def test_large_gap_between_bars_is_flagged():
    dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 25)]  # 22-day gap
    data = _bars([100, 101, 102], dates)
    issues = check_symbol_data_quality(data, date(2024, 1, 25), DEFAULT_CONFIG)
    assert any("data gap" in issue for issue in issues)


def test_normal_weekend_gap_is_not_flagged():
    dates = pd.bdate_range("2024-01-02", "2024-01-10")  # normal weekday gaps only
    data = _bars([100] * len(dates), dates)
    issues = check_symbol_data_quality(data, dates[-1].date(), DEFAULT_CONFIG)
    assert not any("data gap" in issue for issue in issues)


def test_non_positive_price_is_flagged():
    dates = [date(2024, 1, 2), date(2024, 1, 3)]
    data = _bars([100, -5], dates)
    issues = check_symbol_data_quality(data, date(2024, 1, 3), DEFAULT_CONFIG)
    assert any("non-positive" in issue for issue in issues)


def test_high_below_low_is_flagged():
    dates = [date(2024, 1, 2), date(2024, 1, 3)]
    data = _bars([100, 101], dates, highs=[101, 90], lows=[99, 95])
    issues = check_symbol_data_quality(data, date(2024, 1, 3), DEFAULT_CONFIG)
    assert any("High < Low" in issue for issue in issues)


def test_extreme_daily_move_is_flagged():
    dates = [date(2024, 1, 2), date(2024, 1, 3)]
    data = _bars([100, 200], dates)  # +100% in one day
    issues = check_symbol_data_quality(data, date(2024, 1, 3), DEFAULT_CONFIG)
    assert any("suspicious price" in issue and "single-day move" in issue for issue in issues)


def test_moderate_daily_move_is_not_flagged():
    dates = [date(2024, 1, 2), date(2024, 1, 3)]
    data = _bars([100, 110], dates)  # +10%
    issues = check_symbol_data_quality(data, date(2024, 1, 3), DEFAULT_CONFIG)
    assert not any("single-day move" in issue for issue in issues)


def test_thresholds_are_configurable():
    dates = [date(2024, 1, 2), date(2024, 1, 3)]
    data = _bars([100, 110], dates)  # +10%
    strict_config = {**DEFAULT_CONFIG, "max_daily_move_pct": 5.0}
    issues = check_symbol_data_quality(data, date(2024, 1, 3), strict_config)
    assert any("single-day move" in issue for issue in issues)


def test_multiple_issues_can_be_reported_together():
    dates = [date(2024, 1, 2), date(2024, 1, 3)]
    data = _bars([100, -5], dates)  # non-positive price + will also be stale
    issues = check_symbol_data_quality(data, date(2024, 2, 1), DEFAULT_CONFIG)
    assert len(issues) >= 2
