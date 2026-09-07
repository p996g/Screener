"""
Tests for ScreenerEngine (engine.py).

Uses fake filters and a fake fetcher so these tests never touch the network
or real market data — they only verify the engine's own logic: iterating
the universe, invoking every filter, aggregating pass/fail, and enforcing
point-in-time truncation.
"""

from datetime import date, timedelta

import pandas as pd

from engine import ScreenerEngine, SymbolResult
from filters.base import Filter, FilterResult


def _make_ohlcv(dates: list[date], close: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [close] * len(dates),
            "High": [close + 1] * len(dates),
            "Low": [close - 1] * len(dates),
            "Close": [close] * len(dates),
            "Volume": [1_000_000] * len(dates),
        },
        index=pd.DatetimeIndex(dates),
    )


class AlwaysPassFilter(Filter):
    name = "always_pass"

    def pass_fail(self, symbol_data: pd.DataFrame, context: dict | None = None) -> FilterResult:
        return FilterResult(True, "always passes")


class AlwaysFailFilter(Filter):
    name = "always_fail"

    def pass_fail(self, symbol_data: pd.DataFrame, context: dict | None = None) -> FilterResult:
        return FilterResult(False, "always fails")


class RecordsLastCloseFilter(Filter):
    """Fails if the last visible close is above a threshold, so tests can
    prove which rows the filter actually saw."""

    name = "max_close"

    def pass_fail(self, symbol_data: pd.DataFrame, context: dict | None = None) -> FilterResult:
        last_close = float(symbol_data["Close"].iloc[-1])
        max_close = self.config["max_close"]
        if last_close <= max_close:
            return FilterResult(True, f"{last_close} <= {max_close}")
        return FilterResult(False, f"{last_close} <= {max_close} is False")


class FakeFetcher:
    """Returns pre-built DataFrames per symbol, standing in for
    data/fetcher.py's OHLCVFetcher without any network access."""

    def __init__(self, data_by_symbol: dict[str, pd.DataFrame]):
        self.data_by_symbol = data_by_symbol
        self.fetch_calls: list[tuple[str, date]] = []

    def fetch(self, symbol: str, evaluation_date: date) -> pd.DataFrame:
        self.fetch_calls.append((symbol, evaluation_date))
        return self.data_by_symbol.get(symbol, pd.DataFrame())


TODAY = date(2024, 1, 10)


def test_symbol_passing_every_filter_survives():
    dates = [TODAY - timedelta(days=i) for i in range(5)][::-1]
    fetcher = FakeFetcher({"AAA": _make_ohlcv(dates)})
    engine = ScreenerEngine(
        filters=[AlwaysPassFilter(config={})], evaluation_date=TODAY
    )

    results = engine.run(["AAA"], fetcher)

    assert len(results) == 1
    assert results[0].symbol == "AAA"
    assert results[0].passed is True
    assert results[0].filter_results["always_pass"].passed is True


def test_symbol_failing_any_filter_does_not_survive():
    dates = [TODAY - timedelta(days=i) for i in range(5)][::-1]
    fetcher = FakeFetcher({"BBB": _make_ohlcv(dates)})
    engine = ScreenerEngine(
        filters=[AlwaysPassFilter(config={}), AlwaysFailFilter(config={})],
        evaluation_date=TODAY,
    )

    results = engine.run(["BBB"], fetcher)

    assert results[0].passed is False
    assert results[0].filter_results["always_pass"].passed is True
    assert results[0].filter_results["always_fail"].passed is False


def test_every_enabled_filter_runs_and_is_recorded_per_symbol():
    dates = [TODAY - timedelta(days=i) for i in range(3)][::-1]
    fetcher = FakeFetcher({"CCC": _make_ohlcv(dates)})
    filters = [
        AlwaysPassFilter(config={}),
        AlwaysFailFilter(config={}),
        RecordsLastCloseFilter(config={"max_close": 50}),
    ]
    engine = ScreenerEngine(filters=filters, evaluation_date=TODAY)

    results = engine.run(["CCC"], fetcher)

    result = results[0]
    assert set(result.filter_results.keys()) == {
        "always_pass",
        "always_fail",
        "max_close",
    }
    assert result.failure_reasons == [
        "always_fail: always fails",
        "max_close: 100.0 <= 50 is False",
    ]


def test_missing_data_fails_cleanly_with_explanatory_reason():
    fetcher = FakeFetcher({})  # no data for any symbol
    engine = ScreenerEngine(
        filters=[AlwaysPassFilter(config={})], evaluation_date=TODAY
    )

    results = engine.run(["NODATA"], fetcher)

    result = results[0]
    assert result.passed is False
    assert "data_availability" in result.filter_results
    assert result.filter_results["data_availability"].passed is False


def test_engine_truncates_future_bars_even_if_fetcher_does_not():
    """Point-in-time discipline must hold even if a fetcher implementation
    forgets to truncate — the engine is the final backstop against
    look-ahead bias."""
    dates_including_future = [
        TODAY - timedelta(days=1),
        TODAY,
        TODAY + timedelta(days=5),  # future bar, must never reach a filter
    ]
    data_with_future_leak = _make_ohlcv(dates_including_future, close=100.0)
    # Plant a distinct close on the future row so we can prove it was seen
    # (or not seen) by the filter.
    data_with_future_leak.loc[
        pd.Timestamp(TODAY + timedelta(days=5)), "Close"
    ] = 999.0

    fetcher = FakeFetcher({"DDD": data_with_future_leak})
    engine = ScreenerEngine(
        filters=[RecordsLastCloseFilter(config={"max_close": 200})],
        evaluation_date=TODAY,
    )

    results = engine.run(["DDD"], fetcher)

    result = results[0]
    assert result.data.index.max() <= pd.Timestamp(TODAY)
    assert result.filter_results["max_close"].passed is True
    assert "999" not in result.filter_results["max_close"].reason


def test_run_iterates_the_full_universe_in_order():
    dates = [TODAY]
    fetcher = FakeFetcher(
        {
            "AAA": _make_ohlcv(dates),
            "BBB": _make_ohlcv(dates),
            "CCC": _make_ohlcv(dates),
        }
    )
    engine = ScreenerEngine(
        filters=[AlwaysPassFilter(config={})], evaluation_date=TODAY
    )

    results = engine.run(["AAA", "BBB", "CCC"], fetcher)

    assert [r.symbol for r in results] == ["AAA", "BBB", "CCC"]
    assert fetcher.fetch_calls == [
        ("AAA", TODAY),
        ("BBB", TODAY),
        ("CCC", TODAY),
    ]


class RecordsContextFilter(Filter):
    """Captures every context dict it's called with, so tests can assert on
    what the engine actually passes through to filters."""

    name = "records_context"

    def __init__(self, config: dict):
        super().__init__(config)
        self.seen_contexts: list[dict] = []

    def required_symbols(self) -> list[str]:
        return self.config.get("required_symbols", [])

    def pass_fail(self, symbol_data: pd.DataFrame, context: dict | None = None) -> FilterResult:
        self.seen_contexts.append(context)
        return FilterResult(True, "recorded")


def test_context_carries_symbol_and_evaluation_date():
    dates = [TODAY]
    fetcher = FakeFetcher({"AAA": _make_ohlcv(dates)})
    recorder = RecordsContextFilter(config={})
    engine = ScreenerEngine(filters=[recorder], evaluation_date=TODAY)

    engine.run(["AAA"], fetcher)

    assert recorder.seen_contexts[0]["symbol"] == "AAA"
    assert recorder.seen_contexts[0]["evaluation_date"] == TODAY


def test_required_symbols_are_fetched_once_and_shared_via_context():
    """A benchmark symbol requested by a filter's required_symbols() should
    be fetched a single time per run and handed to every symbol's filter
    call via context['reference_data'] — not re-fetched per symbol."""
    dates = [TODAY - timedelta(days=i) for i in range(3)][::-1]
    fetcher = FakeFetcher(
        {
            "AAA": _make_ohlcv(dates),
            "BBB": _make_ohlcv(dates),
            "SPY": _make_ohlcv(dates, close=200.0),
        }
    )
    recorder = RecordsContextFilter(config={"required_symbols": ["SPY"]})
    engine = ScreenerEngine(filters=[recorder], evaluation_date=TODAY)

    engine.run(["AAA", "BBB"], fetcher)

    spy_fetch_count = sum(1 for symbol, _ in fetcher.fetch_calls if symbol == "SPY")
    assert spy_fetch_count == 1

    for ctx in recorder.seen_contexts:
        benchmark_df = ctx["reference_data"]["SPY"]
        assert list(benchmark_df["Close"]) == [200.0, 200.0, 200.0]


def test_symbol_result_passed_is_false_when_no_filters_ran():
    """A SymbolResult with zero filter results (edge case) should not be
    treated as a silent pass."""
    result = SymbolResult(symbol="XXX", data=pd.DataFrame(), filter_results={})
    assert result.passed is False


# -----------------------------------------------------------------------------
# Data-quality gating (see data/quality.py) — runs before any filter, and
# short-circuits the filter pipeline entirely when it flags a symbol.
# -----------------------------------------------------------------------------


def test_data_quality_disabled_by_default_runs_filters_normally():
    dates = [TODAY - timedelta(days=i) for i in range(5)][::-1]
    fetcher = FakeFetcher({"AAA": _make_ohlcv(dates)})
    engine = ScreenerEngine(filters=[AlwaysPassFilter(config={})], evaluation_date=TODAY)

    results = engine.run(["AAA"], fetcher)

    assert "data_quality" not in results[0].filter_results
    assert results[0].passed is True


def test_data_quality_excludes_symbol_before_any_filter_runs():
    # Stale: last bar is 30 days before the evaluation date.
    dates = [TODAY - timedelta(days=35 + i) for i in range(5)][::-1]
    fetcher = FakeFetcher({"AAA": _make_ohlcv(dates)})
    engine = ScreenerEngine(
        filters=[AlwaysPassFilter(config={})],
        evaluation_date=TODAY,
        data_quality_config={"enabled": True, "max_staleness_days": 5},
    )

    results = engine.run(["AAA"], fetcher)
    result = results[0]

    assert result.passed is False
    assert set(result.filter_results.keys()) == {"data_quality"}
    assert "stale data" in result.filter_results["data_quality"].reason
    assert "always_pass" not in result.filter_results  # the filter never ran


def test_data_quality_does_not_exclude_clean_data():
    dates = [TODAY - timedelta(days=i) for i in range(5)][::-1]
    fetcher = FakeFetcher({"AAA": _make_ohlcv(dates)})
    engine = ScreenerEngine(
        filters=[AlwaysPassFilter(config={})],
        evaluation_date=TODAY,
        data_quality_config={"enabled": True, "max_staleness_days": 5},
    )

    results = engine.run(["AAA"], fetcher)

    assert "data_quality" not in results[0].filter_results
    assert results[0].passed is True
