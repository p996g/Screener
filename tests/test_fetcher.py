"""
Tests for data/fetcher.py's failure handling: a network/provider exception
while downloading one symbol must be logged clearly and turned into an
empty (but valid) result, never raised — see the module docstring in
data/fetcher.py for why this is what keeps a scheduled run from crashing
outright over one bad symbol.
"""

import logging
from datetime import date

import data.fetcher as fetcher_module
from data.fetcher import OHLCVFetcher


def _fetcher(**overrides):
    config = {
        "provider": "yfinance",
        "interval": "1d",
        "lookback_days": 30,
        "use_cache": False,
        "cache_dir": None,
    }
    config.update(overrides)
    return OHLCVFetcher(config)


def test_fetch_returns_empty_dataframe_when_download_raises(monkeypatch):
    def boom(*args, **kwargs):
        raise ConnectionError("network unreachable")

    monkeypatch.setattr(fetcher_module.yf, "download", boom)

    result = _fetcher().fetch("AAPL", date(2024, 1, 2))

    assert result.empty
    assert list(result.columns) == fetcher_module.OHLCV_COLUMNS


def test_fetch_failure_is_logged_clearly(monkeypatch, caplog):
    def boom(*args, **kwargs):
        raise ConnectionError("network unreachable")

    monkeypatch.setattr(fetcher_module.yf, "download", boom)

    with caplog.at_level(logging.ERROR):
        _fetcher().fetch("AAPL", date(2024, 1, 2))

    assert "AAPL" in caplog.text
    assert "Data fetch failed" in caplog.text


def test_fetch_does_not_raise_even_for_an_unexpected_exception_type(monkeypatch):
    def boom(*args, **kwargs):
        raise ValueError("yfinance internal error")

    monkeypatch.setattr(fetcher_module.yf, "download", boom)

    # Must not raise.
    result = _fetcher().fetch("MSFT", date(2024, 1, 2))
    assert result.empty


def test_fetch_succeeds_normally_when_download_does_not_raise(monkeypatch):
    import pandas as pd

    good_data = pd.DataFrame(
        {"Open": [1], "High": [1], "Low": [1], "Close": [1], "Volume": [1]},
        index=pd.date_range("2024-01-02", periods=1),
    )
    monkeypatch.setattr(fetcher_module.yf, "download", lambda *a, **k: good_data)

    result = _fetcher().fetch("AAPL", date(2024, 1, 2))
    assert not result.empty
