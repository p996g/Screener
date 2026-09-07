"""
Tests for main.py's failure handling: a run that blows up inside the
engine must be caught, logged, and reported as a failure — never crash the
process or vanish silently (see ProfileRunResult / run()).

The engine itself is replaced with tiny stand-ins so these tests never
touch the network; they only exercise main.py's own try/except and
aggregation logic.
"""

import logging

import main as main_module


def _minimal_config(**profile_overrides):
    return {
        "universe": {"source": "static_list", "tickers": ["AAPL"]},
        "data": {
            "provider": "yfinance",
            "interval": "1d",
            "lookback_days": 30,
            "use_cache": False,
            "cache_dir": None,
        },
        "filters": {"liquidity": {"enabled": False}},
        "ranking": {"weights": {}, "top_n": 10, "min_score": None},
        "output": {
            "formats": [],
            "csv_path": "unused.csv",
            "dashboard_path": "unused_dashboard.html",
            "show_filter_reasons": True,
            "console_columns": [],
            "save_history": False,
        },
        "profiles": {"demo": {}, **profile_overrides},
    }


class ExplodingEngine:
    """Stands in for ScreenerEngine: always raises, simulating a total
    data-fetch outage or other unexpected runtime failure."""

    def __init__(self, filters, evaluation_date, data_quality_config=None):
        pass

    def run(self, universe, fetcher):
        raise RuntimeError("network is down")


class EmptyEngine:
    """Stands in for ScreenerEngine: always succeeds with zero results."""

    def __init__(self, filters, evaluation_date, data_quality_config=None):
        pass

    def run(self, universe, fetcher):
        return []


def test_run_profile_catches_engine_failure_and_returns_it_as_an_error(monkeypatch):
    monkeypatch.setattr(main_module, "ScreenerEngine", ExplodingEngine)
    result = main_module.run_profile(_minimal_config(), "demo")

    assert result.error == "network is down"
    assert result.ranked == []
    assert result.results == []


def test_run_profile_does_not_set_error_on_success(monkeypatch):
    monkeypatch.setattr(main_module, "ScreenerEngine", EmptyEngine)
    result = main_module.run_profile(_minimal_config(), "demo")
    assert result.error is None


def _isolate_output(config: dict, tmp_path) -> None:
    """Point history_dir/dashboard_path at tmp_path so tests calling
    main.run() never write into the real project's output/ directory
    (write_dashboard runs unconditionally on every run())."""
    config["output"]["history_dir"] = str(tmp_path / "history")
    config["output"]["dashboard_path"] = str(tmp_path / "dashboard.html")


def test_run_single_profile_returns_true_on_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(main_module, "ScreenerEngine", ExplodingEngine)
    config = _minimal_config()
    _isolate_output(config, tmp_path)

    assert main_module.run(config, "demo") is True


def test_run_single_profile_returns_false_on_success(monkeypatch, tmp_path):
    monkeypatch.setattr(main_module, "ScreenerEngine", EmptyEngine)
    config = _minimal_config()
    _isolate_output(config, tmp_path)

    assert main_module.run(config, "demo") is False


def test_run_all_returns_true_if_any_profile_fails(monkeypatch, tmp_path):
    calls = {"n": 0}

    class SometimesExplodingEngine:
        def __init__(self, filters, evaluation_date, data_quality_config=None):
            pass

        def run(self, universe, fetcher):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("first profile's fetch failed")
            return []

    monkeypatch.setattr(main_module, "ScreenerEngine", SometimesExplodingEngine)
    config = _minimal_config(a={}, b={})
    config["profiles"] = {"a": {}, "b": {}}
    _isolate_output(config, tmp_path)

    assert main_module.run(config, "all") is True
    assert calls["n"] == 2  # the second profile still ran despite the first failing


def test_run_single_profile_actually_writes_its_csv_report(monkeypatch, tmp_path):
    """Regression test: run() must pass run_profile's *output section*
    (not the whole resolved profile config) into write_reports, or the
    report silently never gets written/printed at all."""
    monkeypatch.setattr(main_module, "ScreenerEngine", EmptyEngine)
    config = _minimal_config()
    _isolate_output(config, tmp_path)
    config["output"]["formats"] = ["csv"]
    config["output"]["csv_path"] = str(tmp_path / "demo.csv")

    main_module.run(config, "demo")

    assert (tmp_path / "demo.csv").exists()


def test_run_single_profile_failure_actually_writes_its_csv_report(monkeypatch, tmp_path):
    monkeypatch.setattr(main_module, "ScreenerEngine", ExplodingEngine)
    config = _minimal_config()
    _isolate_output(config, tmp_path)
    config["output"]["formats"] = ["csv"]
    config["output"]["csv_path"] = str(tmp_path / "demo.csv")

    main_module.run(config, "demo")

    csv_path = tmp_path / "demo.csv"
    assert csv_path.exists()
    assert "RUN FAILED" in csv_path.read_text() or "network is down" in csv_path.read_text()


def test_run_writes_a_history_snapshot(monkeypatch, tmp_path):
    monkeypatch.setattr(main_module, "ScreenerEngine", EmptyEngine)
    config = _minimal_config()
    _isolate_output(config, tmp_path)
    config["output"]["save_history"] = True

    main_module.run(config, "demo", run_label="demo")

    saved = list((tmp_path / "history").glob("*_demo.csv"))
    assert len(saved) == 1


def test_run_regenerates_the_dashboard(monkeypatch, tmp_path):
    monkeypatch.setattr(main_module, "ScreenerEngine", EmptyEngine)
    config = _minimal_config()
    _isolate_output(config, tmp_path)

    main_module.run(config, "demo")

    assert (tmp_path / "dashboard.html").exists()
    assert "Screener Dashboard" in (tmp_path / "dashboard.html").read_text()


def test_configure_logging_creates_log_file_directory(tmp_path):
    config = {"logging": {"level": "INFO", "file": str(tmp_path / "nested" / "screener.log")}}
    try:
        main_module.configure_logging(config)
        assert (tmp_path / "nested").is_dir()
    finally:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            handler.close()
