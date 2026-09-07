"""
Tests for the combined, grouped-by-profile report (output/report.py), plus
failure-reporting (`error`/`errors_by_profile`) and the history snapshot
writer.
"""

from datetime import datetime, timezone

import pandas as pd

from engine import SymbolResult
from filters.base import FilterResult
from output.report import save_history_snapshot, write_combined_report, write_reports
from ranking import RankedSymbol


def _result(symbol: str, close: float, passed: bool = True) -> SymbolResult:
    return SymbolResult(
        symbol=symbol,
        data=pd.DataFrame({"Close": [close]}),
        filter_results={"dummy": FilterResult(passed, "ok" if passed else "nope")},
        score_components={},
    )


def _ranked(symbol: str, score: float, close: float) -> RankedSymbol:
    return RankedSymbol(symbol=symbol, score=score, close=close, component_scores={})


def test_combined_report_tags_every_row_with_its_profile():
    ranked_by_profile = {
        "momentum_screen": [_ranked("AAA", 0.9, 10.0), _ranked("BBB", 0.5, 20.0)],
        "breakout_watch": [_ranked("CCC", 0.7, 30.0)],
    }
    results_by_profile = {
        "momentum_screen": [_result("AAA", 10.0), _result("BBB", 20.0)],
        "breakout_watch": [_result("CCC", 30.0)],
    }

    df = write_combined_report(
        ranked_by_profile, results_by_profile, {"formats": []}
    )

    assert list(df["profile"]) == ["momentum_screen", "momentum_screen", "breakout_watch"]
    assert list(df["symbol"]) == ["AAA", "BBB", "CCC"]


def test_combined_report_rank_restarts_per_profile():
    ranked_by_profile = {
        "screen_a": [_ranked("AAA", 0.9, 10.0), _ranked("BBB", 0.5, 20.0)],
        "screen_b": [_ranked("CCC", 0.7, 30.0)],
    }
    results_by_profile = {
        "screen_a": [_result("AAA", 10.0), _result("BBB", 20.0)],
        "screen_b": [_result("CCC", 30.0)],
    }

    df = write_combined_report(ranked_by_profile, results_by_profile, {"formats": []})

    ranks_by_profile = df.groupby("profile")["rank"].apply(list).to_dict()
    assert ranks_by_profile["screen_a"] == [1, 2]
    assert ranks_by_profile["screen_b"] == [1]


def test_combined_report_handles_a_profile_with_no_survivors():
    ranked_by_profile = {"empty_screen": [], "screen_b": [_ranked("CCC", 0.7, 30.0)]}
    results_by_profile = {
        "empty_screen": [_result("XXX", 10.0, passed=False)],
        "screen_b": [_result("CCC", 30.0)],
    }

    df = write_combined_report(ranked_by_profile, results_by_profile, {"formats": []})

    assert list(df["profile"]) == ["screen_b"]
    assert list(df["symbol"]) == ["CCC"]


def test_combined_report_console_output_sections_each_profile(capsys):
    ranked_by_profile = {"screen_a": [_ranked("AAA", 0.9, 10.0)], "screen_b": []}
    results_by_profile = {
        "screen_a": [_result("AAA", 10.0)],
        "screen_b": [_result("XXX", 10.0, passed=False)],
    }

    write_combined_report(
        ranked_by_profile,
        results_by_profile,
        {"formats": ["console"], "console_columns": ["symbol", "score"]},
    )

    out = capsys.readouterr().out
    assert "=== screen_a ===" in out
    assert "AAA" in out
    assert "=== screen_b ===" in out
    assert "No qualifying candidates today." in out


# -----------------------------------------------------------------------------
# Failure handling: a failed run must be an explicit, visible report row/
# banner — never indistinguishable from "no symbols passed today".
# -----------------------------------------------------------------------------


def test_write_reports_renders_a_failed_run_as_an_explicit_row_not_empty():
    df = write_reports([], [], {"formats": []}, error="network is down")
    assert list(df["symbol"]) == ["(RUN FAILED)"]
    assert list(df["error"]) == ["network is down"]


def test_write_reports_console_shows_run_failed_banner(capsys):
    write_reports([], [], {"formats": ["console"]}, error="network is down")
    out = capsys.readouterr().out
    assert "RUN FAILED: network is down" in out


def test_write_combined_report_marks_only_the_failed_profile():
    ranked_by_profile = {"ok_screen": [_ranked("AAA", 0.9, 10.0)], "broken_screen": []}
    results_by_profile = {"ok_screen": [_result("AAA", 10.0)], "broken_screen": []}

    df = write_combined_report(
        ranked_by_profile,
        results_by_profile,
        {"formats": []},
        errors_by_profile={"broken_screen": "fetch timed out"},
    )

    ok_rows = df[df["profile"] == "ok_screen"]
    broken_rows = df[df["profile"] == "broken_screen"]
    assert list(ok_rows["symbol"]) == ["AAA"]
    assert list(broken_rows["symbol"]) == ["(RUN FAILED)"]
    assert list(broken_rows["error"]) == ["fetch timed out"]


def test_write_combined_report_console_shows_run_failed_for_broken_profile_only(capsys):
    ranked_by_profile = {"ok_screen": [_ranked("AAA", 0.9, 10.0)], "broken_screen": []}
    results_by_profile = {"ok_screen": [_result("AAA", 10.0)], "broken_screen": []}

    write_combined_report(
        ranked_by_profile,
        results_by_profile,
        {"formats": ["console"], "console_columns": ["symbol", "score"]},
        errors_by_profile={"broken_screen": "fetch timed out"},
    )

    out = capsys.readouterr().out
    assert "=== ok_screen ===" in out
    assert "AAA" in out
    assert "=== broken_screen ===" in out
    assert "RUN FAILED: fetch timed out" in out


# -----------------------------------------------------------------------------
# save_history_snapshot
# -----------------------------------------------------------------------------


def test_save_history_snapshot_writes_a_timestamped_file(tmp_path):
    report_df = pd.DataFrame([{"symbol": "AAA", "score": 1.0}])
    when = datetime(2026, 9, 6, 13, 30, 5, tzinfo=timezone.utc)

    path = save_history_snapshot(
        report_df, {"save_history": True, "history_dir": str(tmp_path)}, run_label="all", when=when
    )

    assert path == str(tmp_path / "20260906T133005Z_all.csv")
    assert (tmp_path / "20260906T133005Z_all.csv").exists()


def test_save_history_snapshot_disabled_returns_none_and_writes_nothing(tmp_path):
    report_df = pd.DataFrame([{"symbol": "AAA", "score": 1.0}])
    when = datetime(2026, 9, 6, 13, 30, 5, tzinfo=timezone.utc)

    path = save_history_snapshot(
        report_df, {"save_history": False, "history_dir": str(tmp_path)}, run_label="all", when=when
    )

    assert path is None
    assert list(tmp_path.iterdir()) == []
