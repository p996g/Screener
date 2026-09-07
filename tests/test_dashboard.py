"""
Tests for output/dashboard.py: history scanning, the new/dropped/big-move
diff logic, and the rendered HTML itself.
"""

from dataclasses import dataclass, field

import pandas as pd

from engine import SymbolResult
from filters.base import FilterResult
from output.dashboard import (
    build_today_rows,
    compute_changes,
    load_latest_previous_snapshot,
    render_document,
    render_fragment,
    scan_recent_run_counts,
    write_dashboard,
)
from ranking import RankedSymbol


# A minimal stand-in for main.ProfileRunResult — dashboard.py only ever
# duck-types on .profile_name/.error/.ranked/.results, so a real import of
# main.py (which imports output.dashboard) isn't needed here.
@dataclass
class _FakeRunResult:
    profile_name: str
    ranked: list = field(default_factory=list)
    results: list = field(default_factory=list)
    error: str | None = None


def _symbol_result(symbol: str, close: float, filters: dict[str, tuple[bool, str]]) -> SymbolResult:
    return SymbolResult(
        symbol=symbol,
        data=pd.DataFrame({"Close": [close]}),
        filter_results={name: FilterResult(passed, reason) for name, (passed, reason) in filters.items()},
        score_components={},
    )


def _ranked(symbol: str, score: float, close: float, components: dict[str, float] | None = None) -> RankedSymbol:
    return RankedSymbol(symbol=symbol, score=score, close=close, component_scores=components or {})


def _write_history_csv(history_dir, filename: str, rows: list[dict]) -> None:
    history_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(history_dir / filename, index=False)


# -----------------------------------------------------------------------------
# load_latest_previous_snapshot
# -----------------------------------------------------------------------------


def test_load_latest_previous_snapshot_none_when_directory_missing(tmp_path):
    assert load_latest_previous_snapshot(str(tmp_path / "nope"), "demo") is None


def test_load_latest_previous_snapshot_none_when_profile_never_appears(tmp_path):
    _write_history_csv(tmp_path, "20260101T000000Z_all.csv", [{"profile": "other", "symbol": "AAA", "rank": 1}])
    assert load_latest_previous_snapshot(str(tmp_path), "demo") is None


def test_load_latest_previous_snapshot_uses_the_newest_matching_file(tmp_path):
    _write_history_csv(tmp_path, "20260101T000000Z_all.csv", [{"profile": "demo", "symbol": "OLD", "rank": 1}])
    _write_history_csv(tmp_path, "20260102T000000Z_all.csv", [{"profile": "demo", "symbol": "NEW", "rank": 1}])

    result = load_latest_previous_snapshot(str(tmp_path), "demo")

    assert list(result["symbol"]) == ["NEW"]


def test_load_latest_previous_snapshot_filters_to_the_requested_profile(tmp_path):
    _write_history_csv(
        tmp_path,
        "20260101T000000Z_all.csv",
        [{"profile": "demo", "symbol": "AAA", "rank": 1}, {"profile": "other", "symbol": "ZZZ", "rank": 1}],
    )

    result = load_latest_previous_snapshot(str(tmp_path), "demo")

    assert list(result["symbol"]) == ["AAA"]


# -----------------------------------------------------------------------------
# scan_recent_run_counts
# -----------------------------------------------------------------------------


def test_scan_recent_run_counts_empty_when_no_history(tmp_path):
    assert scan_recent_run_counts(str(tmp_path / "nope"), "demo", 10) == []


def test_scan_recent_run_counts_counts_rows_per_run(tmp_path):
    _write_history_csv(
        tmp_path, "20260101T000000Z_all.csv",
        [{"profile": "demo", "symbol": "A", "rank": 1}, {"profile": "demo", "symbol": "B", "rank": 2}],
    )
    _write_history_csv(tmp_path, "20260102T000000Z_all.csv", [{"profile": "demo", "symbol": "A", "rank": 1}])

    points = scan_recent_run_counts(str(tmp_path), "demo", 10)

    assert [p["count"] for p in points] == [2, 1]
    assert all(p["failed"] is False for p in points)


def test_scan_recent_run_counts_flags_failed_runs_as_zero_candidates(tmp_path):
    _write_history_csv(
        tmp_path, "20260101T000000Z_all.csv",
        [{"profile": "demo", "symbol": "(RUN FAILED)", "error": "network down"}],
    )

    points = scan_recent_run_counts(str(tmp_path), "demo", 10)

    assert points == [{"timestamp": "20260101T000000Z", "count": 0, "failed": True}]


def test_scan_recent_run_counts_respects_limit(tmp_path):
    for i in range(5):
        _write_history_csv(tmp_path, f"2026010{i}T000000Z_all.csv", [{"profile": "demo", "symbol": "A", "rank": 1}])

    points = scan_recent_run_counts(str(tmp_path), "demo", 2)

    assert len(points) == 2
    assert points[-1]["timestamp"] == "20260104T000000Z"


# -----------------------------------------------------------------------------
# compute_changes
# -----------------------------------------------------------------------------


def test_compute_changes_no_prior_run():
    today_rows = [{"symbol": "AAA", "rank": 1}]
    changes = compute_changes(today_rows, None, big_move_threshold=3)
    assert changes == {"has_prior_run": False, "new": [], "dropped": [], "big_moves": []}


def test_compute_changes_detects_new_and_dropped_names():
    today_rows = [{"symbol": "AAA", "rank": 1}, {"symbol": "BBB", "rank": 2}]
    previous_df = pd.DataFrame([{"symbol": "BBB", "rank": 1}, {"symbol": "CCC", "rank": 2}])

    changes = compute_changes(today_rows, previous_df, big_move_threshold=3)

    assert changes["has_prior_run"] is True
    assert changes["new"] == ["AAA"]
    assert changes["dropped"] == ["CCC"]


def test_compute_changes_flags_big_moves_above_threshold_only():
    today_rows = [{"symbol": "AAA", "rank": 1}, {"symbol": "BBB", "rank": 2}]
    previous_df = pd.DataFrame([{"symbol": "AAA", "rank": 5}, {"symbol": "BBB", "rank": 3}])

    changes = compute_changes(today_rows, previous_df, big_move_threshold=3)

    # AAA moved 5 -> 1 (delta 4, big); BBB moved 3 -> 2 (delta 1, not big)
    assert [m["symbol"] for m in changes["big_moves"]] == ["AAA"]
    assert changes["big_moves"][0]["delta"] == 4


def test_compute_changes_ignores_the_run_failed_sentinel_in_previous_snapshot():
    today_rows = [{"symbol": "AAA", "rank": 1}]
    previous_df = pd.DataFrame([{"symbol": "(RUN FAILED)", "error": "boom"}])

    changes = compute_changes(today_rows, previous_df, big_move_threshold=3)

    assert changes["has_prior_run"] is True
    assert changes["new"] == ["AAA"]
    assert changes["dropped"] == []


# -----------------------------------------------------------------------------
# build_today_rows
# -----------------------------------------------------------------------------


def test_build_today_rows_carries_full_filter_detail():
    ranked = [_ranked("AAA", 0.75, 123.45, {"momentum": 0.8})]
    results = [_symbol_result("AAA", 123.45, {"trend": (True, "above 50-day MA"), "momentum": (True, "+5%")})]
    run_result = _FakeRunResult(profile_name="demo", ranked=ranked, results=results)

    rows = build_today_rows(run_result)

    assert rows[0]["rank"] == 1
    assert rows[0]["symbol"] == "AAA"
    assert rows[0]["components"] == {"momentum": 0.8}
    assert ("trend", True, "above 50-day MA") in rows[0]["filters"]


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------


def _sample_run_result(error=None):
    if error:
        return _FakeRunResult(profile_name="demo", error=error)
    ranked = [_ranked("AAA", 0.9, 100.0, {"momentum": 0.9})]
    results = [_symbol_result("AAA", 100.0, {"momentum": (True, "+10% <script>alert(1)</script>")})]
    return _FakeRunResult(profile_name="demo", ranked=ranked, results=results)


def test_render_document_has_full_html_structure():
    from datetime import datetime, timezone

    doc = render_document([_sample_run_result()], {}, {}, 3, datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert doc.startswith("<!DOCTYPE html>")
    assert "<html" in doc and "<head>" in doc and "<body>" in doc
    assert "<title>Screener Dashboard</title>" in doc
    assert "#1A1F2E" in doc  # navy background
    assert "#00D4AA" in doc  # teal accent
    assert "#F5B642" in doc  # amber highlight


def test_render_fragment_has_no_wrapper_tags():
    from datetime import datetime, timezone

    fragment = render_fragment([_sample_run_result()], {}, {}, 3, datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert "<!DOCTYPE" not in fragment
    assert "<html" not in fragment
    assert "<head>" not in fragment
    assert "<body>" not in fragment
    assert "<title>Screener Dashboard</title>" in fragment


def test_render_document_escapes_untrusted_text():
    from datetime import datetime, timezone

    doc = render_document([_sample_run_result()], {}, {}, 3, datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert "<script>alert(1)</script>" not in doc
    assert "&lt;script&gt;" in doc


def test_render_document_shows_failed_profile_banner():
    from datetime import datetime, timezone

    doc = render_document(
        [_sample_run_result(error="network is down")], {}, {}, 3, datetime(2026, 1, 1, tzinfo=timezone.utc)
    )
    assert "RUN FAILED: network is down" in doc


def test_render_document_marks_new_symbols():
    from datetime import datetime, timezone

    # A previous snapshot that doesn't include AAA -> AAA renders as "new".
    previous_df = pd.DataFrame([{"symbol": "ZZZ", "rank": 1}])
    doc = render_document(
        [_sample_run_result()], {"demo": previous_df}, {}, 3, datetime(2026, 1, 1, tzinfo=timezone.utc)
    )
    assert 'class="badge badge-new"' in doc
    assert ">AAA<" in doc or "AAA<" in doc


def test_render_document_shows_empty_state_as_no_qualifying_candidates():
    from datetime import datetime, timezone

    empty_run_result = _FakeRunResult(profile_name="demo", ranked=[], results=[])
    doc = render_document([empty_run_result], {}, {}, 3, datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert "No qualifying candidates today." in doc


def test_render_document_lists_data_quality_exclusions():
    from datetime import datetime, timezone

    excluded = _symbol_result("BADCO", 10.0, {"data_quality": (False, "stale data: 30d old")})
    run_result = _FakeRunResult(profile_name="demo", ranked=[], results=[excluded])

    doc = render_document([run_result], {}, {}, 3, datetime(2026, 1, 1, tzinfo=timezone.utc))

    assert "Data quality exclusions (1)" in doc
    assert "BADCO" in doc
    assert "stale data: 30d old" in doc
    # Still explicit about there being no ranked candidates, separately.
    assert "No qualifying candidates today." in doc


def test_render_document_omits_data_quality_section_when_nothing_excluded():
    doc_run = _sample_run_result()
    from datetime import datetime, timezone

    doc = render_document([doc_run], {}, {}, 3, datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert "Data quality exclusions" not in doc


def test_dashboard_footer_contains_the_disclaimer():
    from datetime import datetime, timezone

    doc = render_document([_sample_run_result()], {}, {}, 3, datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert "does not know news, context" in doc
    assert "not signals" in doc


# -----------------------------------------------------------------------------
# write_dashboard (end to end)
# -----------------------------------------------------------------------------


def test_write_dashboard_writes_a_file_with_expected_content(tmp_path):
    from datetime import datetime, timezone

    output_config = {
        "dashboard_path": str(tmp_path / "dashboard.html"),
        "history_dir": str(tmp_path / "history"),
        "dashboard_recent_runs": 10,
        "dashboard_big_move_threshold": 3,
    }

    path = write_dashboard(
        [_sample_run_result()], {"demo": None}, output_config, datetime(2026, 1, 1, tzinfo=timezone.utc)
    )

    assert path == str(tmp_path / "dashboard.html")
    content = (tmp_path / "dashboard.html").read_text()
    assert "AAA" in content
    assert "demo" in content


def test_write_dashboard_creates_missing_parent_directory(tmp_path):
    from datetime import datetime, timezone

    output_config = {"dashboard_path": str(tmp_path / "nested" / "dashboard.html"), "history_dir": str(tmp_path / "history")}

    write_dashboard([_sample_run_result()], {"demo": None}, output_config, datetime(2026, 1, 1, tzinfo=timezone.utc))

    assert (tmp_path / "nested" / "dashboard.html").exists()
