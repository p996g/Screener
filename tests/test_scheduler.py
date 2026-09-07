"""
Tests for scheduler.py: the pure "when is a run due" logic shared by both
deployment modes, plus the always-on loop's firing/dedup/error-swallowing
behavior.
"""

from datetime import date, datetime, timezone

from zoneinfo import ZoneInfo

from scheduler import compute_runs_for_date, find_due_run, is_market_day, run_loop

NY = ZoneInfo("America/New_York")


def _schedule(tolerance=20, extra_run_times=None, holidays=None, weekdays_only=True):
    return {
        "timezone": "America/New_York",
        "market_open": "09:30",
        "market_close": "16:00",
        "weekdays_only": weekdays_only,
        "holidays": holidays or [],
        "trigger_tolerance_minutes": tolerance,
        "run_times": extra_run_times
        or [{"name": "pre_open", "anchor": "market_open", "offset_minutes": -30, "profile": "all"}],
    }


# -----------------------------------------------------------------------------
# is_market_day
# -----------------------------------------------------------------------------


def test_is_market_day_true_on_weekday():
    assert is_market_day(date(2024, 1, 2), {"weekdays_only": True}) is True  # Tuesday


def test_is_market_day_false_on_weekend():
    assert is_market_day(date(2024, 1, 6), {"weekdays_only": True}) is False  # Saturday


def test_is_market_day_false_on_configured_holiday():
    config = {"weekdays_only": True, "holidays": ["2024-01-02"]}
    assert is_market_day(date(2024, 1, 2), config) is False


def test_is_market_day_ignores_weekend_when_disabled():
    assert is_market_day(date(2024, 1, 6), {"weekdays_only": False}) is True


# -----------------------------------------------------------------------------
# compute_runs_for_date
# -----------------------------------------------------------------------------


def test_compute_runs_applies_offsets_from_named_anchors():
    schedule_config = _schedule(
        extra_run_times=[
            {"name": "pre_open", "anchor": "market_open", "offset_minutes": -30, "profile": "momentum_screen"},
            {"name": "post_close", "anchor": "market_close", "offset_minutes": 30, "profile": "all"},
        ]
    )
    runs = compute_runs_for_date(schedule_config, date(2024, 1, 2))

    assert [r.name for r in runs] == ["pre_open", "post_close"]
    assert runs[0].when == datetime(2024, 1, 2, 9, 0, tzinfo=NY)
    assert runs[1].when == datetime(2024, 1, 2, 16, 30, tzinfo=NY)
    assert runs[0].profile == "momentum_screen"


def test_compute_runs_empty_on_weekend():
    schedule_config = _schedule()
    assert compute_runs_for_date(schedule_config, date(2024, 1, 6)) == []


def test_compute_runs_sorted_chronologically_regardless_of_config_order():
    schedule_config = _schedule(
        extra_run_times=[
            {"name": "late", "anchor": "market_close", "offset_minutes": 0, "profile": "all"},
            {"name": "early", "anchor": "market_open", "offset_minutes": 0, "profile": "all"},
        ]
    )
    runs = compute_runs_for_date(schedule_config, date(2024, 1, 2))
    assert [r.name for r in runs] == ["early", "late"]


# -----------------------------------------------------------------------------
# find_due_run
# -----------------------------------------------------------------------------


def test_find_due_run_at_exact_scheduled_time():
    due = find_due_run(_schedule(), datetime(2024, 1, 2, 9, 0, tzinfo=NY))
    assert due is not None and due.name == "pre_open"


def test_find_due_run_within_tolerance_after_scheduled_time():
    due = find_due_run(_schedule(tolerance=20), datetime(2024, 1, 2, 9, 15, tzinfo=NY))
    assert due is not None


def test_find_due_run_none_before_scheduled_time():
    due = find_due_run(_schedule(), datetime(2024, 1, 2, 8, 59, tzinfo=NY))
    assert due is None


def test_find_due_run_none_after_tolerance_window():
    due = find_due_run(_schedule(tolerance=20), datetime(2024, 1, 2, 9, 25, tzinfo=NY))
    assert due is None


def test_find_due_run_none_on_weekend():
    due = find_due_run(_schedule(), datetime(2024, 1, 6, 9, 0, tzinfo=NY))
    assert due is None


def test_find_due_run_none_on_configured_holiday():
    due = find_due_run(_schedule(holidays=["2024-01-02"]), datetime(2024, 1, 2, 9, 0, tzinfo=NY))
    assert due is None


# -----------------------------------------------------------------------------
# The DST-bracketing cron pair (.github/workflows/screener.yml) — proves the
# "two triggers an hour apart" trick actually fires exactly once per season.
# -----------------------------------------------------------------------------


def test_dst_bracketing_pair_fires_exactly_once_in_edt_summer():
    schedule_config = _schedule(tolerance=20)
    # 2024-07-01 is EDT (UTC-4): 13:00 UTC -> 09:00 ET (correct), 14:00 UTC -> 10:00 ET (wrong).
    correct = datetime(2024, 7, 1, 13, 0, tzinfo=timezone.utc).astimezone(NY)
    wrong = datetime(2024, 7, 1, 14, 0, tzinfo=timezone.utc).astimezone(NY)
    assert find_due_run(schedule_config, correct) is not None
    assert find_due_run(schedule_config, wrong) is None


def test_dst_bracketing_pair_fires_exactly_once_in_est_winter():
    schedule_config = _schedule(tolerance=20)
    # 2024-01-15 is EST (UTC-5): 13:00 UTC -> 08:00 ET (wrong, too early), 14:00 UTC -> 09:00 ET (correct).
    wrong = datetime(2024, 1, 15, 13, 0, tzinfo=timezone.utc).astimezone(NY)
    correct = datetime(2024, 1, 15, 14, 0, tzinfo=timezone.utc).astimezone(NY)
    assert find_due_run(schedule_config, wrong) is None
    assert find_due_run(schedule_config, correct) is not None


# -----------------------------------------------------------------------------
# run_loop (deployment mode (a))
# -----------------------------------------------------------------------------


def _schedule_due_now(name="test_run", profile="all"):
    """A schedule whose one run_time is due right now (anchored to the
    current minute), so a single loop iteration fires it deterministically
    regardless of wall-clock time."""
    now = datetime.now(NY)
    return {
        "timezone": "America/New_York",
        "market_open": f"{now.hour:02d}:{now.minute:02d}",
        "market_close": "16:00",
        "weekdays_only": False,  # avoid flakiness if the suite runs on a weekend
        "poll_interval_seconds": 0,
        "run_times": [{"name": name, "anchor": "market_open", "offset_minutes": 0, "profile": profile}],
    }


def test_run_loop_fires_a_due_run_within_one_iteration():
    fired = []
    run_loop(_schedule_due_now(), on_run=fired.append, max_iterations=1)
    assert len(fired) == 1
    assert fired[0].name == "test_run"


def test_run_loop_does_not_refire_the_same_run_twice():
    fired = []
    run_loop(_schedule_due_now(), on_run=fired.append, max_iterations=3)
    assert len(fired) == 1


def test_run_loop_swallows_exceptions_from_on_run():
    def exploding_on_run(run):
        raise RuntimeError("boom")

    # Must not raise.
    run_loop(_schedule_due_now(), on_run=exploding_on_run, max_iterations=1)
