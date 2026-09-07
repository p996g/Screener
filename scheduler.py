"""
Computes the screener's scheduled run times and (for deployment mode (a))
waits for them.

Two independent things live here, both driven by settings.yaml's
`schedule` section so retiming or adding a run is a config-only change:

  - Pure functions (`is_market_day`, `compute_runs_for_date`, `find_due_run`)
    that answer "which run(s), if any, are due" for a given date/instant.
    These are the single source of truth both deployment modes below build
    on, so they can never drift apart.
  - `run_loop()`: a small, dependency-free scheduler for deployment mode
    (a) — an always-on process that sleeps, wakes up periodically, and
    fires any run whose scheduled time has just passed.

Deployment mode (b), GitHub Actions, doesn't use `run_loop` at all — cron
already IS its scheduler. It instead calls `find_due_run()` once per
workflow trigger (see main.py's "scheduled-run" mode) to decide whether
*this* invocation actually corresponds to a real scheduled time, since
GitHub Actions cron is UTC-only and can't express "30 minutes before the
NYSE open" directly (see the comment on `trigger_tolerance_minutes` in
settings.yaml and .github/workflows/screener.yml for how that's handled).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScheduledRun:
    name: str
    profile: str
    when: datetime  # tz-aware, in schedule.timezone


def is_market_day(day: date, schedule_config: dict) -> bool:
    """True if `day` is a day the screener should run on at all.

    Weekends are always excluded when `weekdays_only` is true (the
    default). Exchange holidays aren't inferred from any calendar data —
    list the ones you care about in `schedule.holidays` (YYYY-MM-DD); no
    holiday-calendar dependency is wired into this scaffold.
    """
    if schedule_config.get("weekdays_only", True) and day.weekday() >= 5:
        return False
    holidays = set(schedule_config.get("holidays") or [])
    return day.isoformat() not in holidays


def compute_runs_for_date(schedule_config: dict, day: date) -> list[ScheduledRun]:
    """Every scheduled run that falls on `day`, in chronological order.
    Empty if `day` isn't a market day (see `is_market_day`)."""
    if not is_market_day(day, schedule_config):
        return []

    tz = ZoneInfo(schedule_config["timezone"])
    anchors = {
        "market_open": _time_on(day, schedule_config["market_open"], tz),
        "market_close": _time_on(day, schedule_config["market_close"], tz),
    }

    runs = [
        ScheduledRun(
            name=entry["name"],
            profile=entry.get("profile", "all"),
            when=anchors[entry["anchor"]] + timedelta(minutes=entry.get("offset_minutes", 0)),
        )
        for entry in schedule_config.get("run_times", [])
    ]
    runs.sort(key=lambda run: run.when)
    return runs


def find_due_run(schedule_config: dict, now: datetime) -> ScheduledRun | None:
    """The single scheduled run (if any) whose time has arrived, within
    `schedule.trigger_tolerance_minutes` of `now`. Used by the one-shot
    "scheduled-run" CLI mode (GitHub Actions): a run counts as due once
    `now` reaches its scheduled time, and stops counting as due after the
    tolerance window passes, so a workflow trigger that fires well before
    or well after the real target time correctly no-ops instead of firing
    at the wrong moment or firing twice.
    """
    tolerance = timedelta(minutes=schedule_config.get("trigger_tolerance_minutes", 20))
    for run in compute_runs_for_date(schedule_config, now.date()):
        if run.when <= now <= run.when + tolerance:
            return run
    return None


def _time_on(day: date, hhmm: str, tz: ZoneInfo) -> datetime:
    hour, minute = (int(part) for part in hhmm.split(":"))
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=tz)


def run_loop(schedule_config: dict, on_run, max_iterations: int | None = None) -> None:
    """Block (or, with `max_iterations` set, iterate a bounded number of
    times — used by tests) calling `on_run(scheduled_run)` once for each
    scheduled run at (approximately) its scheduled time.

    A hand-rolled poll loop rather than a scheduling library: it wakes up
    every `poll_interval_seconds`, fires anything whose time has arrived
    since the last check, and never fires the same (date, run name) twice
    even if the loop was asleep past the exact moment.

    An exception raised by `on_run` is logged and swallowed so one bad run
    never brings down the whole always-on process — the loop just waits
    for the next scheduled time.
    """
    poll_interval = schedule_config.get("poll_interval_seconds", 30)
    tz = ZoneInfo(schedule_config["timezone"])
    already_fired: set[tuple[date, str]] = set()

    logger.info("Scheduler loop started (poll every %ss).", poll_interval)
    iterations = 0
    while max_iterations is None or iterations < max_iterations:
        now = datetime.now(tz)
        already_fired = {key for key in already_fired if key[0] == now.date()}

        for run in compute_runs_for_date(schedule_config, now.date()):
            key = (now.date(), run.name)
            if key in already_fired or run.when > now:
                continue
            already_fired.add(key)
            logger.info(
                "Firing scheduled run '%s' (profile=%s), due at %s.",
                run.name, run.profile, run.when,
            )
            try:
                on_run(run)
            except Exception:
                logger.exception("Scheduled run '%s' raised unexpectedly.", run.name)

        iterations += 1
        if max_iterations is None or iterations < max_iterations:
            time.sleep(poll_interval)
