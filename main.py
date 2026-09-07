"""
Entry point: load config, run the screen, report the results.

Usage:
    python main.py [mode] [path/to/settings.yaml]

`mode` is one of:
  - a profile name defined in settings.yaml's `profiles` section
  - "all" (the default when no mode is given) — runs every defined
    profile and produces one combined report grouped by profile
  - "schedule" — deployment mode (a): blocks forever, running the
    profiles configured under settings.yaml's `schedule.run_times` at
    their scheduled times (see scheduler.py)
  - "scheduled-run" — deployment mode (b): a single check-and-maybe-run,
    used by the GitHub Actions workflow (.github/workflows/screener.yml)
    on every cron trigger; no-ops cleanly if nothing is actually due right
    now (see scheduler.find_due_run)

If no config path is given, config/settings.yaml is used.

Every run (whichever mode triggered it) writes a timestamped snapshot of
its report to `output.history_dir`, regenerates output/dashboard.html
(see output/dashboard.py), evaluates and delivers any alert rules (see
alerts.py) as its last step, and returns/exits non-zero if any profile's
run failed outright (see ProfileRunResult) — a fetch failure for one
symbol is handled further down the stack (data/fetcher.py) and never
reaches this level at all.
"""

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import yaml

from alerts import dispatch_alerts, evaluate_rules, resolve_dashboard_link
from data.fetcher import build_fetcher, resolve_evaluation_date
from data.universe import load_universe
from engine import ScreenerEngine, SymbolResult
from filters.library import build_enabled_filters
from output.dashboard import load_latest_previous_snapshot, write_dashboard
from output.report import save_history_snapshot, write_combined_report, write_reports
from profiles import list_profile_names, resolve_profile_config
from ranking import RankedSymbol, rank_survivors
from scheduler import find_due_run, run_loop

DEFAULT_CONFIG_PATH = "config/settings.yaml"
ALL_PROFILES = "all"
SCHEDULE_LOOP_MODE = "schedule"
SCHEDULED_RUN_MODE = "scheduled-run"

logger = logging.getLogger(__name__)


@dataclass
class ProfileRunResult:
    profile_name: str
    profile_config: dict
    ranked: list[RankedSymbol]
    results: list[SymbolResult]
    error: str | None


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def configure_logging(config: dict) -> None:
    logging_config = config.get("logging", {}) or {}
    level = getattr(logging, str(logging_config.get("level", "INFO")).upper(), logging.INFO)
    handlers = [logging.StreamHandler()]

    log_file = logging_config.get("file")
    if log_file:
        directory = os.path.dirname(log_file)
        if directory:
            os.makedirs(directory, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def run_profile(config: dict, profile_name: str) -> ProfileRunResult:
    """Resolve one named profile and run the full pipeline (universe ->
    fetch -> filter -> rank) for it.

    Config problems (an unknown profile, a bad filter name, a malformed
    universe) raise immediately — those are bugs in settings.yaml that
    should fail loudly and fast. Only the data-fetching/engine phase,
    where a transient network or provider failure is a normal and
    expected possibility in an unattended scheduled run, is caught: it's
    logged clearly and turned into a ProfileRunResult.error rather than
    crashing the whole process.
    """
    profile_config = resolve_profile_config(config, profile_name)
    universe = load_universe(profile_config["universe"])
    fetcher = build_fetcher(profile_config["data"])
    evaluation_date = resolve_evaluation_date(profile_config["data"])
    filters = build_enabled_filters(profile_config["filters"])
    engine = ScreenerEngine(
        filters=filters,
        evaluation_date=evaluation_date,
        data_quality_config=profile_config.get("data_quality"),
    )

    logger.info(
        "[%s] Screening %d symbols as of %s through %d filter(s)...",
        profile_name, len(universe), evaluation_date, len(filters),
    )

    try:
        results = engine.run(universe, fetcher)
    except Exception as exc:
        logger.error("[%s] run failed: %s", profile_name, exc, exc_info=True)
        return ProfileRunResult(profile_name, profile_config, ranked=[], results=[], error=str(exc))

    survivors = [r for r in results if r.passed]
    logger.info("[%s] %d/%d symbols passed all filters.", profile_name, len(survivors), len(results))

    ranked = rank_survivors(results, profile_config["ranking"])
    return ProfileRunResult(profile_name, profile_config, ranked=ranked, results=results, error=None)


def run(config: dict, mode: str = ALL_PROFILES, run_label: str | None = None) -> bool:
    """Run `mode` (a profile name, or "all") once: write its report(s), a
    timestamped history snapshot, and regenerate output/dashboard.html as
    the last step. Returns True if anything failed.

    `run_label` overrides the label used for the history filename (used by
    the scheduler to tag a snapshot with the scheduled run's own name,
    e.g. "pre_open", instead of just the profile name).
    """
    when = datetime.now(timezone.utc)
    label = run_label or mode

    if mode == ALL_PROFILES:
        profile_names = list_profile_names(config)
        if not profile_names:
            raise ValueError(
                "profile 'all' requested but no profiles are defined in "
                "settings.yaml's `profiles` section"
            )
        run_results = [run_profile(config, name) for name in profile_names]
    else:
        run_results = [run_profile(config, mode)]

    # Look up each profile's most recent *prior* snapshot before writing
    # anything new below, so "previous run" never means "this run".
    history_dir = config["output"].get("history_dir", "output/history")
    previous_by_profile = {
        r.profile_name: load_latest_previous_snapshot(history_dir, r.profile_name)
        for r in run_results
    }

    if mode == ALL_PROFILES:
        ranked_by_profile = {r.profile_name: r.ranked for r in run_results}
        results_by_profile = {r.profile_name: r.results for r in run_results}
        errors_by_profile = {r.profile_name: r.error for r in run_results if r.error is not None}
        report_df = write_combined_report(
            ranked_by_profile, results_by_profile, config["output"], errors_by_profile
        )
    else:
        result = run_results[0]
        report_df = write_reports(
            result.ranked, result.results, result.profile_config["output"], error=result.error
        )

    had_failure = any(r.error is not None for r in run_results)

    # Every history snapshot carries a "profile" column, even for a single
    # -profile run, so the dashboard's history/diff logic never needs to
    # special-case which kind of report produced a given file.
    history_df = report_df.copy()
    if "profile" not in history_df.columns:
        history_df.insert(0, "profile", mode)
    save_history_snapshot(history_df, config["output"], run_label=label, when=when)

    write_dashboard(run_results, previous_by_profile, config["output"], when)

    alerts_config = config.get("alerts") or {}
    if alerts_config.get("enabled"):
        triggered = evaluate_rules(run_results, previous_by_profile, alerts_config.get("rules") or {})
        dashboard_link = resolve_dashboard_link(alerts_config, config["output"])
        dispatch_alerts(triggered, alerts_config, dashboard_link)

    return had_failure


def run_scheduler_loop(config: dict) -> None:
    def on_run(scheduled_run) -> None:
        run(config, scheduled_run.profile, run_label=scheduled_run.name)

    run_loop(config["schedule"], on_run)


def run_scheduled_check(config: dict) -> bool:
    """Deployment mode (b): fire at most one due run, if any, and return
    whether it failed. See scheduler.find_due_run for why this is safe to
    call from a cron trigger that fires more often than the real cadence."""
    now = datetime.now(ZoneInfo(config["schedule"]["timezone"]))
    due_run = find_due_run(config["schedule"], now)
    if due_run is None:
        logger.info("No scheduled run is due right now (checked at %s) — skipping.", now)
        return False

    return run(config, due_run.profile, run_label=due_run.name)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else ALL_PROFILES
    config_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CONFIG_PATH
    config = load_config(config_path)
    configure_logging(config)

    if mode == SCHEDULE_LOOP_MODE:
        run_scheduler_loop(config)
        return

    if mode == SCHEDULED_RUN_MODE:
        had_failure = run_scheduled_check(config)
    else:
        had_failure = run(config, mode)

    if had_failure:
        sys.exit(1)


if __name__ == "__main__":
    main()
