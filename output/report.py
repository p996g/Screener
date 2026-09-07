"""
Produces the ranked watchlist output.

Two independent output formats are supported, both driven by
`output.formats` in settings.yaml:
  - "console": a plain-text table printed to stdout.
  - "csv":     a full dump (every ranked symbol, every score component)
               written to `output.csv_path`.

Both formats can optionally include each symbol's per-filter pass/fail
reasons (`output.show_filter_reasons`), which is what keeps the screener's
output explainable rather than a black-box list of tickers.

`write_reports` produces one profile's report. `write_combined_report`
does the same for every profile in a single run (main.py's "all" profile),
laying each profile's own ranking out separately (rank restarts at 1 per
profile) and tagging every row with a `profile` column so a single CSV
still lets you filter/pivot by profile afterward.

`save_history_snapshot` additionally writes a timestamped copy of any
report under `output.history_dir`, building a day-over-day archive of what
the screen said — called once per run by main.py regardless of which of
these two functions produced the report.

Failure handling: both report functions accept an `error` (or, for the
combined report, `errors_by_profile`) argument. When a profile's run
failed (see main.py's ProfileRunResult), its report is a single explicit
"RUN FAILED: <reason>" row/banner instead of a silently-empty watchlist —
the whole point being that a failure is never indistinguishable from "no
symbols passed today".

Small list is fine: nothing here ever pads a short or empty result to look
fuller. A profile with zero survivors renders as an explicit "No
qualifying candidates today" line, not a blank table — an empty screen is
a legitimate, fully-supported outcome, not a degraded one.

Data-quality exclusions (see data/quality.py) are symbols the engine
excluded before any filter ran, on the same point-in-time data — the
console output calls these out by name and reason so they're never
confused with "just didn't pass".
"""

import os
from datetime import datetime

import pandas as pd

from engine import SymbolResult
from ranking import RankedSymbol


def build_report_dataframe(
    ranked: list[RankedSymbol],
    results_by_symbol: dict[str, SymbolResult],
    show_filter_reasons: bool,
    error: str | None = None,
) -> pd.DataFrame:
    if error is not None:
        return pd.DataFrame([{"symbol": "(RUN FAILED)", "error": error}])

    rows = []
    for rank_position, ranked_symbol in enumerate(ranked, start=1):
        result = results_by_symbol[ranked_symbol.symbol]
        row = {
            "rank": rank_position,
            "symbol": ranked_symbol.symbol,
            "score": round(ranked_symbol.score, 4),
            "close": round(ranked_symbol.close, 2),
            "passed": result.passed,
            **{
                f"component_{name}": round(value, 4)
                for name, value in ranked_symbol.component_scores.items()
            },
        }
        if show_filter_reasons:
            row["filter_reasons"] = "; ".join(
                f"{name}: {fr.reason}"
                for name, fr in result.filter_results.items()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _data_quality_exclusion_lines(results: list[SymbolResult]) -> list[str]:
    """One line per symbol the engine excluded via the data-quality check
    (see data/quality.py) before any filter ran — kept separate from
    filter pass/fail so a data problem is never mistaken for "just didn't
    qualify"."""
    return [
        f"  {result.symbol}: {result.filter_results['data_quality'].reason}"
        for result in results
        if "data_quality" in result.filter_results
    ]


def _print_data_quality_exclusions(results: list[SymbolResult]) -> None:
    lines = _data_quality_exclusion_lines(results)
    if not lines:
        return
    print(f"Data quality exclusions ({len(lines)}):")
    for line in lines:
        print(line)


def write_console_report(
    report_df: pd.DataFrame, console_columns: list[str]
) -> None:
    if "error" in report_df.columns and not report_df.empty:
        for message in report_df["error"]:
            print(f"RUN FAILED: {message}")
        return

    if report_df.empty:
        print("No qualifying candidates today.")
        return

    available_columns = [c for c in console_columns if c in report_df.columns]
    print(report_df[available_columns].to_string(index=False))


def write_csv_report(report_df: pd.DataFrame, csv_path: str) -> None:
    directory = os.path.dirname(csv_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    report_df.to_csv(csv_path, index=False)


def write_reports(
    ranked: list[RankedSymbol],
    results: list[SymbolResult],
    output_config: dict,
    error: str | None = None,
) -> pd.DataFrame:
    """Build the report and emit it in every format listed in
    `output.formats`. Returns the report DataFrame either way, so callers
    (and tests) can inspect it without re-parsing a file.

    If `error` is given, the run is reported as failed (see module
    docstring) instead of rendering `ranked`/`results` at all.
    """
    results_by_symbol = {r.symbol: r for r in results}
    report_df = build_report_dataframe(
        ranked, results_by_symbol, output_config.get("show_filter_reasons", True), error=error
    )

    formats = output_config.get("formats", [])
    if "console" in formats:
        write_console_report(report_df, output_config.get("console_columns", []))
        if error is None:
            _print_data_quality_exclusions(results)
    if "csv" in formats:
        write_csv_report(report_df, output_config["csv_path"])

    return report_df


def write_combined_report(
    ranked_by_profile: dict[str, list[RankedSymbol]],
    results_by_profile: dict[str, list[SymbolResult]],
    output_config: dict,
    errors_by_profile: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build and emit one combined report spanning every profile in
    `ranked_by_profile` (in the order given), each tagged with a `profile`
    column. Used by main.py's "all" profile to run every profile in one
    pass and produce a single grouped report. A profile named in
    `errors_by_profile` is reported as failed for that profile only —
    the others still report normally.
    """
    errors_by_profile = errors_by_profile or {}
    show_filter_reasons = output_config.get("show_filter_reasons", True)

    per_profile_frames = []
    for profile_name, ranked in ranked_by_profile.items():
        results_by_symbol = {r.symbol: r for r in results_by_profile[profile_name]}
        profile_df = build_report_dataframe(
            ranked, results_by_symbol, show_filter_reasons,
            error=errors_by_profile.get(profile_name),
        )
        profile_df.insert(0, "profile", profile_name)
        per_profile_frames.append(profile_df)

    combined_df = (
        pd.concat(per_profile_frames, ignore_index=True)
        if per_profile_frames
        else pd.DataFrame(columns=["profile"])
    )

    formats = output_config.get("formats", [])
    if "console" in formats:
        _write_combined_console_report(
            combined_df, ranked_by_profile.keys(), results_by_profile,
            errors_by_profile, output_config.get("console_columns", []),
        )
    if "csv" in formats:
        write_csv_report(combined_df, output_config["csv_path"])

    return combined_df


def _write_combined_console_report(
    combined_df: pd.DataFrame,
    profile_names,
    results_by_profile: dict[str, list[SymbolResult]],
    errors_by_profile: dict[str, str],
    console_columns: list[str],
) -> None:
    available_columns = [c for c in console_columns if c in combined_df.columns]
    for profile_name in profile_names:
        print(f"=== {profile_name} ===")
        profile_rows = combined_df[combined_df["profile"] == profile_name]

        if profile_name in errors_by_profile:
            print(f"  RUN FAILED: {errors_by_profile[profile_name]}")
        else:
            if profile_rows.empty:
                print("  No qualifying candidates today.")
            else:
                print(profile_rows[available_columns].to_string(index=False))

            exclusion_lines = _data_quality_exclusion_lines(results_by_profile[profile_name])
            if exclusion_lines:
                print(f"  Data quality exclusions ({len(exclusion_lines)}):")
                for line in exclusion_lines:
                    print(f"  {line}")
        print()


def save_history_snapshot(
    report_df: pd.DataFrame, output_config: dict, run_label: str, when: datetime
) -> str | None:
    """Save a timestamped copy of `report_df` under `output.history_dir`,
    building a day-over-day archive of what the screen said. `run_label`
    identifies the run in the filename (a profile name, "all", or a
    scheduled run's own name). No-ops (returns None) if
    `output.save_history` is false.
    """
    if not output_config.get("save_history", True):
        return None

    history_dir = output_config.get("history_dir", "output/history")
    timestamp = when.strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(history_dir, f"{timestamp}_{run_label}.csv")
    write_csv_report(report_df, path)
    return path
