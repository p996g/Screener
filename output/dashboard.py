"""
Renders the self-contained HTML dashboard (output/dashboard.html) after
every run.

Design: the page is fully rendered by Python, once, per run — there is no
client-side templating, no JSON blob, no fetch(), and no JavaScript at
all. Expand/collapse for each symbol's filter-by-filter detail uses plain
HTML5 <details>/<summary>, which every browser supports natively. This is
what "dependency-light" and "no server required, openable in any browser"
mean here: double-clicking the file works, forever, with nothing else
running.

Three data sources feed each profile's section:
  - `run_results` (today): the in-memory ProfileRunResult objects from
    this run — full filter-by-filter detail is only available in memory,
    never round-tripped through a CSV, so this must run before those
    objects go out of scope.
  - `previous_by_profile`: the most recent *prior* history snapshot for
    each profile (looked up by main.py *before* today's snapshot is
    written, so "previous" never means "today") — used for the "what
    changed" section.
  - The history directory itself, rescanned fresh (including today's just
    -written snapshot) for the "recent runs" trend view.
"""

from __future__ import annotations

import html
import os
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from main import ProfileRunResult

RUN_FAILED_SENTINEL = "(RUN FAILED)"


# -----------------------------------------------------------------------------
# History I/O — reading, not writing (writing is main.py/output.report's job)
# -----------------------------------------------------------------------------


def load_latest_previous_snapshot(history_dir: str, profile_name: str) -> pd.DataFrame | None:
    """The most recent existing history file (by filename, which sorts
    chronologically) that has rows for `profile_name`. Every history file
    is guaranteed to carry a "profile" column (main.py ensures this even
    for single-profile runs), so this works uniformly regardless of
    whether a given file was a combined "all" snapshot or a single
    profile's own. Returns None if no such snapshot exists yet — this is
    the normal, expected case the very first time a profile is ever run.
    """
    candidate = None
    for filename in _sorted_history_files(history_dir):
        df = _read_history_csv(os.path.join(history_dir, filename))
        if df is None or "profile" not in df.columns:
            continue
        rows = df[df["profile"] == profile_name]
        if not rows.empty:
            candidate = rows  # keep overwriting; last one wins (files sort oldest -> newest)
    return candidate


def scan_recent_run_counts(history_dir: str, profile_name: str, limit: int) -> list[dict]:
    """Up to `limit` most recent runs that included `profile_name`, oldest
    first: [{"timestamp": "20260906T133000Z", "count": int, "failed": bool}].
    A failed run counts as 0 candidates but is flagged so the dashboard can
    render it distinctly rather than looking like a quiet day.
    """
    points = []
    for filename in _sorted_history_files(history_dir):
        df = _read_history_csv(os.path.join(history_dir, filename))
        if df is None or "profile" not in df.columns:
            continue
        rows = df[df["profile"] == profile_name]
        if rows.empty:
            continue
        failed = "error" in rows.columns and rows["error"].notna().any()
        count = 0 if failed else len(rows)
        points.append({"timestamp": filename.split("_", 1)[0], "count": count, "failed": failed})
    return points[-limit:]


def _sorted_history_files(history_dir: str) -> list[str]:
    if not os.path.isdir(history_dir):
        return []
    return sorted(f for f in os.listdir(history_dir) if f.endswith(".csv"))


def _read_history_csv(path: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Data shaping
# -----------------------------------------------------------------------------


def build_today_rows(run_result: "ProfileRunResult") -> list[dict]:
    """One dict per surviving, ranked symbol, carrying everything the
    dashboard needs: rank, score, close, the normalized component
    breakdown, and the full filter-by-filter pass/fail/reason list (only
    available here, from the live objects — never reconstructable from a
    CSV alone)."""
    results_by_symbol = {r.symbol: r for r in run_result.results}
    rows = []
    for position, ranked_symbol in enumerate(run_result.ranked, start=1):
        result = results_by_symbol[ranked_symbol.symbol]
        rows.append(
            {
                "rank": position,
                "symbol": ranked_symbol.symbol,
                "score": ranked_symbol.score,
                "close": ranked_symbol.close,
                "components": dict(ranked_symbol.component_scores),
                "filters": [
                    (name, fr.passed, fr.reason) for name, fr in result.filter_results.items()
                ],
            }
        )
    return rows


def compute_changes(
    today_rows: list[dict], previous_df: pd.DataFrame | None, big_move_threshold: int
) -> dict:
    """New names, dropped names, and rank moves of at least
    `big_move_threshold` positions, versus the previous snapshot."""
    today_rank_by_symbol = {row["symbol"]: row["rank"] for row in today_rows}
    today_symbols = set(today_rank_by_symbol)

    if previous_df is None:
        return {"has_prior_run": False, "new": [], "dropped": [], "big_moves": []}

    usable = previous_df
    if "symbol" in usable.columns:
        usable = usable[usable["symbol"] != RUN_FAILED_SENTINEL]
    if "symbol" not in usable.columns or "rank" not in usable.columns:
        return {"has_prior_run": True, "new": sorted(today_symbols), "dropped": [], "big_moves": []}

    previous_rank_by_symbol = dict(zip(usable["symbol"], usable["rank"]))
    previous_symbols = set(previous_rank_by_symbol)

    new_names = sorted(today_symbols - previous_symbols)
    dropped_names = sorted(previous_symbols - today_symbols)

    big_moves = []
    for symbol in today_symbols & previous_symbols:
        previous_rank = int(previous_rank_by_symbol[symbol])
        current_rank = int(today_rank_by_symbol[symbol])
        delta = previous_rank - current_rank  # positive == moved up (toward #1)
        if abs(delta) >= big_move_threshold:
            big_moves.append(
                {"symbol": symbol, "previous_rank": previous_rank, "current_rank": current_rank, "delta": delta}
            )
    big_moves.sort(key=lambda move: -abs(move["delta"]))

    return {"has_prior_run": True, "new": new_names, "dropped": dropped_names, "big_moves": big_moves}


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------


def _e(value) -> str:
    return html.escape(str(value), quote=True)


def _render_filter_item(name: str, passed: bool, reason: str) -> str:
    icon, css_class = ("&#10003;", "filter-pass") if passed else ("&#10007;", "filter-fail")
    return f'<li class="{css_class}"><span class="filter-icon">{icon}</span> <b>{_e(name)}</b> — {_e(reason)}</li>'


def _render_component_chip(name: str, value: float) -> str:
    alpha = round(0.15 + 0.6 * max(0.0, min(1.0, value)), 3)
    return (
        f'<span class="chip" style="background: rgba(0,212,170,{alpha})">'
        f"{_e(name)} {value:.2f}</span>"
    )


def _render_symbol_row(row: dict, is_new: bool) -> str:
    new_badge = ' <span class="badge badge-new">NEW</span>' if is_new else ""
    score_pct = max(0.0, min(1.0, row["score"])) * 100
    chips = "".join(_render_component_chip(name, value) for name, value in row["components"].items())
    filters_html = "".join(
        _render_filter_item(name, passed, reason) for name, passed, reason in row["filters"]
    )
    return f"""
    <details class="row">
      <summary>
        <span class="col-rank">#{row['rank']}</span>
        <span class="col-symbol">{_e(row['symbol'])}{new_badge}</span>
        <span class="col-score">
          <span class="bar-track"><span class="bar-fill" style="width:{score_pct:.1f}%"></span></span>
          <span class="score-value">{row['score']:.3f}</span>
        </span>
        <span class="col-close">${row['close']:.2f}</span>
      </summary>
      <div class="detail">
        <div class="chips">{chips or '<span class="muted">No score components for this profile</span>'}</div>
        <ul class="filters">{filters_html}</ul>
      </div>
    </details>"""


def _render_changes_block(changes: dict) -> str:
    if not changes["has_prior_run"]:
        return '<p class="muted">No prior run yet to compare against.</p>'

    parts = []

    if changes["new"]:
        chips = "".join(f'<span class="badge badge-new">{_e(s)}</span>' for s in changes["new"])
        parts.append(f'<div class="change-row"><span class="change-label">New ({len(changes["new"])})</span>{chips}</div>')

    if changes["dropped"]:
        chips = "".join(f'<span class="badge badge-dropped">{_e(s)}</span>' for s in changes["dropped"])
        parts.append(f'<div class="change-row"><span class="change-label">Dropped ({len(changes["dropped"])})</span>{chips}</div>')

    if changes["big_moves"]:
        items = []
        for move in changes["big_moves"]:
            arrow, css = ("&#9650;", "move-up") if move["delta"] > 0 else ("&#9660;", "move-down")
            items.append(
                f'<span class="badge badge-move {css}">{_e(move["symbol"])} '
                f'{move["previous_rank"]}&rarr;{move["current_rank"]} {arrow}{abs(move["delta"])}</span>'
            )
        parts.append(f'<div class="change-row"><span class="change-label">Big moves</span>{"".join(items)}</div>')

    if not parts:
        return '<p class="muted">No changes since the last run.</p>'

    return "".join(parts)


def _render_history_bars(points: list[dict]) -> str:
    if not points:
        return '<p class="muted">No run history yet.</p>'

    max_count = max((p["count"] for p in points), default=0) or 1
    bars = []
    for point in points:
        height_pct = 8 if point["failed"] else max(6, round(point["count"] / max_count * 100))
        css_class = "history-bar failed" if point["failed"] else "history-bar"
        label = _format_run_timestamp(point["timestamp"])
        title = f"{label}: FAILED" if point["failed"] else f"{label}: {point['count']} candidate(s)"
        bars.append(f'<div class="{css_class}" style="height:{height_pct}%" title="{_e(title)}"></div>')
    return f'<div class="history-bars">{"".join(bars)}</div>'


def _format_run_timestamp(raw: str) -> str:
    try:
        return datetime.strptime(raw, "%Y%m%dT%H%M%SZ").strftime("%b %d, %H:%M UTC")
    except ValueError:
        return raw


def _render_data_quality_exclusions(run_result: "ProfileRunResult") -> str:
    """Symbols the engine excluded via the data-quality check (see
    data/quality.py) before any filter ran. Rendered separately from the
    results table — a data problem is a different kind of story than "this
    symbol just didn't qualify", and small-list-is-fine means we'd rather
    say so explicitly than leave it unexplained."""
    excluded = [
        (result.symbol, result.filter_results["data_quality"].reason)
        for result in run_result.results
        if "data_quality" in result.filter_results
    ]
    if not excluded:
        return ""
    items = "".join(f"<li><b>{_e(symbol)}</b> — {_e(reason)}</li>" for symbol, reason in excluded)
    return f"""
      <div class="section-label">Data quality exclusions ({len(excluded)})</div>
      <div class="quality-box"><ul class="quality-list">{items}</ul></div>"""


def _render_profile_section(
    run_result: "ProfileRunResult",
    previous_df: pd.DataFrame | None,
    history_points: list[dict],
    big_move_threshold: int,
) -> str:
    profile_name = run_result.profile_name

    if run_result.error:
        return f"""
    <section class="card" id="{_e(profile_name)}">
      <h2>{_e(profile_name)}</h2>
      <div class="status-failed">RUN FAILED: {_e(run_result.error)}</div>
      <div class="section-label">Recent history (last {len(history_points)} runs)</div>
      {_render_history_bars(history_points)}
    </section>"""

    today_rows = build_today_rows(run_result)
    changes = compute_changes(today_rows, previous_df, big_move_threshold)
    new_symbols = set(changes["new"])

    # Small list is fine: an empty result is never padded, and renders as
    # an explicit, unambiguous message rather than a blank table.
    if today_rows:
        rows_html = "".join(_render_symbol_row(row, row["symbol"] in new_symbols) for row in today_rows)
    else:
        rows_html = '<p class="muted">No qualifying candidates today.</p>'

    return f"""
    <section class="card" id="{_e(profile_name)}">
      <h2>{_e(profile_name)}</h2>

      <div class="section-label">Today's results ({len(today_rows)})</div>
      <div class="rows">{rows_html}</div>
      {_render_data_quality_exclusions(run_result)}

      <div class="section-label">Changes since last run</div>
      <div class="changes">{_render_changes_block(changes)}</div>

      <div class="section-label">Recent history (last {len(history_points)} runs)</div>
      {_render_history_bars(history_points)}
    </section>"""


_STYLE = """
<title>Screener Dashboard</title>
<style>
  :root {
    --bg: #1A1F2E;
    --panel: #232A3D;
    --panel-border: #2E3650;
    --teal: #00D4AA;
    --amber: #F5B642;
    --red: #EF6461;
    --text: #E6E9F0;
    --text-muted: #8B93A7;
    --track: #161B28;
  }
  * { box-sizing: border-box; }
  html, body {
    background: var(--bg);
    color: var(--text);
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }
  body { padding: 32px clamp(16px, 5vw, 48px); }
  h1 { color: var(--teal); margin: 0 0 4px 0; font-size: 26px; }
  .meta { color: var(--text-muted); font-size: 13px; margin-bottom: 8px; }
  .banner-failed {
    background: rgba(239,72,77,0.12);
    border: 1px solid var(--red);
    color: var(--red);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 20px;
    font-size: 14px;
  }
  .profile-nav { display: flex; gap: 8px; flex-wrap: wrap; margin: 16px 0 28px 0; }
  .profile-nav a {
    color: var(--teal);
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 999px;
    padding: 6px 14px;
    font-size: 13px;
    text-decoration: none;
  }
  .profile-nav a:hover { border-color: var(--teal); }
  .card {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 24px;
  }
  .card h2 { color: var(--teal); margin: 0 0 16px 0; font-size: 19px; }
  .section-label {
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 11px;
    margin: 20px 0 10px 0;
  }
  .section-label:first-of-type { margin-top: 0; }
  .status-failed {
    background: rgba(239,72,77,0.12);
    border: 1px solid var(--red);
    color: var(--red);
    border-radius: 8px;
    padding: 12px 14px;
    font-size: 14px;
  }
  .muted { color: var(--text-muted); font-size: 13px; margin: 0; }

  .quality-box {
    background: var(--track);
    border: 1px solid var(--panel-border);
    border-radius: 8px;
    padding: 10px 14px;
  }
  ul.quality-list { margin: 0; padding: 0 0 0 18px; font-size: 13px; color: var(--text-muted); }
  ul.quality-list li { line-height: 1.5; }
  ul.quality-list b { color: var(--text); }

  .rows { display: flex; flex-direction: column; gap: 8px; overflow-x: auto; }
  details.row {
    background: var(--track);
    border: 1px solid var(--panel-border);
    border-radius: 10px;
    padding: 4px 14px;
  }
  details.row[open] { border-color: var(--teal); }
  details.row summary {
    list-style: none;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 10px 0;
    min-width: 560px;
  }
  details.row summary::-webkit-details-marker { display: none; }
  details.row summary::before {
    content: "\\25B8";
    color: var(--text-muted);
    width: 12px;
    transition: transform 0.15s ease;
    flex: none;
  }
  details.row[open] summary::before { transform: rotate(90deg); }
  .col-rank { color: var(--text-muted); width: 34px; flex: none; font-variant-numeric: tabular-nums; }
  .col-symbol { width: 130px; flex: none; font-weight: 600; }
  .col-score { display: flex; align-items: center; gap: 10px; flex: 1 1 auto; }
  .col-close { color: var(--text-muted); width: 90px; flex: none; text-align: right; font-variant-numeric: tabular-nums; }
  .bar-track { background: #0E121C; border-radius: 4px; height: 6px; width: 120px; overflow: hidden; flex: none; }
  .bar-fill { background: var(--teal); height: 100%; display: block; }
  .score-value { font-variant-numeric: tabular-nums; color: var(--text-muted); font-size: 13px; width: 48px; }

  .detail { padding: 4px 0 14px 26px; }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
  .chip { border-radius: 999px; padding: 3px 10px; font-size: 12px; color: var(--text); }
  ul.filters { margin: 0; padding: 0; list-style: none; display: flex; flex-direction: column; gap: 4px; }
  ul.filters li { font-size: 13px; line-height: 1.4; }
  .filter-icon { display: inline-block; width: 14px; }
  .filter-pass { color: var(--text); }
  .filter-pass .filter-icon { color: var(--teal); }
  .filter-fail { color: var(--text-muted); }
  .filter-fail .filter-icon { color: var(--red); }

  .changes { display: flex; flex-direction: column; gap: 10px; }
  .change-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .change-label { color: var(--text-muted); font-size: 12px; width: 110px; flex: none; }
  .badge { border-radius: 999px; padding: 3px 10px; font-size: 12px; font-weight: 600; }
  .badge-new { background: var(--amber); color: #1A1F2E; }
  .badge-dropped { background: rgba(239,72,77,0.15); color: var(--red); border: 1px solid rgba(239,72,77,0.4); }
  .badge-move { background: rgba(0,212,170,0.12); color: var(--teal); border: 1px solid rgba(0,212,170,0.35); }
  .badge-move.move-down { background: rgba(239,72,77,0.12); color: var(--red); border-color: rgba(239,72,77,0.35); }

  .history-bars { display: flex; align-items: flex-end; gap: 5px; height: 64px; }
  .history-bar { flex: 1 1 auto; max-width: 22px; background: var(--teal); border-radius: 3px 3px 0 0; min-height: 4px; }
  .history-bar.failed { background: var(--red); }

  footer { margin-top: 32px; }
  .disclaimer {
    border: 1px solid var(--panel-border);
    border-left: 3px solid var(--amber);
    border-radius: 6px;
    background: var(--panel);
    color: var(--text);
    padding: 12px 16px;
    font-size: 13px;
    line-height: 1.5;
    margin-bottom: 10px;
  }
  .footer-meta { color: var(--text-muted); font-size: 12px; }
</style>
"""

DISCLAIMER_TEXT = (
    "This screener finds candidates by fixed rules. It does not know news, "
    "context, or whether a rule stopped making sense. Candidates feed a "
    "research process; they are not signals."
)


def _render_body(
    run_results: list["ProfileRunResult"],
    previous_by_profile: dict,
    history_by_profile: dict,
    big_move_threshold: int,
    generated_at: datetime,
) -> str:
    """Everything that goes in <body> — no <title>/<style>/wrapper tags."""
    failed_profiles = [r.profile_name for r in run_results if r.error]
    banner = ""
    if failed_profiles:
        names = ", ".join(failed_profiles)
        banner = f'<div class="banner-failed">&#9888; {len(failed_profiles)} profile(s) failed this run: {_e(names)}</div>'

    nav = "".join(f'<a href="#{_e(r.profile_name)}">{_e(r.profile_name)}</a>' for r in run_results)

    sections = "".join(
        _render_profile_section(
            r, previous_by_profile.get(r.profile_name), history_by_profile.get(r.profile_name, []), big_move_threshold
        )
        for r in run_results
    )

    generated_label = generated_at.strftime("%Y-%m-%d %H:%M:%S UTC")

    return f"""<h1>Screener Dashboard</h1>
<div class="meta">Generated {generated_label}</div>
{banner}
<nav class="profile-nav">{nav}</nav>
{sections}
<footer>
  <div class="disclaimer">{_e(DISCLAIMER_TEXT)}</div>
  <div class="footer-meta">Regenerated automatically at the end of every run &mdash; see DEPLOYMENT.md.</div>
</footer>"""


def render_fragment(
    run_results: list["ProfileRunResult"],
    previous_by_profile: dict,
    history_by_profile: dict,
    big_move_threshold: int,
    generated_at: datetime,
) -> str:
    """The Artifact-compatible page content: <title>, <style>, and body
    markup, with no <!DOCTYPE>/<html>/<head>/<body> wrapper tags — publish
    this directly as an Artifact to preview a generated dashboard."""
    body = _render_body(run_results, previous_by_profile, history_by_profile, big_move_threshold, generated_at)
    return f"{_STYLE}\n{body}\n"


def render_document(
    run_results: list["ProfileRunResult"],
    previous_by_profile: dict,
    history_by_profile: dict,
    big_move_threshold: int,
    generated_at: datetime,
) -> str:
    """The complete, standalone HTML document — what actually gets
    written to output/dashboard.html so it opens directly in any browser
    with nothing else running."""
    body = _render_body(run_results, previous_by_profile, history_by_profile, big_move_threshold, generated_at)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        f"{_STYLE}\n"
        "</head>\n"
        "<body>\n"
        f"{body}\n"
        "</body>\n"
        "</html>\n"
    )


def write_dashboard(
    run_results: list["ProfileRunResult"],
    previous_by_profile: dict,
    output_config: dict,
    generated_at: datetime,
) -> str:
    """Regenerate output/dashboard.html from this run's results. Called as
    the last step of every run (see main.py's run())."""
    history_dir = output_config.get("history_dir", "output/history")
    recent_runs = output_config.get("dashboard_recent_runs", 10)
    big_move_threshold = output_config.get("dashboard_big_move_threshold", 3)
    dashboard_path = output_config.get("dashboard_path", "output/dashboard.html")

    history_by_profile = {
        r.profile_name: scan_recent_run_counts(history_dir, r.profile_name, recent_runs)
        for r in run_results
    }

    document = render_document(run_results, previous_by_profile, history_by_profile, big_move_threshold, generated_at)

    directory = os.path.dirname(dashboard_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(dashboard_path, "w") as f:
        f.write(document)

    return dashboard_path
