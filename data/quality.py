"""
Data-quality checks that run before screening, per symbol.

The point: a filter has no way to tell "this stock genuinely gapped 80% on
real news" from "the data provider returned garbage for this symbol" — it
just sees numbers and does math on them. Silently screening bad data
produces a confidently-wrong result that looks exactly like a real one. So
this runs first (see engine.py), on the same point-in-time-truncated data
the filters would otherwise see, and any symbol it flags is excluded with
an explicit, visible reason — never silently dropped, never scored anyway.

Three kinds of problems, each independently configurable in settings.yaml
under `data_quality`:
  - stale data:   the most recent bar is older than expected as of the
                   evaluation date (a halted, delisted, or mis-fetched symbol).
  - gaps:         a suspiciously large jump between two consecutive bars'
                   dates (missing history, not just a normal weekend).
  - suspicious
    prices:       non-positive prices, High < Low, or a single-day move
                   past a configurable magnitude.
"""

from datetime import date

import pandas as pd

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def check_symbol_data_quality(data: pd.DataFrame, evaluation_date: date, config: dict) -> list[str]:
    """Returns a list of human-readable issue descriptions for `data` (a
    single symbol's point-in-time-truncated OHLCV history). Empty list
    means no issues found. `data.empty` is not this module's concern —
    the engine already handles "no data at all" as its own distinct
    failure before this ever runs.
    """
    if data.empty:
        return []

    issues: list[str] = []
    issues.extend(_check_staleness(data, evaluation_date, config))
    issues.extend(_check_gaps(data, config))
    issues.extend(_check_suspicious_prices(data, config))
    return issues


def _check_staleness(data: pd.DataFrame, evaluation_date: date, config: dict) -> list[str]:
    max_staleness_days = config.get("max_staleness_days", 5)
    last_bar_date = data.index.max().date()
    staleness_days = (evaluation_date - last_bar_date).days
    if staleness_days > max_staleness_days:
        return [
            f"stale data: most recent bar is {last_bar_date.isoformat()} "
            f"({staleness_days}d before the evaluation date), exceeds "
            f"max_staleness_days {max_staleness_days}"
        ]
    return []


def _check_gaps(data: pd.DataFrame, config: dict) -> list[str]:
    max_gap_days = config.get("max_gap_days", 5)
    if len(data) < 2:
        return []

    sorted_dates = data.index.sort_values()
    gaps_days = sorted_dates.to_series().diff().dt.days.dropna()
    if gaps_days.empty:
        return []

    worst_gap = int(gaps_days.max())
    if worst_gap > max_gap_days:
        gap_end = sorted_dates[gaps_days.values.argmax() + 1].date()
        return [
            f"data gap: {worst_gap}d between consecutive bars (ending "
            f"{gap_end.isoformat()}), exceeds max_gap_days {max_gap_days}"
        ]
    return []


def _check_suspicious_prices(data: pd.DataFrame, config: dict) -> list[str]:
    issues = []

    price_columns = data[["Open", "High", "Low", "Close"]]
    if (price_columns <= 0).any().any():
        issues.append("suspicious price: non-positive Open/High/Low/Close value present")

    if (data["High"] < data["Low"]).any():
        issues.append("suspicious price: High < Low on at least one bar")

    max_daily_move_pct = config.get("max_daily_move_pct", 50.0)
    daily_moves_pct = data["Close"].pct_change().abs() * 100
    worst_move = daily_moves_pct.max()
    if pd.notna(worst_move) and worst_move > max_daily_move_pct:
        worst_date = daily_moves_pct.idxmax().date()
        issues.append(
            f"suspicious price: {worst_move:.1f}% single-day move on "
            f"{worst_date.isoformat()} exceeds max_daily_move_pct {max_daily_move_pct:.1f}%"
        )

    return issues
