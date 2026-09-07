"""
Look-back audit: verifies no filter in the library ever lets a bar dated
after the evaluation date influence its result.

Re-uses the exact technique already proven in engine.py's own test suite
(tests/test_engine.py's test_engine_truncates_future_bars_even_if_fetcher_
does_not): feed the real ScreenerEngine two versions of the same history
for the same evaluation_date — one "clean" (nothing after the evaluation
date), one "poisoned" (the same history with deliberately extreme,
easily-detected bars appended *after* the evaluation date) — and assert
every filter's pass_fail and score_component come out bit-for-bit
identical either way. If they don't, something downstream of the fetcher
is leaking future data into a result — since the engine only ever hands a
filter data truncated to the evaluation date, an identical result under
poisoning is exactly what "point-in-time safe" means.

Where the original test exercised one hand-written fake filter, this runs
against every filter actually registered in filters/library.py's
FILTER_REGISTRY, using each one's own default config straight out of
config/settings.yaml — so it audits what's really shipped, and stays
current automatically as filters are added. Run it directly:

    python audit.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yaml

from engine import ScreenerEngine
from filters.library import FILTER_REGISTRY

EVALUATION_DATE = date(2024, 6, 3)
HISTORY_DAYS = 400
POISON_DAYS = 10
AUDIT_SYMBOL = "AUDIT"


@dataclass(frozen=True)
class AuditFinding:
    filter_name: str
    ok: bool
    detail: str


class _StaticFetcher:
    """Always returns the same pre-built data for AUDIT_SYMBOL (and an
    empty frame for anything else, e.g. a filter's benchmark symbol) —
    no network, fully deterministic."""

    def __init__(self, data: pd.DataFrame):
        self._data = data

    def fetch(self, symbol: str, evaluation_date: date) -> pd.DataFrame:
        if symbol == AUDIT_SYMBOL:
            return self._data
        return pd.DataFrame()


def _synthetic_history(end_date: date, n_days: int = HISTORY_DAYS, seed: int = 42) -> pd.DataFrame:
    """A deterministic, plausible-looking daily OHLCV history ending on
    `end_date` — long enough to satisfy every filter's longest lookback
    (e.g. a 200-day trend)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=end_date, periods=n_days)
    daily_returns = rng.normal(loc=0.0003, scale=0.015, size=n_days)
    closes = 100.0 * np.cumprod(1 + daily_returns)
    highs = closes * (1 + rng.uniform(0.0, 0.01, size=n_days))
    lows = closes * (1 - rng.uniform(0.0, 0.01, size=n_days))
    opens = closes * (1 + rng.uniform(-0.005, 0.005, size=n_days))
    volumes = rng.integers(1_000_000, 5_000_000, size=n_days)
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=dates,
    )


def _poison_future(clean: pd.DataFrame, evaluation_date: date, extra_days: int = POISON_DAYS) -> pd.DataFrame:
    """`clean` plus `extra_days` bars dated after `evaluation_date`, each
    set to an extreme, unmistakable value — if any of this leaks into a
    filter's result, the result will visibly change."""
    future_dates = pd.bdate_range(start=pd.Timestamp(evaluation_date) + timedelta(days=1), periods=extra_days)
    poison = pd.DataFrame(
        {
            "Open": [999_999.0] * extra_days,
            "High": [999_999.0] * extra_days,
            "Low": [0.0001] * extra_days,
            "Close": [999_999.0] * extra_days,
            "Volume": [999_999_999] * extra_days,
        },
        index=future_dates,
    )
    return pd.concat([clean, poison]).sort_index()


def audit_filter(filter_name: str, filter_config: dict) -> AuditFinding:
    """Runs one filter (by its FILTER_REGISTRY name and a config dict for
    it) against a clean history and a poisoned variant of the same
    history for the same evaluation date, and reports whether the result
    differed."""
    filter_cls = FILTER_REGISTRY[filter_name]
    filt = filter_cls(filter_config)

    clean = _synthetic_history(EVALUATION_DATE)
    poisoned = _poison_future(clean, EVALUATION_DATE)

    engine = ScreenerEngine(filters=[filt], evaluation_date=EVALUATION_DATE)
    clean_result = engine.run([AUDIT_SYMBOL], _StaticFetcher(clean))[0]
    poisoned_result = engine.run([AUDIT_SYMBOL], _StaticFetcher(poisoned))[0]

    clean_fr = clean_result.filter_results.get(filter_name)
    poisoned_fr = poisoned_result.filter_results.get(filter_name)
    if clean_fr is None or poisoned_fr is None:
        return AuditFinding(filter_name, False, "filter produced no result on synthetic audit data")

    if clean_fr.passed != poisoned_fr.passed or clean_fr.reason != poisoned_fr.reason:
        return AuditFinding(
            filter_name,
            False,
            f"pass_fail changed when future data was present: "
            f"clean=({clean_fr.passed}, {clean_fr.reason!r}) vs "
            f"poisoned=({poisoned_fr.passed}, {poisoned_fr.reason!r})",
        )

    clean_score = clean_result.score_components.get(filter_name)
    poisoned_score = poisoned_result.score_components.get(filter_name)
    if clean_score != poisoned_score:
        return AuditFinding(
            filter_name,
            False,
            f"score_component changed when future data was present: {clean_score!r} vs {poisoned_score!r}",
        )

    return AuditFinding(filter_name, True, "result unaffected by bars after the evaluation date")


def run_lookback_audit(filters_config: dict) -> list[AuditFinding]:
    """Audits every filter registered in FILTER_REGISTRY, using its own
    config block from `filters_config` (typically settings.yaml's
    top-level `filters` section) — `enabled` is ignored; every registered
    filter is audited regardless of whether any profile currently uses it.
    """
    findings = []
    for filter_name in sorted(FILTER_REGISTRY):
        config = filters_config.get(filter_name)
        if config is None:
            findings.append(
                AuditFinding(filter_name, False, "no config block found in settings.yaml to audit against")
            )
            continue
        findings.append(audit_filter(filter_name, config))
    return findings


def main() -> None:
    with open("config/settings.yaml") as f:
        config = yaml.safe_load(f)

    findings = run_lookback_audit(config["filters"])

    for finding in findings:
        status = "PASS" if finding.ok else "FAIL"
        print(f"[{status}] {finding.filter_name}: {finding.detail}")

    failed = [f for f in findings if not f.ok]
    if failed:
        print(f"\n{len(failed)}/{len(findings)} filter(s) failed the look-back audit.")
        sys.exit(1)

    print(f"\nAll {len(findings)} filter(s) passed the look-back audit.")


if __name__ == "__main__":
    main()
