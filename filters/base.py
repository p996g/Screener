"""
Abstract base class that every screener filter inherits from.

A "filter" is a single pass/fail test applied to one symbol's point-in-time
OHLCV history (e.g. "is the price above its 50-day moving average?"). The
engine (see engine.py) runs every enabled filter against every symbol in the
universe and records the outcome, so the final report can always explain
*why* a symbol passed or failed.

Design rules for subclasses (enforced by convention, not by the type system):
  - All thresholds/parameters must come from the `config` dict passed into
    __init__ — never hard-code a number in a filter implementation. This is
    what lets someone customize the screener by editing settings.yaml alone.
  - `pass_fail` must only look at the rows present in `symbol_data` and
    `context`. The engine is responsible for truncating all data (including
    any reference/benchmark data in `context`) to the evaluation date before
    it ever reaches a filter, so a correctly written filter is automatically
    point-in-time safe — it simply has no future data to leak.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FilterResult:
    """Outcome of running one filter against one symbol."""

    passed: bool
    reason: str  # human-readable explanation, used for both pass and fail


class Filter(ABC):
    """Base class for all screener filters.

    Subclasses must set a class-level `name` (used as the key in
    settings.yaml's `filters:` section and in the engine's results) and
    implement `pass_fail`.
    """

    name: str = "base_filter"

    def __init__(self, config: dict):
        """
        Parameters
        ----------
        config:
            The sub-dictionary from settings.yaml for this filter, e.g. for
            `name = "min_price"` this is the contents of
            `filters.min_price` — nothing more, nothing less. Subclasses
            pull every parameter they need out of this dict.
        """
        self.config = config

    @abstractmethod
    def pass_fail(
        self, symbol_data: pd.DataFrame, context: dict | None = None
    ) -> FilterResult:
        """Evaluate this filter against one symbol's OHLCV history.

        Parameters
        ----------
        symbol_data:
            A DataFrame of OHLCV bars for a single symbol, indexed by date
            in ascending order, already truncated to the evaluation date
            (no future bars). Expected columns: "Open", "High", "Low",
            "Close", "Volume".
        context:
            Optional extra, run-level information the engine makes
            available to every filter call:
              - "symbol": the ticker being evaluated.
              - "evaluation_date": the date the whole run is as-of.
              - "reference_data": dict of {symbol: DataFrame} for any extra
                symbols requested via `required_symbols()` (e.g. a
                benchmark for relative momentum), truncated to the
                evaluation date exactly like `symbol_data`.
            Most filters never need this and can ignore it entirely.

        Returns
        -------
        FilterResult
            `passed=True/False` plus a `reason` string explaining the
            outcome (e.g. "close 4.20 < min_price 5.00").
        """
        raise NotImplementedError

    def required_symbols(self) -> list[str]:
        """Extra ticker symbols (beyond the one being screened) this filter
        needs data for, e.g. a benchmark for relative momentum. The engine
        fetches each of these once per run (not once per symbol) and makes
        them available via `context["reference_data"]`. Most filters need
        nothing extra, hence the empty default.
        """
        return []

    def score_component(
        self, symbol_data: pd.DataFrame, context: dict | None = None
    ) -> float | None:
        """Optional continuous "strength" signal this filter contributes to
        ranking (see ranking.py) — e.g. momentum magnitude, distance above
        a trend line, volume surge multiple, closeness to a breakout level.

        This is deliberately separate from pass_fail: pass_fail is a gate
        (did the symbol qualify at all), this is a score (how good was it
        among symbols that qualified). A filter with no meaningful notion
        of "how good" (e.g. a static price range or blacklist) simply
        doesn't override this and contributes nothing to ranking.

        Convention: a higher returned value must always mean "more
        desirable" — if the filter's own config makes a *lower* raw
        quantity more desirable (e.g. `direction: below`), flip the sign
        before returning so ranking.py never needs to know about it.

        Return None (the default) when there isn't enough data to compute
        this, or when the filter has no score component at all.
        """
        return None

    def _result(self, passed: bool, detail: str) -> FilterResult:
        """Build a FilterResult with a consistently formatted one-line
        reason: "<passed|failed> <ClassName>: <detail>"."""
        status = "passed" if passed else "failed"
        return FilterResult(
            passed=passed, reason=f"{status} {type(self).__name__}: {detail}"
        )
