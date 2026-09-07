"""
Runs the universe through the filter pipeline.

For every symbol, the engine:
  1. Fetches point-in-time OHLCV data (already truncated to the evaluation
     date by the fetcher — the engine re-asserts this truncation itself,
     see _truncate_to_evaluation_date, so the pipeline stays look-ahead
     safe even if a fetcher implementation forgets to).
  2. If `data_quality` is enabled (settings.yaml), runs the data-quality
     checks (see data/quality.py) against that same point-in-time data. A
     symbol with an issue (stale data, a gap, a suspicious price) is
     excluded right here, before any filter sees it — never silently
     screened on data that might be garbage.
  3. Runs every enabled filter against that data, in order, passing along a
     shared `context` (the symbol, the evaluation date, and any reference
     data a filter asked for via `required_symbols()` — e.g. a benchmark
     for relative momentum).
  4. Records each filter's pass/fail result *and* its optional score
     component (see Filter.score_component in filters/base.py), so the
     outcome for every symbol is fully explainable and, for survivors,
     ready to be scored and ranked (see ranking.py).

A symbol "survives" only if it passes every enabled filter (and, first,
the data-quality check).

Before screening the universe, the engine fetches each distinct extra
symbol requested by `filter.required_symbols()` exactly once (not once per
screened symbol) and truncates it to the same evaluation date, so a
benchmark like SPY is only downloaded a single time per run.
"""

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from data.quality import check_symbol_data_quality
from filters.base import Filter, FilterResult


@dataclass
class SymbolResult:
    """Everything the engine learned about one symbol."""

    symbol: str
    data: pd.DataFrame
    filter_results: dict[str, FilterResult] = field(default_factory=dict)
    score_components: dict[str, float | None] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """A symbol survives only if every recorded filter passed (and at
        least one filter actually ran)."""
        if not self.filter_results:
            return False
        return all(result.passed for result in self.filter_results.values())

    @property
    def failure_reasons(self) -> list[str]:
        return [
            f"{name}: {result.reason}"
            for name, result in self.filter_results.items()
            if not result.passed
        ]


class ScreenerEngine:
    """Applies a list of Filters to every symbol in a universe."""

    def __init__(
        self,
        filters: list[Filter],
        evaluation_date: date,
        data_quality_config: dict | None = None,
    ):
        self.filters = filters
        self.evaluation_date = evaluation_date
        self.data_quality_config = data_quality_config or {}

    def run(self, universe: list[str], fetcher) -> list[SymbolResult]:
        """Fetch data and evaluate every filter for every symbol.

        `fetcher` must expose `fetch(symbol, evaluation_date) -> DataFrame`
        (see data/fetcher.py's OHLCVFetcher). Kept duck-typed so tests can
        pass a lightweight stand-in with no network access.
        """
        reference_data = self._fetch_reference_data(fetcher)

        results: list[SymbolResult] = []
        for symbol in universe:
            data = fetcher.fetch(symbol, self.evaluation_date)
            data = self._truncate_to_evaluation_date(data)
            context = {
                "symbol": symbol,
                "evaluation_date": self.evaluation_date,
                "reference_data": reference_data,
            }
            results.append(self._evaluate_symbol(symbol, data, context))
        return results

    def _fetch_reference_data(self, fetcher) -> dict[str, pd.DataFrame]:
        """Fetch every extra symbol requested by any filter's
        `required_symbols()`, once each, truncated to the evaluation date."""
        needed_symbols: set[str] = set()
        for filt in self.filters:
            needed_symbols.update(s for s in filt.required_symbols() if s)

        reference_data = {}
        for symbol in needed_symbols:
            data = fetcher.fetch(symbol, self.evaluation_date)
            reference_data[symbol] = self._truncate_to_evaluation_date(data)
        return reference_data

    def _evaluate_symbol(
        self, symbol: str, data: pd.DataFrame, context: dict
    ) -> SymbolResult:
        if data.empty:
            return SymbolResult(
                symbol=symbol,
                data=data,
                filter_results={
                    "data_availability": FilterResult(
                        False, "no price data available as of evaluation date"
                    )
                },
            )

        if self.data_quality_config.get("enabled"):
            issues = check_symbol_data_quality(data, self.evaluation_date, self.data_quality_config)
            if issues:
                return SymbolResult(
                    symbol=symbol,
                    data=data,
                    filter_results={"data_quality": FilterResult(False, "; ".join(issues))},
                )

        filter_results = {}
        score_components = {}
        for filt in self.filters:
            filter_results[filt.name] = filt.pass_fail(data, context)
            score_components[filt.name] = filt.score_component(data, context)

        return SymbolResult(
            symbol=symbol,
            data=data,
            filter_results=filter_results,
            score_components=score_components,
        )

    def _truncate_to_evaluation_date(self, data: pd.DataFrame) -> pd.DataFrame:
        """Defense-in-depth: drop any bar dated after evaluation_date so no
        filter can ever see future data, even if the fetcher didn't already
        guarantee it."""
        if data.empty:
            return data
        eval_ts = pd.Timestamp(self.evaluation_date)
        return data.loc[data.index <= eval_ts]
