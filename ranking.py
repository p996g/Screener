"""
Scores and ranks the symbols that survived the filter pipeline.

Unlike the filter pipeline (a hard pass/fail gate), ranking is about
ordering the survivors by *how good* they are. Each filter may optionally
expose a continuous "score component" via `Filter.score_component()` (see
filters/base.py) — e.g. momentum strength, trend quality, volume surge
magnitude, proximity to a trigger level. The engine computes these
alongside pass/fail for every symbol and stores them on `SymbolResult`
(engine.py), so this module only ever does aggregation math:

  1. Take every symbol that passed all filters (the survivors).
  2. For each filter name listed in `ranking.weights` (settings.yaml) that
     actually produced a score for at least one survivor, min-max
     normalize that component to [0, 1] across the survivor set.
  3. Combine the normalized components into one composite score using the
     configured weights (renormalized to sum to 1 automatically, so they
     don't need to add up to 1 in the config).
  4. Drop anything below `ranking.min_score`, sort by composite score, and
     keep at most `ranking.top_n`.

Because every filter's score_component is already oriented so "higher is
always better" (that's the contract in filters/base.py), this module never
needs a per-factor "higher_is_better" flag — re-weighting is purely a
matter of editing the numbers in `ranking.weights`.
"""

from dataclasses import dataclass

from engine import SymbolResult


@dataclass
class RankedSymbol:
    symbol: str
    score: float
    close: float
    component_scores: dict[str, float]  # filter name -> normalized [0, 1] contribution


def rank_survivors(results: list[SymbolResult], ranking_config: dict) -> list[RankedSymbol]:
    """Score and sort every symbol that passed all filters.

    Symbols that failed the filter pipeline are excluded entirely — ranking
    only ever orders candidates that are already eligible.
    """
    survivors = [r for r in results if r.passed]
    configured_weights = ranking_config.get("weights", {})

    active_components = _active_components(survivors, configured_weights)
    normalized_by_component = {
        name: _normalize_component(
            {r.symbol: r.score_components.get(name) for r in survivors}
        )
        for name in active_components
    }
    total_weight = sum(configured_weights[name] for name in active_components) or 1.0

    ranked = []
    for r in survivors:
        component_scores = {
            name: normalized_by_component[name][r.symbol] for name in active_components
        }
        score = sum(
            (configured_weights[name] / total_weight) * component_scores[name]
            for name in active_components
        )
        ranked.append(
            RankedSymbol(
                symbol=r.symbol,
                score=score,
                close=float(r.data["Close"].iloc[-1]),
                component_scores=component_scores,
            )
        )

    ranked.sort(key=lambda rs: rs.score, reverse=True)

    min_score = ranking_config.get("min_score")
    if min_score is not None:
        ranked = [rs for rs in ranked if rs.score >= min_score]

    top_n = ranking_config.get("top_n")
    if top_n is not None:
        ranked = ranked[:top_n]

    return ranked


def _active_components(
    survivors: list[SymbolResult], configured_weights: dict
) -> list[str]:
    """Filter names from `ranking.weights` that have a non-zero weight and
    actually produced a score for at least one survivor. A weight entered
    for a disabled filter, or one with no score_component, is silently
    excluded rather than distorting the composite score with an all-neutral
    component."""
    return [
        name
        for name, weight in configured_weights.items()
        if weight and any(r.score_components.get(name) is not None for r in survivors)
    ]


def _normalize_component(raw_by_symbol: dict[str, float | None]) -> dict[str, float]:
    """Min-max normalize to [0, 1]. A symbol missing this component (None)
    gets a neutral 0.5 rather than being penalized or favored. If every
    survivor ties (or all are missing), every symbol gets 0.5."""
    known_values = [v for v in raw_by_symbol.values() if v is not None]
    if not known_values:
        return {symbol: 0.5 for symbol in raw_by_symbol}

    lo, hi = min(known_values), max(known_values)
    if hi == lo:
        return {symbol: 0.5 for symbol in raw_by_symbol}

    return {
        symbol: 0.5 if value is None else (value - lo) / (hi - lo)
        for symbol, value in raw_by_symbol.items()
    }
