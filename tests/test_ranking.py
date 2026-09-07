"""
Tests for the composite scoring/ranking system (ranking.py).

Builds SymbolResult objects directly with hand-picked score_components
(bypassing the engine and any real filters entirely) so these tests can
prove the aggregation math — normalization, weighting, min_score, top_n —
in isolation, and prove that re-weighting the same underlying data changes
the resulting ranking exactly as expected.
"""

import pandas as pd
import pytest

from engine import SymbolResult
from filters.base import FilterResult
from ranking import rank_survivors


def _survivor(symbol: str, close: float, **score_components) -> SymbolResult:
    data = pd.DataFrame({"Close": [close]})
    return SymbolResult(
        symbol=symbol,
        data=data,
        filter_results={"dummy": FilterResult(True, "ok")},
        score_components=score_components,
    )


def _failed(symbol: str) -> SymbolResult:
    return SymbolResult(
        symbol=symbol,
        data=pd.DataFrame({"Close": [100.0]}),
        filter_results={"dummy": FilterResult(False, "nope")},
        score_components={"momentum": 999.0},  # must never appear in ranking
    )


# A: strong momentum, weak trend. B: weak momentum, strong trend. C: middling both.
SURVIVORS = [
    _survivor("A", close=10.0, momentum=0.10, trend=0.01),
    _survivor("B", close=20.0, momentum=0.01, trend=0.10),
    _survivor("C", close=30.0, momentum=0.05, trend=0.05),
]


def test_only_survivors_are_ranked():
    results = SURVIVORS + [_failed("D")]
    ranked = rank_survivors(results, {"weights": {"momentum": 1.0}})
    assert {rs.symbol for rs in ranked} == {"A", "B", "C"}


def test_reweighting_changes_the_ranking_order():
    """The exact scenario the feature exists for: same underlying data, two
    different weight profiles, two different winners."""
    momentum_heavy = rank_survivors(
        SURVIVORS, {"weights": {"momentum": 0.9, "trend": 0.1}}
    )
    trend_heavy = rank_survivors(
        SURVIVORS, {"weights": {"momentum": 0.1, "trend": 0.9}}
    )

    assert momentum_heavy[0].symbol == "A"
    assert trend_heavy[0].symbol == "B"
    # Same three symbols, genuinely different order, not just a tiebreak.
    assert [rs.symbol for rs in momentum_heavy] != [rs.symbol for rs in trend_heavy]


def test_component_scores_are_normalized_into_unit_range():
    ranked = rank_survivors(SURVIVORS, {"weights": {"momentum": 0.5, "trend": 0.5}})
    for rs in ranked:
        for value in rs.component_scores.values():
            assert 0.0 <= value <= 1.0


def test_weights_need_not_sum_to_one():
    """Weights are renormalized automatically -- {0.9, 0.1} and {9, 1}
    should produce identical composite scores."""
    small = rank_survivors(SURVIVORS, {"weights": {"momentum": 0.9, "trend": 0.1}})
    large = rank_survivors(SURVIVORS, {"weights": {"momentum": 9, "trend": 1}})

    small_scores = {rs.symbol: rs.score for rs in small}
    large_scores = {rs.symbol: rs.score for rs in large}
    for symbol in small_scores:
        assert small_scores[symbol] == pytest.approx(large_scores[symbol])


def test_weight_for_absent_component_is_ignored_not_penalized():
    """A weight entered for a filter that produced no score anywhere (e.g.
    it's disabled, or has no score_component) should be dropped from the
    denominator rather than dragging every composite score toward zero."""
    ranked = rank_survivors(
        SURVIVORS, {"weights": {"momentum": 0.5, "volume_surge": 0.5}}
    )
    # volume_surge never appears in SURVIVORS -> only momentum should count,
    # renormalized to weight 1.0, so the strongest momentum symbol (A) gets
    # a perfect composite score of 1.0.
    top = ranked[0]
    assert top.symbol == "A"
    assert top.score == 1.0
    assert "volume_surge" not in top.component_scores


def test_missing_value_for_one_symbol_gets_neutral_normalization():
    survivors = [
        _survivor("A", close=10.0, momentum=0.10),
        _survivor("B", close=20.0, momentum=0.0),
        _survivor("C", close=30.0),  # no momentum key at all
    ]
    ranked = rank_survivors(survivors, {"weights": {"momentum": 1.0}})
    by_symbol = {rs.symbol: rs for rs in ranked}
    assert by_symbol["C"].component_scores["momentum"] == 0.5


def test_min_score_can_produce_an_empty_result():
    ranked = rank_survivors(
        SURVIVORS, {"weights": {"momentum": 1.0}, "min_score": 1.5}
    )
    assert ranked == []


def test_min_score_keeps_only_symbols_at_or_above_the_floor():
    ranked = rank_survivors(
        SURVIVORS, {"weights": {"momentum": 1.0}, "min_score": 0.5}
    )
    # momentum normalized: A=1.0, C=(0.05-0.01)/0.09=0.444, B=0.0
    assert [rs.symbol for rs in ranked] == ["A"]


def test_top_n_truncates_the_ranked_list():
    ranked = rank_survivors(
        SURVIVORS, {"weights": {"momentum": 1.0, "trend": 0.0}, "top_n": 1}
    )
    assert len(ranked) == 1
    assert ranked[0].symbol == "A"


def test_no_weights_configured_yields_zero_scores_but_no_crash():
    ranked = rank_survivors(SURVIVORS, {"weights": {}})
    assert {rs.symbol for rs in ranked} == {"A", "B", "C"}
    assert all(rs.score == 0.0 for rs in ranked)


def test_no_survivors_returns_empty_list():
    ranked = rank_survivors([_failed("D")], {"weights": {"momentum": 1.0}})
    assert ranked == []
