"""
Built-in filter implementations.

Each filter is a small subclass of `Filter` (see filters/base.py) that reads
every parameter it needs out of the config dict it's constructed with —
never from a hard-coded constant. To add a new filter:

  1. Write a class here that subclasses Filter and implements pass_fail().
  2. Register it in FILTER_REGISTRY below under the name it will use in
     settings.yaml (filters.<name>).
  3. Add a `filters.<name>: {enabled: true, ...}` block to settings.yaml.

The engine never needs to change when a filter is added — it only ever asks
FILTER_REGISTRY for whatever names have `enabled: true` in settings.yaml.
"""

import math
from datetime import date, datetime

import pandas as pd

from filters.base import Filter, FilterResult


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------


def _pct_return(closes: pd.Series, lookback_days: int) -> float:
    """Fractional price change from `lookback_days` bars ago to the latest
    close, e.g. 0.05 means +5%."""
    past_close = float(closes.iloc[-(lookback_days + 1)])
    last_close = float(closes.iloc[-1])
    if past_close == 0:
        return 0.0
    return (last_close - past_close) / past_close


def _true_range(data: pd.DataFrame) -> pd.Series:
    prev_close = data["Close"].shift(1)
    return pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - prev_close).abs(),
            (data["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _atr_pct(data: pd.DataFrame, lookback_days: int) -> float:
    """Average True Range over `lookback_days`, expressed as a % of the
    latest close so it's comparable across symbols at different price
    levels."""
    atr = float(_true_range(data).tail(lookback_days).mean())
    last_close = float(data["Close"].iloc[-1])
    if last_close == 0:
        return 0.0
    return atr / last_close * 100


def _realized_vol_pct(data: pd.DataFrame, lookback_days: int) -> float:
    """Annualized realized volatility (stdev of daily returns * sqrt(252)),
    as a percentage, over the trailing `lookback_days` returns."""
    returns = data["Close"].pct_change().dropna().tail(lookback_days)
    daily_std = float(returns.std())
    return daily_std * math.sqrt(252) * 100


# -----------------------------------------------------------------------------
# Filters
# -----------------------------------------------------------------------------


class LiquidityFilter(Filter):
    """Rejects thinly-traded symbols using average dollar volume or average
    share volume over a trailing window as a liquidity proxy."""

    name = "liquidity"

    def pass_fail(self, symbol_data: pd.DataFrame, context: dict | None = None) -> FilterResult:
        metric = self.config["metric"]  # "dollar_volume" or "share_volume"
        lookback_days = self.config["lookback_days"]
        min_value = self.config["min_value"]

        if len(symbol_data) < lookback_days:
            return self._result(
                False,
                f"only {len(symbol_data)} bars available, need {lookback_days} "
                f"for avg {metric}",
            )

        window = symbol_data.tail(lookback_days)
        if metric == "dollar_volume":
            series = window["Close"] * window["Volume"]
            unit = "dollar volume"
        elif metric == "share_volume":
            series = window["Volume"]
            unit = "share volume"
        else:
            raise ValueError(
                f"Unknown liquidity metric '{metric}' — expected "
                f"'dollar_volume' or 'share_volume'"
            )

        avg_value = float(series.mean())
        passed = avg_value >= min_value
        comparator = ">=" if passed else "<"
        return self._result(
            passed,
            f"avg {unit} {avg_value:,.0f} over {lookback_days}d "
            f"{comparator} min {min_value:,.0f}",
        )


class PriceRangeFilter(Filter):
    """Requires the last close to fall within a configurable [min, max]
    price band. Either bound may be null for a one-sided range."""

    name = "price_range"

    def pass_fail(self, symbol_data: pd.DataFrame, context: dict | None = None) -> FilterResult:
        min_price = self.config.get("min_price")
        max_price = self.config.get("max_price")

        if symbol_data.empty:
            return self._result(False, "no price data available")

        last_close = float(symbol_data["Close"].iloc[-1])

        if min_price is not None and last_close < min_price:
            return self._result(
                False, f"close {last_close:.2f} below min {min_price:.2f}"
            )
        if max_price is not None and last_close > max_price:
            return self._result(
                False, f"close {last_close:.2f} above max {max_price:.2f}"
            )

        lo = f"{min_price:.2f}" if min_price is not None else "-inf"
        hi = f"{max_price:.2f}" if max_price is not None else "+inf"
        return self._result(True, f"close {last_close:.2f} within [{lo}, {hi}]")


class TrendFilter(Filter):
    """Requires the last close to be above/below a moving average of a
    configurable length, optionally also requiring that moving average
    itself to be rising or falling."""

    name = "trend"

    def _moving_average(self, symbol_data: pd.DataFrame) -> float | None:
        period = self.config["period"]
        if len(symbol_data) < period:
            return None
        return float(symbol_data["Close"].tail(period).mean())

    def pass_fail(self, symbol_data: pd.DataFrame, context: dict | None = None) -> FilterResult:
        period = self.config["period"]
        direction = self.config["direction"]  # "above" or "below"
        require_slope = self.config.get("require_slope", "none")  # none|rising|falling
        slope_lookback_days = self.config.get("slope_lookback_days", 5)

        needs_slope = require_slope in ("rising", "falling")
        min_bars_needed = period + (slope_lookback_days if needs_slope else 0)
        if len(symbol_data) < min_bars_needed:
            return self._result(
                False,
                f"only {len(symbol_data)} bars available, need "
                f"{min_bars_needed} for {period}-day MA"
                + (" slope" if needs_slope else ""),
            )

        closes = symbol_data["Close"]
        ma_now = self._moving_average(symbol_data)
        last_close = float(closes.iloc[-1])

        price_ok = last_close >= ma_now if direction == "above" else last_close <= ma_now
        actual_relation = "above" if last_close >= ma_now else "below"
        detail = (
            f"close {last_close:.2f} is {actual_relation} {period}-day MA "
            f"{ma_now:.2f} (required {direction})"
        )

        slope_ok = True
        if needs_slope:
            ma_past = float(
                closes.iloc[-(period + slope_lookback_days) : -slope_lookback_days].mean()
            )
            if ma_now > ma_past:
                actual_slope = "rising"
            elif ma_now < ma_past:
                actual_slope = "falling"
            else:
                actual_slope = "flat"
            slope_ok = actual_slope == require_slope
            detail += (
                f", {period}-day MA {actual_slope} ({ma_past:.2f} -> {ma_now:.2f}, "
                f"required {require_slope})"
            )

        return self._result(price_ok and slope_ok, detail)

    def score_component(self, symbol_data: pd.DataFrame, context: dict | None = None) -> float | None:
        """Trend quality: the signed % distance of price from its MA,
        oriented so that a stronger trend *in the configured direction*
        always scores higher (a downtrend screen scores steeper declines
        higher, not lower)."""
        ma_now = self._moving_average(symbol_data)
        if ma_now is None or ma_now == 0:
            return None
        last_close = float(symbol_data["Close"].iloc[-1])
        distance = (last_close - ma_now) / ma_now
        direction = self.config["direction"]
        return distance if direction == "above" else -distance


class MomentumFilter(Filter):
    """Requires the return over a configurable lookback to be above/below a
    threshold. If `benchmark_symbol` is set, the threshold is applied to
    the symbol's return *relative to* the benchmark's return over the same
    window (excess return) instead of the symbol's absolute return."""

    name = "momentum"

    def required_symbols(self) -> list[str]:
        benchmark_symbol = self.config.get("benchmark_symbol")
        return [benchmark_symbol] if benchmark_symbol else []

    def _compute_return(
        self, symbol_data: pd.DataFrame, context: dict | None
    ) -> tuple[float | None, str]:
        """Returns (metric, label_or_error). `metric` is None when there
        isn't enough data (or benchmark data) to compute a return; in that
        case `label_or_error` explains why."""
        lookback_days = self.config["lookback_days"]
        benchmark_symbol = self.config.get("benchmark_symbol")

        if len(symbol_data) <= lookback_days:
            return None, (
                f"only {len(symbol_data)} bars available, need more than "
                f"{lookback_days} for {lookback_days}-day momentum"
            )

        symbol_return = _pct_return(symbol_data["Close"], lookback_days)

        if not benchmark_symbol:
            return symbol_return, f"{lookback_days}-day return {symbol_return:+.2%}"

        reference_data = (context or {}).get("reference_data", {})
        benchmark_data = reference_data.get(benchmark_symbol)
        if benchmark_data is None or len(benchmark_data) <= lookback_days:
            return None, (
                f"benchmark '{benchmark_symbol}' data unavailable for "
                f"{lookback_days}-day relative momentum"
            )
        benchmark_return = _pct_return(benchmark_data["Close"], lookback_days)
        metric = symbol_return - benchmark_return
        return metric, f"{lookback_days}-day return {metric:+.2%} vs {benchmark_symbol}"

    def pass_fail(self, symbol_data: pd.DataFrame, context: dict | None = None) -> FilterResult:
        direction = self.config["direction"]  # "above" or "below"
        threshold = self.config["threshold"]

        metric, label_or_error = self._compute_return(symbol_data, context)
        if metric is None:
            return self._result(False, label_or_error)

        passed = metric > threshold if direction == "above" else metric < threshold
        return self._result(
            passed, f"{label_or_error} ({direction} threshold {threshold:+.2%})"
        )

    def score_component(self, symbol_data: pd.DataFrame, context: dict | None = None) -> float | None:
        """Momentum strength: the (possibly benchmark-relative) return
        itself, oriented so a `direction: below` screen (hunting for
        weakness) scores the weakest symbols highest."""
        metric, _ = self._compute_return(symbol_data, context)
        if metric is None:
            return None
        direction = self.config["direction"]
        return metric if direction == "above" else -metric


class ProximityFilter(Filter):
    """Requires the last close to be within a configurable percentage of a
    reference level — an N-day high, an N-day low, or a moving average.
    Used for breakout screens (reference="high") and pullback screens
    (reference="sma" or "low"). `side` optionally restricts which side of
    the reference level the price must be on."""

    name = "proximity"

    def _reference_level(self, symbol_data: pd.DataFrame) -> tuple[float | None, str]:
        """Returns (level, label_or_error). `level` is None when there
        isn't enough data, in which case `label_or_error` explains why."""
        reference = self.config["reference"]  # "high", "low", or "sma"

        if reference in ("high", "low"):
            lookback_days = self.config["lookback_days"]
            if len(symbol_data) < lookback_days:
                return None, (
                    f"only {len(symbol_data)} bars available, need "
                    f"{lookback_days} for {lookback_days}-day {reference}"
                )
            window = symbol_data.tail(lookback_days)
            level = (
                float(window["High"].max())
                if reference == "high"
                else float(window["Low"].min())
            )
            return level, f"{lookback_days}-day {reference}"

        if reference == "sma":
            sma_period = self.config["sma_period"]
            if len(symbol_data) < sma_period:
                return None, (
                    f"only {len(symbol_data)} bars available, need "
                    f"{sma_period} for SMA{sma_period}"
                )
            level = float(symbol_data["Close"].tail(sma_period).mean())
            return level, f"SMA{sma_period}"

        raise ValueError(
            f"Unknown proximity reference '{reference}' — expected "
            f"'high', 'low', or 'sma'"
        )

    def pass_fail(self, symbol_data: pd.DataFrame, context: dict | None = None) -> FilterResult:
        max_distance_pct = self.config["max_distance_pct"]
        side = self.config.get("side", "any")  # any | at_or_above | at_or_below

        level, label_or_error = self._reference_level(symbol_data)
        if level is None:
            return self._result(False, label_or_error)
        if level == 0:
            return self._result(
                False, f"{label_or_error} is zero, cannot compute distance"
            )
        label = label_or_error

        last_close = float(symbol_data["Close"].iloc[-1])
        distance_pct = (last_close - level) / level * 100
        within_band = abs(distance_pct) <= max_distance_pct

        if side == "at_or_above":
            side_ok = distance_pct >= 0
        elif side == "at_or_below":
            side_ok = distance_pct <= 0
        elif side == "any":
            side_ok = True
        else:
            raise ValueError(
                f"Unknown proximity side '{side}' — expected 'any', "
                f"'at_or_above', or 'at_or_below'"
            )

        direction_word = "above" if distance_pct >= 0 else "below"
        detail = (
            f"close {last_close:.2f} is {abs(distance_pct):.2f}% "
            f"{direction_word} {label} {level:.2f} "
            f"(band {max_distance_pct:.2f}%, side {side})"
        )
        return self._result(within_band and side_ok, detail)

    def score_component(self, symbol_data: pd.DataFrame, context: dict | None = None) -> float | None:
        """Proximity to trigger: negative absolute distance from the
        reference level, so the closest symbols (best breakout/pullback
        setups) score highest."""
        level, _ = self._reference_level(symbol_data)
        if level is None or level == 0:
            return None
        last_close = float(symbol_data["Close"].iloc[-1])
        distance_pct = (last_close - level) / level * 100
        return -abs(distance_pct)


class VolatilityFilter(Filter):
    """Requires a volatility measure (ATR as % of price, or annualized
    realized volatility) over a trailing lookback to fall within a
    configurable [min, max] band. Either bound may be null."""

    name = "volatility"

    def pass_fail(self, symbol_data: pd.DataFrame, context: dict | None = None) -> FilterResult:
        metric = self.config["metric"]  # "atr" or "realized_vol"
        lookback_days = self.config["lookback_days"]
        min_value = self.config.get("min_value")
        max_value = self.config.get("max_value")
        required_bars = lookback_days + 1

        if len(symbol_data) < required_bars:
            return self._result(
                False,
                f"only {len(symbol_data)} bars available, need "
                f"{required_bars} for {lookback_days}-day {metric}",
            )

        if metric == "atr":
            value = _atr_pct(symbol_data, lookback_days)
            label = f"{lookback_days}-day ATR"
            unit = "% of price"
        elif metric == "realized_vol":
            value = _realized_vol_pct(symbol_data, lookback_days)
            label = f"{lookback_days}-day annualized realized volatility"
            unit = "%"
        else:
            raise ValueError(
                f"Unknown volatility metric '{metric}' — expected 'atr' or "
                f"'realized_vol'"
            )

        if min_value is not None and value < min_value:
            return self._result(
                False, f"{label} {value:.2f}{unit} below min {min_value:.2f}"
            )
        if max_value is not None and value > max_value:
            return self._result(
                False, f"{label} {value:.2f}{unit} above max {max_value:.2f}"
            )

        return self._result(True, f"{label} {value:.2f}{unit} within configured band")


class VolumeSurgeFilter(Filter):
    """Requires the most recent day's volume, as a multiple of its own
    trailing average volume, to be above a threshold (`direction: above` —
    a surge, e.g. breakout confirmation) or below it (`direction: below` —
    contracting volume, e.g. the "volume dries up before a breakout"
    pattern). Same underlying multiple, just compared the other way."""

    name = "volume_surge"

    def _surge_multiple(self, symbol_data: pd.DataFrame) -> tuple[float | None, float | None, float | None]:
        """Returns (multiple, last_volume, avg_volume), each None if there
        isn't enough data or the average is zero."""
        avg_lookback_days = self.config["avg_lookback_days"]
        required_bars = avg_lookback_days + 1
        if len(symbol_data) < required_bars:
            return None, None, None

        volumes = symbol_data["Volume"]
        avg_volume = float(volumes.iloc[-(avg_lookback_days + 1) : -1].mean())
        last_volume = float(volumes.iloc[-1])
        if avg_volume == 0:
            return None, last_volume, avg_volume

        return last_volume / avg_volume, last_volume, avg_volume

    def pass_fail(self, symbol_data: pd.DataFrame, context: dict | None = None) -> FilterResult:
        avg_lookback_days = self.config["avg_lookback_days"]
        direction = self.config.get("direction", "above")  # "above" or "below"
        threshold_multiple = self.config["threshold_multiple"]
        required_bars = avg_lookback_days + 1

        if len(symbol_data) < required_bars:
            return self._result(
                False,
                f"only {len(symbol_data)} bars available, need "
                f"{required_bars} for {avg_lookback_days}-day avg volume",
            )

        multiple, last_volume, avg_volume = self._surge_multiple(symbol_data)
        if multiple is None:
            return self._result(
                False,
                f"{avg_lookback_days}-day avg volume is zero, cannot "
                f"compute surge multiple",
            )

        passed = multiple >= threshold_multiple if direction == "above" else multiple <= threshold_multiple
        comparator = ">=" if direction == "above" else "<="
        return self._result(
            passed,
            f"volume {last_volume:,.0f} is {multiple:.2f}x its "
            f"{avg_lookback_days}-day avg {avg_volume:,.0f} "
            f"(required {comparator} {threshold_multiple:.2f}x)",
        )

    def score_component(self, symbol_data: pd.DataFrame, context: dict | None = None) -> float | None:
        """Volume surge magnitude: the multiple itself, oriented so that a
        `direction: below` (drying-up) screen scores the most contracted
        volume highest, just like Momentum/TrendFilter orient by
        `direction`."""
        multiple, _, _ = self._surge_multiple(symbol_data)
        if multiple is None:
            return None
        direction = self.config.get("direction", "above")
        return multiple if direction == "above" else -multiple


class GapFilter(Filter):
    """Excludes symbols around a known event date (e.g. earnings) or on a
    static blacklist.

    No live events feed is wired up in this scaffold, so the "event
    calendar" is a hand-curated `event_dates` map in settings.yaml
    (symbol -> list of "YYYY-MM-DD" dates). A symbol is excluded if the
    evaluation date falls within `exclusion_days` of any of its listed
    event dates. Swapping in a real events API later only means changing
    how `event_dates` is populated — this filter's logic doesn't change.
    """

    name = "gap_event"

    def pass_fail(self, symbol_data: pd.DataFrame, context: dict | None = None) -> FilterResult:
        context = context or {}
        symbol = context.get("symbol")
        evaluation_date = context.get("evaluation_date")

        excluded_symbols = set(self.config.get("excluded_symbols", []))
        if symbol in excluded_symbols:
            return self._result(False, f"{symbol} is on the excluded_symbols list")

        exclusion_days = self.config["exclusion_days"]
        event_dates_by_symbol = self.config.get("event_dates", {}) or {}
        event_dates = event_dates_by_symbol.get(symbol, [])

        if evaluation_date is not None and event_dates:
            eval_date = (
                evaluation_date
                if isinstance(evaluation_date, date)
                else pd.Timestamp(evaluation_date).date()
            )
            for raw_event_date in event_dates:
                event_date = datetime.strptime(raw_event_date, "%Y-%m-%d").date()
                days_away = abs((event_date - eval_date).days)
                if days_away <= exclusion_days:
                    return self._result(
                        False,
                        f"known event on {event_date.isoformat()} is "
                        f"{days_away}d from evaluation date (exclusion "
                        f"window {exclusion_days}d)",
                    )

        return self._result(
            True, f"no excluded event within {exclusion_days}d of evaluation date"
        )


# Maps the names used in settings.yaml's `filters` section to the Filter
# subclass that implements them. This is the only place that needs to
# change when a new filter is added to the library.
FILTER_REGISTRY = {
    LiquidityFilter.name: LiquidityFilter,
    PriceRangeFilter.name: PriceRangeFilter,
    TrendFilter.name: TrendFilter,
    MomentumFilter.name: MomentumFilter,
    ProximityFilter.name: ProximityFilter,
    VolatilityFilter.name: VolatilityFilter,
    VolumeSurgeFilter.name: VolumeSurgeFilter,
    GapFilter.name: GapFilter,
}


def build_enabled_filters(filters_config: dict) -> list[Filter]:
    """Construct one Filter instance for every entry in settings.yaml's
    `filters` section that has `enabled: true`, in the order they appear in
    the file. Each filter is independently switched on/off and configured
    entirely by its own block — no separate master list to keep in sync.
    """
    filters: list[Filter] = []
    for name, block in filters_config.items():
        if not isinstance(block, dict) or not block.get("enabled", False):
            continue
        if name not in FILTER_REGISTRY:
            raise ValueError(
                f"Unknown filter '{name}' in settings.yaml's filters "
                f"section — no matching class in FILTER_REGISTRY"
            )
        filter_cls = FILTER_REGISTRY[name]
        filters.append(filter_cls(block))
    return filters
