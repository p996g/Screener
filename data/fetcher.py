"""
Pulls OHLCV market data for the universe.

Data source: yfinance (free, no API key required). All behavior — lookback
window, interval, caching, timeouts — is driven by the `data` section of
settings.yaml; see build_fetcher() for how a config dict becomes a fetcher.

Point-in-time discipline: `fetch()` always drops any bar dated after
`evaluation_date` before returning, regardless of what the underlying
provider happened to return. This is the single choke point that keeps
look-ahead bias out of the rest of the pipeline — filters and ranking never
need to think about it again.

Failure handling: a network error or provider hiccup while downloading one
symbol is logged clearly here and turned into an empty result rather than
raised — the engine already treats an empty result as a clean, explainable
"no price data available" failure for that symbol (see engine.py), so a
single flaky download never crashes an entire scheduled run.
"""

import logging
import os
from datetime import date, datetime

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


class OHLCVFetcher:
    """Fetches (and optionally caches) daily OHLCV history per symbol."""

    def __init__(self, data_config: dict):
        self.provider = data_config["provider"]
        self.interval = data_config["interval"]
        self.lookback_days = data_config["lookback_days"]
        self.use_cache = data_config.get("use_cache", False)
        self.cache_dir = data_config.get("cache_dir")
        self.request_timeout_seconds = data_config.get(
            "request_timeout_seconds", 30
        )

        if self.provider != "yfinance":
            raise ValueError(
                f"Unsupported data.provider '{self.provider}' — "
                f"expected 'yfinance'"
            )

        if self.use_cache and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def fetch(self, symbol: str, evaluation_date: date) -> pd.DataFrame:
        """Return OHLCV bars for `symbol`, truncated to `evaluation_date`.

        Returns an empty DataFrame (with the expected columns) if no data
        is available, rather than raising, so the engine can record a
        clean "no data" failure per symbol instead of crashing the run.
        """
        raw = self._load_from_cache(symbol)
        if raw is None:
            raw = self._download(symbol, evaluation_date)
            self._save_to_cache(symbol, raw)

        if raw.empty:
            return raw

        eval_ts = pd.Timestamp(evaluation_date)
        point_in_time = raw.loc[raw.index <= eval_ts].copy()
        return point_in_time

    def _download(self, symbol: str, evaluation_date: date) -> pd.DataFrame:
        start = pd.Timestamp(evaluation_date) - pd.Timedelta(
            days=self.lookback_days * 2  # buffer for weekends/holidays
        )
        end = pd.Timestamp(evaluation_date) + pd.Timedelta(days=1)  # yfinance end is exclusive

        try:
            raw = yf.download(
                symbol,
                start=start.date().isoformat(),
                end=end.date().isoformat(),
                interval=self.interval,
                progress=False,
                timeout=self.request_timeout_seconds,
                auto_adjust=False,
            )
        except Exception:
            logger.error(
                "Data fetch failed for %s (evaluation_date=%s); treating as "
                "no data available for this symbol.",
                symbol, evaluation_date, exc_info=True,
            )
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        if raw.empty:
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw = raw[OHLCV_COLUMNS].sort_index()
        return raw

    def _cache_path(self, symbol: str) -> str | None:
        if not (self.use_cache and self.cache_dir):
            return None
        return os.path.join(self.cache_dir, f"{symbol}.csv")

    def _load_from_cache(self, symbol: str) -> pd.DataFrame | None:
        path = self._cache_path(symbol)
        if not path or not os.path.exists(path):
            return None
        cached = pd.read_csv(path, index_col=0, parse_dates=True)
        # Refresh if the cache doesn't reach up to today — cheap staleness
        # check so a same-day rerun is fast but a new day pulls fresh data.
        if cached.empty or cached.index.max().date() < date.today():
            return None
        return cached

    def _save_to_cache(self, symbol: str, data: pd.DataFrame) -> None:
        path = self._cache_path(symbol)
        if not path or data.empty:
            return
        data.to_csv(path)


def build_fetcher(data_config: dict) -> OHLCVFetcher:
    return OHLCVFetcher(data_config)


def resolve_evaluation_date(data_config: dict) -> date:
    """Turn `data.evaluation_date` (null or "YYYY-MM-DD") into a concrete
    date. `null` means "as of today"."""
    raw = data_config.get("evaluation_date")
    if raw is None:
        return date.today()
    return datetime.strptime(raw, "%Y-%m-%d").date()
