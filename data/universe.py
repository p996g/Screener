"""
Defines and loads the ticker universe the screener runs against.

The universe is entirely config-driven: settings.yaml's `universe.source`
picks a strategy, and any parameters that strategy needs (e.g. a static
ticker list) live alongside it in `universe`. Adding a new source (say,
pulling live S&P 500 constituents from an index provider) means adding a
branch here — the rest of the codebase only ever calls `load_universe`.
"""


def load_universe(universe_config: dict) -> list[str]:
    """Resolve the list of ticker symbols to screen.

    Parameters
    ----------
    universe_config:
        The `universe` section of settings.yaml.

    Returns
    -------
    list[str]
        Uppercased, de-duplicated ticker symbols, in the order first seen.
    """
    source = universe_config.get("source")

    if source == "static_list":
        raw_tickers = universe_config.get("tickers", [])
        return _clean_tickers(raw_tickers)

    raise ValueError(
        f"Unsupported universe.source '{source}' — expected one of: "
        f"static_list"
    )


def _clean_tickers(raw_tickers: list[str]) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for raw in raw_tickers:
        ticker = raw.strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        cleaned.append(ticker)
    return cleaned
