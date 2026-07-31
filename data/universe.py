"""Builds the tradeable stock universe via FMP's company screener.

The screener call is a coarse, cheap pre-filter (price / market cap /
exchange / not-ETF / not-fund) so we don't waste API calls or cache space on
illiquid junk. Precise average-*dollar*-volume filtering (FMP's screener
only filters on raw share volume) happens later in scanner.py once real
OHLCV history is cached, where it can be computed exactly as a rolling mean.
"""
from __future__ import annotations

from data.fmp_client import FMPClient
from settings import UniverseSettings


def build_universe(client: FMPClient, settings: UniverseSettings) -> list:
    """Return a sorted, de-duplicated list of ticker symbols matching the
    coarse universe filters, capped at settings.max_symbols.

    Deliberately does NOT try to approximate the dollar-volume liquidity
    floor here: FMP's screener only filters on raw *share* volume, and
    share count varies wildly with price (a $2 stock and a $200 stock can
    have the same dollar volume with a 100x difference in share count), so
    any fixed conversion is wrong for most of the universe. The precise
    dollar-volume filter is applied for real in scanner.py, after actual
    price/volume history is fetched -- see filter_by_liquidity()."""
    base_filters = {
        "priceMoreThan": settings.min_price,
        "marketCapMoreThan": settings.min_market_cap,
        "marketCapLowerThan": settings.max_market_cap or None,
        "isEtf": False if settings.exclude_etf else None,
        "isFund": False if settings.exclude_funds else None,
        "isActivelyTrading": True,
        "limit": settings.max_symbols,
    }

    seen = {}
    exchanges = settings.exchanges or [None]
    for exch in exchanges:
        filters = dict(base_filters)
        if exch:
            filters["exchange"] = exch
        rows = client.screen_stocks(**filters)
        for row in rows:
            symbol = row.get("symbol")
            if symbol:
                seen[symbol] = row

    symbols = sorted(seen.keys())
    return symbols[: settings.max_symbols]
