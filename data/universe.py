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
    coarse universe filters, capped at settings.max_symbols."""
    # Rough share-volume floor derived from the dollar-volume floor, using
    # the price floor as a conservative divisor -- refined precisely later.
    approx_volume_floor = None
    if settings.min_avg_dollar_volume and settings.min_price:
        approx_volume_floor = settings.min_avg_dollar_volume / max(settings.min_price, 1.0)

    base_filters = {
        "priceMoreThan": settings.min_price,
        "marketCapMoreThan": settings.min_market_cap,
        "marketCapLowerThan": settings.max_market_cap or None,
        "volumeMoreThan": approx_volume_floor,
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
