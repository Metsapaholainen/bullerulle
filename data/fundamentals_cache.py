"""Local on-disk cache of per-symbol fundamentals (growth, float, earnings
date), mirroring data/cache.py's OHLCV cache but TTL-based instead of
date-range-based -- fundamentals update quarterly at most, not daily, so
"is this fresh enough" is a simple age check rather than a coverage check.

Deliberately NOT threaded into load_scan_universe_history()/the main
universe fetch: fundamentals are only ever fetched for a small shortlist
(last scan's matches, momentum leaders, or a custom symbol list) from the
Scanner tab's Fundamentals panel, so the bulk OHLCV load stays exactly as
fast/cheap as it always has been.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from data.fmp_client import FMPClient, FMPError
from fundamentals_classifier import detect_earnings_deceleration

FUNDAMENTALS_CACHE_DIR = Path(__file__).parent.parent / "cache" / "fundamentals"
DEFAULT_TTL_SECONDS = 7 * 24 * 3600  # 1 week -- generous but safe for quarterly-updated data
EARNINGS_TTL_SECONDS = 24 * 3600  # earnings dates can shift, refresh daily


def _cache_path(symbol: str) -> Path:
    return FUNDAMENTALS_CACHE_DIR / f"{symbol.upper()}.json"


def _load_cached(symbol: str) -> Optional[dict]:
    path = _cache_path(symbol)
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache(symbol: str, data: dict) -> None:
    FUNDAMENTALS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_cache_path(symbol), "w") as f:
        json.dump({"fetched_at": time.time(), "data": data}, f)


def _fetch_fundamentals(client: FMPClient, symbol: str) -> dict:
    """Each underlying call is independently tolerant of failure -- a symbol
    missing one field (e.g. float unavailable for some tickers) still
    returns whatever else succeeded rather than failing the whole fetch."""
    out = {
        "revenue_growth_pct": None, "eps_growth_pct": None, "float_shares": None, "free_float_pct": None,
        "market_cap": None, "company_name": None, "sector": None,
        "eps_beat_pct": None, "eps_beat_date": None,
        "eps_growth_history": None, "eps_history": None,
        "earnings_decelerating": False, "earnings_deceleration_reason": None,
    }

    try:
        # Raw quarterly EPS/revenue -- also the basis for growth below.
        #
        # Growth here is computed as quarter-vs-same-quarter-one-year-ago
        # (YoY), the standard CANSLIM/Qullamaggie convention ("compared to
        # the same quarter last year") that the rest of this app assumes.
        # FMP's own /income-statement-growth endpoint was used here
        # previously and looked plausible, but is actually QoQ (this
        # quarter vs the immediately prior quarter) -- verified live
        # against NVDA: it reported 19.8% "growth" for a quarter whose
        # revenue was actually +85% vs the same quarter a year earlier, and
        # matched the QoQ figure exactly. Computing directly from quarter
        # vs quarter-4-back avoids that mismatch entirely.
        income = client.income_statement(symbol, period="quarter", limit=8)
        if income:
            out["eps_history"] = [
                {"date": r.get("date"), "eps": r.get("eps"), "revenue": r.get("revenue")} for r in income
            ]
            yoy_growth = []
            for i in range(len(income) - 4):
                cur, prior = income[i], income[i + 4]
                cur_rev, prior_rev = cur.get("revenue"), prior.get("revenue")
                cur_eps, prior_eps = cur.get("eps"), prior.get("eps")
                rev_g = (cur_rev - prior_rev) / abs(prior_rev) if cur_rev is not None and prior_rev else None
                eps_g = (cur_eps - prior_eps) / abs(prior_eps) if cur_eps is not None and prior_eps else None
                yoy_growth.append({"date": cur.get("date"), "growthEPS": eps_g, "growthRevenue": rev_g})
            if yoy_growth:
                out["eps_growth_history"] = yoy_growth
                latest = yoy_growth[0]
                out["revenue_growth_pct"] = (
                    round(latest["growthRevenue"] * 100.0, 2) if latest["growthRevenue"] is not None else None
                )
                out["eps_growth_pct"] = (
                    round(latest["growthEPS"] * 100.0, 2) if latest["growthEPS"] is not None else None
                )
    except FMPError:
        pass

    if out["eps_growth_history"]:
        # "Two consecutive quarters of major earnings deceleration" (O'Neil) --
        # informational only, never blocks a scan match; see PATTERN_EXPLANATIONS
        # / the Fundamentals panel for how this is surfaced.
        decel = detect_earnings_deceleration(out["eps_growth_history"])
        out["earnings_decelerating"] = decel["decelerating"]
        out["earnings_deceleration_reason"] = decel["reason"]

    try:
        # /profile carries market cap, full company name, and sector in one
        # call -- replaces the earlier /quote-based lookup (which had name
        # + market cap but no sector).
        profile = client.profile(symbol)
        if profile:
            out["market_cap"] = profile.get("marketCap")
            out["company_name"] = profile.get("companyName")
            out["sector"] = profile.get("sector")
    except FMPError:
        pass

    try:
        float_info = client.shares_float(symbol)
        if float_info:
            out["float_shares"] = float_info.get("floatShares")
            out["free_float_pct"] = float_info.get("freeFloat")
    except FMPError:
        pass

    try:
        insider = client.insider_trade_statistics(symbol)
        out["insider_acquired_disposed_ratio"] = insider[0].get("acquiredDisposedRatio") if insider else None
    except FMPError:
        out["insider_acquired_disposed_ratio"] = None

    try:
        # Most recent REPORTED quarter only -- rows for future/not-yet-
        # reported dates have epsActual/epsEstimated as None and must be
        # skipped; don't assume the API returns them in date order.
        rows = client.earnings_surprise(symbol, limit=4)
        reported = sorted(
            (r for r in rows if r.get("epsActual") is not None and r.get("epsEstimated")),
            key=lambda r: r.get("date", ""), reverse=True,
        )
        if reported:
            latest = reported[0]
            eps_actual, eps_estimated = latest["epsActual"], latest["epsEstimated"]
            out["eps_beat_pct"] = round((eps_actual - eps_estimated) / abs(eps_estimated) * 100.0, 2)
            out["eps_beat_date"] = latest.get("date")
    except FMPError:
        pass

    return out


REQUIRED_FIELDS = {
    "revenue_growth_pct", "eps_growth_pct", "float_shares", "free_float_pct",
    "market_cap", "company_name", "sector", "insider_acquired_disposed_ratio",
    "eps_beat_pct", "eps_beat_date",
    "eps_growth_history", "eps_history", "earnings_decelerating", "earnings_deceleration_reason",
}


def get_fundamentals(
    client: FMPClient, symbol: str, ttl_seconds: float = DEFAULT_TTL_SECONDS, force_refresh: bool = False
) -> dict:
    """Returns a dict with revenue_growth_pct/eps_growth_pct/float_shares/
    free_float_pct/insider_acquired_disposed_ratio/market_cap/company_name/
    sector -- any field can be None if that particular FMP call failed or
    the data wasn't available.

    A cached entry missing one of REQUIRED_FIELDS entirely (as opposed to
    present-but-None) is treated as stale regardless of its TTL and
    refetched -- guards against a cache file written by an older version of
    this module (e.g. before `sector` was added) silently serving an
    incomplete result forever, since the age check alone wouldn't catch it."""
    cached = None if force_refresh else _load_cached(symbol)
    if cached is not None and (time.time() - cached.get("fetched_at", 0)) < ttl_seconds:
        cached_data = cached.get("data", {})
        if REQUIRED_FIELDS <= set(cached_data.keys()):
            return cached_data

    data = _fetch_fundamentals(client, symbol)
    _save_cache(symbol, data)
    return data


def get_fundamentals_bulk(
    client: FMPClient, symbols: list, ttl_seconds: float = DEFAULT_TTL_SECONDS, on_progress=None
) -> dict:
    """{symbol: fundamentals_dict} for every symbol, skipping ones that
    error out entirely (shouldn't happen given get_fundamentals's internal
    tolerance, but mirrors data/cache.py's get_history_bulk defensive style)."""
    out = {}
    for i, sym in enumerate(symbols):
        try:
            out[sym] = get_fundamentals(client, sym, ttl_seconds=ttl_seconds)
        except Exception:
            out[sym] = {}
        if on_progress:
            on_progress(i + 1, len(symbols), sym)
    return out


def _earnings_calendar_cache_path(as_of: str, lookahead_days: int) -> Path:
    return FUNDAMENTALS_CACHE_DIR / f"_earnings_calendar_{as_of}_{lookahead_days}.json"


def get_earnings_dates(
    client: FMPClient, symbols: list, as_of: str, lookahead_days: int = 14, ttl_seconds: float = EARNINGS_TTL_SECONDS
) -> dict:
    """One shared bulk call to /earnings-calendar for [as_of, as_of+lookahead_days],
    cached and then filtered locally to just `symbols` -- keeps this at O(1)
    API calls regardless of how many symbols are being checked.

    Returns {symbol: "YYYY-MM-DD" or None}."""
    import pandas as pd

    path = _earnings_calendar_cache_path(as_of, lookahead_days)
    rows = None
    if path.exists():
        try:
            with open(path, "r") as f:
                cached = json.load(f)
            if (time.time() - cached.get("fetched_at", 0)) < ttl_seconds:
                rows = cached["rows"]
        except Exception:
            rows = None

    if rows is None:
        to_date = (pd.Timestamp(as_of) + pd.Timedelta(days=lookahead_days)).strftime("%Y-%m-%d")
        rows = client.earnings_calendar(from_date=str(as_of), to_date=to_date)
        FUNDAMENTALS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"fetched_at": time.time(), "rows": rows}, f)

    symbol_set = set(symbols)
    result = {sym: None for sym in symbols}
    for row in rows:
        sym = row.get("symbol")
        if sym in symbol_set and result.get(sym) is None:
            result[sym] = row.get("date")
    return result
