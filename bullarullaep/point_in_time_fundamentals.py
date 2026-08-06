"""Point-in-time-correct fundamentals -- the fix for a real look-ahead-bias
risk in the parent project's `data/fundamentals_cache.py`, whose own module
docstring already flags this: it stores only "most recently known" growth/
EPS-beat, not what was actually knowable ON THE SIGNAL DATE. A backtest
scored against "most recently known" data implicitly lets a trade on date D
see quarters reported after D -- exactly the "free clairvoyance" that
wrecks fundamental backtests in practice.

The fix: store each symbol's quarterly (report_date, eps_actual,
eps_estimated, revenue_actual, revenue_estimated) history, and expose
`fundamentals_as_of(quarters, as_of)`, which filters to `report_date <=
as_of` before computing anything -- so live scanning and backtesting both
go through the exact same function and can never see data that wasn't
public yet.

Built from `earnings_surprise()` alone (NOT joined with `income_statement()`,
the parent project's usual raw-EPS source) -- verified live against NVDA
that the two use DIFFERENT date conventions: `income_statement()`'s `date`
is the quarter's PERIOD-END date, while `earnings_surprise()`'s `date` is
the actual report/announcement date, ~3-4 weeks later, and the two never
match on date equality. `earnings_surprise()` alone already carries actual
AND estimated EPS/revenue for a quarter alongside its real report date, so
it's sufficient by itself and avoids a join that would silently never find
a match.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from data.fmp_client import FMPClient, FMPError

PIT_CACHE_DIR = Path(__file__).parent.parent / "cache" / "ep_pit_fundamentals"
DEFAULT_TTL_SECONDS = 7 * 24 * 3600  # quarterly data changes quarterly -- 1 week is plenty fresh


def _cache_path(symbol: str) -> Path:
    return PIT_CACHE_DIR / f"{symbol.upper()}.json"


def _load_cached(symbol: str) -> Optional[dict]:
    path = _cache_path(symbol)
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache(symbol: str, quarters: list) -> None:
    PIT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_cache_path(symbol), "w") as f:
        json.dump({"fetched_at": time.time(), "quarters": quarters}, f)


def _fetch_quarterly_history(client: FMPClient, symbol: str, limit: int = 12) -> list:
    try:
        rows = client.earnings_surprise(symbol, limit=limit)
    except FMPError:
        rows = []
    quarters = []
    for row in rows:
        report_date = row.get("date")
        eps_actual = row.get("epsActual")
        # Future/not-yet-reported quarters have epsActual=None -- exclude
        # entirely; there's nothing point-in-time-useful about a quarter
        # that hasn't happened yet.
        if not report_date or eps_actual is None:
            continue
        quarters.append({
            "report_date": report_date,
            "eps_actual": eps_actual,
            "revenue_actual": row.get("revenueActual"),
            "eps_estimated": row.get("epsEstimated"),
            "revenue_estimated": row.get("revenueEstimated"),
        })
    quarters.sort(key=lambda q: q["report_date"], reverse=True)
    return quarters


def get_quarterly_history(
    client: FMPClient, symbol: str, ttl_seconds: float = DEFAULT_TTL_SECONDS, force_refresh: bool = False
) -> list:
    """Cached quarterly (report_date, eps_actual, revenue_actual,
    eps_estimated, revenue_estimated) records, most-recent-report-first."""
    cached = None if force_refresh else _load_cached(symbol)
    if cached is not None and (time.time() - cached.get("fetched_at", 0)) < ttl_seconds:
        return cached.get("quarters", [])
    quarters = _fetch_quarterly_history(client, symbol)
    _save_cache(symbol, quarters)
    return quarters


def get_quarterly_history_bulk(
    client: FMPClient, symbols: list, ttl_seconds: float = DEFAULT_TTL_SECONDS, on_progress=None
) -> dict:
    """{symbol: quarters_list}, tolerant of individual symbol failures --
    mirrors data/fundamentals_cache.py::get_fundamentals_bulk's convention."""
    out = {}
    for i, sym in enumerate(symbols):
        try:
            out[sym] = get_quarterly_history(client, sym, ttl_seconds=ttl_seconds)
        except Exception:
            out[sym] = []
        if on_progress:
            on_progress(i + 1, len(symbols), sym)
    return out


def fundamentals_as_of(quarters: list, as_of) -> dict:
    """The core look-ahead-bias fix: given the full quarterly history from
    get_quarterly_history(), return growth/EPS-beat computed using ONLY
    quarters whose report_date is on or before `as_of` -- a quarter reported
    the day after as_of is invisible, exactly as it would have been to a
    real trader on that date.

    Returns a dict shaped to match data/fundamentals_cache.py's
    revenue_growth_pct/eps_growth_pct/eps_beat_pct/eps_beat_date fields, so
    it's a drop-in for anything already consuming that shape (e.g.
    setups/episodic_pivot.py's scan(), or bullarullaep/detector.py)."""
    as_of_ts = pd.Timestamp(as_of)
    known = sorted(
        (q for q in quarters if q.get("report_date") and pd.Timestamp(q["report_date"]) <= as_of_ts),
        key=lambda q: q["report_date"], reverse=True,
    )

    out = {
        "revenue_growth_pct": None, "eps_growth_pct": None,
        "eps_beat_pct": None, "eps_beat_date": None,
        "quarters_known_as_of": len(known),
    }
    if len(known) >= 5:
        # Same YoY convention as the parent project's fundamentals_cache.py:
        # quarter vs. the same quarter 4 back, computed directly from raw
        # actuals rather than FMP's own (QoQ) growth endpoint.
        current, year_ago = known[0], known[4]
        cur_rev, prior_rev = current.get("revenue_actual"), year_ago.get("revenue_actual")
        cur_eps, prior_eps = current.get("eps_actual"), year_ago.get("eps_actual")
        if cur_rev is not None and prior_rev:
            out["revenue_growth_pct"] = round((cur_rev - prior_rev) / abs(prior_rev) * 100.0, 2)
        if cur_eps is not None and prior_eps:
            out["eps_growth_pct"] = round((cur_eps - prior_eps) / abs(prior_eps) * 100.0, 2)

    if known:
        latest = known[0]
        eps_actual, eps_estimated = latest.get("eps_actual"), latest.get("eps_estimated")
        if eps_actual is not None and eps_estimated:
            out["eps_beat_pct"] = round((eps_actual - eps_estimated) / abs(eps_estimated) * 100.0, 2)
            out["eps_beat_date"] = latest.get("report_date")

    return out
