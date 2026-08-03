"""Thin wrapper around the Financial Modeling Prep (FMP) 'stable' REST API.

Only the handful of endpoints this project needs are wrapped here. If FMP
ever renames a path, fix it in ENDPOINTS below -- nothing else in the
codebase needs to change.
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://financialmodelingprep.com/stable"

ENDPOINTS = {
    "company_screener": "/company-screener",
    "historical_price_eod_full": "/historical-price-eod/full",
    "quote": "/quote",
    "profile": "/profile",
    "income_statement_growth": "/income-statement-growth",
    "income_statement": "/income-statement",
    "shares_float": "/shares-float",
    "earnings_calendar": "/earnings-calendar",
    "earnings_company": "/earnings",
    "insider_trade_statistics": "/insider-trading/statistics",
}


class FMPError(RuntimeError):
    pass


def _resolve_api_key(explicit: Optional[str]) -> Optional[str]:
    """Explicit arg > Streamlit Cloud secrets > local .env/os.environ.

    The `st.secrets` check is wrapped defensively: importing streamlit is
    always safe (it's a hard dependency of this project), but *reading*
    `st.secrets` raises if no secrets.toml exists (the normal case for local
    dev), so any exception there just falls through to the .env path."""
    if explicit:
        return explicit
    try:
        import streamlit as st

        if hasattr(st, "secrets") and "FMP_API_KEY" in st.secrets:
            return st.secrets["FMP_API_KEY"]
    except Exception:
        pass
    return os.environ.get("FMP_API_KEY")


class FMPClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        backoff_seconds: float = 1.5,
    ):
        self.api_key = _resolve_api_key(api_key)
        if not self.api_key:
            raise FMPError(
                "No FMP API key found. Set FMP_API_KEY in your environment or .env file "
                "(see .env.example)."
            )
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.session = requests.Session()

    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        params = dict(params or {})
        params["apikey"] = self.api_key
        url = f"{BASE_URL}{path}"
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code == 429:
                    time.sleep(self.backoff_seconds * (attempt + 1))
                    continue
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, dict) and "Error Message" in data:
                    raise FMPError(data["Error Message"])
                return data
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(self.backoff_seconds * (attempt + 1))
        raise FMPError(f"FMP request to {path} failed after {self.max_retries} attempts: {last_exc}")

    def screen_stocks(self, **filters: Any) -> list:
        """Wraps /company-screener. Pass FMP filter kwargs directly,
        e.g. marketCapMoreThan=5e7, priceMoreThan=1, exchange='NASDAQ'."""
        clean = {k: v for k, v in filters.items() if v is not None}
        data = self._get(ENDPOINTS["company_screener"], params=clean)
        return data if isinstance(data, list) else []

    def historical_eod_full(
        self, symbol: str, from_date: Optional[str] = None, to_date: Optional[str] = None
    ) -> list:
        params: dict = {"symbol": symbol}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        data = self._get(ENDPOINTS["historical_price_eod_full"], params=params)
        if isinstance(data, dict) and "historical" in data:
            return data["historical"]
        return data if isinstance(data, list) else []

    def get_quotes(self, symbols: list) -> list:
        """Wraps /quote for one or many symbols (comma-joined) -- used by
        Live Mode to get each symbol's current price/day-high/day-low/volume
        without pulling full intraday bars."""
        if not symbols:
            return []
        data = self._get(ENDPOINTS["quote"], params={"symbol": ",".join(symbols)})
        return data if isinstance(data, list) else [data] if isinstance(data, dict) else []

    def profile(self, symbol: str) -> dict:
        """Wraps /profile -- one company per symbol, carrying `companyName`,
        `sector`, and `marketCap` in a single call (used by the fundamentals
        cache instead of /quote so sector comes along for free)."""
        data = self._get(ENDPOINTS["profile"], params={"symbol": symbol})
        if isinstance(data, list) and data:
            return data[0]
        return data if isinstance(data, dict) else {}

    def income_statement_growth(self, symbol: str, period: str = "quarter", limit: int = 1) -> list:
        """Wraps /income-statement-growth -- fields of interest are
        `growthRevenue`/`growthEPS` (decimal fractions, e.g. 0.11 = 11%)."""
        data = self._get(
            ENDPOINTS["income_statement_growth"], params={"symbol": symbol, "period": period, "limit": limit}
        )
        return data if isinstance(data, list) else []

    def income_statement(self, symbol: str, period: str = "quarter", limit: int = 8) -> list:
        """Wraps /income-statement -- raw (not growth-rate) `eps`/`revenue`
        per period, most-recent-first. Used for the log-scale earnings trend
        chart, where income_statement_growth's decimal growth rates aren't
        enough (need the actual EPS level to plot)."""
        data = self._get(
            ENDPOINTS["income_statement"], params={"symbol": symbol, "period": period, "limit": limit}
        )
        return data if isinstance(data, list) else []

    def shares_float(self, symbol: str) -> dict:
        """Wraps /shares-float -- `floatShares`/`freeFloat`/`outstandingShares`."""
        data = self._get(ENDPOINTS["shares_float"], params={"symbol": symbol})
        if isinstance(data, list) and data:
            return data[0]
        return data if isinstance(data, dict) else {}

    def earnings_calendar(self, from_date: str, to_date: str) -> list:
        """Wraps /earnings-calendar -- ONE shared date-range call across every
        US-listed ticker (no symbol filter exists on this endpoint), so
        callers must filter the result to their own symbol set themselves.
        Keep `from_date`/`to_date` narrow -- even a ~6-week window returns
        hundreds of KB across the whole market."""
        data = self._get(ENDPOINTS["earnings_calendar"], params={"from": from_date, "to": to_date})
        return data if isinstance(data, list) else []

    def earnings_surprise(self, symbol: str, limit: int = 4) -> list:
        """Wraps /earnings (per-symbol) -- actual-vs-estimate EPS/revenue per
        report date, e.g. {"date": ..., "epsActual": 1.87, "epsEstimated":
        1.76, "revenueActual": ..., "revenueEstimated": ...}. Rows for
        future/not-yet-reported dates have epsActual/revenueActual as None --
        callers must filter those out. Best-effort: pre-revenue or
        no-coverage symbols may return an empty list."""
        data = self._get(ENDPOINTS["earnings_company"], params={"symbol": symbol, "limit": limit})
        return data if isinstance(data, list) else []

    def insider_trade_statistics(self, symbol: str) -> list:
        """Wraps /insider-trade-statistics -- best-effort/optional; some FMP
        plan tiers may not include this, so callers should tolerate FMPError."""
        data = self._get(ENDPOINTS["insider_trade_statistics"], params={"symbol": symbol})
        return data if isinstance(data, list) else []
