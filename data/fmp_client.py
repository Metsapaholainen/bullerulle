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
}


class FMPError(RuntimeError):
    pass


class FMPClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        backoff_seconds: float = 1.5,
    ):
        self.api_key = api_key or os.environ.get("FMP_API_KEY")
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
