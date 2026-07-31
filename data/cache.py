"""Local on-disk cache of daily OHLCV bars, one parquet file per symbol.

This is what makes the Designer/Parameter Finder interactive: raw price
history is fetched from FMP once per symbol/date-range and reused across
every parameter tweak afterwards, since indicator/backtest recomputation
never needs to touch the network again.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from data.fmp_client import FMPClient

CACHE_DIR = Path(__file__).parent.parent / "cache"

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol.upper()}.parquet"


def load_cached(symbol: str) -> Optional[pd.DataFrame]:
    path = _cache_path(symbol)
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            return None
    return None


def save_cache(symbol: str, df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_cache_path(symbol))


def _raw_to_df(raw: list) -> pd.DataFrame:
    if not raw:
        empty = pd.DataFrame(columns=OHLCV_COLUMNS)
        empty.index = pd.DatetimeIndex([], name="date")
        return empty
    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    cols = [c for c in OHLCV_COLUMNS if c in df.columns]
    return df[cols].astype(float)


def get_history(
    client: FMPClient,
    symbol: str,
    start: str,
    end: str,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return daily OHLCV for `symbol` covering [start, end], fetching from
    FMP only if the local cache doesn't already cover that range."""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    cached = None if force_refresh else load_cached(symbol)

    needs_fetch = True
    if cached is not None and not cached.empty:
        if cached.index.min() <= start_ts and cached.index.max() >= end_ts:
            needs_fetch = False

    if needs_fetch:
        raw = client.historical_eod_full(symbol, from_date=start, to_date=end)
        fresh = _raw_to_df(raw)
        if cached is not None and not cached.empty:
            combined = pd.concat([cached, fresh])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        else:
            combined = fresh
        save_cache(symbol, combined)
        df = combined
    else:
        df = cached

    mask = (df.index >= start_ts) & (df.index <= end_ts)
    return df.loc[mask].copy()


def get_history_bulk(
    client: FMPClient,
    symbols: list,
    start: str,
    end: str,
    force_refresh: bool = False,
    on_progress=None,
) -> dict:
    """Fetch/cache history for many symbols. Returns {symbol: DataFrame},
    skipping symbols with no usable data."""
    out = {}
    for i, sym in enumerate(symbols):
        try:
            df = get_history(client, sym, start, end, force_refresh=force_refresh)
            if not df.empty:
                out[sym] = df
        except Exception:
            pass
        if on_progress:
            on_progress(i + 1, len(symbols), sym)
    return out


def apply_live_quotes(history: dict, quotes: list) -> dict:
    """Merge FMP /quote snapshots into cached daily history as a synthetic
    "today" bar (open/dayHigh/dayLow/price/volume), replacing any existing
    row for today so repeated calls during the session update in place
    rather than duplicate. Used by Live Mode -- one lightweight quote call
    per refresh instead of re-pulling full daily history.

    Symbols not present in `history` (not previously loaded) are skipped;
    Live Mode only refreshes the current bar for already-loaded symbols."""
    today = pd.Timestamp.today().normalize()
    for q in quotes:
        symbol = q.get("symbol")
        if not symbol or symbol not in history:
            continue
        price = q.get("price")
        if price is None:
            continue
        row = {
            "open": q.get("open", price),
            "high": q.get("dayHigh", price),
            "low": q.get("dayLow", price),
            "close": price,
            "volume": q.get("volume", 0) or 0,
        }
        df = history[symbol]
        for col, val in row.items():
            df.loc[today, col] = val
        history[symbol] = df.sort_index()
    return history
