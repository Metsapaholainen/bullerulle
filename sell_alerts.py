"""Sell-side watchlist: alert when a stock's daily close drops below its
N-day moving average (10-day by default, 20-day when configured per symbol)
-- a classic trailing-stop/sell trigger, the mirror image of this app's
buy-side setups.

The watchlist itself is a small committed YAML file (config/sell_watchlist.yaml)
rather than a per-user session list, so the same file can be read by both the
interactive Streamlit tab and the scheduled CLI/GitHub Actions job that
provides the "alert me even if I never open the app" path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

import github_store

GITHUB_STORE_PATH = "config/sell_watchlist.yaml"


def load_watchlist(path: Optional[Path] = None) -> list:
    """Returns [{"symbol": str, "ma_period": int}, ...], or [] if nothing is
    stored yet. With no explicit `path` (normal app usage), reads through
    github_store.py -- the GitHub data-store branch when a token is
    configured, otherwise a local fallback file -- so the watchlist
    survives a Streamlit Cloud restart instead of living only on the
    container's ephemeral disk. Pass an explicit `path` to read a specific
    local file instead (used by tests)."""
    if path is not None:
        if not path.exists():
            return []
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
    else:
        raw = github_store.read_file(GITHUB_STORE_PATH)
        data = (yaml.safe_load(raw) or {}) if raw else {}
    entries = data.get("watchlist") or []
    return [{"symbol": str(e["symbol"]).upper(), "ma_period": int(e.get("ma_period", 10))} for e in entries]


def save_watchlist(entries: list, path: Optional[Path] = None) -> None:
    normalized = [{"symbol": e["symbol"].upper(), "ma_period": int(e["ma_period"])} for e in entries]
    content = yaml.safe_dump({"watchlist": normalized}, sort_keys=False)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return
    github_store.write_file(GITHUB_STORE_PATH, content, "Update sell_watchlist.yaml")


def check_watchlist(history: dict, watchlist: list) -> pd.DataFrame:
    """For each watchlist entry, compare the latest close to the trailing
    `ma_period`-day SMA -- the same close.rolling(n).mean().iloc[-1] idiom
    scanner.market_direction() uses for the 50/200 and 10/20-day reads.

    Returns one row per entry: symbol, ma_period, date, close, sma,
    pct_from_ma, is_below (True = sell trigger). `is_below` is None when
    there isn't enough history yet to compute the MA."""
    rows = []
    for entry in watchlist:
        symbol, ma_period = entry["symbol"], entry["ma_period"]
        df = history.get(symbol)
        if df is None or df.empty:
            rows.append(
                {"symbol": symbol, "ma_period": ma_period, "date": None,
                 "close": None, "sma": None, "pct_from_ma": None, "is_below": None}
            )
            continue

        close = df["close"].dropna()
        last_close = float(close.iloc[-1])
        last_date = close.index[-1]
        sma = close.rolling(ma_period).mean().iloc[-1]

        if pd.isna(sma):
            rows.append(
                {"symbol": symbol, "ma_period": ma_period, "date": last_date,
                 "close": last_close, "sma": None, "pct_from_ma": None, "is_below": None}
            )
            continue

        sma = float(sma)
        pct_from_ma = (last_close / sma - 1.0) * 100.0
        rows.append(
            {
                "symbol": symbol,
                "ma_period": ma_period,
                "date": last_date,
                "close": last_close,
                "sma": sma,
                "pct_from_ma": pct_from_ma,
                "is_below": last_close < sma,
            }
        )
    return pd.DataFrame(rows)
