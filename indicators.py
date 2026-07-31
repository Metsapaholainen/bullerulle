"""Vectorized pandas indicators shared by the setup detectors, the scanner,
and the backtester. Everything here operates on already-cached OHLCV data --
no network calls -- which is what keeps parameter changes fast/interactive.
"""
from __future__ import annotations

import pandas as pd


def add_moving_averages(df: pd.DataFrame, ema_periods=(10, 20), sma_periods=(50, 200)) -> pd.DataFrame:
    df = df.copy()
    for p in ema_periods:
        df[f"ema_{p}"] = df["close"].ewm(span=p, adjust=False).mean()
    for p in sma_periods:
        df[f"sma_{p}"] = df["close"].rolling(p).mean()
    return df


def add_adr_pct(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Average Daily Range %, Qullamaggie-style: mean(high/low - 1) * 100
    over the lookback window. Used both as a tradeability filter and as the
    unit for stop-loss / extension distances."""
    df = df.copy()
    daily_range_pct = (df["high"] / df["low"] - 1.0) * 100.0
    df[f"adr_pct_{lookback}"] = daily_range_pct.rolling(lookback).mean()
    return df


def add_volume_stats(df: pd.DataFrame, avg_period: int = 50) -> pd.DataFrame:
    df = df.copy()
    df[f"vol_avg_{avg_period}"] = df["volume"].rolling(avg_period).mean()
    df[f"vol_ratio_{avg_period}"] = df["volume"] / df[f"vol_avg_{avg_period}"]
    return df


def add_gap_pct(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    prev_close = df["close"].shift(1)
    df["gap_pct"] = (df["open"] - prev_close) / prev_close * 100.0
    return df


def add_rolling_range_pct(df: pd.DataFrame, window: int, col_name: str) -> pd.DataFrame:
    """(rolling max high - rolling min low) / rolling min low * 100 over
    `window` days -- used both to size a prior "run" and to measure how tight
    (or loose) a consolidation base is."""
    df = df.copy()
    roll_max_high = df["high"].rolling(window).max()
    roll_min_low = df["low"].rolling(window).min()
    df[col_name] = (roll_max_high - roll_min_low) / roll_min_low * 100.0
    return df


def add_consecutive_up_days(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    up = (df["close"] > df["close"].shift(1)).astype(int)
    grp = (up != up.shift(1)).cumsum()
    df["consecutive_up_days"] = up.groupby(grp).cumsum() * up
    return df


def add_core_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience: everything the three setups need, computed once."""
    df = add_moving_averages(df, ema_periods=(10, 20), sma_periods=(50, 200))
    df = add_adr_pct(df, lookback=20)
    df = add_volume_stats(df, avg_period=50)
    df = add_gap_pct(df)
    df = add_consecutive_up_days(df)
    return df


# --- Relative Strength (RS) rating, computed cross-sectionally across the universe ---

def build_close_panel(history: dict) -> pd.DataFrame:
    """history: {symbol: OHLCV DataFrame}. Returns a wide DataFrame of daily
    close prices, one column per symbol, aligned on the union of all dates
    and forward-filled (a symbol with no data yet/anymore is left as NaN so
    it's excluded from that date's percentile rank)."""
    closes = {sym: d["close"] for sym, d in history.items() if not d.empty}
    panel = pd.DataFrame(closes).sort_index()
    return panel.ffill(limit=5)  # tolerate short gaps (holidays/data hiccups), not long absences


def compute_weighted_return_panel(close_panel: pd.DataFrame, lookback_periods, period_weights) -> pd.DataFrame:
    total_weight = sum(period_weights) or 1.0
    score = pd.DataFrame(0.0, index=close_panel.index, columns=close_panel.columns)
    for period, weight in zip(lookback_periods, period_weights):
        ret = close_panel.pct_change(periods=period)
        score = score.add(ret * (weight / total_weight), fill_value=0.0)
    return score


def compute_rs_rating_panel(close_panel: pd.DataFrame, lookback_periods, period_weights) -> pd.DataFrame:
    """IBD-style percentile rank (1-99) of a weighted multi-period return,
    computed independently for every date so it's valid to use inside a
    historical backtest, not just for a live scan."""
    score = compute_weighted_return_panel(close_panel, lookback_periods, period_weights)
    pct_rank = score.rank(axis=1, pct=True, na_option="keep")
    return pct_rank * 98.0 + 1.0
