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


def add_sma_slope(df: pd.DataFrame, period: int = 50, lookback: int = 20) -> pd.DataFrame:
    """% change in the `period`-day SMA over the last `lookback` days.

    "Above the 50-day" and "the 50-day is rising" are different claims: a
    stock can sit above a rolling-over 50-day for weeks on the way down.
    Qullamaggie's rule is to surf *rising* averages, so the quality score
    wants the slope, not just the position."""
    df = df.copy()
    sma_col = f"sma_{period}"
    if sma_col not in df.columns:
        df[sma_col] = df["close"].rolling(period).mean()
    prior = df[sma_col].shift(lookback)
    df[f"sma_{period}_slope_pct"] = (df[sma_col] - prior) / prior * 100.0
    return df


def add_volume_dryup(df: pd.DataFrame, base_days: int, avg_period: int = 50) -> pd.DataFrame:
    """Ratio of the base window's average volume to the longer-run average.

    "Volume drying up into the pivot" is one of the few genuinely
    predictive parts of a base -- sellers are exhausted, so it takes less
    buying to move price. Below ~0.9 means the base is quieter than the
    stock's own norm. The window excludes today (`.shift(1)`) so a breakout
    day's own volume surge can't mask the dry-up that preceded it."""
    df = df.copy()
    avg_col = f"vol_avg_{avg_period}"
    if avg_col not in df.columns:
        df = add_volume_stats(df, avg_period=avg_period)
    base_avg_vol = df["volume"].shift(1).rolling(base_days).mean()
    df[f"vol_dryup_{base_days}"] = base_avg_vol / df[avg_col]
    return df


def add_efficiency_ratio(df: pd.DataFrame, window: int, col_name: str = None) -> pd.DataFrame:
    """Kaufman efficiency ratio: |net change| / sum(|daily changes|) over
    `window` days, in [0, 1].

    This is the quantitative form of what Qullamaggie calls "linearity" and
    what he says you can only "train your eye on" -- 1.0 is a perfectly
    straight move, 0.1 is the same net distance travelled via chop. Two
    bases with identical high-low ranges score very differently here, which
    is exactly the distinction a range-based tightness filter can't make."""
    df = df.copy()
    net_change = (df["close"] - df["close"].shift(window)).abs()
    path_length = df["close"].diff().abs().rolling(window).sum()
    name = col_name or f"efficiency_ratio_{window}"
    df[name] = (net_change / path_length).replace([float("inf"), float("-inf")], pd.NA)
    return df


def rs_line_at_high(close: pd.Series, benchmark_close: pd.Series, window: int = 63) -> pd.Series:
    """Bool series: is the stock/benchmark ratio at a `window`-day high?

    The RS *line* making a new high while price is still building a base is
    the classic "leader" tell -- it means the stock outperformed even while
    consolidating. Distinct from the RS *rating*, which is a cross-sectional
    percentile at a point in time; this is a within-symbol time series.
    Returns all-False when the benchmark isn't loaded rather than raising,
    so callers don't need to special-case a missing SPY."""
    if benchmark_close is None or benchmark_close.empty:
        return pd.Series(False, index=close.index)
    bench = benchmark_close.reindex(close.index).ffill(limit=5)
    rs_line = close / bench
    rolling_max = rs_line.rolling(window, min_periods=max(2, window // 4)).max()
    return (rs_line >= rolling_max) & rs_line.notna()


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
