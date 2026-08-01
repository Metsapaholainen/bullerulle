"""Legout's TC2000 momentum/technical scans (https://legout.github.io/), translated
into pandas from their exact published PCF formulas.

This is intentionally separate from the setups/ package: these are fixed,
single-day threshold scans (no window-size sweeping, no shape_match/pivot
concept), not tunable chart patterns, so they don't fit the SETUP_REGISTRY
architecture used for Breakout/EP/O'Neil patterns.

Data-availability notes (documented per-scan below):
  - Fundamental fields TC2000 exposes (market_cap bounds, float_shares,
    held_percent_insiders, revenue_growth, earnings_estimate_p1y_growth,
    ibd_industry_rs_rank, ibd_sect_rs_rank) are NOT in this app's price/
    volume/RS data model. Clauses using them are skipped and noted; the
    price/volume/RS part of each formula is applied faithfully.
  - "TI" (Trend Intensity) is not publicly documented by TC2000 in a way
    that matches all three observed thresholds (0.9 floor, 1.05/1.08 bars).
    The most consistent interpretation across all three usages is a simple
    price/moving-average ratio: ti(n) = close / SMA(n). Documented here as
    a best-effort reconstruction, not an official TC2000 formula.
  - "SLOPE(n, X)" is approximated as the average per-bar change over the
    window: (X - X.shift(n)) / n -- a simplified stand-in for a true linear
    regression slope.
  - Scans needing fundamentals only (Insiders, Ben's Focus) or genuinely
    ambiguous compound logic (Stockbees Combo) are omitted entirely -- see
    SKIPPED_SCANS at the bottom.
  - "EP" / "EP with Growth" are omitted here since this app already has a
    full Episodic Pivot setup (setups/episodic_pivot.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _adr_pct(df: pd.DataFrame, n: int = 20) -> pd.Series:
    return (df["high"] / df["low"] - 1.0).rolling(n).mean() * 100.0


def _natr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prior_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prior_close).abs(), (df["low"] - prior_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean() / df["close"] * 100.0


def _slope(s: pd.Series, n: int) -> pd.Series:
    return (s - s.shift(n)) / n


def _ti(df: pd.DataFrame, n: int = 50) -> pd.Series:
    return df["close"] / _sma(df["close"], n)


def _pocket_pivot(df: pd.DataFrame, lookback: int = 10, ma_period: int = 10) -> pd.Series:
    up_day = df["close"] > df["close"].shift(1)
    down_volume = df["volume"].where(~up_day, 0.0)
    max_down_vol = down_volume.shift(1).rolling(lookback).max()
    return up_day & (df["volume"] > max_down_vol) & (df["close"] > _sma(df["close"], ma_period))


def _strong_close_upper(df: pd.DataFrame, frac: float = 0.7) -> pd.Series:
    """Close sits in the top `frac` of the day's range."""
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    return (df["close"] - df["low"]) > frac * rng


# ---------------------------------------------------------------------------
# Individual scans. Each returns (bool Series match, float Series sort_key).
# rs, when accepted, is this app's own RS Rating series (1-99 percentile) --
# used as a stand-in for ibd_rs_rank.
# ---------------------------------------------------------------------------

def biggest_gainers(df: pd.DataFrame, lookback_days: int, rs=None):
    close, volume = df["close"], df["volume"]
    cond = (
        (close > 4)
        & (_sma(volume, 50) > 250_000)
        & (close * volume > 2_000_000)
        & (_adr_pct(df, 20) > 4)
        & (_ti(df, 50) > 0.9)
    )
    return cond, close / close.rolling(lookback_days).min()


def five_day_gainers(df: pd.DataFrame, rs=None):
    close, volume = df["close"], df["volume"]
    c5 = close.shift(5)
    cond = (
        (close > 4) & (_sma(volume, 50) > 250_000) & (close * volume > 2_000_000)
        & (_adr_pct(df, 20) > 4) & (close / c5 > 1.2)
    )
    return cond, close / c5


def top_gainers(df: pd.DataFrame, rs=None):
    close, volume = df["close"], df["volume"]
    c1 = close.shift(1)
    vol_sma50 = _sma(volume, 50)
    cond = (close > 4) & (vol_sma50 > 200_000) & (vol_sma50 * close > 2_000_000) & (close / c1 > 1.02)
    return cond, close / c1 - 1.0


def top_gainers_from_open(df: pd.DataFrame, rs=None):
    close, open_, volume = df["close"], df["open"], df["volume"]
    c1 = close.shift(1)
    vol_sma50 = _sma(volume, 50)
    cond = (close > 4) & (vol_sma50 > 200_000) & (vol_sma50 * close > 2_000_000) & (close / c1 > 1.02)
    return cond, close / open_


def volume_gainers(df: pd.DataFrame, rs=None):
    close, volume = df["close"], df["volume"]
    c1 = close.shift(1)
    vol_sma50 = _sma(volume, 50)
    cond = (
        (close > 4) & (vol_sma50 > 200_000) & (vol_sma50 * close > 2_000_000)
        & (close / c1 > 1.0) & (volume / vol_sma50 > 2)
    )
    return cond, volume / vol_sma50


def w52_high(df: pd.DataFrame, rs=None):
    close, volume = df["close"], df["volume"]
    vol_sma50 = _sma(volume, 50)
    max252 = close.rolling(252).max()
    cond = (
        (close > 4) & (vol_sma50 > 200_000) & (vol_sma50 * close > 2_000_000)
        & (close >= max252 * 0.999) & (volume / vol_sma50 > 1)
    )
    return cond, volume / vol_sma50


def bens_power_of_3(df: pd.DataFrame, rs=None):
    """Price hugging its 10/21-EMA and 50-SMA within a tight +/-1.5% band, RS>=70."""
    close, volume = df["close"], df["volume"]
    vol_sma50 = _sma(volume, 50)
    ema10, ema21, sma50 = _ema(close, 10), _ema(close, 21), _sma(close, 50)
    cond = (
        (close > 4) & (vol_sma50 > 200_000) & (vol_sma50 * close > 2_000_000)
        & (close / ema10 > 0.985) & (close / ema10 < 1.015)
        & (close / ema21 > 0.985) & (close / ema21 < 1.015)
        & (close / sma50 > 0.985) & (close / sma50 < 1.015)
    )
    if rs is not None:
        cond = cond & (rs >= 70)
    return cond, (rs if rs is not None else close)


def bens_velocity(df: pd.DataFrame, rs=None):
    """float_shares<100M clause SKIPPED -- not in this app's data model."""
    close, volume = df["close"], df["volume"]
    c1 = close.shift(1)
    vol_sma50 = _sma(volume, 50)
    cond = (
        (close > 4) & (vol_sma50 > 200_000) & (vol_sma50 * close > 2_000_000)
        & (close / c1 > 1.03) & (volume / vol_sma50 > 1.3) & (_ti(df, 50) > 1.05)
    )
    if rs is not None:
        cond = cond & (rs > 70)
    return cond, (rs if rs is not None else close / c1)


def blake_davis_strength(df: pd.DataFrame, rs=None):
    """ibd_industry_rs_rank clause SKIPPED -- no industry classification data."""
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    c1 = close.shift(1)
    vol_sma50 = _sma(volume, 50)
    cond = (
        (close > 7) & (vol_sma50 > 200_000) & (vol_sma50 * close > 2_000_000)
        & _strong_close_upper(df, 0.7) & (close / c1 > 1.0)
    )
    if rs is not None:
        cond = cond & (rs > 70)
    return cond, (rs if rs is not None else close)


def rsnhbp(df: pd.DataFrame, lookback_days: int, rs=None):
    """'RS new high, base pattern': price AND RS both at (or near) an N-period high."""
    close, volume = df["close"], df["volume"]
    vol_sma50 = _sma(volume, 50)
    max_close = close.rolling(lookback_days).max()
    cond = (close >= max_close * 0.999) & (vol_sma50 > 200_000) & (vol_sma50 * close > 2_000_000)
    if rs is not None:
        rs_max = rs.rolling(lookback_days).max()
        cond = cond & (rs >= rs_max * 0.999)
    return cond, close / close.rolling(63).min()


def bradass_pocket_pivot(df: pd.DataFrame, rs=None):
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    c1 = close.shift(1)
    vol_sma50 = _sma(volume, 50)
    ema200 = _ema(close, 200)
    cond = (
        (close > 3) & (vol_sma50 > 200_000) & (vol_sma50 * close > 2_000_000)
        & (volume > vol_sma50) & (close / c1 >= 1.05) & (close > ema200)
        & _strong_close_upper(df, 0.5)
        & _pocket_pivot(df)
    )
    return cond, volume / vol_sma50


def stockbees_4pct_gainers(df: pd.DataFrame, rs=None):
    close, open_, volume = df["close"], df["open"], df["volume"]
    c1, c2 = close.shift(1), close.shift(2)
    o1 = open_.shift(1)
    vol_sma50 = _sma(volume, 50)
    cond = (
        (close > 4) & (close / c1 > 1.04)
        & ((c1 - o1) < (close - open_))
        & (c1 / c2 <= 1.02)
        & (vol_sma50 > 200_000) & (volume > volume.shift(1))
        & _strong_close_upper(df, 0.7)
    )
    return cond, close / c1 - 1.0


def stockbees_ants(df: pd.DataFrame, rs=None):
    """Quiet, tight (+/-1%) low-volatility grind with above-average trend strength."""
    close, volume = df["close"], df["volume"]
    c1 = close.shift(1)
    min3vol = volume.rolling(3).min()
    cond = (
        (close > 3) & (min3vol > 100_000) & (_ti(df, 50) > 1.04)
        & (close / c1 >= 0.99) & (close / c1 <= 1.01)
    )
    return cond, close / c1 - 1.0


def stockbees_breakout_base(df: pd.DataFrame, base_days: int, rs=None):
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    c1 = close.shift(1)
    vol_sma50 = _sma(volume, 50)
    min_base = close.rolling(base_days).min().shift(1)
    max_base = close.rolling(base_days).max().shift(1)
    cond = (
        (c1 / min_base <= 1.1) & (close / max_base >= 0.9)
        & (close / c1 >= 1.04) & (volume > 200_000) & (vol_sma50 * close > 2_000_000)
        & _strong_close_upper(df, 0.7)
    )
    return cond, close / close.rolling(base_days * 2).min()


def darvas_scan(df: pd.DataFrame, rs=None):
    """ibd_sect_rs_rank and 'with Growth' fundamental clauses SKIPPED."""
    close, volume = df["close"], df["volume"]
    vol_sma50 = _sma(volume, 50)
    max252, min252 = close.rolling(252).max(), close.rolling(252).min()
    cond = (
        (close > 4) & (close / max252 > 0.85) & (close / min252 >= 2)
        & (vol_sma50 > 200_000) & (vol_sma50 * close > 2_000_000)
        & _pocket_pivot(df)
    )
    return cond, close / min252


def high_trend_intensity(df: pd.DataFrame, rs=None):
    close, volume = df["close"], df["volume"]
    vol_sma50 = _sma(volume, 50)
    cond = (
        (close > 4) & (close < 20) & (vol_sma50 > 200_000) & (vol_sma50 * close > 2_000_000)
        & (_ti(df, 50) > 1.08) & (_adr_pct(df, 20) > 4)
    )
    return cond, close / close.rolling(126).min()


def htf_scan(df: pd.DataFrame, rs=None):
    """High Trend Factor: a mature, decelerating-volume, still-accelerating uptrend."""
    close, volume = df["close"], df["volume"]
    vol_sma20 = _sma(volume, 20)
    vol_sma50 = _sma(volume, 50)
    sma50, sma200 = _sma(close, 50), _sma(close, 200)
    natr14 = _natr(df, 14)
    min40 = close.rolling(40).min()
    roc60 = close / close.shift(60) - 1.0
    cond = (
        (close > 4) & (vol_sma20 > 250_000)
        & (_slope(vol_sma50, 10) < 0)
        & (close > sma50) & (sma50 > sma200) & (_slope(sma200, 10) > 0)
        & (close / min40 > 1.9) & (roc60 * 100 > 50)
        & (natr14 < 8) & (_slope(natr14, 10) < 0)
    )
    return cond, roc60


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# Each entry: label, source formula (as transcribed from legout.github.io),
# the scan function, optional fixed kwargs, whether it consumes the RS series,
# and default enabled/disabled state for the app's checklist UI.

LEGOUT_SCANS = {
    "biggest_gainers_1m": dict(
        label="Biggest Gainers (1 Month)", fn=biggest_gainers, kwargs={"lookback_days": 21},
        uses_rs=False, default_on=True,
        formula="close>4, avgvol(50)>250000, close*volume>2000000, adr(20)>4, ti(50)>.9; ranked by close/min(21)",
    ),
    "biggest_gainers_3m": dict(
        label="Biggest Gainers (3 Month)", fn=biggest_gainers, kwargs={"lookback_days": 63},
        uses_rs=False, default_on=False, formula="same as 1-Month, ranked by close/min(63)",
    ),
    "biggest_gainers_6m": dict(
        label="Biggest Gainers (6 Month)", fn=biggest_gainers, kwargs={"lookback_days": 126},
        uses_rs=False, default_on=False, formula="same as 1-Month, ranked by close/min(126)",
    ),
    "biggest_gainers_12m": dict(
        label="Biggest Gainers (12 Month)", fn=biggest_gainers, kwargs={"lookback_days": 252},
        uses_rs=False, default_on=False, formula="same as 1-Month, ranked by close/min(252)",
    ),
    "five_day_gainers": dict(
        label="5 Day Gainers", fn=five_day_gainers, kwargs={}, uses_rs=False, default_on=True,
        formula="close>4, avgvol(50)>250000, close*volume>2000000, adr(20)>4, close/close(5)>1.2",
    ),
    "top_gainers": dict(
        label="Top Gainers", fn=top_gainers, kwargs={}, uses_rs=False, default_on=True,
        formula="close>4, avgvol(50)>200000, avgvol(50)*close>2000000, close/close(1)>1.02",
    ),
    "top_gainers_from_open": dict(
        label="Top Gainers From Open", fn=top_gainers_from_open, kwargs={}, uses_rs=False, default_on=False,
        formula="same gate as Top Gainers; ranked by close/open",
    ),
    "volume_gainers": dict(
        label="Volume Gainers", fn=volume_gainers, kwargs={}, uses_rs=False, default_on=True,
        formula="close>4, avgvol(50)>200000, avgvol(50)*close>2000000, close/close(1)>1, volume/avgvol(50)>2",
    ),
    "w52_high": dict(
        label="52 Week High", fn=w52_high, kwargs={}, uses_rs=False, default_on=True,
        formula="close>4, avgvol(50)>200000, avgvol(50)*close>2000000, close>=max(252), volume/avgvol(50)>1",
    ),
    "bens_power_of_3": dict(
        label="Ben's Power of 3", fn=bens_power_of_3, kwargs={}, uses_rs=True, default_on=False,
        formula="close hugging ema(10)/ema(21)/sma(50) within +/-1.5%, RS>=70",
    ),
    "bens_velocity": dict(
        label="Ben's Velocity", fn=bens_velocity, kwargs={}, uses_rs=True, default_on=False,
        formula="close/close(1)>1.03, volume/avgvol(50)>1.3, ti(50)>1.05, RS>70 (float_shares<100M SKIPPED)",
    ),
    "blake_davis_strength": dict(
        label="Blake Davis Strength", fn=blake_davis_strength, kwargs={}, uses_rs=True, default_on=False,
        formula="close>7, strong upper-range close, close/close(1)>1, RS>70 (industry RS SKIPPED)",
    ),
    "rsnhbp_1m": dict(
        label="RS New High, Base Pattern (1M)", fn=rsnhbp, kwargs={"lookback_days": 21},
        uses_rs=True, default_on=False, formula="price AND RS both at ~21-day high",
    ),
    "rsnhbp_3m": dict(
        label="RS New High, Base Pattern (3M)", fn=rsnhbp, kwargs={"lookback_days": 63},
        uses_rs=True, default_on=False, formula="price AND RS both at ~63-day high",
    ),
    "rsnhbp_6m": dict(
        label="RS New High, Base Pattern (6M)", fn=rsnhbp, kwargs={"lookback_days": 126},
        uses_rs=True, default_on=False, formula="price AND RS both at ~126-day high",
    ),
    "rsnhbp_12m": dict(
        label="RS New High, Base Pattern (12M)", fn=rsnhbp, kwargs={"lookback_days": 252},
        uses_rs=True, default_on=False, formula="price AND RS both at ~252-day (52-week) high",
    ),
    "bradass_pocket_pivot": dict(
        label="Bradass Pocket Pivot (EOD)", fn=bradass_pocket_pivot, kwargs={}, uses_rs=False, default_on=True,
        formula="close>3, volume>avgvol(50), close/close(1)>=1.05, close>ema(200), strong close, pocket pivot",
    ),
    "stockbees_4pct_gainers": dict(
        label="Stockbees 4% Gainers", fn=stockbees_4pct_gainers, kwargs={}, uses_rs=False, default_on=True,
        formula="close>4, gain>4%, today's body > yesterday's, strong close, rising volume",
    ),
    "stockbees_ants": dict(
        label="Stockbees Ants (tight grind)", fn=stockbees_ants, kwargs={}, uses_rs=False, default_on=False,
        formula="close>3, min(vol,3)>100000, ti(50)>1.04, |close/close(1)-1|<=1%",
    ),
    "stockbees_breakout_3m_base": dict(
        label="Stockbees Breakout (3M Base)", fn=stockbees_breakout_base, kwargs={"base_days": 63},
        uses_rs=False, default_on=False, formula="breaking a ~3-month base on a >=4% gain, strong close",
    ),
    "stockbees_breakout_1m_base": dict(
        label="Stockbees Breakout (1M Base)", fn=stockbees_breakout_base, kwargs={"base_days": 21},
        uses_rs=False, default_on=False, formula="breaking a ~1-month base on a >=4% gain, strong close",
    ),
    "darvas_scan": dict(
        label="Darvas Scan", fn=darvas_scan, kwargs={}, uses_rs=False, default_on=False,
        formula="within 15% of 52w high, 2x+ off 52w low, pocket pivot (sector RS/growth SKIPPED)",
    ),
    "high_trend_intensity": dict(
        label="High Trend Intensity", fn=high_trend_intensity, kwargs={}, uses_rs=False, default_on=False,
        formula="4<close<20, ti(50)>1.08, adr(20)>4",
    ),
    "htf_scan": dict(
        label="High Trend Factor (HTF)", fn=htf_scan, kwargs={}, uses_rs=False, default_on=False,
        formula="close>sma50>sma200 rising, close/min(40)>1.9, 60-day RoC>50%, decelerating volume & NATR",
    ),
}

# Explicitly not implemented, with reasons -- surfaced in the app so nothing
# silently vanishes from the original 32-scan list.
SKIPPED_SCANS = {
    "Ben's Focus": "needs revenue_growth / earnings_estimate_p1y_growth (fundamentals not fetched)",
    "Insiders": "needs held_percent_insiders / float_shares (fundamentals not fetched)",
    "EP (Episodic Pivot)": "already covered by this app's own Episodic Pivot setup",
    "EP with Growth": "same as EP, plus growth fundamentals (not fetched)",
    "Darvas with Growth": "same as Darvas Scan, plus growth fundamentals (not fetched)",
    "Stockbees Combo": "source formula mixes AND/OR clauses ambiguously -- skipped rather than guess",
    "RSNHBP (5 Day)": "source formula is identical to the 12-Month variant on legout's site (likely a copy/paste slip) -- not duplicated",
}


def scan_legout_signals_over_history(history: dict, scan_key: str, rs_panel: dict | None = None) -> dict:
    """For backtesting: run one Legout scan across *every* historical bar
    (not just the latest) for every symbol. Mirrors
    scanner.scan_signals_over_history()'s shape -- returns {symbol: DataFrame}
    with a boolean `match` column aligned to that symbol's OHLCV, ready to
    hand straight to backtest.engine.run_backtest(). Each scan function
    already computes its condition as a full boolean Series internally
    (run_legout_scans just discards everything but the last bar to build
    today's watchlist) -- this keeps the whole thing instead."""
    spec = LEGOUT_SCANS.get(scan_key)
    if spec is None:
        raise ValueError(f"Unknown Legout scan: {scan_key}")
    fn = spec["fn"]
    kwargs = dict(spec.get("kwargs", {}))
    out = {}
    for symbol, df in history.items():
        if df is None or df.empty or len(df) < 60:
            continue
        rs = rs_panel.get(symbol) if (rs_panel and spec["uses_rs"]) else None
        try:
            if spec["uses_rs"]:
                match, _sort_key = fn(df, rs=rs, **kwargs)
            else:
                match, _sort_key = fn(df, **kwargs)
        except Exception:
            continue
        enriched = df.copy()
        enriched["match"] = match.reindex(df.index).fillna(False)
        out[symbol] = enriched
    return out


def run_legout_scans(history: dict, enabled_keys: list, rs_panel: dict | None = None) -> pd.DataFrame:
    """history: {symbol: OHLCV DataFrame}. rs_panel: optional {symbol: RS-rating Series}.
    Returns a DataFrame of the latest-bar matches across all enabled scans, one row per (symbol, scan)."""
    rows = []
    for key in enabled_keys:
        spec = LEGOUT_SCANS.get(key)
        if spec is None:
            continue
        fn = spec["fn"]
        kwargs = dict(spec.get("kwargs", {}))
        for symbol, df in history.items():
            if df is None or len(df) < 60:
                continue
            rs = rs_panel.get(symbol) if (rs_panel and spec["uses_rs"]) else None
            try:
                if spec["uses_rs"]:
                    match, sort_key = fn(df, rs=rs, **kwargs)
                else:
                    match, sort_key = fn(df, **kwargs)
            except Exception:
                continue
            if len(match) == 0 or not bool(match.iloc[-1]):
                continue
            rows.append({
                "symbol": symbol,
                "scan": spec["label"],
                "scan_key": key,
                "sort_key": float(sort_key.iloc[-1]) if pd.notna(sort_key.iloc[-1]) else np.nan,
                "close": float(df["close"].iloc[-1]),
                "date": df.index[-1],
            })
    if not rows:
        return pd.DataFrame(columns=["symbol", "scan", "scan_key", "sort_key", "close", "date"])
    result = pd.DataFrame(rows)
    return result.sort_values(["scan", "sort_key"], ascending=[True, False]).reset_index(drop=True)
