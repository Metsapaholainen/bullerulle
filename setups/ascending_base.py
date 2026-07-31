"""O'Neil 'Ascending Base' (How to Make Money in Stocks).

A staircase pattern: a series of pullbacks (commonly 3) over ~9-16 weeks,
each with a higher low and a higher high than the one before, each pullback
shallower than a typical correction (~10-20%). It's a choppier, less tidy
base than a cup or flat base, but the ascending structure itself signals
persistent demand. Breakout above the most recent high on volume confirms it.

The lookback window is split into three equal segments (oldest -> newest)
and each is checked for a higher low/high than the previous one.
"""
from __future__ import annotations

import pandas as pd

from indicators import add_volume_stats
from settings import AscendingBaseSettings


def scan(df: pd.DataFrame, settings: AscendingBaseSettings, rs_rating: pd.Series = None) -> pd.DataFrame:
    df = df.copy()

    vol_ratio_col = f"vol_ratio_{settings.volume_avg_period}"
    if vol_ratio_col not in df.columns:
        df = add_volume_stats(df, avg_period=settings.volume_avg_period)

    window = settings.pattern_lookback_days
    seg = max(window // 3, 1)

    low_seg1 = df["low"].shift(1 + 2 * seg).rolling(seg).min()
    high_seg1 = df["high"].shift(1 + 2 * seg).rolling(seg).max()
    low_seg2 = df["low"].shift(1 + seg).rolling(seg).min()
    high_seg2 = df["high"].shift(1 + seg).rolling(seg).max()
    low_seg3 = df["low"].shift(1).rolling(seg).min()
    high_seg3 = df["high"].shift(1).rolling(seg).max()

    ascending_lows = (low_seg2 >= low_seg1) & (low_seg3 >= low_seg2)
    ascending_highs = (high_seg2 >= high_seg1) & (high_seg3 >= high_seg1)

    seg1_depth_pct = (high_seg1 - low_seg1) / high_seg1 * 100.0
    seg2_depth_pct = (high_seg2 - low_seg2) / high_seg2 * 100.0
    seg3_depth_pct = (high_seg3 - low_seg3) / high_seg3 * 100.0
    max_segment_depth = pd.concat([seg1_depth_pct, seg2_depth_pct, seg3_depth_pct], axis=1).max(axis=1)

    cond = pd.Series(True, index=df.index)
    cond &= ascending_lows
    cond &= ascending_highs
    cond &= max_segment_depth.between(settings.min_segment_depth_pct, settings.max_segment_depth_pct)
    cond &= df["close"] > high_seg3
    cond &= df[vol_ratio_col] >= settings.breakout_volume_ratio_min

    out = pd.DataFrame(index=df.index)
    out["match"] = cond.fillna(False)
    out["max_segment_depth_pct"] = max_segment_depth
    out["volume_ratio"] = df[vol_ratio_col]
    out["resistance"] = high_seg3
    out["score"] = (
        (settings.max_segment_depth_pct - max_segment_depth).clip(lower=0)
        + df[vol_ratio_col].clip(lower=0)
    )
    return out
