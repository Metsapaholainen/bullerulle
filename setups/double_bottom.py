"""O'Neil 'Double Bottom' (W pattern) (How to Make Money in Stocks).

Two similar lows separated by a peak in between -- shaped like a "W". The
second low often slightly undercuts the first (shaking out the last weak
holders) before the stock rallies and breaks out above the middle peak on
volume.

The lookback window is split into three equal segments (oldest -> newest):
the first low, the middle peak, and the second low.
"""
from __future__ import annotations

import pandas as pd

from indicators import add_volume_stats
from settings import DoubleBottomSettings


def scan(df: pd.DataFrame, settings: DoubleBottomSettings, rs_rating: pd.Series = None) -> pd.DataFrame:
    df = df.copy()

    vol_ratio_col = f"vol_ratio_{settings.volume_avg_period}"
    if vol_ratio_col not in df.columns:
        df = add_volume_stats(df, avg_period=settings.volume_avg_period)

    window = settings.pattern_lookback_days
    seg = max(window // 3, 1)

    low1 = df["low"].shift(1 + 2 * seg).rolling(seg).min()
    middle_peak = df["high"].shift(1 + seg).rolling(seg).max()
    low2 = df["low"].shift(1).rolling(seg).min()

    lower_low = pd.concat([low1, low2], axis=1).min(axis=1)
    depth_pct = (middle_peak - lower_low) / middle_peak * 100.0
    low_difference_pct = (low2 - low1).abs() / low1 * 100.0

    shape_cond = pd.Series(True, index=df.index)
    shape_cond &= depth_pct.between(settings.min_depth_pct, settings.max_depth_pct)
    shape_cond &= low_difference_pct <= settings.max_low_difference_pct

    cond = shape_cond & (df["close"] > middle_peak) & (df[vol_ratio_col] >= settings.breakout_volume_ratio_min)

    out = pd.DataFrame(index=df.index)
    out["match"] = cond.fillna(False)
    out["shape_match"] = shape_cond.fillna(False)
    out["depth_pct"] = depth_pct
    out["low_difference_pct"] = low_difference_pct
    out["volume_ratio"] = df[vol_ratio_col]
    out["middle_peak"] = middle_peak
    out["score"] = (
        (settings.max_low_difference_pct - low_difference_pct).clip(lower=0)
        + df[vol_ratio_col].clip(lower=0)
        + (depth_pct.clip(lower=0) / max(settings.min_depth_pct, 1e-9))
    )
    return out
