"""Qullamaggie 'Episodic Pivot' setup: a big gap up on a catalyst (earnings,
news) with heavy volume, out of a previously quiet/neglected base.
"""
from __future__ import annotations

import pandas as pd

from indicators import add_gap_pct, add_volume_stats
from settings import EpisodicPivotSettings


def scan(df: pd.DataFrame, settings: EpisodicPivotSettings, rs_rating: pd.Series = None) -> pd.DataFrame:
    df = df.copy()

    if "gap_pct" not in df.columns:
        df = add_gap_pct(df)

    vol_ratio_col = f"vol_ratio_{settings.volume_avg_period}"
    if vol_ratio_col not in df.columns:
        df = add_volume_stats(df, avg_period=settings.volume_avg_period)

    # Base tightness in the window right before the gap (excludes the gap day itself).
    prior_high = df["high"].shift(1).rolling(settings.lookback_base_days).max()
    prior_low = df["low"].shift(1).rolling(settings.lookback_base_days).min()
    prior_range_pct = (prior_high - prior_low) / prior_low * 100.0

    cond = pd.Series(True, index=df.index)
    cond &= df["gap_pct"] >= settings.min_gap_pct
    cond &= df[vol_ratio_col] >= settings.min_volume_ratio
    cond &= prior_range_pct <= settings.max_prior_range_pct
    cond &= df["close"] >= settings.min_price

    out = pd.DataFrame(index=df.index)
    out["match"] = cond.fillna(False)
    out["gap_pct"] = df["gap_pct"]
    out["volume_ratio"] = df[vol_ratio_col]
    out["prior_range_pct"] = prior_range_pct
    out["score"] = df["gap_pct"].clip(lower=0) / 10.0 + df[vol_ratio_col].clip(lower=0)
    return out
