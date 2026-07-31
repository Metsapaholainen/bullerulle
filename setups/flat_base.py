"""O'Neil 'Flat Base' (How to Make Money in Stocks).

A tight, mostly sideways consolidation (O'Neil: no more than ~10-15% from
top to bottom) lasting at least 5 weeks, typically forming after an existing
uptrend or as a second-stage base following an earlier one. A breakout above
the base's high on above-average volume is the buy signal.
"""
from __future__ import annotations

import pandas as pd

from indicators import add_volume_stats
from settings import FlatBaseSettings


def scan(df: pd.DataFrame, settings: FlatBaseSettings, rs_rating: pd.Series = None) -> pd.DataFrame:
    df = df.copy()

    vol_ratio_col = f"vol_ratio_{settings.volume_avg_period}"
    if vol_ratio_col not in df.columns:
        df = add_volume_stats(df, avg_period=settings.volume_avg_period)

    base_high = df["high"].shift(1).rolling(settings.base_days).max()
    base_low = df["low"].shift(1).rolling(settings.base_days).min()
    range_pct = (base_high - base_low) / base_low * 100.0

    prior_high = df["high"].shift(1 + settings.base_days).rolling(settings.prior_move_lookback_days).max()
    prior_low = df["low"].shift(1 + settings.base_days).rolling(settings.prior_move_lookback_days).min()
    prior_move_pct = (prior_high - prior_low) / prior_low * 100.0

    cond = pd.Series(True, index=df.index)
    cond &= range_pct <= settings.max_range_pct
    cond &= prior_move_pct >= settings.min_prior_move_pct
    cond &= df["close"] > base_high
    cond &= df[vol_ratio_col] >= settings.breakout_volume_ratio_min

    out = pd.DataFrame(index=df.index)
    out["match"] = cond.fillna(False)
    out["range_pct"] = range_pct
    out["prior_move_pct"] = prior_move_pct
    out["volume_ratio"] = df[vol_ratio_col]
    out["base_high"] = base_high
    out["score"] = (
        (settings.max_range_pct - range_pct).clip(lower=0)
        + df[vol_ratio_col].clip(lower=0)
        + (prior_move_pct.clip(lower=0) / max(settings.min_prior_move_pct, 1e-9))
    )
    return out
