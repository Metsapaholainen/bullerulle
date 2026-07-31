"""O'Neil 'Cup with Handle' pattern (How to Make Money in Stocks).

A U-shaped correction after an existing uptrend: price declines gradually,
rounds out at the bottom (unlike a sharp V-shaped reversal), climbs back
toward the old high, then pulls back one more time in a shallow, low-volume
"handle" before breaking out to new highs on a surge of volume.

Detection is a single-window heuristic (like the other setups here), not
pixel-perfect shape recognition: the cup's "left side high" is measured from
the early portion of the lookback window, the deepest point is the window's
low, and the handle is the most recent short sub-window.
"""
from __future__ import annotations

import pandas as pd

from indicators import add_volume_stats
from settings import CupWithHandleSettings


def scan(df: pd.DataFrame, settings: CupWithHandleSettings, rs_rating: pd.Series = None) -> pd.DataFrame:
    df = df.copy()

    vol_ratio_col = f"vol_ratio_{settings.volume_avg_period}"
    if vol_ratio_col not in df.columns:
        df = add_volume_stats(df, avg_period=settings.volume_avg_period)

    cup_window = settings.cup_lookback_days
    left_peak_days = settings.left_peak_days

    # Left-side high: the peak in the early portion of the cup window, i.e.
    # the level price was at right before the correction began.
    left_high = df["high"].shift(1 + cup_window - left_peak_days).rolling(left_peak_days).max()

    # Cup low: the deepest point anywhere across the whole cup window.
    cup_low = df["low"].shift(1).rolling(cup_window).min()

    cup_depth_pct = (left_high - cup_low) / left_high * 100.0

    # Handle: the most recent short sub-window -- price should have already
    # recovered back up near the old high before this pullback formed.
    handle_window = settings.handle_lookback_days
    handle_high = df["high"].shift(1).rolling(handle_window).max()
    handle_low = df["low"].shift(1).rolling(handle_window).min()
    handle_depth_pct = (handle_high - handle_low) / handle_high * 100.0

    recovery_pct = handle_high / left_high * 100.0
    cup_midpoint = cup_low + 0.5 * (left_high - cup_low)
    handle_in_upper_half = handle_low >= cup_midpoint

    # Prior uptrend before the cup even started forming (the window of
    # `prior_uptrend_lookback_days` days ending right where the cup begins).
    prior_high = df["high"].shift(1 + cup_window).rolling(settings.prior_uptrend_lookback_days).max()
    prior_low = df["low"].shift(1 + cup_window).rolling(settings.prior_uptrend_lookback_days).min()
    prior_uptrend_pct = (prior_high - prior_low) / prior_low * 100.0

    pivot = pd.concat([left_high, handle_high], axis=1).max(axis=1)

    cond = pd.Series(True, index=df.index)
    cond &= cup_depth_pct.between(settings.min_cup_depth_pct, settings.max_cup_depth_pct)
    cond &= handle_depth_pct <= settings.max_handle_depth_pct
    cond &= recovery_pct >= settings.min_recovery_pct
    if settings.handle_upper_half_only:
        cond &= handle_in_upper_half
    cond &= prior_uptrend_pct >= settings.min_prior_uptrend_pct
    cond &= df["close"] > pivot
    cond &= df[vol_ratio_col] >= settings.breakout_volume_ratio_min

    out = pd.DataFrame(index=df.index)
    out["match"] = cond.fillna(False)
    out["cup_depth_pct"] = cup_depth_pct
    out["handle_depth_pct"] = handle_depth_pct
    out["recovery_pct"] = recovery_pct
    out["prior_uptrend_pct"] = prior_uptrend_pct
    out["volume_ratio"] = df[vol_ratio_col]
    out["pivot"] = pivot
    out["score"] = (
        (cup_depth_pct.clip(lower=0) / max(settings.min_cup_depth_pct, 1e-9))
        + (settings.max_handle_depth_pct - handle_depth_pct).clip(lower=0)
        + df[vol_ratio_col].clip(lower=0)
    )
    return out
