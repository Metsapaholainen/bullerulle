"""O'Neil 'High Tight Flag' (How to Make Money in Stocks).

O'Neil's rarest and most explosive pattern: a stock rockets up 100%+ (even
120%+) in just 4-8 weeks, then consolidates in a tight "flag" for 3-5 weeks
with a pullback of no more than ~10-25% -- remarkably shallow given how far
and fast the stock just ran. A breakout above the flag on volume signals
continuation. It's also a high-failure-rate pattern precisely because the
prior move is so extreme; treat matches with extra caution.
"""
from __future__ import annotations

import pandas as pd

from indicators import add_volume_stats
from settings import HighTightFlagSettings


def scan(df: pd.DataFrame, settings: HighTightFlagSettings, rs_rating: pd.Series = None) -> pd.DataFrame:
    df = df.copy()

    vol_ratio_col = f"vol_ratio_{settings.volume_avg_period}"
    if vol_ratio_col not in df.columns:
        df = add_volume_stats(df, avg_period=settings.volume_avg_period)

    flag_window = settings.flag_lookback_days
    run_up_window = settings.run_up_lookback_days

    # The explosive prior move, measured in the window before the flag.
    run_up_start_close = df["close"].shift(1 + flag_window + run_up_window)
    run_up_end_close = df["close"].shift(1 + flag_window)
    run_up_pct = (run_up_end_close / run_up_start_close - 1.0) * 100.0

    # The flag itself: how tight is the recent consolidation.
    flag_high = df["high"].shift(1).rolling(flag_window).max()
    flag_low = df["low"].shift(1).rolling(flag_window).min()
    flag_depth_pct = (flag_high - flag_low) / flag_high * 100.0

    shape_cond = pd.Series(True, index=df.index)
    shape_cond &= run_up_pct >= settings.min_run_up_pct
    shape_cond &= flag_depth_pct <= settings.max_flag_depth_pct

    cond = shape_cond & (df["close"] > flag_high) & (df[vol_ratio_col] >= settings.breakout_volume_ratio_min)

    out = pd.DataFrame(index=df.index)
    out["match"] = cond.fillna(False)
    out["shape_match"] = shape_cond.fillna(False)
    out["run_up_pct"] = run_up_pct
    out["flag_depth_pct"] = flag_depth_pct
    out["volume_ratio"] = df[vol_ratio_col]
    out["resistance"] = flag_high
    out["score"] = (
        (run_up_pct.clip(lower=0) / max(settings.min_run_up_pct, 1e-9))
        + (settings.max_flag_depth_pct - flag_depth_pct).clip(lower=0)
        + df[vol_ratio_col].clip(lower=0)
    )
    return out
