"""Pullback: buy the FIRST shallow pullback to a rising EMA in an
already-confirmed RS leader, on light volume (sellers-exhausted proxy),
after having been meaningfully extended away from that same EMA recently
(proof this is a genuine pullback FROM an extended move, not chop that
never left the average).

A genuinely different, complementary style to Squeeze/Breakout: it never
requires price to make a new high to trigger, so it structurally can't feel
"late" the way a breakout does -- the entry is buying weakness within an
already-established uptrend, not chasing strength.

Entry: unmodified entry_mode="next_open" (fills unconditionally at
tomorrow's open) rather than stop_buy -- this style buys near where price
already is, it doesn't need a breakout-confirmation trigger. Stop:
unmodified stop_mode="low_of_signal_day" -- the pullback day's own low
already is "the pocket's low" this style wants, no override needed.
"""
from __future__ import annotations

import pandas as pd

from indicators import add_adr_pct, add_volume_stats
from settings import PullbackSettings


def scan(df: pd.DataFrame, settings: PullbackSettings, rs_rating: pd.Series = None) -> pd.DataFrame:
    df = df.copy()

    adr_col = f"adr_pct_{settings.adr_lookback_days}"
    if adr_col not in df.columns:
        df = add_adr_pct(df, lookback=settings.adr_lookback_days)
    vol_ratio_col = f"vol_ratio_{settings.volume_avg_period}"
    if vol_ratio_col not in df.columns:
        df = add_volume_stats(df, avg_period=settings.volume_avg_period)

    ema_col = f"ema_{settings.ema_period}"
    if ema_col not in df.columns:
        df[ema_col] = df["close"].ewm(span=settings.ema_period, adjust=False).mean()

    # Same shift(1)/shift(lookback+1) construction as breakout.py's own EMA
    # slope check -- today's own bar can't feed into "is it rising".
    ema_slope_pct = (
        (df[ema_col].shift(1) - df[ema_col].shift(settings.ema_slope_lookback_days + 1))
        / df[ema_col].shift(settings.ema_slope_lookback_days + 1) * 100.0
    )

    adr_dollars = df["close"] * df[adr_col] / 100.0

    # How far ABOVE the EMA price reached, in ADRs -- the prior extension,
    # excluding today, over the trailing extension_lookback_days.
    dist_above_ema_adr = ((df["high"] - df[ema_col]) / adr_dollars).clip(lower=0)
    max_extension_adr = dist_above_ema_adr.shift(1).rolling(settings.extension_lookback_days).max()

    # How close TODAY sits to the EMA, in ADRs -- "at the pocket."
    pullback_distance_adr = (df["close"] - df[ema_col]).abs() / adr_dollars
    is_near_ema = pullback_distance_adr <= settings.max_pullback_distance_adr_multiple

    # "FIRST touch": an earlier touch within the trailing window disqualifies
    # today.
    touched_recently = (
        is_near_ema.shift(1).rolling(settings.min_bars_since_last_touch, min_periods=1).max().fillna(0).astype(bool)
    )
    is_first_touch = is_near_ema & ~touched_recently

    pullback_window_vol_ratio = df[vol_ratio_col].rolling(settings.pullback_window_days).mean()
    light_volume = pullback_window_vol_ratio <= settings.max_pullback_volume_ratio

    if rs_rating is not None:
        rs = rs_rating.reindex(df.index)
    else:
        rs = pd.Series(100.0, index=df.index)

    shape_cond = pd.Series(True, index=df.index)
    shape_cond &= df[adr_col] >= settings.min_adr_pct
    shape_cond &= rs >= settings.min_rs_rating
    shape_cond &= ema_slope_pct > settings.min_ema_slope_pct
    shape_cond &= max_extension_adr >= settings.min_extension_adr_multiple
    shape_cond &= light_volume

    cond = shape_cond & is_first_touch

    out = pd.DataFrame(index=df.index)
    out["match"] = cond.fillna(False)
    out["shape_match"] = shape_cond.fillna(False)
    out["adr_pct"] = df[adr_col]
    out["rs_rating"] = rs
    out["ema"] = df[ema_col]
    out["ema_slope_pct"] = ema_slope_pct
    out["pullback_distance_adr"] = pullback_distance_adr
    out["max_extension_adr"] = max_extension_adr
    out["pullback_volume_ratio"] = pullback_window_vol_ratio
    out["score"] = (
        (df[adr_col].clip(lower=0) / max(settings.min_adr_pct, 1e-9))
        + (rs.clip(lower=0) / 100.0) * 2.0
        + (max_extension_adr.clip(lower=0) / max(settings.min_extension_adr_multiple, 1e-9))
        + ((settings.max_pullback_volume_ratio - pullback_window_vol_ratio).clip(lower=0))
    )
    return out
