"""Qullamaggie 'Breakout' setup: a large prior move, a tight multi-week
consolidation surfing the 10/20 EMA, breaking out to new highs on volume.

`scan()` is fully vectorized over a symbol's whole history so it can be used
both for a live scan (last row) and for backtesting (every historical row is
a potential signal day).
"""
from __future__ import annotations

import pandas as pd

from indicators import add_adr_pct, add_volume_stats
from settings import BreakoutSettings


def scan(df: pd.DataFrame, settings: BreakoutSettings, rs_rating: pd.Series = None) -> pd.DataFrame:
    df = df.copy()

    adr_col = f"adr_pct_{settings.adr_lookback_days}"
    if adr_col not in df.columns:
        df = add_adr_pct(df, lookback=settings.adr_lookback_days)

    vol_ratio_col = f"vol_ratio_{settings.volume_avg_period}"
    if vol_ratio_col not in df.columns:
        df = add_volume_stats(df, avg_period=settings.volume_avg_period)

    ema_fast_col = f"ema_{settings.ema_fast_period}"
    ema_slow_col = f"ema_{settings.ema_slow_period}"
    for col, period in ((ema_fast_col, settings.ema_fast_period), (ema_slow_col, settings.ema_slow_period)):
        if col not in df.columns:
            df[col] = df["close"].ewm(span=period, adjust=False).mean()

    # Current base = trailing max_consolidation_days window, NOT counting
    # today -- otherwise a genuine breakout day's own high/low would inflate
    # the "is the base tight" measurement right when we want to detect it.
    base_high = df["high"].shift(1).rolling(settings.max_consolidation_days).max()
    base_low = df["low"].shift(1).rolling(settings.max_consolidation_days).min()
    base_range_pct = (base_high - base_low) / base_low * 100.0

    # Require the *recent* (shorter) window to also be tight -- a crude proxy
    # for "the base is contracting", not just tight on average. Also excludes today.
    recent_high = df["high"].shift(1).rolling(settings.min_consolidation_days).max()
    recent_low = df["low"].shift(1).rolling(settings.min_consolidation_days).min()
    recent_range_pct = (recent_high - recent_low) / recent_low * 100.0

    # Prior move = range expansion in the window immediately preceding the base.
    prior_high = df["high"].shift(settings.max_consolidation_days + 1).rolling(settings.prior_move_lookback_days).max()
    prior_low = df["low"].shift(settings.max_consolidation_days + 1).rolling(settings.prior_move_lookback_days).min()
    prior_move_pct = (prior_high - prior_low) / prior_low * 100.0

    # Genuine net advance over that same prior-move window -- prior_move_pct
    # above is a high-low RANGE %, which a choppy or net-flat/down stock
    # could still satisfy. This checks it was actually "a big move higher".
    prior_window_end_close = df["close"].shift(settings.max_consolidation_days + 1)
    prior_window_start_close = df["close"].shift(settings.max_consolidation_days + settings.prior_move_lookback_days)
    prior_net_move_pct = (prior_window_end_close - prior_window_start_close) / prior_window_start_close * 100.0

    # "Surf the rising 10-/20-day moving averages" -- being above the EMA
    # (below) isn't the same as the EMA itself trending up. Both ends of the
    # slope are shifted by 1 day (excluding today), same as base_high/
    # recent_high/prior_high above -- otherwise, on the actual breakout day,
    # today's own price surge would feed into ema_fast_col/ema_slow_col and
    # trivially satisfy "rising" regardless of whether the base leading up
    # to it was really trending up.
    ema_fast_slope_pct = (
        (df[ema_fast_col].shift(1) - df[ema_fast_col].shift(settings.ema_slope_lookback_days + 1))
        / df[ema_fast_col].shift(settings.ema_slope_lookback_days + 1) * 100.0
    )
    ema_slow_slope_pct = (
        (df[ema_slow_col].shift(1) - df[ema_slow_col].shift(settings.ema_slope_lookback_days + 1))
        / df[ema_slow_col].shift(settings.ema_slope_lookback_days + 1) * 100.0
    )

    # "An orderly pullback ... with higher lows": split the base window into
    # an older first half and a more recent second half (recent_low above IS
    # that second half already) and require the second half's low to hold at
    # or above the first half's low. The two halves are contiguous and
    # non-overlapping, tiling the same span as base_high/base_low exactly.
    first_half_days = max(1, settings.max_consolidation_days - settings.min_consolidation_days)
    first_half_low = df["low"].shift(settings.min_consolidation_days + 1).rolling(first_half_days).min()

    # Resistance = the base's high, not counting today (same window as base_high).
    resistance = base_high

    if rs_rating is not None:
        rs = rs_rating.reindex(df.index)
    else:
        rs = pd.Series(100.0, index=df.index)  # no RS data available -> don't filter on it

    # Shape/quality conditions -- everything about whether this is a *valid*
    # base, independent of whether it's broken out yet. Used both for the
    # final match and, on its own, to find stocks sitting in a good base that
    # simply haven't triggered ("approaching pivot") yet.
    shape_cond = pd.Series(True, index=df.index)
    shape_cond &= df[adr_col] >= settings.min_adr_pct
    shape_cond &= prior_move_pct.between(settings.prior_move_min_pct, settings.prior_move_max_pct)
    # Tightness is checked on the shorter "recent" window, not the full
    # max_consolidation_days window -- a real base is often tight for only
    # part of that lookback, with the rest still holding the prior move.
    # base_range_pct/base_high are reported as diagnostics and used as the
    # breakout resistance level, but aren't themselves a tightness filter.
    shape_cond &= recent_range_pct <= settings.max_consolidation_range_pct
    shape_cond &= rs >= settings.min_rs_rating
    if settings.require_above_ema10:
        shape_cond &= df["close"] > df[ema_fast_col]
    if settings.require_above_ema20:
        shape_cond &= df["close"] > df[ema_slow_col]
    if settings.require_rising_ema10:
        shape_cond &= ema_fast_slope_pct > settings.min_ema_slope_pct
    if settings.require_rising_ema20:
        shape_cond &= ema_slow_slope_pct > settings.min_ema_slope_pct
    if settings.require_higher_lows:
        shape_cond &= recent_low >= first_half_low * (1.0 - settings.higher_lows_tolerance_pct / 100.0)
    if settings.require_net_prior_advance:
        shape_cond &= prior_net_move_pct >= settings.min_prior_net_advance_pct

    # Trigger conditions -- the actual breakout: crossing resistance on volume.
    cond = shape_cond & (df["close"] > resistance) & (df[vol_ratio_col] >= settings.breakout_volume_ratio_min)

    out = pd.DataFrame(index=df.index)
    out["match"] = cond.fillna(False)
    out["shape_match"] = shape_cond.fillna(False)
    out["adr_pct"] = df[adr_col]
    out["prior_move_pct"] = prior_move_pct
    out["base_range_pct"] = base_range_pct
    out["recent_range_pct"] = recent_range_pct
    out["volume_ratio"] = df[vol_ratio_col]
    out["rs_rating"] = rs
    out["resistance"] = resistance
    out["prior_net_move_pct"] = prior_net_move_pct
    out["ema_fast_slope_pct"] = ema_fast_slope_pct
    out["ema_slow_slope_pct"] = ema_slow_slope_pct
    out["first_half_low"] = first_half_low
    out["score"] = (
        (df[adr_col].clip(lower=0) / max(settings.min_adr_pct, 1e-9))
        + (rs.clip(lower=0) / 100.0) * 2.0
        + df[vol_ratio_col].clip(lower=0)
        + (prior_move_pct.clip(lower=0) / 100.0)
    )
    return out
