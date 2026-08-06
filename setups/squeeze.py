"""Squeeze: fires on the QUIET/COILED day, BEFORE the breakout prints.

A momentum leader (RS rating), above a rising 50-day MA, sitting in a range
that's meaningfully tighter than its own ADR over the last few days
(self-normalized -- no hardcoded % threshold, since a 3%-ADR stock and a
10%-ADR stock need very different absolute tightness to mean the same
thing), that has NOT yet closed above its recent pivot high, following a
genuine prior advance (a coiled DEAD stock is not a Squeeze candidate).

`close <= pivot_high` is required here where breakout.py requires
`close > resistance` over a similar window -- the same stock can never
match both Squeeze and Breakout on the same day, by design. This is the
direct fix for "the breakout already happened": the signal is the quiet
day, not the day it broke out.

Entry: unmodified entry_mode="stop_buy" -- the signal day's own high plus
entry_buffer_pct is the resting trigger, live for the next session. A
stock that stays coiled re-fires `match` every day it hasn't broken out;
backtest/engine.py's entries loop already skips a symbol with an open
position, so a multi-day coil can't open duplicate/overlapping trades. This
approximates "leave the resting order up" as "replace it with a fresh one
pegged to yesterday's high each day" -- not literally identical (the
trigger can drift down if the coil prints a lower high day-to-day), worth
knowing rather than assuming away.
"""
from __future__ import annotations

import pandas as pd

from indicators import add_adr_pct, add_moving_averages, add_sma_slope
from settings import SqueezeSettings


def scan(df: pd.DataFrame, settings: SqueezeSettings, rs_rating: pd.Series = None) -> pd.DataFrame:
    df = df.copy()

    adr_col = f"adr_pct_{settings.adr_lookback_days}"
    if adr_col not in df.columns:
        df = add_adr_pct(df, lookback=settings.adr_lookback_days)
    ma_col = f"sma_{settings.ma_period}"
    if ma_col not in df.columns:
        df = add_moving_averages(df, ema_periods=(), sma_periods=(settings.ma_period,))
    slope_col = f"sma_{settings.ma_period}_slope_pct"
    if slope_col not in df.columns:
        df = add_sma_slope(df, period=settings.ma_period, lookback=settings.ma_slope_lookback_days)

    # Today's own quiet range -- INCLUDING today, unlike breakout.py's
    # shift(1) base windows, since Squeeze fires ON the coiled day itself.
    recent_high = df["high"].rolling(settings.range_window_days).max()
    recent_low = df["low"].rolling(settings.range_window_days).min()
    recent_range_pct = (recent_high - recent_low) / recent_low * 100.0
    range_to_adr_ratio = recent_range_pct / df[adr_col]

    # The established pivot high, NOT counting today -- same convention as
    # breakout.py's `resistance`. Squeeze requires price to still be BELOW
    # this; Breakout requires price to be ABOVE it.
    pivot_high = df["high"].shift(1).rolling(settings.pivot_lookback_days).max()

    prior_window_close = df["close"].shift(settings.prior_move_lookback_days)
    prior_move_pct = (df["close"] - prior_window_close) / prior_window_close * 100.0

    if rs_rating is not None:
        rs = rs_rating.reindex(df.index)
    else:
        rs = pd.Series(100.0, index=df.index)

    shape_cond = pd.Series(True, index=df.index)
    shape_cond &= df[adr_col] >= settings.min_adr_pct
    shape_cond &= rs >= settings.min_rs_rating
    shape_cond &= range_to_adr_ratio <= settings.max_range_to_adr_ratio
    shape_cond &= prior_move_pct >= settings.min_prior_move_pct
    if settings.require_above_ma:
        shape_cond &= df["close"] > df[ma_col]
    if settings.require_rising_ma:
        shape_cond &= df[slope_col] > settings.min_ma_slope_pct

    cond = shape_cond & (df["close"] <= pivot_high)

    out = pd.DataFrame(index=df.index)
    out["match"] = cond.fillna(False)
    out["shape_match"] = shape_cond.fillna(False)
    out["adr_pct"] = df[adr_col]
    out["rs_rating"] = rs
    out["pivot_high"] = pivot_high
    out["recent_range_pct"] = recent_range_pct
    out["range_to_adr_ratio"] = range_to_adr_ratio
    out["prior_move_pct"] = prior_move_pct
    out["ma_slope_pct"] = df[slope_col]
    out["score"] = (
        (df[adr_col].clip(lower=0) / max(settings.min_adr_pct, 1e-9))
        + (rs.clip(lower=0) / 100.0) * 2.0
        + ((settings.max_range_to_adr_ratio - range_to_adr_ratio).clip(lower=0) / max(settings.max_range_to_adr_ratio, 1e-9))
        + (prior_move_pct.clip(lower=0) / 100.0)
    )
    return out
