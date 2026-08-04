"""Stockbee (Pradeep Bonde) 'Momentum Burst' -- a 3-5 day swing.

Bonde's thesis: most short-term moves are 3-5 day "bursts" of 8-40%, kicked
off by a range-expansion day out of a quiet consolidation. You take day 1 of
the burst, hold 3-5 days, sell into strength, and repeat -- the edge comes
from doing it hundreds of times, not from picking the one big winner.

Included in this app because it is the closest published system to the way
this account is actually traded: an end-of-day scan, a stop-buy above the
signal day's high the next morning, a 3-5 day hold, sell half, trail the
rest. Qullamaggie credits Bonde as one of his own sources, and the Breakout
setup here is effectively the same idea on a longer timeframe.

The core scan is Bonde's, unchanged since 2014 and quoted verbatim below:

    c/c1 >= 1.04 and v > v1 and v > 100000

Everything else is his separately-published 8-point quality checklist, made
quantitative. Each check is numbered to match his list and quoted in the
comment above it.

WHAT THIS CAN'T CHECK: Bonde (like Qullamaggie) watches how the stock trades
intraday -- whether it holds its gains into the close, how it behaves in the
first hour. This app has daily bars only, so check 8 ("close near the high")
is the closest available proxy for that whole dimension.
"""
from __future__ import annotations

import pandas as pd

from indicators import add_adr_pct, add_efficiency_ratio, add_volume_dryup, add_volume_stats
from settings import MomentumBurstSettings


def scan(df: pd.DataFrame, settings: MomentumBurstSettings, rs_rating: pd.Series = None) -> pd.DataFrame:
    df = df.copy()

    vol_ratio_col = f"vol_ratio_{settings.volume_avg_period}"
    if vol_ratio_col not in df.columns:
        df = add_volume_stats(df, avg_period=settings.volume_avg_period)
    adr_col = f"adr_pct_{settings.adr_lookback_days}"
    if adr_col not in df.columns:
        df = add_adr_pct(df, lookback=settings.adr_lookback_days)
    dryup_col = f"vol_dryup_{settings.consolidation_days}"
    if dryup_col not in df.columns:
        df = add_volume_dryup(df, base_days=settings.consolidation_days,
                              avg_period=settings.volume_avg_period)
    er_col = f"efficiency_ratio_{settings.prior_uptrend_lookback_days}"
    if er_col not in df.columns:
        df = add_efficiency_ratio(df, window=settings.prior_uptrend_lookback_days, col_name=er_col)
    # Bonde's check 5 is about the uptrend *before* the consolidation, so the
    # measurement window has to end where the base begins. Reading it at
    # today's bar instead would blend the base's own sideways chop into the
    # "was the advance linear" answer and mark orderly setups as choppy.
    prior_efficiency = df[er_col].shift(settings.consolidation_days)

    prev_close = df["close"].shift(1)
    gain_pct = (df["close"] - prev_close) / prev_close * 100.0
    day_range = df["high"] - df["low"]

    # --- consolidation window, excluding today ----------------------------
    # Everything describing "the base" shifts past today, so the burst day's
    # own expansion can't make its own preceding base look wide or loud.
    base_high = df["high"].shift(1).rolling(settings.consolidation_days).max()
    base_low = df["low"].shift(1).rolling(settings.consolidation_days).min()
    base_range_pct = (base_high - base_low) / base_low * 100.0

    # (1) "stock should have range expansion on breakout day"
    avg_range = day_range.shift(1).rolling(settings.range_expansion_lookback).mean()
    range_expansion_ratio = day_range / avg_range

    # (3) "day before breakout should be narrow range day or negative day"
    prior_range = day_range.shift(1)
    prior_was_narrow = prior_range <= avg_range.shift(1)
    prior_was_negative = df["close"].shift(1) < df["close"].shift(2)
    quiet_prior_day = prior_was_narrow | prior_was_negative

    # (4) "pre breakout there should not be many big range moves or breakdowns"
    daily_move_pct = (df["close"] - prev_close).abs() / prev_close * 100.0
    max_base_move_pct = daily_move_pct.shift(1).rolling(settings.consolidation_days).max()

    # (8) "stock should close near high on breakout day"
    close_position_pct = (df["close"] - df["low"]) / day_range.replace(0, pd.NA) * 100.0

    dollar_volume = df["close"] * df["volume"]

    if rs_rating is not None:
        rs = rs_rating.reindex(df.index)
    else:
        rs = pd.Series(100.0, index=df.index)  # no RS data -> don't filter on it

    # --- shape: is this a valid base to burst OUT of? ---------------------
    # Split from the trigger the same way breakout.py does, so this setup can
    # also feed "approaching pivot" (a quiet, orderly base that simply hasn't
    # had its expansion day yet).
    shape_cond = pd.Series(True, index=df.index)
    shape_cond &= dollar_volume >= settings.min_dollar_volume
    shape_cond &= df[adr_col] >= settings.min_adr_pct
    shape_cond &= rs >= settings.min_rs_rating
    # (6) "correction or consolidation should be orderly during the entire move"
    shape_cond &= base_range_pct <= settings.max_consolidation_range_pct
    # (4)
    shape_cond &= max_base_move_pct <= settings.max_base_daily_move_pct
    # (5) "stock should have linearity in prior uptrend before the consolidation"
    shape_cond &= prior_efficiency >= settings.min_efficiency_ratio
    # (7) "volume during consolidation should be preferably orderly and lower"
    if settings.require_volume_dryup:
        shape_cond &= df[dryup_col] <= settings.volume_dryup_ratio_max
    # (3)
    if settings.require_quiet_prior_day:
        shape_cond &= quiet_prior_day
    # "Avoid day 2/3 entries" -- by then the burst is underway and the stop
    # is too far below to be worth taking. consecutive_up_days counts today,
    # so `<` keeps a first up-day and rejects the third in a row.
    shape_cond &= df["consecutive_up_days"] < settings.max_consecutive_up_days

    # --- trigger: Bonde's core scan + the two same-day quality checks ------
    trigger_cond = gain_pct >= settings.min_gain_pct               # c/c1 >= 1.04
    if settings.require_volume_above_prior:
        trigger_cond &= df["volume"] > df["volume"].shift(1)       # v > v1
    trigger_cond &= df["volume"] >= settings.min_volume            # v > 100000
    trigger_cond &= range_expansion_ratio >= settings.min_range_expansion_ratio   # (1)
    trigger_cond &= close_position_pct >= settings.min_close_position_pct         # (8)

    cond = shape_cond & trigger_cond

    out = pd.DataFrame(index=df.index)
    out["match"] = cond.fillna(False)
    out["shape_match"] = shape_cond.fillna(False)
    out["gain_pct"] = gain_pct
    out["range_expansion_ratio"] = range_expansion_ratio
    out["close_position_pct"] = close_position_pct
    out["base_range_pct"] = base_range_pct
    out["max_base_move_pct"] = max_base_move_pct
    out["vol_dryup_ratio"] = df[dryup_col]
    out["efficiency_ratio"] = prior_efficiency
    out["volume_ratio"] = df[vol_ratio_col]
    out["dollar_volume"] = dollar_volume
    out["adr_pct"] = df[adr_col]
    out["rs_rating"] = rs
    # The level a stop-buy would sit above. Named `resistance` to match
    # breakout.py so the chart's RESISTANCE_FIELDS lookup and the Orders tab
    # find it without a special case. For a burst that's the signal day's own
    # high, not the base high -- the expansion day IS the breakout.
    out["resistance"] = df["high"]
    out["score"] = (
        gain_pct.clip(lower=0) / max(settings.min_gain_pct, 1e-9)
        + range_expansion_ratio.clip(lower=0)
        + (rs.clip(lower=0) / 100.0) * 2.0
        + (close_position_pct.clip(lower=0) / 100.0)
        + (2.0 - df[dryup_col].clip(lower=0, upper=2.0))  # quieter base scores higher
    )
    return out
