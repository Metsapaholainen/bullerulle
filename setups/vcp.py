"""Minervini-style Volatility Contraction Pattern (VCP).

A prior uptrend followed by 2+ successive pullbacks, each SHALLOWER than the
last, with volume declining in parallel across the legs, ending in a very
tight "low-volume pivot" right before the breakout. This is structurally
MORE than this app's Breakout setup: Breakout only checks whether the range
is tight *right now* (a single snapshot); VCP requires a genuine *sequence*
of contracting swing legs plus a volume trend across them. Most simple
momentum-breakout bots only implement the snapshot version, so this
selects a smaller, more specific -- and less crowded -- subset of names.

This is a heuristic swing-leg detector, not pixel-perfect shape recognition:
swing points are found via a simple centered rolling-extreme test, which can
misfire on flat tops/bottoms (ties) or very choppy data. It is Minervini's
DEFINITION made mechanical (multi-leg contraction + volume decay), not a
claim that any specific leg reflects genuine institutional accumulation --
that causal story is plausible but unverified; the structural pattern
itself is what this module actually checks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from indicators import add_adr_pct, add_volume_dryup, add_volume_stats
from settings import VCPSettings

try:
    from indicators import add_level_fakeout_count
except ImportError:  # pragma: no cover - added alongside the fakeout filter feature
    add_level_fakeout_count = None


def scan(df: pd.DataFrame, settings: VCPSettings, rs_rating: pd.Series = None) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    adr_col = f"adr_pct_{settings.adr_lookback_days}"
    if adr_col not in df.columns:
        df = add_adr_pct(df, lookback=settings.adr_lookback_days)
    vol_ratio_col = f"vol_ratio_{settings.volume_avg_period}"
    if vol_ratio_col not in df.columns:
        df = add_volume_stats(df, avg_period=settings.volume_avg_period)
    dryup_col = f"vol_dryup_{settings.pivot_days}"
    if dryup_col not in df.columns:
        df = add_volume_dryup(df, base_days=settings.pivot_days, avg_period=settings.volume_avg_period)

    order = max(1, settings.swing_order_days)

    # --- 1. swing-point detection (one rolling pass) -----------------------
    # A centered window: today is a swing high/low if its own high/low is the
    # extreme point among `order` bars on EITHER side of it. Not knowable
    # until `order` bars later, so both masks are shifted by `order` before
    # use -- the single most important correctness detail here.
    window = 2 * order + 1
    is_swing_high = df["high"] >= df["high"].rolling(window, center=True).max()
    is_swing_high = is_swing_high.fillna(False)
    confirmed_raw = is_swing_high.shift(order).fillna(False)
    # Collapse a run of tied/flat-top bars (all satisfying the extreme test
    # at once) down to just its first bar, so a flat top doesn't spuriously
    # open several new "legs" back to back.
    confirmed_swing_high = confirmed_raw & ~confirmed_raw.shift(1).fillna(False)

    # --- 2. leg assignment (one cumsum pass) --------------------------------
    leg_id = confirmed_swing_high.cumsum()
    loc = pd.Series(np.arange(n), index=df.index)

    # --- 3. per-leg aggregates (one groupby pass each) ----------------------
    # Exactly one non-null "starting high" per leg (the swing-high bar that
    # opened it), so transform("first") -- which skips NaN -- broadcasts that
    # single value across the whole leg without leaking neighbors.
    #
    # confirmed_swing_high is True on the CONFIRMATION row (`order` bars after
    # the actual peak), so `df["high"]` there is some later, lower bar --
    # not the peak itself. Must pull the real peak value back via
    # `.shift(order)` (the same lag confirmed_swing_high itself already
    # encodes), or every leg's high -- and therefore its depth -- is measured
    # against the wrong, understated price.
    swing_high_price = df["high"].shift(order).where(confirmed_swing_high)
    leg_high = swing_high_price.groupby(leg_id).transform("first")
    leg_low = df["low"].groupby(leg_id).transform("min")
    leg_avg_volume = df["volume"].groupby(leg_id).transform("mean")
    leg_depth_pct = (leg_high - leg_low) / leg_high * 100.0
    leg_start_loc = loc.where(confirmed_swing_high).groupby(leg_id).transform("first")

    # One row per unique leg -- used only to look up COMPLETED legs (whose
    # own rows are all strictly in the past relative to any row referencing
    # them via `leg_id - k` for k >= 1), so this is causally safe despite
    # being built in a single pass over the whole frame, the same way a
    # shifted rolling window is: what matters is which calendar rows feed a
    # given row's value, not how many passes computed it.
    depth_by_id = leg_depth_pct.groupby(leg_id).first()
    avgvol_by_id = leg_avg_volume.groupby(leg_id).first()
    start_loc_by_id = leg_start_loc.groupby(leg_id).first()
    high_by_id = leg_high.groupby(leg_id).first()

    # --- 4. bounded lookback over a small constant, not over rows ----------
    max_k = max(2, settings.max_legs_checked)
    leg_cols = {}
    for k in range(1, max_k + 1):
        ref_id = leg_id - k
        leg_cols[f"leg{k}_depth_pct"] = ref_id.map(depth_by_id)
        leg_cols[f"leg{k}_avg_volume"] = ref_id.map(avgvol_by_id)
        leg_cols[f"leg{k}_start_loc"] = ref_id.map(start_loc_by_id)
        leg_cols[f"leg{k}_high"] = ref_id.map(high_by_id)

    # --- 5. monotonic contraction + volume-decline checks, with tolerance --
    # Real data is never perfectly monotonic, so each successive (older) leg
    # only needs to be AT LEAST AS wide/loud within a tolerance band, not
    # strictly larger.
    tol_depth = 1.0 + settings.contraction_tolerance_pct / 100.0
    tol_vol = 1.0 + settings.volume_tolerance_pct / 100.0
    min_legs = max(1, settings.min_legs)

    legs_shallowing = pd.Series(True, index=df.index)
    volume_declining = pd.Series(True, index=df.index)
    for k in range(1, min_legs):
        legs_shallowing &= leg_cols[f"leg{k}_depth_pct"] <= leg_cols[f"leg{k + 1}_depth_pct"] * tol_depth
        volume_declining &= leg_cols[f"leg{k}_avg_volume"] <= leg_cols[f"leg{k + 1}_avg_volume"] * tol_vol

    depth_cols_needed = [f"leg{k}_depth_pct" for k in range(1, min_legs + 1)]
    enough_legs = pd.concat([leg_cols[c] for c in depth_cols_needed], axis=1).notna().all(axis=1)

    # The oldest leg actually used must still fall within the base window --
    # otherwise a stock with very few big swings could satisfy "N contracting
    # legs" using legs from years ago, which isn't a VCP, it's coincidence.
    oldest_start_loc = leg_cols[f"leg{min_legs}_start_loc"]
    legs_within_window = (loc - oldest_start_loc) <= settings.vcp_lookback_days

    # --- 6. final low-volume pivot ------------------------------------------
    pivot_high = df["high"].shift(1).rolling(settings.pivot_days).max()
    pivot_low = df["low"].shift(1).rolling(settings.pivot_days).min()
    pivot_range_pct = (pivot_high - pivot_low) / pivot_low * 100.0
    pivot_tight = pivot_range_pct <= settings.max_pivot_range_pct
    pivot_quiet = df[dryup_col] <= settings.max_pivot_volume_dryup_ratio

    # --- 7. prior uptrend (the base must follow an actual advance) ---------
    window_end_close = df["close"].shift(settings.vcp_lookback_days + 1)
    window_start_close = df["close"].shift(settings.vcp_lookback_days + settings.prior_move_lookback_days)
    prior_net_move_pct = (window_end_close - window_start_close) / window_start_close * 100.0
    prior_move_ok = prior_net_move_pct >= settings.prior_move_min_pct

    if rs_rating is not None:
        rs = rs_rating.reindex(df.index)
    else:
        rs = pd.Series(100.0, index=df.index)  # no RS data available -> don't filter on it

    # Resistance = the most recently confirmed leg's own peak -- the level
    # sitting directly above the current pivot -- NOT a flat vcp_lookback_days
    # rolling max. A straight rolling window over the whole base span also
    # picks up the tail of the PRIOR MOVE that preceded the base (whose own
    # peak, by construction of a genuine net advance, is often taller than
    # where the base's own contracting legs are oscillating), which made a
    # real breakout of the base's own high nearly unreachable in practice
    # (verified against real data: of 222 shape-valid pivots across 366
    # cached symbols, only 7 ever closed above a flat 90-day rolling max,
    # vs. this leg-anchored level tracking the base's own top).
    resistance = leg_cols["leg1_high"]

    shape_cond = (
        enough_legs & legs_within_window & legs_shallowing & volume_declining
        & pivot_tight & pivot_quiet & prior_move_ok
        & (df[adr_col] >= settings.min_adr_pct)
        & (rs >= settings.min_rs_rating)
    )

    recent_fakeout_count = pd.Series(0, index=df.index)
    if settings.fakeout_filter_action != "off" and add_level_fakeout_count is not None:
        recent_fakeout_count = add_level_fakeout_count(
            df, resistance, settings.fakeout_lookback_days,
            settings.fakeout_touch_tolerance_pct, settings.fakeout_min_reversal_pct,
        )
        if settings.fakeout_filter_action == "exclude":
            shape_cond &= recent_fakeout_count == 0

    cond = shape_cond & (df["close"] > resistance) & (df[vol_ratio_col] >= settings.breakout_volume_ratio_min)

    out = pd.DataFrame(index=df.index)
    out["match"] = cond.fillna(False)
    out["shape_match"] = shape_cond.fillna(False)
    out["adr_pct"] = df[adr_col]
    out["rs_rating"] = rs
    out["resistance"] = resistance
    out["prior_net_move_pct"] = prior_net_move_pct
    out["pivot_range_pct"] = pivot_range_pct
    out["pivot_volume_dryup_ratio"] = df[dryup_col]
    out["volume_ratio"] = df[vol_ratio_col]
    out["num_legs_confirmed"] = min_legs
    out["recent_fakeout_count"] = recent_fakeout_count
    for k in range(1, min_legs + 1):
        out[f"leg{k}_depth_pct"] = leg_cols[f"leg{k}_depth_pct"]
    score = (
        (df[adr_col].clip(lower=0) / max(settings.min_adr_pct, 1e-9))
        + (rs.clip(lower=0) / 100.0) * 2.0
        + df[vol_ratio_col].clip(lower=0)
        + ((settings.max_pivot_range_pct - pivot_range_pct).clip(lower=0) / max(settings.max_pivot_range_pct, 1e-9))
    )
    if settings.fakeout_filter_action == "penalize":
        score = score - recent_fakeout_count * settings.fakeout_penalty_score
    out["score"] = score
    return out
