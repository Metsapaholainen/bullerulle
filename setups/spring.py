"""Wyckoff-style Phase B/C "Spring" (shakeout inside a quiet range).

Inside a trading range with volume drying up (Phase B), a brief false
breakdown below range support that reclaims the range within a few bars --
ideally WITHOUT a volume expansion (contrast with a genuine breakdown,
which does expand). `match` fires on the RECLAIM day, while price is still
inside/near the range -- deliberately BEFORE any resistance-breakout bot
would ever see this stock, since it hasn't broken out of anything yet.

This is a classical, coherent framework (Wyckoff, later formalized by
Pruden), not a modern peer-reviewed finding, and it's described here purely
mechanically: a structural false-breakdown-and-reclaim pattern. There is no
claim about "smart money" intent -- that framing is retail folklore this
project deliberately avoids.
"""
from __future__ import annotations

import pandas as pd

from indicators import add_adr_pct, add_volume_stats
from settings import SpringSettings


def scan(df: pd.DataFrame, settings: SpringSettings, rs_rating: pd.Series = None) -> pd.DataFrame:
    df = df.copy()

    adr_col = f"adr_pct_{settings.adr_lookback_days}"
    if adr_col not in df.columns:
        df = add_adr_pct(df, lookback=settings.adr_lookback_days)
    vol_avg_col = f"vol_avg_{settings.volume_avg_period}"
    if vol_avg_col not in df.columns:
        df = add_volume_stats(df, avg_period=settings.volume_avg_period)

    # The range, excluding today, same convention as every other setup here.
    range_high = df["high"].shift(1).rolling(settings.range_lookback_days).max()
    range_low = df["low"].shift(1).rolling(settings.range_lookback_days).min()
    range_pct = (range_high - range_low) / range_low * 100.0

    range_avg_volume = df["volume"].shift(1).rolling(settings.range_lookback_days).mean()
    range_vol_dryup_ratio = range_avg_volume / df[vol_avg_col]

    # How far BELOW today's range support did today's low reach -- a
    # positive value means an undercut; anything within
    # [penetration_min_pct, penetration_max_pct] counts as a candidate
    # spring rather than noise (too shallow) or a real breakdown (too deep).
    penetration_pct = (range_low - df["low"]) / range_low * 100.0
    is_undercut_bar = penetration_pct.between(settings.penetration_min_pct, settings.penetration_max_pct)

    # Did an undercut happen within the trailing reclaim_max_days (incl. today)?
    undercut_recently = is_undercut_bar.rolling(settings.reclaim_max_days, min_periods=1).max().astype(bool)
    reclaim_cond = (df["close"] >= range_low) & undercut_recently

    # The true undercut low over that same trailing window -- this is what a
    # spring entry's stop should actually reference, NOT the reclaim day's
    # own low (which can sit well above it if the reclaim happened on a
    # strong up day). Exposed as `stop_reference_low` for
    # resolve_entry_and_stop() to pick up.
    spring_low = df["low"].rolling(settings.reclaim_max_days, min_periods=1).min()
    reclaim_window_avg_volume = df["volume"].rolling(settings.reclaim_max_days, min_periods=1).mean()
    reclaim_volume_ratio = reclaim_window_avg_volume / range_avg_volume.replace(0, pd.NA)
    volume_did_not_expand = reclaim_volume_ratio <= settings.max_reclaim_volume_ratio

    # Fire only on the FIRST day of a reclaim streak, so a multi-day hold
    # above range_low doesn't re-trigger every day.
    trigger_cond = reclaim_cond & ~reclaim_cond.shift(1).fillna(False)

    if rs_rating is not None:
        rs = rs_rating.reindex(df.index)
    else:
        rs = pd.Series(100.0, index=df.index)

    shape_cond = pd.Series(True, index=df.index)
    shape_cond &= range_pct <= settings.max_range_pct
    shape_cond &= range_vol_dryup_ratio <= settings.volume_dryup_ratio_max
    shape_cond &= df[adr_col] >= settings.min_adr_pct
    shape_cond &= rs >= settings.min_rs_rating

    cond = shape_cond & trigger_cond & volume_did_not_expand

    out = pd.DataFrame(index=df.index)
    out["match"] = cond.fillna(False)
    out["shape_match"] = shape_cond.fillna(False)
    out["adr_pct"] = df[adr_col]
    out["rs_rating"] = rs
    out["range_low"] = range_low
    out["range_high"] = range_high
    out["range_pct"] = range_pct
    out["range_vol_dryup_ratio"] = range_vol_dryup_ratio
    out["penetration_pct"] = penetration_pct
    out["reclaim_volume_ratio"] = reclaim_volume_ratio
    out["stop_reference_low"] = spring_low
    out["score"] = (
        (df[adr_col].clip(lower=0) / max(settings.min_adr_pct, 1e-9))
        + (rs.clip(lower=0) / 100.0) * 2.0
        + ((settings.volume_dryup_ratio_max - range_vol_dryup_ratio).clip(lower=0) / max(settings.volume_dryup_ratio_max, 1e-9))
        + ((settings.max_reclaim_volume_ratio - reclaim_volume_ratio).clip(lower=0))
    )
    return out
