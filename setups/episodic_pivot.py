"""Qullamaggie 'Episodic Pivot' setup: a big gap up on a catalyst (earnings,
news) with heavy volume, out of a previously quiet/neglected base.

Gap/volume conditions here are evaluated on daily EOD bars only -- the
article's "10%+ gap on big volume in the first 5-30 minutes" can't be
checked intraday since this codebase has no intraday data source (confirmed:
data/fmp_client.py and data/cache.py only fetch/cache daily bars). This is a
deliberate, acknowledged limitation, not an oversight.
"""
from __future__ import annotations

import pandas as pd

from indicators import add_gap_pct, add_volume_stats
from settings import EpisodicPivotSettings


def scan(
    df: pd.DataFrame, settings: EpisodicPivotSettings, rs_rating: pd.Series = None, fundamentals: dict = None
) -> pd.DataFrame:
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

    # "Best EPs are on stocks that have gone sideways for 3-6 months or
    # more" -- a longer, coarser quiet-window check than prior_range_pct's
    # tight-basing check above (this one just rejects a stock that already
    # ran hard in the months before the gap, tight base or not).
    quiet_base_run_pct = (
        df["close"].shift(1) / df["close"].shift(settings.quiet_base_lookback_days + 1) - 1.0
    ) * 100.0
    cond &= quiet_base_run_pct.abs() <= settings.max_quiet_base_run_pct

    # "Avoid stocks that have already made a big move from a previous EP" --
    # a prior gap of similar size within the trailing window is a soft
    # penalty (halved score below), not a hard exclude, since the article
    # says "avoid," not "never."
    prior_gap_flag = df["gap_pct"] >= settings.prior_ep_min_gap_pct
    had_prior_ep = (
        prior_gap_flag.shift(1).rolling(settings.prior_ep_lookback_days).sum() > settings.max_prior_ep_count
    ).fillna(False)

    # Growth-quality scoring: opt-in via the `fundamentals` kwarg (a plain
    # dict for this single symbol, not a time series -- this app's data
    # model doesn't carry historical-as-of fundamentals, so this reflects
    # "most recent known growth," documented as a simplification).
    growth_pct = None
    growth_tier = "unknown"
    if fundamentals:
        rev_g = fundamentals.get("revenue_growth_pct")
        eps_g = fundamentals.get("eps_growth_pct")
        candidates = [g for g in (rev_g, eps_g) if g is not None]
        if candidates:
            growth_pct = max(candidates)
            if growth_pct >= settings.ideal_growth_pct:
                growth_tier = "5-star"
            elif growth_pct >= settings.min_growth_pct_floor:
                growth_tier = "floor-met"
            else:
                growth_tier = "below-floor"
    if settings.require_growth_floor and growth_pct is not None:
        cond &= growth_pct >= settings.min_growth_pct_floor

    # "A significant beat to analyst expectations" -- a single most-recent-
    # quarter figure (data/fundamentals_cache.py), same "most recent known"
    # simplification as growth_pct above. Never hard-fails the scan unless
    # require_eps_beat is explicitly turned on (off by default: beat-data
    # coverage is spotty for smaller/newer names).
    eps_beat_pct = fundamentals.get("eps_beat_pct") if fundamentals else None
    beat_qualifies = eps_beat_pct is not None and eps_beat_pct >= settings.min_eps_beat_pct_for_bonus
    if settings.require_eps_beat and eps_beat_pct is not None:
        cond &= eps_beat_pct >= settings.min_eps_beat_pct_for_bonus

    if growth_tier == "5-star" and beat_qualifies:
        growth_bonus = 3.0
    elif growth_tier == "5-star":
        growth_bonus = 2.0
    elif growth_tier == "floor-met":
        growth_bonus = 1.0 + (0.5 if beat_qualifies else 0.0)
    else:
        growth_bonus = 0.5 if beat_qualifies else 0.0

    base_score = df["gap_pct"].clip(lower=0) / 10.0 + df[vol_ratio_col].clip(lower=0) + growth_bonus
    score = base_score.where(~had_prior_ep, base_score / 2.0)

    out = pd.DataFrame(index=df.index)
    out["match"] = cond.fillna(False)
    out["gap_pct"] = df["gap_pct"]
    out["volume_ratio"] = df[vol_ratio_col]
    out["prior_range_pct"] = prior_range_pct
    out["quiet_base_run_pct"] = quiet_base_run_pct
    out["had_prior_ep"] = had_prior_ep
    out["growth_pct"] = growth_pct
    out["growth_tier"] = growth_tier
    out["eps_beat_pct"] = eps_beat_pct
    out["score"] = score
    return out
