"""BullaRullaEP's episodic-pivot detector -- forked from the parent
project's `setups/episodic_pivot.py`, upgraded with point-in-time
fundamentals, a backward-looking earnings-catalyst confirmation, and
scoring bonuses from the LLM catalyst classifier and (optional, exploratory)
social-sentiment chatter spike.

Gap/volume conditions are still evaluated on daily EOD bars only for the
HARD GATES below -- same acknowledged limitation as the parent module, this
codebase still has no true intraday data source. What's different here:
`bullarullaep/scan_confirm.py` (pass 2) does a real premarket confirmation
pass using a live quote before actually alerting, so the EOD gates below are
the coarse first cut, not the final word, the way they were in the parent
project's single-pass design.

Growth/EPS-beat inputs can be supplied two ways:
  - Point-in-time COLUMNS already on `df` (`revenue_growth_pct`,
    `eps_growth_pct`, `eps_beat_pct`, `is_earnings_day`) -- the backtest
    path: the caller pre-computes these per bar via
    `point_in_time_fundamentals.fundamentals_as_of()` against the same
    cached quarterly history, so a signal on date D can never see a quarter
    reported after D. This is the fix for the parent module's own
    documented look-ahead-bias risk.
  - A single `fundamentals` dict / `is_earnings_day` bool, broadcast across
    every row -- the live-scan path, where "today" is the only row that
    matters and a fresh point-in-time snapshot IS "as of today" by
    construction.
`catalyst`/`social_sentiment_spike` are inherently live-only signals (no
historical news/sentiment archive exists to backtest them against) and are
only ever applied to the LAST row -- an honest limitation, not an oversight,
matching the parent module's own "documented simplification" precedent.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from indicators import add_gap_pct, add_volume_stats

from bullarullaep.settings import EPSettings


def scan(
    df: pd.DataFrame,
    settings: EPSettings,
    fundamentals: Optional[dict] = None,
    is_earnings_day: bool = False,
    catalyst=None,
    social_sentiment_spike: Optional[float] = None,
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
    # penalty (halved score below), not a hard exclude.
    prior_gap_flag = df["gap_pct"] >= settings.prior_ep_min_gap_pct
    had_prior_ep = (
        prior_gap_flag.shift(1).rolling(settings.prior_ep_lookback_days).sum() > settings.max_prior_ep_count
    ).fillna(False)

    # Point-in-time-preferring growth/EPS-beat inputs (see module docstring).
    if "revenue_growth_pct" in df.columns or "eps_growth_pct" in df.columns:
        revenue_growth_pct = df.get("revenue_growth_pct", pd.Series(np.nan, index=df.index))
        eps_growth_pct = df.get("eps_growth_pct", pd.Series(np.nan, index=df.index))
        eps_beat_pct_col = df.get("eps_beat_pct", pd.Series(np.nan, index=df.index))
    elif fundamentals:
        revenue_growth_pct = pd.Series(fundamentals.get("revenue_growth_pct"), index=df.index, dtype="float64")
        eps_growth_pct = pd.Series(fundamentals.get("eps_growth_pct"), index=df.index, dtype="float64")
        eps_beat_pct_col = pd.Series(fundamentals.get("eps_beat_pct"), index=df.index, dtype="float64")
    else:
        revenue_growth_pct = pd.Series(np.nan, index=df.index)
        eps_growth_pct = pd.Series(np.nan, index=df.index)
        eps_beat_pct_col = pd.Series(np.nan, index=df.index)

    growth_pct = pd.concat([revenue_growth_pct, eps_growth_pct], axis=1).max(axis=1, skipna=True)

    growth_tier = pd.Series(
        np.select(
            [growth_pct.isna(), growth_pct >= settings.ideal_growth_pct, growth_pct >= settings.min_growth_pct_floor],
            ["unknown", "5-star", "floor-met"],
            default="below-floor",
        ),
        index=df.index,
    )
    if settings.require_growth_floor:
        cond &= growth_pct.isna() | (growth_pct >= settings.min_growth_pct_floor)

    beat_qualifies = eps_beat_pct_col.notna() & (eps_beat_pct_col >= settings.min_eps_beat_pct_for_bonus)
    if settings.require_eps_beat:
        cond &= eps_beat_pct_col.isna() | beat_qualifies

    growth_bonus = pd.Series(
        np.select(
            [
                (growth_tier == "5-star") & beat_qualifies,
                growth_tier == "5-star",
                (growth_tier == "floor-met") & beat_qualifies,
                growth_tier == "floor-met",
                beat_qualifies,
            ],
            [3.0, 2.0, 1.5, 1.0, 0.5],
            default=0.0,
        ),
        index=df.index, dtype="float64",
    )

    # Backward-looking catalyst confirmation, point-in-time if a column is
    # present (backtest path), else the scalar arg broadcast to the last row
    # only (live-scan path -- see module docstring).
    if "is_earnings_day" in df.columns:
        is_earnings_day_s = df["is_earnings_day"].fillna(False).astype(bool)
    else:
        is_earnings_day_s = pd.Series(False, index=df.index)
        if len(is_earnings_day_s) and is_earnings_day:
            is_earnings_day_s.iloc[-1] = True

    # LLM catalyst-classifier bonus -- live-only, last row only (see module
    # docstring). Never gates `cond`: a misclassification should never
    # silently kill a real trade.
    catalyst_type = pd.Series(None, index=df.index, dtype="object")
    catalyst_confidence = pd.Series(np.nan, index=df.index)
    catalyst_rationale = pd.Series(None, index=df.index, dtype="object")
    catalyst_bonus = pd.Series(0.0, index=df.index)
    if settings.enable_catalyst_classifier and catalyst is not None and len(df):
        last = df.index[-1]
        catalyst_type.loc[last] = catalyst.catalyst_type
        catalyst_confidence.loc[last] = catalyst.confidence
        catalyst_rationale.loc[last] = catalyst.rationale
        if catalyst.is_genuine and catalyst.confidence >= settings.min_catalyst_confidence:
            catalyst_bonus.loc[last] = settings.catalyst_bonus_weight * catalyst.confidence

    # Exploratory social-sentiment chatter-spike bonus -- same live-only,
    # last-row-only treatment. OFF by default (see settings.py: FMP's
    # social-sentiment endpoint is unavailable on this project's current
    # plan tier, confirmed 403 live).
    social_spike_col = pd.Series(np.nan, index=df.index)
    social_bonus = pd.Series(0.0, index=df.index)
    if settings.enable_social_sentiment and social_sentiment_spike is not None and len(df):
        last = df.index[-1]
        social_spike_col.loc[last] = social_sentiment_spike
        if social_sentiment_spike >= settings.social_sentiment_spike_threshold:
            social_bonus.loc[last] = 1.0

    base_score = df["gap_pct"].clip(lower=0) / 10.0 + df[vol_ratio_col].clip(lower=0) + growth_bonus
    base_score = base_score + catalyst_bonus + social_bonus
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
    out["eps_beat_pct"] = eps_beat_pct_col
    out["is_earnings_day"] = is_earnings_day_s
    out["catalyst_type"] = catalyst_type
    out["catalyst_confidence"] = catalyst_confidence
    out["catalyst_rationale"] = catalyst_rationale
    out["social_sentiment_spike"] = social_spike_col
    out["score"] = score
    return out
