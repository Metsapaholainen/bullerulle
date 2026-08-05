"""Setup quality score (0-6.5) and its 1-6 star bucketing.

WHAT THIS IS, AND WHAT IT ISN'T
-------------------------------
Qullamaggie does not publish a numeric grade for his setups. He talks about
"first-tier" vs "second-tier" watchlist names and about "the best of the
best", but there is no 5-star scale anywhere in his own writing. The
4/5/6-star language comes from third-party TradingView indicators built by
other people from his material -- the most explicit of them scores seven
components to a 6.5-point total, which is where "6 star" comes from.

So this module is *our* score, built from *his* documented criteria. Each
component below traces to something he actually said, quoted in the
comments. Don't present the number as his.

WHY A SCORE INSTEAD OF MORE FILTERS
-----------------------------------
Hard thresholds stack multiplicatively: ADR>=5 AND RS>=90 AND prior move
30-100% AND tight base AND above both EMAs took the Breakout scan down to
~5 matches out of 400 symbols. Every one of those gates is a real Qullamaggie
criterion, but as an AND-chain they encode "perfect or nothing", which isn't
how he actually trades -- he ranks a watchlist and takes the best of what's
there on the day. Scoring reproduces that: loosen the gates so candidates
surface, then let the score sort them.

Fully vectorized over the whole index (same contract as the setup scan()
functions) so the score is available for live scans, Approaching Pivot, and
historical backtest filtering alike -- not just the most recent bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from indicators import (
    add_adr_pct,
    add_efficiency_ratio,
    add_sma_slope,
    add_volume_dryup,
    add_volume_stats,
    rs_line_at_high,
)

# Component ceilings. These sum to MAX_QUALITY_SCORE and are weighted by how
# load-bearing each criterion is in his own description: the prior move is
# the single thing he leads with ("a big move higher, 30-100%+"), so it gets
# the largest share; MA alignment and linearity are confirmations rather than
# entry reasons, so they get half-points.
WEIGHT_PRIOR_MOVE = 1.5
WEIGHT_ABOVE_RISING_50 = 1.0
WEIGHT_RS = 1.0
WEIGHT_TIGHTNESS = 1.0
WEIGHT_VOLUME = 1.0
WEIGHT_MA_ALIGNMENT = 0.5
WEIGHT_LINEARITY = 0.5

MAX_QUALITY_SCORE = (
    WEIGHT_PRIOR_MOVE + WEIGHT_ABOVE_RISING_50 + WEIGHT_RS + WEIGHT_TIGHTNESS
    + WEIGHT_VOLUME + WEIGHT_MA_ALIGNMENT + WEIGHT_LINEARITY
)  # 6.5

QUALITY_COMPONENT_COLUMNS = [
    "q_prior_move",
    "q_above_rising_50",
    "q_rs",
    "q_tightness",
    "q_volume",
    "q_ma_alignment",
    "q_linearity",
]

# Default base window when the caller doesn't know the setup's own size. 20
# trading days ~= 4 weeks, the middle of his "2-8 weeks" consolidation range.
DEFAULT_BASE_DAYS = 20


def _tiered(series: pd.Series, thresholds, points) -> pd.Series:
    """Step function: the highest threshold the value clears wins.

    `thresholds` ascending for >= tests. Written as a cumulative sum of
    boolean masks rather than nested np.where so adding a tier later is a
    one-line change and the tiers stay readable next to the comments that
    justify them."""
    out = pd.Series(0.0, index=series.index)
    values = series.fillna(-np.inf)
    for threshold, point in zip(thresholds, points):
        out = out.where(values < threshold, point)
    return out


def _tiered_below(series: pd.Series, thresholds, points) -> pd.Series:
    """Same as _tiered but for "smaller is better" metrics (tightness).
    `thresholds` descending for <= tests."""
    out = pd.Series(0.0, index=series.index)
    values = series.fillna(np.inf)
    for threshold, point in zip(thresholds, points):
        out = out.where(values > threshold, point)
    return out


def compute_quality_panel(
    df: pd.DataFrame,
    settings=None,
    rs_rating: pd.Series = None,
    benchmark_close: pd.Series = None,
    base_days: int = None,
    prior_move_lookback_days: int = 63,
    tightness_window_days: int = None,
) -> pd.DataFrame:
    """Score every bar of `df` 0-6.5 and bucket into 1-6 stars.

    `df` should be the output of indicators.add_core_indicators() (anything
    missing is computed here). `rs_rating` is the cross-sectional 1-99 series
    from compute_rs_rating_panel; `benchmark_close` is SPY's close, used only
    for the RS-line bonus and safely omittable.

    `base_days` should be the setup's own consolidation window when known
    (auto-detect picks a different one per match), so linearity and the
    prior-move boundary are measured over the base the detector actually
    found rather than a fixed guess.

    `tightness_window_days` is a SEPARATE, usually SHORTER window for the
    tightness and volume-dry-up components. This matters concretely: for
    Breakout, `base_days` is `max_consolidation_days` (the full base, ~40
    days by default), but the detector's own tightness check is against
    `min_consolidation_days` (~16 days) -- the tail end of the base, after it
    has narrowed. Scoring tightness over the full window instead measured
    something the detector never claimed and, on 135 real Breakout trades,
    came out **zero for every single one** (mean full-window range was 47%,
    nowhere near any of the tightness tiers) -- a silently dead component,
    not a mis-weighted one. Defaults to `base_days` for setups that don't
    have a separate short window (the five O'Neil patterns, whose "base"
    field already IS the tight part)."""
    base_days = int(base_days or DEFAULT_BASE_DAYS)
    base_days = max(2, base_days)
    tightness_window_days = int(tightness_window_days or base_days)
    tightness_window_days = max(2, min(tightness_window_days, base_days))
    df = df.copy()

    # --- make sure every input column exists -------------------------------
    if "adr_pct_20" not in df.columns:
        df = add_adr_pct(df, lookback=20)
    if "vol_ratio_50" not in df.columns:
        df = add_volume_stats(df, avg_period=50)
    for period in (10, 20):
        col = f"ema_{period}"
        if col not in df.columns:
            df[col] = df["close"].ewm(span=period, adjust=False).mean()
    if "sma_50" not in df.columns:
        df["sma_50"] = df["close"].rolling(50).mean()
    if "sma_50_slope_pct" not in df.columns:
        df = add_sma_slope(df, period=50, lookback=20)
    dryup_col = f"vol_dryup_{tightness_window_days}"
    if dryup_col not in df.columns:
        df = add_volume_dryup(df, base_days=tightness_window_days, avg_period=50)
    er_col = f"efficiency_ratio_{base_days}"
    if er_col not in df.columns:
        df = add_efficiency_ratio(df, window=base_days, col_name=er_col)

    out = pd.DataFrame(index=df.index)

    # --- 1. Prior move (1.5) ----------------------------------------------
    # "You want a stock that has already made a big move higher -- 30-100%+."
    # Measured close-to-close over the window ENDING where the base begins,
    # so it's the genuine net advance, not a high-low range a choppy stock
    # could also satisfy. Both ends shift past the base so today's breakout
    # can't contribute to its own "prior" move.
    #
    # Tiering below is evidence-revised, not just theory: backtested against
    # 135 real Breakout trades (400 symbols, 2022-2026), this is the ONLY one
    # of the seven components whose relationship to becoming a big (trailing-
    # stop-exit) winner was more than noise -- and it was a clean, monotonic
    # one. Quartiles of this exact raw value against trail-exit rate:
    #   Q1 (<=-6%)  -> 5.9%   Q2 (-6..9%) -> 20.6%
    #   Q3 (9..30%) -> 21.2%  Q4 (>=30%)  -> 29.4%
    # The OLD tiering (cliffs at 20/30/50%, zero below 20) missed this
    # entirely: a -27% decline and a +19% advance both scored 0, despite the
    # data showing the latter has ~3.5x the trail-exit rate of the former.
    # This wider tiering is the one deliberate change made on that evidence;
    # the other six components' correlations with trade outcome were all
    # within noise at this sample size (n=135, |r|~0.1-0.2, not significant)
    # and are left as originally designed rather than reweighted on chance.
    window_end_close = df["close"].shift(base_days + 1)
    window_start_close = df["close"].shift(base_days + prior_move_lookback_days)
    prior_net_move_pct = (window_end_close - window_start_close) / window_start_close * 100.0
    out["q_prior_move"] = _tiered(prior_net_move_pct, (0.0, 15.0, 30.0, 50.0), (0.4, 0.8, 1.2, 1.5))

    # --- 2. Above a rising 50-day (1.0) -----------------------------------
    # "I never buy a stock below its 50-day moving average." Split into
    # position (0.5) and slope (0.5) -- see add_sma_slope's docstring for why
    # those are different claims.
    above_50 = (df["close"] > df["sma_50"]).fillna(False)
    rising_50 = (df["sma_50_slope_pct"] > 0).fillna(False)
    out["q_above_rising_50"] = above_50.astype(float) * 0.5 + (above_50 & rising_50).astype(float) * 0.5

    # --- 3. Relative strength (1.0) ---------------------------------------
    # "Top 1-2% of all stocks over 1, 3 and 6 months." RS 97+ is roughly that
    # top 1-2%; 90 and 80 are the "still a leader" and "acceptable" rungs.
    # A new high in the RS *line* while price is still based earns the top
    # tier outright -- outperforming during a consolidation is the strongest
    # single tell that institutions are accumulating.
    if rs_rating is not None:
        rs = rs_rating.reindex(df.index)
    else:
        rs = pd.Series(np.nan, index=df.index)
    rs_points = _tiered(rs, (80.0, 90.0, 97.0), (0.5, 0.75, 1.0))
    if benchmark_close is not None:
        rs_line_high = rs_line_at_high(df["close"], benchmark_close, window=63)
        rs_points = rs_points.where(~rs_line_high, np.maximum(rs_points, 0.75))
    # No RS data at all (single-symbol use, benchmark not loaded) shouldn't
    # silently zero out a whole component and cap everything at 5 stars --
    # award the middle tier so the remaining six components still separate.
    out["q_rs"] = rs_points.where(rs.notna(), 0.5)

    # --- 4. Base tightness (1.0) ------------------------------------------
    # "The range should tighten from left to right." Measured over
    # tightness_window_days (the RECENT/short window), not the full base --
    # see the docstring above for why that distinction is load-bearing.
    tight_high = df["high"].shift(1).rolling(tightness_window_days).max()
    tight_low = df["low"].shift(1).rolling(tightness_window_days).min()
    base_range_pct = (tight_high - tight_low) / tight_low * 100.0
    out["q_tightness"] = _tiered_below(base_range_pct, (15.0, 10.0, 6.0, 3.0), (0.25, 0.5, 0.75, 1.0))

    # --- 5. Volume dry-up then expansion (1.0) ----------------------------
    # "Volume dries up in the consolidation, then expands on the breakout."
    # Both halves are needed for the full point: quiet base (0.5) AND a real
    # volume surge on the signal day (0.5).
    dried_up = (df[dryup_col] < 0.9).fillna(False)
    expanded = (df["vol_ratio_50"] >= 1.5).fillna(False)
    out["q_volume"] = dried_up.astype(float) * 0.5 + expanded.astype(float) * 0.5

    # --- 6. Moving-average alignment (0.5) --------------------------------
    # "Surf the rising 10-, 20- and 50-day moving averages." Full stack in
    # order is the textbook momentum configuration.
    aligned = (
        (df["close"] > df["ema_10"])
        & (df["ema_10"] > df["ema_20"])
        & (df["ema_20"] > df["sma_50"])
    ).fillna(False)
    out["q_ma_alignment"] = aligned.astype(float) * WEIGHT_MA_ALIGNMENT

    # --- 7. Linearity / orderliness (0.5) ---------------------------------
    # "An orderly pullback ... with higher lows", and his repeated emphasis
    # on "linearity" -- which he says you have to train your eye on. The
    # efficiency ratio is that judgement made measurable (0.25), plus the
    # explicit higher-lows check (0.25).
    orderly = (df[er_col] >= 0.3).fillna(False)
    half = max(1, base_days // 2)
    recent_low = df["low"].shift(1).rolling(half).min()
    older_low = df["low"].shift(half + 1).rolling(base_days - half).min()
    higher_lows = (recent_low >= older_low).fillna(False)
    out["q_linearity"] = orderly.astype(float) * 0.25 + higher_lows.astype(float) * 0.25

    # --- total + stars -----------------------------------------------------
    out["quality_score"] = out[QUALITY_COMPONENT_COLUMNS].sum(axis=1)
    out["stars"] = stars_from_score(out["quality_score"])

    # Diagnostics -- surfaced in scan tables so a low score is explainable
    # ("why is this only 3 stars?") without re-deriving it by hand.
    out["prior_net_move_pct_q"] = prior_net_move_pct
    out["base_range_pct_q"] = base_range_pct
    out["vol_dryup_ratio"] = df[dryup_col]
    out["efficiency_ratio"] = df[er_col]
    return out


def stars_from_score(score) -> pd.Series:
    """0-6.5 -> 1-6 stars. floor(), so "4.3" reads as 4 stars -- the
    intuitive mapping. Clipped at 6 because the 6.5 ceiling would otherwise
    produce a lone 7th star for a perfect setup."""
    if isinstance(score, pd.Series):
        return np.floor(score).clip(1, 6).astype(int)
    return int(min(6, max(1, np.floor(score))))


def quality_at(
    df: pd.DataFrame,
    date,
    settings=None,
    rs_rating: pd.Series = None,
    benchmark_close: pd.Series = None,
    base_days: int = None,
) -> dict:
    """Single-date convenience wrapper -- returns a plain dict of the
    component scores, total and stars, or an empty dict if the date isn't in
    the frame. Used by the scan record builders, which want one row."""
    panel = compute_quality_panel(
        df, settings=settings, rs_rating=rs_rating,
        benchmark_close=benchmark_close, base_days=base_days,
    )
    if date not in panel.index:
        return {}
    row = panel.loc[date]
    return {k: (float(v) if k != "stars" else int(v)) for k, v in row.items() if pd.notna(v)}
