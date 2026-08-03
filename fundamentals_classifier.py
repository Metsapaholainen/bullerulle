"""Peter Lynch-style growth categorization and the cross-setup "avoid buying
near earnings" filter -- both driven by the fundamentals dict shape produced
by data/fundamentals_cache.py.

Not a chart-pattern setup (hence living at the project root alongside
legout_scans.py/pattern_diagrams.py, not under setups/).

The Lynch classification is a best-effort simplification: real "cyclical"/
"turnaround"/"asset play" calls need sector context and multi-year history
this app doesn't fetch. From revenue/EPS growth alone, only the growth-rate
axis (fast grower vs stalwart vs slow grower) is really distinguishable --
documented here rather than pretending at false precision.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

LYNCH_CATEGORIES = {
    "fast_grower": "Fast Grower (25%+ earnings growth)",
    "stalwart": "Stalwart / Medium Grower (8-25% growth)",
    "slow_grower": "Slow Grower / Sluggard (~flat)",
    "turnaround": "Possible Turnaround (negative EPS growth, positive revenue growth)",
    "cyclical": "Cyclical / Declining",
    "unknown": "Unclassified (insufficient data)",
}


def classify_lynch(fundamentals: dict) -> str:
    """Returns a key into LYNCH_CATEGORIES."""
    if not fundamentals:
        return "unknown"
    eps_growth = fundamentals.get("eps_growth_pct")
    rev_growth = fundamentals.get("revenue_growth_pct")
    growth = eps_growth if eps_growth is not None else rev_growth
    if growth is None:
        return "unknown"
    if growth >= 25:
        return "fast_grower"
    if 8 <= growth < 25:
        return "stalwart"
    if -5 <= growth < 8:
        return "slow_grower"
    if growth < -5 and (rev_growth or 0) > 0:
        return "turnaround"
    return "cyclical"


def detect_earnings_deceleration(
    growth_history: list, threshold_pct: float = 66.7, metric: str = "growthEPS"
) -> dict:
    """O'Neil's "two quarters of major earnings deceleration" rule: before
    turning negative on a stock's earnings, look for TWO CONSECUTIVE
    quarter-over-quarter transitions where growth fell by `threshold_pct`%
    or more from its own prior quarter (his examples: 100% -> 30%, or 50% ->
    15% -- both ~a two-thirds decline). A single bad quarter isn't enough;
    only a stock is "decelerating" once BOTH of its two most recent
    transitions clear that bar.

    `growth_history` is a list of dicts, most-recent-quarter first, each
    with a `date` and the growth fraction under `metric` (e.g. FMP's
    `growthEPS`, a decimal like 0.35 for 35% -- NOT the *100 percentage
    fields used elsewhere in this module, since this comes straight off
    data/fundamentals_cache.py's raw quarterly history, not _fetch_fundamentals's
    single-quarter summary).

    A transition only counts as deceleration when the PRIOR quarter's growth
    was itself positive -- "decline of two-thirds from the previous rate"
    presumes coming down from a real growth rate, not comparing off a
    negative or zero base where the math is meaningless.

    Returns {"decelerating": bool, "quarters": [(date, growth_pct), ...] for
    the 3 quarters examined, "reason": human-readable str or None}."""
    rates = [(row.get("date"), row.get(metric)) for row in (growth_history or [])[:3]]
    if len(rates) < 3 or any(g is None for _, g in rates):
        return {"decelerating": False, "quarters": rates, "reason": None}

    (d0, g0), (d1, g1), (d2, g2) = rates  # most-recent-first

    def _decelerated(newer: float, older: float) -> bool:
        return older > 0 and newer <= older * (1 - threshold_pct / 100.0)

    transition_1 = _decelerated(g0, g1)  # most recent quarter vs the one before it
    transition_2 = _decelerated(g1, g2)  # that quarter vs the one before it
    decelerating = transition_1 and transition_2

    reason = None
    if decelerating:
        reason = (
            f"Growth decelerated {threshold_pct:.0f}%+ two quarters running: "
            f"{g2 * 100:.0f}% ({d2}) -> {g1 * 100:.0f}% ({d1}) -> {g0 * 100:.0f}% ({d0})"
        )
    return {"decelerating": decelerating, "quarters": rates, "reason": reason}


def days_to_earnings(symbol: str, earnings_dates: dict, as_of) -> Optional[int]:
    if not earnings_dates:
        return None
    next_date = earnings_dates.get(symbol)
    if not next_date:
        return None
    return int((pd.Timestamp(next_date) - pd.Timestamp(as_of)).days)


def filter_earnings_avoidance(rows_df: pd.DataFrame, earnings_dates: dict, as_of, avoid_days_before: int = 3) -> pd.DataFrame:
    """'Avoid buying 3-days before earnings' (Qullamaggie's Laws of Swing) --
    applies to ANY setup's scan output, not just Episodic Pivot. Drops rows
    whose symbol has an earnings date within `avoid_days_before` days after
    `as_of` (inclusive of today, i.e. 0 <= days_to <= avoid_days_before)."""
    if rows_df.empty or not earnings_dates:
        return rows_df
    as_of_ts = pd.Timestamp(as_of)

    def _too_close(symbol: str) -> bool:
        next_date = earnings_dates.get(symbol)
        if not next_date:
            return False
        days_to = (pd.Timestamp(next_date) - as_of_ts).days
        return 0 <= days_to <= avoid_days_before

    return rows_df[~rows_df["symbol"].map(_too_close)].reset_index(drop=True)
