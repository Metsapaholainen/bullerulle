"""Synthetic fixtures for the two anti-crowd Breakout toggles (opt-in, off
by default): the recent-fakeout filter (a mechanical proxy for the Osler
order-clustering evidence) and the `confirmed_breakout` diagnostic (does the
breakout still hold a full day later, checked only against yesterday's
already-known resistance -- zero look-ahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from settings import BreakoutSettings
from setups import breakout

RESISTANCE = 100.0


def _build_base(with_fakeout: bool, breakout_vol: float = 2_000_000, fakeout_close: float = 96.0):
    """A prior move up to RESISTANCE, a tight 50-day base, an optional failed
    test of RESISTANCE partway through the base, then a genuine breakout on
    volume."""
    rows = []
    for _ in range(30):
        rows.append((60.0, 61.0, 59.0, 60.0, 500_000))
    for p in np.linspace(60.0, RESISTANCE, 63):
        rows.append((p, p * 1.01, p * 0.99, p, 800_000))

    def _base_day(i):
        p = 96.0 + (i % 5) * 0.3
        return (p, min(p + 0.5, RESISTANCE - 0.2), p - 0.5, p, 500_000)

    for i in range(40):
        rows.append(_base_day(i))
    if with_fakeout:
        # A wick that touches near resistance but closes well back below it --
        # a genuine failed test, not just noise.
        rows.append((98.0, 99.8, 95.8, fakeout_close, 900_000))
    for i in range(9 if with_fakeout else 10):
        rows.append(_base_day(i))
    rows.append((RESISTANCE * 1.005, RESISTANCE * 1.05, RESISTANCE * 0.999, RESISTANCE * 1.04, breakout_vol))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])
    df.index = pd.date_range("2020-01-01", periods=len(df), freq="B")
    return df


BASE_SETTINGS = dict(
    min_adr_pct=0.0, prior_move_lookback_days=63, prior_move_min_pct=10.0, prior_move_max_pct=200.0,
    min_consolidation_days=10, max_consolidation_days=40, max_consolidation_range_pct=15.0,
    require_above_ema10=False, require_above_ema20=False, min_rs_rating=0.0,
    breakout_volume_ratio_min=1.5,
)


def test_baseline_breakout_matches_with_filter_off():
    df = _build_base(with_fakeout=False)
    s = BreakoutSettings(**BASE_SETTINGS, fakeout_filter_action="off")
    result = breakout.scan(df, s)
    assert bool(result["match"].iloc[-1]) is True


def test_recent_fakeout_is_counted_and_excluded():
    df = _build_base(with_fakeout=True)
    s_exclude = BreakoutSettings(
        **BASE_SETTINGS, fakeout_filter_action="exclude",
        fakeout_lookback_days=60, fakeout_touch_tolerance_pct=1.5, fakeout_min_reversal_pct=1.0,
    )
    result = breakout.scan(df, s_exclude)
    assert result["recent_fakeout_count"].iloc[-1] >= 1
    assert bool(result["match"].iloc[-1]) is False  # excluded despite a genuine raw breakout

    # Same price action, but the filter is off -- diagnostic isn't computed,
    # and it doesn't block the match (never a reason to lose signals silently).
    s_off = BreakoutSettings(**BASE_SETTINGS, fakeout_filter_action="off")
    result_off = breakout.scan(df, s_off)
    assert bool(result_off["match"].iloc[-1]) is True


def test_penalize_docks_score_without_excluding():
    df = _build_base(with_fakeout=True)
    s_penalize = BreakoutSettings(
        **BASE_SETTINGS, fakeout_filter_action="penalize", fakeout_penalty_score=1.0,
        fakeout_lookback_days=60, fakeout_touch_tolerance_pct=1.5, fakeout_min_reversal_pct=1.0,
    )
    s_off = BreakoutSettings(**BASE_SETTINGS, fakeout_filter_action="off")
    result_penalize = breakout.scan(df, s_penalize)
    result_off = breakout.scan(df, s_off)
    assert bool(result_penalize["match"].iloc[-1]) is True  # still matches, just scores lower
    assert result_penalize["score"].iloc[-1] < result_off["score"].iloc[-1]


def _build_with_day_after(day_after_close: float):
    df = _build_base(with_fakeout=False)
    breakout_row = df.iloc[-1]
    low = min(breakout_row["close"], day_after_close) - 0.5
    extra = pd.DataFrame(
        [(breakout_row["close"], breakout_row["close"] * 1.02, low, day_after_close, 1_000_000)],
        columns=["open", "high", "low", "close", "volume"],
        index=[df.index[-1] + pd.tseries.offsets.BDay(1)],
    )
    return pd.concat([df, extra])


def test_confirmed_breakout_distinguishes_hold_from_failure_using_only_lagged_values():
    s = BreakoutSettings(**BASE_SETTINGS, fakeout_filter_action="off")

    df_hold = _build_with_day_after(day_after_close=RESISTANCE * 1.05)
    result_hold = breakout.scan(df_hold, s)
    assert bool(result_hold["confirmed_breakout"].iloc[-1]) is True

    df_fail = _build_with_day_after(day_after_close=RESISTANCE * 0.97)
    result_fail = breakout.scan(df_fail, s)
    assert bool(result_fail["confirmed_breakout"].iloc[-1]) is False


def test_require_close_confirmation_swaps_match_to_the_confirmed_column():
    s = BreakoutSettings(**BASE_SETTINGS, fakeout_filter_action="off", require_close_confirmation=True)

    df_hold = _build_with_day_after(day_after_close=RESISTANCE * 1.05)
    result_hold = breakout.scan(df_hold, s)
    assert bool(result_hold["match"].iloc[-1]) is True
    assert bool(result_hold["match"].iloc[-2]) is False  # the raw breakout day itself no longer matches

    df_fail = _build_with_day_after(day_after_close=RESISTANCE * 0.97)
    result_fail = breakout.scan(df_fail, s)
    assert bool(result_fail["match"].iloc[-1]) is False
