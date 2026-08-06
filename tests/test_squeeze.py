"""Synthetic fixtures for setups/squeeze.py -- fires on the QUIET/COILED
day, BEFORE the breakout prints: a genuine prior advance into a tight base
that drifts quietly higher (range tight vs. its own ADR, above a rising
MA), staying below its own recent pivot high. Controls: a base drifting
DOWN (falling MA -- not a Squeeze candidate) and a base that's already
closed above its pivot high (that's Breakout's day, not Squeeze's).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from settings import SqueezeSettings
from setups import squeeze

PEAK = 100.0
SETTINGS = SqueezeSettings(
    min_rs_rating=0.0, ma_period=20, require_above_ma=True, require_rising_ma=True,
    ma_slope_lookback_days=10, range_window_days=3, adr_lookback_days=20,
    max_range_to_adr_ratio=1.5, pivot_lookback_days=20, prior_move_lookback_days=70,
    min_prior_move_pct=20.0, min_adr_pct=0.0,
)


def _build_fixture(base_end_frac=0.95, coil_end_frac=0.97, coil_days=6, coil_amp=0.5):
    """40 quiet lead-in bars -> a 40-day genuine advance up to PEAK -> a
    20-day base drifting toward `base_end_frac` * PEAK -> a `coil_days`-day
    tight coil drifting toward `coil_end_frac` * PEAK, staying under PEAK."""
    rows = []
    price = 50.0
    for _ in range(40):
        rows.append((price, price * 1.01, price * 0.99, price, 500_000))
    for p in np.linspace(50.0, PEAK, 40):
        rows.append((p, p * 1.008, p * 0.992, p, 700_000))
    base_start = PEAK * 0.90
    base_end = PEAK * base_end_frac
    for p in np.linspace(base_start, base_end, 20):
        rows.append((p, p + 1.5, p - 1.5, p, 500_000))
    for p in np.linspace(base_end, PEAK * coil_end_frac, coil_days):
        rows.append((p, p + coil_amp, p - coil_amp, p, 400_000))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])
    df.index = pd.date_range("2020-01-01", periods=len(df), freq="B")
    return df


def test_quiet_coil_under_rising_ma_matches():
    df = _build_fixture()
    result = squeeze.scan(df, SETTINGS)
    assert bool(result["match"].iloc[-1]) is True
    assert bool(result["shape_match"].iloc[-1]) is True
    assert result["range_to_adr_ratio"].iloc[-1] <= SETTINGS.max_range_to_adr_ratio
    assert df["close"].iloc[-1] <= result["pivot_high"].iloc[-1]


def test_falling_ma_does_not_match():
    """Same coil shape, but the base/coil drift DOWN instead of up -- the
    50-day MA is falling, so this isn't a Squeeze candidate."""
    df = _build_fixture(base_end_frac=0.85, coil_end_frac=0.83)
    result = squeeze.scan(df, SETTINGS)
    assert bool(result["match"].iloc[-1]) is False
    assert result["ma_slope_pct"].iloc[-1] < 0


def test_already_broken_out_does_not_match():
    """The coil drifts up and closes ABOVE the established pivot high --
    that's a Breakout day, not a Squeeze day. Squeeze and Breakout can
    never match the same stock on the same day, by design."""
    df = _build_fixture(coil_end_frac=1.05)
    result = squeeze.scan(df, SETTINGS)
    assert bool(result["match"].iloc[-1]) is False
    assert df["close"].iloc[-1] > result["pivot_high"].iloc[-1]


def test_no_duplicate_shape_match_bug_range_window_causality():
    """Sanity check the range window is causal -- shape_match on the final
    coil day shouldn't depend on data that hasn't happened yet (there's
    nothing after the last row, so this just confirms scan() doesn't error
    or produce NaN=truthy surprises at the edge of history)."""
    df = _build_fixture()
    result = squeeze.scan(df, SETTINGS)
    assert result["match"].iloc[-1] in (True, False)
    assert not result["match"].isna().any()
