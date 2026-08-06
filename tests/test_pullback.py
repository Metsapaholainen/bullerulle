"""Synthetic fixtures for setups/pullback.py -- buy the FIRST shallow
pullback to a rising EMA in an already-extended leader, on light volume.
Controls: the same pullback on heavy volume (a real breakdown, not sellers
exhausted) and pure chop that never extended away from the EMA in the
first place (nothing to "pull back" from).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from settings import PullbackSettings
from setups import pullback

SETTINGS = PullbackSettings(
    ema_period=20, ema_slope_lookback_days=10, min_ema_slope_pct=0.0, min_rs_rating=0.0,
    extension_lookback_days=20, min_extension_adr_multiple=2.0,
    max_pullback_distance_adr_multiple=1.0, min_bars_since_last_touch=10,
    volume_avg_period=50, pullback_window_days=3, max_pullback_volume_ratio=0.9,
    adr_lookback_days=20, min_adr_pct=0.0,
)


def _build_fixture(pullback_vol=400_000, pullback_price=76.0):
    """40 quiet lead-in bars -> a 30-day extended run (50 -> 80, well above
    the 20-EMA) -> 5 more days holding near the highs -> a pullback day
    dipping toward the (still-rising) EMA."""
    rows = []
    for _ in range(40):
        rows.append((50.0, 50.5, 49.5, 50.0, 700_000))
    for p in np.linspace(50.0, 80.0, 30):
        rows.append((p, p * 1.01, p * 0.995, p, 900_000))
    for p in np.linspace(80.0, 82.0, 5):
        rows.append((p, p * 1.01, p * 0.995, p, 800_000))
    rows.append((pullback_price * 1.01, pullback_price * 1.02, pullback_price * 0.98, pullback_price, pullback_vol))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])
    df.index = pd.date_range("2020-01-01", periods=len(df), freq="B")
    return df


def _build_chop_fixture():
    """Flat chop that never extends away from its own EMA -- nothing to
    pull back FROM."""
    rows = []
    price = 70.0
    for i in range(80):
        p = price + (i % 5) * 0.3
        rows.append((p, p + 0.5, p - 0.5, p, 500_000))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])
    df.index = pd.date_range("2020-01-01", periods=len(df), freq="B")
    return df


def test_light_volume_pullback_to_rising_ema_matches():
    df = _build_fixture()
    result = pullback.scan(df, SETTINGS)
    assert bool(result["match"].iloc[-1]) is True
    assert bool(result["shape_match"].iloc[-1]) is True
    assert result["pullback_distance_adr"].iloc[-1] <= SETTINGS.max_pullback_distance_adr_multiple
    assert result["max_extension_adr"].iloc[-1] >= SETTINGS.min_extension_adr_multiple


def test_heavy_volume_pullback_does_not_match():
    """Same price action, but 7.5x the volume -- a real breakdown, not a
    quiet, sellers-exhausted pullback."""
    df = _build_fixture(pullback_vol=3_000_000)
    result = pullback.scan(df, SETTINGS)
    assert bool(result["match"].iloc[-1]) is False
    assert result["pullback_volume_ratio"].iloc[-1] > SETTINGS.max_pullback_volume_ratio


def test_chop_with_no_prior_extension_does_not_match():
    """Never extended away from its own EMA, so there's nothing to pull
    back FROM -- sitting near the EMA the whole time isn't a pullback."""
    df = _build_chop_fixture()
    result = pullback.scan(df, SETTINGS)
    assert bool(result["match"].iloc[-1]) is False
    assert result["max_extension_adr"].iloc[-1] < SETTINGS.min_extension_adr_multiple


def test_second_touch_within_the_window_does_not_match():
    """The FIRST touch should match; sitting near the EMA again a couple
    days later (well inside min_bars_since_last_touch) should not."""
    df = _build_fixture()
    extra = pd.DataFrame(
        [(76.5, 77.0, 75.5, 76.3, 400_000)],
        columns=["open", "high", "low", "close", "volume"],
        index=[df.index[-1] + pd.tseries.offsets.BDay(1)],
    )
    df2 = pd.concat([df, extra])
    result = pullback.scan(df2, SETTINGS)
    assert bool(result["match"].iloc[-2]) is True   # the first touch
    assert bool(result["match"].iloc[-1]) is False  # too soon after the first touch
