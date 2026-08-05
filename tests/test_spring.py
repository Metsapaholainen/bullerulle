"""Synthetic fixtures for setups/spring.py -- a quiet range, a brief undercut
of range support, and a reclaim within a few days on volume that did NOT
expand (the Wyckoff "spring"), vs. a control where the reclaim happens on
expanding volume (a genuine breakdown-turned-bounce, not a quiet spring).

Note on `range_low`/`range_high`: both are a trailing rolling window ending
YESTERDAY, so once the undercut day itself falls inside that window (i.e.
starting the very next bar), the reported range support mechanically drops
to include it -- this is real, intended behavior (support really did get
probed), not a bug, but it means a "hasn't reclaimed yet" day in the fixture
must close below the POST-undercut range low, not just the original one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from settings import SpringSettings
from setups import spring

SETTINGS = SpringSettings(
    range_lookback_days=40, max_range_pct=30.0, volume_dryup_ratio_max=0.85,
    penetration_min_pct=0.5, penetration_max_pct=8.0, reclaim_max_days=3,
    max_reclaim_volume_ratio=1.3, min_adr_pct=0.5, min_rs_rating=0.0,
)


def _build_fixture(
    undercut_low=92.0, undercut_close=93.0, mid_low=91.5, mid_close=91.8,
    reclaim_low=94.0, reclaim_close=96.0,
    undercut_vol=500_000, mid_vol=500_000, reclaim_vol=500_000,
    lead_vol=2_500_000,
):
    """40 quiet lead-in bars (elevated volume, establishing the stock's own
    normal average) -> 40 bars oscillating in a flat 95-100 range on quiet
    volume -> an undercut day -> a day that still hasn't reclaimed -> a
    reclaim day."""
    rows = []
    for _ in range(40):
        rows.append((155.0, 156.0, 154.0, 155.0, lead_vol))
    for t in np.linspace(0, 2 * np.pi, 40):
        p = 97.5 + 2.5 * np.sin(t)
        rows.append((p, min(p + 1.0, 100.0), max(p - 1.0, 95.0), p, 500_000))
    rows.append((94.0, 94.5, undercut_low, undercut_close, undercut_vol))
    rows.append((92.0, 93.0, mid_low, mid_close, mid_vol))
    rows.append((94.5, 97.0, reclaim_low, reclaim_close, reclaim_vol))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])
    df.index = pd.date_range("2020-01-01", periods=len(df), freq="B")
    return df


def test_reclaim_day_matches_and_prior_days_do_not():
    df = _build_fixture()
    result = spring.scan(df, SETTINGS)
    assert bool(result["match"].iloc[-1]) is True   # reclaim day
    assert bool(result["match"].iloc[-2]) is False  # still-below-support day
    assert bool(result["match"].iloc[-3]) is False  # undercut day itself
    assert bool(result["shape_match"].iloc[-1]) is True


def test_stop_reference_low_is_the_true_undercut_low_not_the_reclaim_bar():
    """The reclaim day's OWN low sits well above the true low reached during
    the spring -- resolve_entry_and_stop needs the actual undercut low, which
    stop_reference_low must expose."""
    df = _build_fixture()
    result = spring.scan(df, SETTINGS)
    reclaim_day_low = df["low"].iloc[-1]
    true_undercut_low = min(df["low"].iloc[-3], df["low"].iloc[-2], df["low"].iloc[-1])
    assert result["stop_reference_low"].iloc[-1] == true_undercut_low
    assert true_undercut_low < reclaim_day_low


def test_expanding_volume_reclaim_does_not_match():
    """Control: price action is identical, but the reclaim happens on 6x
    volume -- a genuine breakdown-turned-bounce, not a quiet Wyckoff spring."""
    df = _build_fixture(reclaim_vol=3_000_000)
    result = spring.scan(df, SETTINGS)
    assert bool(result["match"].iloc[-1]) is False
    assert result["reclaim_volume_ratio"].iloc[-1] > SETTINGS.max_reclaim_volume_ratio


def test_no_undercut_never_matches():
    """No day ever pierces range support at all -- there's nothing to
    reclaim, so `match` should never fire regardless of the later close."""
    df = _build_fixture(
        undercut_low=96.0, undercut_close=96.5, mid_low=96.2, mid_close=96.8, reclaim_low=96.3,
    )
    result = spring.scan(df, SETTINGS)
    assert bool(result["match"].any()) is False
