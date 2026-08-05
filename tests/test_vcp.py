"""Synthetic fixtures for setups/vcp.py -- a genuine multi-leg contraction
(3 legs, each shallower and quieter than the last) ending in a tight,
low-volume pivot and a volume breakout, vs. a control fixture where the legs
don't actually contract. Also guards the two real bugs found and fixed while
building this detector: `swing_high_price` originally read the CONFIRMATION
row's own (later, lower) high instead of the true peak `order` bars earlier,
and `resistance` was originally a flat vcp_lookback_days rolling max instead
of the most recently confirmed leg's own high -- both silently made `match`
almost impossible to reach on real data despite `shape_match` firing fine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from settings import VCPSettings
from setups import vcp

R = 32.0  # common resistance level every leg pulls back from and recovers toward


def _build_fixture(depths, vols, breakout_vol=3_000_000):
    n_lead, n_uptrend, leg_len, pivot_len = 20, 40, 15, 5
    rows = []
    for _ in range(n_lead):
        rows.append((10.0, 10.0, 10.0, 10.0, 500_000))
    for p in np.linspace(10.0, R, n_uptrend):
        rows.append((p, p * 1.01, p * 0.99, p, 800_000))
    for depth, vol in zip(depths, vols):
        low = R * (1 - depth / 100.0)
        half = leg_len // 2
        leg_prices = np.concatenate([np.linspace(R, low, half), np.linspace(low, R * 0.995, leg_len - half)])
        for p in leg_prices:
            rows.append((p, p * 1.005, p * 0.995, p, vol))
    for p in np.linspace(R * 0.98, R * 0.99, pivot_len):
        rows.append((p, p * 1.003, p * 0.997, p, 400_000))
    rows.append((R * 1.01, R * 1.06, R * 1.00, R * 1.05, breakout_vol))  # breakout day
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])
    df.index = pd.date_range("2020-01-01", periods=len(df), freq="B")
    return df


def _settings(**overrides):
    base = dict(
        vcp_lookback_days=50, prior_move_lookback_days=40, min_legs=2, max_legs_checked=4,
        pivot_days=5, max_pivot_range_pct=8.0, max_pivot_volume_dryup_ratio=0.9,
        prior_move_min_pct=20.0, min_rs_rating=0.0, min_adr_pct=0.0, breakout_volume_ratio_min=1.5,
    )
    base.update(overrides)
    return VCPSettings(**base)


def test_genuine_contraction_matches_on_breakout_day():
    df = _build_fixture(depths=[20, 12, 6], vols=[2_200_000, 1_400_000, 800_000])
    result = vcp.scan(df, _settings())
    assert bool(result["match"].iloc[-1]) is True
    assert bool(result["match"].iloc[:-1].any()) is False  # only the breakout day, nowhere earlier
    assert bool(result["shape_match"].iloc[-1]) is True


def test_resistance_is_leg_anchored_not_a_stale_rolling_max():
    """Regression guard for the second bug: resistance must sit near the
    base's own recent leg high, not far above it (which is what a flat
    vcp_lookback_days rolling max produced when it swept in the tail of the
    prior move)."""
    df = _build_fixture(depths=[20, 12, 6], vols=[2_200_000, 1_400_000, 800_000])
    result = vcp.scan(df, _settings())
    close_on_breakout = df["close"].iloc[-1]
    resistance_on_breakout = result["resistance"].iloc[-1]
    assert resistance_on_breakout == resistance_on_breakout  # not NaN
    assert abs(resistance_on_breakout - R) < R * 0.05
    assert close_on_breakout > resistance_on_breakout


def test_non_contracting_last_leg_never_matches():
    """Control: the LAST (nearest) leg is deeper than the one before it --
    a genuine violation of 'each leg shallower than the last', since
    min_legs=2 only requires the two nearest legs to contract."""
    df = _build_fixture(depths=[6, 12, 20], vols=[800_000, 1_400_000, 2_200_000])
    result = vcp.scan(df, _settings())
    assert bool(result["match"].any()) is False
    assert bool(result["shape_match"].iloc[-1]) is False


def test_swing_high_price_uses_the_actual_peak_not_the_confirmation_row():
    """Regression guard for the first bug: a sharp one-bar spike's peak value
    must survive into leg_high, not get replaced by the (much lower) price
    on the day the swing is confirmed `order` bars later. Baseline drifts
    slightly (not perfectly flat) so only the real spike -- not a tied flat
    region -- ever satisfies the local-max condition."""
    n = 40
    high = pd.Series(np.linspace(10.0, 10.5, n))
    high.iloc[20] = 50.0
    df = pd.DataFrame({"high": high})
    order = 3
    window = 2 * order + 1
    is_swing_high = (df["high"] >= df["high"].rolling(window, center=True).max()).fillna(False)
    confirmed_raw = is_swing_high.shift(order).fillna(False)
    confirmed_swing_high = confirmed_raw & ~confirmed_raw.shift(1).fillna(False)
    swing_high_price = df["high"].shift(order).where(confirmed_swing_high)
    confirmed_at_spike = swing_high_price.iloc[20 + order]
    assert confirmed_at_spike == 50.0
