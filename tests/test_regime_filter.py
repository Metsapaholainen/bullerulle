"""Synthetic fixtures for scanner.compute_regime_series and run_backtest's
`regime_multiplier` kwarg: a calm benchmark stretch (size_multiplier stays
1.0) vs. a genuine drawdown+high-vol stretch (flagged 'crowded', downsized
or gated), plus an end-to-end check that run_backtest actually scales share
counts only on the dates the regime series says to.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from settings import BacktestSettings
from scanner import compute_regime_series
from backtest.engine import run_backtest

N_CALM = 300
N_STRESS = 60


def _build_benchmark():
    """A long calm stretch (small, gently varying daily range, no
    drawdown -- a monotonic drift up) followed by a genuine crash (steep
    decline, wide daily range). The calm section's range varies slightly
    day to day (not a constant) so percentile-ranking doesn't degenerate
    into tie-breaking noise on an artificially flat series."""
    rows = []
    price = 100.0
    for i in range(N_CALM):
        price *= 1.0002
        spread = 0.006 + 0.002 * np.sin(i * 0.31) + 0.0005 * np.cos(i * 0.17)
        rows.append((price, price * (1 + spread), price * (1 - spread), price, 1_000_000))
    for i in range(N_STRESS):
        price *= 0.99
        spread = 0.035 + 0.004 * np.sin(i * 0.4)
        rows.append((price, price * (1 + spread), price * (1 - spread), price, 2_000_000))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])
    df.index = pd.date_range("2018-01-01", periods=len(df), freq="B")
    return df


REGIME_SETTINGS = dict(
    regime_drawdown_lookback_days=252, regime_max_drawdown_pct=-15.0,
    regime_vol_lookback_days=20, regime_vol_percentile_window_days=252,
    regime_vol_percentile_threshold=85.0, regime_combine_mode="any",
)


def test_calm_section_is_not_flagged():
    bench = _build_benchmark()
    bt = BacktestSettings(**REGIME_SETTINGS, regime_action="downsize", regime_size_multiplier=0.5)
    regime = compute_regime_series(bench, bt)
    mid_calm = N_CALM - 150  # well clear of both the startup ramp and the crash boundary
    assert bool(regime["crowded"].iloc[mid_calm]) is False
    assert regime["size_multiplier"].iloc[mid_calm] == 1.0


def test_stress_section_is_flagged_and_downsized():
    bench = _build_benchmark()
    bt = BacktestSettings(**REGIME_SETTINGS, regime_action="downsize", regime_size_multiplier=0.5)
    regime = compute_regime_series(bench, bt)
    mid_stress = N_CALM + N_STRESS // 2
    assert bool(regime["crowded"].iloc[mid_stress]) is True
    assert regime["size_multiplier"].iloc[mid_stress] == 0.5
    assert regime["drawdown_pct"].iloc[mid_stress] < REGIME_SETTINGS["regime_max_drawdown_pct"]


def test_gate_mode_zeroes_the_multiplier_instead_of_scaling():
    bench = _build_benchmark()
    bt = BacktestSettings(**REGIME_SETTINGS, regime_action="gate")
    regime = compute_regime_series(bench, bt)
    mid_stress = N_CALM + N_STRESS // 2
    assert regime["size_multiplier"].iloc[mid_stress] == 0.0


def _build_signals(bench: pd.DataFrame, match_dates: list):
    sym = pd.DataFrame(
        {"open": 50.0, "high": 51.0, "low": 49.0, "close": 50.0, "volume": 1_000_000,
         "adr_pct": 2.0, "match": False},
        index=bench.index,
    )
    for d in match_dates:
        sym.loc[d, "match"] = True
    return {"TEST": sym}


def test_run_backtest_downsizes_shares_only_on_crowded_dates():
    bench = _build_benchmark()
    bt = BacktestSettings(
        **REGIME_SETTINGS, regime_action="downsize", regime_size_multiplier=0.4,
        entry_mode="next_open", stop_mode="adr_multiple", stop_adr_multiple=1.0,
        avoid_chase_adr_multiple=0.0, max_position_pct_of_avg_volume=0.0,
    )
    regime = compute_regime_series(bench, bt)
    mult = regime["size_multiplier"]

    calm_date = bench.index[N_CALM - 150]
    stress_date = bench.index[N_CALM + N_STRESS // 2]
    assert mult[calm_date] == 1.0
    assert mult[stress_date] == 0.4

    signals = _build_signals(bench, [calm_date, stress_date])
    baseline = run_backtest(bt, signals, side="long")
    regimed = run_backtest(bt, signals, side="long", regime_multiplier=mult)

    baseline_by_date = {t.entry_date: t.shares for t in baseline.trades}
    regimed_by_date = {t.entry_date: t.shares for t in regimed.trades}

    # entry_date is the bar AFTER the signal (next_open) -- match against
    # whichever calendar date each trade actually landed on.
    calm_entry = min(d for d in regimed_by_date if d > calm_date)
    stress_entry = min(d for d in regimed_by_date if d > stress_date)

    assert regimed_by_date[calm_entry] == baseline_by_date[calm_entry]  # untouched
    assert regimed_by_date[stress_entry] < baseline_by_date[stress_entry]  # downsized
    assert abs(regimed_by_date[stress_entry] / baseline_by_date[stress_entry] - 0.4) < 0.01


def test_run_backtest_gate_mode_drops_crowded_trades_entirely():
    bench = _build_benchmark()
    bt = BacktestSettings(
        **REGIME_SETTINGS, regime_action="gate",
        entry_mode="next_open", stop_mode="adr_multiple", stop_adr_multiple=1.0,
        avoid_chase_adr_multiple=0.0, max_position_pct_of_avg_volume=0.0,
    )
    regime = compute_regime_series(bench, bt)
    mult = regime["size_multiplier"]

    calm_date = bench.index[N_CALM - 150]
    stress_date = bench.index[N_CALM + N_STRESS // 2]

    signals = _build_signals(bench, [calm_date, stress_date])
    result = run_backtest(bt, signals, side="long", regime_multiplier=mult)

    entry_dates = [t.entry_date for t in result.trades]
    assert any(d > calm_date and d <= calm_date + pd.Timedelta(days=5) for d in entry_dates)
    assert not any(d > stress_date and d <= stress_date + pd.Timedelta(days=5) for d in entry_dates)
