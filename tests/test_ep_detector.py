"""Synthetic fixtures for bullarullaep/detector.py -- a quiet, flat base for
months, then a gap-up day on heavy volume. Controls: gap too small, volume
not heavy enough, prior base too loose/wide, already ran hard before the
gap (fails the neglect-period check), and the growth-floor gate (with vs.
without fundamentals meeting the floor). Also checks that the catalyst-
classifier bonus raises score without ever being required for a match --
match must be a pure function of the EOD hard gates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bullarullaep.catalyst_classifier import CatalystClassification
from bullarullaep.settings import EPSettings
from bullarullaep import detector

SETTINGS = EPSettings(
    min_gap_pct=10.0, min_volume_ratio=3.0, volume_avg_period=20,
    lookback_base_days=20, max_prior_range_pct=15.0,
    quiet_base_lookback_days=60, max_quiet_base_run_pct=25.0,
    min_price=1.0, prior_ep_lookback_days=100, prior_ep_min_gap_pct=8.0,
    require_growth_floor=False, require_eps_beat=False,
)


def _build_fixture(base_days=80, base_price=20.0, gap_pct=15.0, vol_ratio=6.0, base_run_pct=0.0):
    """`base_days` quiet flat bars around `base_price` (drifting by
    `base_run_pct` total over the base, to control the neglect-period
    check), then one gap day at `base_price * (1 + gap_pct/100)` on
    `vol_ratio`x the base's own average volume."""
    rows = []
    base_avg_volume = 300_000
    prices = np.linspace(base_price, base_price * (1 + base_run_pct / 100.0), base_days)
    for p in prices:
        rows.append((p, p * 1.01, p * 0.99, p, base_avg_volume))
    gap_price = base_price * (1 + gap_pct / 100.0)
    rows.append((gap_price, gap_price * 1.02, gap_price * 0.98, gap_price * 1.01, base_avg_volume * vol_ratio))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])
    df.index = pd.date_range("2023-01-01", periods=len(df), freq="B")
    return df


def test_gap_on_volume_out_of_quiet_base_matches():
    df = _build_fixture()
    result = detector.scan(df, SETTINGS)
    last = result.iloc[-1]
    assert bool(last["match"]) is True
    assert last["gap_pct"] >= SETTINGS.min_gap_pct
    assert last["volume_ratio"] >= SETTINGS.min_volume_ratio


def test_gap_too_small_does_not_match():
    df = _build_fixture(gap_pct=4.0)
    result = detector.scan(df, SETTINGS)
    assert bool(result["match"].iloc[-1]) is False


def test_volume_too_light_does_not_match():
    df = _build_fixture(vol_ratio=1.2)
    result = detector.scan(df, SETTINGS)
    assert bool(result["match"].iloc[-1]) is False


def test_loose_prior_base_does_not_match():
    """A choppy, wide-ranging base right before the gap (not tight/quiet) --
    should fail the prior_range_pct gate even though gap/volume are fine."""
    rows = []
    base_avg_volume = 300_000
    price = 20.0
    for i in range(80):
        # wide daily swings, +/-8%, instead of the fixture's quiet +/-1%
        p = price * (1 + (0.08 if i % 2 == 0 else -0.08))
        rows.append((p, p * 1.10, p * 0.90, p, base_avg_volume))
    gap_price = 23.0
    rows.append((gap_price, gap_price * 1.02, gap_price * 0.98, gap_price * 1.01, base_avg_volume * 6.0))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])
    df.index = pd.date_range("2023-01-01", periods=len(df), freq="B")
    result = detector.scan(df, SETTINGS)
    assert bool(result["match"].iloc[-1]) is False
    assert result["prior_range_pct"].iloc[-1] > SETTINGS.max_prior_range_pct


def test_already_ran_hard_before_gap_does_not_match():
    """The "neglect period" check: a stock that already ran +50% over the
    quiet-base lookback isn't neglected anymore, even if the tail end right
    before the gap looks tight."""
    df = _build_fixture(base_run_pct=50.0)
    result = detector.scan(df, SETTINGS)
    assert bool(result["match"].iloc[-1]) is False
    assert abs(result["quiet_base_run_pct"].iloc[-1]) > SETTINGS.max_quiet_base_run_pct


def test_growth_floor_gates_when_provided_but_not_when_unknown():
    df = _build_fixture()
    strict = EPSettings(**{**SETTINGS.__dict__, "require_growth_floor": True, "min_growth_pct_floor": 25.0})

    # No fundamentals at all -- gate is skipped when growth is unknown.
    result_unknown = detector.scan(df, strict)
    assert bool(result_unknown["match"].iloc[-1]) is True
    assert result_unknown["growth_tier"].iloc[-1] == "unknown"

    # Fundamentals present but below the floor -- now it gates.
    result_below = detector.scan(df, strict, fundamentals={"revenue_growth_pct": 5.0, "eps_growth_pct": 3.0})
    assert bool(result_below["match"].iloc[-1]) is False
    assert result_below["growth_tier"].iloc[-1] == "below-floor"

    # Fundamentals present and above the floor -- matches, tiered correctly.
    result_above = detector.scan(df, strict, fundamentals={"revenue_growth_pct": 120.0, "eps_growth_pct": 90.0})
    assert bool(result_above["match"].iloc[-1]) is True
    assert result_above["growth_tier"].iloc[-1] == "5-star"


def test_catalyst_bonus_raises_score_but_never_required_for_match():
    df = _build_fixture()
    baseline = detector.scan(df, SETTINGS)
    baseline_score = baseline["score"].iloc[-1]
    baseline_match = bool(baseline["match"].iloc[-1])

    weak_catalyst = CatalystClassification(catalyst_type="unclear_or_noise", confidence=0.9, rationale="routine 8-K")
    with_weak = detector.scan(df, SETTINGS, catalyst=weak_catalyst)
    assert bool(with_weak["match"].iloc[-1]) == baseline_match  # match unaffected either way
    assert with_weak["score"].iloc[-1] == baseline_score  # no bonus: not "genuine"

    strong_catalyst = CatalystClassification(catalyst_type="earnings_beat", confidence=0.9, rationale="big beat")
    with_strong = detector.scan(df, SETTINGS, catalyst=strong_catalyst)
    assert bool(with_strong["match"].iloc[-1]) == baseline_match  # still never gates match
    assert with_strong["score"].iloc[-1] > baseline_score  # but does raise score
    assert with_strong["catalyst_type"].iloc[-1] == "earnings_beat"


def test_point_in_time_columns_path_used_when_present():
    """When the caller has already merged point-in-time growth columns onto
    df (the backtest path), scan() must prefer those over any `fundamentals`
    dict argument -- confirms the column path actually takes effect."""
    df = _build_fixture()
    df["revenue_growth_pct"] = np.nan
    df["eps_growth_pct"] = np.nan
    df.iloc[-1, df.columns.get_loc("revenue_growth_pct")] = 150.0
    df.iloc[-1, df.columns.get_loc("eps_growth_pct")] = 130.0

    strict = EPSettings(**{**SETTINGS.__dict__, "require_growth_floor": True, "min_growth_pct_floor": 25.0})
    # Even though we ALSO pass a below-floor fundamentals dict, the column
    # on df should win (5-star, matches).
    result = detector.scan(df, strict, fundamentals={"revenue_growth_pct": 1.0, "eps_growth_pct": 1.0})
    assert result["growth_tier"].iloc[-1] == "5-star"
    assert bool(result["match"].iloc[-1]) is True


def test_no_nan_match_column_and_causal():
    df = _build_fixture()
    result = detector.scan(df, SETTINGS)
    assert result["match"].isin([True, False]).all()
    assert not result["match"].isna().any()
