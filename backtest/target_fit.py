"""Reverse parameter finder: given a symbol and the approximate date of a
known good move, (1) report the setup's diagnostic indicator values that
stock actually had at that date, then (2) search threshold combinations
that would flag it, ranked by how well they perform across the broader
universe/history -- not just this one example -- so a lucky one-stock fit
doesn't get recommended without also showing it's an overfit.

Scope: this searches the *threshold* parameters (ADR% floor, RS floor,
volume-ratio floor, consolidation-tightness ceiling, etc.), not the
underlying window/period lengths (e.g. how many days count as "the
consolidation") -- those stay fixed at whatever the base settings specify,
since varying both simultaneously blows up the search space far beyond
what's useful interactively. Change the base settings' window/period
fields yourself (e.g. in the Designer tab) if you want to explore those too.
"""
from __future__ import annotations

import itertools
from typing import Optional

import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from backtest.optimizer import set_nested
from backtest.stats import compute_stats
from indicators import add_core_indicators, build_close_panel, compute_rs_rating_panel
from scanner import scan_signals_over_history
from settings import Settings
from setups import SETUP_REGISTRY

# Which settings fields are "thresholds" worth searching per setup, paired
# with (kind, diagnostic_column) -- kind "floor" means the setting is a
# minimum (>=), "ceiling" means it's a maximum (<=).
THRESHOLD_SPECS = {
    "breakout": {
        "min_adr_pct": ("floor", "adr_pct"),
        "prior_move_min_pct": ("floor", "prior_move_pct"),
        "max_consolidation_range_pct": ("ceiling", "base_range_pct"),
        "breakout_volume_ratio_min": ("floor", "volume_ratio"),
        "min_rs_rating": ("floor", "rs_rating"),
    },
    "episodic_pivot": {
        "min_gap_pct": ("floor", "gap_pct"),
        "min_volume_ratio": ("floor", "volume_ratio"),
        "max_prior_range_pct": ("ceiling", "prior_range_pct"),
    },
    "parabolic_short": {
        "min_extension_adr_multiple": ("floor", "extension_adr_multiple"),
        "min_run_up_pct": ("floor", "run_up_pct"),
    },
    "cup_with_handle": {
        "min_cup_depth_pct": ("floor", "cup_depth_pct"),
        "max_handle_depth_pct": ("ceiling", "handle_depth_pct"),
        "min_recovery_pct": ("floor", "recovery_pct"),
        "min_prior_uptrend_pct": ("floor", "prior_uptrend_pct"),
        "breakout_volume_ratio_min": ("floor", "volume_ratio"),
    },
    "double_bottom": {
        "max_low_difference_pct": ("ceiling", "low_difference_pct"),
        "breakout_volume_ratio_min": ("floor", "volume_ratio"),
    },
    "flat_base": {
        "max_range_pct": ("ceiling", "range_pct"),
        "min_prior_move_pct": ("floor", "prior_move_pct"),
        "breakout_volume_ratio_min": ("floor", "volume_ratio"),
    },
    "ascending_base": {
        "max_segment_depth_pct": ("ceiling", "max_segment_depth_pct"),
        "breakout_volume_ratio_min": ("floor", "volume_ratio"),
    },
    "high_tight_flag": {
        "min_run_up_pct": ("floor", "run_up_pct"),
        "max_flag_depth_pct": ("ceiling", "flag_depth_pct"),
        "breakout_volume_ratio_min": ("floor", "volume_ratio"),
    },
}


def inspect_target(
    target_df: pd.DataFrame,
    settings: Settings,
    setup_name: str,
    target_date,
    rs_rating: Optional[pd.Series] = None,
) -> dict:
    """Report the setup's diagnostic indicator values for one symbol on one
    date, using the base settings' window/period definitions."""
    spec = SETUP_REGISTRY[setup_name]
    setup_settings = getattr(settings, spec["settings_attr"])
    enriched = add_core_indicators(target_df)
    result = spec["module"].scan(enriched, setup_settings, rs_rating=rs_rating)

    target_ts = pd.Timestamp(target_date)
    if target_ts not in result.index:
        loc = result.index.get_indexer([target_ts], method="nearest")[0]
        target_ts = result.index[loc]

    row = result.loc[target_ts].to_dict()
    row["date_used"] = target_ts
    row["matched_with_current_settings"] = bool(row.get("match", False))
    return row


def _candidate_values(kind: str, observed: float, n: int = 4) -> list:
    if observed is None or not np.isfinite(observed):
        return []
    if kind == "floor":
        # thresholds at/just-below the observed value: a floor == observed
        # just barely still catches the target; lower floors are looser
        # and more likely (or less likely) to generalize.
        lo, hi = observed * 0.5, observed
    else:  # ceiling
        lo, hi = observed, observed * 1.5
    lo, hi = min(lo, hi), max(lo, hi)
    return sorted(set(np.round(np.linspace(lo, hi, n), 2)))


def find_parameters_for_target(
    base_settings: Settings,
    history: dict,
    setup_name: str,
    target_symbol: str,
    target_date,
    n_candidates_per_param: int = 2,
    top_k: int = 10,
    max_combinations: int = 200,
    on_progress=None,
) -> dict:
    """Full reverse-fit workflow. Returns {"observed": dict, "candidates":
    DataFrame} where `candidates` has one row per searched combination with
    its params, whether it still catches the target, and its broad-universe
    backtest stats."""
    if target_symbol not in history:
        raise ValueError(f"{target_symbol} not found in the loaded history")

    spec = SETUP_REGISTRY[setup_name]
    side = spec["side"]
    threshold_specs = THRESHOLD_SPECS[setup_name]

    close_panel = build_close_panel(history)
    rs_panel = compute_rs_rating_panel(
        close_panel, base_settings.rs_rating.lookback_periods, base_settings.rs_rating.period_weights
    )
    rs_series = rs_panel[target_symbol] if target_symbol in rs_panel.columns else None

    observed = inspect_target(history[target_symbol], base_settings, setup_name, target_date, rs_rating=rs_series)

    param_grid = {}
    for param_name, (kind, diag_key) in threshold_specs.items():
        candidates = _candidate_values(kind, observed.get(diag_key), n=n_candidates_per_param)
        if candidates:
            param_grid[f"{spec['settings_attr']}.{param_name}"] = candidates

    if not param_grid:
        return {"observed": observed, "candidates": pd.DataFrame()}

    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    if len(combos) > max_combinations:
        # Full Cartesian product across every threshold can blow up fast
        # (5 breakout params x n candidates each) -- evenly subsample rather
        # than silently grinding through thousands of near-duplicate combos.
        step = len(combos) / max_combinations
        combos = [combos[int(i * step)] for i in range(max_combinations)]

    rows = []
    for i, combo in enumerate(combos):
        settings = base_settings.copy()
        for key, value in zip(keys, combo):
            set_nested(settings, key, value)

        target_check = inspect_target(
            history[target_symbol], settings, setup_name, observed["date_used"], rs_rating=rs_series
        )
        catches_target = bool(target_check.get("matched_with_current_settings", False))

        signals = scan_signals_over_history(settings, history, setup_name)
        result = run_backtest(settings.backtest, signals, side=side)
        stats = compute_stats(result)

        row = dict(zip(keys, combo))
        row["catches_target"] = catches_target
        row.update(stats)
        rows.append(row)

        if on_progress:
            on_progress(i + 1, len(combos), dict(zip(keys, combo)))

    candidates_df = pd.DataFrame(rows)
    if not candidates_df.empty:
        # Overfitting guard: rank combos that catch the target first, but
        # among those, by broad-universe expectancy -- so a combo that only
        # works for this one stock and performs poorly elsewhere sorts low
        # instead of being silently recommended.
        candidates_df = candidates_df.sort_values(
            ["catches_target", "expectancy_r"], ascending=[False, False], na_position="last"
        ).reset_index(drop=True)
        candidates_df = candidates_df.head(top_k)

    return {"observed": observed, "candidates": candidates_df}
