"""Grid search over parameter combinations for a given setup, each combo
backtested across the full universe/history and ranked by performance.

This is the "which of these various settings actually hold up historically"
tool -- distinct from the Designer tab (one setting at a time, instant
feedback) and the Parameter Finder (reverse-fit to a single known example).
"""
from __future__ import annotations

import itertools
from typing import Callable, Optional

import pandas as pd

from backtest.engine import run_backtest
from backtest.stats import compute_stats
from scanner import scan_signals_over_history
from settings import Settings
from setups import SETUP_REGISTRY


def set_nested(settings: Settings, dotted_key: str, value) -> None:
    section, attr = dotted_key.split(".", 1)
    setattr(getattr(settings, section), attr, value)


def run_grid_search(
    base_settings: Settings,
    history: dict,
    setup_name: str,
    param_grid: dict,
    on_progress: Optional[Callable] = None,
) -> pd.DataFrame:
    """`param_grid` example: {"breakout.min_adr_pct": [3, 4, 5],
    "breakout.min_rs_rating": [80, 90, 95]}. Returns one row per combination
    with the params used plus backtest stats, ranked by expectancy (R)."""
    if setup_name not in SETUP_REGISTRY:
        raise ValueError(f"Unknown setup: {setup_name}")
    side = SETUP_REGISTRY[setup_name]["side"]

    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values())) if keys else [()]

    rows = []
    for i, combo in enumerate(combos):
        settings = base_settings.copy()
        for key, value in zip(keys, combo):
            set_nested(settings, key, value)

        signals = scan_signals_over_history(settings, history, setup_name)
        # setup_name matters: it's what activates bt.ep_stop_mode_override for
        # episodic_pivot. Omitting it here meant every grid search silently
        # ignored that override while the single-run backtest honoured it,
        # so the two disagreed on the same settings.
        result = run_backtest(settings.backtest, signals, side=side, setup_name=setup_name)
        stats = compute_stats(result)

        row = dict(zip(keys, combo))
        row.update(stats)
        rows.append(row)

        if on_progress:
            on_progress(i + 1, len(combos), dict(zip(keys, combo)))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.sort_values("expectancy_r", ascending=False, na_position="last").reset_index(drop=True)
