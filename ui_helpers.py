"""Color-coded, beginner-friendly framing for backtest numbers.

Turns raw stats (win rate, expectancy, CAGR, drawdown, Sharpe) into a
green/yellow/red signal using common rule-of-thumb thresholds for swing
trading systems. These are heuristics to help someone new to backtesting
get a quick read, not a guarantee -- a green number means "historically
reasonable for this kind of strategy," not "will win."
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

GOOD = "#2ecc71"
OK = "#f1c40f"
BAD = "#e74c3c"
NEUTRAL = "#888888"

# metric -> (good_cutoff, ok_cutoff). For every metric here, value >= good_cutoff
# is green, value >= ok_cutoff is yellow, otherwise red. This works uniformly
# for "higher is better" metrics (win rate, expectancy, CAGR, Sharpe) and for
# max_drawdown_pct too, since drawdown is stored as a negative number and
# "less negative" (closer to zero) is better -- so its cutoffs are just
# ordered the same way (-15 > -30).
METRIC_RULES = {
    "win_rate": (50.0, 35.0),
    "expectancy_r": (0.3, 0.0),
    "avg_r": (0.2, 0.0),
    "cagr": (20.0, 0.0),
    "sharpe": (1.0, 0.5),
    "max_drawdown_pct": (-15.0, -30.0),
}

METRIC_LABELS = {
    "win_rate": "Win rate",
    "expectancy_r": "Expectancy",
    "avg_r": "Avg R",
    "cagr": "CAGR",
    "sharpe": "Sharpe",
    "max_drawdown_pct": "Max drawdown",
}

METRIC_TIPS = {
    "win_rate": "Share of trades that were profitable. Above 50% is strong for a breakout system, "
    "but a lower win rate can still be profitable if winners are much bigger than losers -- check Expectancy too.",
    "expectancy_r": "Average R (multiple of risk) gained or lost per trade. Positive means the system "
    "makes money on average across these trades -- this is the single most important number here.",
    "avg_r": "Average R multiple per trade (similar to expectancy, unweighted).",
    "cagr": "Annualized growth rate of the backtest's equity curve.",
    "sharpe": "Return per unit of volatility/risk taken. Above 1 is generally considered good; below 0.5 is weak.",
    "max_drawdown_pct": "The largest peak-to-trough decline in equity during the backtest. Smaller "
    "(closer to 0) is better -- e.g. -50% means the account's value was cut in half at some point.",
}


def color_for(metric: str, value) -> str:
    if metric not in METRIC_RULES or value is None:
        return NEUTRAL
    if isinstance(value, float) and not np.isfinite(value):
        return NEUTRAL
    good_cutoff, ok_cutoff = METRIC_RULES[metric]
    if value >= good_cutoff:
        return GOOD
    elif value >= ok_cutoff:
        return OK
    else:
        return BAD


def render_legend() -> None:
    st.caption(
        f"🟢 :green[good]  🟡 :orange[mixed / borderline]  🔴 :red[weak] -- rule-of-thumb thresholds "
        "for swing-trading systems, not guarantees. Expand 'What do these numbers mean?' below for details."
    )


def render_stat_badges(stats: dict) -> None:
    """Render win rate / expectancy / CAGR / max DD / Sharpe as colored cards."""
    keys = ["win_rate", "expectancy_r", "cagr", "max_drawdown_pct", "sharpe"]
    display = {
        "win_rate": f"{stats['win_rate']:.1f}%" if pd.notna(stats.get("win_rate")) else "-",
        "expectancy_r": f"{stats['expectancy_r']:.2f}R" if pd.notna(stats.get("expectancy_r")) else "-",
        "cagr": f"{stats['cagr']:.1f}%" if pd.notna(stats.get("cagr")) else "-",
        "max_drawdown_pct": f"{stats['max_drawdown_pct']:.1f}%" if pd.notna(stats.get("max_drawdown_pct")) else "-",
        "sharpe": f"{stats['sharpe']:.2f}" if pd.notna(stats.get("sharpe")) else "-",
    }

    render_legend()
    cols = st.columns(len(keys))
    for col, key in zip(cols, keys):
        raw_value = stats.get(key)
        color = color_for(key, raw_value) if pd.notna(raw_value) else NEUTRAL
        col.markdown(
            f"""<div style="border:1px solid {color}; border-radius:8px; padding:8px; text-align:center;">
<div style="font-size:0.8em; opacity:0.8;">{METRIC_LABELS[key]}</div>
<div style="font-size:1.4em; font-weight:700; color:{color};">{display[key]}</div>
</div>""",
            unsafe_allow_html=True,
        )
    st.caption(f"Trades: {stats.get('trade_count', 0)}")

    with st.expander("What do these numbers mean?"):
        for key in keys:
            st.markdown(f"**{METRIC_LABELS[key]}**: {METRIC_TIPS[key]}")


def style_results_dataframe(df: pd.DataFrame):
    """Return a pandas Styler that background-colors any recognized metric
    columns (win_rate, expectancy_r, cagr, max_drawdown_pct, sharpe, ...) so
    grid-search / candidate tables get the same green/yellow/red read."""
    if df is None or df.empty:
        return df
    color_cols = [c for c in df.columns if c in METRIC_RULES]
    if not color_cols:
        return df

    def _style_column(series: pd.Series, metric: str):
        return [f"background-color: {color_for(metric, v)}; color: black;" for v in series]

    styler = df.style
    for col in color_cols:
        styler = styler.apply(lambda s, m=col: _style_column(s, m), subset=[col])
    return styler
