"""Streamlit app: Scanner / Designer / Backtest & Optimizer / Parameter Finder / Settings.

Data loading (network-bound) is an explicit sidebar action; every tab below
operates on the already-cached `st.session_state["history"]` and recomputes
indicators/setups/backtests in pandas, which is fast enough to feel
interactive when you tweak a setting.
"""
from __future__ import annotations

import traceback

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import presets
from backtest.engine import run_backtest
from backtest.optimizer import run_grid_search, set_nested
from backtest.stats import compute_stats, stats_summary_text
from backtest.target_fit import find_parameters_for_target
from data.cache import apply_live_quotes, get_history_bulk
from data.fmp_client import FMPClient, FMPError
from scanner import (
    DEFAULT_BENCHMARK_SYMBOL,
    SECONDARY_BENCHMARK_SYMBOL,
    TIMEFRAME_LABELS,
    ensure_benchmark_loaded,
    find_approaching_pivot,
    history_lookback_days,
    load_scan_universe_history,
    market_direction,
    momentum_shortlist,
    relative_strength_resilience,
    run_scan,
    scan_signals_over_history,
    top_movers_by_timeframe,
)
from data.fundamentals_cache import get_earnings_dates, get_fundamentals_bulk
from fundamentals_classifier import classify_lynch, days_to_earnings, filter_earnings_avoidance, LYNCH_CATEGORIES
from legout_scans import LEGOUT_SCANS, SKIPPED_SCANS, run_legout_scans
from notifications import send_ntfy_alert, send_sell_alerts
from paper_trading import (
    join_results as pt_join_results,
    load_rules as pt_load_rules,
    load_trades as pt_load_trades,
    save_rules as pt_save_rules,
    save_trades as pt_save_trades,
    simulate as pt_simulate,
)
from pattern_diagrams import get_pattern_diagram as _get_pattern_diagram_uncached


@st.cache_data(show_spinner=False)
def get_pattern_diagram(setup_key: str):
    # These are pure, deterministic schematics (no randomness, no external
    # data) but were being rebuilt on every single rerun anywhere in the
    # app -- all 8 of them, unconditionally, since expander contents still
    # execute even when visually collapsed. Caching removes that dead
    # weight from every other interaction on the page.
    return _get_pattern_diagram_uncached(setup_key)
from sell_alerts import check_watchlist, load_watchlist, save_watchlist
from settings import BacktestSettings, Settings
from setups import SETUP_REGISTRY
from setups.explanations import PATTERN_EXPLANATIONS
from ui_helpers import render_stat_badges, style_results_dataframe

LIVE_MODE_MAX_SYMBOLS = 30
# Auto-fetching fundamentals costs ~3 FMP calls/symbol -- fine for a
# momentum-prefiltered shortlist (tens of symbols), but scanning the whole
# loaded universe unfiltered could mean hundreds of symbols x 3 calls each.
# Above this size, skip the automatic fetch rather than risk a multi-minute
# Run Scan (the standalone Fundamentals panel is still available for a
# smaller custom list).
MAX_FUNDAMENTALS_AUTO_FETCH = 200
LABEL_TO_SETUP_KEY = {spec["label"]: key for key, spec in SETUP_REGISTRY.items()}

st.set_page_config(page_title="Qullamaggie Scanner", layout="wide")

if "settings" not in st.session_state:
    st.session_state.settings = presets.load_default()
if "settings_version" not in st.session_state:
    # Bumped every time st.session_state.settings is replaced wholesale
    # (preset applied, reset to defaults) -- every settings-bound widget
    # below has this suffixed onto its key, so a version bump gives them
    # all fresh keys and forces them to actually re-read the new settings
    # values instead of ignoring value=/index= in favor of their own stale
    # keyed state (a Streamlit widget only honors value=/index= the very
    # first time its key appears).
    st.session_state.settings_version = 0
if "history" not in st.session_state:
    st.session_state.history = {}
if "last_scan" not in st.session_state:
    st.session_state.last_scan = pd.DataFrame()

# Apply anything queued by the Parameter Finder's "Copy to Backtest &
# Optimizer" button *before* any widget below is instantiated -- Streamlit
# forbids writing to st.session_state[key] in the same run after a widget
# with that key has already been created, so the write has to happen here,
# at the top, on the rerun that follows the button click.
_pending = st.session_state.pop("pf_pending_apply", None)
if _pending:
    st.session_state["bt_setup"] = _pending["setup"]
    st.session_state["opt_setup"] = _pending["setup"]
    for _grid_key, _grid_value in _pending["grid_values"].items():
        st.session_state[_grid_key] = _grid_value


def get_client():
    try:
        return FMPClient()
    except FMPError as exc:
        st.error(str(exc))
        st.stop()


# --------------------------------------------------------------------------
# Sidebar: data loading + preset management
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Data")
    custom_symbols = st.text_input(
        "Custom symbols (comma-separated, optional)",
        help="Leave blank to use the full screener-built universe. Fill in "
        "a few tickers (e.g. NVDA,SMCI,CVNA) for fast iteration.",
    )
    as_of = st.date_input("As-of date", value=pd.Timestamp.today())

    _auto_min_days = int(history_lookback_days(st.session_state.settings) * 1.6)
    history_years = st.slider(
        "History to load (years)", 1.0, 10.0, max(2.0, round(_auto_min_days / 365.25 * 2) / 2), 0.5,
        key="history_years",
        help=f"How far back to fetch price data -- this is what the Backtest tab can actually test over, "
        f"not just what the live scan needs. The current settings need at least ~{_auto_min_days / 365.25:.1f} "
        "years of history for indicators to compute correctly; raising this doesn't change scan results, "
        "it only gives the backtester a longer, more statistically meaningful window (and means a slower "
        "'Load / Refresh Data', since more bars per symbol get fetched).",
    )

    with st.expander("Universe filters (market cap, liquidity, momentum)"):
        u = st.session_state.settings.universe
        u.min_market_cap = st.number_input(
            "Min market cap ($)", value=float(u.min_market_cap), step=50_000_000.0, format="%.0f", key=f"u_min_mcap_{st.session_state.settings_version}"
        )
        u.max_market_cap = st.number_input(
            "Max market cap ($, 0 = no cap)", value=float(u.max_market_cap), step=500_000_000.0, format="%.0f", key=f"u_max_mcap_{st.session_state.settings_version}"
        )
        u.min_avg_dollar_volume = st.number_input(
            "Min avg $ volume (liquidity floor)", value=float(u.min_avg_dollar_volume), step=1_000_000.0, format="%.0f", key=f"u_min_dv_{st.session_state.settings_version}"
        )
        u.max_symbols = st.number_input("Max universe size", value=int(u.max_symbols), step=50, key=f"u_max_syms_{st.session_state.settings_version}")
        u.momentum_top_pct = st.slider(
            "Momentum leaders: top % of movers", 0.5, 20.0, float(u.momentum_top_pct), 0.5, key=f"u_mom_pct_{st.session_state.settings_version}"
        )
        st.caption(
            f"Currently: ${u.min_market_cap:,.0f}"
            + (f" - ${u.max_market_cap:,.0f}" if u.max_market_cap else "+")
            + f" market cap, ${u.min_avg_dollar_volume:,.0f}+ avg $ volume."
        )

    if st.button("Load / Refresh Data", type="primary"):
        symbols = [s.strip().upper() for s in custom_symbols.split(",") if s.strip()] or None
        progress = st.progress(0.0, text="Loading...")

        def on_progress(i, n, symbol):
            progress.progress(i / n, text=f"Loading {symbol} ({i}/{n})")

        try:
            client = get_client()
            st.session_state.history = load_scan_universe_history(
                st.session_state.settings,
                as_of=str(as_of),
                client=client,
                symbols=symbols,
                on_progress=on_progress,
                min_history_days=int(history_years * 365.25),
            )
            # Also load SPY and QQQ (cheap, two extra symbols) so the
            # market-direction banner is available without a separate step.
            start_for_benchmark = (pd.Timestamp(as_of) - pd.Timedelta(days=420)).strftime("%Y-%m-%d")
            ensure_benchmark_loaded(st.session_state.history, client, start_for_benchmark, str(as_of))
            ensure_benchmark_loaded(
                st.session_state.history, client, start_for_benchmark, str(as_of),
                benchmark_symbol=SECONDARY_BENCHMARK_SYMBOL,
            )
            st.session_state.market_direction = market_direction(
                st.session_state.history.get(DEFAULT_BENCHMARK_SYMBOL)
            )
            st.session_state.market_direction_secondary = market_direction(
                st.session_state.history.get(SECONDARY_BENCHMARK_SYMBOL)
            )
            progress.empty()
            st.success(f"Loaded {len(st.session_state.history)} symbols.")
        except Exception as exc:
            progress.empty()
            st.error(f"Data load failed: {exc}")
            st.code(traceback.format_exc())

    st.caption(f"{len(st.session_state.history)} symbols currently cached in memory.")

    st.divider()
    st.header("Presets")
    builtin_names = presets.list_builtin_presets()
    existing = presets.list_presets()
    preset_options = ["(current)"] + [f"⭐ {n}" for n in builtin_names] + existing
    chosen = st.selectbox("Load preset", preset_options, help="⭐ presets are built-in, sourced directly from Qullamaggie's documented small/mid-cap vs large-cap thresholds.")
    if chosen != "(current)" and st.button("Apply selected preset"):
        st.session_state.settings = presets._resolve_preset(chosen)
        st.session_state.settings_version += 1
        st.rerun()

    new_name = st.text_input("Save current settings as")
    if st.button("Save preset") and new_name:
        presets.save_preset(new_name, st.session_state.settings)
        st.success(f"Saved preset '{new_name}'.")
        st.rerun()


settings: Settings = st.session_state.settings
history = st.session_state.history

# CANSLIM's "M" -- market direction. O'Neil's premise: ~3 of every 4 stocks
# follow the broad market's trend, so this is shown up top on every tab as
# context for reading any individual match, not tucked away in a submenu.
# Shown for both SPY (broad market) and QQQ (Nasdaq-100 -- often the more
# relevant read for this scanner's growth/tech-skewed universe).
def _render_market_banner(symbol: str, md: dict) -> None:
    fast_line = ""
    if md.get("fast_direction"):
        fast_line = (
            f"""<div style="font-size:0.85em; opacity:0.85; margin-top:2px;">"""
            f"""{md['fast_emoji']} Short-term ({symbol} 10/20-day): """
            f"""<b>{md['fast_direction']}</b> -- {md['fast_detail']}</div>"""
        )
    st.markdown(
        f"""<div style="border:1px solid #444; border-radius:8px; padding:8px 14px; margin-bottom:10px;">
<b>{md['emoji']} Market ({symbol}): {md['direction']}</b> -- {md['detail']}
{fast_line}
</div>""",
        unsafe_allow_html=True,
    )


md = st.session_state.get("market_direction")
md_secondary = st.session_state.get("market_direction_secondary")
if md or md_secondary:
    col_md, col_md2 = st.columns(2)
    with col_md:
        if md:
            _render_market_banner(DEFAULT_BENCHMARK_SYMBOL, md)
    with col_md2:
        if md_secondary:
            _render_market_banner(SECONDARY_BENCHMARK_SYMBOL, md_secondary)
elif history:
    st.caption("Market direction unavailable -- reload data from the sidebar to fetch the SPY/QQQ benchmarks.")

tab_scanner, tab_designer, tab_backtest, tab_finder, tab_live, tab_sell, tab_paper, tab_settings = st.tabs(
    ["Scanner", "Designer", "Backtest & Optimizer", "Parameter Finder", "Live Mode", "Sell Alerts", "Paper Trading", "Settings"]
)


# --------------------------------------------------------------------------
# Chart helper
# --------------------------------------------------------------------------
CHART_PERIODS = {
    "5 days": 5, "1 month": 21, "3 months": 63, "6 months": 126, "1 year": 252, "2 years": 504, "All": None,
}


def slice_for_period(df: pd.DataFrame, period_label: str) -> pd.DataFrame:
    days = CHART_PERIODS.get(period_label)
    if not days or len(df) <= days:
        return df
    return df.tail(days)


AUTO_CHART_PERIOD = "Auto (fits the pattern)"


def recommended_period_days(row, setup_key: str, settings: Settings) -> int:
    """A fixed '1 year' default squeezes a 3-week flag or a 10-week Double
    Bottom into mostly-irrelevant history, while barely fitting a 9-month
    Cup with Handle. Instead, size the chart to the *actual* pattern window
    used for this specific match (from auto-detect, if that's what found
    it) plus margin for context -- a short pattern gets a tight zoom, a long
    one gets a wide one."""
    if row is None or not setup_key or setup_key not in PATTERN_WINDOW_FIELDS or settings is None:
        return CHART_PERIODS["1 year"]
    setup_settings = getattr(settings, SETUP_REGISTRY[setup_key]["settings_attr"], None)
    window_days = 0
    for field_name, _, _ in PATTERN_WINDOW_FIELDS[setup_key]:
        row_value = row.get(field_name) if hasattr(row, "get") else None
        value = row_value if row_value is not None and pd.notna(row_value) else getattr(setup_settings, field_name, None)
        if value:
            window_days = max(window_days, int(value))
    if window_days <= 0:
        return CHART_PERIODS["1 year"]
    # Show the pattern window plus ~60% margin beforehand (the quiet
    # history leading into it), with a floor so very short patterns still
    # get enough surrounding context to read.
    return max(int(window_days * 1.6), 60)


def resolve_chart_df(full_df: pd.DataFrame, period_label: str, row=None, setup_key: str = None, settings: Settings = None) -> pd.DataFrame:
    """Shared by every 'Chart period' picker: resolves the Auto option
    against a specific match's row/setup, otherwise defers to the fixed
    CHART_PERIODS buckets."""
    if period_label == AUTO_CHART_PERIOD:
        days = recommended_period_days(row, setup_key, settings)
        return full_df.tail(days) if days and len(full_df) > days else full_df
    return slice_for_period(full_df, period_label)


# For each setup: which settings fields define the lookback window(s) worth
# shading on the chart (field_name, label, fill color), so you can see
# exactly which candles the detector considered -- not just the day it fired.
PATTERN_WINDOW_FIELDS = {
    "breakout": [("max_consolidation_days", "Base", "rgba(100,149,237,0.15)")],
    "episodic_pivot": [("lookback_base_days", "Prior base", "rgba(100,149,237,0.15)")],
    "parabolic_short": [("run_up_lookback_days", "Run-up", "rgba(255,0,0,0.10)")],
    "cup_with_handle": [
        ("cup_lookback_days", "Cup", "rgba(100,149,237,0.12)"),
        ("handle_lookback_days", "Handle", "rgba(255,165,0,0.30)"),
    ],
    "double_bottom": [("pattern_lookback_days", "W pattern", "rgba(100,149,237,0.15)")],
    "flat_base": [("base_days", "Base", "rgba(100,149,237,0.15)")],
    "ascending_base": [("pattern_lookback_days", "Ascending base", "rgba(100,149,237,0.15)")],
    "high_tight_flag": [
        ("run_up_lookback_days", "Run-up", "rgba(255,0,0,0.08)"),
        ("flag_lookback_days", "Flag", "rgba(255,165,0,0.30)"),
    ],
}
# Whichever of these diagnostic columns is present holds the pattern's key
# breakout/resistance level, drawn as a horizontal reference line.
RESISTANCE_FIELDS = ["resistance", "pivot", "base_high", "middle_peak"]


def compute_adr_pct_at(df: pd.DataFrame, date, lookback: int = 20):
    """ADR% (mean of high/low - 1, over the trailing `lookback` bars ending
    at `date`) -- the same fallback formula the backtest engine itself uses
    to size stops, so the chart's stop/target lines match what a real trade
    would actually use."""
    window = df.loc[:date].tail(lookback)
    if window.empty:
        return None
    return float(((window["high"] / window["low"] - 1.0) * 100.0).mean())


def _hash_settings_for_cache(s: Settings) -> str:
    """Settings is a plain (non-frozen) dataclass, so it isn't natively
    hashable -- this gives st.cache_data a stable, content-based key so the
    plot cache correctly invalidates only when settings actually change."""
    return str(s.to_dict())


@st.cache_data(show_spinner=False, hash_funcs={Settings: _hash_settings_for_cache})
def plot_symbol(
    df: pd.DataFrame, symbol: str, marker_date=None, setup_key=None, settings=None, row=None, side="long",
    entry_date=None, entry_price=None, stop_price=None, target_price=None,
    partial_date=None, partial_price=None,
    exit_date=None, exit_price=None, exit_reason=None, r_multiple=None,
):
    # Cached: this is a pure function of its arguments (no st.* calls, no
    # hidden global state besides static module-level constants), and it's
    # rebuilt on every single page rerun otherwise -- up to 5 of these plus
    # 8 static pattern diagrams fire on ANY widget interaction anywhere in
    # the app, since Streamlit reruns the whole script every time. Caching
    # means switching a chart period/symbol elsewhere doesn't pay the cost
    # of rebuilding every *other* unrelated chart on the page.
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name=symbol
        )
    )

    # ADR% (average daily range, the same trailing-20-day formula used to
    # size stops elsewhere in this app) as of the last bar currently shown --
    # shown right in the chart title so it's visible next to every chart
    # without touching each of this function's call sites individually.
    chart_title = symbol
    if not df.empty:
        chart_adr_pct = compute_adr_pct_at(df, df.index[-1], lookback=20)
        if chart_adr_pct is not None:
            chart_title = f"{symbol} -- ADR%: {chart_adr_pct:.1f}%"
    for period, color in ((10, "orange"), (20, "blue")):
        ema = df["close"].ewm(span=period, adjust=False).mean()
        fig.add_trace(go.Scatter(x=df.index, y=ema, line=dict(width=1, color=color), name=f"EMA{period}"))

    # Simple (not exponential) moving averages -- a separate toggleable layer
    # on top of the EMA10/EMA20 pair above. Dashed + a distinct palette so
    # they read as a different family of line at a glance; each is its own
    # named trace, so clicking its legend entry shows/hides just that one --
    # no extra UI needed, this is native Plotly legend behavior.
    for period, color in ((10, "#b0b0b0"), (20, "#00bcd4"), (50, "#9c27b0"), (200, "#f44336")):
        sma = df["close"].rolling(period).mean()
        fig.add_trace(
            go.Scatter(x=df.index, y=sma, line=dict(width=1, color=color, dash="dot"), name=f"SMA{period}")
        )

    if marker_date is not None and marker_date in df.index:
        fig.add_vline(x=marker_date, line_dash="dash", line_color="green", annotation_text="signal day")

        if setup_key and settings is not None and setup_key in PATTERN_WINDOW_FIELDS:
            setup_settings = getattr(settings, SETUP_REGISTRY[setup_key]["settings_attr"])
            marker_loc = df.index.get_loc(marker_date)
            for field_name, label, color in PATTERN_WINDOW_FIELDS[setup_key]:
                # Prefer the actual window size that produced this match (set
                # by auto-detect, which may differ from the Designer tab's
                # current default) over the settings object's fixed value.
                row_value = row.get(field_name) if row is not None and hasattr(row, "get") else None
                window_days = row_value if row_value is not None and pd.notna(row_value) else getattr(
                    setup_settings, field_name, None
                )
                if not window_days:
                    continue
                start_loc = max(marker_loc - int(window_days), 0)
                fig.add_vrect(
                    x0=df.index[start_loc], x1=marker_date, fillcolor=color, line_width=0,
                    annotation_text=label, annotation_position="top left",
                )

        if row is not None:
            for field in RESISTANCE_FIELDS:
                value = row.get(field) if hasattr(row, "get") else None
                if value is not None and pd.notna(value):
                    fig.add_hline(
                        y=value, line_dash="dot", line_color="yellow",
                        annotation_text=f"{field} = {value:.2f}", annotation_position="right",
                    )
                    break

        # An explicit marker right on the breakout candle -- dashed lines
        # spanning the whole chart can be easy to lose among the candles,
        # so this pins down exactly which bar triggered the signal.
        breakout_high = df.loc[marker_date, "high"]
        fig.add_trace(
            go.Scatter(
                x=[marker_date],
                y=[breakout_high * 1.03],
                mode="markers+text",
                marker=dict(symbol="star", size=16, color="lime", line=dict(width=1, color="black")),
                text=["Breakout"],
                textposition="top center",
                textfont=dict(color="lime"),
                name="Breakout day",
                showlegend=False,
            )
        )

        # Entry/stop/target zone -- the same ADR-based stop and R-multiple
        # target the backtest engine actually uses, so "where to sell" on
        # the chart matches what a real trade would do, not a guess.
        if settings is not None:
            marker_loc = df.index.get_loc(marker_date)
            entry_loc = marker_loc + settings.backtest.entry_delay_days
            if entry_loc < len(df.index):
                entry_date = df.index[entry_loc]
                entry_price = df.loc[entry_date, "open"]
                is_projected = False
            else:
                entry_date = marker_date
                entry_price = df.loc[marker_date, "close"]
                is_projected = True  # no next bar yet -- this signal hasn't been entered

            adr_pct = compute_adr_pct_at(df, marker_date)
            if adr_pct and entry_price:
                stop_distance = entry_price * (adr_pct / 100.0) * settings.backtest.stop_adr_multiple
                sign = 1.0 if side == "long" else -1.0
                stop_price = entry_price - sign * stop_distance
                target_price = entry_price + sign * settings.backtest.partial_profit_r_multiple * stop_distance

                fig.add_trace(
                    go.Scatter(
                        x=[entry_date], y=[entry_price], mode="markers",
                        marker=dict(symbol="diamond", size=12, color="cyan", line=dict(width=1, color="black")),
                        name="Entry" + (" (projected)" if is_projected else ""),
                        showlegend=False,
                    )
                )
                fig.add_hline(
                    y=stop_price, line_dash="dash", line_color="red",
                    annotation_text=f"Stop {stop_price:.2f}", annotation_position="bottom right",
                )
                fig.add_hline(
                    y=target_price, line_dash="dash", line_color="mediumseagreen",
                    annotation_text=f"Target {target_price:.2f} ({settings.backtest.partial_profit_r_multiple:.1f}R)",
                    annotation_position="top right",
                )
                lo, hi = sorted([stop_price, entry_price])
                fig.add_hrect(y0=lo, y1=hi, fillcolor="rgba(255,0,0,0.06)", line_width=0)
                lo, hi = sorted([entry_price, target_price])
                fig.add_hrect(y0=lo, y1=hi, fillcolor="rgba(60,179,113,0.08)", line_width=0)

        # Realized paper-trade outcome -- draws the ACTUAL entry/stop/target/
        # partial/exit from a real simulated Trade result, passed directly by
        # the caller (Paper Trading tab) rather than recomputed from
        # `settings` like the static projection above. Independent of the
        # `settings is not None` block so it works even when settings=None.
        # Uses pd.notna() throughout (not `is not None`) since these values
        # arrive via a DataFrame row -- missing entries show up as NaT/NaN,
        # not Python None, once mixed into a column with real values.
        if pd.notna(entry_date) and pd.notna(entry_price):
            fig.add_trace(
                go.Scatter(
                    x=[entry_date], y=[entry_price], mode="markers",
                    marker=dict(symbol="diamond", size=12, color="cyan", line=dict(width=1, color="black")),
                    name="Entry", showlegend=False,
                )
            )
            if pd.notna(stop_price):
                fig.add_hline(
                    y=stop_price, line_dash="dash", line_color="red",
                    annotation_text=f"Stop {stop_price:.2f}", annotation_position="bottom right",
                )
            if pd.notna(target_price):
                fig.add_hline(
                    y=target_price, line_dash="dash", line_color="mediumseagreen",
                    annotation_text=f"Target {target_price:.2f}", annotation_position="top right",
                )
            if pd.notna(partial_date) and pd.notna(partial_price):
                fig.add_trace(
                    go.Scatter(
                        x=[partial_date], y=[partial_price], mode="markers+text",
                        marker=dict(symbol="diamond", size=13, color="orange", line=dict(width=1, color="black")),
                        text=["Sold half"], textposition="bottom center", textfont=dict(color="orange"),
                        name="Partial exit", showlegend=False,
                    )
                )
            if pd.notna(exit_date) and pd.notna(exit_price):
                r_val = r_multiple if pd.notna(r_multiple) else 0
                exit_color = "lime" if r_val > 0 else ("red" if r_val < 0 else "gray")
                fig.add_trace(
                    go.Scatter(
                        x=[exit_date], y=[exit_price], mode="markers+text",
                        marker=dict(symbol="x", size=14, color=exit_color, line=dict(width=1, color="black")),
                        text=[exit_reason or "exit"], textposition="top center", textfont=dict(color=exit_color),
                        name="Exit", showlegend=False,
                    )
                )

    fig.update_layout(
        title=dict(text=chart_title, font=dict(size=13)),
        height=450, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=35, b=10),
    )
    return fig


# --------------------------------------------------------------------------
# Tab: Designer (fast iterate settings -> live scan + backtest feedback)
#
# NOTE: this block is written *before* the Scanner block below even though
# Scanner is the first visual tab (tab order is set by the st.tabs(...) call
# above, not by source order). Streamlit re-executes the whole script on
# every widget interaction and all tab bodies run every time regardless of
# which is visually active, so mutating `settings` here first means the
# Scanner/Backtest tabs read the up-to-date values in the *same* rerun
# instead of lagging by one interaction.
# --------------------------------------------------------------------------
with tab_designer:
    st.subheader("Interactive system designer")
    st.caption("Tweak any parameter below -- the scan matches and backtest stats to the right update immediately, using the already-loaded data (no re-fetch).")

    if not history:
        st.info("Load data from the sidebar first.")
    else:
        col_params, col_results = st.columns([1, 1])

        with col_params:
            setup_choice = st.selectbox(
                "Setup to design", list(SETUP_REGISTRY.keys()), format_func=lambda k: SETUP_REGISTRY[k]["label"], key="designer_setup"
            )
            spec = SETUP_REGISTRY[setup_choice]
            s = getattr(settings, spec["settings_attr"])

            s.enabled = st.checkbox("Enabled", value=s.enabled, key=f"{setup_choice}_enabled_{st.session_state.settings_version}")

            if setup_choice == "breakout":
                s.min_adr_pct = st.slider("Min ADR %", 0.5, 15.0, float(s.min_adr_pct), 0.5, key=f"d_bo_adr_{st.session_state.settings_version}")
                s.prior_move_min_pct = st.slider("Prior move min %", 0.0, 200.0, float(s.prior_move_min_pct), 5.0, key=f"d_bo_pmmin_{st.session_state.settings_version}")
                s.prior_move_max_pct = st.slider("Prior move max %", 10.0, 500.0, float(s.prior_move_max_pct), 5.0, key=f"d_bo_pmmax_{st.session_state.settings_version}")
                s.min_consolidation_days = st.slider("Min consolidation (days)", 3, 60, int(s.min_consolidation_days), key=f"d_bo_mincon_{st.session_state.settings_version}")
                s.max_consolidation_days = st.slider("Max consolidation (days)", 5, 90, int(s.max_consolidation_days), key=f"d_bo_maxcon_{st.session_state.settings_version}")
                s.max_consolidation_range_pct = st.slider("Max consolidation range %", 5.0, 80.0, float(s.max_consolidation_range_pct), 1.0, key=f"d_bo_range_{st.session_state.settings_version}")
                s.breakout_volume_ratio_min = st.slider("Min breakout volume ratio", 0.5, 6.0, float(s.breakout_volume_ratio_min), 0.1, key=f"d_bo_vol_{st.session_state.settings_version}")
                s.min_rs_rating = st.slider("Min RS rating", 1, 99, int(s.min_rs_rating), key=f"d_bo_rs_{st.session_state.settings_version}")
                s.require_above_ema10 = st.checkbox("Require close > EMA10", value=s.require_above_ema10, key=f"d_bo_ema10_{st.session_state.settings_version}")
                s.require_above_ema20 = st.checkbox("Require close > EMA20", value=s.require_above_ema20, key=f"d_bo_ema20_{st.session_state.settings_version}")
            elif setup_choice == "episodic_pivot":
                s.min_gap_pct = st.slider("Min gap %", 1.0, 50.0, float(s.min_gap_pct), 1.0, key=f"d_ep_gap_{st.session_state.settings_version}")
                s.min_volume_ratio = st.slider("Min volume ratio", 0.5, 10.0, float(s.min_volume_ratio), 0.1, key=f"d_ep_vol_{st.session_state.settings_version}")
                s.max_prior_range_pct = st.slider("Max prior base range %", 5.0, 60.0, float(s.max_prior_range_pct), 1.0, key=f"d_ep_range_{st.session_state.settings_version}")
                s.lookback_base_days = st.slider("Prior base lookback (days)", 5, 60, int(s.lookback_base_days), key=f"d_ep_lookback_{st.session_state.settings_version}")
                st.markdown("**Quiet-base check** (\"gone sideways for 3-6 months or more\")")
                s.quiet_base_lookback_days = st.slider(
                    "Quiet-base lookback (days)", 40, 200, int(s.quiet_base_lookback_days), 10, key=f"d_ep_quiet_lb_{st.session_state.settings_version}"
                )
                s.max_quiet_base_run_pct = st.slider(
                    "Max run in quiet-base window %", 10.0, 100.0, float(s.max_quiet_base_run_pct), 5.0, key=f"d_ep_quiet_run_{st.session_state.settings_version}"
                )
                st.markdown("**Prior-EP avoidance** (soft penalty, not a hard exclude)")
                s.prior_ep_lookback_days = st.slider(
                    "Prior-EP lookback (days)", 60, 378, int(s.prior_ep_lookback_days), 10, key=f"d_ep_prior_lb_{st.session_state.settings_version}"
                )
                s.prior_ep_min_gap_pct = st.slider(
                    "Prior-EP min gap % (to count as \"already had one\")", 3.0, 30.0, float(s.prior_ep_min_gap_pct), 1.0,
                    key=f"d_ep_prior_gap_{st.session_state.settings_version}",
                )
                st.markdown("**Growth quality** (needs fundamentals fetched -- see Scanner tab)")
                s.min_growth_pct_floor = st.slider(
                    "Growth floor %", 0.0, 100.0, float(s.min_growth_pct_floor), 5.0, key=f"d_ep_growth_floor_{st.session_state.settings_version}"
                )
                s.ideal_growth_pct = st.slider(
                    "Ideal (\"5-star\") growth %", 25.0, 300.0, float(s.ideal_growth_pct), 25.0, key=f"d_ep_growth_ideal_{st.session_state.settings_version}"
                )
                s.require_growth_floor = st.checkbox(
                    "Require growth floor as a hard filter", value=s.require_growth_floor, key=f"d_ep_require_growth_{st.session_state.settings_version}"
                )
            elif setup_choice == "parabolic_short":
                s.min_extension_adr_multiple = st.slider("Min extension (x ADR)", 0.5, 10.0, float(s.min_extension_adr_multiple), 0.1, key=f"d_ps_ext_{st.session_state.settings_version}")
                s.min_run_up_pct = st.slider("Min run-up %", 10.0, 300.0, float(s.min_run_up_pct), 5.0, key=f"d_ps_runup_{st.session_state.settings_version}")
                s.consecutive_up_days_min = st.slider("Min consecutive up days", 1, 10, int(s.consecutive_up_days_min), key=f"d_ps_updays_{st.session_state.settings_version}")
            elif setup_choice == "cup_with_handle":
                s.min_cup_depth_pct = st.slider("Min cup depth %", 5.0, 50.0, float(s.min_cup_depth_pct), 1.0, key=f"d_cwh_mind_{st.session_state.settings_version}")
                s.max_cup_depth_pct = st.slider("Max cup depth %", 10.0, 60.0, float(s.max_cup_depth_pct), 1.0, key=f"d_cwh_maxd_{st.session_state.settings_version}")
                s.max_handle_depth_pct = st.slider("Max handle depth %", 3.0, 30.0, float(s.max_handle_depth_pct), 1.0, key=f"d_cwh_hd_{st.session_state.settings_version}")
                s.min_recovery_pct = st.slider("Min recovery to old high %", 50.0, 100.0, float(s.min_recovery_pct), 1.0, key=f"d_cwh_rec_{st.session_state.settings_version}")
                s.min_prior_uptrend_pct = st.slider("Min prior uptrend %", 0.0, 100.0, float(s.min_prior_uptrend_pct), 5.0, key=f"d_cwh_prior_{st.session_state.settings_version}")
                s.handle_upper_half_only = st.checkbox("Handle must stay in upper half of cup", value=s.handle_upper_half_only, key=f"d_cwh_upperhalf_{st.session_state.settings_version}")
                s.breakout_volume_ratio_min = st.slider("Min breakout volume ratio", 0.5, 6.0, float(s.breakout_volume_ratio_min), 0.1, key=f"d_cwh_vol_{st.session_state.settings_version}")
            elif setup_choice == "double_bottom":
                s.min_depth_pct = st.slider("Min depth %", 3.0, 40.0, float(s.min_depth_pct), 1.0, key=f"d_db_mind_{st.session_state.settings_version}")
                s.max_depth_pct = st.slider("Max depth %", 10.0, 60.0, float(s.max_depth_pct), 1.0, key=f"d_db_maxd_{st.session_state.settings_version}")
                s.max_low_difference_pct = st.slider("Max difference between the two lows %", 1.0, 25.0, float(s.max_low_difference_pct), 1.0, key=f"d_db_lowdiff_{st.session_state.settings_version}")
                s.breakout_volume_ratio_min = st.slider("Min breakout volume ratio", 0.5, 6.0, float(s.breakout_volume_ratio_min), 0.1, key=f"d_db_vol_{st.session_state.settings_version}")
            elif setup_choice == "flat_base":
                s.max_range_pct = st.slider("Max base range %", 5.0, 30.0, float(s.max_range_pct), 1.0, key=f"d_fb_range_{st.session_state.settings_version}")
                s.min_prior_move_pct = st.slider("Min prior move %", 0.0, 100.0, float(s.min_prior_move_pct), 5.0, key=f"d_fb_prior_{st.session_state.settings_version}")
                s.breakout_volume_ratio_min = st.slider("Min breakout volume ratio", 0.5, 6.0, float(s.breakout_volume_ratio_min), 0.1, key=f"d_fb_vol_{st.session_state.settings_version}")
            elif setup_choice == "ascending_base":
                s.min_segment_depth_pct = st.slider("Min per-pullback depth %", 1.0, 20.0, float(s.min_segment_depth_pct), 1.0, key=f"d_ab_mind_{st.session_state.settings_version}")
                s.max_segment_depth_pct = st.slider("Max per-pullback depth %", 10.0, 40.0, float(s.max_segment_depth_pct), 1.0, key=f"d_ab_maxd_{st.session_state.settings_version}")
                s.breakout_volume_ratio_min = st.slider("Min breakout volume ratio", 0.5, 6.0, float(s.breakout_volume_ratio_min), 0.1, key=f"d_ab_vol_{st.session_state.settings_version}")
            else:  # high_tight_flag
                s.min_run_up_pct = st.slider("Min run-up %", 50.0, 300.0, float(s.min_run_up_pct), 5.0, key=f"d_htf_runup_{st.session_state.settings_version}")
                s.max_flag_depth_pct = st.slider("Max flag depth %", 5.0, 40.0, float(s.max_flag_depth_pct), 1.0, key=f"d_htf_flag_{st.session_state.settings_version}")
                s.breakout_volume_ratio_min = st.slider("Min breakout volume ratio", 0.5, 6.0, float(s.breakout_volume_ratio_min), 0.1, key=f"d_htf_vol_{st.session_state.settings_version}")

            with st.expander(f"What does {spec['label']} mean?"):
                st.write(PATTERN_EXPLANATIONS.get(setup_choice, "No explanation available."))

            st.markdown("**Backtest mechanics**")
            bt = settings.backtest
            bt.stop_mode = st.selectbox(
                "Stop placement", ["low_of_signal_day", "adr_multiple"], index=0 if bt.stop_mode == "low_of_signal_day" else 1,
                format_func=lambda v: "Low of signal day (Qullamaggie's rule, capped at the ADR)" if v == "low_of_signal_day" else "Pure ADR multiple from entry",
                key=f"d_bt_stopmode_{st.session_state.settings_version}",
            )
            bt.risk_pct_per_trade = st.slider("Risk % per trade", 0.1, 5.0, float(bt.risk_pct_per_trade), 0.1, key=f"d_bt_risk_{st.session_state.settings_version}")
            bt.stop_adr_multiple = st.slider(
                "Stop distance cap (x ADR)", 0.25, 3.0, float(bt.stop_adr_multiple), 0.25, key=f"d_bt_stop_{st.session_state.settings_version}",
                help="With 'Low of signal day', this is a cap -- the stop never ends up farther than this many ADRs away.",
            )
            bt.max_position_pct_of_equity = st.slider(
                "Max position size (% of equity)", 5.0, 100.0, float(bt.max_position_pct_of_equity), 5.0, key=f"d_bt_maxpos_pct_{st.session_state.settings_version}",
                help="\"Don't put more than 20% of your account into any one share.\"",
            )
            bt.avoid_chase_adr_multiple = st.slider(
                "Anti-chase: skip if signal day's range > this many ADRs (0 = off)", 0.0, 4.0,
                float(bt.avoid_chase_adr_multiple), 0.25, key=f"d_bt_chase_{st.session_state.settings_version}",
            )
            bt.partial_profit_r_multiple = st.slider("Partial profit target (R)", 0.5, 5.0, float(bt.partial_profit_r_multiple), 0.5, key=f"d_bt_ptarget_{st.session_state.settings_version}")
            bt.partial_profit_fraction = st.slider("Partial profit fraction", 0.0, 1.0, float(bt.partial_profit_fraction), 0.05, key=f"d_bt_pfrac_{st.session_state.settings_version}")
            bt.partial_profit_max_days = st.slider(
                "OR take partial after this many days held, if sooner (0 = off)", 0, 20, int(bt.partial_profit_max_days),
                key=f"d_bt_pdays_{st.session_state.settings_version}",
                help="\"Sell 1/3 to 1/2 of the position after 3-5 days, then move the stop to break even\" -- "
                "fires the partial even if the R-target above hasn't been hit yet.",
            )
            bt.move_stop_to_breakeven_after_partial = st.checkbox(
                "Move stop to breakeven after partial", value=bt.move_stop_to_breakeven_after_partial, key=f"d_bt_breakeven_{st.session_state.settings_version}",
            )
            bt.trail_ma_type = st.selectbox(
                "Trailing MA type", ["sma", "ema"], index=0 if bt.trail_ma_type == "sma" else 1, key=f"d_bt_matype_{st.session_state.settings_version}",
            )
            bt.trail_ema_period = st.slider("Trailing MA period", 5, 50, int(bt.trail_ema_period), key=f"d_bt_trail_{st.session_state.settings_version}")
            bt.max_positions = st.slider("Max concurrent positions", 1, 30, int(bt.max_positions), key=f"d_bt_maxpos_{st.session_state.settings_version}")
            bt.max_position_pct_of_avg_volume = st.slider(
                "Max position size (% of avg daily volume, 0 = off)", 0.0, 10.0,
                float(bt.max_position_pct_of_avg_volume), 0.5, key=f"d_bt_liq_cap_{st.session_state.settings_version}",
                help="\"Buy no more than 1% of the average volume\" (Qullamaggie's Laws of Swing).",
            )
            if setup_choice == "episodic_pivot":
                ep_override_options = ["(use Stop placement above)", "low_of_signal_day", "adr_multiple"]
                ep_override_index = (
                    ep_override_options.index(bt.ep_stop_mode_override)
                    if bt.ep_stop_mode_override in ep_override_options
                    else 0
                )
                ep_override_choice = st.selectbox(
                    "EP stop override", ep_override_options, index=ep_override_index, key=f"d_bt_ep_override_{st.session_state.settings_version}",
                    help="EP's stop should exclude the gap (\"calculate stop from the low of the opening candle, "
                    "don't include the gap\") -- the whole-day low used elsewhere is gap-contaminated for EP "
                    "specifically. Only affects Episodic Pivot backtests.",
                )
                bt.ep_stop_mode_override = None if ep_override_choice == "(use Stop placement above)" else ep_override_choice

        with col_results:
            live_results = run_scan(settings, history, as_of=str(as_of), setup_names=[setup_choice])
            st.metric("Current matches", len(live_results))
            if not live_results.empty:
                st.dataframe(live_results, use_container_width=True, height=200)

            st.markdown("**Live backtest (full loaded history)**")
            signals = scan_signals_over_history(settings, history, setup_choice)
            result = run_backtest(settings.backtest, signals, side=spec["side"], setup_name=setup_choice)
            stats = compute_stats(result)
            if stats["trade_count"] == 0:
                st.warning(stats_summary_text(stats))
            else:
                render_stat_badges(stats)
            if not result.equity_curve.empty:
                st.line_chart(result.equity_curve, height=250)


# --------------------------------------------------------------------------
# Tab: Scanner
# --------------------------------------------------------------------------
with tab_scanner:
    st.subheader("Today's matches")
    if not history:
        st.info("Load data from the sidebar first.")
    else:
        with st.expander("💪 Momentum leaders (biggest movers by timeframe)", expanded=False):
            st.caption(
                "The universe filters (sidebar/Settings) already scope to small/mid-caps with a "
                "$20M+ liquidity floor by default. This finds the strongest "
                f"{settings.universe.momentum_top_pct:.0f}% of movers within that universe over "
                "1/3/6/12/18 months -- the biggest winners, independent of whether they're forming a pattern yet."
            )
            if st.button("Find biggest movers"):
                st.session_state.movers_by_tf = top_movers_by_timeframe(
                    history, settings.universe.momentum_timeframes_days, settings.universe.momentum_top_pct
                )
            movers_by_tf = st.session_state.get("movers_by_tf")
            if movers_by_tf:
                mover_cols = st.columns(len(movers_by_tf))
                for col, (days, df) in zip(mover_cols, movers_by_tf.items()):
                    with col:
                        st.markdown(f"**{TIMEFRAME_LABELS.get(days, f'{days}d')}**")
                        if df.empty:
                            st.caption("no data")
                        else:
                            st.dataframe(df.round(1), use_container_width=True, height=200, hide_index=True)

        with st.expander("🛡️ Relative strength: stocks that held up the most", expanded=False):
            st.caption(
                "Classic IBD/O'Neil relative strength: not just 'went up a lot,' but 'isn't giving it "
                "back' -- especially versus the market itself (SPY). Finds stocks with a big prior move "
                "(default: 12 months) whose recent pullback (default: 1 month) is smaller than SPY's own "
                "pullback over the same window. Auto-loads SPY as the benchmark if it isn't already cached."
            )
            rs_col1, rs_col2 = st.columns(2)
            with rs_col1:
                rs_prior_days = st.selectbox(
                    "Prior move window", [63, 126, 252, 378], index=2,
                    format_func=lambda d: TIMEFRAME_LABELS.get(d, f"{d}d"), key="rs_prior_days",
                )
            with rs_col2:
                rs_recent_days = st.selectbox(
                    "Recent pullback window", [10, 21, 42, 63], index=1,
                    format_func=lambda d: {10: "2 weeks", 21: "1 month", 42: "2 months", 63: "3 months"}[d],
                    key="rs_recent_days",
                )
            if st.button("Find stocks holding up the most"):
                try:
                    client = get_client()
                    start = (pd.Timestamp(as_of) - pd.Timedelta(days=int(rs_prior_days * 1.6) + 30)).strftime("%Y-%m-%d")
                    ensure_benchmark_loaded(history, client, start, str(as_of))
                    st.session_state.history = history
                    st.session_state.resilience_df = relative_strength_resilience(
                        history, prior_move_days=rs_prior_days, recent_window_days=rs_recent_days
                    )
                except Exception as exc:
                    st.error(f"Couldn't load benchmark/compute resilience: {exc}")

            resilience_df = st.session_state.get("resilience_df")
            if resilience_df is not None and not resilience_df.empty:
                st.caption("Sorted by resilience vs. SPY first (if SPY loaded), then by prior move size.")
                st.dataframe(resilience_df.round(2), use_container_width=True, height=300)

        with st.expander("🎯 Approaching pivot (sitting in a base, hasn't broken out yet)", expanded=False):
            st.caption(
                "The main scan only flags a stock on the exact day it breaks out -- a rare, discrete event "
                "per stock. This instead finds stocks currently sitting inside a *valid* pattern shape that "
                "simply haven't triggered yet, within a chosen % of the resistance/pivot level -- a watchlist "
                "for what might move next, not just what already did."
            )
            pivot_distance = st.slider(
                "Max distance from pivot (%)", 1.0, 20.0, 8.0, 1.0, key="pivot_distance_pct",
                help="How close (in either direction) to the breakout level counts as 'approaching'.",
            )
            if st.button("Find stocks approaching a pivot"):
                st.session_state.approaching_df = find_approaching_pivot(
                    settings, history, as_of=str(as_of), max_distance_pct=pivot_distance
                )
            approaching_df = st.session_state.get("approaching_df")
            if approaching_df is not None and not approaching_df.empty:
                st.caption("Sorted closest-to-triggering first. Negative pct_to_pivot means it's already just above the level, waiting on volume to confirm.")
                st.dataframe(approaching_df.round(2), use_container_width=True, height=300)
                pick_approach = st.selectbox(
                    "Chart one", approaching_df["symbol"].tolist(), key="approach_chart_pick"
                )
                if pick_approach and pick_approach in history:
                    arow = approaching_df[approaching_df["symbol"] == pick_approach].iloc[0]
                    a_setup_key = LABEL_TO_SETUP_KEY.get(arow["setup"])
                    st.plotly_chart(
                        plot_symbol(
                            history[pick_approach], pick_approach, marker_date=arow["date"],
                            setup_key=a_setup_key, settings=settings, row=arow, side=arow.get("side", "long"),
                        ),
                        use_container_width=True,
                        key="approach_chart",
                    )
            elif approaching_df is not None:
                st.caption("Nothing within that distance right now -- try widening the % above.")

        col_pf, col_ad = st.columns(2)
        with col_pf:
            prefilter_momentum = st.checkbox(
                "Pre-filter to momentum leaders before scanning for patterns",
                value=True,
                help="Only run pattern detection on the union of the top movers above, instead of the whole "
                "loaded universe -- 'find the biggest movers, then see which ones are forming a pattern'.",
                key="scanner_prefilter_momentum",
            )
        with col_ad:
            auto_detect = st.checkbox(
                "🔍 Auto-detect pattern size (recommended)",
                value=True,
                help="Instead of requiring an exact cup depth/width, handle depth, base length, etc. to match "
                "today's settings, tries several realistic sizes for each pattern (spanning what O'Neil's book "
                "documents) and counts it a match if any of them fit. Turn this off to use the exact numbers "
                "from the Designer tab instead.",
                key="scanner_auto_detect",
            )

        col_earn, col_earn_days = st.columns([2, 1])
        with col_earn:
            avoid_earnings = st.checkbox(
                "📅 Avoid signals within N days of earnings",
                value=True,
                help="\"Avoid buying 3-days before earnings\" (Qullamaggie's Laws of Swing) -- fetches next "
                "earnings dates in one bulk call (cheap regardless of universe size) and drops any match "
                "whose symbol reports within the window below.",
                key="scanner_avoid_earnings",
            )
        with col_earn_days:
            avoid_earnings_days = st.number_input(
                "Days", min_value=0, max_value=14, value=3, key="scanner_avoid_earnings_days", disabled=not avoid_earnings
            )

        if st.button("Run Scan"):
            scan_history = history
            if prefilter_momentum:
                shortlist = momentum_shortlist(
                    history, settings.universe.momentum_timeframes_days, settings.universe.momentum_top_pct
                )
                scan_history = {s: history[s] for s in shortlist if s in history}
                st.session_state.scan_shortlist_size = len(scan_history)

            scan_earnings_dates = None
            if avoid_earnings:
                try:
                    scan_earnings_dates = get_earnings_dates(get_client(), list(scan_history.keys()), as_of=str(as_of))
                except Exception as exc:
                    st.warning(f"Earnings-date lookup failed ({exc}) -- running without the earnings-avoidance filter.")

            scan_fundamentals = None
            if len(scan_history) <= MAX_FUNDAMENTALS_AUTO_FETCH:
                try:
                    fund_client = get_client()
                    fund_progress = st.progress(0.0, text="Fetching fundamentals for scan candidates...")
                    scan_fundamentals = get_fundamentals_bulk(
                        fund_client, list(scan_history.keys()),
                        on_progress=lambda i, n, s: fund_progress.progress(i / n, text=f"Fundamentals: {s} ({i}/{n})"),
                    )
                    fund_progress.empty()
                except Exception as exc:
                    st.warning(f"Fundamentals fetch failed ({exc}) -- scanning without growth-based scoring/columns.")
            else:
                st.caption(
                    f"Skipping automatic fundamentals fetch for {len(scan_history)} symbols "
                    f"(cap: {MAX_FUNDAMENTALS_AUTO_FETCH}) -- pre-filter to momentum leaders first for growth-based "
                    "scoring, or use the Fundamentals panel below for a smaller custom list."
                )

            st.session_state.last_scan = run_scan(
                settings, scan_history, as_of=str(as_of), auto_detect=auto_detect,
                fundamentals=scan_fundamentals,
                earnings_dates=scan_earnings_dates, avoid_earnings_window=avoid_earnings,
                avoid_earnings_days=int(avoid_earnings_days),
            )
        if prefilter_momentum and st.session_state.get("scan_shortlist_size") is not None:
            st.caption(f"Scanned {st.session_state.scan_shortlist_size} momentum-leading symbol(s), not the full universe.")
        results = st.session_state.last_scan
        if results.empty:
            st.warning("No matches yet -- click 'Run Scan', or loosen thresholds in the Designer tab.")
        else:
            st.caption(
                "Sorted by RS rating (momentum/leadership) first, then setup score -- grouped below by setup "
                "so each table only shows columns that actually apply to that pattern (a Cup with Handle match "
                "was never going to have a Double Bottom's columns, and vice versa). The rank column reflects "
                "the overall cross-setup priority, so you can still tell what matters most at a glance."
            )
            results_ranked = results.copy()
            results_ranked.insert(0, "rank", results_ranked.index + 1)
            # Show setups in the order their best (highest-priority) match
            # appears in the overall ranking, not alphabetically -- the
            # group containing the #1 overall match comes first.
            setup_order = results_ranked.groupby("setup")["rank"].min().sort_values().index.tolist()
            for setup_label in setup_order:
                group_df = results_ranked[results_ranked["setup"] == setup_label].dropna(axis=1, how="all")
                st.markdown(f"**{setup_label}** ({len(group_df)} match{'es' if len(group_df) != 1 else ''})")
                st.dataframe(group_df, use_container_width=True, height=min(80 + 35 * len(group_df), 350))

            col_pick, col_period = st.columns([3, 1])
            with col_pick:
                pick = st.selectbox("Chart a match", results["symbol"].tolist())
            row = results[results["symbol"] == pick].iloc[0] if pick else None
            picked_setup_key = LABEL_TO_SETUP_KEY.get(row["setup"]) if row is not None else None
            with col_period:
                pick_period = st.selectbox(
                    "Chart period", [AUTO_CHART_PERIOD] + list(CHART_PERIODS.keys()), index=0, key="match_chart_period",
                    help="Auto sizes the chart to this specific match's actual pattern window (e.g. a 3-week flag "
                    "gets a tight zoom, a 9-month cup gets a wide one) instead of a fixed calendar period.",
                )
            if pick and pick in history:
                picked_side = row.get("side", "long")
                st.plotly_chart(
                    plot_symbol(
                        resolve_chart_df(history[pick], pick_period, row, picked_setup_key, settings), pick,
                        marker_date=row["date"], setup_key=picked_setup_key, settings=settings, row=row, side=picked_side,
                    ),
                    use_container_width=True,
                    key="scanner_match_chart",
                )
                st.caption(
                    "Shaded blue/orange region = the window the detector looked at; yellow dotted line = "
                    "breakout/resistance level; green star = signal day; cyan diamond = entry; red dashed = "
                    "stop-loss; green dashed = profit target (same ADR-based stop/R-multiple target the "
                    "backtester uses). \"Entry (projected)\" means there's no next bar yet -- the trade "
                    "hasn't actually been entered."
                )

            matched_setups = [k for k, v in SETUP_REGISTRY.items() if v["label"] in results["setup"].unique()]
            if matched_setups:
                st.markdown("**What these matches mean:**")
                for setup_key in matched_setups:
                    with st.expander(SETUP_REGISTRY[setup_key]["label"]):
                        st.write(PATTERN_EXPLANATIONS.get(setup_key, "No explanation available."))

        with st.expander("📖 Pattern library (all setups, whether matched or not)"):
            st.caption("Diagrams are idealized schematics illustrating the shape -- not real ticker data.")
            for setup_key, spec in SETUP_REGISTRY.items():
                st.markdown(f"**{spec['label']}**")
                diagram_col, text_col = st.columns([1, 1])
                with diagram_col:
                    diagram = get_pattern_diagram(setup_key)
                    if diagram is not None:
                        st.plotly_chart(diagram, use_container_width=True, key=f"pattern_diagram_{setup_key}")
                with text_col:
                    st.write(PATTERN_EXPLANATIONS.get(setup_key, "No explanation available."))
                st.divider()

        with st.expander("🖼️ Browse all charts (eyeball any loaded stock yourself)", expanded=False):
            st.caption(
                "Not just matches -- flip through every symbol currently loaded, at whatever timeframe you "
                "want, to check for patterns yourself. If the symbol also matched in the last scan, its "
                "pattern gets shaded/annotated same as above."
            )
            all_symbols = sorted(history.keys())
            # The Symbol selectbox owns its displayed value via its `key`
            # once set -- Streamlit ignores a later `index=` argument on a
            # keyed widget, so Prev/Next must update that SAME key directly
            # (before the selectbox is instantiated this run), not a
            # separate index variable, or the click gets silently
            # overwritten by the selectbox reading back its own stale
            # keyed state a few lines later.
            if (
                "browse_symbol_select" not in st.session_state
                or st.session_state.browse_symbol_select not in all_symbols
            ):
                st.session_state.browse_symbol_select = all_symbols[0]
            current_bidx = all_symbols.index(st.session_state.browse_symbol_select)

            col_prev, col_pick2, col_next, col_period2 = st.columns([1, 3, 1, 2])
            with col_prev:
                if st.button("⬅ Prev", key="browse_prev"):
                    st.session_state.browse_symbol_select = all_symbols[(current_bidx - 1) % len(all_symbols)]
            with col_next:
                if st.button("Next ➡", key="browse_next"):
                    st.session_state.browse_symbol_select = all_symbols[(current_bidx + 1) % len(all_symbols)]
            with col_pick2:
                browse_symbol = st.selectbox("Symbol", all_symbols, key="browse_symbol_select")
            with col_period2:
                browse_period = st.selectbox(
                    "Chart period", [AUTO_CHART_PERIOD] + list(CHART_PERIODS.keys()), index=0, key="browse_period",
                    help="Auto sizes to this symbol's actual matched pattern window, if it matched; otherwise falls "
                    "back to 1 year.",
                )

            browse_match = None
            if not results.empty:
                sym_matches = results[results["symbol"] == browse_symbol]
                if not sym_matches.empty:
                    browse_match = sym_matches.iloc[0]

            browse_setup_key = LABEL_TO_SETUP_KEY.get(browse_match["setup"]) if browse_match is not None else None
            browse_df = resolve_chart_df(history[browse_symbol], browse_period, browse_match, browse_setup_key, settings)

            if browse_match is not None:
                st.plotly_chart(
                    plot_symbol(
                        browse_df, browse_symbol, marker_date=browse_match["date"],
                        setup_key=browse_setup_key, settings=settings, row=browse_match,
                        side=browse_match.get("side", "long"),
                    ),
                    use_container_width=True,
                    key="browse_chart",
                )
                st.success(f"This matched **{browse_match['setup']}** in the last scan.")
            else:
                st.plotly_chart(plot_symbol(browse_df, browse_symbol), use_container_width=True, key="browse_chart")
                st.caption("No pattern match for this symbol in the last scan (or you haven't run one yet).")

        with st.expander("🧮 Legout scans (TC2000-style, from legout.github.io)", expanded=False):
            st.caption(
                "24 of legout.github.io's ~32 published TC2000 scan formulas, translated into pandas and "
                "applied to whatever's loaded above. These are fixed threshold rules (not tunable in the "
                "Designer tab) -- run whichever you want, on demand."
            )
            legout_keys = list(LEGOUT_SCANS.keys())
            legout_labels = {k: v["label"] for k, v in LEGOUT_SCANS.items()}
            default_selected = [k for k, v in LEGOUT_SCANS.items() if v.get("default_on")]
            legout_selected = st.multiselect(
                "Scans to run",
                legout_keys,
                default=default_selected,
                format_func=lambda k: legout_labels[k],
                key="legout_selected",
            )
            legout_scope = st.radio(
                "Scope",
                ["Momentum leaders only (faster)", "Everything loaded"],
                horizontal=True,
                key="legout_scope",
            )
            if st.button("Run Legout Scans", key="legout_run_btn"):
                legout_history = history
                if legout_scope.startswith("Momentum"):
                    shortlist = momentum_shortlist(
                        history, settings.universe.momentum_timeframes_days, settings.universe.momentum_top_pct
                    )
                    legout_history = {s: history[s] for s in shortlist if s in history}
                from indicators import build_close_panel, compute_rs_rating_panel

                close_panel = build_close_panel(legout_history)
                rs_rating_panel = compute_rs_rating_panel(
                    close_panel, settings.rs_rating.lookback_periods, settings.rs_rating.period_weights
                )
                rs_panel = {sym: rs_rating_panel[sym] for sym in rs_rating_panel.columns}
                legout_matches = run_legout_scans(legout_history, legout_selected, rs_panel)

                # Same fundamentals/earnings-avoidance treatment as the main
                # Run Scan -- reuses the checkbox/setting defined above,
                # applied only to the (typically small) set of matched
                # symbols rather than the whole scanned scope, so this stays
                # cheap regardless of "Everything loaded" vs "Momentum leaders".
                if not legout_matches.empty:
                    legout_match_symbols = legout_matches["symbol"].unique().tolist()
                    if len(legout_match_symbols) <= MAX_FUNDAMENTALS_AUTO_FETCH:
                        try:
                            legout_fund_client = get_client()
                            legout_fundamentals = get_fundamentals_bulk(legout_fund_client, legout_match_symbols)
                            legout_matches["revenue_growth_pct"] = legout_matches["symbol"].map(
                                lambda s: legout_fundamentals.get(s, {}).get("revenue_growth_pct")
                            )
                            legout_matches["eps_growth_pct"] = legout_matches["symbol"].map(
                                lambda s: legout_fundamentals.get(s, {}).get("eps_growth_pct")
                            )
                            legout_matches["lynch_category"] = legout_matches["symbol"].map(
                                lambda s: LYNCH_CATEGORIES[classify_lynch(legout_fundamentals.get(s, {}))]
                            )
                            if avoid_earnings:
                                legout_earnings_dates = get_earnings_dates(
                                    legout_fund_client, legout_match_symbols, as_of=str(as_of)
                                )
                                legout_matches["days_to_earnings"] = legout_matches.apply(
                                    lambda r: days_to_earnings(r["symbol"], legout_earnings_dates, r["date"]), axis=1
                                )
                                legout_matches = filter_earnings_avoidance(
                                    legout_matches, legout_earnings_dates, str(as_of), int(avoid_earnings_days)
                                )
                        except Exception as exc:
                            st.warning(f"Fundamentals enrichment failed ({exc}) -- showing raw matches only.")
                    else:
                        st.caption(
                            f"Skipping fundamentals enrichment for {len(legout_match_symbols)} matched symbols "
                            f"(cap: {MAX_FUNDAMENTALS_AUTO_FETCH})."
                        )

                st.session_state.legout_results = legout_matches

            legout_results = st.session_state.get("legout_results")
            if legout_results is None:
                st.info("Pick some scans above and click 'Run Legout Scans'.")
            elif legout_results.empty:
                st.warning("No matches -- try 'Everything loaded' scope, or different scans.")
            else:
                st.caption(
                    f"{len(legout_results)} match(es) across {legout_results['symbol'].nunique()} symbol(s). "
                    "Click a row to chart that match below -- the selected row stays highlighted."
                )
                legout_table_event = st.dataframe(
                    legout_results,
                    use_container_width=True,
                    height=300,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="legout_results_table",
                )

                legout_match_labels = [
                    f"{r['symbol']} -- {r['scan']} ({pd.Timestamp(r['date']).date()})"
                    for _, r in legout_results.iterrows()
                ]
                # The selectbox below owns its displayed value via its `key`
                # once set -- Streamlit ignores a later `index=` argument on
                # a keyed widget, so both Prev/Next and a table row click
                # must update the SAME key directly (before the selectbox is
                # instantiated this run), not a separate index variable, or
                # the click gets silently overwritten by the selectbox
                # reading back its own stale keyed state a few lines later.
                if (
                    "legout_chart_pick" not in st.session_state
                    or st.session_state.legout_chart_pick not in legout_match_labels
                ):
                    st.session_state.legout_chart_pick = legout_match_labels[0]

                clicked_rows = legout_table_event.selection.rows if legout_table_event.selection else []
                if clicked_rows:
                    st.session_state.legout_chart_pick = legout_match_labels[clicked_rows[0]]

                current_lidx = legout_match_labels.index(st.session_state.legout_chart_pick)

                col_lprev, col_lpick, col_lnext, col_lperiod = st.columns([1, 3, 1, 2])
                with col_lprev:
                    if st.button("⬅ Prev", key="legout_prev"):
                        st.session_state.legout_chart_pick = legout_match_labels[(current_lidx - 1) % len(legout_match_labels)]
                with col_lnext:
                    if st.button("Next ➡", key="legout_next"):
                        st.session_state.legout_chart_pick = legout_match_labels[(current_lidx + 1) % len(legout_match_labels)]
                with col_lpick:
                    legout_pick_label = st.selectbox("Chart a match", legout_match_labels, key="legout_chart_pick")
                with col_lperiod:
                    legout_period = st.selectbox("Chart period", list(CHART_PERIODS.keys()), index=4, key="legout_chart_period")

                legout_row = legout_results.iloc[legout_match_labels.index(legout_pick_label)]
                legout_pick = legout_row["symbol"]
                if legout_pick in history:
                    st.plotly_chart(
                        plot_symbol(
                            slice_for_period(history[legout_pick], legout_period),
                            legout_pick,
                            marker_date=legout_row["date"],
                        ),
                        use_container_width=True,
                        key="legout_chart",
                    )
                    st.caption(f"Green dashed line = the day **{legout_row['scan']}** matched.")

            with st.expander("Formulas used (and what's skipped)"):
                st.markdown("**Implemented:**")
                for key, spec in LEGOUT_SCANS.items():
                    st.markdown(f"- **{spec['label']}**: `{spec['formula']}`")
                st.markdown("**Not implemented, with reasons:**")
                for name, reason in SKIPPED_SCANS.items():
                    st.markdown(f"- **{name}**: {reason}")

        with st.expander("📊 Fundamentals (growth, float, earnings date, Lynch tag)", expanded=False):
            st.caption(
                "Revenue/EPS growth, float shares, insider buy/sell ratio, next earnings date, and a Peter "
                "Lynch-style category tag -- fetched from FMP on demand. Kept separate from 'Load / Refresh "
                "Data' so the main scan stays fast and cheap on API calls; this only fetches for a small "
                "shortlist you choose below."
            )
            fund_scope = st.radio(
                "Symbols to fetch",
                ["Last scan matches", "Legout scan matches", "Momentum leaders", "Custom list"],
                horizontal=True,
                key="fund_scope",
                help="'Legout scan matches' uses whatever's currently in the Legout scans results table above "
                "(run that first if it's empty) -- gives the fuller column set here (float, insider ratio, "
                "next earnings date) plus a chart picker, beyond the few columns already shown inline there.",
            )
            fund_custom = ""
            if fund_scope == "Custom list":
                fund_custom = st.text_input("Symbols (comma-separated)", key="fund_custom_symbols")

            if st.button("Fetch fundamentals", key="fund_fetch_btn"):
                if fund_scope == "Last scan matches":
                    fund_symbols = results["symbol"].unique().tolist() if not results.empty else []
                elif fund_scope == "Legout scan matches":
                    legout_results_for_fund = st.session_state.get("legout_results")
                    fund_symbols = (
                        legout_results_for_fund["symbol"].unique().tolist()
                        if legout_results_for_fund is not None and not legout_results_for_fund.empty
                        else []
                    )
                elif fund_scope == "Momentum leaders":
                    fund_symbols = momentum_shortlist(
                        history, settings.universe.momentum_timeframes_days, settings.universe.momentum_top_pct
                    )
                else:
                    fund_symbols = [s.strip().upper() for s in fund_custom.split(",") if s.strip()]
                fund_symbols = [s for s in fund_symbols if s in history]

                if not fund_symbols and fund_scope == "Legout scan matches":
                    st.warning("No Legout scan matches yet -- run the Legout scans section above first.")
                elif not fund_symbols:
                    st.warning("No symbols to fetch -- pick a scope with matches, or type some symbols.")
                else:
                    try:
                        client = get_client()
                        progress = st.progress(0.0, text="Fetching fundamentals...")
                        fund_data = get_fundamentals_bulk(
                            client, fund_symbols,
                            on_progress=lambda i, n, s: progress.progress(i / n, text=f"{s} ({i}/{n})"),
                        )
                        fund_earnings = get_earnings_dates(client, fund_symbols, as_of=str(as_of))
                        progress.empty()
                        fund_rows = []
                        for sym in fund_symbols:
                            fd = fund_data.get(sym, {})
                            fund_rows.append({
                                "symbol": sym,
                                "revenue_growth_pct": fd.get("revenue_growth_pct"),
                                "eps_growth_pct": fd.get("eps_growth_pct"),
                                "float_shares": fd.get("float_shares"),
                                "free_float_pct": fd.get("free_float_pct"),
                                "insider_acq_disp_ratio": fd.get("insider_acquired_disposed_ratio"),
                                "next_earnings_date": fund_earnings.get(sym),
                                "days_to_earnings": days_to_earnings(sym, fund_earnings, str(as_of)),
                                "lynch_category": LYNCH_CATEGORIES[classify_lynch(fd)],
                            })
                        st.session_state.fundamentals_df = pd.DataFrame(fund_rows)
                    except Exception as exc:
                        st.error(f"Fundamentals fetch failed: {exc}")

            fdf = st.session_state.get("fundamentals_df")
            if fdf is None or fdf.empty:
                st.info("Pick a scope above and click 'Fetch fundamentals'.")
            else:
                st.dataframe(fdf, use_container_width=True, height=300)
                fund_pick = st.selectbox("Chart one", fdf["symbol"].tolist(), key="fund_chart_pick")
                if fund_pick in history:
                    st.plotly_chart(plot_symbol(history[fund_pick], fund_pick), use_container_width=True, key="fund_chart")

            with st.expander("What do these columns mean?"):
                st.markdown(
                    "- **revenue_growth_pct / eps_growth_pct**: latest quarter's YoY growth. Qullamaggie's own "
                    "bar for an Episodic Pivot: \"triple digit is ideal, mid/high double digits works really well too.\"\n"
                    "- **float_shares / free_float_pct**: shares actually available to trade -- a smaller float "
                    "moves faster on the same dollar volume.\n"
                    "- **insider_acq_disp_ratio**: last quarter's insider shares acquired ÷ disposed -- above 1 "
                    "means insiders bought more than they sold.\n"
                    "- **days_to_earnings**: days until the next reported earnings date. \"Avoid buying 3-days "
                    "before earnings\" (Qullamaggie's Laws of Swing) -- also enforced directly in the main scan "
                    "via the checkbox above 'Run Scan'.\n"
                    "- **lynch_category**: a best-effort Peter Lynch-style tag from growth rate alone (Fast "
                    "Grower 25%+, Stalwart 8-25%, Slow Grower ~flat, Turnaround/Cyclical for negative growth) -- "
                    "simplified, since true cyclical/turnaround/asset-play calls need sector and balance-sheet "
                    "context this app doesn't fetch."
                )


# --------------------------------------------------------------------------
# Tab: Backtest & Optimizer
# --------------------------------------------------------------------------
with tab_backtest:
    st.subheader("Full backtest")
    if not history:
        st.info("Load data from the sidebar first.")
    else:
        bt_symbol_limit = st.selectbox(
            "Limit to symbol (optional -- backtest one stock instead of the whole loaded universe)",
            ["(all loaded symbols)"] + sorted(history.keys()),
            key="bt_symbol_limit",
        )
        bt_history = (
            {bt_symbol_limit: history[bt_symbol_limit]}
            if bt_symbol_limit != "(all loaded symbols)"
            else history
        )

        bt_setup = st.selectbox(
            "Setup", list(SETUP_REGISTRY.keys()), format_func=lambda k: SETUP_REGISTRY[k]["label"], key="bt_setup"
        )
        if st.button("Run backtest"):
            signals = scan_signals_over_history(settings, bt_history, bt_setup)
            result = run_backtest(settings.backtest, signals, side=SETUP_REGISTRY[bt_setup]["side"], setup_name=bt_setup)
            st.session_state.bt_result = result

        result = st.session_state.get("bt_result")
        if result is not None:
            stats = compute_stats(result)
            if stats["trade_count"] == 0:
                st.warning("No trades were generated with these settings over this period.")
            else:
                render_stat_badges(stats)
            if not result.equity_curve.empty:
                st.line_chart(result.equity_curve, height=300)
            st.dataframe(result.trades_df(), use_container_width=True, height=300)

        st.divider()
        st.subheader("Parameter grid optimizer")
        st.caption("Comma-separated values per parameter, e.g. '3,4,5' for Min ADR %.")

        opt_setup = st.selectbox(
            "Setup ", list(SETUP_REGISTRY.keys()), format_func=lambda k: SETUP_REGISTRY[k]["label"], key="opt_setup"
        )
        opt_attr = SETUP_REGISTRY[opt_setup]["settings_attr"]

        grid_param_options = {
            "breakout": ["min_adr_pct", "prior_move_min_pct", "max_consolidation_range_pct", "breakout_volume_ratio_min", "min_rs_rating"],
            "episodic_pivot": ["min_gap_pct", "min_volume_ratio", "max_prior_range_pct"],
            "parabolic_short": ["min_extension_adr_multiple", "min_run_up_pct"],
            "cup_with_handle": ["min_cup_depth_pct", "max_handle_depth_pct", "min_recovery_pct", "breakout_volume_ratio_min"],
            "double_bottom": ["max_low_difference_pct", "breakout_volume_ratio_min"],
            "flat_base": ["max_range_pct", "min_prior_move_pct", "breakout_volume_ratio_min"],
            "ascending_base": ["max_segment_depth_pct", "breakout_volume_ratio_min"],
            "high_tight_flag": ["min_run_up_pct", "max_flag_depth_pct", "breakout_volume_ratio_min"],
        }[opt_setup]

        param_grid = {}
        for p in grid_param_options:
            raw = st.text_input(f"{p} values", key=f"grid_{opt_setup}_{p}")
            if raw.strip():
                try:
                    param_grid[f"{opt_attr}.{p}"] = [float(v) for v in raw.split(",")]
                except ValueError:
                    st.warning(f"Could not parse values for {p}")

        if st.button("Run grid search") and param_grid:
            n_combos = 1
            for v in param_grid.values():
                n_combos *= len(v)
            progress = st.progress(0.0, text=f"Running {n_combos} combinations...")

            def on_opt_progress(i, n, combo):
                progress.progress(i / n, text=f"{i}/{n}: {combo}")

            grid_results = run_grid_search(settings, bt_history, opt_setup, param_grid, on_progress=on_opt_progress)
            progress.empty()
            st.session_state.grid_results = grid_results

        grid_results = st.session_state.get("grid_results")
        if grid_results is not None and not grid_results.empty:
            st.caption("🟢 good  🟡 mixed/borderline  🔴 weak, per-column -- same rule-of-thumb thresholds as above.")
            st.dataframe(style_results_dataframe(grid_results), use_container_width=True, height=350)

        st.divider()
        st.subheader("Compare all patterns")
        st.caption(
            "Which pattern actually held up historically? Runs every enabled setup's backtest over the "
            + ("selected symbol" if bt_symbol_limit != "(all loaded symbols)" else "whole loaded universe")
            + " with its current settings, ranked by expectancy."
        )
        if st.button("Compare all patterns"):
            compare_rows = []
            compare_progress = st.progress(0.0, text="Backtesting each pattern...")
            enabled = [name for name, spec in SETUP_REGISTRY.items() if getattr(settings, spec["settings_attr"]).enabled]
            for i, name in enumerate(enabled):
                spec = SETUP_REGISTRY[name]
                signals = scan_signals_over_history(settings, bt_history, name)
                result = run_backtest(settings.backtest, signals, side=spec["side"], setup_name=name)
                stats = compute_stats(result)
                compare_rows.append({"pattern": spec["label"], "side": spec["side"], **stats})
                compare_progress.progress((i + 1) / len(enabled), text=f"{spec['label']} done")
            compare_progress.empty()
            compare_df = pd.DataFrame(compare_rows).sort_values(
                "expectancy_r", ascending=False, na_position="last"
            ).reset_index(drop=True)
            st.session_state.pattern_comparison = compare_df

        pattern_comparison = st.session_state.get("pattern_comparison")
        if pattern_comparison is not None and not pattern_comparison.empty:
            st.caption("🟢 good  🟡 mixed/borderline  🔴 weak, per-column -- ranked by expectancy (most useful historically first).")
            st.dataframe(style_results_dataframe(pattern_comparison), use_container_width=True, height=300)
            best = pattern_comparison.iloc[0]
            if pd.notna(best.get("expectancy_r")) and best["trade_count"] > 0:
                st.success(
                    f"Best performer here: **{best['pattern']}** -- expectancy {best['expectancy_r']:.2f}R "
                    f"over {int(best['trade_count'])} trades, win rate {best['win_rate']:.1f}%, "
                    f"CAGR {best['cagr']:.1f}%. Small sample sizes (especially with 'Limit to symbol') can be "
                    "noisy -- treat this as a starting point, not proof."
                )


# --------------------------------------------------------------------------
# Tab: Parameter Finder (reverse-fit to a known example)
# --------------------------------------------------------------------------
with tab_finder:
    st.subheader("What made this trade work?")
    st.caption(
        "Give a symbol and the approximate date of a known good move. First "
        "you'll see the indicator values that stock actually had at that "
        "moment; then you can search for threshold combinations that would "
        "flag it, validated against the whole loaded universe/history so a "
        "one-off overfit doesn't masquerade as a real edge."
    )
    if not history:
        st.info("Load data from the sidebar first (include the target symbol in Custom symbols, or make sure it's in the screener-built universe).")
    else:
        pf_symbol = st.selectbox("Symbol", sorted(history.keys()), key="pf_symbol")
        pf_date = st.date_input("Approximate date of the move (e.g. the breakout day)", key="pf_date")
        pf_setup = st.selectbox(
            "Setup type", list(SETUP_REGISTRY.keys()), format_func=lambda k: SETUP_REGISTRY[k]["label"], key="pf_setup"
        )

        if st.button("Inspect target"):
            from backtest.target_fit import inspect_target
            from indicators import build_close_panel, compute_rs_rating_panel

            close_panel = build_close_panel(history)
            rs_panel = compute_rs_rating_panel(close_panel, settings.rs_rating.lookback_periods, settings.rs_rating.period_weights)
            rs_series = rs_panel[pf_symbol] if pf_symbol in rs_panel.columns else None
            observed = inspect_target(history[pf_symbol], settings, pf_setup, pf_date, rs_rating=rs_series)
            st.session_state.pf_observed = observed
            st.json({k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in observed.items()})

            marker_date = observed.get("date_used")
            st.plotly_chart(
                plot_symbol(
                    history[pf_symbol], pf_symbol, marker_date=marker_date,
                    setup_key=pf_setup, settings=settings, row=observed,
                    side=SETUP_REGISTRY[pf_setup]["side"],
                ),
                use_container_width=True,
                key="param_finder_chart",
            )
            st.caption(
                "Shaded region = the window the detector looked at; dashed yellow line = its "
                "breakout/resistance level; green line = the target date."
            )

        if st.button("Search parameters that capture this"):
            progress = st.progress(0.0, text="Searching...")

            def on_pf_progress(i, n, combo):
                progress.progress(i / n, text=f"{i}/{n}: {combo}")

            try:
                out = find_parameters_for_target(
                    settings, history, pf_setup, pf_symbol, pf_date, on_progress=on_pf_progress
                )
                progress.empty()
                st.session_state.pf_result = out
                st.session_state.pf_setup_used = pf_setup
            except Exception as exc:
                progress.empty()
                st.error(f"Search failed: {exc}")
                st.code(traceback.format_exc())

        pf_result = st.session_state.get("pf_result")
        if pf_result is not None:
            st.write("Observed values at target date:")
            st.json({k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in pf_result["observed"].items()})

            candidates = pf_result["candidates"]
            st.write("Candidate parameter sets (ranked: catches target first, then broad-universe expectancy):")
            st.caption("🟢 good  🟡 mixed/borderline  🔴 weak, per-column -- rule-of-thumb thresholds, not guarantees.")
            st.dataframe(style_results_dataframe(candidates), use_container_width=True, height=350)

            if not candidates.empty:
                st.markdown("**Send a candidate to Backtest & Optimizer**")
                param_cols = [c for c in candidates.columns if "." in c]
                row_labels = [
                    f"Row {i}: " + ", ".join(f"{c.split('.', 1)[1]}={candidates.loc[i, c]}" for c in param_cols)
                    for i in candidates.index
                ]
                chosen_idx = st.selectbox(
                    "Candidate to use as a starting point",
                    candidates.index.tolist(),
                    format_func=lambda i: row_labels[i],
                    key="pf_chosen_candidate",
                )
                if st.button("Copy to Backtest & Optimizer"):
                    used_setup = st.session_state.pf_setup_used
                    row = candidates.loc[chosen_idx]

                    grid_values = {}
                    for col in param_cols:
                        set_nested(settings, col, row[col])
                        param_name = col.split(".", 1)[1]
                        value = row[col]
                        # pre-fill the grid-search box for this param with a small
                        # range around the chosen value, so there's something to
                        # tweak from rather than a single fixed number
                        grid_key = f"grid_{used_setup}_{param_name}"
                        grid_values[grid_key] = f"{value * 0.8:.2f},{value:.2f},{value * 1.2:.2f}"

                    # Can't write st.session_state["bt_setup"]/"opt_setup"/grid_*
                    # directly here -- those widgets were already instantiated
                    # earlier in this same run. Queue the values and apply them
                    # at the top of the script on the rerun instead.
                    st.session_state.pf_pending_apply = {"setup": used_setup, "grid_values": grid_values}
                    st.session_state.pf_copy_message = (
                        f"Copied row {chosen_idx}'s parameters into the current settings "
                        f"({SETUP_REGISTRY[used_setup]['label']}) and pre-filled the grid "
                        "search ranges. Switch to the Backtest & Optimizer tab to review and tweak."
                    )
                    st.rerun()

        if st.session_state.get("pf_copy_message"):
            st.success(st.session_state.pf_copy_message)


# --------------------------------------------------------------------------
# Tab: Live Mode -- auto-refreshing watchlist scan
#
# EOD price data itself doesn't change intraday, so "live" here means: poll
# FMP's lightweight /quote endpoint on a timer to get each watchlist symbol's
# current price/day-high/day-low/volume, synthesize that into a provisional
# "today" bar on top of the already-loaded daily history, and re-run the scan
# against it -- so matches reflect the current session, not just yesterday's
# close. Requires the watchlist symbols to already have full daily history
# loaded from the sidebar first (Live Mode only refreshes today's bar, it
# doesn't pull years of history on every tick).
# --------------------------------------------------------------------------
with tab_live:
    st.subheader("Live Mode")
    st.caption(
        "Polls current price/volume for a small watchlist on a timer and re-scans for forming patterns, "
        "prioritized by RS rating (momentum). Load full history for these symbols from the sidebar first -- "
        "Live Mode only refreshes today's bar, not years of history, on every tick."
    )

    if "live_mode_on" not in st.session_state:
        st.session_state.live_mode_on = False
    if "live_watchlist" not in st.session_state:
        st.session_state.live_watchlist = []

    col_a, col_b = st.columns([2, 1])
    with col_a:
        watchlist_input = st.text_input(
            f"Watchlist symbols, comma-separated (max {LIVE_MODE_MAX_SYMBOLS})",
            value=",".join(st.session_state.live_watchlist),
            key="live_watchlist_input",
        )
    with col_b:
        refresh_label = st.selectbox("Refresh every", ["30s", "60s", "2min", "5min"], index=1, key="live_refresh_label")
    refresh_seconds = {"30s": 30, "60s": 60, "2min": 120, "5min": 300}[refresh_label]

    st.session_state.live_watchlist = [
        s.strip().upper() for s in watchlist_input.split(",") if s.strip()
    ][:LIVE_MODE_MAX_SYMBOLS]

    live_on = st.toggle("Live Mode on", value=st.session_state.live_mode_on, key="live_mode_toggle")
    st.session_state.live_mode_on = live_on

    missing = [s for s in st.session_state.live_watchlist if s not in history]
    if missing:
        st.warning(
            f"Not yet loaded (load from the sidebar first, including these in Custom symbols): {', '.join(missing)}"
        )

    @st.fragment(run_every=refresh_seconds if live_on else None)
    def render_live_panel():
        if not st.session_state.live_mode_on:
            st.info("Live Mode is off.")
            return
        watchlist = [s for s in st.session_state.live_watchlist if s in st.session_state.history]
        if not watchlist:
            st.warning("Add at least one already-loaded symbol to the watchlist above.")
            return
        try:
            client = get_client()
            quotes = client.get_quotes(watchlist)
            st.session_state.history = apply_live_quotes(st.session_state.history, quotes)
            live_results = run_scan(
                st.session_state.settings,
                {s: st.session_state.history[s] for s in watchlist},
                auto_detect=True,
            )
            st.caption(f"Last refreshed {pd.Timestamp.now().strftime('%H:%M:%S')} -- watching {len(watchlist)} symbol(s).")
            if live_results.empty:
                st.write("No pattern matches right now.")
            else:
                st.dataframe(live_results, use_container_width=True, height=300)
        except Exception as exc:
            st.error(f"Live refresh failed: {exc}")

    render_live_panel()


# --------------------------------------------------------------------------
# Tab: Sell Alerts -- watchlist of held stocks, alert when the daily close
# drops below a configurable 10- or 20-day moving average (a classic
# trailing-stop/sell trigger, the mirror image of the buy-side setups
# elsewhere in this app). The watchlist lives in config/sell_watchlist.yaml
# (committed) so the same file also drives the scheduled GitHub Actions
# check (`python cli.py sell-alerts`) that fires even when nobody has this
# tab open -- Streamlit Community Cloud has no background/cron capability
# of its own, so that scheduled job is the *reliable* path; this tab is for
# interactive management, an on-demand check, and live monitoring while
# markets are open and the tab stays open.
# --------------------------------------------------------------------------
with tab_sell:
    st.subheader("Sell Alerts")
    st.caption(
        "Watch stocks you're holding and get a push notification the moment the daily close drops below "
        "the moving average you choose (10-day is the common default; use 20-day for slower-moving names). "
        "Delivered via ntfy.sh -- set NTFY_TOPIC in your .env/secrets, then subscribe to that same topic in "
        "the free ntfy app or at ntfy.sh/<your-topic> to receive them. Changes made here are saved to "
        "config/sell_watchlist.yaml; on Streamlit Cloud the container's disk doesn't persist, so edits there "
        "only stick for the current session -- edit the file on GitHub (or run the app locally) to change "
        "what the scheduled daily check below uses."
    )

    if "sell_watchlist" not in st.session_state:
        st.session_state.sell_watchlist = load_watchlist()
    if "sell_alerts_sent_today" not in st.session_state:
        st.session_state.sell_alerts_sent_today = set()

    st.markdown("**Watchlist**")
    if not st.session_state.sell_watchlist:
        st.info("No symbols yet -- add one below.")
    else:
        for i, entry in enumerate(st.session_state.sell_watchlist):
            col_sym, col_ma, col_remove = st.columns([2, 2, 1])
            col_sym.write(entry["symbol"])
            col_ma.write(f"{entry['ma_period']}-day MA")
            if col_remove.button("Remove", key=f"sell_remove_{i}"):
                st.session_state.sell_watchlist.pop(i)
                save_watchlist(st.session_state.sell_watchlist)
                st.rerun()

    col_add_sym, col_add_ma, col_add_btn = st.columns([2, 2, 1])
    with col_add_sym:
        new_symbol = st.text_input("Add symbol", key="sell_new_symbol").strip().upper()
    with col_add_ma:
        new_ma_period = st.radio("MA period", [10, 20], horizontal=True, key="sell_new_ma_period")
    with col_add_btn:
        st.write("")
        if st.button("Add", key="sell_add_btn") and new_symbol:
            existing_symbols = {e["symbol"] for e in st.session_state.sell_watchlist}
            if new_symbol not in existing_symbols:
                st.session_state.sell_watchlist.append({"symbol": new_symbol, "ma_period": int(new_ma_period)})
                save_watchlist(st.session_state.sell_watchlist)
                st.rerun()

    st.divider()

    col_check, col_test = st.columns([1, 1])
    with col_check:
        sell_check_now = st.button("Check now", key="sell_check_now")
    with col_test:
        if st.button("Send test alert", key="sell_test_alert"):
            try:
                send_ntfy_alert("Sell Alerts test", "If you got this, your ntfy.sh setup works.")
                st.success("Test alert sent -- check your ntfy topic.")
            except Exception as exc:
                st.error(f"Test alert failed: {exc}")

    if sell_check_now:
        if not st.session_state.sell_watchlist:
            st.warning("Add at least one symbol to the watchlist above first.")
        else:
            sell_symbols = [e["symbol"] for e in st.session_state.sell_watchlist]
            try:
                sell_client = get_client()
                sell_end = pd.Timestamp.today().normalize()
                sell_start = sell_end - pd.Timedelta(days=90)
                sell_history = get_history_bulk(
                    sell_client, sell_symbols, sell_start.strftime("%Y-%m-%d"), sell_end.strftime("%Y-%m-%d")
                )
                st.session_state.sell_check_results = check_watchlist(sell_history, st.session_state.sell_watchlist)
            except Exception as exc:
                st.error(f"Check failed: {exc}")

    sell_results = st.session_state.get("sell_check_results")
    if sell_results is not None and not sell_results.empty:
        def _style_sell_row(row):
            if row["is_below"] is True:
                return ["background-color: #e74c3c; color: black;"] * len(row)
            elif row["is_below"] is False:
                return ["background-color: #2ecc71; color: black;"] * len(row)
            return [""] * len(row)

        st.dataframe(
            sell_results.style.apply(_style_sell_row, axis=1),
            use_container_width=True,
            height=min(80 + 35 * len(sell_results), 350),
        )

        below_rows = sell_results[sell_results["is_below"] == True]
        if not below_rows.empty:
            fresh_alerts = below_rows[~below_rows["symbol"].isin(st.session_state.sell_alerts_sent_today)]
            if not fresh_alerts.empty:
                try:
                    alerted = send_sell_alerts(fresh_alerts)
                    st.session_state.sell_alerts_sent_today.update(alerted)
                    st.warning(f"Sell alert sent for: {', '.join(alerted)}")
                except Exception as exc:
                    st.error(f"Alert sending failed: {exc}")
            else:
                st.warning(f"Below MA (already alerted this session): {', '.join(below_rows['symbol'])}")

    st.divider()
    st.markdown("**Live checking** (while this tab stays open, during market hours)")

    if "sell_live_on" not in st.session_state:
        st.session_state.sell_live_on = False

    col_live_toggle, col_live_refresh = st.columns([1, 1])
    with col_live_toggle:
        sell_live_on = st.toggle("Live checking on", value=st.session_state.sell_live_on, key="sell_live_toggle")
        st.session_state.sell_live_on = sell_live_on
    with col_live_refresh:
        sell_refresh_label = st.selectbox(
            "Refresh every", ["30s", "60s", "2min", "5min"], index=1, key="sell_live_refresh_label"
        )
    sell_refresh_seconds = {"30s": 30, "60s": 60, "2min": 120, "5min": 300}[sell_refresh_label]

    @st.fragment(run_every=sell_refresh_seconds if sell_live_on else None)
    def render_sell_live_panel():
        if not st.session_state.sell_live_on:
            st.info("Live checking is off.")
            return
        watchlist = st.session_state.sell_watchlist
        if not watchlist:
            st.warning("Add at least one symbol to the watchlist above.")
            return
        symbols = [e["symbol"] for e in watchlist]
        missing = [s for s in symbols if s not in st.session_state.history]
        if missing:
            st.warning(
                f"Not yet loaded (load from the sidebar first, including these in Custom symbols): {', '.join(missing)}"
            )
        watch_symbols = [s for s in symbols if s in st.session_state.history]
        if not watch_symbols:
            return
        try:
            live_client = get_client()
            quotes = live_client.get_quotes(watch_symbols)
            st.session_state.history = apply_live_quotes(st.session_state.history, quotes)
            live_watchlist = [e for e in watchlist if e["symbol"] in watch_symbols]
            live_results = check_watchlist(st.session_state.history, live_watchlist)
            st.caption(f"Last refreshed {pd.Timestamp.now().strftime('%H:%M:%S')} -- watching {len(watch_symbols)} symbol(s).")
            st.dataframe(live_results, use_container_width=True, height=min(80 + 35 * len(live_results), 300))

            below_now = live_results[live_results["is_below"] == True]
            fresh_live_alerts = below_now[~below_now["symbol"].isin(st.session_state.sell_alerts_sent_today)]
            if not fresh_live_alerts.empty:
                alerted = send_sell_alerts(fresh_live_alerts)
                st.session_state.sell_alerts_sent_today.update(alerted)
                st.warning(f"Sell alert sent for: {', '.join(alerted)}")
        except Exception as exc:
            st.error(f"Live check failed: {exc}")

    render_sell_live_panel()


# --------------------------------------------------------------------------
# Tab: Paper Trading -- log a hand-picked (symbol, decision date) "buy-in"
# and simulate it through the exact same trade-management engine (stop-loss,
# sell-half at an R-target or day-limit, then a trailing moving-average exit
# for the rest) that every automated setup in this app already uses, via
# `paper_trading.py` -- a thin journal layer over backtest.engine.run_backtest,
# not a new simulator. Uses its OWN fixed exit-rule settings (below),
# independent of the Designer tab's "Backtest mechanics", so a logged trade's
# shown outcome never silently shifts just because those are being tuned
# elsewhere for scanning purposes.
# --------------------------------------------------------------------------
with tab_paper:
    st.subheader("Paper Trading")
    st.caption(
        "Log a stock and the day you'd have decided to buy it -- this simulates a real 3-5 day swing trade "
        "(stop loss from entry, sell half at a profit target or after a few days, then trail the rest on a "
        "moving average) using the exact same trade-management engine as the Backtest tab, so the numbers "
        "are grounded in real mechanics, not a guess. The point is to build a track record of your own "
        "pattern-spotting and see, in aggregate, whether it would have made money. The 'decision date' is "
        "when you'd have decided to buy -- the simulated entry fills at the next trading day's open, same as "
        "every automated setup elsewhere in this app."
    )

    if "paper_trades" not in st.session_state:
        st.session_state.paper_trades = pt_load_trades()
    if "paper_rules" not in st.session_state:
        st.session_state.paper_rules = pt_load_rules()

    with st.expander("Paper trading rules (independent of the Designer tab)", expanded=False):
        st.caption(
            "These stay fixed regardless of what you tune in Designer -> Backtest mechanics, so your journal's "
            "past results don't silently change out from under you."
        )
        pr = st.session_state.paper_rules
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            pr.stop_adr_multiple = st.slider(
                "Stop distance (x ADR)", 0.25, 3.0, float(pr.stop_adr_multiple), 0.25, key="pt_stop_adr"
            )
            pr.partial_profit_r_multiple = st.slider(
                "Sell-half target (R)", 0.5, 5.0, float(pr.partial_profit_r_multiple), 0.5, key="pt_partial_r"
            )
        with col_r2:
            pr.partial_profit_max_days = st.slider(
                "Sell half by day N regardless", 1, 15, int(pr.partial_profit_max_days), key="pt_partial_days"
            )
            pr.partial_profit_fraction = st.slider(
                "Fraction sold at partial", 0.1, 1.0, float(pr.partial_profit_fraction), 0.05, key="pt_partial_frac"
            )
        with col_r3:
            pr.trail_ema_period = st.slider(
                "Trailing MA period (days)", 5, 50, int(pr.trail_ema_period), key="pt_trail_period"
            )
            pr.trail_ma_type = st.selectbox(
                "Trailing MA type", ["sma", "ema"], index=0 if pr.trail_ma_type == "sma" else 1, key="pt_trail_type"
            )
        col_r4, col_r5 = st.columns(2)
        with col_r4:
            pr.initial_capital = st.number_input(
                "Hypothetical starting capital ($)", value=float(pr.initial_capital), step=5000.0, key="pt_capital"
            )
            pr.risk_pct_per_trade = st.slider(
                "Risk % per trade", 0.1, 5.0, float(pr.risk_pct_per_trade), 0.1, key="pt_risk_pct"
            )
        with col_r5:
            pr.max_positions = st.slider("Max concurrent positions", 1, 20, int(pr.max_positions), key="pt_max_pos")
            pr.max_position_pct_of_equity = st.slider(
                "Max position size (% of equity)", 5.0, 100.0, float(pr.max_position_pct_of_equity), 5.0,
                key="pt_max_pos_pct",
            )
        pt_save_rules(pr)

    st.markdown("**Log a buy-in**")
    col_add_sym, col_add_date, col_add_tag = st.columns([1, 1, 2])
    with col_add_sym:
        paper_new_symbol = st.text_input("Symbol", key="paper_new_symbol").strip().upper()
    with col_add_date:
        paper_new_date = st.date_input("Decision date", value=pd.Timestamp(as_of), key="paper_new_date")
    with col_add_tag:
        paper_new_tag = st.selectbox(
            "Pattern you think you spotted",
            ["Gut feel / other"] + [spec["label"] for spec in SETUP_REGISTRY.values()],
            key="paper_new_tag",
        )
    paper_new_notes = st.text_input("Notes (optional)", key="paper_new_notes")
    if st.button("Log buy-in", key="paper_add_btn") and paper_new_symbol:
        new_entry_key = (paper_new_symbol, str(paper_new_date))
        existing_keys = {(e["symbol"], e["decision_date"]) for e in st.session_state.paper_trades}
        if new_entry_key in existing_keys:
            st.warning(f"Already logged {paper_new_symbol} on {paper_new_date} -- remove it below first if you want to re-log it.")
        else:
            st.session_state.paper_trades.append(
                {
                    "symbol": paper_new_symbol,
                    "decision_date": str(paper_new_date),
                    "setup_tag": paper_new_tag,
                    "notes": paper_new_notes,
                }
            )
            pt_save_trades(st.session_state.paper_trades)
            st.rerun()

    st.divider()

    if not st.session_state.paper_trades:
        st.info("No paper trades logged yet -- add one above.")
    else:
        paper_result, paper_skipped = pt_simulate(history, st.session_state.paper_trades, st.session_state.paper_rules)
        paper_joined = pt_join_results(st.session_state.paper_trades, paper_result, paper_skipped)

        stats = compute_stats(paper_result)
        render_stat_badges(stats)
        if paper_result.equity_curve is not None and not paper_result.equity_curve.empty:
            eq_fig = go.Figure()
            eq_fig.add_trace(
                go.Scatter(x=paper_result.equity_curve.index, y=paper_result.equity_curve.values, mode="lines", line=dict(color="mediumseagreen"))
            )
            eq_fig.update_layout(
                title=dict(text="Hypothetical account equity", font=dict(size=13)),
                height=250, margin=dict(l=10, r=10, t=35, b=10),
            )
            st.plotly_chart(eq_fig, use_container_width=True, key="paper_equity_chart")

        st.markdown("**Logged trades**")

        def _style_paper_row(row):
            status = row["status"]
            color = {"Win": "#2ecc71", "Loss": "#e74c3c", "Open": "#f1c40f", "Not simulated": "#888888"}.get(status)
            if color is None:
                return [""] * len(row)
            return [f"background-color: {color}; color: black;"] * len(row)

        display_cols = [
            "symbol", "decision_date", "setup_tag", "entry_date", "entry_price",
            "partial_date", "partial_price", "exit_date", "exit_price", "exit_reason",
            "r_multiple", "pnl", "status",
        ]
        paper_table_event = st.dataframe(
            paper_joined[display_cols].style.apply(_style_paper_row, axis=1),
            use_container_width=True,
            height=min(80 + 35 * len(paper_joined), 350),
            on_select="rerun",
            selection_mode="single-row",
            key="paper_results_table",
        )

        paper_labels = [f"{r['symbol']} -- {r['decision_date']}" for _, r in paper_joined.iterrows()]
        if (
            "paper_chart_pick" not in st.session_state
            or st.session_state.paper_chart_pick not in paper_labels
        ):
            st.session_state.paper_chart_pick = paper_labels[0]

        clicked_paper_rows = paper_table_event.selection.rows if paper_table_event.selection else []
        if clicked_paper_rows:
            st.session_state.paper_chart_pick = paper_labels[clicked_paper_rows[0]]

        col_paper_pick, col_paper_remove = st.columns([3, 1])
        with col_paper_pick:
            paper_pick_label = st.selectbox("Chart this trade", paper_labels, key="paper_chart_pick")
        paper_pick_idx = paper_labels.index(paper_pick_label)
        with col_paper_remove:
            st.write("")
            if st.button("Remove this trade", key="paper_remove_btn"):
                st.session_state.paper_trades.pop(paper_pick_idx)
                pt_save_trades(st.session_state.paper_trades)
                st.rerun()

        paper_pick_row = paper_joined.iloc[paper_pick_idx]
        paper_pick_symbol = paper_pick_row["symbol"]
        if paper_pick_symbol not in history:
            st.warning(f"{paper_pick_symbol} isn't loaded -- load it from the sidebar first to see its chart.")
        else:
            st.plotly_chart(
                plot_symbol(
                    history[paper_pick_symbol],
                    paper_pick_symbol,
                    marker_date=pd.Timestamp(paper_pick_row["decision_date"]),
                    side="long",
                    entry_date=paper_pick_row["entry_date"],
                    entry_price=paper_pick_row["entry_price"],
                    partial_date=paper_pick_row["partial_date"],
                    partial_price=paper_pick_row["partial_price"],
                    exit_date=paper_pick_row["exit_date"],
                    exit_price=paper_pick_row["exit_price"],
                    exit_reason=paper_pick_row["exit_reason"],
                    r_multiple=paper_pick_row["r_multiple"],
                ),
                use_container_width=True,
                key="paper_chart",
            )
            if paper_pick_row["notes"]:
                st.caption(f"Notes: {paper_pick_row['notes']}")


# --------------------------------------------------------------------------
# Tab: Settings (preset management + raw view)
# --------------------------------------------------------------------------
with tab_settings:
    st.subheader("Current settings")
    st.json(settings.to_dict())

    st.divider()
    st.subheader("Compare presets")
    all_presets = [f"⭐ {n}" for n in presets.list_builtin_presets()] + presets.list_presets()
    to_compare = st.multiselect("Presets to compare", all_presets)
    if to_compare:
        st.dataframe(presets.compare_presets(to_compare), use_container_width=True, height=500)

    st.divider()
    if st.button("Reset to defaults"):
        st.session_state.settings = presets.load_default()
        st.session_state.settings_version += 1
        st.rerun()
