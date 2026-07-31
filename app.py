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
from data.cache import apply_live_quotes
from data.fmp_client import FMPClient, FMPError
from scanner import (
    TIMEFRAME_LABELS,
    ensure_benchmark_loaded,
    load_scan_universe_history,
    momentum_shortlist,
    relative_strength_resilience,
    run_scan,
    scan_signals_over_history,
    top_movers_by_timeframe,
)
from settings import Settings
from setups import SETUP_REGISTRY
from setups.explanations import PATTERN_EXPLANATIONS
from ui_helpers import render_stat_badges, style_results_dataframe

LIVE_MODE_MAX_SYMBOLS = 30

st.set_page_config(page_title="Qullamaggie Scanner", layout="wide")

if "settings" not in st.session_state:
    st.session_state.settings = presets.load_default()
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

    with st.expander("Universe filters (market cap, liquidity, momentum)"):
        u = st.session_state.settings.universe
        u.min_market_cap = st.number_input(
            "Min market cap ($)", value=float(u.min_market_cap), step=50_000_000.0, format="%.0f", key="u_min_mcap"
        )
        u.max_market_cap = st.number_input(
            "Max market cap ($, 0 = no cap)", value=float(u.max_market_cap), step=500_000_000.0, format="%.0f", key="u_max_mcap"
        )
        u.min_avg_dollar_volume = st.number_input(
            "Min avg $ volume (liquidity floor)", value=float(u.min_avg_dollar_volume), step=1_000_000.0, format="%.0f", key="u_min_dv"
        )
        u.max_symbols = st.number_input("Max universe size", value=int(u.max_symbols), step=50, key="u_max_syms")
        u.momentum_top_pct = st.slider(
            "Momentum leaders: top % of movers", 0.5, 20.0, float(u.momentum_top_pct), 0.5, key="u_mom_pct"
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
    existing = presets.list_presets()
    chosen = st.selectbox("Load preset", ["(current)"] + existing)
    if chosen != "(current)" and st.button("Apply selected preset"):
        st.session_state.settings = presets.load_preset(chosen)
        st.rerun()

    new_name = st.text_input("Save current settings as")
    if st.button("Save preset") and new_name:
        presets.save_preset(new_name, st.session_state.settings)
        st.success(f"Saved preset '{new_name}'.")
        st.rerun()


settings: Settings = st.session_state.settings
history = st.session_state.history

tab_scanner, tab_designer, tab_backtest, tab_finder, tab_live, tab_settings = st.tabs(
    ["Scanner", "Designer", "Backtest & Optimizer", "Parameter Finder", "Live Mode", "Settings"]
)


# --------------------------------------------------------------------------
# Chart helper
# --------------------------------------------------------------------------
def plot_symbol(df: pd.DataFrame, symbol: str, marker_date=None):
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name=symbol
        )
    )
    for period, color in ((10, "orange"), (20, "blue")):
        ema = df["close"].ewm(span=period, adjust=False).mean()
        fig.add_trace(go.Scatter(x=df.index, y=ema, line=dict(width=1, color=color), name=f"EMA{period}"))
    if marker_date is not None and marker_date in df.index:
        fig.add_vline(x=marker_date, line_dash="dash", line_color="green")
    fig.update_layout(height=450, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
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

            s.enabled = st.checkbox("Enabled", value=s.enabled, key=f"{setup_choice}_enabled")

            if setup_choice == "breakout":
                s.min_adr_pct = st.slider("Min ADR %", 0.5, 15.0, float(s.min_adr_pct), 0.5, key="d_bo_adr")
                s.prior_move_min_pct = st.slider("Prior move min %", 0.0, 200.0, float(s.prior_move_min_pct), 5.0, key="d_bo_pmmin")
                s.prior_move_max_pct = st.slider("Prior move max %", 10.0, 500.0, float(s.prior_move_max_pct), 5.0, key="d_bo_pmmax")
                s.min_consolidation_days = st.slider("Min consolidation (days)", 3, 60, int(s.min_consolidation_days), key="d_bo_mincon")
                s.max_consolidation_days = st.slider("Max consolidation (days)", 5, 90, int(s.max_consolidation_days), key="d_bo_maxcon")
                s.max_consolidation_range_pct = st.slider("Max consolidation range %", 5.0, 80.0, float(s.max_consolidation_range_pct), 1.0, key="d_bo_range")
                s.breakout_volume_ratio_min = st.slider("Min breakout volume ratio", 0.5, 6.0, float(s.breakout_volume_ratio_min), 0.1, key="d_bo_vol")
                s.min_rs_rating = st.slider("Min RS rating", 1, 99, int(s.min_rs_rating), key="d_bo_rs")
                s.require_above_ema10 = st.checkbox("Require close > EMA10", value=s.require_above_ema10, key="d_bo_ema10")
                s.require_above_ema20 = st.checkbox("Require close > EMA20", value=s.require_above_ema20, key="d_bo_ema20")
            elif setup_choice == "episodic_pivot":
                s.min_gap_pct = st.slider("Min gap %", 1.0, 50.0, float(s.min_gap_pct), 1.0, key="d_ep_gap")
                s.min_volume_ratio = st.slider("Min volume ratio", 0.5, 10.0, float(s.min_volume_ratio), 0.1, key="d_ep_vol")
                s.max_prior_range_pct = st.slider("Max prior base range %", 5.0, 60.0, float(s.max_prior_range_pct), 1.0, key="d_ep_range")
                s.lookback_base_days = st.slider("Prior base lookback (days)", 5, 60, int(s.lookback_base_days), key="d_ep_lookback")
            elif setup_choice == "parabolic_short":
                s.min_extension_adr_multiple = st.slider("Min extension (x ADR)", 0.5, 10.0, float(s.min_extension_adr_multiple), 0.1, key="d_ps_ext")
                s.min_run_up_pct = st.slider("Min run-up %", 10.0, 300.0, float(s.min_run_up_pct), 5.0, key="d_ps_runup")
                s.consecutive_up_days_min = st.slider("Min consecutive up days", 1, 10, int(s.consecutive_up_days_min), key="d_ps_updays")
            elif setup_choice == "cup_with_handle":
                s.min_cup_depth_pct = st.slider("Min cup depth %", 5.0, 50.0, float(s.min_cup_depth_pct), 1.0, key="d_cwh_mind")
                s.max_cup_depth_pct = st.slider("Max cup depth %", 10.0, 60.0, float(s.max_cup_depth_pct), 1.0, key="d_cwh_maxd")
                s.max_handle_depth_pct = st.slider("Max handle depth %", 3.0, 30.0, float(s.max_handle_depth_pct), 1.0, key="d_cwh_hd")
                s.min_recovery_pct = st.slider("Min recovery to old high %", 50.0, 100.0, float(s.min_recovery_pct), 1.0, key="d_cwh_rec")
                s.min_prior_uptrend_pct = st.slider("Min prior uptrend %", 0.0, 100.0, float(s.min_prior_uptrend_pct), 5.0, key="d_cwh_prior")
                s.handle_upper_half_only = st.checkbox("Handle must stay in upper half of cup", value=s.handle_upper_half_only, key="d_cwh_upperhalf")
                s.breakout_volume_ratio_min = st.slider("Min breakout volume ratio", 0.5, 6.0, float(s.breakout_volume_ratio_min), 0.1, key="d_cwh_vol")
            elif setup_choice == "double_bottom":
                s.min_depth_pct = st.slider("Min depth %", 3.0, 40.0, float(s.min_depth_pct), 1.0, key="d_db_mind")
                s.max_depth_pct = st.slider("Max depth %", 10.0, 60.0, float(s.max_depth_pct), 1.0, key="d_db_maxd")
                s.max_low_difference_pct = st.slider("Max difference between the two lows %", 1.0, 25.0, float(s.max_low_difference_pct), 1.0, key="d_db_lowdiff")
                s.breakout_volume_ratio_min = st.slider("Min breakout volume ratio", 0.5, 6.0, float(s.breakout_volume_ratio_min), 0.1, key="d_db_vol")
            elif setup_choice == "flat_base":
                s.max_range_pct = st.slider("Max base range %", 5.0, 30.0, float(s.max_range_pct), 1.0, key="d_fb_range")
                s.min_prior_move_pct = st.slider("Min prior move %", 0.0, 100.0, float(s.min_prior_move_pct), 5.0, key="d_fb_prior")
                s.breakout_volume_ratio_min = st.slider("Min breakout volume ratio", 0.5, 6.0, float(s.breakout_volume_ratio_min), 0.1, key="d_fb_vol")
            elif setup_choice == "ascending_base":
                s.min_segment_depth_pct = st.slider("Min per-pullback depth %", 1.0, 20.0, float(s.min_segment_depth_pct), 1.0, key="d_ab_mind")
                s.max_segment_depth_pct = st.slider("Max per-pullback depth %", 10.0, 40.0, float(s.max_segment_depth_pct), 1.0, key="d_ab_maxd")
                s.breakout_volume_ratio_min = st.slider("Min breakout volume ratio", 0.5, 6.0, float(s.breakout_volume_ratio_min), 0.1, key="d_ab_vol")
            else:  # high_tight_flag
                s.min_run_up_pct = st.slider("Min run-up %", 50.0, 300.0, float(s.min_run_up_pct), 5.0, key="d_htf_runup")
                s.max_flag_depth_pct = st.slider("Max flag depth %", 5.0, 40.0, float(s.max_flag_depth_pct), 1.0, key="d_htf_flag")
                s.breakout_volume_ratio_min = st.slider("Min breakout volume ratio", 0.5, 6.0, float(s.breakout_volume_ratio_min), 0.1, key="d_htf_vol")

            with st.expander(f"What does {spec['label']} mean?"):
                st.write(PATTERN_EXPLANATIONS.get(setup_choice, "No explanation available."))

            st.markdown("**Backtest mechanics**")
            bt = settings.backtest
            bt.risk_pct_per_trade = st.slider("Risk % per trade", 0.1, 5.0, float(bt.risk_pct_per_trade), 0.1, key="d_bt_risk")
            bt.stop_adr_multiple = st.slider("Stop distance (x ADR)", 0.25, 3.0, float(bt.stop_adr_multiple), 0.25, key="d_bt_stop")
            bt.partial_profit_r_multiple = st.slider("Partial profit target (R)", 0.5, 5.0, float(bt.partial_profit_r_multiple), 0.5, key="d_bt_ptarget")
            bt.partial_profit_fraction = st.slider("Partial profit fraction", 0.0, 1.0, float(bt.partial_profit_fraction), 0.05, key="d_bt_pfrac")
            bt.trail_ema_period = st.slider("Trailing EMA period", 5, 50, int(bt.trail_ema_period), key="d_bt_trail")
            bt.max_positions = st.slider("Max concurrent positions", 1, 30, int(bt.max_positions), key="d_bt_maxpos")

        with col_results:
            live_results = run_scan(settings, history, as_of=str(as_of), setup_names=[setup_choice])
            st.metric("Current matches", len(live_results))
            if not live_results.empty:
                st.dataframe(live_results, use_container_width=True, height=200)

            st.markdown("**Live backtest (full loaded history)**")
            signals = scan_signals_over_history(settings, history, setup_choice)
            result = run_backtest(settings.backtest, signals, side=spec["side"])
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

        prefilter_momentum = st.checkbox(
            "Pre-filter to momentum leaders before scanning for patterns",
            value=True,
            help="Only run pattern detection on the union of the top movers above, instead of the whole "
            "loaded universe -- 'find the biggest movers, then see which ones are forming a pattern'.",
            key="scanner_prefilter_momentum",
        )

        if st.button("Run Scan"):
            scan_history = history
            if prefilter_momentum:
                shortlist = momentum_shortlist(
                    history, settings.universe.momentum_timeframes_days, settings.universe.momentum_top_pct
                )
                scan_history = {s: history[s] for s in shortlist if s in history}
                st.session_state.scan_shortlist_size = len(scan_history)
            st.session_state.last_scan = run_scan(settings, scan_history, as_of=str(as_of))
        if prefilter_momentum and st.session_state.get("scan_shortlist_size") is not None:
            st.caption(f"Scanned {st.session_state.scan_shortlist_size} momentum-leading symbol(s), not the full universe.")
        results = st.session_state.last_scan
        if results.empty:
            st.warning("No matches yet -- click 'Run Scan', or loosen thresholds in the Designer tab.")
        else:
            st.caption("Sorted by RS rating (momentum/leadership) first, then setup score.")
            st.dataframe(results, use_container_width=True, height=350)
            pick = st.selectbox("Chart a match", results["symbol"].tolist())
            if pick and pick in history:
                row = results[results["symbol"] == pick].iloc[0]
                st.plotly_chart(plot_symbol(history[pick], pick, marker_date=row["date"]), use_container_width=True)

            matched_setups = [k for k, v in SETUP_REGISTRY.items() if v["label"] in results["setup"].unique()]
            if matched_setups:
                st.markdown("**What these matches mean:**")
                for setup_key in matched_setups:
                    with st.expander(SETUP_REGISTRY[setup_key]["label"]):
                        st.write(PATTERN_EXPLANATIONS.get(setup_key, "No explanation available."))

        with st.expander("📖 Pattern library (all setups, whether matched or not)"):
            for setup_key, spec in SETUP_REGISTRY.items():
                st.markdown(f"**{spec['label']}**")
                st.write(PATTERN_EXPLANATIONS.get(setup_key, "No explanation available."))


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
            result = run_backtest(settings.backtest, signals, side=SETUP_REGISTRY[bt_setup]["side"])
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
                result = run_backtest(settings.backtest, signals, side=spec["side"])
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
            st.plotly_chart(plot_symbol(history[pf_symbol], pf_symbol, marker_date=marker_date), use_container_width=True)

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
# Tab: Settings (preset management + raw view)
# --------------------------------------------------------------------------
with tab_settings:
    st.subheader("Current settings")
    st.json(settings.to_dict())

    st.divider()
    st.subheader("Compare presets")
    all_presets = presets.list_presets()
    to_compare = st.multiselect("Presets to compare", all_presets)
    if to_compare:
        st.dataframe(presets.compare_presets(to_compare), use_container_width=True, height=500)

    st.divider()
    if st.button("Reset to defaults"):
        st.session_state.settings = presets.load_default()
        st.rerun()
