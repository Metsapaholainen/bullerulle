"""Streamlit app: Scanner / Designer / Backtest & Optimizer / Parameter Finder / Settings.

Data loading (network-bound) is an explicit sidebar action; every tab below
operates on the already-cached `st.session_state["history"]` and recomputes
indicators/setups/backtests in pandas, which is fast enough to feel
interactive when you tweak a setting.
"""
from __future__ import annotations

import traceback
from dataclasses import asdict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import presets
from backtest.engine import resolve_entry_and_stop, run_backtest
from backtest.optimizer import run_grid_search, set_nested
from backtest.stats import compute_stats, stats_summary_text
from backtest.target_fit import find_parameters_for_target
from data.cache import apply_live_quotes, get_history, get_history_bulk
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
from data.fundamentals_cache import get_earnings_dates, get_fundamentals, get_fundamentals_bulk
from fundamentals_classifier import classify_lynch, days_to_earnings, filter_earnings_avoidance, LYNCH_CATEGORIES
from indicators import add_adr_pct, add_volume_stats, build_close_panel, compute_rs_rating_panel
from stop_calculator import compute_stop_and_size
from legout_scans import LEGOUT_SCANS, SKIPPED_SCANS, run_legout_scans, scan_legout_signals_over_history
from notifications import send_ntfy_alert, send_sell_alerts
from paper_trading import (
    freeze_closed_trades as pt_freeze_closed_trades,
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
from watchlist import load_watchlist as load_general_watchlist, save_watchlist as save_general_watchlist

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
if "watchlist" not in st.session_state:
    # General-purpose "flag this to look at later" list -- distinct from the
    # Sell Alerts watchlist (which carries a moving-average rule). Loaded
    # once here so the "+ Watchlist" button next to any chart and the
    # dedicated Watchlist tab always see the same in-session list.
    st.session_state.watchlist = load_general_watchlist()
if "paper_trades" not in st.session_state:
    # Loaded once here (rather than only inside the Paper Trading tab body)
    # so the "+ Paper Trading" button next to any chart can append to it
    # regardless of which tab happens to run first in a given script pass.
    st.session_state.paper_trades = pt_load_trades()
if "paper_rules" not in st.session_state:
    st.session_state.paper_rules = pt_load_rules()

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


def _hash_settings_for_cache(s: Settings) -> str:
    """Settings is a plain (non-frozen) dataclass, so it isn't natively
    hashable -- this gives st.cache_data a stable, content-based key so the
    plot cache correctly invalidates only when settings actually change."""
    return str(s.to_dict())


@st.cache_data(show_spinner=False, hash_funcs={Settings: _hash_settings_for_cache})
def _cached_load_universe(settings: Settings, as_of: str, symbols: Optional[tuple], min_history_days: int) -> dict:
    """The expensive step ("Load / Refresh Data") wrapped in Streamlit's
    process-level cache -- unlike st.session_state (which resets on every
    browser refresh/new session), @st.cache_data's cache lives as long as
    this server process keeps running, so a page refresh followed by the
    exact same load (same settings/symbols/as-of date) returns instantly
    instead of re-running the full universe screen + per-symbol history
    fetch (which, for the full ~1500-symbol universe, is the ~30-minute
    cost users were hitting on every refresh). Call `.clear()` on this
    function first to force a genuine re-fetch (the sidebar's "Force fresh
    reload" checkbox does this).

    The progress bar is created AND updated entirely inside this function
    (not passed in from the caller) -- @st.cache_data replays whatever
    st.* element calls happened during a cache-missed run when a later
    call hits the cache, and that replay only works for elements/blocks
    created inside the cached function itself. An earlier version created
    the progress bar in the sidebar and mutated it via a callback passed
    in here, which crashed on the very next cache hit ("a streamlit
    element is called on some layout block created outside the
    function")."""
    progress = st.progress(0.0, text="Loading...")

    def on_progress(i, n, symbol):
        progress.progress(i / n, text=f"Loading {symbol} ({i}/{n})")

    try:
        client = get_client()
        history = load_scan_universe_history(
            settings, as_of=as_of, client=client,
            symbols=list(symbols) if symbols else None,
            min_history_days=min_history_days,
            on_progress=on_progress,
        )
        start_for_benchmark = (pd.Timestamp(as_of) - pd.Timedelta(days=420)).strftime("%Y-%m-%d")
        ensure_benchmark_loaded(history, client, start_for_benchmark, as_of)
        ensure_benchmark_loaded(history, client, start_for_benchmark, as_of, benchmark_symbol=SECONDARY_BENCHMARK_SYMBOL)
    finally:
        progress.empty()
    return {
        "history": history,
        "market_direction": market_direction(history.get(DEFAULT_BENCHMARK_SYMBOL)),
        "market_direction_secondary": market_direction(history.get(SECONDARY_BENCHMARK_SYMBOL)),
    }


def sync_paper_trade_prices() -> tuple:
    """Refreshes just the symbols with a logged Paper Trade -- a handful of
    tickers, not the whole loaded universe -- so a logged trade's current
    price stays live without needing a full "Load / Refresh Data" (which
    reloads everything) or "Force fresh reload" (which re-fetches the whole
    universe). Two steps: (1) any symbol with a logged trade that isn't in
    `history` at all yet gets its full daily history fetched from scratch
    (mirrors Live Mode's per-symbol get_history, but self-serves instead of
    just warning "load it from the sidebar first"); (2) every one of those
    symbols gets a lightweight /quote merged in as today's bar (same
    mechanism Live Mode/Sell Alerts already use for a live current price).

    Returns (synced_symbols, failed_symbols). Clears the paper-trading
    fingerprint so the next render recomputes the simulation against the
    refreshed prices, even if today's date was already the last bar (a plain
    date-based fingerprint wouldn't otherwise notice the price changed)."""
    symbols = sorted({t["symbol"] for t in st.session_state.paper_trades})
    if not symbols:
        return [], []

    client = get_client()
    missing = [s for s in symbols if s not in st.session_state.history or st.session_state.history[s].empty]
    failed = []
    if missing:
        start = (pd.Timestamp(as_of) - pd.Timedelta(days=int(history_years * 365.25))).strftime("%Y-%m-%d")
        end = str(as_of)
        fetched = get_history_bulk(client, missing, start, end)
        st.session_state.history.update(fetched)
        failed.extend(s for s in missing if s not in fetched)

    quote_symbols = [s for s in symbols if s in st.session_state.history]
    try:
        quotes = client.get_quotes(quote_symbols)
        st.session_state.history = apply_live_quotes(st.session_state.history, quotes)
    except Exception:
        failed.extend(s for s in quote_symbols if s not in failed)

    st.session_state.pop("paper_fingerprint", None)
    synced = [s for s in symbols if s not in failed]
    return synced, failed


def render_symbol_status_badge(symbol: str) -> None:
    """Shows whether `symbol` is already tracked somewhere in this app --
    already on the Watchlist, already logged in Paper Trading, both, or
    neither ("new find") -- so a fresh scan match is visually distinct from
    something you're already watching. Called from render_add_to_watchlist_button,
    which sits next to every chart in the app, so this shows up everywhere
    those buttons already do without needing a separate call at each site."""
    on_watchlist = symbol in st.session_state.get("watchlist", [])
    in_paper_trading = any(t["symbol"] == symbol for t in st.session_state.get("paper_trades", []))
    if on_watchlist and in_paper_trading:
        st.caption("📋 On Watchlist  ·  📝 In Paper Trading")
    elif on_watchlist:
        st.caption("📋 On Watchlist")
    elif in_paper_trading:
        st.caption("📝 In Paper Trading")
    else:
        st.caption("✨ New find")


def render_add_to_watchlist_button(symbol: str, key_suffix: str) -> None:
    """A small "+ Watchlist" button dropped next to any chart -- flags the
    currently-viewed symbol for the dedicated Watchlist tab. `key_suffix`
    just needs to be unique per call site (e.g. the caller's own chart key)
    so multiple chart sections on the same page don't collide on widget key."""
    render_symbol_status_badge(symbol)
    if st.button("+ Watchlist", key=f"add_watchlist_{key_suffix}"):
        if symbol in st.session_state.watchlist:
            st.info(f"{symbol} is already on your watchlist.")
        else:
            st.session_state.watchlist.append(symbol)
            save_general_watchlist(st.session_state.watchlist)
            st.success(f"Added {symbol} to your watchlist.")


def render_add_to_paper_trading_button(symbol: str, decision_date, key_suffix: str, setup_tag: str = None) -> None:
    """A small "+ Paper Trading" button dropped next to any chart -- logs a
    paper trade for the currently-viewed symbol, defaulting the decision
    date to whatever this chart's own signal date is (or the most recent
    loaded bar if there isn't one, i.e. "I'm looking at this right now").
    Deduped the same way the Paper Trading tab's own form is (by symbol +
    decision date) so clicking twice doesn't log a duplicate. `key_suffix`
    just needs to be unique per call site."""
    if st.button("+ Paper Trading", key=f"add_paper_{key_suffix}"):
        if decision_date is None:
            st.warning(f"{symbol} isn't loaded -- can't log a decision date.")
            return
        entry_date_str = pd.Timestamp(decision_date).strftime("%Y-%m-%d")
        already_logged = any(
            t["symbol"] == symbol and t["decision_date"] == entry_date_str for t in st.session_state.paper_trades
        )
        if already_logged:
            st.info(f"{symbol} on {entry_date_str} is already logged in Paper Trading.")
        else:
            st.session_state.paper_trades.append(
                {"symbol": symbol, "decision_date": entry_date_str, "setup_tag": setup_tag or "", "notes": ""}
            )
            pt_save_trades(st.session_state.paper_trades)
            st.success(f"Logged {symbol} ({entry_date_str}) to Paper Trading.")


def render_add_to_sell_alerts_button(symbol: str, key_suffix: str, ma_period: int = 10) -> None:
    """A small "+ Sell Alert" button -- adds the currently-viewed symbol to
    the Sell Alerts watchlist (close-below-`ma_period`-day-MA rule), deduped
    by symbol against whatever's already there. `key_suffix` just needs to
    be unique per call site."""
    if st.button("+ Sell Alert", key=f"add_sellalert_{key_suffix}"):
        existing_symbols = {e["symbol"] for e in st.session_state.sell_watchlist}
        if symbol in existing_symbols:
            st.info(f"{symbol} is already on your Sell Alerts watchlist.")
        else:
            st.session_state.sell_watchlist.append({"symbol": symbol, "ma_period": ma_period})
            save_watchlist(st.session_state.sell_watchlist)
            st.success(f"Added {symbol} to Sell Alerts ({ma_period}-day MA).")


def _history_fingerprint(history: dict) -> tuple:
    return tuple(sorted((sym, str(df.index[-1]), len(df)) for sym, df in history.items() if not df.empty))


@st.cache_data(show_spinner=False)
def _cached_rs_rating_panel(_history: dict, history_fp: tuple, lookback_periods: tuple, period_weights: tuple) -> pd.DataFrame:
    """Cross-sectional RS rating (1-99 percentile rank of weighted multi-
    period return) computed across whatever's currently loaded -- the same
    build_close_panel/compute_rs_rating_panel machinery run_scan() already
    uses internally, exposed here so any chart (not just a scan-matched
    row) can show a symbol's RS rating. `_history` (leading underscore) is
    excluded from the cache key since a dict of DataFrames isn't hashable;
    `history_fp` is a plain hashable summary standing in for it, so the
    cache still correctly invalidates when the loaded data changes.

    NOTE: this is relative to whatever's currently loaded, not literally
    "the whole market" -- with only a handful of symbols loaded, the rank
    is a coarse approximation, not a true IBD-style rating."""
    close_panel = build_close_panel(_history)
    return compute_rs_rating_panel(close_panel, list(lookback_periods), list(period_weights))


def get_symbol_rs_rating(symbol: str, history: dict, at_date=None) -> Optional[float]:
    """RS rating for `symbol` as of `at_date` (or the most recent available
    date if omitted/not found in the index), or None if the symbol isn't
    loaded. Cheap on top of the cached panel above -- just a column lookup."""
    if symbol not in history or history[symbol].empty:
        return None
    rs = st.session_state.settings.rs_rating
    panel = _cached_rs_rating_panel(
        history, _history_fingerprint(history), tuple(rs.lookback_periods), tuple(rs.period_weights)
    )
    if symbol not in panel.columns:
        return None
    col = panel[symbol].dropna()
    if col.empty:
        return None
    if at_date is not None:
        col = col[col.index <= pd.Timestamp(at_date)]
        if col.empty:
            return None
    return float(col.iloc[-1])


def _lightweight_fundamentals(symbol: str) -> dict:
    """Cached market_cap/company_name/sector/revenue_growth_pct/eps_growth_pct/
    rs_rating lookup for chart call sites with no scan-result row already
    carrying these fields (Browse all charts' no-match branch, Paper
    Trading). A network hiccup here should never block the chart itself
    from rendering, so any failure just yields an empty dict (plot_symbol
    treats every one of these kwargs as optional)."""
    out = {}
    try:
        fd = get_fundamentals(FMPClient(), symbol)
        out = {
            "market_cap": fd.get("market_cap"), "company_name": fd.get("company_name"),
            "sector": fd.get("sector"),
            "revenue_growth_pct": fd.get("revenue_growth_pct"), "eps_growth_pct": fd.get("eps_growth_pct"),
        }
    except Exception:
        pass
    try:
        out["rs_rating"] = get_symbol_rs_rating(symbol, history)
    except Exception:
        pass
    return out


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

    force_fresh_load = st.checkbox(
        "Force fresh reload (ignore cache)", value=False, key="force_fresh_load",
        help="Clears this app session's in-memory memo of the load, so the exact same "
        "symbols/settings/date get re-processed instead of returning instantly. Note it does NOT re-download "
        "price bars that are already in the local parquet cache and still cover the requested range -- those "
        "always come from disk. That's deliberate (it's what makes a cancelled load resume quickly), but it "
        "means this won't pull down a fresh copy of data you already have.",
    )

    load_col, cancel_col = st.columns([3, 1])
    with load_col:
        do_load = st.button("Load / Refresh Data", type="primary", key="btn_load_data")
    with cancel_col:
        # Rendered unconditionally and BEFORE the blocking load below, so it
        # exists in the DOM for the whole time the load runs. Clicking it
        # sends a rerun request; Streamlit interrupts the in-flight script at
        # its next progress-bar update (every element call checks for pending
        # execution-control requests). The exception it raises derives from
        # BaseException, so neither the `except Exception` here nor the
        # per-symbol one in data/cache.py swallows it, and @st.cache_data
        # only writes its entry on a clean return -- so a cancelled load
        # leaves no partial cache entry and no held lock.
        cancel_load = st.button(
            "Cancel", key="btn_cancel_load",
            help="Abandon a load that's in progress. Everything already fetched stays in the local "
            "cache on disk, so starting again picks up from where it stopped.",
        )

    if cancel_load:
        st.session_state["load_cancelled"] = True

    if do_load and not cancel_load:
        symbols = [s.strip().upper() for s in custom_symbols.split(",") if s.strip()] or None
        st.session_state.pop("load_cancelled", None)
        try:
            if force_fresh_load:
                _cached_load_universe.clear()
            loaded = _cached_load_universe(
                st.session_state.settings,
                str(as_of),
                tuple(symbols) if symbols else None,
                int(history_years * 365.25),
            )
            st.session_state.history = loaded["history"]
            st.session_state.market_direction = loaded["market_direction"]
            st.session_state.market_direction_secondary = loaded["market_direction_secondary"]
            st.success(f"Loaded {len(st.session_state.history)} symbols.")
            # Charts themselves aren't warmed here -- plot_symbol() isn't
            # defined yet at this point in the script (this sidebar section
            # runs before the "Chart helper" section further down), so a
            # flag is set instead and the actual warm-up runs once
            # plot_symbol exists, right before the tab bodies below.
            st.session_state._warm_charts_pending = True
        except Exception as exc:
            st.error(f"Data load failed: {exc}")
            st.code(traceback.format_exc())

    if st.session_state.pop("load_cancelled", False):
        st.warning(
            "Load cancelled. Nothing already in memory was touched, and every symbol fetched before you "
            "cancelled is saved to the local cache -- click Load / Refresh Data again and it'll skip "
            "straight past those and resume."
        )

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
# Also read here (not just inside tab_scanner's body) so it's always defined
# for other tabs -- tab_scanner only reassigns this from st.session_state.last_scan
# inside its own "history is loaded" branch, so anything relying on `results`
# from elsewhere (e.g. the Watchlist tab) could see a NameError if `history`
# started this run empty but got populated later in the same pass (e.g. the
# Paper Trading tab's sync button fetching a symbol that's also watchlisted,
# further down the script, executing after tab_scanner already decided to
# skip its own results assignment).
results = st.session_state.last_scan

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

(
    tab_scanner, tab_orders, tab_designer, tab_backtest, tab_finder, tab_live,
    tab_sell, tab_paper, tab_watchlist, tab_stopcalc, tab_settings,
) = st.tabs(
    ["Scanner", "📋 Tomorrow's Orders", "Designer", "Backtest & Optimizer", "Parameter Finder",
     "Live Mode", "Sell Alerts", "Paper Trading", "Watchlist", "Stop Calculator", "Settings"]
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


def slice_around_date(df: pd.DataFrame, period_label: str, center_date) -> pd.DataFrame:
    """Like slice_for_period, but windows AROUND `center_date` instead of
    the series' tail. slice_for_period is right for a signal that's always
    near the most recent bar (every automated setup, Legout scans), but
    Paper Trading logs decision dates that can be anywhere in the loaded
    history -- "last N days from today" would often miss the trade's window
    entirely for an older entry. Splits the period's day-count roughly
    1/3 before the date, 2/3 after (so the runup into the decision and the
    trade's aftermath both get reasonable room)."""
    days = CHART_PERIODS.get(period_label)
    if not days or len(df) <= days or center_date not in df.index:
        return df
    loc = df.index.get_loc(center_date)
    before, after = int(days * 0.33), int(days * 0.67)
    start = max(loc - before, 0)
    end = min(loc + after, len(df.index))
    return df.iloc[start:end]


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


VOLUME_MA_PERIOD = 20


def _with_volume_ma(full_df: pd.DataFrame) -> pd.DataFrame:
    """Attach the volume EMA BEFORE the frame gets sliced to a chart period.

    Computing it inside plot_symbol would run it over the visible window
    only, so the first 20 bars of a 3-month chart would show a ramping
    average that isn't the stock's real 20-day norm -- and the whole point
    of the line is to judge whether today's volume is unusual."""
    if "volume" not in full_df.columns:
        return full_df
    out = full_df.copy()
    out["vol_ema"] = out["volume"].ewm(span=VOLUME_MA_PERIOD, adjust=False).mean()
    return out


def resolve_chart_df(full_df: pd.DataFrame, period_label: str, row=None, setup_key: str = None, settings: Settings = None) -> pd.DataFrame:
    """Shared by every 'Chart period' picker: resolves the Auto option
    against a specific match's row/setup, otherwise defers to the fixed
    CHART_PERIODS buckets."""
    full_df = _with_volume_ma(full_df)
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
    "momentum_burst": [("consolidation_days", "Quiet base", "rgba(100,149,237,0.18)")],
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


def _format_market_cap(value) -> str:
    value = float(value)
    if value >= 1e12:
        return f"${value / 1e12:.2f}T"
    if value >= 1e9:
        return f"${value / 1e9:.2f}B"
    if value >= 1e6:
        return f"${value / 1e6:.1f}M"
    return f"${value:,.0f}"


@st.cache_data(show_spinner=False)
def plot_eps_log_chart(eps_history: tuple, symbol: str):
    """Quarterly EPS on a log y-axis -- "one inch anywhere on the scale
    represents the same percentage change," so a steepening bar-to-bar climb
    reads as accelerating growth and a flattening/declining climb reads as
    decelerating growth, at a glance, the same way O'Neil describes reading
    a log-scale price chart. `eps_history` is a tuple of (date, eps) pairs,
    most-recent-first (tuple, not list/dict, so this cached function has a
    hashable argument).

    Log scale can't represent a zero/negative EPS quarter -- those bars are
    dropped from the chart (noted in a caption by the caller), not silently
    clamped to some arbitrary small positive number that would misrepresent
    the actual loss."""
    quarters = [(d, e) for d, e in reversed(eps_history) if e is not None and e > 0]
    fig = go.Figure()
    if quarters:
        dates = [q[0] for q in quarters]
        values = [q[1] for q in quarters]
        fig.add_trace(go.Bar(x=dates, y=values, name="EPS", marker_color="#4C78A8"))
    fig.update_layout(
        yaxis_type="log", height=220, margin=dict(l=40, r=20, t=30, b=30),
        title=f"{symbol} quarterly EPS (log scale)", showlegend=False,
    )
    return fig


@st.cache_data(show_spinner=False, hash_funcs={Settings: _hash_settings_for_cache})
def plot_symbol(
    df: pd.DataFrame, symbol: str, marker_date=None, setup_key=None, settings=None, row=None, side="long",
    entry_date=None, entry_price=None, stop_price=None, target_price=None,
    partial_date=None, partial_price=None,
    exit_date=None, exit_price=None, exit_reason=None, r_multiple=None,
    market_cap=None, company_name=None, revenue_growth_pct=None, eps_growth_pct=None, sector=None,
    rs_rating=None,
):
    # Cached: this is a pure function of its arguments (no st.* calls, no
    # hidden global state besides static module-level constants), and it's
    # rebuilt on every single page rerun otherwise -- up to 5 of these plus
    # 8 static pattern diagrams fire on ANY widget interaction anywhere in
    # the app, since Streamlit reruns the whole script every time. Caching
    # means switching a chart period/symbol elsewhere doesn't pay the cost
    # of rebuilding every *other* unrelated chart on the page.
    # Two panes: price on row 1, volume on row 2. Every add_trace/add_hline/
    # add_hrect/add_vrect below MUST pass row=1, col=1 -- on a subplots figure
    # plotly defaults those to ALL subplots, which would draw the stop/target
    # price lines across the volume pane (pinned to its bottom edge, since
    # volumes are in the millions) and duplicate every annotation.
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.03,
    )
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name=symbol
        ),
        row=1, col=1,
    )

    if "volume" in df.columns and df["volume"].notna().any():
        # Bars tinted by the candle's own direction, so an up-day volume
        # spike (accumulation) reads differently from a down-day spike
        # (distribution) without needing a separate indicator.
        vol_colors = [
            "rgba(38,166,154,0.55)" if c >= o else "rgba(239,83,80,0.55)"
            for c, o in zip(df["close"], df["open"])
        ]
        fig.add_trace(
            go.Bar(
                x=df.index, y=df["volume"], marker_color=vol_colors, name="Volume",
                showlegend=False, hovertemplate="%{y:,.0f}<extra></extra>",
            ),
            row=2, col=1,
        )
        # Volume EMA20 -- precomputed on the FULL frame in _with_volume_ma()
        # so it isn't biased by the visible window. Falls back to computing
        # it here for the few call sites that pass a raw slice.
        vol_ma = df["vol_ema"] if "vol_ema" in df.columns else df["volume"].ewm(
            span=VOLUME_MA_PERIOD, adjust=False
        ).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index, y=vol_ma, line=dict(width=1.4, color="#f0b90b"),
                name=f"Vol EMA{VOLUME_MA_PERIOD}", showlegend=False, hoverinfo="skip",
            ),
            row=2, col=1,
        )

    # ADR% (average daily range, the same trailing-20-day formula used to
    # size stops elsewhere in this app) plus company name/market cap/growth
    # (when the caller has them -- not every call site fetches fundamentals)
    # shown right in the chart title, so it's visible next to every chart
    # without needing separate columns/panels. ADR% is colored green above
    # a "worth trading" threshold -- a high ADR is a good sign for this
    # trading style, not a risk flag, unlike most other volatility metrics.
    chart_title = f"{symbol} -- {company_name}" if company_name else symbol
    if sector:
        chart_title += f" ({sector})"
    if not df.empty:
        chart_adr_pct = compute_adr_pct_at(df, df.index[-1], lookback=20)
        detail_parts = []
        if chart_adr_pct is not None:
            adr_color = "#2ecc71" if chart_adr_pct >= 4.0 else "#cccccc"
            detail_parts.append(f'<span style="color:{adr_color}">ADR%: {chart_adr_pct:.1f}%</span>')
        if market_cap:
            detail_parts.append(f"Mkt Cap: {_format_market_cap(market_cap)}")
        if revenue_growth_pct is not None and pd.notna(revenue_growth_pct):
            detail_parts.append(f"Rev Growth (YoY): {revenue_growth_pct:+.1f}%")
        if eps_growth_pct is not None and pd.notna(eps_growth_pct):
            detail_parts.append(f"EPS Growth (YoY): {eps_growth_pct:+.1f}%")
        if rs_rating is not None and pd.notna(rs_rating):
            rs_color = "#2ecc71" if rs_rating >= 90 else "#cccccc"
            detail_parts.append(f'<span style="color:{rs_color}">RS Rating: {rs_rating:.0f}</span>')
        if detail_parts:
            chart_title += "<br>" + "  |  ".join(detail_parts)
    for period, color in ((10, "orange"), (20, "blue")):
        ema = df["close"].ewm(span=period, adjust=False).mean()
        fig.add_trace(go.Scatter(x=df.index, y=ema, line=dict(width=1, color=color), name=f"EMA{period}"), row=1, col=1)

    # Simple (not exponential) moving averages -- a separate toggleable layer
    # on top of the EMA10/EMA20 pair above. Dashed + a distinct palette so
    # they read as a different family of line at a glance; each is its own
    # named trace, so clicking its legend entry shows/hides just that one --
    # no extra UI needed, this is native Plotly legend behavior.
    for period, color in ((10, "#b0b0b0"), (20, "#00bcd4"), (50, "#9c27b0"), (200, "#f44336")):
        sma = df["close"].rolling(period).mean()
        fig.add_trace(
            go.Scatter(x=df.index, y=sma, line=dict(width=1, color=color, dash="dot"), name=f"SMA{period}"), row=1, col=1)

    if marker_date is not None and marker_date in df.index:
        # Two explicit calls rather than row="all": that variant works, but
        # emits one annotation PER row, leaving a second "signal day" label
        # floating over the volume bars. Row 2 gets the line without a label.
        fig.add_vline(
            x=marker_date, line_dash="dash", line_color="green",
            annotation_text="signal day", row=1, col=1,
        )
        fig.add_vline(x=marker_date, line_dash="dash", line_color="green", row=2, col=1)

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
                    annotation_text=label, annotation_position="top left", row=1, col=1)

        if row is not None:
            for field in RESISTANCE_FIELDS:
                value = row.get(field) if hasattr(row, "get") else None
                if value is not None and pd.notna(value):
                    fig.add_hline(
                        y=value, line_dash="dot", line_color="yellow",
                        annotation_text=f"{field} = {value:.2f}", annotation_position="right", row=1, col=1)
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
            ), row=1, col=1)

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
                    ), row=1, col=1)
                fig.add_hline(
                    y=stop_price, line_dash="dash", line_color="red",
                    annotation_text=f"Stop {stop_price:.2f}", annotation_position="bottom right", row=1, col=1)
                fig.add_hline(
                    y=target_price, line_dash="dash", line_color="mediumseagreen",
                    annotation_text=f"Target {target_price:.2f} ({settings.backtest.partial_profit_r_multiple:.1f}R)",
                    annotation_position="top right", row=1, col=1)
                lo, hi = sorted([stop_price, entry_price])
                fig.add_hrect(y0=lo, y1=hi, fillcolor="rgba(255,0,0,0.06)", line_width=0, row=1, col=1)
                lo, hi = sorted([entry_price, target_price])
                fig.add_hrect(y0=lo, y1=hi, fillcolor="rgba(60,179,113,0.08)", line_width=0, row=1, col=1)

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
                ), row=1, col=1)
            if pd.notna(stop_price):
                fig.add_hline(
                    y=stop_price, line_dash="dash", line_color="red",
                    annotation_text=f"Stop {stop_price:.2f}", annotation_position="bottom right", row=1, col=1)
            if pd.notna(target_price):
                fig.add_hline(
                    y=target_price, line_dash="dash", line_color="mediumseagreen",
                    annotation_text=f"Target {target_price:.2f}", annotation_position="top right", row=1, col=1)
            if pd.notna(partial_date) and pd.notna(partial_price):
                fig.add_trace(
                    go.Scatter(
                        x=[partial_date], y=[partial_price], mode="markers+text",
                        marker=dict(symbol="diamond", size=13, color="orange", line=dict(width=1, color="black")),
                        text=["Sold half"], textposition="bottom center", textfont=dict(color="orange"),
                        name="Partial exit", showlegend=False,
                    ), row=1, col=1)
            if pd.notna(exit_date) and pd.notna(exit_price):
                r_val = r_multiple if pd.notna(r_multiple) else 0
                exit_color = "lime" if r_val > 0 else ("red" if r_val < 0 else "gray")
                fig.add_trace(
                    go.Scatter(
                        x=[exit_date], y=[exit_price], mode="markers+text",
                        marker=dict(symbol="x", size=14, color=exit_color, line=dict(width=1, color="black")),
                        text=[exit_reason or "exit"], textposition="top center", textfont=dict(color=exit_color),
                        name="Exit", showlegend=False,
                    ), row=1, col=1)

    fig.update_layout(
        title=dict(text=chart_title, font=dict(size=13)),
        # 450 -> 560 so the price pane keeps roughly the height it had and
        # the volume pane is genuinely additive rather than a squeeze.
        height=560,
        xaxis_rangeslider_visible=False,
        # Bottom margin up from 10: the date labels now sit under row 2.
        margin=dict(l=10, r=10, t=50, b=28),
        bargap=0,  # volume bars butt together, like a real volume pane
    )
    # shared_xaxes=True already hides row 1's tick labels and links the zoom,
    # so only the volume axis needs styling. "~s" renders 12,400,000 as 12M.
    fig.update_yaxes(title_text="Vol", title_font=dict(size=10), tickformat="~s", row=2, col=1)
    return fig


if st.session_state.pop("_warm_charts_pending", False) and history:
    # Warms plot_symbol()'s cache for every just-loaded symbol's default
    # "Browse all charts" view (no scan match yet at this point -- Run Scan
    # hasn't necessarily happened) right after data loads, rather than only
    # after a scan completes -- so switching stocks in Browse all
    # charts/the Watchlist tab is instant even before running a scan. Runs
    # once per load (the flag is popped so a later rerun doesn't repeat it).
    _warm_symbols = [s for s, df in history.items() if not df.empty]
    if _warm_symbols:
        _warm_progress = st.progress(0.0, text="Pre-building charts for instant browsing...")
        for _wi, _wsym in enumerate(_warm_symbols):
            try:
                plot_symbol(
                    resolve_chart_df(history[_wsym], AUTO_CHART_PERIOD, None, None, settings), _wsym,
                    **_lightweight_fundamentals(_wsym),
                )
            except Exception:
                pass
            _warm_progress.progress(
                (_wi + 1) / len(_warm_symbols), text=f"Pre-building charts... ({_wi + 1}/{len(_warm_symbols)})"
            )
        _warm_progress.empty()
        st.caption(f"Pre-built {len(_warm_symbols)} chart(s) for instant browsing.")


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
            st.markdown("**\"Surf the rising moving averages\"**")
            s.require_rising_ema10 = st.checkbox("Require EMA10 sloping up", value=s.require_rising_ema10, key=f"d_bo_risingema10_{st.session_state.settings_version}")
            s.require_rising_ema20 = st.checkbox("Require EMA20 sloping up", value=s.require_rising_ema20, key=f"d_bo_risingema20_{st.session_state.settings_version}")
            s.ema_slope_lookback_days = st.slider("EMA slope lookback (days)", 2, 20, int(s.ema_slope_lookback_days), key=f"d_bo_slopelb_{st.session_state.settings_version}")
            s.min_ema_slope_pct = st.slider("Min EMA slope % over lookback", -5.0, 20.0, float(s.min_ema_slope_pct), 0.5, key=f"d_bo_slopemin_{st.session_state.settings_version}")
            st.markdown("**\"Orderly pullback with higher lows\"**")
            s.require_higher_lows = st.checkbox("Require higher lows in base", value=s.require_higher_lows, key=f"d_bo_higherlows_{st.session_state.settings_version}")
            s.higher_lows_tolerance_pct = st.slider("Higher-lows tolerance %", 0.0, 15.0, float(s.higher_lows_tolerance_pct), 0.5, key=f"d_bo_hltol_{st.session_state.settings_version}")
            st.markdown("**\"A big move higher\" (not just a wide/choppy range)**")
            s.require_net_prior_advance = st.checkbox("Require genuine net advance", value=s.require_net_prior_advance, key=f"d_bo_netadv_{st.session_state.settings_version}")
            s.min_prior_net_advance_pct = st.slider("Min prior net advance %", 0.0, 100.0, float(s.min_prior_net_advance_pct), 5.0, key=f"d_bo_netadvmin_{st.session_state.settings_version}")
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
            st.markdown("**Earnings beat** (\"a significant beat to analyst expectations\")")
            s.min_eps_beat_pct_for_bonus = st.slider(
                "EPS beat % for bonus tier", 0.0, 100.0, float(s.min_eps_beat_pct_for_bonus), 5.0, key=f"d_ep_beatbonus_{st.session_state.settings_version}"
            )
            s.require_eps_beat = st.checkbox(
                "Require EPS beat as a hard filter (when data available)", value=s.require_eps_beat, key=f"d_ep_requirebeat_{st.session_state.settings_version}"
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
        elif setup_choice == "high_tight_flag":
            s.min_run_up_pct = st.slider("Min run-up %", 50.0, 300.0, float(s.min_run_up_pct), 5.0, key=f"d_htf_runup_{st.session_state.settings_version}")
            s.max_flag_depth_pct = st.slider("Max flag depth %", 5.0, 40.0, float(s.max_flag_depth_pct), 1.0, key=f"d_htf_flag_{st.session_state.settings_version}")
            s.breakout_volume_ratio_min = st.slider("Min breakout volume ratio", 0.5, 6.0, float(s.breakout_volume_ratio_min), 0.1, key=f"d_htf_vol_{st.session_state.settings_version}")
        elif setup_choice == "momentum_burst":
            st.caption("Bonde's core scan -- these three are his, unchanged since 2014.")
            s.min_gain_pct = st.slider("Min gain on the burst day %", 2.0, 12.0, float(s.min_gain_pct), 0.5, key=f"d_mb_gain_{st.session_state.settings_version}")
            s.min_dollar_volume = st.number_input("Min dollar volume", 0, 1_000_000_000, int(s.min_dollar_volume), 500_000, key=f"d_mb_dv_{st.session_state.settings_version}")
            s.min_close_position_pct = st.slider("Min close position in the day's range %", 0.0, 100.0, float(s.min_close_position_pct), 5.0, key=f"d_mb_closepos_{st.session_state.settings_version}", help="100 = closed exactly on the high. Bonde: 'stock should close near high on breakout day'.")
            st.caption("Quality checks -- his 8-point checklist, loosened so they exclude clear violations rather than acting as an AND-chain.")
            s.min_range_expansion_ratio = st.slider("Min range expansion (x the 5-day avg)", 1.0, 3.0, float(s.min_range_expansion_ratio), 0.05, key=f"d_mb_rexp_{st.session_state.settings_version}")
            s.consolidation_days = st.slider("Consolidation length (days)", 3, 30, int(s.consolidation_days), 1, key=f"d_mb_cons_{st.session_state.settings_version}")
            s.max_consolidation_range_pct = st.slider("Max consolidation range %", 5.0, 50.0, float(s.max_consolidation_range_pct), 1.0, key=f"d_mb_range_{st.session_state.settings_version}")
            s.max_base_daily_move_pct = st.slider("Max single-day move inside the base %", 3.0, 25.0, float(s.max_base_daily_move_pct), 1.0, key=f"d_mb_basemove_{st.session_state.settings_version}")
            s.min_efficiency_ratio = st.slider("Min prior-trend linearity (0-1)", 0.0, 0.8, float(s.min_efficiency_ratio), 0.05, key=f"d_mb_eff_{st.session_state.settings_version}", help="Kaufman efficiency ratio over the advance BEFORE the base: net move divided by total distance travelled. 1.0 is a straight line, 0.1 is the same net move via chop.")
            s.require_volume_dryup = st.checkbox("Require volume dry-up in the base", value=bool(s.require_volume_dryup), key=f"d_mb_dryup_{st.session_state.settings_version}")
            s.volume_dryup_ratio_max = st.slider("Max base volume vs the 50-day average", 0.3, 2.0, float(s.volume_dryup_ratio_max), 0.05, key=f"d_mb_dryupr_{st.session_state.settings_version}")
            s.require_quiet_prior_day = st.checkbox("Require a narrow or down day before the burst", value=bool(s.require_quiet_prior_day), key=f"d_mb_quiet_{st.session_state.settings_version}")
            s.max_consecutive_up_days = st.slider("Skip after this many consecutive up days", 1, 8, int(s.max_consecutive_up_days), 1, key=f"d_mb_consec_{st.session_state.settings_version}", help="'Avoid day 2/3 entries' -- by then the burst is underway and your stop sits too far below to be worth taking.")
            s.min_adr_pct = st.slider("Min ADR %", 0.5, 12.0, float(s.min_adr_pct), 0.5, key=f"d_mb_adr_{st.session_state.settings_version}")
            s.min_rs_rating = st.slider("Min RS rating", 0, 99, int(s.min_rs_rating), 1, key=f"d_mb_rs_{st.session_state.settings_version}")
        else:
            # Reached only if a setup is registered without its own branch
            # above. Previously this was `else: # high_tight_flag`, which
            # silently rendered HTF's sliders bound to whatever settings
            # object the new setup actually uses -- edits went to fields that
            # didn't exist, or worse, to ones that did and meant something else.
            st.info(
                f"No parameter controls have been written for **{spec['label']}** yet. "
                "It still scans and backtests using its configured defaults; edit them in "
                "config/default_settings.yaml, or add an `elif` branch here."
            )

        with st.expander(f"What does {spec['label']} mean?"):
            st.write(PATTERN_EXPLANATIONS.get(setup_choice, "No explanation available."))

        st.markdown("**Backtest mechanics**")
        bt = settings.backtest
        bt.entry_mode = st.selectbox(
            "Entry execution", ["stop_buy", "next_open"], index=0 if bt.entry_mode == "stop_buy" else 1,
            format_func=lambda v: (
                "Stop-buy above the signal day's high (how you actually trade)" if v == "stop_buy"
                else "Fill at the next open, unconditionally (legacy)"
            ),
            key=f"d_bt_entrymode_{st.session_state.settings_version}",
            help="Stop-buy only fills when price actually trades through the trigger, so signals that opened "
            "and immediately rolled over produce no trade at all -- which is most of the point. Expect a "
            "meaningfully lower trade count than 'next open'; that drop IS the filter working.",
        )
        if bt.entry_mode == "stop_buy":
            bt.entry_buffer_pct = st.slider(
                "Trigger buffer above the high (%)", 0.0, 1.0, float(bt.entry_buffer_pct), 0.01,
                key=f"d_bt_buffer_{st.session_state.settings_version}",
                help="How far beyond the signal day's high the resting order sits, so a single tick touching "
                "the exact high doesn't fill you on a level that was never really taken out.",
            )
            bt.max_gap_fill_pct = st.slider(
                "Stand aside if it gaps more than this % past the trigger (0 = take any gap)", 0.0, 15.0,
                float(bt.max_gap_fill_pct), 0.5, key=f"d_bt_maxgap_{st.session_state.settings_version}",
                help="The nearest daily-bar substitute for his opening-range-high filter: he skips a stock "
                "that gaps up and then fails to clear its early high. A resting order can't see that, but it "
                "can at least refuse a fill far above where the setup justified.",
            )
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
        bt.stop_exceeds_adr_action = st.selectbox(
            "When the signal day's low is further than that cap",
            ["cap", "skip"], index=0 if bt.stop_exceeds_adr_action == "cap" else 1,
            format_func=lambda v: (
                "Tighten the stop to the cap" if v == "cap"
                else "Skip the trade (his actual rule)"
            ),
            key=f"d_bt_stopaction_{st.session_state.settings_version}",
            help="Tightening keeps the trade but moves the stop off the structure, so an ordinary pullback "
            "into the base takes you out. Skipping matches \"the stop should be no more than the ATR\" -- if "
            "it would be wider, the setup isn't takeable at acceptable risk. This binds far more often under "
            "stop-buy entries, because the higher fill widens the distance to the signal day's low.",
        )
        bt.check_same_bar_stop = st.checkbox(
            "Also check whether the entry bar itself hit the stop",
            value=bool(bt.check_same_bar_stop), key=f"d_bt_samebar_{st.session_state.settings_version}",
            help="Off by default, and it will LOWER your win rate. Exits are processed before entries, so a "
            "fill that reverses through its own stop the same session is currently carried to the next day "
            "for free. Under stop-buy you enter near the top of the bar's range, which makes that more "
            "likely -- in synthetic testing it affected about a third of fills. Turning this on is more "
            "honest but pessimistic: a daily bar can't tell you whether the low came before or after your fill.",
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
                help="Per the source article, EP's stop rule is worded identically to Breakout's -- \"the stop "
                "is at the lows of the day\" -- so the default (use Stop placement above, i.e. the signal "
                "day's whole-day low) already matches it. This is an optional alternative to experiment with, "
                "not a correction. Only affects Episodic Pivot backtests.",
            )
            bt.ep_stop_mode_override = None if ep_override_choice == "(use Stop placement above)" else ep_override_choice

    with col_results:
        if not history:
            st.info("Load data from the sidebar first to see live matches and backtest stats.")
        else:
            # Fingerprint-cached: this used to re-run a full scan +
            # scan_signals_over_history + run_backtest across the ENTIRE
            # loaded universe on every single script rerun -- not just while
            # actively tweaking a slider here, but on ANY widget interaction
            # anywhere in the app, since Streamlit reruns the whole script
            # regardless of which tab is visually active. At a handful of
            # test symbols that's invisible; at a real ~1000+ symbol
            # universe it's a 60-90+ second stall on every unrelated click
            # (e.g. just switching charts in the Scanner tab). Only actually
            # recompute when something this result depends on changed.
            designer_history_fp = tuple(
                sorted((sym, str(df.index[-1])) for sym, df in history.items() if not df.empty)
            )
            designer_fingerprint = (
                setup_choice, str(as_of), tuple(asdict(s).items()), tuple(asdict(bt).items()), designer_history_fp,
            )
            if st.session_state.get("designer_fingerprint") != designer_fingerprint:
                st.session_state.designer_live_results = run_scan(
                    settings, history, as_of=str(as_of), setup_names=[setup_choice]
                )
                designer_signals = scan_signals_over_history(settings, history, setup_choice)
                st.session_state.designer_bt_result = run_backtest(
                    settings.backtest, designer_signals, side=spec["side"], setup_name=setup_choice
                )
                st.session_state.designer_fingerprint = designer_fingerprint
            live_results = st.session_state.designer_live_results
            result = st.session_state.designer_bt_result

            st.metric("Current matches", len(live_results))
            if not live_results.empty:
                st.dataframe(live_results, use_container_width=True, height=200)

            st.markdown("**Live backtest (full loaded history)**")
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
            col_pivot_dist, col_pivot_pre = st.columns([2, 1])
            with col_pivot_dist:
                pivot_distance = st.slider(
                    "Max distance from pivot (%)", 1.0, 20.0, 8.0, 1.0, key="pivot_distance_pct",
                    help="How close (in either direction) to the breakout level counts as 'approaching'.",
                )
            with col_pivot_pre:
                pivot_prefilter = st.checkbox(
                    "Momentum leaders only", value=True, key="pivot_prefilter",
                    help="Narrow to the momentum shortlist first, exactly as Run Scan does. This check runs "
                    "every setup at every auto-detected size against every symbol -- roughly 0.2s each -- so "
                    "on a full universe it's the difference between seconds and several minutes. There's also "
                    "little point ranking bases in stocks that aren't going anywhere.",
                )
            if st.button("Find stocks approaching a pivot"):
                pivot_history = history
                if pivot_prefilter:
                    shortlist = momentum_shortlist(
                        history, settings.universe.momentum_timeframes_days, settings.universe.momentum_top_pct
                    )
                    pivot_history = {s: history[s] for s in shortlist if s in history} or history
                pivot_progress = st.progress(0.0, text="Checking bases...")

                def _on_pivot_progress(i, n, sym):
                    pivot_progress.progress(i / n, text=f"Checking bases... {sym} ({i}/{n})")

                try:
                    st.session_state.approaching_df = find_approaching_pivot(
                        settings, pivot_history, as_of=str(as_of), max_distance_pct=pivot_distance,
                        on_progress=_on_pivot_progress,
                    )
                finally:
                    pivot_progress.empty()
                st.caption(f"Checked {len(pivot_history)} symbol(s).")
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
                            **_lightweight_fundamentals(pick_approach),
                        ),
                        use_container_width=True,
                        key="approach_chart",
                    )
                    col_approach_wl, col_approach_pt = st.columns(2)
                    with col_approach_wl:
                        render_add_to_watchlist_button(pick_approach, "approach")
                    with col_approach_pt:
                        render_add_to_paper_trading_button(pick_approach, arow["date"], "approach", setup_tag=arow["setup"])
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

        min_adr_pct = st.number_input(
            "📈 Minimum ADR% (20-day avg daily range)", min_value=0.0, max_value=50.0, value=4.0, step=0.5,
            key="scanner_min_adr_pct",
            help="Drops any match with less tradeable daily range than this, regardless of which setup found "
            "it -- e.g. set to 4.0 to hide anything with less than 4% average daily range. 0 disables the filter.",
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

            new_scan_results = run_scan(
                settings, scan_history, as_of=str(as_of), auto_detect=auto_detect,
                fundamentals=scan_fundamentals,
                earnings_dates=scan_earnings_dates, avoid_earnings_window=avoid_earnings,
                min_adr_pct=float(min_adr_pct),
                avoid_earnings_days=int(avoid_earnings_days),
            )
            st.session_state.last_scan = new_scan_results

            if not new_scan_results.empty:
                # Warm plot_symbol()'s cache for every match's default chart
                # (same args the "Chart a match" picker below uses at its
                # default settings) right now, once, so switching between
                # matches afterwards is an instant cache hit instead of a
                # fresh render each time.
                warm_symbols = [s for s in new_scan_results["symbol"].unique().tolist() if s in history]
                warm_progress = st.progress(0.0, text="Pre-building charts...")
                for wi, wsym in enumerate(warm_symbols):
                    wrow = new_scan_results[new_scan_results["symbol"] == wsym].iloc[0]
                    wsetup_key = LABEL_TO_SETUP_KEY.get(wrow["setup"])
                    try:
                        plot_symbol(
                            resolve_chart_df(history[wsym], AUTO_CHART_PERIOD, wrow, wsetup_key, settings), wsym,
                            marker_date=wrow["date"], setup_key=wsetup_key, settings=settings, row=wrow,
                            side=wrow.get("side", "long"),
                            market_cap=wrow.get("market_cap"), company_name=wrow.get("company_name"),
                            sector=wrow.get("sector"),
                            revenue_growth_pct=wrow.get("revenue_growth_pct"), eps_growth_pct=wrow.get("eps_growth_pct"),
                            rs_rating=wrow.get("rs_rating"),
                        )
                    except Exception:
                        pass
                    warm_progress.progress(
                        (wi + 1) / len(warm_symbols), text=f"Pre-building charts... ({wi + 1}/{len(warm_symbols)})"
                    )
                warm_progress.empty()
        if prefilter_momentum and st.session_state.get("scan_shortlist_size") is not None:
            st.caption(f"Scanned {st.session_state.scan_shortlist_size} momentum-leading symbol(s), not the full universe.")
        results = st.session_state.last_scan
        # Quality gate + cap, applied AFTER the scan. The scan itself is
        # deliberately wide (see the note on BreakoutSettings' defaults);
        # this is where you tighten, because a star score ranks across setups
        # in a way each setup's own threshold never could. The controls are
        # rendered BEFORE the empty check on purpose -- if the filter clears
        # the list, you still need the slider on screen to lower it again.
        if not results.empty and "stars" in results.columns:
            star_counts = results["stars"].value_counts()
            available = "  ".join(f"{int(k)}★: {v}" for k, v in sorted(star_counts.items(), reverse=True))
            col_stars, col_topn = st.columns([2, 1])
            with col_stars:
                min_stars = st.slider(
                    "Minimum quality (stars)", 0, 6, 0, 1, key="scanner_min_stars",
                    help="A 0-6.5 score across seven of Qullamaggie's documented criteria -- prior move, above "
                    "a rising 50-day, RS, base tightness, volume dry-up then expansion, moving-average "
                    "alignment, and linearity -- bucketed into 1-6 stars. The star SCALE is this app's "
                    "encoding of those criteria; he doesn't publish a numeric grade himself.\n\n"
                    "DEFAULTS TO 0 ON PURPOSE. Filtering to 3 stars cuts Breakout's expectancy by roughly half "
                    "even after fixing a real measurement bug and re-tuning the one component (prior move "
                    "size) that showed a genuine, data-verified relationship to becoming a big winner. Even "
                    "filtering on THAT component alone never beat taking every match. Use the stars to decide "
                    "what to look at first (the list "
                    "is sorted by them), not to throw candidates away.",
                )
            with col_topn:
                top_n = st.number_input(
                    "Show top", 1, 200, 10, 1, key="scanner_top_n",
                    help="Cap on how many matches to display, after the star filter. The list is already "
                    "ranked best-first, so this only trims the tail.",
                )
            st.caption(
                f"Matched today, by quality -- {available}. Sorted best-first; stars are a **reading order**, "
                "not a filter (see the slider's help for why raising it measurably hurt Breakout)."
            )
            filtered = results[results["stars"].fillna(0) >= min_stars]
            if filtered.empty:
                st.warning(
                    f"Nothing at {min_stars}★ or better today. Lower the bar above to see what did match -- "
                    "on a quiet day the honest answer may be that there's nothing here worth taking."
                )
            results = filtered.head(int(top_n))

        if results.empty:
            if st.session_state.last_scan.empty:
                st.warning("No matches yet -- click 'Run Scan', or loosen thresholds in the Designer tab.")
        else:
            st.caption(
                "Ranked by star quality first, then RS rating (momentum/leadership), then the setup's own score "
                "-- grouped below by setup so each table only shows columns that actually apply to that pattern "
                "(a Cup with Handle match was never going to have a Double Bottom's columns, and vice versa). "
                "The rank column reflects the overall cross-setup priority."
            )
            results_ranked = results.reset_index(drop=True).copy()
            results_ranked.insert(0, "rank", results_ranked.index + 1)
            # Show setups in the order their best (highest-priority) match
            # appears in the overall ranking, not alphabetically -- the
            # group containing the #1 overall match comes first.
            setup_order = results_ranked.groupby("setup")["rank"].min().sort_values().index.tolist()
            scanner_group_clicks = {}
            for setup_label in setup_order:
                group_df = results_ranked[results_ranked["setup"] == setup_label].dropna(axis=1, how="all")
                st.markdown(f"**{setup_label}** ({len(group_df)} match{'es' if len(group_df) != 1 else ''})")
                group_event = st.dataframe(
                    group_df, use_container_width=True, height=min(80 + 35 * len(group_df), 350),
                    on_select="rerun", selection_mode="single-row", key=f"scanner_group_table_{setup_label}",
                )
                group_clicked = group_event.selection.rows if group_event.selection else []
                if group_clicked:
                    scanner_group_clicks[setup_label] = group_df.iloc[group_clicked[0]]["symbol"]

            # A click on any per-setup table above should drive the same
            # "Chart a match" picker below -- write it into that selectbox's
            # OWN session-state key before the selectbox is instantiated (a
            # keyed widget ignores value=/index= once its key already holds
            # a value, the same rule already established for Legout scans
            # and Paper Trading's row-click-to-chart elsewhere in this app).
            all_match_symbols = results["symbol"].tolist()
            if (
                "scanner_chart_pick" not in st.session_state
                or st.session_state.scanner_chart_pick not in all_match_symbols
            ):
                st.session_state.scanner_chart_pick = all_match_symbols[0]
            if scanner_group_clicks:
                # Only one table can have actually fired a selection event in
                # a given rerun (clicking a row in one table doesn't set
                # another table's selection.rows), so any single value here is
                # the one that was just clicked.
                st.session_state.scanner_chart_pick = next(iter(scanner_group_clicks.values()))

            col_pick, col_period = st.columns([3, 1])
            with col_pick:
                pick = st.selectbox("Chart a match", all_match_symbols, key="scanner_chart_pick")
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
                        market_cap=row.get("market_cap"), company_name=row.get("company_name"),
                        sector=row.get("sector"),
                        revenue_growth_pct=row.get("revenue_growth_pct"), eps_growth_pct=row.get("eps_growth_pct"),
                        rs_rating=row.get("rs_rating"),
                    ),
                    use_container_width=True,
                    key="scanner_match_chart",
                )
                col_scanner_wl, col_scanner_pt = st.columns(2)
                with col_scanner_wl:
                    render_add_to_watchlist_button(pick, "scanner_match")
                with col_scanner_pt:
                    render_add_to_paper_trading_button(pick, row["date"], "scanner_match", setup_tag=row["setup"])
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
                        market_cap=browse_match.get("market_cap"), company_name=browse_match.get("company_name"),
                        sector=browse_match.get("sector"),
                        revenue_growth_pct=browse_match.get("revenue_growth_pct"), eps_growth_pct=browse_match.get("eps_growth_pct"),
                        rs_rating=browse_match.get("rs_rating"),
                    ),
                    use_container_width=True,
                    key="browse_chart",
                )
                col_browse_wl, col_browse_pt = st.columns(2)
                with col_browse_wl:
                    render_add_to_watchlist_button(browse_symbol, "browse")
                with col_browse_pt:
                    render_add_to_paper_trading_button(
                        browse_symbol, browse_match["date"], "browse", setup_tag=browse_match["setup"]
                    )
                st.success(f"This matched **{browse_match['setup']}** in the last scan.")
            else:
                browse_fund = _lightweight_fundamentals(browse_symbol)
                st.plotly_chart(
                    plot_symbol(browse_df, browse_symbol, **browse_fund), use_container_width=True, key="browse_chart"
                )
                col_browse_nm_wl, col_browse_nm_pt = st.columns(2)
                with col_browse_nm_wl:
                    render_add_to_watchlist_button(browse_symbol, "browse_nomatch")
                with col_browse_nm_pt:
                    render_add_to_paper_trading_button(
                        browse_symbol, history[browse_symbol].index[-1], "browse_nomatch"
                    )
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

                if not legout_matches.empty:
                    # Warm plot_symbol()'s cache for every match's default
                    # chart (same args the chart picker below uses at its
                    # default "1 year" period) right now, once, so flipping
                    # between matches afterwards is instant instead of a
                    # fresh render each time.
                    legout_warm_symbols = [s for s in legout_matches["symbol"].unique().tolist() if s in history]
                    legout_warm_progress = st.progress(0.0, text="Pre-building charts...")
                    for lwi, lwsym in enumerate(legout_warm_symbols):
                        lwrow = legout_matches[legout_matches["symbol"] == lwsym].iloc[0]
                        try:
                            plot_symbol(
                                slice_for_period(history[lwsym], "1 year"), lwsym, marker_date=lwrow["date"],
                                **_lightweight_fundamentals(lwsym),
                            )
                        except Exception:
                            pass
                        legout_warm_progress.progress(
                            (lwi + 1) / len(legout_warm_symbols),
                            text=f"Pre-building charts... ({lwi + 1}/{len(legout_warm_symbols)})",
                        )
                    legout_warm_progress.empty()

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
                            **_lightweight_fundamentals(legout_pick),
                        ),
                        use_container_width=True,
                        key="legout_chart",
                    )
                    col_legout_wl, col_legout_pt = st.columns(2)
                    with col_legout_wl:
                        render_add_to_watchlist_button(legout_pick, "legout")
                    with col_legout_pt:
                        render_add_to_paper_trading_button(
                            legout_pick, legout_row["date"], "legout", setup_tag=legout_row["scan"]
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
                                "company_name": fd.get("company_name"),
                                "market_cap": fd.get("market_cap"),
                                "sector": fd.get("sector"),
                                "revenue_growth_pct": fd.get("revenue_growth_pct"),
                                "eps_growth_pct": fd.get("eps_growth_pct"),
                                "float_shares": fd.get("float_shares"),
                                "free_float_pct": fd.get("free_float_pct"),
                                "insider_acq_disp_ratio": fd.get("insider_acquired_disposed_ratio"),
                                "next_earnings_date": fund_earnings.get(sym),
                                "days_to_earnings": days_to_earnings(sym, fund_earnings, str(as_of)),
                                "lynch_category": LYNCH_CATEGORIES[classify_lynch(fd)],
                                "earnings_decelerating": "⚠️ Decelerating" if fd.get("earnings_decelerating") else "",
                            })
                        st.session_state.fundamentals_df = pd.DataFrame(fund_rows)
                        # Raw per-symbol fundamentals (incl. eps_history/eps_growth_history/
                        # earnings_deceleration_reason) kept separately so the log-scale
                        # chart below survives a rerun (e.g. picking a different symbol)
                        # without re-fetching.
                        st.session_state.fundamentals_raw = fund_data
                    except Exception as exc:
                        st.error(f"Fundamentals fetch failed: {exc}")

            fdf = st.session_state.get("fundamentals_df")
            if fdf is None or fdf.empty:
                st.info("Pick a scope above and click 'Fetch fundamentals'.")
            else:
                st.dataframe(fdf, use_container_width=True, height=300)
                fund_pick = st.selectbox("Chart one", fdf["symbol"].tolist(), key="fund_chart_pick")
                if fund_pick in history:
                    fund_row = fdf[fdf["symbol"] == fund_pick].iloc[0]
                    st.plotly_chart(
                        plot_symbol(
                            history[fund_pick], fund_pick,
                            market_cap=fund_row.get("market_cap"), company_name=fund_row.get("company_name"),
                            sector=fund_row.get("sector"),
                            revenue_growth_pct=fund_row.get("revenue_growth_pct"), eps_growth_pct=fund_row.get("eps_growth_pct"),
                            rs_rating=get_symbol_rs_rating(fund_pick, history),
                        ),
                        use_container_width=True, key="fund_chart",
                    )
                    col_fund_wl, col_fund_pt = st.columns(2)
                    with col_fund_wl:
                        render_add_to_watchlist_button(fund_pick, "fund")
                    with col_fund_pt:
                        render_add_to_paper_trading_button(fund_pick, history[fund_pick].index[-1], "fund")

                    fund_raw = st.session_state.get("fundamentals_raw", {}).get(fund_pick, {})
                    eps_history = fund_raw.get("eps_history")
                    if eps_history:
                        if fund_raw.get("earnings_decelerating"):
                            st.warning(f"⚠️ {fund_raw.get('earnings_deceleration_reason')}")
                        eps_pairs = tuple((r.get("date"), r.get("eps")) for r in eps_history)
                        st.plotly_chart(
                            plot_eps_log_chart(eps_pairs, fund_pick), use_container_width=True, key="fund_eps_log_chart",
                        )
                        if any(r.get("eps") is not None and r.get("eps") <= 0 for r in eps_history):
                            st.caption("Quarters with zero/negative EPS are omitted above -- a log scale can't represent them.")

            with st.expander("What do these columns mean?"):
                st.markdown(
                    "- **revenue_growth_pct / eps_growth_pct**: latest quarter vs. the same quarter one year ago "
                    "(YoY) -- computed directly from raw quarterly figures, not FMP's own growth-rate endpoint "
                    "(which is quarter-over-quarter, not YoY). Qullamaggie's own bar for an Episodic Pivot: "
                    "\"triple digit is ideal, mid/high double digits works really well too.\"\n"
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
                    "context this app doesn't fetch.\n"
                    "- **earnings_decelerating**: O'Neil's rule -- flagged when YoY EPS growth has fallen by "
                    "two-thirds or more from its own prior quarter, in each of the last two quarters running "
                    "(e.g. 100% -> 30% -> 9%). \"You may want to avoid that company\" -- shown as a warning only, "
                    "never blocks a scan match. Chart one below to see the log-scale quarterly EPS trend."
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
            "momentum_burst": ["min_gain_pct", "min_range_expansion_ratio", "max_consolidation_range_pct", "min_efficiency_ratio", "min_rs_rating"],
        # .get() rather than a raw subscript: a setup registered without an
        # entry here used to raise KeyError at render time and take the whole
        # tab down. Falling back to every numeric field on its settings object
        # is a strictly better default than crashing.
        }.get(opt_setup)
        if grid_param_options is None:
            opt_settings_obj = getattr(settings, opt_attr)
            grid_param_options = [
                f for f, v in asdict(opt_settings_obj).items() if isinstance(v, (int, float)) and not isinstance(v, bool)
            ]

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

        st.divider()
        st.subheader("Compare Legout scans")
        st.caption(
            "Backtests legout.github.io's TC2000-style scans the same way as the patterns above -- same "
            "stop/partial/trail mechanics, same stats -- so you can see which of these actually made money "
            "historically instead of just triggering often. Each scan already computes its match condition "
            "across the full history internally; the Scanner tab's version just looks at the latest bar to "
            "build today's watchlist, this uses every historical bar instead."
        )
        legout_compare_selected = st.multiselect(
            "Scans to compare", list(LEGOUT_SCANS.keys()),
            default=[k for k, v in LEGOUT_SCANS.items() if v.get("default_on")],
            format_func=lambda k: LEGOUT_SCANS[k]["label"], key="legout_compare_selected",
        )
        if st.button("Compare Legout scans") and legout_compare_selected:
            from indicators import build_close_panel, compute_rs_rating_panel

            legout_compare_rows = []
            legout_compare_progress = st.progress(0.0, text="Backtesting each scan...")
            legout_rs_panel = None
            if any(LEGOUT_SCANS[k]["uses_rs"] for k in legout_compare_selected):
                close_panel = build_close_panel(bt_history)
                if not close_panel.empty:
                    rs_df = compute_rs_rating_panel(
                        close_panel, settings.rs_rating.lookback_periods, settings.rs_rating.period_weights
                    )
                    legout_rs_panel = {sym: rs_df[sym] for sym in rs_df.columns}
            for i, key in enumerate(legout_compare_selected):
                spec = LEGOUT_SCANS[key]
                legout_signals = scan_legout_signals_over_history(bt_history, key, rs_panel=legout_rs_panel)
                legout_result = run_backtest(settings.backtest, legout_signals, side="long")
                legout_stats = compute_stats(legout_result)
                legout_compare_rows.append({"pattern": spec["label"], "side": "long", **legout_stats})
                legout_compare_progress.progress(
                    (i + 1) / len(legout_compare_selected), text=f"{spec['label']} done"
                )
            legout_compare_progress.empty()
            legout_compare_df = pd.DataFrame(legout_compare_rows).sort_values(
                "expectancy_r", ascending=False, na_position="last"
            ).reset_index(drop=True)
            st.session_state.legout_pattern_comparison = legout_compare_df

        legout_pattern_comparison = st.session_state.get("legout_pattern_comparison")
        if legout_pattern_comparison is not None and not legout_pattern_comparison.empty:
            st.caption("🟢 good  🟡 mixed/borderline  🔴 weak, per-column -- ranked by expectancy (most useful historically first).")
            st.dataframe(style_results_dataframe(legout_pattern_comparison), use_container_width=True, height=300)
            legout_best = legout_pattern_comparison.iloc[0]
            if pd.notna(legout_best.get("expectancy_r")) and legout_best["trade_count"] > 0:
                st.success(
                    f"Best performer here: **{legout_best['pattern']}** -- expectancy {legout_best['expectancy_r']:.2f}R "
                    f"over {int(legout_best['trade_count'])} trades, win rate {legout_best['win_rate']:.1f}%, "
                    f"CAGR {legout_best['cagr']:.1f}%. Small sample sizes (especially with 'Limit to symbol') can be "
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
            pf_rs_rating = None
            if rs_series is not None and marker_date is not None:
                rs_asof = rs_series[rs_series.index <= pd.Timestamp(marker_date)]
                if not rs_asof.empty:
                    pf_rs_rating = rs_asof.iloc[-1]
            st.plotly_chart(
                plot_symbol(
                    history[pf_symbol], pf_symbol, marker_date=marker_date,
                    setup_key=pf_setup, settings=settings, row=observed,
                    side=SETUP_REGISTRY[pf_setup]["side"],
                    rs_rating=pf_rs_rating,
                ),
                use_container_width=True,
                key="param_finder_chart",
            )
            col_pf_wl, col_pf_pt = st.columns(2)
            with col_pf_wl:
                render_add_to_watchlist_button(pf_symbol, "param_finder")
            with col_pf_pt:
                render_add_to_paper_trading_button(
                    pf_symbol, marker_date, "param_finder", setup_tag=SETUP_REGISTRY[pf_setup]["label"]
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
        col_sync_btn, col_sync_auto, col_sync_interval = st.columns([1.4, 1, 1])
        with col_sync_btn:
            if st.button("🔄 Sync Paper Trades now", key="pt_sync_btn"):
                pt_synced, pt_failed = sync_paper_trade_prices()
                if pt_synced:
                    st.success(
                        f"Synced current price for: {', '.join(pt_synced)}"
                        + (f" (couldn't fetch: {', '.join(pt_failed)})" if pt_failed else "")
                    )
                elif pt_failed:
                    st.error(f"Couldn't fetch: {', '.join(pt_failed)}")
        with col_sync_auto:
            pt_auto_sync = st.toggle("Auto-sync", value=st.session_state.get("pt_auto_sync", False), key="pt_auto_sync")
        with col_sync_interval:
            pt_sync_interval_label = st.selectbox(
                "Every", ["30s", "60s", "2min", "5min"], index=1, key="pt_sync_interval_label", disabled=not pt_auto_sync,
            )

        if pt_auto_sync:
            pt_sync_seconds = {"30s": 30, "60s": 60, "2min": 120, "5min": 300}[pt_sync_interval_label]

            @st.fragment(run_every=pt_sync_seconds)
            def _paper_auto_sync_tick():
                # Deliberately does NOT force a full st.rerun() -- a
                # fragment's own background timer tick isn't a normal
                # foreground user click, and forcing a full rerun from it
                # was observed to reset the active tab back to Scanner,
                # which is worse than the staleness it's meant to fix. This
                # silently keeps st.session_state.history current; the
                # trade table/stats below pick up the refreshed prices the
                # next time anything triggers a normal rerun (switching
                # tabs, tweaking a slider, clicking "Sync Paper Trades now"
                # for an immediate look).
                sync_paper_trade_prices()
                st.caption(f"Auto-synced in the background at {pd.Timestamp.now().strftime('%H:%M:%S')}.")

            _paper_auto_sync_tick()
            st.caption("Results below reflect the latest auto-synced prices as of your next interaction with the page.")

        # Streamlit reruns this whole script on ANY widget interaction
        # anywhere on the page, so a plain pt_simulate() call here would
        # re-run a full backtest.engine day-loop (plus re-computing every
        # indicator over each symbol's whole history) on every unrelated
        # click too -- e.g. just picking which trade to chart. Recompute
        # only when what it actually depends on changes: the trade list,
        # the rules, or fresh data having been loaded for a symbol in use.
        paper_symbols_in_use = {t["symbol"] for t in st.session_state.paper_trades}
        paper_fingerprint = (
            tuple((t["symbol"], t["decision_date"]) for t in st.session_state.paper_trades),
            tuple(asdict(st.session_state.paper_rules).items()),
            tuple(sorted(
                (s, str(history[s].index[-1]))
                for s in paper_symbols_in_use
                if s in history and not history[s].empty
            )),
        )
        if st.session_state.get("paper_fingerprint") != paper_fingerprint:
            st.session_state.paper_result, st.session_state.paper_skipped = pt_simulate(
                history, st.session_state.paper_trades, st.session_state.paper_rules
            )
            # Permanently record any trade that closed for real this run --
            # otherwise its win/loss would only ever be recomputed live from
            # `history`, and would silently revert to "Not simulated" once
            # the loaded date window later moves past it.
            updated_trades, froze_any = pt_freeze_closed_trades(
                st.session_state.paper_trades, st.session_state.paper_result
            )
            if froze_any:
                st.session_state.paper_trades = updated_trades
                pt_save_trades(st.session_state.paper_trades)
            st.session_state.paper_fingerprint = paper_fingerprint
        paper_result = st.session_state.paper_result
        paper_skipped = st.session_state.paper_skipped
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
            color = {
                "Win": "#2ecc71", "Loss": "#e74c3c", "Open": "#f1c40f",
                "Pending": "#3498db", "Not simulated": "#888888",
            }.get(status)
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

        st.markdown("**Closed trades (history)**")
        paper_closed = paper_joined[paper_joined["status"].isin(["Win", "Loss", "Breakeven"])].copy()
        if paper_closed.empty:
            st.caption(
                "Nothing closed yet -- wins/losses collect here permanently once a logged trade hits its "
                "stop, target, or trailing exit, regardless of what's currently loaded above."
            )
        else:
            paper_closed["_exit_sort"] = pd.to_datetime(paper_closed["exit_date"])
            paper_closed = paper_closed.sort_values("_exit_sort", ascending=False).drop(columns="_exit_sort")
            closed_wins = int((paper_closed["status"] == "Win").sum())
            closed_losses = int((paper_closed["status"] == "Loss").sum())
            closed_total = len(paper_closed)
            win_rate = (closed_wins / closed_total * 100) if closed_total else 0.0
            avg_r = paper_closed["r_multiple"].mean()
            st.caption(
                f"{closed_total} closed -- {closed_wins}W / {closed_losses}L ({win_rate:.0f}% win rate), "
                f"avg {avg_r:+.2f}R. Kept here permanently once realized, even after the loaded date range "
                "moves past a trade."
            )
            st.dataframe(
                paper_closed[display_cols].style.apply(_style_paper_row, axis=1),
                use_container_width=True,
                height=min(80 + 35 * len(paper_closed), 350),
                key="paper_closed_table",
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

        col_paper_pick, col_paper_period, col_paper_remove = st.columns([3, 2, 1])
        with col_paper_pick:
            paper_pick_label = st.selectbox("Chart this trade", paper_labels, key="paper_chart_pick")
        paper_pick_idx = paper_labels.index(paper_pick_label)
        with col_paper_period:
            paper_period = st.selectbox(
                "Chart period", list(CHART_PERIODS.keys()), index=4, key="paper_chart_period",
                help="Windowed around the decision date, not 'today' -- an older logged trade won't get lost "
                "off the left edge of a chart sized to the most recent bars.",
            )
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
            paper_decision_date = pd.Timestamp(paper_pick_row["decision_date"])
            paper_chart_df = slice_around_date(history[paper_pick_symbol], paper_period, paper_decision_date)
            st.plotly_chart(
                plot_symbol(
                    paper_chart_df,
                    paper_pick_symbol,
                    marker_date=paper_decision_date,
                    side="long",
                    entry_date=paper_pick_row["entry_date"],
                    entry_price=paper_pick_row["entry_price"],
                    partial_date=paper_pick_row["partial_date"],
                    partial_price=paper_pick_row["partial_price"],
                    exit_date=paper_pick_row["exit_date"],
                    exit_price=paper_pick_row["exit_price"],
                    exit_reason=paper_pick_row["exit_reason"],
                    r_multiple=paper_pick_row["r_multiple"],
                    **_lightweight_fundamentals(paper_pick_symbol),
                ),
                use_container_width=True,
                key="paper_chart",
            )
            render_add_to_watchlist_button(paper_pick_symbol, "paper")
            if paper_pick_row["notes"]:
                st.caption(f"Notes: {paper_pick_row['notes']}")


# --------------------------------------------------------------------------
# Tab: Watchlist -- symbols flagged via the "+ Watchlist" button next to any
# chart in the app. No moving-average rule attached (that's Sell Alerts) --
# just a quick way to bookmark something to look at again, browsed the same
# way as "Browse all charts" but scoped to just this list.
# --------------------------------------------------------------------------
with tab_watchlist:
    st.subheader("Watchlist")
    st.caption(
        "Symbols you've flagged with the \"+ Watchlist\" button next to any chart in the app. Add/remove "
        "below, or just browse their charts."
    )

    if not st.session_state.watchlist:
        st.info("Nothing on your watchlist yet -- click \"+ Watchlist\" next to any chart to add it.")
    else:
        for i, sym in enumerate(st.session_state.watchlist):
            col_sym, col_remove = st.columns([4, 1])
            col_sym.write(sym)
            if col_remove.button("Remove", key=f"watchlist_remove_{i}"):
                st.session_state.watchlist.pop(i)
                save_general_watchlist(st.session_state.watchlist)
                st.rerun()

    col_wl_add_sym, col_wl_add_btn = st.columns([3, 1])
    with col_wl_add_sym:
        watchlist_new_symbol = st.text_input("Add symbol", key="watchlist_new_symbol").strip().upper()
    with col_wl_add_btn:
        st.write("")
        if st.button("Add", key="watchlist_add_btn") and watchlist_new_symbol:
            if watchlist_new_symbol not in st.session_state.watchlist:
                st.session_state.watchlist.append(watchlist_new_symbol)
                save_general_watchlist(st.session_state.watchlist)
                st.rerun()

    st.divider()

    watchlist_loaded_symbols = sorted(s for s in st.session_state.watchlist if s in history)
    if not watchlist_loaded_symbols:
        st.warning(
            "None of your watchlist symbols are loaded yet -- load them from the sidebar first "
            "(Custom symbols) to see their charts here."
        )
    else:
        if (
            "watchlist_symbol_select" not in st.session_state
            or st.session_state.watchlist_symbol_select not in watchlist_loaded_symbols
        ):
            st.session_state.watchlist_symbol_select = watchlist_loaded_symbols[0]
        current_wlidx = watchlist_loaded_symbols.index(st.session_state.watchlist_symbol_select)

        col_wl_prev, col_wl_pick, col_wl_next, col_wl_period = st.columns([1, 3, 1, 2])
        with col_wl_prev:
            if st.button("⬅ Prev", key="watchlist_prev"):
                st.session_state.watchlist_symbol_select = watchlist_loaded_symbols[(current_wlidx - 1) % len(watchlist_loaded_symbols)]
        with col_wl_next:
            if st.button("Next ➡", key="watchlist_next"):
                st.session_state.watchlist_symbol_select = watchlist_loaded_symbols[(current_wlidx + 1) % len(watchlist_loaded_symbols)]
        with col_wl_pick:
            watchlist_symbol = st.selectbox("Symbol", watchlist_loaded_symbols, key="watchlist_symbol_select")
        with col_wl_period:
            watchlist_period = st.selectbox(
                "Chart period", [AUTO_CHART_PERIOD] + list(CHART_PERIODS.keys()), index=0, key="watchlist_chart_period",
            )

        watchlist_match = None
        if not results.empty:
            sym_matches = results[results["symbol"] == watchlist_symbol]
            if not sym_matches.empty:
                watchlist_match = sym_matches.iloc[0]
        watchlist_setup_key = LABEL_TO_SETUP_KEY.get(watchlist_match["setup"]) if watchlist_match is not None else None
        watchlist_df = resolve_chart_df(history[watchlist_symbol], watchlist_period, watchlist_match, watchlist_setup_key, settings)

        if watchlist_match is not None:
            st.plotly_chart(
                plot_symbol(
                    watchlist_df, watchlist_symbol, marker_date=watchlist_match["date"],
                    setup_key=watchlist_setup_key, settings=settings, row=watchlist_match,
                    side=watchlist_match.get("side", "long"),
                    market_cap=watchlist_match.get("market_cap"), company_name=watchlist_match.get("company_name"),
                    sector=watchlist_match.get("sector"),
                    revenue_growth_pct=watchlist_match.get("revenue_growth_pct"), eps_growth_pct=watchlist_match.get("eps_growth_pct"),
                    rs_rating=watchlist_match.get("rs_rating"),
                ),
                use_container_width=True,
                key="watchlist_chart",
            )
            col_wlt_pt, col_wlt_sa = st.columns(2)
            with col_wlt_pt:
                render_add_to_paper_trading_button(
                    watchlist_symbol, watchlist_match["date"], "watchlist_tab", setup_tag=watchlist_match["setup"]
                )
            with col_wlt_sa:
                render_add_to_sell_alerts_button(watchlist_symbol, "watchlist_tab")
        else:
            st.plotly_chart(
                plot_symbol(watchlist_df, watchlist_symbol, **_lightweight_fundamentals(watchlist_symbol)),
                use_container_width=True, key="watchlist_chart",
            )
            col_wlt_nm_pt, col_wlt_nm_sa = st.columns(2)
            with col_wlt_nm_pt:
                render_add_to_paper_trading_button(
                    watchlist_symbol, history[watchlist_symbol].index[-1], "watchlist_tab_nomatch"
                )
            with col_wlt_nm_sa:
                render_add_to_sell_alerts_button(watchlist_symbol, "watchlist_tab_nomatch")


# --------------------------------------------------------------------------
# Tab: Stop Calculator -- look up a single ticker and get the stop price +
# suggested share count for placing the trade on a different platform.
# --------------------------------------------------------------------------
with tab_stopcalc:
    st.subheader("Stop Loss Calculator")
    st.caption(
        "Look up any ticker and get the stop price + suggested share count for a trade you'll place "
        "elsewhere. Uses the exact same stop/position-sizing formulas as Backtest & Optimizer, but with "
        "its own fixed settings below -- independent of the Designer tab, so this doesn't drift just "
        "because you're experimenting with sliders elsewhere."
    )

    with st.expander("Stop calculator settings (independent of the Designer tab)", expanded=False):
        sc_col1, sc_col2 = st.columns(2)
        with sc_col1:
            st.session_state.sc_side = st.radio(
                "Side", ["long", "short"], index=0 if st.session_state.get("sc_side", "long") == "long" else 1,
                key="sc_side_radio", horizontal=True,
            )
            st.session_state.sc_stop_mode = st.selectbox(
                "Stop placement", ["low_of_signal_day", "adr_multiple"],
                index=0 if st.session_state.get("sc_stop_mode", "low_of_signal_day") == "low_of_signal_day" else 1,
                format_func=lambda v: "Low of signal day (capped at the ADR)" if v == "low_of_signal_day" else "Pure ADR multiple from entry",
                key="sc_stop_mode_select",
            )
            st.session_state.sc_stop_adr_multiple = st.slider(
                "Stop distance cap (x ADR)", 0.25, 3.0, float(st.session_state.get("sc_stop_adr_multiple", 1.0)), 0.25,
                key="sc_stop_adr_slider",
            )
        with sc_col2:
            st.session_state.sc_equity = st.number_input(
                "Account equity ($)", min_value=0.0, value=float(st.session_state.get("sc_equity", 100_000.0)), step=1000.0,
                key="sc_equity_input",
            )
            st.session_state.sc_risk_pct = st.slider(
                "Risk % per trade", 0.1, 5.0, float(st.session_state.get("sc_risk_pct", 1.0)), 0.1, key="sc_risk_pct_slider",
            )
            st.session_state.sc_max_position_pct = st.slider(
                "Max position size (% of equity)", 5.0, 100.0, float(st.session_state.get("sc_max_position_pct", 20.0)), 5.0,
                key="sc_max_position_pct_slider",
            )
            st.session_state.sc_max_liquidity_pct = st.slider(
                "Max position (% of avg volume, 0 disables)", 0.0, 5.0, float(st.session_state.get("sc_max_liquidity_pct", 1.0)), 0.5,
                key="sc_max_liquidity_pct_slider",
            )

    col_sc_symbol, col_sc_entry, col_sc_btn = st.columns([2, 2, 1])
    with col_sc_symbol:
        sc_symbol = st.text_input("Ticker", key="sc_symbol_input").strip().upper()
    with col_sc_entry:
        sc_entry_override = st.number_input(
            "Entry price override (optional, 0 = use latest close)", min_value=0.0, value=0.0, step=0.01, key="sc_entry_override",
        )
    with col_sc_btn:
        st.write("")
        sc_go = st.button("Calculate", key="sc_calculate_btn")

    # Computed once on "Calculate" and stashed in session_state, then always
    # rendered from there (not gated on sc_go) -- otherwise clicking
    # "+ Watchlist"/"+ Paper Trading" below triggers its own rerun where
    # sc_go is False again, and the whole results section would vanish right
    # as you click it.
    if sc_go and sc_symbol:
        try:
            if sc_symbol in history and not history[sc_symbol].empty:
                sc_df = history[sc_symbol].copy()
            else:
                sc_client = get_client()
                sc_end = pd.Timestamp.today().strftime("%Y-%m-%d")
                sc_start = (pd.Timestamp.today() - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
                sc_df = get_history(sc_client, sc_symbol, sc_start, sc_end)

            if sc_df is None or sc_df.empty or len(sc_df) < 25:
                st.session_state.sc_computed = None
                st.error(f"Not enough history for {sc_symbol} to compute a 20-day ADR.")
            else:
                sc_df = add_adr_pct(sc_df, lookback=20)
                sc_df = add_volume_stats(sc_df, avg_period=50)
                sc_last = sc_df.iloc[-1]
                sc_entry_price = sc_entry_override if sc_entry_override > 0 else float(sc_last["close"])

                sc_result = compute_stop_and_size(
                    entry_price=sc_entry_price,
                    signal_low=float(sc_last["low"]),
                    signal_high=float(sc_last["high"]),
                    adr_pct=float(sc_last["adr_pct_20"]) if pd.notna(sc_last["adr_pct_20"]) else 0.0,
                    side=st.session_state.sc_side,
                    stop_mode=st.session_state.sc_stop_mode,
                    stop_adr_multiple=st.session_state.sc_stop_adr_multiple,
                    risk_pct_per_trade=st.session_state.sc_risk_pct,
                    equity=st.session_state.sc_equity,
                    max_position_pct_of_equity=st.session_state.sc_max_position_pct,
                    max_position_pct_of_avg_volume=st.session_state.sc_max_liquidity_pct,
                    avg_volume=float(sc_last["vol_avg_50"]) if pd.notna(sc_last.get("vol_avg_50")) else None,
                )
                st.session_state.sc_computed = {
                    "symbol": sc_symbol, "df": sc_df, "result": sc_result, "side": st.session_state.sc_side,
                }
        except FMPError as exc:
            st.session_state.sc_computed = None
            st.error(f"Couldn't fetch {sc_symbol}: {exc}")
    elif sc_go:
        st.warning("Enter a ticker first.")

    sc_computed = st.session_state.get("sc_computed")
    if sc_computed:
        sc_symbol, sc_df, sc_result, sc_calc_side = (
            sc_computed["symbol"], sc_computed["df"], sc_computed["result"], sc_computed["side"]
        )
        if sc_result.get("error"):
            st.error(sc_result["error"])
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Entry price", f"${sc_result['entry_price']:.2f}")
            m2.metric("Stop price", f"${sc_result['stop_price']:.2f}", f"-{sc_result['stop_distance_pct']:.1f}%" if sc_calc_side == "long" else f"+{sc_result['stop_distance_pct']:.1f}%")
            m3.metric("Suggested shares", f"{sc_result['shares']:,}")
            m4, m5, m6 = st.columns(3)
            m4.metric("Position value", f"${sc_result['position_value']:,.0f}")
            m5.metric("Total $ at risk", f"${sc_result['total_risk_dollars']:,.0f}")
            m6.metric("Stop distance", f"${sc_result['stop_distance']:.2f}")
            if sc_result["capped_by"]:
                st.caption(f"Share count capped by: {', '.join(sc_result['capped_by'])}.")
            if sc_result["shares"] <= 0:
                st.warning("Suggested share count is 0 -- risk/position settings are too tight for this stop distance.")

            st.plotly_chart(
                plot_symbol(sc_df.tail(252), sc_symbol, **_lightweight_fundamentals(sc_symbol)),
                use_container_width=True, key="sc_chart",
            )
            col_sc_wl, col_sc_pt = st.columns(2)
            with col_sc_wl:
                render_add_to_watchlist_button(sc_symbol, "stopcalc")
            with col_sc_pt:
                render_add_to_paper_trading_button(sc_symbol, sc_df.index[-1], "stopcalc")


# --------------------------------------------------------------------------
# Tab: Tomorrow's Orders -- ready-to-place stop-buy tickets
#
# The point of this tab: everything else in the app tells you WHAT matched.
# This tells you what to actually put in the broker before the open, with the
# trigger, the stop, and the share count already worked out -- because the
# whole premise of trading this way is that you won't be watching intraday.
# --------------------------------------------------------------------------
with tab_orders:
    st.subheader("Tomorrow's orders")
    st.caption(
        "Turns the latest scan into stop-buy tickets you can place before the open. Every number here comes "
        "from the same code the backtest uses to fill and stop a trade, so what you place is what was tested."
    )

    if not history:
        st.info("Load data from the sidebar first, then run a scan on the Scanner tab.")
    elif st.session_state.last_scan.empty:
        st.info("No scan results yet -- run a scan on the Scanner tab first.")
    else:
        scan_all = st.session_state.last_scan

        col_acct, col_risk, col_stars2 = st.columns(3)
        with col_acct:
            account_size = st.number_input(
                "Account size ($)", 1_000, 100_000_000, int(settings.backtest.initial_capital), 1_000,
                key="orders_account",
            )
        with col_risk:
            risk_pct = st.slider(
                "Risk per trade (%)", 0.1, 3.0, float(settings.backtest.risk_pct_per_trade), 0.05,
                key="orders_risk_pct",
                help="Qullamaggie's own answer: \"most of the time 0.3-0.5%, rarely more than 1%.\"",
            )
        with col_stars2:
            orders_min_stars = st.slider(
                "Minimum stars", 0, 6, 0, 1, key="orders_min_stars",
                help="Defaults to 0 because raising it measurably hurt the one setup with a real edge -- "
                "see \"Which system should I actually trade?\" below. Use the star column to decide which "
                "tickets to place first, not which to discard.",
            )

        setup_labels = sorted(scan_all["setup"].unique().tolist())
        # Momentum Burst and Breakout by default: the two setups whose hold
        # period and entry mechanics actually match placing a stop-buy and
        # holding 3-5 days. The O'Neil patterns are built for multi-week
        # holds -- available here, just not the default.
        preferred = [
            l for l in setup_labels
            if l.startswith("Momentum Burst") or l == "Breakout"
        ] or setup_labels
        chosen_setups = st.multiselect(
            "Setups to draw from", setup_labels, default=preferred, key="orders_setups",
            help="Defaults to Momentum Burst and Breakout -- the two with a 3-5 day horizon. The O'Neil "
            "patterns are designed for multi-week holds, so they'll behave differently on this timetable.",
        )

        picks = scan_all[scan_all["setup"].isin(chosen_setups)]
        if "stars" in picks.columns:
            picks = picks[picks["stars"].fillna(0) >= orders_min_stars]

        if picks.empty:
            st.warning(
                f"Nothing at {orders_min_stars}★ or better in the selected setups. Lower the bar, widen the "
                "setups, or accept that today has nothing worth an order -- the last one is a real answer."
            )
        else:
            bt = settings.backtest
            order_rows, skipped_rows = [], []
            for _, r in picks.iterrows():
                sym = r["symbol"]
                if sym not in history or history[sym].empty:
                    skipped_rows.append({"symbol": sym, "reason": "not loaded"})
                    continue
                sdf = history[sym]
                sig_date = pd.Timestamp(r["date"])
                if sig_date not in sdf.index:
                    skipped_rows.append({"symbol": sym, "reason": "signal date not in history"})
                    continue
                bar = sdf.loc[sig_date]
                adr_at = compute_adr_pct_at(sdf, sig_date)
                signal_bar = {
                    "high": float(bar["high"]), "low": float(bar["low"]),
                    "adr_pct": adr_at, "vol_avg_50": None,
                }
                # entry_bar=None: tomorrow hasn't happened. The trigger, stop
                # and size are all knowable now; only the fill isn't, so the
                # plan assumes a fill at the trigger.
                plan = resolve_entry_and_stop(bt, "long", signal_bar, None)
                if not plan.tradeable:
                    skipped_rows.append({"symbol": sym, "reason": plan.skip_reason})
                    continue

                avg_vol = float(sdf["volume"].tail(50).mean()) if "volume" in sdf.columns else None
                sizing = compute_stop_and_size(
                    entry_price=plan.entry_price, signal_low=float(bar["low"]), signal_high=float(bar["high"]),
                    adr_pct=adr_at, side="long", stop_mode=bt.stop_mode,
                    stop_adr_multiple=bt.stop_adr_multiple, risk_pct_per_trade=risk_pct,
                    equity=float(account_size),
                    max_position_pct_of_equity=bt.max_position_pct_of_equity,
                    max_position_pct_of_avg_volume=bt.max_position_pct_of_avg_volume,
                    avg_volume=avg_vol,
                    stop_exceeds_adr_action=bt.stop_exceeds_adr_action,
                    stop_distance=plan.stop_distance,
                )
                if sizing.get("error"):
                    skipped_rows.append({"symbol": sym, "reason": sizing["error"]})
                    continue

                order_rows.append({
                    "symbol": sym,
                    "setup": r["setup"],
                    "★": int(r["stars"]) if pd.notna(r.get("stars")) else None,
                    "stop-buy trigger": round(plan.trigger_price, 2) if plan.trigger_price else None,
                    "limit cap": round(plan.limit_cap_price, 2) if plan.limit_cap_price else None,
                    "stop": round(plan.stop_price, 2),
                    "risk/share": round(plan.stop_distance, 2),
                    "shares": sizing["shares"],
                    "position $": round(sizing["position_value"], 0),
                    "% of acct": round(sizing["position_value"] / account_size * 100.0, 1),
                    "risk $": round(sizing["total_risk_dollars"], 0),
                    "sell-half at": round(plan.target_price, 2),
                    "ADR%": round(adr_at, 1) if adr_at else None,
                    "RS": round(r["rs_rating"], 0) if pd.notna(r.get("rs_rating")) else None,
                    "days to earnings": r.get("days_to_earnings"),
                    "capped by": ", ".join(sizing["capped_by"]) or "",
                })

            if not order_rows:
                st.warning("Every candidate was filtered out at the order stage -- see the skipped list below.")
            else:
                orders_df = pd.DataFrame(order_rows)
                total_risk = orders_df["risk $"].sum()
                total_pos = orders_df["position $"].sum()
                c1, c2, c3 = st.columns(3)
                c1.metric("Orders", len(orders_df))
                c2.metric("Total at risk", f"${total_risk:,.0f}", f"{total_risk / account_size * 100:.1f}% of account")
                c3.metric("Total position value", f"${total_pos:,.0f}", f"{total_pos / account_size * 100:.0f}% of account")
                if total_risk / account_size * 100 > 6:
                    st.warning(
                        f"These orders risk {total_risk / account_size * 100:.1f}% of the account if every one "
                        "fills and every one stops out. They won't all fill -- but they can all fill on the same "
                        "strong morning, which is exactly when you'd least expect to be at full risk."
                    )
                st.dataframe(orders_df, use_container_width=True, height=min(80 + 35 * len(orders_df), 420))
                gap_note = (
                    f"a **limit cap** column is shown, so you can place a stop-LIMIT and refuse anything worse"
                    if bt.max_gap_fill_pct > 0 else
                    "**max gap fill %** is currently 0 in the Designer tab, so a gap of any size gets filled"
                )
                st.caption(
                    f"The stop, share count and sell-half target are exact. The **trigger** is exact too, but "
                    f"the price you actually get is not: if the stock opens *above* your trigger, you fill at "
                    f"the open, not the trigger. Measured on a 5% gap that's about 3% worse than the numbers "
                    f"above -- and the stop/risk figures shift with it. Right now {gap_note}."
                )
                st.download_button(
                    "Download as CSV", orders_df.to_csv(index=False).encode(),
                    file_name=f"orders_{as_of}.csv", mime="text/csv", key="orders_csv",
                )

            if skipped_rows:
                with st.expander(f"Skipped ({len(skipped_rows)}) -- and why"):
                    st.caption(
                        "These matched the scan but didn't become orders. Worth reading rather than ignoring: "
                        "\"anti-chase\" and \"stop wider than the ADR cap\" are both the system correctly "
                        "declining a trade, not a failure."
                    )
                    st.dataframe(pd.DataFrame(skipped_rows), use_container_width=True)

    # Reference material, deliberately OUTSIDE the has-results branch: the
    # exit plan and the system comparison are worth reading on a day with no
    # picks at all -- arguably especially then.
    st.divider()
    st.markdown("**The plan once you're filled**")
    st.info(
        f"1. Sell **{settings.backtest.partial_profit_fraction:.0%}** into strength at the "
        f"**sell-half** target (≈{settings.backtest.partial_profit_r_multiple:.0f}R above entry), or after "
        f"**{settings.backtest.partial_profit_max_days} days**, whichever comes first.\n\n"
        f"2. Move the stop on the rest to **breakeven**.\n\n"
        f"3. Trail the remainder with the **{settings.backtest.trail_ema_period}-day "
        f"{settings.backtest.trail_ma_type.upper()}**, exiting on the first **close** below it. "
        "Closes only -- an intraday poke below the average doesn't count.\n\n"
        "That's the whole exit. The hard part isn't knowing it, it's doing it on the days it feels wrong."
    )

    with st.expander("⚠️ How a stop-buy differs from what Qullamaggie actually does"):
        st.markdown(
            "He doesn't place resting premarket orders. He watches the open and buys the **opening range "
            "high** -- the stock has to take out its first 1-, 5- or 60-minute high before he'll touch it. "
            "If it gaps up and then fails to clear that level, **he skips the trade entirely**.\n\n"
            "That filter is a real part of the edge, and a resting stop-buy doesn't have it: it fills you "
            "on exactly those failed gaps. The closest substitute here is **max gap fill %** in the "
            "Designer tab, which stands aside when the stock opens too far past your trigger. It is a "
            "blunter tool than watching the open -- worth knowing, not worth pretending otherwise.\n\n"
            "The other honest gap: your stop can't be \"the low of the day\" when you place the order "
            "before the day exists. What's used instead is the **signal day's low**, capped at "
            f"**{settings.backtest.stop_adr_multiple:.1f}× ADR**."
        )

    with st.expander("🤔 Which system should I actually trade?"):
        st.markdown(
            "Measured, not guessed. Every long setup was backtested twice on 400 cached symbols over "
            "2022-08 to 2026-08, once with the old next-open fill and once with **your** stop-buy "
            "execution. Ranked by expectancy in R under stop-buy:\n"
        )
        st.dataframe(
            pd.DataFrame([
                {"setup": "Breakout", "trades": 135, "win %": 54.8, "expectancy (R)": 0.473,
                 "profit factor": 2.15, "max DD %": -14.0},
                {"setup": "Cup with Handle", "trades": 65, "win %": 50.8, "expectancy (R)": 0.390,
                 "profit factor": 1.76, "max DD %": -15.1},
                {"setup": "Ascending Base", "trades": 133, "win %": 51.1, "expectancy (R)": 0.087,
                 "profit factor": 1.28, "max DD %": -14.6},
                {"setup": "Double Bottom", "trades": 490, "win %": 42.0, "expectancy (R)": 0.016,
                 "profit factor": 1.06, "max DD %": -88.3},
                {"setup": "Flat Base", "trades": 81, "win %": 46.9, "expectancy (R)": -0.059,
                 "profit factor": 0.90, "max DD %": -10.8},
                {"setup": "Episodic Pivot", "trades": 3, "win %": 33.3, "expectancy (R)": -0.231,
                 "profit factor": 0.67, "max DD %": -0.9},
                {"setup": "Momentum Burst", "trades": 84, "win %": 39.3, "expectancy (R)": -0.233,
                 "profit factor": 0.62, "max DD %": -17.3},
            ]),
            use_container_width=True, hide_index=True,
        )
        st.markdown(
            "**Breakout wins, and it isn't close.** Expectancy 0.47R at a profit factor of 2.15 over 135 "
            "trades. The reason is visible in the exits: its trailing-stop exits average **+3.6R** while "
            "its stopped-out trades average -0.29R. The winners run far enough to pay for the losers "
            "several times over. That is the entire edge, and it means the discipline that matters most "
            "is *not* cutting a winner early.\n\n"
            "**Momentum Burst tested negative here, and I'd previously have told you it was your best "
            "fit.** It isn't, on this evidence. Its trail exits average only +0.75R — the winners never "
            "get big enough. I checked whether the exit rules were simply wrong for a 3-5 day burst by "
            "re-running it with faster trails (3/5/8-day) and earlier partials; it stayed negative in "
            "every variant, so the problem is the entries, not the exits. Plausible reasons: this "
            "universe skews mid/large-cap while Bonde's method leans on smaller, more volatile names, "
            "and he treats his scan as a shortlist to judge by eye rather than a mechanical system. "
            "**Don't trade it on my say-so.** It's kept here because the scan is faithful to his "
            "published rules and you may want to tune it — not because it's earned a place yet.\n\n"
            "**Stop-buy execution helped the good setups and exposed the weak ones.** Breakout's "
            "expectancy nearly doubled (0.249 → 0.473) and Cup with Handle's almost tripled, while "
            "Double Bottom's and Flat Base's collapsed. Filtering out the signals that never traded "
            "through their trigger removes noise from a real edge and removes the *illusion* of one "
            "where the fills were doing the work.\n\n"
            "**The star score does NOT work as a filter, and I dug into why rather than just noting it.** "
            "For each of the seven components I correlated its value at the signal date against the "
            "eventual trade R-multiple, across all 135 Breakout trades and again restricted to just the "
            "26 trail-exit trades (the ones the edge actually lives in). Every single component came back "
            "statistically indistinguishable from zero on trade *magnitude* — including on the trail-exit "
            "subset, where you'd most expect one to show up. One did show a real, monotonic relationship, "
            "but to a different thing: **prior move size predicted the *probability* of becoming a "
            "trail-exit winner** (quartiles of raw prior-move%: 5.9% → 20.6% → 21.2% → 29.4% trail-exit "
            "rate — a genuine 5x spread), which lines up with his own emphasis on a big prior move. Along "
            "the way I also found the tightness component was silently dead: it measured the wrong window "
            "(the full base instead of the tight recent tail Breakout's own detector actually checks) and "
            "came out exactly 0.0 for all 135 trades. Both are fixed — tightness now measures the correct "
            "window, and prior-move's tiering was widened to capture the quartile pattern instead of a "
            "cliff that scored a −27% decline and a +19% advance identically.\n\n"
            "**Neither fix repaired the filter.** After both changes, raising the bar to 3★ still cuts "
            "Breakout's expectancy by roughly half. I went further and filtered on prior-move *alone* "
            "(dropping the other six components entirely, so nothing could dilute it) — expectancy still "
            "never beat taking every match, at any threshold. The signal is real (the trail-exit-rate "
            "climb is too clean to be coincidence) but too weak, at 135 trades, to build a profitable "
            "filter from. I'd also flagged Cup with Handle earlier as an exception where filtering helped "
            "(0.39 → 0.54 at 3★+) — that claim did not survive the prior-move retiering (now 0.39 → 0.33 "
            "on the same threshold, from a shared component change), which is itself evidence it was "
            "small-sample noise rather than a real effect. Use the stars to decide **what to look at "
            "first**, not what to discard, for any setup. Both star sliders default to 0 for that reason.\n\n"
            "**Two red flags in that table.** Double Bottom shows an 88% maximum drawdown across 490 "
            "trades — a barely-positive expectancy attached to a near-total loss of capital is not a "
            "tradeable system, it's a warning. And Episodic Pivot's 3 trades are far too few to conclude "
            "anything; its own criteria are intraday and it clusters around earnings, so treat that row "
            "as \"unmeasured\", not \"bad\".\n\n"
            "---\n\n"
            "**Read these as a ranking, not as returns.** The universe is momentum-pre-filtered and made "
            "of companies that still exist today, which flatters everything in the table. 400 symbols "
            "over four years is also a modest sample — Breakout's 135 trades is enough to take seriously, "
            "Episodic Pivot's 3 is not. Re-run the comparison on the Backtest tab against your own "
            "universe before you change how you size anything."
        )


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
