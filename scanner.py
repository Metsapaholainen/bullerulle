"""Full scan orchestration: universe -> cached history -> indicators ->
setup detection -> one ranked DataFrame of matches.

Split into two steps on purpose:
  1. `load_scan_universe_history()` -- the expensive, network-touching step.
  2. `run_scan()` -- pure computation over already-fetched history.
This split is what lets the Designer tab change settings and re-run `run_scan`
against the same in-memory/cached history in a second or two.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from data.fmp_client import FMPClient
from data.cache import get_history, get_history_bulk
from data.universe import build_universe
from indicators import add_core_indicators, build_close_panel, compute_rs_rating_panel
from settings import Settings
from setups import SETUP_REGISTRY

DEFAULT_BENCHMARK_SYMBOL = "SPY"


def history_lookback_days(settings: Settings) -> int:
    """Enough calendar days of history to satisfy every rolling window used
    across all setups plus the RS rating, with headroom."""
    candidates = [
        settings.breakout.max_consolidation_days + settings.breakout.prior_move_lookback_days + 30,
        settings.episodic_pivot.lookback_base_days + 30,
        settings.parabolic_short.run_up_lookback_days + 30,
        settings.cup_with_handle.cup_lookback_days + settings.cup_with_handle.prior_uptrend_lookback_days + 30,
        settings.double_bottom.pattern_lookback_days + 30,
        settings.flat_base.base_days + settings.flat_base.prior_move_lookback_days + 30,
        settings.ascending_base.pattern_lookback_days + 30,
        settings.high_tight_flag.flag_lookback_days + settings.high_tight_flag.run_up_lookback_days + 30,
        max(settings.rs_rating.lookback_periods) + 30,
        max(settings.universe.momentum_timeframes_days, default=0) + 30,
        300,  # floor so sma_200 etc. have room even with small windows configured
    ]
    return max(candidates)


def load_scan_universe_history(
    settings: Settings,
    as_of: Optional[str] = None,
    client: Optional[FMPClient] = None,
    symbols: Optional[list] = None,
    on_progress=None,
) -> dict:
    """Builds (or reuses a given) universe and returns {symbol: OHLCV
    DataFrame} covering enough history for every indicator/setup, ending at
    `as_of` (defaults to today)."""
    client = client or FMPClient()
    as_of_dt = pd.Timestamp(as_of) if as_of else pd.Timestamp.today().normalize()
    lookback_calendar_days = int(history_lookback_days(settings) * 1.6)  # pad for weekends/holidays
    start = (as_of_dt - pd.Timedelta(days=lookback_calendar_days)).strftime("%Y-%m-%d")
    end = as_of_dt.strftime("%Y-%m-%d")

    if symbols is None:
        symbols = build_universe(client, settings.universe)

    return get_history_bulk(client, symbols, start, end, on_progress=on_progress)


def enabled_setup_names(settings: Settings) -> list:
    return [
        name
        for name, spec in SETUP_REGISTRY.items()
        if getattr(settings, spec["settings_attr"]).enabled
    ]


# Human-readable labels for the standard momentum lookback windows (trading
# days -> approximate calendar months), used across the app for display.
TIMEFRAME_LABELS = {21: "1 month", 63: "3 months", 126: "6 months", 252: "12 months", 378: "18 months"}


def top_movers_by_timeframe(history: dict, timeframes_days: list, top_pct: float = 2.0) -> dict:
    """For each lookback window (in trading days), rank every symbol's
    trailing % return and return the top `top_pct`% -- e.g. the strongest
    2% of movers over the last 3 months. Returns {days: DataFrame[symbol,
    return_pct]} sorted by return descending."""
    close_panel = build_close_panel(history)
    results = {}
    for days in timeframes_days:
        if close_panel.empty or len(close_panel) <= days:
            results[days] = pd.DataFrame(columns=["symbol", "return_pct"])
            continue
        returns = (close_panel.pct_change(periods=days).iloc[-1] * 100.0).dropna()
        if returns.empty:
            results[days] = pd.DataFrame(columns=["symbol", "return_pct"])
            continue
        cutoff = returns.quantile(1 - top_pct / 100.0)
        top = returns[returns >= cutoff].sort_values(ascending=False)
        results[days] = pd.DataFrame({"symbol": top.index, "return_pct": top.values})
    return results


def momentum_shortlist(history: dict, timeframes_days: list, top_pct: float = 2.0) -> list:
    """Union of the top `top_pct`% movers across every timeframe -- the
    momentum-focused shortlist to run pattern detection against instead of
    the full (possibly 1000+ symbol) universe."""
    by_timeframe = top_movers_by_timeframe(history, timeframes_days, top_pct)
    union = set()
    for df in by_timeframe.values():
        union.update(df["symbol"].tolist())
    return sorted(union)


def ensure_benchmark_loaded(
    history: dict, client: FMPClient, start: str, end: str, benchmark_symbol: str = DEFAULT_BENCHMARK_SYMBOL
) -> pd.DataFrame:
    """Fetch/cache the benchmark (default SPY) over the same date range as
    the rest of the loaded universe, for the relative-strength/resilience
    screen below. Returns its OHLCV DataFrame (also inserted into `history`
    under its own symbol key so callers can chart it if they want)."""
    if benchmark_symbol not in history or history[benchmark_symbol].empty:
        history[benchmark_symbol] = get_history(client, benchmark_symbol, start, end)
    return history[benchmark_symbol]


def relative_strength_resilience(
    history: dict,
    prior_move_days: int = 252,
    recent_window_days: int = 21,
    benchmark_symbol: str = DEFAULT_BENCHMARK_SYMBOL,
) -> pd.DataFrame:
    """Find stocks that made a big prior move and have held onto it --
    "relative strength" in the classic IBD/O'Neil sense: not just "went up
    a lot" but "isn't giving it back," especially compared to the market.

    For each symbol: `prior_move_pct` (return over the last `prior_move_days`,
    e.g. 252 = ~12 months) and `pullback_from_high_pct` (how far the latest
    close sits below the highest close in just the last `recent_window_days`,
    e.g. 21 = ~1 month -- 0 means sitting right at a new high, more negative
    means it's given more back). If the benchmark (default SPY) is available
    in `history`, also reports `resilience_vs_benchmark` = how much less (or
    more) the stock has pulled back than the benchmark over that same recent
    window -- positive means the stock is genuinely outperforming the market,
    not just quiet because nothing's happened. Sorted by prior move first,
    resilience second."""
    close_panel = build_close_panel(history)
    if close_panel.empty:
        return pd.DataFrame()

    benchmark_pullback_pct = None
    if benchmark_symbol in close_panel.columns:
        bench = close_panel[benchmark_symbol].dropna()
        if len(bench) > recent_window_days:
            bench_recent = bench.iloc[-recent_window_days:]
            benchmark_pullback_pct = (bench.iloc[-1] / bench_recent.max() - 1.0) * 100.0

    rows = []
    for symbol in close_panel.columns:
        if symbol == benchmark_symbol:
            continue
        s = close_panel[symbol].dropna()
        if len(s) <= max(prior_move_days, recent_window_days):
            continue

        prior_move_pct = (s.iloc[-1] / s.iloc[-prior_move_days] - 1.0) * 100.0
        recent = s.iloc[-recent_window_days:]
        pullback_from_high_pct = (s.iloc[-1] / recent.max() - 1.0) * 100.0

        row = {
            "symbol": symbol,
            "prior_move_pct": prior_move_pct,
            "pullback_from_high_pct": pullback_from_high_pct,
        }
        if benchmark_pullback_pct is not None:
            # Less negative (or positive) than the benchmark's own pullback
            # means the stock is holding up better than the market itself.
            row["resilience_vs_benchmark"] = pullback_from_high_pct - benchmark_pullback_pct
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    sort_cols = ["prior_move_pct"]
    if "resilience_vs_benchmark" in df.columns:
        sort_cols = ["resilience_vs_benchmark", "prior_move_pct"]
    return df.sort_values(sort_cols, ascending=False).reset_index(drop=True)


def run_scan(
    settings: Settings,
    history: dict,
    as_of: Optional[str] = None,
    setup_names: Optional[list] = None,
) -> pd.DataFrame:
    """Given already-fetched history, run every enabled setup for the most
    recent (or `as_of`) date per symbol and return one ranked DataFrame."""
    setup_names = setup_names or enabled_setup_names(settings)
    if not setup_names or not history:
        return pd.DataFrame(columns=["symbol", "setup", "date", "close", "score"])

    close_panel = build_close_panel(history)
    if close_panel.empty:
        return pd.DataFrame(columns=["symbol", "setup", "date", "close", "score"])

    rs_panel = compute_rs_rating_panel(
        close_panel, settings.rs_rating.lookback_periods, settings.rs_rating.period_weights
    )
    as_of_ts = pd.Timestamp(as_of) if as_of else close_panel.index.max()

    rows = []
    for symbol, raw_df in history.items():
        if raw_df.empty:
            continue
        df_upto = raw_df.loc[:as_of_ts]
        if df_upto.empty:
            continue
        eval_date = df_upto.index.max()

        enriched = add_core_indicators(df_upto)
        rs_series = rs_panel[symbol] if symbol in rs_panel.columns else None

        for setup_name in setup_names:
            spec = SETUP_REGISTRY[setup_name]
            setup_settings = getattr(settings, spec["settings_attr"])
            result = spec["module"].scan(enriched, setup_settings, rs_rating=rs_series)
            if eval_date not in result.index:
                continue
            row = result.loc[eval_date]
            if bool(row.get("match", False)):
                record = {
                    "symbol": symbol,
                    "setup": spec["label"],
                    "side": spec["side"],
                    "date": eval_date,
                    "close": enriched.loc[eval_date, "close"],
                }
                record.update({k: v for k, v in row.items() if k != "match"})
                # Always attach RS rating regardless of whether this setup's own
                # match condition uses it, so results can be prioritized by
                # momentum/leadership across every pattern, not just Breakout.
                record["rs_rating"] = rs_series.get(eval_date) if rs_series is not None else float("nan")
                rows.append(record)

    if not rows:
        return pd.DataFrame(columns=["symbol", "setup", "date", "close", "score"])

    # Prioritize high-momentum (high RS rating) leaders first -- O'Neil and
    # Qullamaggie alike emphasize that these patterns matter most in market
    # leaders, not laggards that happen to technically match. Score is the
    # tie-breaker within the same RS tier.
    return (
        pd.DataFrame(rows)
        .sort_values(["rs_rating", "score"], ascending=[False, False], na_position="last")
        .reset_index(drop=True)
    )


def scan_signals_over_history(
    settings: Settings,
    history: dict,
    setup_name: str,
) -> dict:
    """For backtesting: run one setup across *every* historical bar (not
    just the latest) for every symbol. Returns {symbol: DataFrame} where
    each DataFrame is the setup's full diagnostic output (incl. a `match`
    boolean column) aligned to that symbol's dates."""
    if setup_name not in SETUP_REGISTRY:
        raise ValueError(f"Unknown setup: {setup_name}")
    spec = SETUP_REGISTRY[setup_name]
    setup_settings = getattr(settings, spec["settings_attr"])

    close_panel = build_close_panel(history)
    rs_panel = (
        compute_rs_rating_panel(close_panel, settings.rs_rating.lookback_periods, settings.rs_rating.period_weights)
        if not close_panel.empty
        else pd.DataFrame()
    )

    out = {}
    for symbol, raw_df in history.items():
        if raw_df.empty:
            continue
        enriched = add_core_indicators(raw_df)
        rs_series = rs_panel[symbol] if symbol in rs_panel.columns else None
        out[symbol] = spec["module"].scan(enriched, setup_settings, rs_rating=rs_series).join(
            enriched[["open", "high", "low", "close", "volume"]]
        )
    return out
