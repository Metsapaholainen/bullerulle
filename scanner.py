"""Full scan orchestration: universe -> cached history -> indicators ->
setup detection -> one ranked DataFrame of matches.

Split into two steps on purpose:
  1. `load_scan_universe_history()` -- the expensive, network-touching step.
  2. `run_scan()` -- pure computation over already-fetched history.
This split is what lets the Designer tab change settings and re-run `run_scan`
against the same in-memory/cached history in a second or two.
"""
from __future__ import annotations

import copy
from typing import Optional

import pandas as pd

from data.fmp_client import FMPClient
from data.cache import get_history, get_history_bulk
from data.universe import build_universe
from indicators import add_core_indicators, build_close_panel, compute_rs_rating_panel
from settings import Settings
from setups import SETUP_REGISTRY

DEFAULT_BENCHMARK_SYMBOL = "SPY"
# Nasdaq-100 proxy -- shown alongside SPY in the market-direction banner
# since this scanner's universe skews growth/tech, where QQQ's trend is
# often the more relevant read than the broader S&P 500 alone.
SECONDARY_BENCHMARK_SYMBOL = "QQQ"

# Setups whose scan() accepts a `fundamentals` kwarg for internal scoring
# (currently just Episodic Pivot). Every other setup's results still get
# fundamentals columns attached below -- this only controls which setups'
# own match/score logic *uses* the data internally.
FUNDAMENTALS_AWARE_SETUPS = {"episodic_pivot"}


def history_lookback_days(settings: Settings) -> int:
    """Enough calendar days of history to satisfy every rolling window used
    across all setups plus the RS rating, with headroom."""
    candidates = [
        settings.breakout.max_consolidation_days + settings.breakout.prior_move_lookback_days + 30,
        max(
            settings.episodic_pivot.lookback_base_days,
            settings.episodic_pivot.quiet_base_lookback_days,
            settings.episodic_pivot.prior_ep_lookback_days,
        )
        + 30,
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
    min_history_days: Optional[int] = None,
) -> dict:
    """Builds (or reuses a given) universe and returns {symbol: OHLCV
    DataFrame} covering enough history for every indicator/setup, ending at
    `as_of` (defaults to today).

    `min_history_days` (calendar days) lets a caller request *more* history
    than the auto-computed minimum -- e.g. to backtest over several years
    instead of just enough bars to satisfy the longest pattern window/RS
    lookback. It only ever extends the window, never shrinks it below what
    indicators/setups actually need."""
    client = client or FMPClient()
    as_of_dt = pd.Timestamp(as_of) if as_of else pd.Timestamp.today().normalize()
    lookback_calendar_days = int(history_lookback_days(settings) * 1.6)  # pad for weekends/holidays
    if min_history_days:
        lookback_calendar_days = max(lookback_calendar_days, int(min_history_days))
    start = (as_of_dt - pd.Timedelta(days=lookback_calendar_days)).strftime("%Y-%m-%d")
    end = as_of_dt.strftime("%Y-%m-%d")

    auto_built = symbols is None
    if auto_built:
        symbols = build_universe(client, settings.universe)

    history = get_history_bulk(client, symbols, start, end, on_progress=on_progress)

    # Only apply the liquidity floor to an auto-built (whole-market) universe --
    # if you explicitly typed in custom symbols, you get exactly those symbols,
    # regardless of liquidity.
    if auto_built:
        history = filter_by_liquidity(history, settings.universe.min_avg_dollar_volume)

    return history


def filter_by_liquidity(history: dict, min_avg_dollar_volume: float, lookback_days: int = 20) -> dict:
    """Precise post-fetch liquidity filter: drop symbols whose trailing
    `lookback_days`-day average *dollar* volume (close * volume) falls below
    `min_avg_dollar_volume`. This is exact, unlike trying to approximate a
    dollar-volume floor as a raw share-volume filter at the FMP screener
    level (share count for the same dollar volume varies wildly with price,
    so any fixed conversion there is wrong for most of the universe)."""
    if not min_avg_dollar_volume:
        return history
    out = {}
    for symbol, df in history.items():
        if df.empty:
            continue
        if len(df) < lookback_days:
            out[symbol] = df  # too little history to judge yet -- don't punish new listings
            continue
        dollar_volume = (df["close"] * df["volume"]).tail(lookback_days).mean()
        if dollar_volume >= min_avg_dollar_volume:
            out[symbol] = df
    return out


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


def market_direction(benchmark_df: pd.DataFrame) -> dict:
    """Classify the broad market's trend from the benchmark's (default SPY)
    price relative to its 50/200-day moving averages -- the "M" in CANSLIM.
    O'Neil's core premise: about three out of four stocks tend to follow the
    market's overall direction, so it's worth knowing before acting on any
    individual scan match, win or lose.

    Returns {"direction": "Uptrend"|"Downtrend"|"Choppy / Mixed", "emoji": str,
    "detail": str, "close": float, "sma50": float, "sma200": float} or an
    empty dict if there isn't enough history yet (needs 200+ bars)."""
    if benchmark_df is None or benchmark_df.empty or len(benchmark_df) < 200:
        return {}

    close = benchmark_df["close"].dropna()
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]
    last = close.iloc[-1]
    above50 = last > sma50
    above200 = last > sma200
    golden_cross = sma50 > sma200

    if above50 and above200 and golden_cross:
        direction, emoji = "Uptrend", "🟢"
        detail = "Above both the 50- and 200-day average, with the 50 above the 200 -- a healthy backdrop for long setups."
    elif not above50 and not above200 and not golden_cross:
        direction, emoji = "Downtrend", "🔴"
        detail = "Below both the 50- and 200-day average, with the 50 below the 200 -- a hostile backdrop; most breakouts fail in conditions like this."
    else:
        direction, emoji = "Choppy / Mixed", "🟡"
        detail = "Mixed signals between the 50/200-day averages -- no clear trend either way; be more selective and size down."

    # Faster secondary read: "we always use the 10/20 SMA as our guide" on
    # whether to be aggressive or defensive (Qullamaggie, Nasdaq 90s-vs-today
    # article) -- a quicker-reacting confirmation alongside the 50/200 read
    # above, not a replacement for it.
    sma10 = close.rolling(10).mean().iloc[-1]
    sma20 = close.rolling(20).mean().iloc[-1]
    fast_above10 = last > sma10
    fast_above20 = last > sma20
    fast_cross = sma10 > sma20

    if fast_above10 and fast_above20 and fast_cross:
        fast_direction, fast_emoji = "Aggressive", "🟢"
        fast_detail = "Above both the 10- and 20-day average, with the 10 above the 20 -- short-term momentum favors being aggressive."
    elif not fast_above10 and not fast_above20 and not fast_cross:
        fast_direction, fast_emoji = "Defensive", "🔴"
        fast_detail = "Below both the 10- and 20-day average, with the 10 below the 20 -- favor playing defense, even if the longer-term 50/200 read is fine."
    else:
        fast_direction, fast_emoji = "Neutral", "🟡"
        fast_detail = "Mixed short-term signal between the 10/20-day averages."

    return {
        "direction": direction,
        "emoji": emoji,
        "detail": detail,
        "close": float(last),
        "sma50": float(sma50),
        "sma200": float(sma200),
        "sma10": float(sma10),
        "sma20": float(sma20),
        "fast_direction": fast_direction,
        "fast_emoji": fast_emoji,
        "fast_detail": fast_detail,
    }


# Auto-detect: instead of requiring one exact window-length per pattern
# (e.g. "the cup is exactly 130 days"), sweep a handful of realistic sizes
# spanning what O'Neil's book actually documents, and count it a match if
# *any* of them fits. Depth/quality thresholds (min/max % fields) stay as
# range checks from the current settings -- only the window-length fields
# (how wide/long the pattern is) get swept, since that's what forces you to
# guess a stock's exact pattern size today.
PATTERN_SIZE_VARIANTS = {
    "breakout": [
        {"max_consolidation_days": d, "min_consolidation_days": max(5, int(d * 0.4))}
        for d in (20, 30, 40, 55, 70)
    ],
    "episodic_pivot": [{"lookback_base_days": d} for d in (10, 15, 20, 30, 40)],
    "parabolic_short": [{"run_up_lookback_days": d} for d in (10, 15, 20, 30)],
    "cup_with_handle": [
        {
            "cup_lookback_days": d,
            "left_peak_days": max(10, d // 3),
            "handle_lookback_days": max(8, d // 9),
            "prior_uptrend_lookback_days": max(30, d // 3),
        }
        for d in (45, 65, 90, 130, 180, 250)  # ~9-50 weeks, spanning O'Neil's 7-65wk range
    ],
    "double_bottom": [{"pattern_lookback_days": d} for d in (30, 45, 65, 90, 120)],
    "flat_base": [{"base_days": d} for d in (15, 20, 25, 35, 50)],
    "ascending_base": [{"pattern_lookback_days": d} for d in (45, 60, 75, 100, 130)],
    "high_tight_flag": [
        {"flag_lookback_days": d, "run_up_lookback_days": int(d * 1.5)} for d in (12, 18, 25, 35)
    ],
}


def build_size_variants(base_settings: Settings, setup_name: str) -> list:
    """Returns a list of setup-settings objects (deep copies of the base
    setup's settings with different window-length fields) spanning that
    pattern's realistic size range. With no registered variants for a setup,
    just returns its current settings unchanged."""
    spec = SETUP_REGISTRY[setup_name]
    base = getattr(base_settings, spec["settings_attr"])
    overrides_list = PATTERN_SIZE_VARIANTS.get(setup_name)
    if not overrides_list:
        return [base]
    variants = []
    for overrides in overrides_list:
        variant = copy.deepcopy(base)
        for field, value in overrides.items():
            setattr(variant, field, value)
        variants.append(variant)
    return variants


def run_scan(
    settings: Settings,
    history: dict,
    as_of: Optional[str] = None,
    setup_names: Optional[list] = None,
    auto_detect: bool = False,
    fundamentals: Optional[dict] = None,
    earnings_dates: Optional[dict] = None,
    avoid_earnings_window: bool = True,
    avoid_earnings_days: int = 3,
    min_adr_pct: float = 0.0,
) -> pd.DataFrame:
    """Given already-fetched history, run every enabled setup for the most
    recent (or `as_of`) date per symbol and return one ranked DataFrame.

    With `auto_detect=True`, each setup is tried across several realistic
    pattern sizes (see PATTERN_SIZE_VARIANTS) instead of just the one
    exact size in `settings` -- a symbol counts as a match if *any* size
    fits, and the best-scoring size's diagnostics are kept.

    `fundamentals` (optional {symbol: {revenue_growth_pct, eps_growth_pct,
    float_shares, free_float_pct, insider_acquired_disposed_ratio}}) and
    `earnings_dates` (optional {symbol: "YYYY-MM-DD" or None}) are both
    fetched separately (data/fundamentals_cache.py, on demand for a
    shortlist, not the whole universe) and passed in here. When present,
    every result row -- regardless of which setup produced it -- gets
    revenue_growth_pct/eps_growth_pct/days_to_earnings/lynch_category
    columns attached (mirroring how rs_rating is already universal below).
    Only Episodic Pivot's own scan() additionally *uses* fundamentals for
    internal scoring (see FUNDAMENTALS_AWARE_SETUPS). If
    `avoid_earnings_window` is True and earnings_dates was supplied, rows
    within `avoid_earnings_days` of an earnings date are dropped --
    "avoid buying 3-days before earnings" (Qullamaggie's Laws of Swing),
    applied across every setup, not just EP. `min_adr_pct` drops any row
    whose 20-day ADR% at the signal date is below the threshold -- a global
    tradeability filter applied across every setup, independent of
    whatever any individual setup's own settings require."""
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
            variants = (
                build_size_variants(settings, setup_name)
                if auto_detect
                else [getattr(settings, spec["settings_attr"])]
            )

            scan_kwargs = {}
            if setup_name in FUNDAMENTALS_AWARE_SETUPS and fundamentals is not None:
                scan_kwargs["fundamentals"] = fundamentals.get(symbol, {})

            best_row = None
            best_variant_settings = None
            for variant_settings in variants:
                result = spec["module"].scan(enriched, variant_settings, rs_rating=rs_series, **scan_kwargs)
                if eval_date not in result.index:
                    continue
                candidate = result.loc[eval_date]
                if bool(candidate.get("match", False)):
                    if best_row is None or candidate.get("score", 0) > best_row.get("score", 0):
                        best_row = candidate
                        best_variant_settings = variant_settings

            if best_row is not None:
                record = {
                    "symbol": symbol,
                    "setup": spec["label"],
                    "side": spec["side"],
                    "date": eval_date,
                    "close": enriched.loc[eval_date, "close"],
                    "adr_pct": enriched.loc[eval_date, "adr_pct_20"],
                }
                record.update({k: v for k, v in best_row.items() if k != "match"})
                # Carry the actual window-length values that produced this
                # match (may differ from `settings`' defaults under
                # auto_detect) so charting can shade the real size used,
                # not whatever the Designer tab happens to be set to.
                for overrides in PATTERN_SIZE_VARIANTS.get(setup_name, []):
                    for field in overrides:
                        record[field] = getattr(best_variant_settings, field, None)
                # Always attach RS rating regardless of whether this setup's own
                # match condition uses it, so results can be prioritized by
                # momentum/leadership across every pattern, not just Breakout.
                record["rs_rating"] = rs_series.get(eval_date) if rs_series is not None else float("nan")
                rows.append(record)

    if not rows:
        return pd.DataFrame(columns=["symbol", "setup", "date", "close", "score"])

    result_df = pd.DataFrame(rows)

    # Universal fundamentals attachment -- every setup's results get these
    # columns (not just Episodic Pivot's own internal scoring above),
    # mirroring how rs_rating is already attached to every row regardless
    # of whether that setup's match condition uses it.
    if fundamentals:
        from fundamentals_classifier import classify_lynch, days_to_earnings, LYNCH_CATEGORIES

        result_df["revenue_growth_pct"] = result_df["symbol"].map(
            lambda s: fundamentals.get(s, {}).get("revenue_growth_pct")
        )
        result_df["eps_growth_pct"] = result_df["symbol"].map(
            lambda s: fundamentals.get(s, {}).get("eps_growth_pct")
        )
        result_df["market_cap"] = result_df["symbol"].map(lambda s: fundamentals.get(s, {}).get("market_cap"))
        result_df["company_name"] = result_df["symbol"].map(lambda s: fundamentals.get(s, {}).get("company_name"))
        result_df["sector"] = result_df["symbol"].map(lambda s: fundamentals.get(s, {}).get("sector"))
        result_df["lynch_category"] = result_df["symbol"].map(
            lambda s: LYNCH_CATEGORIES[classify_lynch(fundamentals.get(s, {}))]
        )
    if earnings_dates:
        from fundamentals_classifier import days_to_earnings, filter_earnings_avoidance

        result_df["days_to_earnings"] = result_df.apply(
            lambda r: days_to_earnings(r["symbol"], earnings_dates, r["date"]), axis=1
        )
        if avoid_earnings_window:
            result_df = filter_earnings_avoidance(result_df, earnings_dates, as_of_ts, avoid_earnings_days)

    if min_adr_pct > 0:
        result_df = result_df[result_df["adr_pct"] >= min_adr_pct]

    if result_df.empty:
        return pd.DataFrame(columns=["symbol", "setup", "date", "close", "score"])

    # Prioritize high-momentum (high RS rating) leaders first -- O'Neil and
    # Qullamaggie alike emphasize that these patterns matter most in market
    # leaders, not laggards that happen to technically match. Score is the
    # tie-breaker within the same RS tier.
    return (
        result_df
        .sort_values(["rs_rating", "score"], ascending=[False, False], na_position="last")
        .reset_index(drop=True)
    )


# Setups with a meaningful "sitting in a valid base, hasn't triggered yet"
# state, mapped to the diagnostic column holding their resistance/pivot
# level. Excludes episodic_pivot (a gap event, not something you sit inside)
# and parabolic_short (an ongoing extension, not a level you approach).
APPROACHING_PIVOT_SETUPS = {
    "breakout": "resistance",
    "cup_with_handle": "pivot",
    "double_bottom": "middle_peak",
    "flat_base": "base_high",
    "ascending_base": "resistance",
    "high_tight_flag": "resistance",
}


def find_approaching_pivot(
    settings: Settings,
    history: dict,
    as_of: Optional[str] = None,
    setup_names: Optional[list] = None,
    max_distance_pct: float = 8.0,
    auto_detect: bool = True,
) -> pd.DataFrame:
    """Stocks sitting inside a *valid* base/pattern shape that haven't
    triggered the actual breakout yet -- within `max_distance_pct`% of the
    resistance/pivot level (either just under it, or just over it without
    volume having confirmed yet). Complements run_scan(), which only shows
    the exact day a stock already broke out."""
    candidates = setup_names or enabled_setup_names(settings)
    setup_names = [s for s in candidates if s in APPROACHING_PIVOT_SETUPS]
    if not setup_names or not history:
        return pd.DataFrame(columns=["symbol", "setup", "date", "close", "pivot", "pct_to_pivot"])

    close_panel = build_close_panel(history)
    if close_panel.empty:
        return pd.DataFrame(columns=["symbol", "setup", "date", "close", "pivot", "pct_to_pivot"])

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
        close_price = enriched.loc[eval_date, "close"]

        for setup_name in setup_names:
            spec = SETUP_REGISTRY[setup_name]
            pivot_field = APPROACHING_PIVOT_SETUPS[setup_name]
            variants = (
                build_size_variants(settings, setup_name)
                if auto_detect
                else [getattr(settings, spec["settings_attr"])]
            )

            best = None
            for variant_settings in variants:
                result = spec["module"].scan(enriched, variant_settings, rs_rating=rs_series)
                if eval_date not in result.index:
                    continue
                candidate = result.loc[eval_date]
                if not bool(candidate.get("shape_match", False)):
                    continue
                if bool(candidate.get("match", False)):
                    continue  # already broken out -- that's run_scan's job, not this
                pivot_value = candidate.get(pivot_field)
                if pivot_value is None or pd.isna(pivot_value) or pivot_value <= 0:
                    continue
                pct_to_pivot = (pivot_value - close_price) / close_price * 100.0
                if abs(pct_to_pivot) > max_distance_pct:
                    continue
                if best is None or abs(pct_to_pivot) < abs(best["pct_to_pivot"]):
                    best = {"pct_to_pivot": pct_to_pivot, "pivot": pivot_value, "row": candidate, "variant": variant_settings}

            if best is not None:
                record = {
                    "symbol": symbol,
                    "setup": spec["label"],
                    "side": spec["side"],
                    "date": eval_date,
                    "close": close_price,
                    "pivot": best["pivot"],
                    "pct_to_pivot": best["pct_to_pivot"],
                }
                record["rs_rating"] = rs_series.get(eval_date) if rs_series is not None else float("nan")
                for overrides in PATTERN_SIZE_VARIANTS.get(setup_name, []):
                    for field in overrides:
                        record[field] = getattr(best["variant"], field, None)
                rows.append(record)

    if not rows:
        return pd.DataFrame(columns=["symbol", "setup", "date", "close", "pivot", "pct_to_pivot"])

    return (
        pd.DataFrame(rows)
        .sort_values(by="pct_to_pivot", key=lambda s: s.abs(), ascending=True)
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
