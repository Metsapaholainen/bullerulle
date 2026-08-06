"""Pass 2 of BullaRullaEP's daily schedule -- the pre-open premarket
confirmation. Re-checks pass-1's candidates PLUS any symbol newly reporting
earnings before today's open (pass 1 ran ~7 hours earlier, before most
before-market-open reports are out -- this is how Qullamaggie's favorite
case, a same-day BMO earnings gap, still gets caught).

Pulls a live `/quote` for each candidate and merges it into cached history
as a synthetic "today so far" bar, exactly the way `data/cache.py::
apply_live_quotes()` already does elsewhere in this codebase (open/high/low
approx premarket price/dayHigh/dayLow, volume approx current cumulative
volume) -- documented honestly as an approximation, not lab-grade premarket
OHLCV, since FMP's `/quote` is a single live snapshot, not a full premarket
session history.

Feeds that synthetic bar into the UNCHANGED
`backtest/engine.py::resolve_entry_and_stop()` to get today's stop-buy
trigger (above the premarket high -- this is what mechanically implements
"wait for it to move above the early high," enforced by the order type
itself, not a third intraday poll), limit cap, and initial stop (naturally
"low of the gap day so far"). Sizes the position via
`stop_calculator.compute_stop_and_size()` and sends the final, actionable
alert.

Meant to run once per weekday, shortly before the US open, via
.github/workflows/ep_scan_confirm.yml (~9:15 ET / ~16:15 Finnish summer);
also runnable by hand via `python bullarullaep/cli.py scan-confirm`.
"""
from __future__ import annotations

import sys
from datetime import date
from typing import Optional

import pandas as pd

from backtest.engine import resolve_entry_and_stop
from data.cache import apply_live_quotes, get_history_bulk
from data.fmp_client import FMPClient, FMPError
from data.fundamentals_cache import get_earnings_dates_window
from fundamentals_classifier import days_since_earnings
from indicators import add_adr_pct

from bullarullaep import alerts, detector, watchlist_store
from bullarullaep.catalyst_classifier import classify_catalyst
from bullarullaep.point_in_time_fundamentals import fundamentals_as_of, get_quarterly_history_bulk
from bullarullaep.scan_early import history_lookback_days
from bullarullaep.settings import EPConfig
from stop_calculator import compute_stop_and_size


def _progress_printer():
    def on_progress(i, n, symbol):
        print(f"\r{i}/{n}: {symbol}    ", end="", file=sys.stderr)

    return on_progress


def _todays_earnings_symbols(client: FMPClient, as_of: str) -> list:
    """Every symbol reporting earnings on `as_of` market-wide -- catches a
    same-day before-market-open report pass 1 (run hours earlier) couldn't
    have seen yet. No universe filtering here; the shape gates below (price,
    liquidity via the cached-history requirement, growth floor, etc.) do
    that filtering downstream, same as any other candidate."""
    try:
        rows = client.earnings_calendar(from_date=as_of, to_date=as_of)
    except FMPError:
        return []
    return sorted({r["symbol"] for r in rows if r.get("symbol")})


def run_scan_confirm(
    config: Optional[EPConfig] = None,
    client: Optional[FMPClient] = None,
    symbols: Optional[list] = None,
    as_of: Optional[str] = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    config = config or EPConfig.load()
    client = client or FMPClient()
    as_of = as_of or date.today().isoformat()

    if symbols:
        candidate_symbols = list(dict.fromkeys(symbols))
    else:
        pass1 = [c["symbol"] for c in watchlist_store.load_early_candidates(as_of) if c.get("symbol")]
        today_earnings = _todays_earnings_symbols(client, as_of)
        candidate_symbols = list(dict.fromkeys(pass1 + today_earnings))

    if not candidate_symbols:
        print("No pass-1 candidates and no symbols reporting earnings today -- nothing to confirm.", file=sys.stderr)
        return pd.DataFrame()

    lookback_days = int(history_lookback_days(config.ep) * 1.6)
    start = (pd.Timestamp(as_of) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    # end=as_of - 1 bar: we want cached history UP TO but not including today
    # (today's real bar doesn't exist as an EOD bar yet -- it gets built
    # synthetically below from the live quote).
    prior_end = (pd.Timestamp(as_of) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    history = get_history_bulk(client, candidate_symbols, start, prior_end, on_progress=_progress_printer())
    print(file=sys.stderr)

    try:
        quotes = client.get_quotes(list(history.keys()))
    except FMPError:
        quotes = []
    history = apply_live_quotes(history, quotes)
    quoted_symbols = {q.get("symbol") for q in quotes if q.get("symbol")}

    quarters_by_symbol = get_quarterly_history_bulk(client, list(history.keys()))
    earnings_window = get_earnings_dates_window(
        client, list(history.keys()), as_of,
        lookback_days=config.ep.catalyst_earnings_window_days, lookahead_days=1,
    )
    try:
        news_rows = client.stock_news(symbols=list(history.keys()), limit=200)
    except FMPError:
        news_rows = []
    news_by_symbol: dict = {}
    for row in news_rows:
        sym, title = row.get("symbol"), row.get("title")
        if sym and title:
            news_by_symbol.setdefault(sym, []).append(title)

    orders = []
    for symbol, df in history.items():
        # Skip anything we didn't actually get a live premarket quote for --
        # without it there's no confirmed gap, only yesterday's stale close.
        if symbol not in quoted_symbols or len(df) < 30:
            continue

        df = add_adr_pct(df, lookback=20)
        df["adr_pct"] = df["adr_pct_20"]

        fundamentals = fundamentals_as_of(quarters_by_symbol.get(symbol, []), as_of)
        days_since = days_since_earnings(symbol, earnings_window, as_of)
        is_earnings_day = days_since is not None and 0 <= days_since <= config.ep.catalyst_earnings_window_days

        catalyst = None
        if config.ep.enable_catalyst_classifier:
            gap_pct_preview = detector.scan(df, config.ep)["gap_pct"].iloc[-1]
            catalyst = classify_catalyst(
                symbol, news_by_symbol.get(symbol, []),
                gap_pct=float(gap_pct_preview) if pd.notna(gap_pct_preview) else None,
                is_earnings_day=is_earnings_day,
            )

        out = detector.scan(
            df, config.ep, fundamentals=fundamentals, is_earnings_day=is_earnings_day, catalyst=catalyst
        )
        last = out.iloc[-1]
        if not bool(last["match"]):
            continue

        signal_bar = df.iloc[-1]
        plan = resolve_entry_and_stop(config.backtest, "long", signal_bar, entry_bar=None)
        if not plan.tradeable:
            print(f"{symbol}: not tradeable -- {plan.skip_reason}", file=sys.stderr)
            continue

        avg_volume = df["volume"].iloc[:-1].tail(50).mean() if len(df) > 1 else None
        sizing = compute_stop_and_size(
            entry_price=plan.entry_price,
            signal_low=signal_bar["low"],
            signal_high=signal_bar["high"],
            adr_pct=plan.adr_pct,
            side="long",
            stop_mode=config.backtest.stop_mode,
            stop_adr_multiple=config.backtest.stop_adr_multiple,
            risk_pct_per_trade=config.backtest.risk_pct_per_trade,
            equity=config.backtest.initial_capital,
            max_position_pct_of_equity=config.backtest.max_position_pct_of_equity,
            max_position_pct_of_avg_volume=config.backtest.max_position_pct_of_avg_volume,
            avg_volume=avg_volume,
            stop_exceeds_adr_action=config.backtest.stop_exceeds_adr_action,
            stop_distance=plan.stop_distance,
        )
        if sizing.get("error") or sizing.get("shares", 0) <= 0:
            print(f"{symbol}: sizing failed -- {sizing.get('error', 'zero shares')}", file=sys.stderr)
            continue

        row = last.to_dict()
        row.update({
            "symbol": symbol,
            "as_of": as_of,
            "trigger_price": plan.trigger_price,
            "limit_cap_price": plan.limit_cap_price,
            "entry_price_planned": plan.entry_price,
            "stop_price": sizing["stop_price"],
            "shares": sizing["shares"],
            "position_value": sizing["position_value"],
            "total_risk_dollars": sizing["total_risk_dollars"],
        })
        orders.append(row)

    orders_df = pd.DataFrame(orders).sort_values("score", ascending=False).reset_index(drop=True) if orders else pd.DataFrame()

    if not orders_df.empty:
        if dry_run:
            print(f"[dry-run] Would send {len(orders_df)} confirmed order(s):", file=sys.stderr)
            cols = ["symbol", "gap_pct", "trigger_price", "limit_cap_price", "stop_price", "shares", "score"]
            print(orders_df[[c for c in cols if c in orders_df.columns]].to_string(index=False))
        else:
            alerted = alerts.send_confirmed_alerts(orders_df)
            watchlist_store.append_alert_log(orders_df[orders_df["symbol"].isin(alerted)].to_dict("records"))
    else:
        print("No candidates confirmed a real premarket gap/volume today.", file=sys.stderr)

    return orders_df
