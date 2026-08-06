"""Pass 1 of BullaRullaEP's daily schedule -- the early/overnight catalyst
watchlist. Runs the EOD hard gates against yesterday's confirmed close
(today's bar doesn't exist yet), enriches the resulting shortlist with
point-in-time growth/EPS-beat, a backward-looking earnings-catalyst check,
recent news headlines, and an LLM catalyst classification, then pushes an
informational "heads up" ntfy alert -- NOT yet actionable, since premarket
gap/volume isn't confirmed until pass 2 (scan_confirm.py). Persists the
candidate list via watchlist_store so pass 2 can read it without re-running
the universe scan from scratch.

Meant to run once per weekday morning via
.github/workflows/ep_scan_early.yml (~9:00 Finnish time); also runnable by
hand via `python bullarullaep/cli.py scan-early`.
"""
from __future__ import annotations

import sys
from datetime import date
from typing import Optional

import pandas as pd

from data.cache import get_history_bulk
from data.fmp_client import FMPClient, FMPError
from data.fundamentals_cache import get_earnings_dates_window
from data.universe import build_universe
from fundamentals_classifier import days_since_earnings

from bullarullaep import alerts, detector, watchlist_store
from bullarullaep.catalyst_classifier import classify_catalyst
from bullarullaep.point_in_time_fundamentals import fundamentals_as_of, get_quarterly_history_bulk
from bullarullaep.settings import EPConfig


def history_lookback_days(ep_settings) -> int:
    """EP only needs enough history for its own longest window (the
    neglect-period check) plus a buffer -- doesn't need the parent
    project's multi-setup scanner.history_lookback_days()."""
    return max(
        ep_settings.lookback_base_days, ep_settings.quiet_base_lookback_days, ep_settings.prior_ep_lookback_days
    ) + 30


def _progress_printer():
    def on_progress(i, n, symbol):
        print(f"\r{i}/{n}: {symbol}    ", end="", file=sys.stderr)

    return on_progress


def run_scan_early(
    config: Optional[EPConfig] = None,
    client: Optional[FMPClient] = None,
    symbols: Optional[list] = None,
    as_of: Optional[str] = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    config = config or EPConfig.load()
    client = client or FMPClient()
    as_of = as_of or date.today().isoformat()

    universe = symbols or build_universe(client, config.universe)
    lookback_days = int(history_lookback_days(config.ep) * 1.6)  # pad for weekends/holidays
    start = (pd.Timestamp(as_of) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    history = get_history_bulk(client, universe, start, as_of, on_progress=_progress_printer())
    print(file=sys.stderr)

    # First pass: cheap, price/volume-only shape gates, no fundamentals/news
    # fetched yet -- narrows the (potentially thousands-wide) universe down
    # to a shortlist before spending any fundamentals/news/LLM budget. The
    # shape-pass `out` is kept per symbol so gap_pct etc. don't need
    # recomputing in the enrichment pass below.
    shape_out_by_symbol = {}
    for symbol, df in history.items():
        if len(df) < 30:
            continue
        out = detector.scan(df, config.ep)
        if bool(out["match"].iloc[-1]):
            shape_out_by_symbol[symbol] = out

    if not shape_out_by_symbol:
        print("No shape-gate candidates today.", file=sys.stderr)
        watchlist_store.save_early_candidates(as_of, [])
        return pd.DataFrame()

    shape_candidates = list(shape_out_by_symbol.keys())

    # Point-in-time fundamentals + backward-looking earnings confirmation +
    # news headlines + catalyst classification -- only for the shortlist.
    quarters_by_symbol = get_quarterly_history_bulk(client, shape_candidates)
    earnings_window = get_earnings_dates_window(
        client, shape_candidates, as_of,
        lookback_days=config.ep.catalyst_earnings_window_days, lookahead_days=1,
    )
    try:
        news_rows = client.stock_news(symbols=shape_candidates, limit=200)
    except FMPError:
        news_rows = []
    news_by_symbol: dict = {}
    for row in news_rows:
        sym, title = row.get("symbol"), row.get("title")
        if sym and title:
            news_by_symbol.setdefault(sym, []).append(title)

    rows = []
    for symbol in shape_candidates:
        df = history[symbol]
        fundamentals = fundamentals_as_of(quarters_by_symbol.get(symbol, []), as_of)
        days_since = days_since_earnings(symbol, earnings_window, as_of)
        is_earnings_day = days_since is not None and 0 <= days_since <= config.ep.catalyst_earnings_window_days

        catalyst = None
        if config.ep.enable_catalyst_classifier:
            headlines = news_by_symbol.get(symbol, [])
            gap_pct = shape_out_by_symbol[symbol]["gap_pct"].iloc[-1]
            catalyst = classify_catalyst(
                symbol, headlines,
                gap_pct=float(gap_pct) if pd.notna(gap_pct) else None,
                is_earnings_day=is_earnings_day,
            )

        out = detector.scan(
            df, config.ep, fundamentals=fundamentals, is_earnings_day=is_earnings_day, catalyst=catalyst
        )
        last = out.iloc[-1]
        if not bool(last["match"]):
            continue  # e.g. the growth-floor gate can still reject once real fundamentals arrive
        row = last.to_dict()
        row["symbol"] = symbol
        row["as_of"] = as_of
        rows.append(row)

    candidates_df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()
    watchlist_store.save_early_candidates(
        as_of, candidates_df.to_dict("records") if not candidates_df.empty else []
    )

    if not candidates_df.empty:
        if dry_run:
            print(f"[dry-run] Would alert {len(candidates_df)} candidate(s):", file=sys.stderr)
            cols = ["symbol", "gap_pct", "volume_ratio", "growth_tier", "catalyst_type", "score"]
            print(candidates_df[[c for c in cols if c in candidates_df.columns]].to_string(index=False))
        else:
            alerts.send_early_watchlist_alert(candidates_df)
    else:
        print("No candidates cleared the full pass-1 gates today.", file=sys.stderr)

    return candidates_df
