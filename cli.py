"""Headless CLI, for scheduled/non-interactive runs.

    python cli.py scan
    python cli.py scan --preset aggressive --symbols NVDA,SMCI,CVNA
    python cli.py backtest --setup breakout --start 2018-01-01 --end 2024-12-31
    python cli.py sell-alerts
    python cli.py sell-alerts --dry-run
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

import presets
from backtest.engine import run_backtest
from backtest.stats import compute_stats, stats_summary_text
from data.cache import get_history_bulk
from data.fmp_client import FMPClient, FMPError
from notifications import send_sell_alerts
from scanner import (
    DEFAULT_BENCHMARK_SYMBOL,
    compute_regime_series,
    ensure_benchmark_loaded,
    load_scan_universe_history,
    run_scan,
    scan_signals_over_history,
)
from sell_alerts import check_watchlist, load_watchlist
from setups import SETUP_REGISTRY


def _load_settings(preset_name):
    return presets.load_preset(preset_name) if preset_name else presets.load_default()


def _progress_printer():
    def on_progress(i, n, symbol):
        print(f"\rLoading {i}/{n}: {symbol}    ", end="", file=sys.stderr)

    return on_progress


def cmd_scan(args):
    settings = _load_settings(args.preset)
    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    try:
        client = FMPClient()
    except FMPError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    history = load_scan_universe_history(
        settings, as_of=args.as_of, client=client, symbols=symbols, on_progress=_progress_printer()
    )
    print(file=sys.stderr)

    results = run_scan(settings, history, as_of=args.as_of)
    if results.empty:
        print("No matches.")
        return

    out_path = args.output or f"scan_{args.as_of or date.today().isoformat()}.csv"
    results.to_csv(out_path, index=False)
    print(f"Wrote {len(results)} matches to {out_path}")
    print(results.to_string(index=False))


def cmd_backtest(args):
    settings = _load_settings(args.preset)
    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    try:
        client = FMPClient()
    except FMPError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    history = load_scan_universe_history(
        settings, as_of=args.end, client=client, symbols=symbols, on_progress=_progress_printer()
    )
    print(file=sys.stderr)

    start_ts, end_ts = pd.Timestamp(args.start), pd.Timestamp(args.end)
    history = {
        sym: df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
        for sym, df in history.items()
    }
    history = {sym: df for sym, df in history.items() if not df.empty}

    regime_multiplier = None
    if settings.backtest.regime_filter_enabled:
        benchmark_df = ensure_benchmark_loaded(history, client, args.start, args.end)
        regime = compute_regime_series(benchmark_df, settings.backtest)
        if not regime.empty:
            regime_multiplier = regime["size_multiplier"]

    signals = scan_signals_over_history(settings, history, args.setup)
    result = run_backtest(
        settings.backtest, signals, side=SETUP_REGISTRY[args.setup]["side"], regime_multiplier=regime_multiplier
    )
    stats = compute_stats(result)
    print(stats_summary_text(stats))

    if args.output:
        result.trades_df().to_csv(args.output, index=False)
        print(f"Wrote trade log to {args.output}")


def cmd_sell_alerts(args):
    """Checks the sell watchlist against its configured moving average and
    pushes an ntfy.sh alert for anything currently closing below it. Designed
    to be invoked once a day by an external scheduler (GitHub Actions, cron,
    Task Scheduler) so alerts fire even when nobody has the app open --
    Streamlit Community Cloud has no background/cron capability of its own."""
    watchlist = load_watchlist(Path(args.watchlist) if args.watchlist else None)
    if not watchlist:
        print("Watchlist is empty -- add symbols via the app's Sell Alerts tab or edit the YAML directly.")
        return

    try:
        client = FMPClient()
    except FMPError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    symbols = [e["symbol"] for e in watchlist]
    end = date.today()
    start = end - timedelta(days=90)
    history = get_history_bulk(client, symbols, start.isoformat(), end.isoformat(), on_progress=_progress_printer())
    print(file=sys.stderr)

    results = check_watchlist(history, watchlist)
    print(results.to_string(index=False))

    below = results[results["is_below"] == True]  # noqa: E712 (explicit True, not just truthy, since is_below can be None)
    if below.empty:
        print("\nNothing below its moving average today.")
        return

    if args.dry_run:
        print(f"\n[dry-run] Would alert for: {', '.join(below['symbol'])}")
        return

    alerted = send_sell_alerts(below)
    print(f"\nSent sell alerts for: {', '.join(alerted)}")


def main():
    parser = argparse.ArgumentParser(description="Qullamaggie-style scanner/backtester CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="Run a scan and write a ranked CSV")
    p_scan.add_argument("--preset", help="Named preset to use (default: config/default_settings.yaml)")
    p_scan.add_argument("--symbols", help="Comma-separated symbols instead of the full screener universe")
    p_scan.add_argument("--as-of", dest="as_of", help="Scan date (YYYY-MM-DD), default: today")
    p_scan.add_argument("--output", help="Output CSV path")
    p_scan.set_defaults(func=cmd_scan)

    p_bt = sub.add_parser("backtest", help="Run a historical backtest for one setup")
    p_bt.add_argument("--setup", required=True, choices=list(SETUP_REGISTRY.keys()))
    p_bt.add_argument("--preset")
    p_bt.add_argument("--symbols")
    p_bt.add_argument("--start", required=True)
    p_bt.add_argument("--end", required=True)
    p_bt.add_argument("--output", help="Optional trade-log CSV path")
    p_bt.set_defaults(func=cmd_backtest)

    p_sell = sub.add_parser("sell-alerts", help="Check the sell watchlist and push an ntfy.sh alert if needed")
    p_sell.add_argument("--watchlist", help="Path to the watchlist YAML (default: config/sell_watchlist.yaml)")
    p_sell.add_argument("--dry-run", action="store_true", help="Print what would be alerted without sending")
    p_sell.set_defaults(func=cmd_sell_alerts)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
