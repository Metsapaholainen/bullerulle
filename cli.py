"""Headless CLI, for scheduled/non-interactive runs.

    python cli.py scan
    python cli.py scan --preset aggressive --symbols NVDA,SMCI,CVNA
    python cli.py backtest --setup breakout --start 2018-01-01 --end 2024-12-31
"""
from __future__ import annotations

import argparse
import sys
from datetime import date

import pandas as pd

import presets
from backtest.engine import run_backtest
from backtest.stats import compute_stats, stats_summary_text
from data.fmp_client import FMPClient, FMPError
from scanner import load_scan_universe_history, run_scan, scan_signals_over_history
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

    signals = scan_signals_over_history(settings, history, args.setup)
    result = run_backtest(settings.backtest, signals, side=SETUP_REGISTRY[args.setup]["side"])
    stats = compute_stats(result)
    print(stats_summary_text(stats))

    if args.output:
        result.trades_df().to_csv(args.output, index=False)
        print(f"Wrote trade log to {args.output}")


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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
