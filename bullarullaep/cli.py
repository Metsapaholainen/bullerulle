"""Headless CLI for BullaRullaEP, for scheduled/non-interactive runs.

Must be run as a module (`-m`), not as a script -- this file imports
sibling top-level packages (backtest/, data/, indicators.py, ...) from the
parent repo root, which only ends up on sys.path when Python resolves the
package via `-m` from the repo root, not when the file is executed directly.

    python -m bullarullaep.cli scan-early
    python -m bullarullaep.cli scan-early --symbols NVDA,SMCI,CVNA --dry-run
    python -m bullarullaep.cli scan-confirm
    python -m bullarullaep.cli scan-confirm --dry-run
    python -m bullarullaep.cli backtest --start 2022-01-01 --end 2026-08-01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from backtest.engine import run_backtest
from backtest.stats import compute_stats, stats_summary_text
from data.cache import get_history_bulk
from data.fmp_client import FMPClient, FMPError
from data.universe import build_universe
from indicators import add_core_indicators

from bullarullaep import detector
from bullarullaep.point_in_time_fundamentals import fundamentals_as_of, get_quarterly_history_bulk
from bullarullaep.scan_confirm import run_scan_confirm
from bullarullaep.scan_early import history_lookback_days, run_scan_early
from bullarullaep.settings import EPConfig


def _load_config(path):
    return EPConfig.load(Path(path)) if path else EPConfig.load()


def _client_or_exit() -> FMPClient:
    try:
        return FMPClient()
    except FMPError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def _progress_printer():
    def on_progress(i, n, symbol):
        print(f"\r{i}/{n}: {symbol}    ", end="", file=sys.stderr)

    return on_progress


def cmd_scan_early(args):
    config = _load_config(args.config)
    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    client = _client_or_exit()
    df = run_scan_early(config=config, client=client, symbols=symbols, as_of=args.as_of, dry_run=args.dry_run)
    if not df.empty and args.output:
        df.to_csv(args.output, index=False)
        print(f"Wrote {len(df)} candidates to {args.output}")


def cmd_scan_confirm(args):
    config = _load_config(args.config)
    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    client = _client_or_exit()
    df = run_scan_confirm(config=config, client=client, symbols=symbols, as_of=args.as_of, dry_run=args.dry_run)
    if not df.empty and args.output:
        df.to_csv(args.output, index=False)
        print(f"Wrote {len(df)} orders to {args.output}")


def _add_point_in_time_columns(df: pd.DataFrame, quarters: list, catalyst_window_days: int) -> pd.DataFrame:
    """The backtest-side half of the point-in-time-correctness fix: adds
    revenue_growth_pct/eps_growth_pct/eps_beat_pct/is_earnings_day to EVERY
    historical bar, each computed as of THAT bar's own date via
    fundamentals_as_of() against the same cached quarterly history -- pure
    local computation, no extra API calls (quarters is already fully
    fetched once per symbol). This is what makes detector.scan() causally
    valid across a whole backtest instead of just for "today"."""
    df = df.copy()
    report_dates = sorted(pd.Timestamp(q["report_date"]) for q in quarters if q.get("report_date"))
    revenue_growth, eps_growth, eps_beat, earnings_flag = [], [], [], []
    for dt in df.index:
        f = fundamentals_as_of(quarters, dt)
        revenue_growth.append(f["revenue_growth_pct"])
        eps_growth.append(f["eps_growth_pct"])
        eps_beat.append(f["eps_beat_pct"])
        recent = any(0 <= (dt - rd).days <= catalyst_window_days for rd in report_dates if rd <= dt)
        earnings_flag.append(recent)
    df["revenue_growth_pct"] = revenue_growth
    df["eps_growth_pct"] = eps_growth
    df["eps_beat_pct"] = eps_beat
    df["is_earnings_day"] = earnings_flag
    return df


def scan_signals_over_history(config: EPConfig, history: dict, quarters_by_symbol: dict) -> dict:
    """EP's analogue of scanner.py::scan_signals_over_history -- runs the
    detector across every historical bar (not just the latest) for every
    symbol, point-in-time-correct, for backtesting. No catalyst-classifier
    or social-sentiment bonus is applied here (see detector.py's docstring:
    both are inherently live-only, no historical news/sentiment archive
    exists to backtest them against honestly) -- the backtest measures the
    EOD hard gates + point-in-time growth/EPS-beat scoring only, same
    disclosed scope as the parent project's own episodic_pivot.py."""
    out = {}
    for symbol, raw_df in history.items():
        if raw_df.empty:
            continue
        enriched = add_core_indicators(raw_df)
        pit = _add_point_in_time_columns(enriched, quarters_by_symbol.get(symbol, []), config.ep.catalyst_earnings_window_days)
        result = detector.scan(pit, config.ep).join(enriched[["open", "high", "low", "close", "volume"]])
        out[symbol] = result
    return out


def cmd_backtest(args):
    config = _load_config(args.config)
    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    client = _client_or_exit()

    universe = symbols or build_universe(client, config.universe)
    lookback_days = int(history_lookback_days(config.ep) * 1.6)
    start_fetch = (pd.Timestamp(args.start) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    history = get_history_bulk(client, universe, start_fetch, args.end, on_progress=_progress_printer())
    print(file=sys.stderr)

    quarters_by_symbol = get_quarterly_history_bulk(client, list(history.keys()), on_progress=_progress_printer())
    print(file=sys.stderr)

    start_ts, end_ts = pd.Timestamp(args.start), pd.Timestamp(args.end)
    history = {sym: df.loc[(df.index >= start_ts) & (df.index <= end_ts)] for sym, df in history.items()}
    history = {sym: df for sym, df in history.items() if not df.empty}

    signals = scan_signals_over_history(config, history, quarters_by_symbol)
    result = run_backtest(config.backtest, signals, side="long", setup_name="episodic_pivot")
    stats = compute_stats(result)
    print(stats_summary_text(stats))

    if args.output:
        result.trades_df().to_csv(args.output, index=False)
        print(f"Wrote trade log to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="BullaRullaEP: episodic-pivot-only scanner/backtester CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_early = sub.add_parser("scan-early", help="Pass 1: overnight catalyst watchlist")
    p_early.add_argument("--config", help="Path to a saved EP config YAML (default: bullarullaep/config/default_ep_settings.yaml)")
    p_early.add_argument("--symbols", help="Comma-separated symbols instead of the full screener universe")
    p_early.add_argument("--as-of", dest="as_of", help="Scan date (YYYY-MM-DD), default: today")
    p_early.add_argument("--output", help="Optional CSV path for the candidate list")
    p_early.add_argument("--dry-run", action="store_true", help="Print what would be alerted without sending")
    p_early.set_defaults(func=cmd_scan_early)

    p_confirm = sub.add_parser("scan-confirm", help="Pass 2: premarket confirmation + actionable alert")
    p_confirm.add_argument("--config", help="Path to a saved EP config YAML")
    p_confirm.add_argument("--symbols", help="Comma-separated symbols instead of pass-1's saved candidates")
    p_confirm.add_argument("--as-of", dest="as_of", help="Scan date (YYYY-MM-DD), default: today")
    p_confirm.add_argument("--output", help="Optional CSV path for the confirmed order tickets")
    p_confirm.add_argument("--dry-run", action="store_true", help="Print what would be alerted without sending")
    p_confirm.set_defaults(func=cmd_scan_confirm)

    p_bt = sub.add_parser("backtest", help="Run a historical backtest of the EP detector")
    p_bt.add_argument("--config", help="Path to a saved EP config YAML")
    p_bt.add_argument("--symbols", help="Comma-separated symbols instead of the full screener universe")
    p_bt.add_argument("--start", required=True)
    p_bt.add_argument("--end", required=True)
    p_bt.add_argument("--output", help="Optional trade-log CSV path")
    p_bt.set_defaults(func=cmd_backtest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
