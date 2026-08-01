"""Paper Trading journal: log a hand-picked (symbol, decision_date) "buy-in"
and simulate it through the exact same trade-management engine (stop-loss,
sell-half at an R-target or day-limit, then a trailing moving-average exit
for the remainder) that every automated setup in this app already uses --
reusing backtest.engine directly rather than a bespoke simulator.

Uses its OWN fixed BacktestSettings, independent of whatever's currently
tuned in the Designer tab's "Backtest mechanics" section, so a logged
trade's shown outcome never silently shifts just because you're
experimenting with scan/backtest settings elsewhere.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from backtest.engine import run_backtest, BacktestResult
from indicators import add_core_indicators
from settings import BacktestSettings

STORE_DIR = Path(__file__).parent / "paper_trading_store"
TRADES_PATH = STORE_DIR / "trades.yaml"
RULES_PATH = STORE_DIR / "rules.yaml"


def load_trades(path: Optional[Path] = None) -> list:
    path = path or TRADES_PATH
    if not path.exists():
        return []
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    return data.get("trades") or []


def save_trades(entries: list, path: Optional[Path] = None) -> Path:
    path = path or TRADES_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump({"trades": entries}, f, sort_keys=False)
    return path


def load_rules(path: Optional[Path] = None) -> BacktestSettings:
    path = path or RULES_PATH
    if not path.exists():
        return BacktestSettings()
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    defaults = asdict(BacktestSettings())
    defaults.update(data)
    return BacktestSettings(**defaults)


def save_rules(rules: BacktestSettings, path: Optional[Path] = None) -> Path:
    path = path or RULES_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(asdict(rules), f, sort_keys=False)
    return path


def build_signals(history: dict, trades: list):
    """Groups logged entries by symbol and builds the {symbol: enriched_df}
    shape backtest.engine.run_backtest() expects (bool `match` column plus
    OHLCV) -- the exact shape scanner.scan_signals_over_history() already
    produces for detector-driven signals, built here from hand-picked dates
    instead. Returns (signals_dict, skipped) where `skipped` lists entries
    that couldn't even be placed (symbol not loaded, or decision_date
    outside the loaded history's range) so the caller can report them
    instead of having them silently vanish."""
    by_symbol: dict = {}
    skipped = []
    for entry in trades:
        symbol = entry["symbol"]
        decision_date = pd.Timestamp(entry["decision_date"])
        if symbol not in history or history[symbol].empty:
            skipped.append({**entry, "reason": "Symbol not loaded -- load it from the sidebar first."})
            continue
        if symbol not in by_symbol:
            enriched = add_core_indicators(history[symbol].copy())
            enriched["match"] = False
            by_symbol[symbol] = enriched
        df = by_symbol[symbol]
        if decision_date not in df.index:
            skipped.append({**entry, "reason": "Decision date is outside the currently loaded history."})
            continue
        df.loc[decision_date, "match"] = True
    return by_symbol, skipped


def simulate(history: dict, trades: list, rules: BacktestSettings):
    """Runs every logged entry through ONE run_backtest() call, so they all
    share a single simulated account (cash, equity, max_positions cap) --
    "one hypothetical account" rather than isolated per-trade sandboxes."""
    signals, skipped = build_signals(history, trades)
    if not signals:
        return BacktestResult(trades=[], equity_curve=pd.Series(dtype=float)), skipped
    result = run_backtest(rules, signals, side="long")
    return result, skipped


def join_results(trades: list, result: BacktestResult, skipped: list) -> pd.DataFrame:
    """One row per logged entry: matched to its realized Trade by
    (symbol, signal_date) where the engine actually produced one, otherwise
    an explicit not-simulated reason -- so a logged entry never just
    disappears without explanation."""
    skipped_reasons = {(s["symbol"], s["decision_date"]): s["reason"] for s in skipped}
    trades_by_key = {(t.symbol, t.signal_date.strftime("%Y-%m-%d")): t for t in result.trades}

    rows = []
    for entry in trades:
        key = (entry["symbol"], entry["decision_date"])
        trade = trades_by_key.get(key)
        base = {
            "symbol": entry["symbol"],
            "decision_date": entry["decision_date"],
            "setup_tag": entry.get("setup_tag") or "",
            "notes": entry.get("notes") or "",
        }
        if trade is not None:
            still_open_at_end = trade.exit_reason == "end_of_data"
            r = trade.r_multiple or 0
            rows.append(
                {
                    **base,
                    "entry_date": trade.entry_date,
                    "entry_price": trade.entry_price,
                    "partial_date": trade.partial_date,
                    "partial_price": trade.partial_price,
                    "exit_date": trade.exit_date,
                    "exit_price": trade.exit_price,
                    "exit_reason": "Still open (as of last bar)" if still_open_at_end else trade.exit_reason,
                    "r_multiple": trade.r_multiple,
                    "pnl": trade.pnl,
                    "status": "Open" if still_open_at_end else ("Win" if r > 0 else ("Loss" if r < 0 else "Breakeven")),
                }
            )
        else:
            reason = skipped_reasons.get(
                key, "Not simulated -- likely produced no valid trade (insufficient ADR history, or filtered)."
            )
            rows.append(
                {
                    **base,
                    "entry_date": None, "entry_price": None, "partial_date": None, "partial_price": None,
                    "exit_date": None, "exit_price": None, "exit_reason": reason,
                    "r_multiple": None, "pnl": None, "status": "Not simulated",
                }
            )
    return pd.DataFrame(rows)
