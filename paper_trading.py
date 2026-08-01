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

import github_store
from backtest.engine import run_backtest, BacktestResult
from indicators import add_core_indicators
from settings import BacktestSettings

GITHUB_TRADES_PATH = "paper_trading_store/trades.yaml"
GITHUB_RULES_PATH = "paper_trading_store/rules.yaml"


def load_trades(path: Optional[Path] = None) -> list:
    """With no explicit `path` (normal app usage), reads through
    github_store.py so the journal survives a Streamlit Cloud restart
    instead of living only on the container's ephemeral disk. Pass an
    explicit `path` to read a specific local file instead (used by tests)."""
    if path is not None:
        if not path.exists():
            return []
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
    else:
        raw = github_store.read_file(GITHUB_TRADES_PATH)
        data = (yaml.safe_load(raw) or {}) if raw else {}
    return data.get("trades") or []


def save_trades(entries: list, path: Optional[Path] = None) -> None:
    content = yaml.safe_dump({"trades": entries}, sort_keys=False)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return
    github_store.write_file(GITHUB_TRADES_PATH, content, "Update paper trading journal")


def load_rules(path: Optional[Path] = None) -> BacktestSettings:
    if path is not None:
        if not path.exists():
            return BacktestSettings()
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
    else:
        raw = github_store.read_file(GITHUB_RULES_PATH)
        if not raw:
            return BacktestSettings()
        data = yaml.safe_load(raw) or {}
    defaults = asdict(BacktestSettings())
    defaults.update(data)
    return BacktestSettings(**defaults)


def save_rules(rules: BacktestSettings, path: Optional[Path] = None) -> None:
    content = yaml.safe_dump(asdict(rules), sort_keys=False)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return
    github_store.write_file(GITHUB_RULES_PATH, content, "Update paper trading rules")


def build_signals(history: dict, trades: list, entry_delay_days: int = 1):
    """Groups logged entries by symbol and builds the {symbol: enriched_df}
    shape backtest.engine.run_backtest() expects (bool `match` column plus
    OHLCV) -- the exact shape scanner.scan_signals_over_history() already
    produces for detector-driven signals, built here from hand-picked dates
    instead. Returns (signals_dict, skipped) where `skipped` lists entries
    that couldn't even be placed, each tagged with a `status` of either
    "not_simulated" (symbol not loaded, or the decision date itself isn't a
    valid trading day in the loaded history) or "pending" (the decision date
    is fine, but there's no bar yet `entry_delay_days` days after it -- i.e.
    it hasn't "opened" yet, not a real problem) -- so a logged entry never
    just disappears without an accurate explanation."""
    by_symbol: dict = {}
    skipped = []
    for entry in trades:
        symbol = entry["symbol"]
        decision_date = pd.Timestamp(entry["decision_date"])
        if symbol not in history or history[symbol].empty:
            skipped.append(
                {**entry, "status": "not_simulated", "reason": "Symbol not loaded -- load it from the sidebar first."}
            )
            continue
        if symbol not in by_symbol:
            enriched = add_core_indicators(history[symbol].copy())
            enriched["match"] = False
            by_symbol[symbol] = enriched
        df = by_symbol[symbol]
        if decision_date not in df.index:
            skipped.append(
                {**entry, "status": "not_simulated", "reason": "Decision date is outside the currently loaded history."}
            )
            continue
        decision_loc = df.index.get_loc(decision_date)
        if decision_loc + entry_delay_days >= len(df.index):
            skipped.append(
                {
                    **entry,
                    "status": "pending",
                    "reason": "Pending -- reload data from the sidebar after the next trading day closes to see the entry fill.",
                }
            )
            continue
        df.loc[decision_date, "match"] = True
    return by_symbol, skipped


def simulate(history: dict, trades: list, rules: BacktestSettings):
    """Runs every logged entry through ONE run_backtest() call, so they all
    share a single simulated account (cash, equity, max_positions cap) --
    "one hypothetical account" rather than isolated per-trade sandboxes."""
    signals, skipped = build_signals(history, trades, entry_delay_days=rules.entry_delay_days)
    if not signals:
        return BacktestResult(trades=[], equity_curve=pd.Series(dtype=float)), skipped
    result = run_backtest(rules, signals, side="long")
    return result, skipped


# Fields frozen onto a trade entry once it genuinely closes (see
# freeze_closed_trades) -- kept as plain strings/floats so they round-trip
# through YAML cleanly.
REALIZED_FIELDS = [
    "entry_date", "entry_price", "partial_date", "partial_price",
    "exit_date", "exit_price", "exit_reason", "r_multiple", "pnl", "status",
]


def _to_native_float(value):
    """Trade fields (entry_price, r_multiple, etc.) arrive as numpy float64
    -- pandas .loc lookups return numpy scalars, not native Python floats.
    PyYAML's safe_dump can't represent numpy types at all, so anything
    frozen into a trade entry must be cast to a plain float first, or
    save_trades() raises a RepresenterError the next time this trade
    closes and gets frozen."""
    return None if value is None else float(value)


def freeze_closed_trades(trades: list, result: BacktestResult) -> tuple[list, bool]:
    """Once a logged trade actually closes (a real stop/target/trail exit --
    not just running out of loaded data), permanently record its realized
    outcome in the trade entry itself. Without this, a closed trade's
    result is only ever recomputed live from `history` -- if the loaded
    date window later moves past that trade (a rolling "History to load
    (years)" range, or the symbol just isn't reloaded), it would silently
    fall back to "Not simulated" and the win/loss would vanish from view.
    Freezing it here means the Paper Trading journal keeps showing what
    actually happened, indefinitely, regardless of what's currently loaded.

    Returns (possibly-updated trades list, whether anything changed) so the
    caller can skip an unnecessary save when nothing closed this run."""
    trades_by_key = {(t.symbol, t.signal_date.strftime("%Y-%m-%d")): t for t in result.trades}
    changed = False
    updated = []
    for entry in trades:
        key = (entry["symbol"], entry["decision_date"])
        trade = trades_by_key.get(key)
        already_frozen = entry.get("status") in ("Win", "Loss", "Breakeven")
        if trade is not None and trade.exit_reason != "end_of_data" and not already_frozen:
            r = trade.r_multiple or 0
            entry = {
                **entry,
                "entry_date": trade.entry_date.strftime("%Y-%m-%d"),
                "entry_price": _to_native_float(trade.entry_price),
                "partial_date": trade.partial_date.strftime("%Y-%m-%d") if trade.partial_date is not None else None,
                "partial_price": _to_native_float(trade.partial_price),
                "exit_date": trade.exit_date.strftime("%Y-%m-%d"),
                "exit_price": _to_native_float(trade.exit_price),
                "exit_reason": trade.exit_reason,
                "r_multiple": _to_native_float(trade.r_multiple),
                "pnl": _to_native_float(trade.pnl),
                "status": "Win" if r > 0 else ("Loss" if r < 0 else "Breakeven"),
            }
            changed = True
        updated.append(entry)
    return updated, changed


def join_results(trades: list, result: BacktestResult, skipped: list) -> pd.DataFrame:
    """One row per logged entry: matched to its realized Trade by
    (symbol, signal_date) where the engine actually produced one this run;
    otherwise a previously frozen closed outcome (see freeze_closed_trades)
    if there is one; otherwise an explicit "Pending" or "Not simulated"
    reason -- so a logged entry never just disappears without explanation."""
    skipped_by_key = {(s["symbol"], s["decision_date"]): s for s in skipped}
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
        elif entry.get("status") in ("Win", "Loss", "Breakeven"):
            # Not re-simulable this run (e.g. loaded history no longer
            # covers this date range), but its outcome was frozen when it
            # closed -- show that instead of losing it to "Not simulated".
            rows.append({**base, **{field: entry.get(field) for field in REALIZED_FIELDS}})
        else:
            skip_info = skipped_by_key.get(key)
            if skip_info is not None:
                reason, status = skip_info["reason"], ("Pending" if skip_info["status"] == "pending" else "Not simulated")
            else:
                reason = "Not simulated -- likely produced no valid trade (insufficient ADR history, or filtered)."
                status = "Not simulated"
            rows.append(
                {
                    **base,
                    "entry_date": None, "entry_price": None, "partial_date": None, "partial_price": None,
                    "exit_date": None, "exit_price": None, "exit_reason": reason,
                    "r_multiple": None, "pnl": None, "status": status,
                }
            )
    return pd.DataFrame(rows)
