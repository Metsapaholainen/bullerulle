"""Event-driven, bar-based (daily) portfolio backtest engine.

Mechanics, matching how these setups are actually traded rather than a naive
close-to-close backtest:
  - Entry on the next bar's open after a signal day (`entry_delay_days`).
  - Initial stop sized in ADR multiples (`stop_adr_multiple`), which also
    drives position size via a fixed risk-% per trade.
  - Optional partial profit-take at an R-multiple, trailing the remainder
    off a close below (long) / above (short) an EMA.
  - A portfolio-level `max_positions` cap shared across all symbols.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from settings import BacktestSettings


@dataclass
class Trade:
    symbol: str
    side: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    initial_shares: float
    stop_price: float
    target_price: Optional[float]
    shares: float = 0.0
    partial_taken: bool = False
    partial_pnl: float = 0.0
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    r_multiple: Optional[float] = None

    def __post_init__(self):
        if self.shares == 0.0:
            self.shares = self.initial_shares


@dataclass
class BacktestResult:
    trades: list = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    side: str = "long"

    def trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(
                columns=[
                    "symbol", "side", "signal_date", "entry_date", "entry_price",
                    "exit_date", "exit_price", "exit_reason", "shares", "pnl", "r_multiple",
                ]
            )
        rows = []
        for t in self.trades:
            rows.append(
                {
                    "symbol": t.symbol,
                    "side": t.side,
                    "signal_date": t.signal_date,
                    "entry_date": t.entry_date,
                    "entry_price": t.entry_price,
                    "exit_date": t.exit_date,
                    "exit_price": t.exit_price,
                    "exit_reason": t.exit_reason,
                    "shares": t.initial_shares,
                    "pnl": t.pnl,
                    "r_multiple": t.r_multiple,
                }
            )
        return pd.DataFrame(rows)


def _prepare_signal_frame(df: pd.DataFrame, trail_ema_period: int) -> pd.DataFrame:
    d = df.copy()
    d["trail_ema"] = d["close"].ewm(span=trail_ema_period, adjust=False).mean()
    if "adr_pct" not in d.columns:
        if "adr_pct_20" in d.columns:
            d["adr_pct"] = d["adr_pct_20"]
        else:
            d["adr_pct"] = (d["high"] / d["low"] - 1.0).rolling(20).mean() * 100.0
    return d


def run_backtest(
    bt: BacktestSettings,
    signals: dict,
    side: str = "long",
) -> BacktestResult:
    """`signals`: {symbol: DataFrame} as produced by
    scanner.scan_signals_over_history -- must have a boolean `match` column
    plus open/high/low/close/volume (and ideally adr_pct)."""
    prepared = {sym: _prepare_signal_frame(df, bt.trail_ema_period) for sym, df in signals.items() if not df.empty}

    if not prepared:
        return BacktestResult(trades=[], equity_curve=pd.Series(dtype=float), side=side)

    all_dates = sorted(set().union(*[set(df.index) for df in prepared.values()]))

    # Precompute, for every symbol, the entry date for every signal (offset
    # by entry_delay_days within that symbol's own trading calendar).
    entries_by_date = defaultdict(list)
    for symbol, df in prepared.items():
        idx = df.index
        match_locs = np.where(df["match"].fillna(False).values)[0]
        for loc in match_locs:
            entry_loc = loc + bt.entry_delay_days
            if entry_loc < len(idx):
                entries_by_date[idx[entry_loc]].append((symbol, idx[loc]))

    cash = float(bt.initial_capital)
    open_positions: dict = {}
    closed_trades = []
    equity_curve = []
    sign = 1.0 if side == "long" else -1.0

    for date in all_dates:
        # --- 1. manage exits for open positions ---
        for symbol in list(open_positions.keys()):
            trade = open_positions[symbol]
            df = prepared[symbol]
            if date not in df.index:
                continue
            bar = df.loc[date]

            exit_price = None
            exit_reason = None

            if side == "long":
                if bar["low"] <= trade.stop_price:
                    exit_price, exit_reason = trade.stop_price, "stop"
                elif not trade.partial_taken and trade.target_price and bar["high"] >= trade.target_price:
                    partial_shares = trade.shares * bt.partial_profit_fraction
                    proceeds = partial_shares * trade.target_price * (1 - bt.slippage_pct / 100.0)
                    cash += proceeds
                    trade.partial_pnl += (trade.target_price - trade.entry_price) * partial_shares
                    trade.shares -= partial_shares
                    trade.partial_taken = True
                elif bar["close"] < bar["trail_ema"]:
                    exit_price, exit_reason = bar["close"], "trail"
            else:  # short
                if bar["high"] >= trade.stop_price:
                    exit_price, exit_reason = trade.stop_price, "stop"
                elif not trade.partial_taken and trade.target_price and bar["low"] <= trade.target_price:
                    # buy back (cover) the partial tranche at the target price
                    partial_shares = trade.shares * bt.partial_profit_fraction
                    cash -= partial_shares * trade.target_price * (1 + bt.slippage_pct / 100.0)
                    trade.partial_pnl += (trade.entry_price - trade.target_price) * partial_shares
                    trade.shares -= partial_shares
                    trade.partial_taken = True
                elif bar["close"] > bar["trail_ema"]:
                    exit_price, exit_reason = bar["close"], "trail"

            if exit_price is not None:
                if side == "long":
                    proceeds = trade.shares * exit_price * (1 - bt.slippage_pct / 100.0)
                    cash += proceeds
                    total_pnl = trade.partial_pnl + (exit_price - trade.entry_price) * trade.shares
                else:
                    proceeds = trade.shares * exit_price * (1 + bt.slippage_pct / 100.0)
                    cash -= proceeds
                    total_pnl = trade.partial_pnl + (trade.entry_price - exit_price) * trade.shares

                commission = bt.commission_per_share * trade.initial_shares
                total_pnl -= commission
                cash -= commission

                stop_distance = abs(trade.entry_price - trade.stop_price)
                trade.exit_date = date
                trade.exit_price = exit_price
                trade.exit_reason = exit_reason
                trade.pnl = total_pnl
                trade.r_multiple = total_pnl / (stop_distance * trade.initial_shares) if stop_distance > 0 else np.nan
                closed_trades.append(trade)
                del open_positions[symbol]

        # --- 2. equity mark-to-market (before new entries, using today's close) ---
        # Longs: cash already paid for shares, so equity = cash + market value held.
        # Shorts: cash already received short-sale proceeds, so equity = cash - liability to buy back.
        open_market_value = sum(
            prepared[s].loc[date, "close"] * t.shares
            for s, t in open_positions.items()
            if date in prepared[s].index
        )
        equity = cash + open_market_value if side == "long" else cash - open_market_value

        # --- 3. new entries ---
        for symbol, signal_date in entries_by_date.get(date, []):
            if symbol in open_positions:
                continue
            if len(open_positions) >= bt.max_positions:
                continue
            df = prepared[symbol]
            if date not in df.index or signal_date not in df.index:
                continue

            raw_entry_price = df.loc[date, "open"]
            if not np.isfinite(raw_entry_price) or raw_entry_price <= 0:
                continue
            entry_price = raw_entry_price * (1 + bt.slippage_pct / 100.0 * sign)

            adr_pct = df.loc[signal_date, "adr_pct"]
            if not np.isfinite(adr_pct) or adr_pct <= 0:
                continue
            stop_distance = entry_price * (adr_pct / 100.0) * bt.stop_adr_multiple
            if stop_distance <= 0:
                continue

            risk_amount = equity * bt.risk_pct_per_trade / 100.0
            shares = risk_amount / stop_distance
            affordable_shares = cash / entry_price if entry_price > 0 else 0
            shares = max(min(shares, affordable_shares), 0.0)
            if shares <= 0:
                continue

            if side == "long":
                stop_price = entry_price - stop_distance
                target_price = entry_price + bt.partial_profit_r_multiple * stop_distance
                cash -= shares * entry_price
            else:
                stop_price = entry_price + stop_distance
                target_price = entry_price - bt.partial_profit_r_multiple * stop_distance
                cash += shares * entry_price

            open_positions[symbol] = Trade(
                symbol=symbol,
                side=side,
                signal_date=signal_date,
                entry_date=date,
                entry_price=entry_price,
                initial_shares=shares,
                stop_price=stop_price,
                target_price=target_price,
            )

        equity_curve.append((date, equity))

    # close anything still open at the last available price
    last_date = all_dates[-1]
    for symbol, trade in list(open_positions.items()):
        df = prepared[symbol]
        last_px = df["close"].iloc[-1]
        if side == "long":
            proceeds = trade.shares * last_px * (1 - bt.slippage_pct / 100.0)
            cash += proceeds
            total_pnl = trade.partial_pnl + (last_px - trade.entry_price) * trade.shares
        else:
            proceeds = trade.shares * last_px * (1 + bt.slippage_pct / 100.0)
            cash -= proceeds
            total_pnl = trade.partial_pnl + (trade.entry_price - last_px) * trade.shares
        stop_distance = abs(trade.entry_price - trade.stop_price)
        trade.exit_date = df.index[-1]
        trade.exit_price = last_px
        trade.exit_reason = "end_of_data"
        trade.pnl = total_pnl
        trade.r_multiple = total_pnl / (stop_distance * trade.initial_shares) if stop_distance > 0 else np.nan
        closed_trades.append(trade)

    equity_series = pd.Series(dict(equity_curve)).sort_index()
    return BacktestResult(trades=closed_trades, equity_curve=equity_series, side=side)
