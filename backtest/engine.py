"""Event-driven, bar-based (daily) portfolio backtest engine.

Mechanics, matching how these setups are actually traded (and, where stated,
Qullamaggie's own documented rules) rather than a naive close-to-close backtest:
  - Entry either at the next bar's open (`entry_mode='next_open'`) or via a
    resting premarket stop order just beyond the signal day's extreme
    (`'stop_buy'`, the default) -- in which case a signal whose trigger is
    never traded through produces NO trade at all.
  - Anti-chase filter: skip a signal if its own day's range already exceeds
    `avoid_chase_adr_multiple` ADRs ("if price change on day is more than the
    ATR, skip it").
  - Initial stop: either the signal day's actual low, capped at
    `stop_adr_multiple` ADRs ("stop_mode='low_of_signal_day'`, Qullamaggie's
    stated rule), or a pure ADR-multiple from entry (`'adr_multiple'`) --
    which also drives position size via a fixed risk-% per trade, itself
    capped at `max_position_pct_of_equity` of the account on any one position.
  - Optional partial profit-take at an R-multiple, OR after
    `partial_profit_max_days` bars held (whichever comes first) -- "sell 1/3
    to 1/2 after 3-5 days." If taken, the stop can move to breakeven
    (`move_stop_to_breakeven_after_partial`), then the remainder trails off a
    close below (long) / above (short) an SMA or EMA.
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
    partial_date: Optional[pd.Timestamp] = None
    partial_price: Optional[float] = None
    partial_shares: Optional[float] = None
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    r_multiple: Optional[float] = None
    initial_stop_price: float = None  # the *original* stop, for R-multiple -- stop_price may later move to breakeven
    bars_held: int = 0  # bars elapsed since entry_date; drives the time-based partial-profit fallback

    def __post_init__(self):
        if self.shares == 0.0:
            self.shares = self.initial_shares
        if self.initial_stop_price is None:
            self.initial_stop_price = self.stop_price


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
                    "partial_date", "partial_price", "partial_shares",
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
                    "partial_date": t.partial_date,
                    "partial_price": t.partial_price,
                    "partial_shares": t.partial_shares,
                }
            )
        return pd.DataFrame(rows)


def _prepare_signal_frame(df: pd.DataFrame, trail_period: int, trail_ma_type: str = "sma") -> pd.DataFrame:
    d = df.copy()
    if trail_ma_type == "ema":
        d["trail_ema"] = d["close"].ewm(span=trail_period, adjust=False).mean()
    else:
        d["trail_ema"] = d["close"].rolling(trail_period).mean()
    if "adr_pct" not in d.columns:
        if "adr_pct_20" in d.columns:
            d["adr_pct"] = d["adr_pct_20"]
        else:
            d["adr_pct"] = (d["high"] / d["low"] - 1.0).rolling(20).mean() * 100.0
    if "vol_avg_50" not in d.columns:
        d["vol_avg_50"] = d["volume"].rolling(50).mean()
    return d


def _apply_partial(trade: Trade, price: float, side: str, bt: BacktestSettings, cash: float, date=None) -> float:
    """Sell/cover `partial_profit_fraction` of the position at `price`, and --
    if configured -- move the stop to breakeven. Returns the updated cash."""
    partial_shares = trade.shares * bt.partial_profit_fraction
    if side == "long":
        cash += partial_shares * price * (1 - bt.slippage_pct / 100.0)
        trade.partial_pnl += (price - trade.entry_price) * partial_shares
    else:
        cash -= partial_shares * price * (1 + bt.slippage_pct / 100.0)
        trade.partial_pnl += (trade.entry_price - price) * partial_shares
    trade.shares -= partial_shares
    trade.partial_taken = True
    trade.partial_date = date
    trade.partial_price = price
    trade.partial_shares = partial_shares
    if bt.move_stop_to_breakeven_after_partial:
        trade.stop_price = max(trade.stop_price, trade.entry_price) if side == "long" else min(trade.stop_price, trade.entry_price)
    return cash


@dataclass
class EntryPlan:
    """The outcome of deciding whether, where and with what stop to enter.

    `tradeable=False` carries a human-readable `skip_reason` rather than
    silently vanishing, so the Orders tab can explain why a scan match
    didn't become an order instead of just omitting it."""
    tradeable: bool
    skip_reason: Optional[str] = None
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    stop_distance: Optional[float] = None
    target_price: Optional[float] = None
    trigger_price: Optional[float] = None      # None under entry_mode="next_open"
    limit_cap_price: Optional[float] = None    # for placing a stop-LIMIT order
    adr_pct: Optional[float] = None


def resolve_entry_and_stop(
    bt: BacktestSettings,
    side: str,
    signal_bar,
    entry_bar=None,
    stop_mode: str = None,
) -> EntryPlan:
    """Single source of truth for "do we take this, at what price, with what stop".

    Used by run_backtest (below), by the Orders tab to build real order
    tickets, by stop_calculator, and by the chart's projected entry/stop
    overlay -- all four previously carried their own copy of this arithmetic
    and had already drifted apart from each other.

    `entry_bar` is the bar the order is live for (the one after the signal).
    Pass None to plan an order for a session that hasn't happened yet: the
    trigger, stop and size are all knowable in advance, only the fill isn't,
    so the returned plan assumes a fill at the trigger.
    """
    stop_mode = stop_mode or bt.stop_mode
    sign = 1.0 if side == "long" else -1.0

    adr_pct = signal_bar.get("adr_pct") if hasattr(signal_bar, "get") else signal_bar["adr_pct"]
    if adr_pct is None or not np.isfinite(adr_pct) or adr_pct <= 0:
        return EntryPlan(False, "no ADR data")

    # Anti-chase: a pure property of the signal day, independent of the fill.
    # "If price change on day is more than the ATR, skip it."
    if bt.avoid_chase_adr_multiple > 0:
        signal_range_pct = (signal_bar["high"] / signal_bar["low"] - 1.0) * 100.0
        if np.isfinite(signal_range_pct) and signal_range_pct > adr_pct * bt.avoid_chase_adr_multiple:
            return EntryPlan(False, f"already moved {signal_range_pct:.1f}% on the signal day (anti-chase)")

    trigger_price = None
    limit_cap_price = None

    if bt.entry_mode == "stop_buy":
        # A resting order just beyond the signal day's extreme.
        if side == "long":
            trigger_price = signal_bar["high"] * (1 + bt.entry_buffer_pct / 100.0)
        else:
            trigger_price = signal_bar["low"] * (1 - bt.entry_buffer_pct / 100.0)
        if not np.isfinite(trigger_price) or trigger_price <= 0:
            return EntryPlan(False, "no valid trigger price")
        if bt.max_gap_fill_pct > 0:
            limit_cap_price = trigger_price * (1 + sign * bt.max_gap_fill_pct / 100.0)

        if entry_bar is None:
            # Planning ahead: assume a fill exactly at the trigger.
            raw_entry_price = trigger_price
        else:
            triggered = (
                entry_bar["high"] >= trigger_price if side == "long"
                else entry_bar["low"] <= trigger_price
            )
            if not triggered:
                return EntryPlan(False, "never traded through the trigger", trigger_price=trigger_price)
            # A gap far past the trigger fills at a price the setup never
            # justified. Note this can only fire when open is already beyond
            # the trigger, so it can't conflict with the trigger test above.
            if limit_cap_price is not None:
                gapped_past = (
                    entry_bar["open"] > limit_cap_price if side == "long"
                    else entry_bar["open"] < limit_cap_price
                )
                if gapped_past:
                    return EntryPlan(
                        False, f"gapped past the trigger (opened {entry_bar['open']:.2f})",
                        trigger_price=trigger_price, limit_cap_price=limit_cap_price,
                    )
            raw_entry_price = (
                max(entry_bar["open"], trigger_price) if side == "long"
                else min(entry_bar["open"], trigger_price)
            )
    else:
        if entry_bar is None:
            return EntryPlan(False, "no entry bar yet (next_open needs one)")
        raw_entry_price = entry_bar["open"]

    if not np.isfinite(raw_entry_price) or raw_entry_price <= 0:
        return EntryPlan(False, "no valid entry price")
    entry_price = raw_entry_price * (1 + bt.slippage_pct / 100.0 * sign)

    adr_stop_distance = entry_price * (adr_pct / 100.0) * bt.stop_adr_multiple
    if adr_stop_distance <= 0:
        return EntryPlan(False, "ADR stop distance is zero")

    if stop_mode == "low_of_signal_day":
        if side == "long":
            raw_stop_distance = entry_price - signal_bar["low"]
        else:
            raw_stop_distance = signal_bar["high"] - entry_price
        if not np.isfinite(raw_stop_distance) or raw_stop_distance <= 0:
            stop_distance = adr_stop_distance
        elif raw_stop_distance > adr_stop_distance:
            # The signal day's low is further than stop_adr_multiple ADRs from
            # where we actually got filled. See stop_exceeds_adr_action's
            # comment in settings.py -- this binds on most trades under
            # stop_buy, which is why the choice is exposed at all.
            if bt.stop_exceeds_adr_action == "skip":
                return EntryPlan(
                    False,
                    f"stop would be {raw_stop_distance / adr_stop_distance:.1f}x the ADR cap",
                    trigger_price=trigger_price, limit_cap_price=limit_cap_price, adr_pct=adr_pct,
                )
            stop_distance = adr_stop_distance
        else:
            stop_distance = raw_stop_distance
    else:
        stop_distance = adr_stop_distance

    if stop_distance <= 0:
        return EntryPlan(False, "stop distance is zero")

    stop_price = entry_price - sign * stop_distance
    target_price = entry_price + sign * bt.partial_profit_r_multiple * stop_distance
    return EntryPlan(
        True, None,
        entry_price=entry_price, stop_price=stop_price, stop_distance=stop_distance,
        target_price=target_price, trigger_price=trigger_price,
        limit_cap_price=limit_cap_price, adr_pct=adr_pct,
    )


def run_backtest(
    bt: BacktestSettings,
    signals: dict,
    side: str = "long",
    setup_name: str = None,
) -> BacktestResult:
    """`signals`: {symbol: DataFrame} as produced by
    scanner.scan_signals_over_history -- must have a boolean `match` column
    plus open/high/low/close/volume (and ideally adr_pct).

    `setup_name`, if passed as "episodic_pivot", activates
    `bt.ep_stop_mode_override` (if set) for these trades only. Per the
    source article, EP's stop rule is worded identically to Breakout's --
    "the stop is at the lows of the day" -- so the default (override=None,
    i.e. plain `stop_mode="low_of_signal_day"`, the signal day's whole-day
    low, ADR-capped) already matches it. The override exists only as an
    optional alternative to experiment with, not because the default is
    wrong. Every other setup, and EP with no override configured, behaves
    exactly as before."""
    effective_stop_mode = (
        bt.ep_stop_mode_override if (setup_name == "episodic_pivot" and bt.ep_stop_mode_override) else bt.stop_mode
    )
    prepared = {
        sym: _prepare_signal_frame(df, bt.trail_ema_period, bt.trail_ma_type)
        for sym, df in signals.items()
        if not df.empty
    }

    if not prepared:
        return BacktestResult(trades=[], equity_curve=pd.Series(dtype=float), side=side)

    all_dates = sorted(set().union(*[set(df.index) for df in prepared.values()]))

    # stop_buy models a premarket order placed for the session immediately
    # after the signal, so the delay is structurally fixed at 1 bar: 0 would
    # derive the trigger from the same bar it fills on (look-ahead), and >=2
    # would arm a stale trigger from days earlier. Coerced rather than
    # raised -- optimizer.py grid-loops over run_backtest with no exception
    # handling, so a raise here would kill an entire parameter sweep.
    effective_entry_delay = 1 if bt.entry_mode == "stop_buy" else bt.entry_delay_days

    # Precompute, for every symbol, the entry date for every signal (offset
    # by the effective delay within that symbol's own trading calendar).
    # Under stop_buy this is the date the order is LIVE for, not a guaranteed
    # fill -- whether it actually fills is decided per-bar below, since that
    # needs the entry bar's high/open.
    entries_by_date = defaultdict(list)
    for symbol, df in prepared.items():
        idx = df.index
        match_locs = np.where(df["match"].fillna(False).values)[0]
        for loc in match_locs:
            entry_loc = loc + effective_entry_delay
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
            trade.bars_held += 1

            exit_price = None
            exit_reason = None

            # Partial profit fires at the R-multiple target, OR after
            # partial_profit_max_days bars held -- whichever comes first
            # ("sell 1/3 to 1/2 after 3-5 days").
            time_partial_due = (
                not trade.partial_taken
                and bt.partial_profit_max_days > 0
                and trade.bars_held >= bt.partial_profit_max_days
            )

            if side == "long":
                if bar["low"] <= trade.stop_price:
                    exit_price, exit_reason = trade.stop_price, "stop"
                elif not trade.partial_taken and trade.target_price and bar["high"] >= trade.target_price:
                    cash = _apply_partial(trade, trade.target_price, side, bt, cash, date=date)
                elif time_partial_due:
                    cash = _apply_partial(trade, bar["close"], side, bt, cash, date=date)
                elif bar["close"] < bar["trail_ema"]:
                    exit_price, exit_reason = bar["close"], "trail"
            else:  # short
                if bar["high"] >= trade.stop_price:
                    exit_price, exit_reason = trade.stop_price, "stop"
                elif not trade.partial_taken and trade.target_price and bar["low"] <= trade.target_price:
                    cash = _apply_partial(trade, trade.target_price, side, bt, cash, date=date)
                elif time_partial_due:
                    cash = _apply_partial(trade, bar["close"], side, bt, cash, date=date)
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

                stop_distance = abs(trade.entry_price - trade.initial_stop_price)
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

            signal_bar = df.loc[signal_date]
            entry_bar = df.loc[date]

            # Anti-chase, entry fill (incl. whether a stop-buy triggered at
            # all) and stop placement all live in one shared helper so the
            # Orders tab, stop_calculator and the chart overlay can't drift
            # away from what the engine actually does.
            plan = resolve_entry_and_stop(bt, side, signal_bar, entry_bar, effective_stop_mode)
            if not plan.tradeable:
                continue
            entry_price = plan.entry_price
            stop_distance = plan.stop_distance

            risk_amount = equity * bt.risk_pct_per_trade / 100.0
            shares = risk_amount / stop_distance
            affordable_shares = cash / entry_price if entry_price > 0 else 0
            shares = max(min(shares, affordable_shares), 0.0)
            # Never risk more than max_position_pct_of_equity of the account
            # on one position, regardless of what risk-based sizing implies.
            if bt.max_position_pct_of_equity > 0 and entry_price > 0:
                max_shares_by_position_cap = (equity * bt.max_position_pct_of_equity / 100.0) / entry_price
                shares = min(shares, max_shares_by_position_cap)
            # "Buy no more than 1% of the average volume" -- caps share
            # count against the signal day's own liquidity, independent of
            # the equity-based caps above.
            if bt.max_position_pct_of_avg_volume > 0:
                avg_vol_at_signal = signal_bar.get("vol_avg_50")
                if avg_vol_at_signal is not None and np.isfinite(avg_vol_at_signal) and avg_vol_at_signal > 0:
                    max_shares_by_liquidity = avg_vol_at_signal * (bt.max_position_pct_of_avg_volume / 100.0)
                    shares = min(shares, max_shares_by_liquidity)
            if shares <= 0:
                continue

            stop_price = plan.stop_price
            target_price = plan.target_price

            # The entry bar is never seen by the exits loop above (it runs
            # before entries), so a fill that reverses through its own stop
            # the same session would otherwise be carried to the next day for
            # free. Under stop_buy the fill sits near the TOP of the bar's
            # range, which makes that materially more likely -- see
            # check_same_bar_stop in settings.py.
            if bt.check_same_bar_stop:
                stopped_same_bar = (
                    entry_bar["low"] <= stop_price if side == "long"
                    else entry_bar["high"] >= stop_price
                )
                if stopped_same_bar:
                    # Same slippage/commission convention as the main exit
                    # path below: slippage hits the cash proceeds, PnL is
                    # measured on the raw prices, commission is charged once
                    # on the full position.
                    if side == "long":
                        cash += shares * stop_price * (1 - bt.slippage_pct / 100.0) - shares * entry_price
                        total_pnl = (stop_price - entry_price) * shares
                    else:
                        cash += shares * entry_price - shares * stop_price * (1 + bt.slippage_pct / 100.0)
                        total_pnl = (entry_price - stop_price) * shares
                    commission = bt.commission_per_share * shares
                    total_pnl -= commission
                    cash -= commission

                    trade = Trade(
                        symbol=symbol, side=side, signal_date=signal_date, entry_date=date,
                        entry_price=entry_price, initial_shares=shares,
                        stop_price=stop_price, target_price=target_price,
                    )
                    trade.exit_date = date
                    trade.exit_price = stop_price
                    trade.exit_reason = "stop_same_bar"
                    trade.pnl = total_pnl
                    trade.r_multiple = total_pnl / (stop_distance * shares) if stop_distance > 0 else 0.0
                    trade.shares = 0.0
                    closed_trades.append(trade)
                    continue

            if side == "long":
                cash -= shares * entry_price
            else:
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
        stop_distance = abs(trade.entry_price - trade.initial_stop_price)
        trade.exit_date = df.index[-1]
        trade.exit_price = last_px
        trade.exit_reason = "end_of_data"
        trade.pnl = total_pnl
        trade.r_multiple = total_pnl / (stop_distance * trade.initial_shares) if stop_distance > 0 else np.nan
        closed_trades.append(trade)

    equity_series = pd.Series(dict(equity_curve)).sort_index()
    return BacktestResult(trades=closed_trades, equity_curve=equity_series, side=side)
