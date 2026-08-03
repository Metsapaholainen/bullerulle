"""Standalone stop-loss/position-size calculator for a single ticker, meant
for trades placed on a different platform than this app -- you look up a
symbol here, read off the stop price and suggested share count, and enter
those manually elsewhere.

Deliberately mirrors backtest/engine.py's run_backtest() stop-distance and
position-sizing formulas EXACTLY (same ADR cap, same low-of-signal-day rule,
same risk-%/position-%/liquidity-% caps) so the number you get here is
consistent with what the backtester assumes for a real trade -- just applied
once, standalone, instead of inside the day-by-day simulation loop.
"""
from __future__ import annotations

from typing import Optional


def compute_stop_and_size(
    entry_price: float,
    signal_low: float,
    signal_high: float,
    adr_pct: float,
    side: str = "long",
    stop_mode: str = "low_of_signal_day",
    stop_adr_multiple: float = 1.0,
    risk_pct_per_trade: float = 1.0,
    equity: float = 100_000.0,
    max_position_pct_of_equity: float = 20.0,
    max_position_pct_of_avg_volume: float = 1.0,
    avg_volume: Optional[float] = None,
) -> dict:
    """`signal_low`/`signal_high` are the most recent complete bar's low/high
    (the "signal day" in backtest/engine.py's terms). Returns a dict with the
    stop price, stop distance ($ and %), suggested share count (after every
    cap engine.py would also apply), position value, and total $ at risk --
    or an `error` key set to a human-readable reason if no valid stop/size
    could be computed (e.g. non-positive ADR or entry price)."""
    if not (entry_price and entry_price > 0):
        return {"error": "Entry price must be positive."}
    if not (adr_pct and adr_pct > 0):
        return {"error": "ADR % must be positive -- check the symbol has enough history."}

    adr_stop_distance = entry_price * (adr_pct / 100.0) * stop_adr_multiple
    if adr_stop_distance <= 0:
        return {"error": "Computed ADR-based stop distance is zero."}

    if stop_mode == "low_of_signal_day":
        if side == "long":
            raw_stop_distance = entry_price - signal_low
        else:
            raw_stop_distance = signal_high - entry_price
        if raw_stop_distance is None or raw_stop_distance <= 0:
            stop_distance = adr_stop_distance
        else:
            stop_distance = min(raw_stop_distance, adr_stop_distance)
    else:
        stop_distance = adr_stop_distance

    if stop_distance <= 0:
        return {"error": "Computed stop distance is zero -- can't size a position."}

    stop_price = entry_price - stop_distance if side == "long" else entry_price + stop_distance

    risk_amount = equity * risk_pct_per_trade / 100.0
    shares = risk_amount / stop_distance
    affordable_shares = equity / entry_price if entry_price > 0 else 0.0
    shares = max(min(shares, affordable_shares), 0.0)

    capped_by = []
    if max_position_pct_of_equity > 0 and entry_price > 0:
        max_shares_by_position = (equity * max_position_pct_of_equity / 100.0) / entry_price
        if max_shares_by_position < shares:
            capped_by.append("max position % of equity")
        shares = min(shares, max_shares_by_position)
    if max_position_pct_of_avg_volume > 0 and avg_volume is not None and avg_volume > 0:
        max_shares_by_liquidity = avg_volume * (max_position_pct_of_avg_volume / 100.0)
        if max_shares_by_liquidity < shares:
            capped_by.append("max % of average volume")
        shares = min(shares, max_shares_by_liquidity)

    shares = int(shares)  # whole shares for a real order
    position_value = shares * entry_price
    total_risk_dollars = shares * stop_distance

    return {
        "error": None,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "stop_distance": stop_distance,
        "stop_distance_pct": stop_distance / entry_price * 100.0,
        "shares": shares,
        "position_value": position_value,
        "total_risk_dollars": total_risk_dollars,
        "capped_by": capped_by,
    }
