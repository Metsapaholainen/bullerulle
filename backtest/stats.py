"""Performance stats derived from a BacktestResult: win rate, average R,
expectancy, CAGR, max drawdown, Sharpe -- the numbers used to rank presets
and parameter-grid combinations."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.engine import BacktestResult


def compute_stats(result: BacktestResult) -> dict:
    trades = result.trades_df()
    equity = result.equity_curve

    if trades.empty or equity.empty:
        return {
            "trade_count": 0,
            "win_rate": np.nan,
            "avg_r": np.nan,
            "expectancy_r": np.nan,
            "total_pnl": 0.0,
            "cagr": np.nan,
            "max_drawdown_pct": np.nan,
            "sharpe": np.nan,
            "final_equity": equity.iloc[-1] if not equity.empty else np.nan,
        }

    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    win_rate = len(wins) / len(trades) * 100.0
    avg_r = trades["r_multiple"].mean()
    avg_win_r = wins["r_multiple"].mean() if not wins.empty else 0.0
    avg_loss_r = losses["r_multiple"].mean() if not losses.empty else 0.0
    expectancy_r = (win_rate / 100.0) * avg_win_r + (1 - win_rate / 100.0) * avg_loss_r

    daily_returns = equity.pct_change().dropna()
    n_days = (equity.index[-1] - equity.index[0]).days or 1
    years = n_days / 365.25
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown_pct = drawdown.min() * 100.0

    sharpe = (
        daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        if daily_returns.std() not in (0, np.nan) and len(daily_returns) > 1
        else np.nan
    )

    return {
        "trade_count": len(trades),
        "win_rate": win_rate,
        "avg_r": avg_r,
        "expectancy_r": expectancy_r,
        "total_pnl": trades["pnl"].sum(),
        "cagr": cagr * 100.0 if pd.notna(cagr) else np.nan,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe": sharpe,
        "final_equity": equity.iloc[-1],
    }


def stats_summary_text(stats: dict) -> str:
    if stats["trade_count"] == 0:
        return "No trades were generated with these settings over this period."
    return (
        f"{stats['trade_count']} trades | Win rate {stats['win_rate']:.1f}% | "
        f"Avg R {stats['avg_r']:.2f} | Expectancy {stats['expectancy_r']:.2f}R | "
        f"CAGR {stats['cagr']:.1f}% | Max DD {stats['max_drawdown_pct']:.1f}% | "
        f"Sharpe {stats['sharpe']:.2f}"
    )
