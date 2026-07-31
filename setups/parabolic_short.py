"""Qullamaggie 'Parabolic Short' setup: a stock extended many ADRs above its
short-term EMA after a huge, fast run -- the short-side counterpart in his
playbook (over-extension / blowoff-top reversal candidate).
"""
from __future__ import annotations

import pandas as pd

from indicators import add_adr_pct, add_moving_averages, add_consecutive_up_days
from settings import ParabolicShortSettings


def scan(df: pd.DataFrame, settings: ParabolicShortSettings, rs_rating: pd.Series = None) -> pd.DataFrame:
    df = df.copy()

    ema_col = f"ema_{settings.ema_period}"
    if ema_col not in df.columns:
        df[ema_col] = df["close"].ewm(span=settings.ema_period, adjust=False).mean()

    adr_col = "adr_pct_20"
    if adr_col not in df.columns:
        df = add_adr_pct(df, lookback=20)

    if "consecutive_up_days" not in df.columns:
        df = add_consecutive_up_days(df)

    extension_pct = (df["close"] - df[ema_col]) / df[ema_col] * 100.0
    extension_adr_multiple = extension_pct / df[adr_col].replace(0, pd.NA)

    run_up_pct = (df["close"] / df["close"].shift(settings.run_up_lookback_days) - 1.0) * 100.0

    cond = pd.Series(True, index=df.index)
    cond &= extension_adr_multiple >= settings.min_extension_adr_multiple
    cond &= run_up_pct >= settings.min_run_up_pct
    cond &= df["consecutive_up_days"] >= settings.consecutive_up_days_min

    out = pd.DataFrame(index=df.index)
    out["match"] = cond.fillna(False)
    out["extension_adr_multiple"] = extension_adr_multiple
    out["run_up_pct"] = run_up_pct
    out["consecutive_up_days"] = df["consecutive_up_days"]
    out["score"] = extension_adr_multiple.clip(lower=0) + run_up_pct.clip(lower=0) / 50.0
    return out
