"""EP-specific ntfy.sh push-notification formatting, built on top of the
parent project's generic `notifications.send_ntfy_alert()`. Two message
types, matching the two-pass schedule (see scan_early.py/scan_confirm.py):
an informational "heads up" from pass 1 (not yet actionable -- premarket
gap/volume isn't confirmed yet) and an actionable stop-buy-with-limit order
ticket from pass 2.
"""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd

from notifications import send_ntfy_alert


def _resolve_ep_topic(explicit: Optional[str] = None) -> Optional[str]:
    """Explicit arg > NTFY_TOPIC_EP (a separate topic, so EP alerts don't
    mix into whatever else is subscribed to the parent project's
    NTFY_TOPIC) > NTFY_TOPIC (fall back to sharing the one topic if a
    dedicated one was never set up) -- resolution of the shared fallback
    itself happens inside notifications.send_ntfy_alert."""
    if explicit:
        return explicit
    dedicated = os.environ.get("NTFY_TOPIC_EP")
    if dedicated:
        return dedicated
    try:
        import streamlit as st

        if hasattr(st, "secrets") and "NTFY_TOPIC_EP" in st.secrets:
            return st.secrets["NTFY_TOPIC_EP"]
    except Exception:
        pass
    return None  # falls through to NTFY_TOPIC inside send_ntfy_alert


def format_early_candidate_line(row) -> str:
    parts = [f"{row['symbol']} (gap {row.get('gap_pct', 0):+.1f}%, vol {row.get('volume_ratio', 0):.1f}x)"]
    growth_tier = row.get("growth_tier")
    if growth_tier and growth_tier != "unknown":
        parts.append(f"growth={growth_tier}")
    eps_beat_pct = row.get("eps_beat_pct")
    if eps_beat_pct is not None and pd.notna(eps_beat_pct):
        parts.append(f"beat={eps_beat_pct:+.0f}%")
    catalyst_type = row.get("catalyst_type")
    if catalyst_type:
        parts.append(f"catalyst={catalyst_type} ({row.get('catalyst_confidence', 0) or 0:.0%})")
    return " | ".join(parts)


def send_early_watchlist_alert(candidates_df: pd.DataFrame, topic: Optional[str] = None) -> None:
    """Pass 1: informational only -- explicitly NOT an order ticket, since
    premarket gap/volume hasn't been confirmed yet (see scan_confirm.py)."""
    if candidates_df.empty:
        return
    lines = [format_early_candidate_line(row) for _, row in candidates_df.iterrows()]
    title = f"EP watchlist: {len(candidates_df)} candidate(s) to watch before the open"
    message = (
        "Not yet actionable -- premarket gap/volume unconfirmed. Will re-check before the open.\n\n"
        + "\n".join(lines)
    )
    send_ntfy_alert(title, message, topic=_resolve_ep_topic(topic), priority="default")


def format_confirmed_order_line(row) -> str:
    limit_cap = row.get("limit_cap_price")
    line = f"{row['symbol']}: trigger ${row['trigger_price']:.2f}"
    if limit_cap is not None and pd.notna(limit_cap):
        line += f" / limit cap ${limit_cap:.2f}"
    line += f", stop ${row['stop_price']:.2f}, size {int(row['shares'])} sh"
    risk = row.get("total_risk_dollars")
    if risk is not None and pd.notna(risk):
        line += f" (risk ${risk:.0f})"
    return line


def send_confirmed_alerts(orders_df: pd.DataFrame, topic: Optional[str] = None) -> list:
    """Pass 2: one ntfy push per confirmed, actionable EP candidate -- a
    real stop-buy-with-limit order ticket. Returns the list of symbols
    actually alerted."""
    alerted = []
    for _, row in orders_df.iterrows():
        title = f"EP entry: {row['symbol']} (gap {row.get('gap_pct', 0):+.1f}%)"
        message = format_confirmed_order_line(row)
        rationale = row.get("catalyst_rationale")
        if rationale and pd.notna(rationale):
            message += f"\nCatalyst: {rationale}"
        send_ntfy_alert(title, message, topic=_resolve_ep_topic(topic), priority="high")
        alerted.append(row["symbol"])
    return alerted
