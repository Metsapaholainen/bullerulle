"""Push-notification delivery for Sell Alerts, via ntfy.sh (https://ntfy.sh) --
a free, account-free pub/sub service: POST a message to a topic URL and
anyone subscribed to that topic (the free phone app, or just a browser tab)
gets it instantly. The topic name itself is the only "secret" here -- pick
something long and unguessable, since anyone who knows it can read (and post
to) it too.
"""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

NTFY_BASE_URL = "https://ntfy.sh"


def _resolve_ntfy_topic(explicit: Optional[str] = None) -> Optional[str]:
    """Explicit arg > Streamlit Cloud secrets > local .env/os.environ --
    same fallback chain as data.fmp_client._resolve_api_key."""
    if explicit:
        return explicit
    try:
        import streamlit as st

        if hasattr(st, "secrets") and "NTFY_TOPIC" in st.secrets:
            return st.secrets["NTFY_TOPIC"]
    except Exception:
        pass
    return os.environ.get("NTFY_TOPIC")


def send_ntfy_alert(title: str, message: str, topic: Optional[str] = None, priority: str = "high") -> None:
    resolved_topic = _resolve_ntfy_topic(topic)
    if not resolved_topic:
        raise RuntimeError("No ntfy.sh topic configured -- set NTFY_TOPIC in .env or Streamlit secrets.")
    response = requests.post(
        f"{NTFY_BASE_URL}/{resolved_topic}",
        data=message.encode("utf-8"),
        headers={"Title": title, "Priority": priority, "Tags": "warning"},
        timeout=10,
    )
    response.raise_for_status()


def format_sell_alert_message(row) -> str:
    date_str = pd.Timestamp(row["date"]).date().isoformat() if row.get("date") is not None else "?"
    return (
        f"{row['symbol']} closed at ${row['close']:.2f}, below its {row['ma_period']}-day MA "
        f"of ${row['sma']:.2f} ({row['pct_from_ma']:+.1f}%) on {date_str}."
    )


def send_sell_alerts(matches_df: pd.DataFrame, topic: Optional[str] = None) -> list:
    """Sends one ntfy push per row in `matches_df` (the caller filters to
    is_below == True first). Returns the list of symbols actually alerted."""
    alerted = []
    for _, row in matches_df.iterrows():
        message = format_sell_alert_message(row)
        title = f"Sell alert: {row['symbol']} below {row['ma_period']}-day MA"
        send_ntfy_alert(title, message, topic=topic)
        alerted.append(row["symbol"])
    return alerted
