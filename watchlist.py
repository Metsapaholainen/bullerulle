"""General-purpose watchlist: just a plain list of symbols you want to keep
an eye on and flip through charts for -- no moving-average rule attached
(that's what Sell Alerts is for). Persisted the same way as the Sell Alerts
watchlist and the Paper Trading journal (see github_store.py), so it
survives a Streamlit Cloud restart instead of living only on the
container's ephemeral disk.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

import github_store

GITHUB_STORE_PATH = "watchlist.yaml"


def load_watchlist(path: Optional[Path] = None) -> list:
    """Returns a list of uppercased symbol strings. Pass an explicit `path`
    to read a specific local file instead of the normal GitHub-backed store
    (used by tests)."""
    if path is not None:
        if not path.exists():
            return []
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
    else:
        raw = github_store.read_file(GITHUB_STORE_PATH)
        data = (yaml.safe_load(raw) or {}) if raw else {}
    return [str(s).upper() for s in (data.get("watchlist") or [])]


def save_watchlist(symbols: list, path: Optional[Path] = None) -> None:
    normalized = sorted({s.upper() for s in symbols})
    content = yaml.safe_dump({"watchlist": normalized}, sort_keys=False)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return
    github_store.write_file(GITHUB_STORE_PATH, content, "Update watchlist")
