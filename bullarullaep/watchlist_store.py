"""Persistence for BullaRullaEP's daily state -- pass-1's candidate list
(handed off to pass-2 the same day) and a running log of real, live alerts
sent (for honest post-hoc tracking of how the system performs outside of
backtests). Built on the parent project's `github_store.py`, which already
handles GitHub-Contents-API persistence with a local-file fallback, so this
needs no new infrastructure.
"""
from __future__ import annotations

import yaml

import github_store

CANDIDATES_PATH_TEMPLATE = "bullarullaep_store/candidates_{date}.yaml"
ALERT_LOG_PATH = "bullarullaep_store/alert_log.yaml"


def save_early_candidates(as_of: str, candidates: list) -> None:
    """`candidates` is a list of plain dicts (one per symbol), e.g. from
    `DataFrame.to_dict("records")`."""
    path = CANDIDATES_PATH_TEMPLATE.format(date=as_of)
    content = yaml.safe_dump({"as_of": as_of, "candidates": candidates}, sort_keys=False)
    github_store.write_file(path, content, message=f"EP candidates {as_of}")


def load_early_candidates(as_of: str) -> list:
    path = CANDIDATES_PATH_TEMPLATE.format(date=as_of)
    content = github_store.read_file(path)
    if not content:
        return []
    data = yaml.safe_load(content) or {}
    return data.get("candidates", [])


def append_alert_log(entries: list) -> None:
    """Appends real (not backtested) confirmed alerts to a running log, for
    honest tracking of how the system performs live over time. `entries` is
    a list of plain dicts, one per alerted symbol."""
    if not entries:
        return
    existing_content = github_store.read_file(ALERT_LOG_PATH)
    existing_data = (yaml.safe_load(existing_content) or {}) if existing_content else {}
    existing = existing_data.get("entries", [])
    existing.extend(entries)
    content = yaml.safe_dump({"entries": existing}, sort_keys=False)
    github_store.write_file(ALERT_LOG_PATH, content, message=f"EP alert log +{len(entries)}")


def load_alert_log() -> list:
    content = github_store.read_file(ALERT_LOG_PATH)
    if not content:
        return []
    data = yaml.safe_load(content) or {}
    return data.get("entries", [])
