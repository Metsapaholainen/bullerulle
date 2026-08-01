"""GitHub-repo-backed persistence for small, frequently-edited user data
(the Paper Trading journal, the Sell Alerts watchlist, the general
Watchlist) -- so it survives Streamlit Community Cloud's ephemeral
container disk, which wipes any locally-written file on every sleep/wake
or redeploy (see DEPLOY.md).

Data lives as small YAML files on a dedicated branch of this project's own
GitHub repo, read/written via the Contents API -- deliberately NOT
`master`, so these frequent small commits never trigger a Streamlit Cloud
auto-redeploy of the live app (Cloud only tracks `master`).

Falls back to plain local file I/O when no GITHUB_TOKEN is configured, so
local dev/testing needs no PAT just to run the app -- and also as a
last-resort safety net if a GitHub API call fails, so a transient network
error never simply drops the user's edit.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

GITHUB_REPO = "Metsapaholainen/bullerulle"
GITHUB_BRANCH = "data-store"
GITHUB_API_BASE = "https://api.github.com"
_TIMEOUT = 10

# Process-lifetime caches so a save doesn't re-verify the branch and
# re-fetch the current file's sha on every single click -- that was costing
# 3 sequential GitHub API round-trips per write (branch check + get-sha +
# commit) when at most the commit itself is actually necessary once this
# process has already confirmed the branch exists and knows the last sha.
_branch_ready = False
_sha_cache: dict = {}


def _resolve_github_token(explicit: Optional[str] = None) -> Optional[str]:
    """Explicit arg > Streamlit Cloud secrets > local .env/os.environ -- same
    fallback chain as data.fmp_client._resolve_api_key."""
    if explicit:
        return explicit
    try:
        import streamlit as st

        if hasattr(st, "secrets") and "GITHUB_TOKEN" in st.secrets:
            return st.secrets["GITHUB_TOKEN"]
    except Exception:
        pass
    return os.environ.get("GITHUB_TOKEN")


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}


def _local_path(path: str) -> Path:
    return Path(__file__).parent / path


def _read_local(path: str) -> Optional[str]:
    local_path = _local_path(path)
    if not local_path.exists():
        return None
    return local_path.read_text(encoding="utf-8")


def _write_local(path: str, content: str) -> None:
    local_path = _local_path(path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(content, encoding="utf-8")


def _warn(message: str) -> None:
    try:
        import streamlit as st

        st.warning(message)
    except Exception:
        pass


def _branch_exists(token: str) -> bool:
    resp = requests.get(
        f"{GITHUB_API_BASE}/repos/{GITHUB_REPO}/git/ref/heads/{GITHUB_BRANCH}",
        headers=_headers(token), timeout=_TIMEOUT,
    )
    return resp.status_code == 200


def _ensure_branch_exists(token: str) -> None:
    global _branch_ready
    if _branch_ready:
        return
    if _branch_exists(token):
        _branch_ready = True
        return
    # Branch doesn't exist yet -- create it pointing at master's current commit.
    resp = requests.get(
        f"{GITHUB_API_BASE}/repos/{GITHUB_REPO}/git/ref/heads/master",
        headers=_headers(token), timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    master_sha = resp.json()["object"]["sha"]
    create_resp = requests.post(
        f"{GITHUB_API_BASE}/repos/{GITHUB_REPO}/git/refs",
        headers=_headers(token), timeout=_TIMEOUT,
        json={"ref": f"refs/heads/{GITHUB_BRANCH}", "sha": master_sha},
    )
    if create_resp.status_code not in (201, 422):  # 422 = ref already exists (race), fine
        create_resp.raise_for_status()
    _branch_ready = True


def _get_file(token: str, path: str):
    """Returns (content_str, sha), or (None, None) if the file doesn't exist yet on this branch."""
    resp = requests.get(
        f"{GITHUB_API_BASE}/repos/{GITHUB_REPO}/contents/{path}",
        headers=_headers(token), params={"ref": GITHUB_BRANCH}, timeout=_TIMEOUT,
    )
    if resp.status_code == 404:
        return None, None
    resp.raise_for_status()
    data = resp.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    return content, data["sha"]


def read_file(path: str, token: Optional[str] = None) -> Optional[str]:
    """Returns the file's text content. Reads from the GitHub data-store
    branch when a token is configured, otherwise (or on any API failure)
    falls back to a local file at the same relative path."""
    resolved = _resolve_github_token(token)
    if not resolved:
        return _read_local(path)
    try:
        _ensure_branch_exists(resolved)
        content, sha = _get_file(resolved, path)
        if content is not None:
            if sha:
                _sha_cache[path] = sha
            return content
        return _read_local(path)  # not on GitHub yet -- e.g. before the very first write
    except Exception:
        return _read_local(path)


def write_file(path: str, content: str, message: str, token: Optional[str] = None) -> None:
    """Commits `content` to `path` on the GitHub data-store branch when a
    token is configured. On any failure (or no token), writes to a local
    file instead -- so an edit is never silently lost, though a local-only
    write won't survive a Streamlit Cloud restart, which is surfaced as a
    warning so that's never mistaken for durable storage.

    Skips the branch-exists check and the current-file-sha lookup once this
    process already knows them (`_branch_ready`/`_sha_cache`) -- a save used
    to cost 3 sequential GitHub API round-trips (branch check, get-sha,
    commit); after the first successful write for a given path it's just the
    commit itself. If the cached sha turns out to be stale (someone edited
    the file directly on GitHub, or another process wrote first), the PUT
    comes back 409/422 and this refetches the real sha once and retries."""
    resolved = _resolve_github_token(token)
    if not resolved:
        _write_local(path, content)
        return
    try:
        _ensure_branch_exists(resolved)
        sha = _sha_cache.get(path)
        for attempt in range(2):
            body = {
                "message": message,
                "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
                "branch": GITHUB_BRANCH,
            }
            if sha:
                body["sha"] = sha
            resp = requests.put(
                f"{GITHUB_API_BASE}/repos/{GITHUB_REPO}/contents/{path}",
                headers=_headers(resolved), timeout=_TIMEOUT + 5, json=body,
            )
            if resp.status_code in (409, 422) and attempt == 0:
                _, sha = _get_file(resolved, path)
                continue
            resp.raise_for_status()
            new_sha = resp.json().get("content", {}).get("sha")
            if new_sha:
                _sha_cache[path] = new_sha
            return
    except Exception as exc:
        _sha_cache.pop(path, None)
        _write_local(path, content)
        _warn(f"Couldn't save to GitHub ({exc}) -- saved locally instead, which won't survive a Cloud restart.")
