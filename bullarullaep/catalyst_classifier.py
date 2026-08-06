"""LLM-based news-catalyst classifier -- answers "is this a genuine
re-rating catalyst (earnings beat, FDA approval, M&A, major contract) or
just noise (routine 8-K, dilution, reverse split, analyst chatter)?" for a
candidate's most recent headlines.

A SCORING BONUS, never a hard gate -- bullarullaep/detector.py only ever
adds `catalyst_bonus_weight * confidence` on top of the EOD hard gates, it
never excludes a match on a low/failed classification. Same philosophy the
parent project already uses for growth/EPS-beat scoring: a wrong call from
an imperfect signal should never silently kill a real trade.

Degrades gracefully: no ANTHROPIC_API_KEY, a network error, or a malformed
response all fall back to an "unclear_or_noise"/confidence-0.0
classification rather than raising -- a scan should never crash because the
LLM layer had a bad day. No live API calls happen in tests; see
tests/test_catalyst_classifier.py for the mocked contract.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

CATALYST_TYPES = (
    "earnings_beat", "fda_approval", "ma_or_contract", "guidance_raise",
    "other_positive", "unclear_or_noise", "negative",
)

# Haiku by default -- this runs automatically, twice a trading day, across
# every candidate that clears the EOD hard gates; a cheap, fast model is the
# right default for "classify this headline," not a reasoning-heavy task.
# Pass model= to upgrade to Sonnet for better judgment if it's worth the cost.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

CLASSIFY_TOOL = {
    "name": "classify_catalyst",
    "description": "Classify whether a stock's recent news is a genuine re-rating catalyst worth trading a gap-up breakout on.",
    "input_schema": {
        "type": "object",
        "properties": {
            "catalyst_type": {"type": "string", "enum": list(CATALYST_TYPES)},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "rationale": {"type": "string", "description": "One sentence explaining the call."},
        },
        "required": ["catalyst_type", "confidence", "rationale"],
    },
}


@dataclass
class CatalystClassification:
    catalyst_type: str
    confidence: float
    rationale: str

    @property
    def is_genuine(self) -> bool:
        return self.catalyst_type not in ("unclear_or_noise", "negative")


def _unknown(reason: str) -> CatalystClassification:
    return CatalystClassification(catalyst_type="unclear_or_noise", confidence=0.0, rationale=reason)


def _resolve_api_key(explicit: Optional[str] = None) -> Optional[str]:
    """Same fallback chain as data.fmp_client._resolve_api_key /
    notifications._resolve_ntfy_topic: explicit arg > Streamlit secrets (not
    used by this headless project today, but kept for consistency) > env."""
    if explicit:
        return explicit
    try:
        import streamlit as st

        if hasattr(st, "secrets") and "ANTHROPIC_API_KEY" in st.secrets:
            return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY")


def _build_prompt(symbol: str, headlines: list, gap_pct: Optional[float], is_earnings_day: bool) -> str:
    context_lines = [f"Symbol: {symbol}"]
    if gap_pct is not None:
        context_lines.append(f"Today's gap: {gap_pct:+.1f}%")
    context_lines.append(f"Coincides with an earnings report: {'yes' if is_earnings_day else 'no'}")
    context_lines.append("Recent headlines (most recent first):")
    for h in headlines[:10]:
        context_lines.append(f"- {h}")
    return (
        "You are screening for Qullamaggie/Stockbee-style episodic pivots: a stock that gaps up "
        "10%+ on a REAL re-rating catalyst -- a genuine earnings beat with strong growth, FDA/"
        "regulatory approval, M&A, a major contract win, or a significant guidance raise. Routine "
        "8-Ks, dilution/share offerings, reverse splits, analyst-only commentary, or vague/"
        "promotional news are NOT genuine catalysts, even if they coincide with a price move.\n\n"
        + "\n".join(context_lines)
        + "\n\nClassify this using the classify_catalyst tool."
    )


def classify_catalyst(
    symbol: str,
    headlines: list,
    gap_pct: Optional[float] = None,
    is_earnings_day: bool = False,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> CatalystClassification:
    """`headlines` is a list of recent headline strings (most-recent-first,
    e.g. from data/fmp_client.py::stock_news(), `title` field). Returns a
    CatalystClassification -- confidence=0.0/catalyst_type="unclear_or_noise"
    on ANY failure (no headlines, no key, package missing, API error,
    malformed response); never raises."""
    if not headlines:
        return _unknown("no headlines available")

    resolved_key = _resolve_api_key(api_key)
    if not resolved_key:
        return _unknown("no ANTHROPIC_API_KEY configured")

    try:
        import anthropic
    except ImportError:
        return _unknown("anthropic package not installed")

    prompt = _build_prompt(symbol, headlines, gap_pct, is_earnings_day)

    try:
        client = anthropic.Anthropic(api_key=resolved_key)
        response = client.messages.create(
            model=model,
            max_tokens=300,
            tools=[CLASSIFY_TOOL],
            tool_choice={"type": "tool", "name": "classify_catalyst"},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "classify_catalyst":
                data = block.input
                catalyst_type = data.get("catalyst_type", "unclear_or_noise")
                if catalyst_type not in CATALYST_TYPES:
                    catalyst_type = "unclear_or_noise"
                return CatalystClassification(
                    catalyst_type=catalyst_type,
                    confidence=max(0.0, min(1.0, float(data.get("confidence", 0.0)))),
                    rationale=str(data.get("rationale", "")),
                )
        return _unknown("model did not return a tool call")
    except Exception as exc:
        return _unknown(f"classifier error: {exc}")
