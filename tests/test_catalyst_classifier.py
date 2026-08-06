"""bullarullaep/catalyst_classifier.py, tested against a MOCKED Anthropic
client -- no live API calls happen in this test file, per this project's
own convention of never hitting external network/paid APIs from the test
suite. Covers: the graceful-degradation paths (no key, no headlines,
package missing, API error, malformed tool response) and the happy path
with a mocked structured tool-call response.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from bullarullaep.catalyst_classifier import CatalystClassification, classify_catalyst


def test_no_headlines_degrades_gracefully():
    result = classify_catalyst("XYZ", [], api_key="fake-key")
    assert result.catalyst_type == "unclear_or_noise"
    assert result.confidence == 0.0
    assert result.is_genuine is False


def test_no_api_key_degrades_gracefully(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = classify_catalyst("XYZ", ["XYZ beats on strong demand"], api_key=None)
    assert result.catalyst_type == "unclear_or_noise"
    assert result.confidence == 0.0
    assert "no ANTHROPIC_API_KEY" in result.rationale


def test_api_error_degrades_gracefully():
    with patch("anthropic.Anthropic") as mock_anthropic:
        mock_anthropic.return_value.messages.create.side_effect = RuntimeError("connection reset")
        result = classify_catalyst("XYZ", ["XYZ beats on strong demand"], api_key="fake-key")
    assert result.catalyst_type == "unclear_or_noise"
    assert result.confidence == 0.0
    assert "classifier error" in result.rationale


def _mock_tool_use_response(catalyst_type: str, confidence: float, rationale: str):
    block = MagicMock()
    block.type = "tool_use"
    block.name = "classify_catalyst"
    block.input = {"catalyst_type": catalyst_type, "confidence": confidence, "rationale": rationale}
    response = MagicMock()
    response.content = [block]
    return response


def test_happy_path_returns_parsed_classification():
    with patch("anthropic.Anthropic") as mock_anthropic:
        mock_anthropic.return_value.messages.create.return_value = _mock_tool_use_response(
            "earnings_beat", 0.87, "Beat EPS by 25% with raised guidance."
        )
        result = classify_catalyst("NVDA", ["NVDA crushes Q2 estimates"], gap_pct=18.0, api_key="fake-key")

    assert isinstance(result, CatalystClassification)
    assert result.catalyst_type == "earnings_beat"
    assert result.confidence == 0.87
    assert result.is_genuine is True


def test_invalid_catalyst_type_from_model_falls_back_to_unclear():
    with patch("anthropic.Anthropic") as mock_anthropic:
        mock_anthropic.return_value.messages.create.return_value = _mock_tool_use_response(
            "totally_made_up_type", 0.9, "hallucinated enum value"
        )
        result = classify_catalyst("XYZ", ["some headline"], api_key="fake-key")
    assert result.catalyst_type == "unclear_or_noise"


def test_confidence_clamped_to_zero_one_range():
    with patch("anthropic.Anthropic") as mock_anthropic:
        mock_anthropic.return_value.messages.create.return_value = _mock_tool_use_response(
            "ma_or_contract", 1.5, "overconfident model"
        )
        result = classify_catalyst("XYZ", ["some headline"], api_key="fake-key")
    assert result.confidence == 1.0


def test_no_tool_call_in_response_degrades_gracefully():
    response = MagicMock()
    response.content = []
    with patch("anthropic.Anthropic") as mock_anthropic:
        mock_anthropic.return_value.messages.create.return_value = response
        result = classify_catalyst("XYZ", ["some headline"], api_key="fake-key")
    assert result.catalyst_type == "unclear_or_noise"
    assert result.confidence == 0.0
