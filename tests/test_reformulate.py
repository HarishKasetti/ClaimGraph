"""
tests/test_reformulate.py
--------------------------
Unit tests for app.services.reformulate.

All tests that would normally call Ollama are run with the _call_ollama
helper mocked out so the test suite runs without a live Ollama instance.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Module under test
# ---------------------------------------------------------------------------
from app.services.reformulate import (
    _is_question,
    reformulate_to_claim,
    suggest_claims,
)


# ===========================================================================
# _is_question – pure unit tests (no mocking needed)
# ===========================================================================

class TestIsQuestion:
    """Tests for the internal _is_question heuristic."""

    # --- Inputs that SHOULD be detected as questions ---

    def test_ends_with_question_mark(self):
        assert _is_question("Does coffee improve cognitive performance?") is True

    def test_starts_with_does(self):
        assert _is_question("Does caffeine affect sleep") is True

    def test_starts_with_is(self):
        assert _is_question("Is dark matter made of WIMPs?") is True

    def test_starts_with_can(self):
        assert _is_question("Can gut bacteria influence mood?") is True

    def test_starts_with_are(self):
        assert _is_question("Are mRNA vaccines effective against variants?") is True

    def test_starts_with_do(self):
        assert _is_question("Do antidepressants affect neuroplasticity") is True

    def test_starts_with_will(self):
        assert _is_question("Will exercise reduce Alzheimer's risk?") is True

    def test_case_insensitive_prefix(self):
        assert _is_question("IS vitamin D linked to immune function?") is True

    def test_leading_whitespace_trimmed(self):
        assert _is_question("  Does intermittent fasting lower insulin?") is True

    # --- Inputs that should NOT be detected as questions ---

    def test_plain_assertion(self):
        assert _is_question("Caffeine reduces adenosine receptor activity.") is False

    def test_assertion_no_period(self):
        assert _is_question("Exercise reduces the risk of cardiovascular disease") is False

    def test_multi_word_assertion_starting_with_noun(self):
        assert _is_question("Omega-3 fatty acids lower triglyceride levels.") is False

    def test_empty_string(self):
        assert _is_question("") is False

    def test_whitespace_only(self):
        assert _is_question("   ") is False


# ===========================================================================
# reformulate_to_claim
# ===========================================================================

MOCK_REFORMULATED = "Caffeine consumption is associated with reduced adenosine receptor activity."


class TestReformulateToClaimAssertions:
    """When the input is already an assertion, no LLM call should be made."""

    def test_returns_claim_unchanged(self):
        text = "Exercise reduces the risk of cardiovascular disease."
        result = reformulate_to_claim(text)
        assert result == {"claim": text, "reformulated": False}

    def test_reformulated_false(self):
        result = reformulate_to_claim("Aspirin inhibits COX-1 and COX-2 enzymes.")
        assert result["reformulated"] is False

    def test_no_original_key_for_assertion(self):
        result = reformulate_to_claim("Vitamin C is an antioxidant.")
        assert "original" not in result

    def test_assertion_with_numbers(self):
        text = "CRISPR-Cas9 has an off-target edit rate below 1% in most studies."
        result = reformulate_to_claim(text)
        assert result["claim"] == text
        assert result["reformulated"] is False


class TestReformulateToClaimQuestions:
    """When the input is a question, _call_ollama should be invoked."""

    @patch("app.services.reformulate._call_ollama", return_value=MOCK_REFORMULATED)
    def test_question_mark_triggers_reformulation(self, mock_ollama):
        result = reformulate_to_claim("Does caffeine affect sleep quality?")
        mock_ollama.assert_called_once()
        assert result["reformulated"] is True
        assert result["claim"] == MOCK_REFORMULATED

    @patch("app.services.reformulate._call_ollama", return_value=MOCK_REFORMULATED)
    def test_prefix_question_triggers_reformulation(self, mock_ollama):
        result = reformulate_to_claim("Is exercise beneficial for depression?")
        mock_ollama.assert_called_once()
        assert result["reformulated"] is True

    @patch("app.services.reformulate._call_ollama", return_value=MOCK_REFORMULATED)
    def test_original_key_present_for_question(self, mock_ollama):
        q = "Can antibiotics cause antibiotic resistance?"
        result = reformulate_to_claim(q)
        assert result["original"] == q

    @patch("app.services.reformulate._call_ollama", return_value=MOCK_REFORMULATED)
    def test_returned_claim_is_ollama_output(self, mock_ollama):
        result = reformulate_to_claim("Does intermittent fasting reduce insulin?")
        assert result["claim"] == MOCK_REFORMULATED

    @patch("app.services.reformulate._call_ollama", return_value=MOCK_REFORMULATED)
    def test_prompt_contains_user_input(self, mock_ollama):
        question = "Are mRNA vaccines safe?"
        reformulate_to_claim(question)
        call_args = mock_ollama.call_args[0][0]  # first positional argument
        assert question in call_args

    @patch("app.services.reformulate._call_ollama", return_value=MOCK_REFORMULATED)
    def test_prompt_contains_keyword_assertion(self, mock_ollama):
        reformulate_to_claim("Does aspirin prevent heart attacks?")
        call_args = mock_ollama.call_args[0][0]
        assert "assertion" in call_args.lower()

    @patch(
        "app.services.reformulate._call_ollama",
        side_effect=Exception("Ollama not reachable"),
    )
    def test_ollama_error_propagates(self, mock_ollama):
        with pytest.raises(Exception, match="Ollama not reachable"):
            reformulate_to_claim("Is dark matter real?")


# ===========================================================================
# suggest_claims
# ===========================================================================

MOCK_SUGGESTIONS_RAW = (
    "- Regular aerobic exercise reduces LDL cholesterol levels.\n"
    "- Omega-3 supplementation lowers triglycerides in adults.\n"
    "- Sleep deprivation impairs working memory consolidation.\n"
    "- Intermittent fasting decreases fasting insulin concentrations.\n"
)

EXPECTED_SUGGESTIONS = [
    "Regular aerobic exercise reduces LDL cholesterol levels.",
    "Omega-3 supplementation lowers triglycerides in adults.",
    "Sleep deprivation impairs working memory consolidation.",
    "Intermittent fasting decreases fasting insulin concentrations.",
]


class TestSuggestClaims:
    """Tests for suggest_claims."""

    def test_empty_input_returns_empty_list(self):
        assert suggest_claims([]) == []

    @patch("app.services.reformulate._call_ollama", return_value=MOCK_SUGGESTIONS_RAW)
    def test_returns_list_of_strings(self, mock_ollama):
        result = suggest_claims(["Abstract about exercise and cholesterol."])
        assert isinstance(result, list)
        assert all(isinstance(c, str) for c in result)

    @patch("app.services.reformulate._call_ollama", return_value=MOCK_SUGGESTIONS_RAW)
    def test_returns_at_most_four_claims(self, mock_ollama):
        result = suggest_claims(["Abstract 1", "Abstract 2", "Abstract 3"])
        assert len(result) <= 4

    @patch("app.services.reformulate._call_ollama", return_value=MOCK_SUGGESTIONS_RAW)
    def test_parses_dash_prefixed_lines(self, mock_ollama):
        result = suggest_claims(["Some abstract text."])
        assert result == EXPECTED_SUGGESTIONS

    @patch("app.services.reformulate._call_ollama", return_value=MOCK_SUGGESTIONS_RAW)
    def test_abstracts_passed_to_ollama(self, mock_ollama):
        abstracts = ["First abstract.", "Second abstract."]
        suggest_claims(abstracts)
        prompt = mock_ollama.call_args[0][0]
        assert "First abstract." in prompt
        assert "Second abstract." in prompt

    @patch(
        "app.services.reformulate._call_ollama",
        return_value=(
            "Regular aerobic exercise reduces LDL cholesterol levels.\n"
            "Omega-3 supplementation lowers triglycerides in adults.\n"
            "Sleep deprivation impairs working memory.\n"
        ),
    )
    def test_fallback_parsing_no_bullets(self, mock_ollama):
        """Model returns plain lines without bullet markers."""
        result = suggest_claims(["Abstract text."])
        assert len(result) == 3
        assert result[0] == "Regular aerobic exercise reduces LDL cholesterol levels."

    @patch(
        "app.services.reformulate._call_ollama",
        return_value=(
            "- Claim 1.\n- Claim 2.\n- Claim 3.\n- Claim 4.\n- Claim 5.\n"
        ),
    )
    def test_caps_at_four_even_if_model_returns_more(self, mock_ollama):
        result = suggest_claims(["Abstract."])
        assert len(result) == 4

    @patch(
        "app.services.reformulate._call_ollama",
        side_effect=Exception("connection refused"),
    )
    def test_ollama_error_propagates(self, mock_ollama):
        with pytest.raises(Exception, match="connection refused"):
            suggest_claims(["Some abstract."])
