"""
tests/test_contradiction.py
----------------------------
Unit tests for app/services/contradiction.py — Phase 6 Stage 6.

All tests are fully isolated:
  - No Ollama calls (reformulate._call_ollama patched)
  - No MongoDB connections (motor patched)
  - No DeBERTa downloads (stance.get_pipeline patched)
  - No asyncio.run side effects (patched where needed)

Test Groups
-----------
TestExtractMainFinding     -- Ollama call, cache, fallback, cleaning
TestCheckContradictions    -- pair generation, threshold, persistence, empty
TestGenerateContradictionMap -- Plotly JSON structure, zero-contradiction case
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
_PAPER_IDS = ["paper_001", "paper_002", "paper_003"]

_ABSTRACT_A = (
    "Low-dose aspirin significantly reduces the incidence of myocardial "
    "infarction in high-risk adults (RR 0.78, p<0.001)."
)
_ABSTRACT_B = (
    "This RCT found no significant reduction in cardiovascular events with "
    "aspirin compared to placebo, while major bleeding events increased."
)
_ABSTRACT_C = (
    "Aspirin was first synthesised by Felix Hoffmann at Bayer in 1897 and "
    "commercially introduced in 1899."
)

_FINDING_A = "Low-dose aspirin reduces myocardial infarction risk by 22%."
_FINDING_B = "Aspirin shows no cardiovascular benefit and increases bleeding."
_FINDING_C = "Aspirin was synthesised in 1897 by Felix Hoffmann."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipe_result(label: str = "refute", score: float = 0.85):
    """Return a mock DeBERTa pipeline result dict."""
    other_labels = [l for l in ["support", "refute", "no_stance"] if l != label]
    remaining = (1.0 - score) / len(other_labels)
    return {
        "labels": [label] + other_labels,
        "scores": [score] + [remaining] * len(other_labels),
    }


def _make_pipe(label: str = "refute", score: float = 0.85):
    """Return a callable mock pipeline that always returns the given label."""
    def _pipe(text, candidate_labels, hypothesis_template, multi_label):
        return _make_pipe_result(label, score)
    return _pipe


# ---------------------------------------------------------------------------
# TestExtractMainFinding
# ---------------------------------------------------------------------------

class TestExtractMainFinding:
    """Tests for extract_main_finding(paper_id, abstract=...)."""

    def _clear_cache(self):
        import app.services.contradiction as cmod
        cmod._finding_cache.clear()

    def test_calls_ollama_with_abstract(self) -> None:
        """extract_main_finding must pass the abstract to Ollama."""
        self._clear_cache()
        captured_prompts: list[str] = []

        def _mock_ollama(prompt: str) -> str:
            captured_prompts.append(prompt)
            return _FINDING_A

        with patch("app.services.contradiction._call_ollama", side_effect=_mock_ollama):
            from app.services.contradiction import extract_main_finding
            result = extract_main_finding("p_test", abstract=_ABSTRACT_A)

        assert result == _FINDING_A
        assert len(captured_prompts) == 1
        assert _ABSTRACT_A[:100] in captured_prompts[0]

    def test_returns_cached_result_on_second_call(self) -> None:
        """Second call for the same paper_id must NOT call Ollama again."""
        self._clear_cache()
        call_count = {"n": 0}

        def _mock_ollama(prompt: str) -> str:
            call_count["n"] += 1
            return _FINDING_B

        with patch("app.services.contradiction._call_ollama", side_effect=_mock_ollama):
            from app.services.contradiction import extract_main_finding
            r1 = extract_main_finding("p_cached", abstract=_ABSTRACT_B)
            r2 = extract_main_finding("p_cached", abstract=_ABSTRACT_B)

        assert r1 == r2 == _FINDING_B
        assert call_count["n"] == 1, "Ollama should only be called once (cache hit on 2nd)"

    def test_fallback_on_ollama_failure(self) -> None:
        """When Ollama raises, the first 200 chars of abstract must be returned."""
        self._clear_cache()

        def _fail_ollama(prompt: str) -> str:
            raise ConnectionError("Ollama not running")

        with patch("app.services.contradiction._call_ollama", side_effect=_fail_ollama):
            from app.services.contradiction import extract_main_finding
            result = extract_main_finding("p_fail", abstract=_ABSTRACT_A)

        assert len(result) <= 205  # 200 chars + possible ellipsis
        assert result  # must not be empty

    def test_empty_abstract_returns_empty_string(self) -> None:
        """With no abstract available, result must be an empty string."""
        self._clear_cache()

        with patch("app.services.contradiction._call_ollama") as mock_ollama, \
             patch("app.services.contradiction.asyncio") as mock_asyncio:
            mock_asyncio.run = MagicMock(return_value={})
            from app.services.contradiction import extract_main_finding
            result = extract_main_finding("p_empty", abstract="")

        assert result == ""
        mock_ollama.assert_not_called()

    def test_finding_cleaned_of_numbering(self) -> None:
        """Ollama responses starting with '1.' or '1)' must be stripped."""
        self._clear_cache()

        def _numbered_response(prompt: str) -> str:
            return "1. Low-dose aspirin reduces cardiovascular risk."

        with patch("app.services.contradiction._call_ollama", side_effect=_numbered_response):
            from app.services.contradiction import extract_main_finding
            result = extract_main_finding("p_numbered", abstract=_ABSTRACT_A)

        assert not result.startswith("1.")
        assert not result.startswith("1)")

    def test_only_first_sentence_returned(self) -> None:
        """If Ollama returns multiple sentences, only the first is used."""
        self._clear_cache()

        def _multi_sentence(prompt: str) -> str:
            return (
                "Aspirin reduces cardiovascular risk. "
                "This was demonstrated in a large RCT. "
                "Further studies are needed."
            )

        with patch("app.services.contradiction._call_ollama", side_effect=_multi_sentence):
            from app.services.contradiction import extract_main_finding
            result = extract_main_finding("p_multi", abstract=_ABSTRACT_A)

        sentence_count = len([s for s in result.split(". ") if s.strip()])
        assert sentence_count <= 2, (
            f"Expected at most 1-2 sentence fragments, got: {result!r}"
        )

    def test_returns_string_type(self) -> None:
        """Return type must always be str."""
        self._clear_cache()

        with patch("app.services.contradiction._call_ollama", return_value=_FINDING_C):
            from app.services.contradiction import extract_main_finding
            result = extract_main_finding("p_type", abstract=_ABSTRACT_C)

        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# TestCheckContradictions
# ---------------------------------------------------------------------------

class TestCheckContradictions:
    """Tests for check_contradictions(topic_id, paper_ids=...) -> list[dict]."""

    def _clear_cache(self):
        import app.services.contradiction as cmod
        cmod._finding_cache.clear()

    def _preload_findings(self, findings: dict[str, str]) -> None:
        import app.services.contradiction as cmod
        cmod._finding_cache.update(findings)

    def test_flags_refute_above_threshold(self) -> None:
        """Pairs where DeBERTa returns refute > 0.7 must be flagged."""
        self._clear_cache()
        self._preload_findings({
            "paper_A": _FINDING_A,
            "paper_B": _FINDING_B,
        })

        mock_pipe = _make_pipe(label="refute", score=0.85)

        with patch("app.services.contradiction.get_pipeline", return_value=mock_pipe), \
             patch("app.services.contradiction.asyncio.run", return_value=None):

            from app.services.contradiction import check_contradictions
            result = check_contradictions(
                "topic_x", paper_ids=["paper_A", "paper_B"]
            )

        assert len(result) == 1
        assert result[0]["type"] == "contradiction"
        assert result[0]["paper_a_id"] in ("paper_A", "paper_B")
        assert result[0]["paper_b_id"] in ("paper_A", "paper_B")
        assert result[0]["confidence"] > 0.7

    def test_does_not_flag_below_threshold(self) -> None:
        """Pairs where refute score <= 0.7 must NOT be flagged."""
        self._clear_cache()
        self._preload_findings({
            "paper_A": _FINDING_A,
            "paper_B": _FINDING_B,
        })

        mock_pipe = _make_pipe(label="refute", score=0.65)  # below threshold

        with patch("app.services.contradiction.get_pipeline", return_value=mock_pipe), \
             patch("app.services.contradiction.asyncio.run", return_value=None):

            from app.services.contradiction import check_contradictions
            result = check_contradictions(
                "topic_y", paper_ids=["paper_A", "paper_B"]
            )

        assert result == []

    def test_does_not_flag_support_label(self) -> None:
        """Pairs classified as 'support' must never be flagged."""
        self._clear_cache()
        self._preload_findings({
            "paper_A": _FINDING_A,
            "paper_B": _FINDING_C,
        })

        mock_pipe = _make_pipe(label="support", score=0.95)

        with patch("app.services.contradiction.get_pipeline", return_value=mock_pipe), \
             patch("app.services.contradiction.asyncio.run", return_value=None):

            from app.services.contradiction import check_contradictions
            result = check_contradictions(
                "topic_supp", paper_ids=["paper_A", "paper_B"]
            )

        assert result == []

    def test_does_not_flag_no_stance_label(self) -> None:
        """Pairs classified as 'no_stance' must never be flagged."""
        self._clear_cache()
        self._preload_findings({
            "paper_A": _FINDING_A,
            "paper_B": _FINDING_C,
        })

        mock_pipe = _make_pipe(label="no_stance", score=0.90)

        with patch("app.services.contradiction.get_pipeline", return_value=mock_pipe), \
             patch("app.services.contradiction.asyncio.run", return_value=None):

            from app.services.contradiction import check_contradictions
            result = check_contradictions(
                "topic_ns", paper_ids=["paper_A", "paper_B"]
            )

        assert result == []

    def test_correct_number_of_pairs_checked(self) -> None:
        """With n papers, C(n,2) pairs must be checked."""
        self._clear_cache()
        findings = {f"paper_{i}": f"Finding number {i}." for i in range(5)}
        self._preload_findings(findings)

        call_count = {"n": 0}
        def _counting_pipe(text, candidate_labels, hypothesis_template, multi_label):
            call_count["n"] += 1
            return _make_pipe_result("no_stance", 0.80)

        with patch("app.services.contradiction.get_pipeline", return_value=_counting_pipe), \
             patch("app.services.contradiction.asyncio.run", return_value=None):

            from app.services.contradiction import check_contradictions
            check_contradictions("topic_pairs", paper_ids=list(findings.keys()))

        # C(5,2) = 10 pairs — DeBERTa called once per pair
        assert call_count["n"] == 10, (
            f"Expected 10 DeBERTa calls for 5 papers, got {call_count['n']}"
        )

    def test_empty_paper_ids_returns_empty_list(self) -> None:
        """With no paper_ids, result must be an empty list."""
        self._clear_cache()

        with patch("app.services.contradiction.get_pipeline"):
            from app.services.contradiction import check_contradictions
            result = check_contradictions("topic_empty", paper_ids=[])

        assert result == []

    def test_result_dict_has_required_keys(self) -> None:
        """Each contradiction dict must have all required keys."""
        self._clear_cache()
        self._preload_findings({
            "paper_A": _FINDING_A,
            "paper_B": _FINDING_B,
        })

        mock_pipe = _make_pipe(label="refute", score=0.88)

        with patch("app.services.contradiction.get_pipeline", return_value=mock_pipe), \
             patch("app.services.contradiction.asyncio.run", return_value=None):

            from app.services.contradiction import check_contradictions
            result = check_contradictions(
                "topic_keys", paper_ids=["paper_A", "paper_B"]
            )

        assert len(result) == 1
        required_keys = {
            "paper_a_id", "paper_b_id", "finding_a", "finding_b",
            "type", "confidence", "topic_id",
        }
        for key in required_keys:
            assert key in result[0], f"Missing key: {key!r}"

    def test_single_paper_no_pairs_no_crash(self) -> None:
        """With only one paper, no pairs exist — must return empty without crash."""
        self._clear_cache()
        self._preload_findings({"paper_solo": _FINDING_A})

        with patch("app.services.contradiction.get_pipeline"):
            from app.services.contradiction import check_contradictions
            result = check_contradictions("topic_solo", paper_ids=["paper_solo"])

        assert result == []

    def test_skips_pairs_with_empty_finding(self) -> None:
        """Pairs where one paper has an empty finding must be skipped silently."""
        self._clear_cache()
        # paper_B has empty finding
        self._preload_findings({"paper_A": _FINDING_A, "paper_B": ""})

        call_count = {"n": 0}
        def _counting_pipe(text, candidate_labels, hypothesis_template, multi_label):
            call_count["n"] += 1
            return _make_pipe_result("refute", 0.90)

        with patch("app.services.contradiction.get_pipeline", return_value=_counting_pipe), \
             patch("app.services.contradiction.asyncio.run", return_value=None):

            from app.services.contradiction import check_contradictions
            result = check_contradictions(
                "topic_skip", paper_ids=["paper_A", "paper_B"]
            )

        assert result == []
        assert call_count["n"] == 0  # DeBERTa never called

    def test_confidence_is_float_in_unit_interval(self) -> None:
        """Flagged pair confidence must be a float in [0, 1]."""
        self._clear_cache()
        self._preload_findings({
            "paper_A": _FINDING_A,
            "paper_B": _FINDING_B,
        })

        mock_pipe = _make_pipe(label="refute", score=0.81)

        with patch("app.services.contradiction.get_pipeline", return_value=mock_pipe), \
             patch("app.services.contradiction.asyncio.run", return_value=None):

            from app.services.contradiction import check_contradictions
            result = check_contradictions(
                "topic_conf", paper_ids=["paper_A", "paper_B"]
            )

        assert len(result) == 1
        conf = result[0]["confidence"]
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# TestGenerateContradictionMap
# ---------------------------------------------------------------------------

class TestGenerateContradictionMap:
    """Tests for generate_contradiction_map(topic_id, paper_ids=...) -> str."""

    def _clear_cache(self):
        import app.services.contradiction as cmod
        cmod._finding_cache.clear()

    def _preload_findings(self, findings: dict[str, str]) -> None:
        import app.services.contradiction as cmod
        cmod._finding_cache.update(findings)

    def test_returns_valid_json_string(self) -> None:
        """Output must be a non-empty, valid JSON string."""
        self._clear_cache()
        self._preload_findings({"p1": _FINDING_A, "p2": _FINDING_B})
        mock_pipe = _make_pipe("no_stance", 0.80)

        with patch("app.services.contradiction.get_pipeline", return_value=mock_pipe), \
             patch("app.services.contradiction.asyncio.run", return_value=None):
            from app.services.contradiction import generate_contradiction_map
            result = generate_contradiction_map(
                "topic_json",
                paper_ids=["p1", "p2"],
                paper_titles={"p1": "Aspirin cardiovascular benefit study", "p2": "Aspirin harm RCT"},
            )

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_plotly_data_and_layout_keys(self) -> None:
        """Plotly JSON must have 'data' and 'layout' keys."""
        self._clear_cache()
        self._preload_findings({"p1": _FINDING_A, "p2": _FINDING_B})
        mock_pipe = _make_pipe("no_stance", 0.80)

        with patch("app.services.contradiction.get_pipeline", return_value=mock_pipe), \
             patch("app.services.contradiction.asyncio.run", return_value=None):
            from app.services.contradiction import generate_contradiction_map
            result = generate_contradiction_map(
                "topic_keys",
                paper_ids=["p1", "p2"],
                paper_titles={"p1": "Title One Study Two", "p2": "Title Two Study Three"},
            )

        parsed = json.loads(result)
        assert "data" in parsed
        assert "layout" in parsed
        assert len(parsed["data"]) >= 1

    def test_heatmap_trace_type(self) -> None:
        """The first trace must be a heatmap."""
        self._clear_cache()
        self._preload_findings({"p1": _FINDING_A, "p2": _FINDING_B})
        mock_pipe = _make_pipe("support", 0.90)

        with patch("app.services.contradiction.get_pipeline", return_value=mock_pipe), \
             patch("app.services.contradiction.asyncio.run", return_value=None):
            from app.services.contradiction import generate_contradiction_map
            result = generate_contradiction_map(
                "topic_hm",
                paper_ids=["p1", "p2"],
                paper_titles={"p1": "Paper One Study", "p2": "Paper Two Study"},
            )

        parsed = json.loads(result)
        trace = parsed["data"][0]
        assert trace["type"] == "heatmap"

    def test_axis_labels_are_short_titles_not_ids(self) -> None:
        """Axis labels must be short human-readable titles, not raw paper_ids."""
        self._clear_cache()
        self._preload_findings({
            "raw_id_001": _FINDING_A,
            "raw_id_002": _FINDING_B,
        })
        titles = {
            "raw_id_001": "Aspirin Reduces Cardiovascular Risk in Adults",
            "raw_id_002": "Aspirin Shows No Benefit Large RCT",
        }
        mock_pipe = _make_pipe("no_stance", 0.75)

        with patch("app.services.contradiction.get_pipeline", return_value=mock_pipe), \
             patch("app.services.contradiction.asyncio.run", return_value=None):
            from app.services.contradiction import generate_contradiction_map
            result = generate_contradiction_map(
                "topic_labels",
                paper_ids=["raw_id_001", "raw_id_002"],
                paper_titles=titles,
            )

        parsed = json.loads(result)
        x_labels = parsed["data"][0]["x"]
        y_labels = parsed["data"][0]["y"]

        for lbl in x_labels + y_labels:
            assert "raw_id" not in lbl, (
                f"Raw paper_id found in axis label: {lbl!r}"
            )

    def test_all_grey_on_zero_contradictions(self) -> None:
        """When all pairs are no_stance, all z-values must be 0 (grey)."""
        self._clear_cache()
        findings = {
            "p1": "Study one finds X.",
            "p2": "Study two finds Y.",
            "p3": "Study three finds Z.",
        }
        self._preload_findings(findings)
        titles = {k: f"Title {k}" for k in findings}
        mock_pipe = _make_pipe("no_stance", 0.85)

        with patch("app.services.contradiction.get_pipeline", return_value=mock_pipe), \
             patch("app.services.contradiction.asyncio.run", return_value=None):
            from app.services.contradiction import generate_contradiction_map
            result = generate_contradiction_map(
                "topic_grey",
                paper_ids=list(findings.keys()),
                paper_titles=titles,
            )

        parsed = json.loads(result)
        z_matrix = parsed["data"][0]["z"]

        # All off-diagonal values should be 0 (no_stance)
        n = len(z_matrix)
        for i in range(n):
            for j in range(n):
                if i != j:
                    assert z_matrix[i][j] == 0.0, (
                        f"Expected 0.0 for no_stance at ({i},{j}), got {z_matrix[i][j]}"
                    )

    def test_contradiction_produces_negative_z_value(self) -> None:
        """A refute pair must have z=-1.0 in the matrix."""
        self._clear_cache()
        self._preload_findings({"p1": _FINDING_A, "p2": _FINDING_B})
        titles = {"p1": "Paper One Finding", "p2": "Paper Two Contradiction"}

        call_count = {"n": 0}
        def _alternating_pipe(text, candidate_labels, hypothesis_template, multi_label):
            # First call returns refute, symmetrical second call mirrors
            call_count["n"] += 1
            return _make_pipe_result("refute", 0.88)

        with patch("app.services.contradiction.get_pipeline", return_value=_alternating_pipe), \
             patch("app.services.contradiction.asyncio.run", return_value=None):
            from app.services.contradiction import generate_contradiction_map
            result = generate_contradiction_map(
                "topic_refute",
                paper_ids=["p1", "p2"],
                paper_titles=titles,
            )

        parsed = json.loads(result)
        z_matrix = parsed["data"][0]["z"]

        # Off-diagonal should be -1.0 for refute
        off_diagonal_vals = [
            z_matrix[i][j]
            for i in range(len(z_matrix))
            for j in range(len(z_matrix[i]))
            if i != j
        ]
        assert any(v == -1.0 for v in off_diagonal_vals), (
            f"Expected at least one -1.0 (refute) in z_matrix, got: {off_diagonal_vals}"
        )

    def test_support_produces_positive_z_value(self) -> None:
        """A support pair must have z=+1.0 in the matrix."""
        self._clear_cache()
        self._preload_findings({"p1": _FINDING_A, "p2": _FINDING_C})
        titles = {"p1": "Paper Support Finding", "p2": "Paper Background Context"}
        mock_pipe = _make_pipe("support", 0.92)

        with patch("app.services.contradiction.get_pipeline", return_value=mock_pipe), \
             patch("app.services.contradiction.asyncio.run", return_value=None):
            from app.services.contradiction import generate_contradiction_map
            result = generate_contradiction_map(
                "topic_support",
                paper_ids=["p1", "p2"],
                paper_titles=titles,
            )

        parsed = json.loads(result)
        z_matrix = parsed["data"][0]["z"]

        off_diagonal_vals = [
            z_matrix[i][j]
            for i in range(len(z_matrix))
            for j in range(len(z_matrix[i]))
            if i != j
        ]
        assert any(v == 1.0 for v in off_diagonal_vals), (
            f"Expected at least one +1.0 (support) in z_matrix, got: {off_diagonal_vals}"
        )

    def test_matrix_is_nxn(self) -> None:
        """The z-matrix must be square (n x n) for n papers."""
        self._clear_cache()
        n = 4
        findings = {f"p{i}": f"Finding {i}." for i in range(n)}
        self._preload_findings(findings)
        titles = {k: f"Title {k} Word" for k in findings}
        mock_pipe = _make_pipe("no_stance", 0.70)

        with patch("app.services.contradiction.get_pipeline", return_value=mock_pipe), \
             patch("app.services.contradiction.asyncio.run", return_value=None):
            from app.services.contradiction import generate_contradiction_map
            result = generate_contradiction_map(
                "topic_nxn",
                paper_ids=list(findings.keys()),
                paper_titles=titles,
            )

        parsed = json.loads(result)
        z_matrix = parsed["data"][0]["z"]
        assert len(z_matrix) == n, f"Expected {n} rows, got {len(z_matrix)}"
        for row in z_matrix:
            assert len(row) == n, f"Expected {n} cols, got {len(row)}"

    def test_diagonal_is_zero(self) -> None:
        """Diagonal cells (paper vs itself) must always be 0."""
        self._clear_cache()
        findings = {"p1": _FINDING_A, "p2": _FINDING_B, "p3": _FINDING_C}
        self._preload_findings(findings)
        titles = {k: f"Title {k} Long" for k in findings}
        mock_pipe = _make_pipe("refute", 0.95)  # even with strong refute everywhere

        with patch("app.services.contradiction.get_pipeline", return_value=mock_pipe), \
             patch("app.services.contradiction.asyncio.run", return_value=None):
            from app.services.contradiction import generate_contradiction_map
            result = generate_contradiction_map(
                "topic_diag",
                paper_ids=list(findings.keys()),
                paper_titles=titles,
            )

        parsed = json.loads(result)
        z_matrix = parsed["data"][0]["z"]
        for i in range(len(z_matrix)):
            assert z_matrix[i][i] == 0.0, (
                f"Diagonal z[{i}][{i}] should be 0.0, got {z_matrix[i][i]}"
            )

    def test_empty_paper_ids_returns_valid_json(self) -> None:
        """With no paper_ids, must return a valid (empty) Plotly JSON."""
        self._clear_cache()

        with patch("app.services.contradiction.asyncio") as mock_asyncio:
            mock_asyncio.run = MagicMock(return_value=[])
            from app.services.contradiction import generate_contradiction_map
            result = generate_contradiction_map("topic_no_papers", paper_ids=[])

        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_title_truncation_to_4_words(self) -> None:
        """Long titles must be truncated to first 4 words + ellipsis."""
        import app.services.contradiction as cmod
        long_title = "The Effects of Aspirin on Cardiovascular Health in Adults"
        short = cmod._short_title(long_title, n_words=4)
        assert "…" in short
        assert len(short.split()[0:4]) <= 4

    def test_short_title_no_truncation_when_short(self) -> None:
        """Titles with ≤4 words must not get an ellipsis."""
        import app.services.contradiction as cmod
        short_title = "Aspirin Study"
        result = cmod._short_title(short_title, n_words=4)
        assert "…" not in result
        assert result == "Aspirin Study"
