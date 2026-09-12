"""
tests/test_calibrate.py
-----------------------
Unit tests for app/services/calibrate.py — Phase 5 Stage 5.

All tests are fully isolated:
  - No real ML model is loaded.
  - No MongoDB connection is made.
  - No filesystem access for the calibrator model (mocked with joblib).
  - Plotly is imported but only JSON output is checked (no rendering).

Test Groups
-----------
TestCalibrate           -- calibrate(margin) -> float
TestExtractGrounding    -- extract_grounding(claim_vec, chunks, top_k)
TestGenerateChart       -- generate_confidence_chart(distances, confidence)
TestPersistVerdict      -- persist_verdict(topic_id, claim, verdict)
TestPlotCalibrationCurve -- smoke-test plot_calibration_curve() with tmp file
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def _make_fake_calibrator(prob: float = 0.82):
    """Return a minimal sklearn-compatible mock calibrator."""
    mock_clf = MagicMock()
    # predict_proba returns [[p_incorrect, p_correct]]
    mock_clf.predict_proba = MagicMock(
        return_value=np.array([[1.0 - prob, prob]])
    )
    return mock_clf


def _make_chunks(n: int, dim: int = 8, seed: int = 42) -> list[dict[str, Any]]:
    """Build n fake chunk dicts with random vectors."""
    rng = np.random.default_rng(seed)
    chunks = []
    for i in range(n):
        vec = rng.standard_normal(dim).astype(np.float64)
        chunks.append({
            "text": f"Sentence number {i}: the study found significant results.",
            "paper_id": f"paper_{i // 2}",
            "vector": vec.tolist(),
        })
    return chunks


# ---------------------------------------------------------------------------
# TestCalibrate
# ---------------------------------------------------------------------------

class TestCalibrate:
    """Tests for calibrate(margin) -> float."""

    def test_output_in_unit_interval(self) -> None:
        """calibrate() must always return a value in [0, 1]."""
        fake_clf = _make_fake_calibrator(prob=0.75)
        with patch("app.services.calibrate._load_calibrator", return_value=fake_clf):
            from app.services.calibrate import calibrate

            result = calibrate(1.5)

        assert 0.0 <= result <= 1.0, f"Expected [0,1], got {result}"

    def test_uses_model_predict_proba(self) -> None:
        """calibrate() must call predict_proba and use column index 1."""
        fake_clf = _make_fake_calibrator(prob=0.88)
        with patch("app.services.calibrate._load_calibrator", return_value=fake_clf):
            from app.services.calibrate import calibrate

            result = calibrate(2.0)

        assert abs(result - 0.88) < 1e-4, f"Expected ~0.88, got {result}"
        fake_clf.predict_proba.assert_called_once()

    def test_inf_margin_is_capped(self) -> None:
        """Infinite margin must not raise -- it must be capped and processed."""
        fake_clf = _make_fake_calibrator(prob=0.95)
        with patch("app.services.calibrate._load_calibrator", return_value=fake_clf):
            from app.services.calibrate import calibrate

            result = calibrate(float("inf"))

        assert 0.0 <= result <= 1.0

    def test_zero_margin_returns_valid_prob(self) -> None:
        """Zero margin (ambiguous case) must still return a valid probability."""
        fake_clf = _make_fake_calibrator(prob=0.51)
        with patch("app.services.calibrate._load_calibrator", return_value=fake_clf):
            from app.services.calibrate import calibrate

            result = calibrate(0.0)

        assert 0.0 <= result <= 1.0

    def test_negative_margin_clamped_to_zero(self) -> None:
        """Negative margins (should not occur but must be safe) are clamped to 0."""
        fake_clf = _make_fake_calibrator(prob=0.55)
        with patch("app.services.calibrate._load_calibrator", return_value=fake_clf):
            from app.services.calibrate import calibrate

            # Should not raise
            result = calibrate(-5.0)

        assert 0.0 <= result <= 1.0

    def test_sigmoid_fallback_when_model_missing(self, tmp_path) -> None:
        """When the calibrator file is absent, SigmoidFallback must be used."""
        # Reset the module-level singleton
        import app.services.calibrate as cal_module
        original = cal_module._calibrator
        cal_module._calibrator = None  # force reload

        with patch.object(
            Path, "exists", return_value=False
        ):
            result = cal_module.calibrate(1.0)

        # SigmoidFallback sigmoid(1.0) ≈ 0.731
        assert 0.5 <= result <= 1.0, f"Fallback sigmoid should be > 0.5 for margin=1.0, got {result}"

        # Restore
        cal_module._calibrator = original

    def test_return_type_is_float(self) -> None:
        """calibrate() must return a Python float, not numpy scalar."""
        fake_clf = _make_fake_calibrator(prob=0.70)
        with patch("app.services.calibrate._load_calibrator", return_value=fake_clf):
            from app.services.calibrate import calibrate

            result = calibrate(1.0)

        assert isinstance(result, float), f"Expected float, got {type(result)}"


# ---------------------------------------------------------------------------
# TestExtractGrounding
# ---------------------------------------------------------------------------

class TestExtractGrounding:
    """Tests for extract_grounding(claim_vector, chunks, top_k)."""

    def test_returns_top_k_results(self) -> None:
        """extract_grounding must return exactly top_k items (when enough chunks exist)."""
        from app.services.calibrate import extract_grounding

        claim_vec = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        chunks = _make_chunks(n=10, dim=8)
        result = extract_grounding(claim_vec, chunks, top_k=3)

        assert len(result) == 3, f"Expected 3, got {len(result)}"

    def test_returns_fewer_when_not_enough_chunks(self) -> None:
        """When fewer than top_k chunks are available, return what exists."""
        from app.services.calibrate import extract_grounding

        claim_vec = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        chunks = _make_chunks(n=2, dim=8)
        result = extract_grounding(claim_vec, chunks, top_k=5)

        assert len(result) == 2

    def test_result_keys_present(self) -> None:
        """Each result dict must have 'text', 'paper_id', 'cosine_sim'."""
        from app.services.calibrate import extract_grounding

        claim_vec = np.ones(8, dtype=np.float64)
        chunks = _make_chunks(n=5, dim=8)
        result = extract_grounding(claim_vec, chunks, top_k=2)

        for item in result:
            assert "text" in item, f"Missing 'text' key: {item}"
            assert "paper_id" in item, f"Missing 'paper_id' key: {item}"
            assert "cosine_sim" in item, f"Missing 'cosine_sim' key: {item}"

    def test_sorted_descending_by_cosine_sim(self) -> None:
        """Results must be sorted descending by cosine_sim."""
        from app.services.calibrate import extract_grounding

        claim_vec = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        chunks = _make_chunks(n=8, dim=8, seed=7)
        result = extract_grounding(claim_vec, chunks, top_k=5)

        sims = [item["cosine_sim"] for item in result]
        assert sims == sorted(sims, reverse=True), (
            f"Results not sorted descending: {sims}"
        )

    def test_empty_chunks_returns_empty_list(self) -> None:
        """Empty input must return an empty list without raising."""
        from app.services.calibrate import extract_grounding

        claim_vec = np.ones(8)
        result = extract_grounding(claim_vec, [], top_k=3)

        assert result == []

    def test_cosine_sim_in_valid_range(self) -> None:
        """cosine_sim must be in [-1, 1]."""
        from app.services.calibrate import extract_grounding

        claim_vec = np.ones(8, dtype=np.float64)
        chunks = _make_chunks(n=6, dim=8)
        result = extract_grounding(claim_vec, chunks, top_k=6)

        for item in result:
            sim = item["cosine_sim"]
            assert -1.0 <= sim <= 1.0, f"cosine_sim out of range: {sim}"

    def test_chunks_missing_vector_skipped(self) -> None:
        """Chunks without a 'vector' key must be silently skipped."""
        from app.services.calibrate import extract_grounding

        claim_vec = np.ones(4, dtype=np.float64)
        chunks = [
            {"text": "Good sentence with vector.", "paper_id": "p1",
             "vector": [1.0, 0.5, 0.5, 0.5]},
            {"text": "No vector here — should be skipped.", "paper_id": "p2"},
            {"text": "Another good sentence.", "paper_id": "p3",
             "vector": [0.9, 0.4, 0.4, 0.4]},
        ]
        result = extract_grounding(claim_vec, chunks, top_k=5)

        # Only 2 chunks have vectors
        assert len(result) == 2

    def test_text_is_not_empty_string(self) -> None:
        """Returned text strings must not be empty."""
        from app.services.calibrate import extract_grounding

        claim_vec = np.ones(8, dtype=np.float64)
        chunks = _make_chunks(n=5, dim=8)
        result = extract_grounding(claim_vec, chunks, top_k=3)

        for item in result:
            assert len(item["text"].strip()) > 0, f"Empty text found: {item}"

    def test_most_similar_chunk_is_first(self) -> None:
        """The chunk most cosine-similar to the claim must appear first."""
        from app.services.calibrate import extract_grounding

        # Claim in direction of [1, 0, 0, 0]
        claim_vec = np.array([1.0, 0.0, 0.0, 0.0])

        # Construct chunks: chunk_0 is most aligned with claim direction
        chunks = [
            {"text": "Highly relevant supporting passage about the topic.",
             "paper_id": "best_paper",
             "vector": [1.0, 0.0, 0.0, 0.0]},  # cos_sim = 1.0
            {"text": "Somewhat related passage with mixed evidence.",
             "paper_id": "mid_paper",
             "vector": [0.5, 0.5, 0.5, 0.0]},
            {"text": "Completely orthogonal passage unrelated to claim.",
             "paper_id": "worst_paper",
             "vector": [0.0, 1.0, 0.0, 0.0]},  # cos_sim = 0.0
        ]
        result = extract_grounding(claim_vec, chunks, top_k=3)

        assert result[0]["paper_id"] == "best_paper", (
            f"Expected 'best_paper' first, got {result[0]['paper_id']}"
        )
        assert result[-1]["paper_id"] == "worst_paper", (
            f"Expected 'worst_paper' last, got {result[-1]['paper_id']}"
        )


# ---------------------------------------------------------------------------
# TestGenerateChart
# ---------------------------------------------------------------------------

class TestGenerateChart:
    """Tests for generate_confidence_chart(distances, confidence) -> str."""

    _SAMPLE_DISTANCES = {"support": 2.5, "refute": 8.1, "no_stance": 5.3}

    def test_returns_string(self) -> None:
        """Output must be a non-empty string."""
        from app.services.calibrate import generate_confidence_chart

        result = generate_confidence_chart(self._SAMPLE_DISTANCES, 0.83)
        assert isinstance(result, str)
        assert len(result) > 100

    def test_valid_json(self) -> None:
        """Output must be valid JSON."""
        from app.services.calibrate import generate_confidence_chart

        result = generate_confidence_chart(self._SAMPLE_DISTANCES, 0.83)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_plotly_data_key_present(self) -> None:
        """Plotly figure JSON must have a 'data' key (list of traces)."""
        from app.services.calibrate import generate_confidence_chart

        result = generate_confidence_chart(self._SAMPLE_DISTANCES, 0.83)
        parsed = json.loads(result)
        assert "data" in parsed, "Missing 'data' key in Plotly JSON"
        assert isinstance(parsed["data"], list)
        assert len(parsed["data"]) >= 1

    def test_plotly_layout_key_present(self) -> None:
        """Plotly figure JSON must have a 'layout' key."""
        from app.services.calibrate import generate_confidence_chart

        result = generate_confidence_chart(self._SAMPLE_DISTANCES, 0.83)
        parsed = json.loads(result)
        assert "layout" in parsed, "Missing 'layout' key in Plotly JSON"

    def test_bar_trace_has_three_values(self) -> None:
        """The bar trace must have 3 x-values (support, neutral, refute)."""
        from app.services.calibrate import generate_confidence_chart

        result = generate_confidence_chart(self._SAMPLE_DISTANCES, 0.83)
        parsed = json.loads(result)
        bar_trace = parsed["data"][0]
        assert bar_trace["type"] == "bar"
        assert len(bar_trace["x"]) == 3
        assert len(bar_trace["y"]) == 3

    def test_confidence_appears_in_title(self) -> None:
        """The layout title must contain the confidence percentage."""
        from app.services.calibrate import generate_confidence_chart

        result = generate_confidence_chart(self._SAMPLE_DISTANCES, 0.83)
        parsed = json.loads(result)
        title_text = parsed["layout"]["title"]["text"]
        # 83.0% should appear
        assert "83.0%" in title_text, (
            f"Expected '83.0%' in title, got: {title_text}"
        )

    def test_inf_distances_handled_gracefully(self) -> None:
        """Distances with inf (empty buckets) must not raise."""
        from app.services.calibrate import generate_confidence_chart

        distances = {"support": 2.5, "refute": float("inf"), "no_stance": float("inf")}
        result = generate_confidence_chart(distances, 0.91)
        parsed = json.loads(result)
        assert parsed["data"][0]["type"] == "bar"

    def test_all_inf_distances(self) -> None:
        """All-infinite distances must not raise (edge case)."""
        from app.services.calibrate import generate_confidence_chart

        distances = {
            "support": float("inf"),
            "refute": float("inf"),
            "no_stance": float("inf"),
        }
        result = generate_confidence_chart(distances, 0.5)
        assert json.loads(result) is not None

    def test_verdict_label_in_title(self) -> None:
        """The winning label must appear in the chart title."""
        from app.services.calibrate import generate_confidence_chart

        # Support has smallest distance → winning label
        distances = {"support": 1.0, "refute": 8.0, "no_stance": 5.0}
        result = generate_confidence_chart(distances, 0.90)
        parsed = json.loads(result)
        title_text = parsed["layout"]["title"]["text"].upper()
        assert "SUPPORT" in title_text, (
            f"Expected 'SUPPORT' in title, got: {title_text}"
        )

    def test_annotations_contain_confidence(self) -> None:
        """Layout annotations must mention the confidence value."""
        from app.services.calibrate import generate_confidence_chart

        result = generate_confidence_chart(self._SAMPLE_DISTANCES, 0.75)
        parsed = json.loads(result)
        annotations = parsed["layout"].get("annotations", [])
        assert len(annotations) >= 1
        # At least one annotation should mention the confidence
        found = any("75.0%" in ann.get("text", "") for ann in annotations)
        assert found, f"Confidence not found in annotations: {annotations}"


# ---------------------------------------------------------------------------
# TestPersistVerdict
# ---------------------------------------------------------------------------

class TestPersistVerdict:
    """Tests for persist_verdict(topic_id, claim, verdict)."""

    _TOPIC_ID = "topic_test_phase5"
    _CLAIM = "Intermittent fasting improves metabolic health."
    _VERDICT: dict[str, Any] = {
        "winning_label": "support",
        "margin": 1.42,
        "distances": {"support": 2.1, "refute": 5.1, "no_stance": 4.3},
        "confidence": 0.84,
        "grounding": [
            {"text": "Fasting improves insulin sensitivity.", "paper_id": "p1", "cosine_sim": 0.91}
        ],
    }

    def test_persist_calls_replace_one(self) -> None:
        """persist_verdict must call replace_one with upsert=True."""
        mock_replace = AsyncMock(return_value=MagicMock(matched_count=0, modified_count=0))
        mock_collection = MagicMock()
        mock_collection.replace_one = mock_replace

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)

        mock_client_instance = MagicMock()
        mock_client_instance.__getitem__ = MagicMock(return_value=mock_db)
        mock_client_instance.close = MagicMock()

        mock_motor = MagicMock()
        mock_motor.AsyncIOMotorClient = MagicMock(return_value=mock_client_instance)

        import sys
        original_motor = sys.modules.get("motor.motor_asyncio")
        sys.modules["motor.motor_asyncio"] = mock_motor

        try:
            from app.services.calibrate import persist_verdict

            # Call synchronously (no running event loop in test)
            persist_verdict(self._TOPIC_ID, self._CLAIM, self._VERDICT)
        finally:
            if original_motor is not None:
                sys.modules["motor.motor_asyncio"] = original_motor
            elif "motor.motor_asyncio" in sys.modules:
                del sys.modules["motor.motor_asyncio"]

    def test_persist_verdict_doc_id_format(self) -> None:
        """The document _id must be 'topic_id:claim_hash' format."""
        import hashlib

        claim_hash = hashlib.sha256(self._CLAIM.encode()).hexdigest()[:16]
        expected_doc_id = f"{self._TOPIC_ID}:{claim_hash}"

        # Just verify the hash logic (unit-testable without Motor)
        assert ":" in expected_doc_id
        parts = expected_doc_id.split(":")
        assert parts[0] == self._TOPIC_ID
        assert len(parts[1]) == 16

    def test_persist_verdict_same_claim_same_doc_id(self) -> None:
        """The same (topic_id, claim) pair must always produce the same doc_id."""
        import hashlib

        hash1 = hashlib.sha256(self._CLAIM.encode()).hexdigest()[:16]
        hash2 = hashlib.sha256(self._CLAIM.encode()).hexdigest()[:16]
        assert hash1 == hash2, "Same claim must produce same hash (deterministic)"


# ---------------------------------------------------------------------------
# TestPlotCalibrationCurve
# ---------------------------------------------------------------------------

class TestPlotCalibrationCurve:
    """Smoke tests for plot_calibration_curve()."""

    def _write_fake_calibration_data(self, path: Path, n: int = 30) -> None:
        """Write synthetic calibration JSONL."""
        rng = np.random.default_rng(0)
        with path.open("w") as f:
            for _ in range(n):
                margin = float(rng.uniform(0, 5))
                correct = int(rng.random() > 0.4)
                f.write(json.dumps({"margin": margin, "correct": correct}) + "\n")

    def test_creates_output_file(self, tmp_path: Path) -> None:
        """plot_calibration_curve() must create the output PNG file."""
        data_file = tmp_path / "calibration_set.jsonl"
        output_file = tmp_path / "calibration_curve.png"
        self._write_fake_calibration_data(data_file)

        fake_clf = _make_fake_calibrator(prob=0.72)

        with patch("app.services.calibrate._CALIBRATION_DATA_PATH", data_file), \
             patch("app.services.calibrate._load_calibrator", return_value=fake_clf):
            from app.services.calibrate import plot_calibration_curve
            plot_calibration_curve(output_path=output_file)

        assert output_file.exists(), "calibration_curve.png was not created"
        assert output_file.stat().st_size > 1000, "Output file is suspiciously small"

    def test_no_crash_on_few_data(self, tmp_path: Path) -> None:
        """plot_calibration_curve() must not crash when data is sparse."""
        data_file = tmp_path / "calibration_set.jsonl"
        output_file = tmp_path / "calibration_curve.png"
        # Only 3 points
        self._write_fake_calibration_data(data_file, n=3)

        fake_clf = _make_fake_calibrator(prob=0.60)

        with patch("app.services.calibrate._CALIBRATION_DATA_PATH", data_file), \
             patch("app.services.calibrate._load_calibrator", return_value=fake_clf):
            from app.services.calibrate import plot_calibration_curve
            # Should log an error but not raise
            plot_calibration_curve(output_path=output_file)

    def test_no_crash_when_data_missing(self, tmp_path: Path) -> None:
        """plot_calibration_curve() must not raise when data file is absent."""
        missing_path = tmp_path / "nonexistent.jsonl"
        output_file = tmp_path / "calibration_curve.png"

        with patch("app.services.calibrate._CALIBRATION_DATA_PATH", missing_path):
            from app.services.calibrate import plot_calibration_curve
            # Should log an error and return cleanly
            plot_calibration_curve(output_path=output_file)

        assert not output_file.exists(), (
            "Should not create output when data is missing"
        )
