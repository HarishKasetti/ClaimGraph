"""
tests/test_visualize.py
------------------------
Unit tests for app/services/visualize.py — Phase 7 Stage 7.

All tests are fully isolated (no UMAP download, no Qdrant, no FastAPI server):
- UMAP import always patched to trigger PCA fallback (reliable in CI)
- Plotly used directly (it's a pure-Python library, no side effects)

Test Groups
-----------
TestProject3D           -- shape, keys, labels, edge cases
TestGenerateScatter3D   -- JSON validity, trace structure, claim point
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_vectors(n: int = 6, dim: int = 12) -> np.ndarray:
    """Return small random float32 array for fast PCA."""
    rng = np.random.default_rng(42)
    return rng.random((n, dim)).astype(np.float32)


def _make_labels(n: int = 6) -> list[str]:
    labels_cycle = ["support", "refute", "no_stance"]
    return [labels_cycle[i % 3] for i in range(n)]


def _make_paper_ids(n: int = 6) -> list[str]:
    return [f"paper_{i:03d}" for i in range(n)]


def _make_titles(n: int = 6) -> list[str]:
    return [f"Title for Paper {i}" for i in range(n)]


# ---------------------------------------------------------------------------
# TestProject3D
# ---------------------------------------------------------------------------

class TestProject3D:
    """Tests for project_3d(all_vectors, labels, paper_ids, titles=None)."""

    def _run(self, n=6, dim=12, include_claim=True):
        """Helper: run project_3d with PCA forced (UMAP mocked away)."""
        vecs = _make_vectors(n, dim)
        labels = _make_labels(n - 1) + (["claim"] if include_claim else ["no_stance"])
        ids = _make_paper_ids(n - 1) + (["__claim__"] if include_claim else [f"paper_{n-1:03d}"])
        titles = _make_titles(n)

        # Force PCA by making UMAP import fail
        with patch.dict("sys.modules", {"umap": None}):
            from app.services.visualize import project_3d
            result = project_3d(vecs, labels, ids, titles)
        return result

    def test_returns_list_of_dicts(self) -> None:
        result = self._run()
        assert isinstance(result, list)
        assert all(isinstance(p, dict) for p in result)

    def test_length_matches_input(self) -> None:
        n = 8
        result = self._run(n=n)
        assert len(result) == n

    def test_all_required_keys_present(self) -> None:
        result = self._run()
        required = {"x", "y", "z", "label", "paper_id", "title"}
        for i, p in enumerate(result):
            assert required.issubset(p.keys()), (
                f"Point {i} missing keys: {required - p.keys()}"
            )

    def test_xyz_are_floats(self) -> None:
        result = self._run()
        for p in result:
            assert isinstance(p["x"], float)
            assert isinstance(p["y"], float)
            assert isinstance(p["z"], float)

    def test_labels_preserved(self) -> None:
        """Labels in output must match the input labels."""
        n = 6
        vecs = _make_vectors(n)
        labels_in = ["support", "refute", "no_stance", "support", "refute", "claim"]
        ids = _make_paper_ids(n)

        with patch.dict("sys.modules", {"umap": None}):
            from app.services.visualize import project_3d
            result = project_3d(vecs, labels_in, ids)

        labels_out = [p["label"] for p in result]
        assert labels_out == labels_in

    def test_titles_default_to_paper_ids(self) -> None:
        """When titles=None, titles in output must equal paper_ids."""
        n = 4
        vecs = _make_vectors(n)
        ids = _make_paper_ids(n)
        labels = _make_labels(n)

        with patch.dict("sys.modules", {"umap": None}):
            from app.services.visualize import project_3d
            result = project_3d(vecs, labels, ids, titles=None)

        for p in result:
            assert p["title"] == p["paper_id"]

    def test_empty_vectors_returns_empty_list(self) -> None:
        """Empty input must return an empty list without raising."""
        with patch.dict("sys.modules", {"umap": None}):
            from app.services.visualize import project_3d
            result = project_3d(np.array([]).reshape(0, 12), [], [])

        assert result == []

    def test_single_vector_does_not_crash(self) -> None:
        """Edge case: n=1 should not crash (PCA n_components capped)."""
        vecs = _make_vectors(1, 12)
        labels = ["support"]
        ids = ["p0"]

        with patch.dict("sys.modules", {"umap": None}):
            from app.services.visualize import project_3d
            result = project_3d(vecs, labels, ids)

        assert len(result) == 1
        assert "x" in result[0]

    def test_two_vectors_no_crash(self) -> None:
        """n=2 — PCA(n_components=min(3, n-1)=1) — must not crash."""
        vecs = _make_vectors(2, 10)
        labels = ["support", "refute"]
        ids = ["p0", "p1"]

        with patch.dict("sys.modules", {"umap": None}):
            from app.services.visualize import project_3d
            result = project_3d(vecs, labels, ids)

        assert len(result) == 2
        for p in result:
            assert all(k in p for k in ("x", "y", "z"))

    def test_pca_fallback_triggered_on_umap_timeout(self) -> None:
        """Simulating UMAP timeout by raising FuturesTimeoutError."""
        from concurrent.futures import TimeoutError as FuturesTimeoutError

        vecs = _make_vectors(6)
        labels = _make_labels(6)
        ids = _make_paper_ids(6)

        mock_future = MagicMock()
        mock_future.result.side_effect = FuturesTimeoutError("timed out")
        mock_pool = MagicMock()
        mock_pool.submit.return_value = mock_future
        mock_pool.__enter__ = MagicMock(return_value=mock_pool)
        mock_pool.__exit__ = MagicMock(return_value=False)

        with patch("app.services.visualize.ThreadPoolExecutor", return_value=mock_pool):
            from app.services.visualize import project_3d
            result = project_3d(vecs, labels, ids)

        assert len(result) == 6, "PCA fallback should return all points"

    def test_claim_label_preserved(self) -> None:
        """The claim point must keep label='claim' in output."""
        result = self._run(include_claim=True)
        claim_pts = [p for p in result if p["label"] == "claim"]
        assert len(claim_pts) == 1


# ---------------------------------------------------------------------------
# TestGenerateScatter3D
# ---------------------------------------------------------------------------

class TestGenerateScatter3D:
    """Tests for generate_scatter_3d(plot_points, claim_point=None) -> str."""

    def _make_plot_points(self, n: int = 5) -> list[dict]:
        """Create n fake plot points (no claim)."""
        rng = np.random.default_rng(0)
        labels_cycle = ["support", "refute", "no_stance"]
        return [
            {
                "x": float(rng.random()),
                "y": float(rng.random()),
                "z": float(rng.random()),
                "label": labels_cycle[i % 3],
                "paper_id": f"p{i}",
                "title": f"Paper {i} Title",
            }
            for i in range(n)
        ]

    def _make_claim_point(self) -> dict:
        return {
            "x": 0.5, "y": 0.5, "z": 0.5,
            "label": "claim",
            "paper_id": "__claim__",
            "title": "Intermittent fasting improves insulin sensitivity.",
        }

    def test_returns_valid_json_string(self) -> None:
        pts = self._make_plot_points()
        from app.services.visualize import generate_scatter_3d
        result = generate_scatter_3d(pts)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_plotly_data_and_layout_keys(self) -> None:
        pts = self._make_plot_points()
        from app.services.visualize import generate_scatter_3d
        parsed = json.loads(generate_scatter_3d(pts))
        assert "data" in parsed
        assert "layout" in parsed
        assert len(parsed["data"]) >= 1

    def test_scatter3d_trace_type(self) -> None:
        """All traces must be scatter3d."""
        pts = self._make_plot_points()
        from app.services.visualize import generate_scatter_3d
        parsed = json.loads(generate_scatter_3d(pts))
        for trace in parsed["data"]:
            assert trace["type"] == "scatter3d", (
                f"Unexpected trace type: {trace['type']}"
            )

    def test_claim_point_in_separate_trace(self) -> None:
        """Claim point must appear in a trace named 'Claim'."""
        pts = self._make_plot_points()
        claim = self._make_claim_point()
        from app.services.visualize import generate_scatter_3d
        parsed = json.loads(generate_scatter_3d(pts, claim))
        trace_names = [t.get("name", "") for t in parsed["data"]]
        assert "Claim" in trace_names, (
            f"No 'Claim' trace found. Trace names: {trace_names}"
        )

    def test_claim_trace_has_diamond_marker(self) -> None:
        """Claim marker symbol must be 'diamond'."""
        pts = self._make_plot_points()
        claim = self._make_claim_point()
        from app.services.visualize import generate_scatter_3d
        parsed = json.loads(generate_scatter_3d(pts, claim))
        claim_trace = next(t for t in parsed["data"] if t.get("name") == "Claim")
        assert claim_trace["marker"]["symbol"] == "diamond"

    def test_claim_trace_is_gold_colour(self) -> None:
        """Claim marker must be gold (#f1c40f)."""
        pts = self._make_plot_points()
        claim = self._make_claim_point()
        from app.services.visualize import generate_scatter_3d
        parsed = json.loads(generate_scatter_3d(pts, claim))
        claim_trace = next(t for t in parsed["data"] if t.get("name") == "Claim")
        assert claim_trace["marker"]["color"] == "#f1c40f"

    def test_support_trace_is_green(self) -> None:
        """Support trace marker must be #2ecc71 (emerald green)."""
        pts = [
            {"x": 0.1, "y": 0.1, "z": 0.1, "label": "support",
             "paper_id": "p0", "title": "Support paper"}
        ]
        from app.services.visualize import generate_scatter_3d
        parsed = json.loads(generate_scatter_3d(pts))
        support_trace = next(
            (t for t in parsed["data"] if t.get("name") == "Support"), None
        )
        assert support_trace is not None
        assert support_trace["marker"]["color"] == "#2ecc71"

    def test_refute_trace_is_red(self) -> None:
        """Refute trace marker must be #e74c3c (alizarin red)."""
        pts = [
            {"x": 0.2, "y": 0.2, "z": 0.2, "label": "refute",
             "paper_id": "p1", "title": "Refute paper"}
        ]
        from app.services.visualize import generate_scatter_3d
        parsed = json.loads(generate_scatter_3d(pts))
        refute_trace = next(
            (t for t in parsed["data"] if t.get("name") == "Refute"), None
        )
        assert refute_trace is not None
        assert refute_trace["marker"]["color"] == "#e74c3c"

    def test_dark_theme_paper_bgcolor(self) -> None:
        """Layout paper_bgcolor must be the dark theme colour #1a1a2e."""
        from app.services.visualize import generate_scatter_3d
        parsed = json.loads(generate_scatter_3d(self._make_plot_points()))
        assert parsed["layout"]["paper_bgcolor"] == "#1a1a2e"

    def test_empty_points_returns_valid_json(self) -> None:
        """No points must return a valid Plotly JSON without raising."""
        from app.services.visualize import generate_scatter_3d
        result = generate_scatter_3d([], None)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "layout" in parsed

    def test_claim_point_from_plot_points_list(self) -> None:
        """Claim point embedded in plot_points list must still form a Claim trace."""
        pts = self._make_plot_points(4)
        claim_embedded = {
            "x": 0.9, "y": 0.9, "z": 0.9,
            "label": "claim",
            "paper_id": "__claim__",
            "title": "The embedded claim",
        }
        pts.append(claim_embedded)
        from app.services.visualize import generate_scatter_3d
        parsed = json.loads(generate_scatter_3d(pts))
        trace_names = [t.get("name", "") for t in parsed["data"]]
        assert "Claim" in trace_names

    def test_scene_has_three_axes(self) -> None:
        """Layout.scene must define xaxis, yaxis, and zaxis."""
        from app.services.visualize import generate_scatter_3d
        parsed = json.loads(generate_scatter_3d(self._make_plot_points()))
        scene = parsed["layout"].get("scene", {})
        assert "xaxis" in scene
        assert "yaxis" in scene
        assert "zaxis" in scene
