"""
app/services/visualize.py
--------------------------
Phase 7 Stage 7 — 3D Projection & Scatter Visualisation Service.

Provides two public functions:

project_3d(all_vectors, labels, paper_ids, titles=None)
    Projects high-dimensional embeddings (768-D SPECTER2 vectors) down to 3
    coordinates for the interactive scatter plot.  Tries UMAP first with a
    10-second timeout; falls back to sklearn PCA(n_components=3) if UMAP is
    unavailable or exceeds the budget.

generate_scatter_3d(plot_points, claim_point=None)
    Builds a Plotly 3D scatter figure from the projected points:
      - Support papers  → green spheres
      - Refute papers   → red spheres
      - No-stance       → grey spheres
      - Claim point     → gold diamond-shaped marker (visually dominant)
    Returns a self-contained Plotly JSON string.

Design notes
------------
- UMAP timeout is implemented via concurrent.futures.ThreadPoolExecutor.
  The thread may continue running after the timeout — it cannot be killed —
  but the calling code returns immediately and uses PCA instead.
- n_components for both UMAP and PCA is always 3.
- When n_samples < 4, PCA n_components is capped at n_samples - 1.
- If all_vectors is empty, generate_scatter_3d returns an empty-state figure
  rather than raising.
"""

from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_UMAP_TIMEOUT_S: float = 10.0
_N_COMPONENTS: int = 3

# Colour palette — consistent with calibrate.py and contradiction.py dark theme
_COLOURS: dict[str, str] = {
    "support":   "#2ecc71",   # emerald green
    "refute":    "#e74c3c",   # alizarin red
    "no_stance": "#7f8c8d",   # asbestos grey
    "claim":     "#f1c40f",   # sunflower gold
}
_SYMBOLS: dict[str, str] = {
    "support":   "circle",
    "refute":    "circle",
    "no_stance": "circle",
    "claim":     "diamond",
}
_SIZES: dict[str, int] = {
    "support":   7,
    "refute":    7,
    "no_stance": 6,
    "claim":     16,   # visually dominant
}


# ---------------------------------------------------------------------------
# Internal: UMAP runner (called inside thread)
# ---------------------------------------------------------------------------

def _run_umap(vectors: np.ndarray) -> np.ndarray:
    """Fit UMAP(n_components=3) and return projected coordinates."""
    import umap  # late import — heavy dependency

    reducer = umap.UMAP(
        n_components=_N_COMPONENTS,
        n_neighbors=min(15, len(vectors) - 1),
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(vectors).astype(np.float32)


def _run_pca(vectors: np.ndarray) -> np.ndarray:
    """Fit PCA(n_components=3, capped at n_samples-1) and return coords."""
    from sklearn.decomposition import PCA  # late import

    n = len(vectors)
    n_comp = min(_N_COMPONENTS, n - 1, vectors.shape[1])
    if n_comp < 1:
        n_comp = 1

    pca = PCA(n_components=n_comp, random_state=42)
    projected = pca.fit_transform(vectors).astype(np.float32)

    # Pad to exactly 3 columns if n_comp < 3
    if projected.shape[1] < _N_COMPONENTS:
        padding = np.zeros(
            (projected.shape[0], _N_COMPONENTS - projected.shape[1]),
            dtype=np.float32,
        )
        projected = np.hstack([projected, padding])

    return projected


# ---------------------------------------------------------------------------
# Public: project_3d
# ---------------------------------------------------------------------------

def project_3d(
    all_vectors: np.ndarray,
    labels: list[str],
    paper_ids: list[str],
    titles: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """
    Project embeddings to 3-D for the interactive scatter plot.

    Parameters
    ----------
    all_vectors : np.ndarray
        Shape (N, D) float array — one row per point.  The claim vector
        should be the **last** row; its label must be ``"claim"``.
    labels : list[str]
        Per-row label: ``"support"`` / ``"refute"`` / ``"no_stance"`` /
        ``"claim"``.  Must have length N.
    paper_ids : list[str]
        Per-row paper identifier (or ``"__claim__"`` for the claim point).
        Must have length N.
    titles : list[str], optional
        Per-row display title.  Falls back to ``paper_id`` if not provided.

    Returns
    -------
    list of dict
        Each dict:
        ``{"x": float, "y": float, "z": float,
           "label": str, "paper_id": str, "title": str}``

    Strategy
    --------
    1. Try UMAP with a 10-second wall-clock timeout.
    2. On timeout / ImportError / any failure → fall back to PCA.
    """
    if titles is None:
        titles = list(paper_ids)

    n = len(all_vectors)
    if n == 0:
        logger.warning("project_3d: received empty all_vectors — returning [].")
        return []

    vectors = np.asarray(all_vectors, dtype=np.float64)

    # Choose projection method
    projected: np.ndarray | None = None

    if n >= 4:  # UMAP needs at least n_neighbors+1 points
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_run_umap, vectors)
                projected = future.result(timeout=_UMAP_TIMEOUT_S)
                logger.info("project_3d: UMAP succeeded (n=%d)", n)
        except FuturesTimeoutError:
            logger.warning(
                "project_3d: UMAP timed out after %.1fs — falling back to PCA.",
                _UMAP_TIMEOUT_S,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "project_3d: UMAP failed (%s) — falling back to PCA.", exc
            )

    if projected is None:
        logger.info("project_3d: using PCA fallback (n=%d)", n)
        try:
            projected = _run_pca(vectors)
        except Exception as exc:  # noqa: BLE001
            logger.error("project_3d: PCA also failed (%s) — returning zeros.", exc)
            projected = np.zeros((n, _N_COMPONENTS), dtype=np.float32)

    points: list[dict[str, Any]] = []
    for i in range(n):
        points.append({
            "x":        float(projected[i, 0]),
            "y":        float(projected[i, 1]),
            "z":        float(projected[i, 2]),
            "label":    labels[i] if i < len(labels) else "no_stance",
            "paper_id": paper_ids[i] if i < len(paper_ids) else f"p{i}",
            "title":    titles[i] if i < len(titles) else f"p{i}",
        })

    logger.info(
        "project_3d: projected %d points — "
        "support=%d refute=%d no_stance=%d claim=%d",
        n,
        sum(1 for p in points if p["label"] == "support"),
        sum(1 for p in points if p["label"] == "refute"),
        sum(1 for p in points if p["label"] == "no_stance"),
        sum(1 for p in points if p["label"] == "claim"),
    )
    return points


# ---------------------------------------------------------------------------
# Public: generate_scatter_3d
# ---------------------------------------------------------------------------

def generate_scatter_3d(
    plot_points: list[dict[str, Any]],
    claim_point: Optional[dict[str, Any]] = None,
) -> str:
    """
    Build a Plotly 3D scatter figure from projected points.

    Parameters
    ----------
    plot_points : list of dict
        Output of ``project_3d()`` — each dict has ``x, y, z, label,
        paper_id, title``.  May include the claim point (label=="claim") or
        it may be passed separately via ``claim_point``.
    claim_point : dict, optional
        If provided, this dict (same schema) is treated as the claim point
        even if it is absent from ``plot_points``.  Takes precedence.

    Returns
    -------
    str
        Self-contained Plotly JSON string.
        Paste at https://chart-studio.plotly.com/create to render.

    Visual encoding
    ---------------
    - Support  → green circles
    - Refute   → red circles
    - No-stance → grey circles
    - Claim    → gold diamond, 2× larger, always drawn on top
    """
    import plotly.graph_objects as go  # late import

    if not plot_points and claim_point is None:
        logger.warning("generate_scatter_3d: no points provided — returning empty figure.")
        fig = go.Figure()
        fig.update_layout(
            title="No embeddings available",
            paper_bgcolor="#1a1a2e",
            font=dict(color="#ecf0f1"),
        )
        return fig.to_json()

    # Separate claim points out of plot_points (keep paper points clean)
    paper_points = [p for p in plot_points if p.get("label") != "claim"]
    claim_from_list = next(
        (p for p in plot_points if p.get("label") == "claim"), None
    )
    final_claim = claim_point or claim_from_list

    # Build one trace per label group (enables clean legend)
    traces: list[go.Scatter3d] = []

    label_groups = ["support", "refute", "no_stance"]
    label_names = {
        "support":   "Support",
        "refute":    "Refute",
        "no_stance": "Neutral",
    }

    for lbl in label_groups:
        pts = [p for p in paper_points if p.get("label") == lbl]
        if not pts:
            continue

        xs = [p["x"] for p in pts]
        ys = [p["y"] for p in pts]
        zs = [p["z"] for p in pts]
        hover = [
            f"<b>{p.get('title', p.get('paper_id', '?'))}</b><br>"
            f"Label: {lbl}<br>"
            f"({p['x']:.3f}, {p['y']:.3f}, {p['z']:.3f})"
            for p in pts
        ]

        traces.append(
            go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="markers",
                name=label_names[lbl],
                marker=dict(
                    size=_SIZES[lbl],
                    color=_COLOURS[lbl],
                    symbol=_SYMBOLS[lbl],
                    opacity=0.85,
                    line=dict(width=0.5, color="#1a1a2e"),
                ),
                hovertext=hover,
                hoverinfo="text",
            )
        )

    # Claim point — always last so it renders on top
    if final_claim:
        traces.append(
            go.Scatter3d(
                x=[final_claim["x"]],
                y=[final_claim["y"]],
                z=[final_claim["z"]],
                mode="markers+text",
                name="Claim",
                text=["★ Claim"],
                textposition="top center",
                textfont=dict(color=_COLOURS["claim"], size=12),
                marker=dict(
                    size=_SIZES["claim"],
                    color=_COLOURS["claim"],
                    symbol=_SYMBOLS["claim"],
                    opacity=1.0,
                    line=dict(width=2, color="#ffffff"),
                ),
                hovertext=[
                    f"<b>★ CLAIM</b><br>"
                    f"{final_claim.get('title', 'Claim')}<br>"
                    f"({final_claim['x']:.3f}, {final_claim['y']:.3f}, {final_claim['z']:.3f})"
                ],
                hoverinfo="text",
            )
        )

    n_papers = len(paper_points)
    n_support = sum(1 for p in paper_points if p.get("label") == "support")
    n_refute  = sum(1 for p in paper_points if p.get("label") == "refute")
    n_neutral = sum(1 for p in paper_points if p.get("label") == "no_stance")

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=(
                f"<b>ClaimGraph — 3D Embedding Space</b><br>"
                f"<span style='font-size:13px;color:#95a5a6'>"
                f"{n_papers} papers · "
                f"<span style='color:{_COLOURS['support']}'>{n_support} support</span> · "
                f"<span style='color:{_COLOURS['refute']}'>{n_refute} refute</span> · "
                f"<span style='color:{_COLOURS['no_stance']}'>{n_neutral} neutral</span>"
                f"</span>"
            ),
            font=dict(size=16, color="#ecf0f1"),
            x=0.5,
            xanchor="center",
        ),
        scene=dict(
            xaxis=dict(
                title=dict(text="Dim 1", font=dict(color="#bdc3c7", size=11)),
                backgroundcolor="#16213e",
                gridcolor="#2c3e50",
                showbackground=True,
                tickfont=dict(color="#bdc3c7", size=9),
            ),
            yaxis=dict(
                title=dict(text="Dim 2", font=dict(color="#bdc3c7", size=11)),
                backgroundcolor="#16213e",
                gridcolor="#2c3e50",
                showbackground=True,
                tickfont=dict(color="#bdc3c7", size=9),
            ),
            zaxis=dict(
                title=dict(text="Dim 3", font=dict(color="#bdc3c7", size=11)),
                backgroundcolor="#16213e",
                gridcolor="#2c3e50",
                showbackground=True,
                tickfont=dict(color="#bdc3c7", size=9),
            ),
            bgcolor="#16213e",
        ),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(family="Inter, sans-serif", color="#ecf0f1"),
        legend=dict(
            font=dict(color="#ecf0f1", size=12),
            bgcolor="#2c3e50",
            bordercolor="#34495e",
            borderwidth=1,
            x=0.01,
            y=0.99,
        ),
        margin=dict(t=90, b=0, l=0, r=0),
        height=600,
    )

    return fig.to_json()
