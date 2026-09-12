"""
app/services/calibrate.py
--------------------------
Phase 5 Stage 5 — Calibration & Extraction Service.

Public API
----------
calibrate(margin)
    Load the fitted logistic-regression calibrator and convert a raw geometric
    margin to a probability in [0, 1].

extract_grounding(claim_vector, winning_bucket_chunks, top_k=3)
    Find the top-k most cosine-similar chunk dicts to the claim vector from
    the winning stance bucket.  Returns a list of
    ``{"text": str, "paper_id": str, "cosine_sim": float}``.

persist_verdict(topic_id, claim, verdict)
    Write the full verdict document to MongoDB ``verdicts`` collection
    (upsert by topic_id + claim hash).  Works from both sync and async contexts.

generate_confidence_chart(distances, confidence)
    Return a Plotly figure JSON string showing distance-to-each-plane as bars,
    with the calibrated confidence annotated.

plot_calibration_curve()
    Save ``calibration_curve.png`` plotting predicted confidence vs. actual
    accuracy across margin bins — the key defense artifact.

Design notes
------------
- The calibrator is loaded lazily from ``data/calibrator.joblib`` the first
  time ``calibrate()`` is called, then cached as a module-level singleton.
- If the joblib model does not exist (not yet fitted), ``calibrate()`` falls
  back to a sigmoid of the raw margin so the rest of the system can still run.
- ``persist_verdict`` uses Motor (async) internally but wraps the call in
  ``asyncio.run`` when called from a synchronous context.
- ``generate_confidence_chart`` produces a self-contained Plotly JSON string
  that can be pasted directly into https://chart-studio.plotly.com/create
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.spatial.distance import cosine as cosine_dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & env
# ---------------------------------------------------------------------------
_CALIBRATOR_PATH = Path("data/calibrator.joblib")
_CALIBRATION_DATA_PATH = Path("data/calibration_set.jsonl")
_CURVE_OUTPUT = Path("calibration_curve.png")

_MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017/claimgraph")
_DB_NAME: str = _MONGO_URI.rstrip("/").split("/")[-1]

# Margin ceiling (must match fit_calibrator.py)
_MARGIN_CAP = 20.0

# ---------------------------------------------------------------------------
# Lazy calibrator singleton
# ---------------------------------------------------------------------------
_calibrator = None  # type: ignore[assignment]


def _load_calibrator():
    """Return the fitted LogisticRegression, loading from disk once."""
    global _calibrator  # noqa: PLW0603
    if _calibrator is None:
        import joblib  # late import

        if _CALIBRATOR_PATH.exists():
            _calibrator = joblib.load(_CALIBRATOR_PATH)
            logger.info("Calibrator loaded from %s", _CALIBRATOR_PATH)
        else:
            logger.warning(
                "Calibrator not found at %s — using sigmoid fallback. "
                "Run: python -m app.scripts.fit_calibrator",
                _CALIBRATOR_PATH,
            )
            _calibrator = _SigmoidFallback()
    return _calibrator


class _SigmoidFallback:
    """Minimal sigmoid calibrator used when the joblib model is absent."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        # X shape: (N, 1) — margin values
        # sigmoid(margin) mapped to [0.5, 1] because margin >= 0
        vals = 1.0 / (1.0 + np.exp(-X[:, 0]))
        return np.column_stack([1.0 - vals, vals])


# ---------------------------------------------------------------------------
# Public: calibrate
# ---------------------------------------------------------------------------

def calibrate(margin: float) -> float:
    """
    Convert a raw geometric margin to a calibrated probability.

    Parameters
    ----------
    margin : float
        The (second-closest - closest) / closest margin from ``fit_geometry()``.
        May be ``float('inf')`` — capped to ``_MARGIN_CAP`` internally.

    Returns
    -------
    float
        Calibrated probability in [0, 1] that the geometric verdict is correct.
    """
    if not math.isfinite(margin):
        margin = _MARGIN_CAP
    margin = float(np.clip(margin, 0.0, _MARGIN_CAP))
    clf = _load_calibrator()
    X = np.array([[margin]], dtype=np.float64)
    prob = float(clf.predict_proba(X)[0, 1])
    logger.debug("calibrate: margin=%.4f -> confidence=%.4f", margin, prob)
    return round(float(np.clip(prob, 0.0, 1.0)), 6)


# ---------------------------------------------------------------------------
# Public: extract_grounding
# ---------------------------------------------------------------------------

def extract_grounding(
    claim_vector: np.ndarray,
    winning_bucket_chunks: list[dict[str, Any]],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    Find the top-k most cosine-similar chunks to the claim from the winning
    stance bucket.

    Parameters
    ----------
    claim_vector : np.ndarray
        1-D float array (embedding_dim,) representing the claim.
    winning_bucket_chunks : list of dict
        Each dict must have at minimum:
          - ``"text"``     : str  — the chunk text
          - ``"paper_id"`` : str  — source paper identifier
          - ``"vector"``   : list[float] | np.ndarray — chunk embedding
        ``"vector"`` is optional; chunks without it are skipped.
    top_k : int
        Number of top sentences to return (default 3).

    Returns
    -------
    list of dict
        Up to ``top_k`` dicts, each:
        ``{"text": str, "paper_id": str, "cosine_sim": float}``
        sorted descending by cosine_sim.
    """
    if not winning_bucket_chunks:
        return []

    claim_vec = np.asarray(claim_vector, dtype=np.float64).ravel()
    if np.allclose(claim_vec, 0):
        logger.warning("extract_grounding: claim_vector is all-zeros.")
        return []

    scored: list[tuple[float, dict[str, Any]]] = []

    for chunk in winning_bucket_chunks:
        vec_raw = chunk.get("vector")
        text = chunk.get("text", "").strip()
        paper_id = chunk.get("paper_id", "unknown")

        if vec_raw is None or not text:
            continue

        chunk_vec = np.asarray(vec_raw, dtype=np.float64).ravel()
        if np.allclose(chunk_vec, 0):
            continue

        # Cosine similarity = 1 - cosine_distance
        try:
            sim = float(1.0 - cosine_dist(claim_vec, chunk_vec))
        except Exception:  # noqa: BLE001
            sim = 0.0

        scored.append((sim, {"text": text, "paper_id": paper_id, "cosine_sim": round(sim, 6)}))

    # Sort descending by similarity
    scored.sort(key=lambda t: t[0], reverse=True)
    top = [item for _, item in scored[:top_k]]

    logger.debug(
        "extract_grounding: %d chunks scored, returning top-%d (best sim=%.4f)",
        len(scored),
        top_k,
        scored[0][0] if scored else 0.0,
    )
    return top


# ---------------------------------------------------------------------------
# Public: persist_verdict
# ---------------------------------------------------------------------------

def persist_verdict(
    topic_id: str,
    claim: str,
    verdict: dict[str, Any],
) -> None:
    """
    Persist the full verification verdict to MongoDB ``verdicts`` collection.

    Upserts on ``topic_id + claim_hash`` so re-running the pipeline for the
    same claim updates rather than duplicates the record.

    Parameters
    ----------
    topic_id : str
        Research topic identifier.
    claim : str
        The verbatim claim string that was verified.
    verdict : dict
        The full verdict dict, expected to include at minimum:
          - ``"winning_label"``
          - ``"margin"``
          - ``"distances"``
          - ``"confidence"``     (calibrated probability; injected by caller)
          - ``"grounding"``      (list from extract_grounding; injected by caller)
    """
    claim_hash = hashlib.sha256(claim.encode()).hexdigest()[:16]
    doc_id = f"{topic_id}:{claim_hash}"

    document = {
        "_id": doc_id,
        "topic_id": topic_id,
        "claim": claim,
        "claim_hash": claim_hash,
        "verdict": verdict,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        loop = asyncio.get_running_loop()
        # We are inside an async context — schedule as a coroutine
        loop.create_task(_async_persist(doc_id, document))
        logger.info("persist_verdict: scheduled async upsert for doc_id=%r", doc_id)
    except RuntimeError:
        # No running event loop — run synchronously
        asyncio.run(_async_persist(doc_id, document))
        logger.info("persist_verdict: sync upsert done for doc_id=%r", doc_id)


async def _async_persist(doc_id: str, document: dict[str, Any]) -> None:
    """Async Motor upsert into verdicts collection."""
    from motor.motor_asyncio import AsyncIOMotorClient  # late import

    client = AsyncIOMotorClient(_MONGO_URI)
    try:
        db = client[_DB_NAME]
        result = await db["verdicts"].replace_one(
            {"_id": doc_id},
            document,
            upsert=True,
        )
        logger.info(
            "_async_persist: upserted verdict doc_id=%r (matched=%d, modified=%d)",
            doc_id,
            result.matched_count,
            result.modified_count,
        )
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Public: generate_confidence_chart
# ---------------------------------------------------------------------------

def generate_confidence_chart(
    distances: dict[str, float],
    confidence: float,
) -> str:
    """
    Generate a Plotly bar chart JSON showing distance-to-each-stance-plane,
    with the calibrated confidence annotated as a title overlay.

    Parameters
    ----------
    distances : dict
        ``{"support": float, "refute": float, "no_stance": float}``
        Values may be ``float("inf")`` for empty buckets.
    confidence : float
        Calibrated confidence in [0, 1] from ``calibrate()``.

    Returns
    -------
    str
        JSON string representing a complete Plotly figure.
        Paste into https://chart-studio.plotly.com/create to render.
    """
    import plotly.graph_objects as go  # late import

    labels = ["Support", "Neutral", "Refute"]
    keys = ["support", "no_stance", "refute"]
    colors = ["#2ecc71", "#95a5a6", "#e74c3c"]

    values: list[float] = []
    for k in keys:
        v = distances.get(k, float("inf"))
        if not math.isfinite(v):
            v = 0.0  # empty bucket — show as 0 height bar with distinct colour
        values.append(round(v, 4))

    # Identify winning bar (smallest non-zero distance)
    finite_distances = {k: distances.get(k, float("inf")) for k in keys}
    winning_key = min(
        (k for k in finite_distances if math.isfinite(finite_distances[k]) and finite_distances[k] > 0),
        key=lambda k: finite_distances[k],
        default=None,
    )

    # Highlight winning bar with stronger color
    bar_colors = []
    border_colors = []
    winning_label_text = ""
    for k, c in zip(keys, colors):
        if k == winning_key:
            bar_colors.append(c)
            border_colors.append("#ffffff")
            winning_label_text = k.replace("no_stance", "neutral").upper()
        else:
            # Muted version of the color
            bar_colors.append(c + "80")  # 50% alpha via hex
            border_colors.append(c)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker=dict(
                color=bar_colors,
                line=dict(color=border_colors, width=2),
            ),
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
            textfont=dict(size=13, color="#ecf0f1"),
            hovertemplate=(
                "<b>%{x}</b><br>Distance: %{y:.4f}<extra></extra>"
            ),
            name="Distance to plane",
        )
    )

    confidence_pct = f"{confidence * 100:.1f}%"
    verdict_text = winning_label_text or "UNKNOWN"

    fig.update_layout(
        title=dict(
            text=(
                f"<b>Verdict: {verdict_text}</b>  ·  "
                f"Confidence: <b>{confidence_pct}</b>"
            ),
            font=dict(size=18, color="#ecf0f1"),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title="Stance Class",
            title_font=dict(size=14, color="#bdc3c7"),
            tickfont=dict(size=13, color="#bdc3c7"),
            gridcolor="#2c3e50",
        ),
        yaxis=dict(
            title="Distance to Plane",
            title_font=dict(size=14, color="#bdc3c7"),
            tickfont=dict(size=13, color="#bdc3c7"),
            gridcolor="#2c3e50",
            zeroline=False,
        ),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(family="Inter, sans-serif", color="#ecf0f1"),
        margin=dict(t=80, b=60, l=60, r=40),
        showlegend=False,
        annotations=[
            dict(
                text=(
                    f"Calibrated confidence: <b>{confidence_pct}</b><br>"
                    f"<span style='font-size:11px;color:#7f8c8d'>"
                    f"Smallest distance = winning stance class</span>"
                ),
                xref="paper",
                yref="paper",
                x=0.98,
                y=0.96,
                xanchor="right",
                yanchor="top",
                showarrow=False,
                font=dict(size=12, color="#ecf0f1"),
                bgcolor="#2c3e50",
                borderpad=8,
                bordercolor="#34495e",
                borderwidth=1,
            )
        ],
    )

    return fig.to_json()


# ---------------------------------------------------------------------------
# Public: plot_calibration_curve  (defense artifact)
# ---------------------------------------------------------------------------

def plot_calibration_curve(
    output_path: str | Path = _CURVE_OUTPUT,
    n_bins: int = 10,
) -> None:
    """
    Plot predicted confidence vs. actual accuracy across margin bins.

    Reads ``data/calibration_set.jsonl``, runs ``calibrate()`` on each margin,
    bins the predictions, and plots mean predicted vs. actual fraction correct.

    The resulting ``calibration_curve.png`` is the key defense artifact:
    points close to the diagonal indicate reliable probability estimates.

    Parameters
    ----------
    output_path : str or Path
        Where to save the PNG (default: ``calibration_curve.png``).
    n_bins : int
        Number of bins for the reliability diagram (default: 10).
    """
    import matplotlib  # late import
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if not _CALIBRATION_DATA_PATH.exists():
        logger.error(
            "Calibration data not found at %s. Run build_calibration_set first.",
            _CALIBRATION_DATA_PATH,
        )
        return

    margins: list[float] = []
    corrects: list[int] = []

    with _CALIBRATION_DATA_PATH.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                margins.append(float(min(rec["margin"], _MARGIN_CAP)))
                corrects.append(int(rec["correct"]))
            except (KeyError, ValueError, json.JSONDecodeError):
                continue

    if len(margins) < 5:
        logger.error("Too few data points (%d) to plot calibration curve.", len(margins))
        return

    probs = [calibrate(m) for m in margins]

    # Build reliability diagram
    prob_arr = np.array(probs)
    correct_arr = np.array(corrects)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_means_pred: list[float] = []
    bin_means_actual: list[float] = []
    bin_counts: list[int] = []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (prob_arr >= lo) & (prob_arr < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        bin_means_pred.append(float(prob_arr[mask].mean()))
        bin_means_actual.append(float(correct_arr[mask].mean()))
        bin_counts.append(count)

    if not bin_means_pred:
        logger.error("No non-empty bins found in calibration curve.")
        return

    # ── Figure ──────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(8, 9),
        gridspec_kw={"height_ratios": [3, 1]},
        facecolor="#1a1a2e",
    )
    fig.subplots_adjust(hspace=0.05)

    # Dark style
    for ax in (ax1, ax2):
        ax.set_facecolor("#16213e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#34495e")
        ax.tick_params(colors="#bdc3c7", labelsize=11)

    # ── Reliability diagram (top) ────────────────────────────────────────────
    ax1.plot(
        [0, 1], [0, 1],
        "--",
        color="#7f8c8d",
        linewidth=1.5,
        label="Perfect calibration",
        zorder=1,
    )

    sizes = [max(8, c * 2) for c in bin_counts]
    sc = ax1.scatter(
        bin_means_pred,
        bin_means_actual,
        s=sizes,
        c=bin_counts,
        cmap="cool",
        edgecolors="#ecf0f1",
        linewidths=0.8,
        zorder=3,
        label="Calibration points",
    )
    ax1.plot(
        bin_means_pred,
        bin_means_actual,
        "-o",
        color="#3498db",
        linewidth=2,
        markersize=6,
        zorder=2,
    )

    # Shade gap between curve and diagonal
    ax1.fill_between(
        bin_means_pred,
        bin_means_actual,
        bin_means_pred,
        alpha=0.15,
        color="#e74c3c",
        label="Calibration gap",
    )

    cbar = fig.colorbar(sc, ax=ax1, pad=0.01)
    cbar.set_label("Samples per bin", color="#bdc3c7", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="#bdc3c7")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#bdc3c7")

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("")
    ax1.set_ylabel("Fraction Correct (Actual)", color="#bdc3c7", fontsize=12)
    ax1.set_title(
        "ClaimGraph — Calibration Reliability Diagram\n"
        "Predicted Confidence vs. Actual Accuracy",
        color="#ecf0f1",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax1.legend(
        loc="upper left",
        facecolor="#2c3e50",
        edgecolor="#34495e",
        labelcolor="#ecf0f1",
        fontsize=10,
    )
    ax1.grid(True, color="#2c3e50", linewidth=0.8)

    # Overall stats annotation
    overall_acc = float(correct_arr.mean())
    mean_conf = float(prob_arr.mean())
    ece = float(
        sum(
            (bin_counts[i] / len(margins)) * abs(bin_means_pred[i] - bin_means_actual[i])
            for i in range(len(bin_counts))
        )
    )
    ax1.text(
        0.98, 0.04,
        f"n={len(margins)} · Accuracy={overall_acc:.1%} · "
        f"Mean conf={mean_conf:.1%} · ECE={ece:.3f}",
        transform=ax1.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#bdc3c7",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#2c3e50", edgecolor="#34495e"),
    )

    # ── Histogram of predicted confidences (bottom) ──────────────────────────
    ax2.hist(
        probs,
        bins=n_bins,
        range=(0, 1),
        color="#3498db",
        alpha=0.8,
        edgecolor="#1a1a2e",
    )
    ax2.set_xlabel("Predicted Confidence", color="#bdc3c7", fontsize=12)
    ax2.set_ylabel("Count", color="#bdc3c7", fontsize=10)
    ax2.set_xlim(0, 1)
    ax2.grid(True, color="#2c3e50", linewidth=0.8)

    plt.savefig(
        str(output_path),
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    logger.info("Calibration curve saved → %s", output_path)
    print(f"[calibrate] Calibration curve saved → {output_path}")
