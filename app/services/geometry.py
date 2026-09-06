"""
app/services/geometry.py
-------------------------
Geometric claim-verification service for ClaimGraph.

Pipeline (Phase 4 Stage 4)
--------------------------
Step 1  Fit a global PCA(n_components=50) across ALL embeddings for a topic.
        This single shared projection avoids the "too-many-dims, too-few-points"
        trap that kills per-bucket PCA in 768-D space.

Step 2  Project all bucket embeddings AND the claim vector into the 50-D space.

Step 3  Per non-empty bucket:
        - If bucket has >= 3 projected points:
            fit sklearn.covariance.LedoitWolf on the bucket points and
            compute scipy.spatial.distance.mahalanobis(claim_proj, bucket_mean,
            precision_matrix).
        - If bucket has < 3 projected points:
            fall back to cosine distance from claim to bucket centroid.
            Flag low_sample_warning=True in the verdict.

Step 4  Assemble verdict:
        {
          "distances":          {"support": float, "refute": float, "no_stance": float},
          "winning_label":      str,          # label with smallest distance
          "margin":             float,        # (2nd-closest - closest) / closest
          "is_ambiguous":       bool,         # True when margin < 0.20
          "low_sample_warning": bool,         # True when any bucket used cosine fallback
        }

Public API
----------
fit_geometry(bucket_vectors, claim_vector)
    -> GeometryResult dict

is_ambiguous(distances)
    -> bool

Design notes
------------
- PCA n_components is capped at min(50, n_samples-1, n_features) automatically.
- When a bucket is completely empty (no vectors), its distance is set to inf so
  it can never win.
- All inputs are np.ndarray (float32); conversion is done internally.
- No Qdrant / MongoDB I/O in this module -- pure numpy/sklearn/scipy math.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.spatial.distance import cosine as cosine_dist
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PCA_N_COMPONENTS: int = 50
_MIN_MAHALANOBIS_POINTS: int = 3   # below this, fall back to cosine
_AMBIGUITY_THRESHOLD: float = 0.20  # margin < 20% of closest => ambiguous

LABELS = ("support", "refute", "no_stance")


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
GeometryResult = dict[str, Any]


# ---------------------------------------------------------------------------
# Public: is_ambiguous
# ---------------------------------------------------------------------------

def is_ambiguous(distances: dict[str, float]) -> bool:
    """
    Return True when the margin between the closest and second-closest
    distance is less than 20% of the closest distance.

    A small margin means the claim sits roughly equidistant between two
    stance classes -- the geometric verdict should not be trusted alone.

    Parameters
    ----------
    distances : dict
        ``{"support": float, "refute": float, "no_stance": float}``
        Inf is used for empty buckets.

    Returns
    -------
    bool
    """
    finite_vals = sorted(v for v in distances.values() if np.isfinite(v))
    if len(finite_vals) < 2:
        # Only one finite distance -- cannot be ambiguous
        return False
    closest, second = finite_vals[0], finite_vals[1]
    if closest == 0.0:
        return False  # exact hit -- not ambiguous
    margin = (second - closest) / closest
    return margin < _AMBIGUITY_THRESHOLD


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fit_global_pca(all_vectors: np.ndarray) -> PCA:
    """
    Fit a shared PCA on the union of all bucket vectors.

    n_components is capped to prevent sklearn errors when the number of
    samples or features is smaller than the requested 50.
    """
    n_samples, n_features = all_vectors.shape
    n_components = min(_PCA_N_COMPONENTS, n_samples - 1, n_features)
    if n_components < 1:
        n_components = 1
    pca = PCA(n_components=n_components)
    pca.fit(all_vectors)
    logger.debug(
        "_fit_global_pca: fitted PCA with n_components=%d "
        "(from n_samples=%d, n_features=%d)",
        n_components,
        n_samples,
        n_features,
    )
    return pca


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two 1-D vectors, clamped to [0, 2]."""
    if np.allclose(a, 0) or np.allclose(b, 0):
        return 1.0  # undefined -- treat as maximally distant
    return float(cosine_dist(a, b))


def _mahalanobis_distance(
    point: np.ndarray,
    bucket_vectors: np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    Fit LedoitWolf on bucket_vectors and compute Mahalanobis distance
    from point to the bucket mean.

    Returns
    -------
    (distance, precision_matrix)
    """
    lw = LedoitWolf()
    lw.fit(bucket_vectors)
    mean = bucket_vectors.mean(axis=0)
    dist = mahalanobis(point, mean, lw.precision_)
    return float(dist), lw.precision_


# ---------------------------------------------------------------------------
# Public: fit_geometry
# ---------------------------------------------------------------------------

def fit_geometry(
    bucket_vectors: dict[str, np.ndarray],
    claim_vector: np.ndarray,
) -> GeometryResult:
    """
    Fit a shared PCA + per-bucket Ledoit-Wolf geometry and compute the
    claim's distance to each stance class.

    Parameters
    ----------
    bucket_vectors : dict
        Keys are stance labels ("support", "refute", "no_stance").
        Values are float32 arrays of shape (n_papers, embedding_dim).
        Empty buckets may be absent or have shape (0, dim).
    claim_vector : np.ndarray
        1-D float32 array of shape (embedding_dim,).

    Returns
    -------
    GeometryResult dict:
        {
          "distances":          {"support": float, "refute": float, "no_stance": float},
          "winning_label":      str,
          "margin":             float,
          "is_ambiguous":       bool,
          "low_sample_warning": bool,
          "pca_n_components":   int,
        }

    Raises
    ------
    ValueError
        If no bucket contains any vectors.
    """
    # --- Normalise inputs --------------------------------------------------
    claim_vec = np.asarray(claim_vector, dtype=np.float64).ravel()

    # Collect non-empty buckets
    non_empty: dict[str, np.ndarray] = {}
    for label in LABELS:
        vecs = bucket_vectors.get(label)
        if vecs is not None and len(vecs) > 0:
            non_empty[label] = np.asarray(vecs, dtype=np.float64)

    if not non_empty:
        raise ValueError("fit_geometry: all buckets are empty -- nothing to fit.")

    # --- Step 1: fit global PCA on union of all bucket vectors --------------
    all_vecs = np.vstack(list(non_empty.values()))  # (N_total, D)
    # Append claim to ensure the PCA sees its direction too
    all_with_claim = np.vstack([all_vecs, claim_vec.reshape(1, -1)])
    pca = _fit_global_pca(all_with_claim)

    # --- Step 2: project into PCA space ------------------------------------
    projected: dict[str, np.ndarray] = {}
    for label, vecs in non_empty.items():
        projected[label] = pca.transform(vecs)         # (n_bucket, k)

    claim_proj = pca.transform(claim_vec.reshape(1, -1))[0]  # (k,)

    logger.info(
        "fit_geometry: PCA n_components=%d | buckets=%s",
        pca.n_components_,
        {lbl: len(v) for lbl, v in projected.items()},
    )

    # --- Step 3: compute per-bucket distances ------------------------------
    distances: dict[str, float] = {}
    low_sample_warning = False
    method_used: dict[str, str] = {}

    for label in LABELS:
        if label not in projected:
            # Empty bucket -- infinite distance
            distances[label] = float("inf")
            method_used[label] = "empty"
            continue

        proj_vecs = projected[label]
        n = len(proj_vecs)

        if n >= _MIN_MAHALANOBIS_POINTS:
            try:
                dist, _ = _mahalanobis_distance(claim_proj, proj_vecs)
                distances[label] = dist
                method_used[label] = "mahalanobis"
            except Exception as exc:  # noqa: BLE001
                # Numerical failure (e.g. singular precision) -- fall back
                logger.warning(
                    "fit_geometry: Mahalanobis failed for label=%r (%s), "
                    "falling back to cosine.",
                    label, exc,
                )
                centroid = proj_vecs.mean(axis=0)
                distances[label] = _cosine_distance(claim_proj, centroid)
                method_used[label] = "cosine_fallback_numerical"
                low_sample_warning = True
        else:
            # < 3 points: cosine fallback
            centroid = proj_vecs.mean(axis=0)
            distances[label] = _cosine_distance(claim_proj, centroid)
            method_used[label] = "cosine_fallback_small_n"
            low_sample_warning = True
            logger.warning(
                "fit_geometry: bucket %r has only %d point(s) -- "
                "using cosine fallback (low_sample_warning=True).",
                label, n,
            )

    logger.info(
        "fit_geometry: distances=%s | methods=%s",
        {k: f"{v:.4f}" for k, v in distances.items() if np.isfinite(v)},
        method_used,
    )

    # --- Step 4: assemble verdict ------------------------------------------
    finite_distances = {k: v for k, v in distances.items() if np.isfinite(v)}
    if not finite_distances:
        raise ValueError("fit_geometry: all distances are infinite.")

    winning_label = min(finite_distances, key=finite_distances.__getitem__)

    finite_sorted = sorted(finite_distances.values())
    if len(finite_sorted) >= 2:
        closest, second = finite_sorted[0], finite_sorted[1]
        margin = (second - closest) / closest if closest > 0 else float("inf")
    else:
        margin = float("inf")

    ambiguous = is_ambiguous(distances)

    result: GeometryResult = {
        "distances":          distances,
        "winning_label":      winning_label,
        "margin":             float(margin),
        "is_ambiguous":       ambiguous,
        "low_sample_warning": low_sample_warning,
        "pca_n_components":   int(pca.n_components_),
    }
    logger.info(
        "fit_geometry: winning_label=%r margin=%.4f is_ambiguous=%s "
        "low_sample_warning=%s",
        winning_label, margin, ambiguous, low_sample_warning,
    )
    return result
