"""
app/api/routes/verify.py
-------------------------
Phase 7 Stage 7 — Full Pipeline Endpoint.

POST /api/v1/verify
    Runs all phases (1-6) and returns a unified dashboard payload.

Request body
------------
{
  "topic_id": "<topic_id>",
  "claim":    "<raw user claim or question>"
}

Response payload
----------------
{
  "verdict":            "support" | "refute" | "no_stance",
  "confidence":         float [0, 1],
  "reformulated_claim": str,
  "evidence":           [{"text", "paper_id", "cosine_sim"}, ...],
  "scatter_3d":         "<plotly_json>",
  "confidence_chart":   "<plotly_json>",
  "contradiction_map":  "<plotly_json>",
  "low_sample_warning": bool
}

All 7 keys are always present. Partial pipeline failures degrade gracefully:
- Geometry failure → verdict="no_stance", confidence=0.5
- Visualisation failure → empty-state Plotly JSON
- Contradiction map failure → empty-state Plotly JSON

Pipeline steps
--------------
1  reformulate_to_claim(claim)                   Phase 1
2  classify_paper_set(claim, topic_id)           Phase 3  (async)
3  Fetch all paper vectors from Qdrant           Phase 4 prep
4  embed claim via SPECTER2                      Phase 4 prep
5  fit_geometry(bucket_vectors, claim_vector)    Phase 4
6  calibrate(margin)                             Phase 5
7  extract_grounding(claim_vector, chunks, 3)    Phase 5
8  check_contradictions(topic_id)               Phase 6
9  generate_contradiction_map(topic_id)          Phase 6
10 project_3d(all_vectors, labels, ids)          Phase 7
11 generate_scatter_3d(plot_points, claim_pt)    Phase 7
12 generate_confidence_chart(distances, conf)    Phase 5

Design notes
------------
- The route is fully async. Sync-heavy calls (Qdrant scroll, UMAP/PCA,
  DeBERTa inference) are offloaded via asyncio.to_thread() to avoid
  blocking the event loop.
- classify_paper_set is async (Motor under the hood) and awaited directly.
- Paper titles are fetched from MongoDB global_papers for scatter hover text.
- Winning bucket chunks (with vectors) are fetched from Qdrant for grounding.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import defaultdict
from typing import Any, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Verify"])

# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------
_MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017/claimgraph")
_DB_NAME: str = _MONGO_URI.rstrip("/").split("/")[-1]
_QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
_COLLECTION: str = "papers"


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class VerifyRequest(BaseModel):
    topic_id: str
    claim: str


class VerifyResponse(BaseModel):
    verdict: str
    confidence: float
    reformulated_claim: str
    evidence: list[dict[str, Any]]
    scatter_3d: str
    confidence_chart: str
    contradiction_map: str
    low_sample_warning: bool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _fetch_paper_titles(paper_ids: list[str]) -> dict[str, str]:
    """Fetch {paper_id: title} from MongoDB global_papers."""
    from motor.motor_asyncio import AsyncIOMotorClient  # late import

    client = AsyncIOMotorClient(_MONGO_URI)
    try:
        db = client[_DB_NAME]
        titles: dict[str, str] = {}
        async for doc in db["global_papers"].find(
            {"paper_id": {"$in": paper_ids}},
            {"paper_id": 1, "title": 1},
        ):
            titles[doc["paper_id"]] = doc.get("title", doc["paper_id"])
        return titles
    finally:
        client.close()


def _fetch_bucket_vectors_sync(
    paper_ids: list[str],
    collection: str = _COLLECTION,
) -> dict[str, np.ndarray]:
    """
    Retrieve mean embedding per paper_id from Qdrant (sync).
    Returns {paper_id: mean_vector_1D}.
    """
    if not paper_ids:
        return {}

    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchAny

    client = QdrantClient(url=_QDRANT_URL)
    scroll_filter = Filter(
        must=[FieldCondition(key="paper_id", match=MatchAny(any=paper_ids))]
    )

    points: list[Any] = []
    offset = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=2000,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        points.extend(batch)
        if next_offset is None:
            break
        offset = next_offset

    # Group by paper_id, compute mean vector
    paper_vecs: dict[str, list[list[float]]] = defaultdict(list)
    for p in points:
        pid = (p.payload or {}).get("paper_id", "")
        if pid and p.vector:
            paper_vecs[pid].append(p.vector)

    result: dict[str, np.ndarray] = {}
    for pid in paper_ids:
        if pid in paper_vecs:
            result[pid] = np.mean(paper_vecs[pid], axis=0).astype(np.float32)

    return result


def _fetch_winning_chunks_sync(
    paper_ids: list[str],
    collection: str = _COLLECTION,
) -> list[dict[str, Any]]:
    """
    Fetch all chunks (with text + vector) for the winning-bucket papers.
    Used by extract_grounding.
    """
    if not paper_ids:
        return []

    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchAny

    client = QdrantClient(url=_QDRANT_URL)
    scroll_filter = Filter(
        must=[FieldCondition(key="paper_id", match=MatchAny(any=paper_ids))]
    )

    points: list[Any] = []
    offset = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=2000,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        points.extend(batch)
        if next_offset is None:
            break
        offset = next_offset

    chunks: list[dict[str, Any]] = []
    for p in points:
        payload = dict(p.payload or {})
        if p.vector:
            payload["vector"] = p.vector
        chunks.append(payload)

    return chunks


def _embed_claim_sync(claim: str) -> np.ndarray:
    """Embed claim string using SPECTER2 (sync)."""
    from app.services.embed import get_model
    model = get_model()
    vec = model.encode([claim], convert_to_numpy=True, show_progress_bar=False)
    return vec[0].astype(np.float32)


def _empty_plotly_json(message: str = "No data") -> str:
    """Return a minimal valid Plotly JSON for error/empty states."""
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.update_layout(
        title=message,
        paper_bgcolor="#1a1a2e",
        font=dict(color="#ecf0f1"),
    )
    return fig.to_json()


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/verify", response_model=VerifyResponse)
async def verify(req: VerifyRequest) -> VerifyResponse:
    """
    Run the full ClaimGraph verification pipeline and return the dashboard payload.

    The pipeline runs phases 1-7 in order, with graceful degradation on each
    step so the response is always well-formed.
    """
    topic_id = req.topic_id.strip()
    raw_claim = req.claim.strip()

    if not topic_id or not raw_claim:
        raise HTTPException(status_code=422, detail="topic_id and claim are required.")

    logger.info(
        "POST /verify  topic_id=%r  claim=%r", topic_id, raw_claim[:80]
    )

    # ── 1. Reformulate claim (Phase 1) ─────────────────────────────────────
    reformulated_claim = raw_claim
    try:
        from app.services.reformulate import reformulate_to_claim  # late import
        ref_result = await asyncio.to_thread(reformulate_to_claim, raw_claim)
        reformulated_claim = ref_result.get("claim", raw_claim)
        logger.info("Step 1 reformulated_claim=%r", reformulated_claim[:80])
    except Exception as exc:
        logger.warning("Step 1 reformulate failed (%s) — using raw claim.", exc)

    # ── 2. Classify paper stances (Phase 3) ────────────────────────────────
    buckets: dict[str, list[str]] = {"support": [], "refute": [], "no_stance": []}
    try:
        from app.services.stance import classify_paper_set  # late import
        buckets = await classify_paper_set(reformulated_claim, topic_id)
        logger.info(
            "Step 2 stance buckets — support=%d refute=%d no_stance=%d",
            len(buckets["support"]), len(buckets["refute"]), len(buckets["no_stance"]),
        )
    except Exception as exc:
        logger.warning("Step 2 classify_paper_set failed (%s) — empty buckets.", exc)

    # Collect all paper_ids across buckets
    all_paper_ids: list[str] = []
    paper_labels: list[str] = []
    for lbl in ("support", "refute", "no_stance"):
        for pid in buckets.get(lbl, []):
            if pid not in all_paper_ids:
                all_paper_ids.append(pid)
                paper_labels.append(lbl)

    # ── 3. Fetch paper vectors from Qdrant ─────────────────────────────────
    paper_vec_map: dict[str, np.ndarray] = {}
    try:
        paper_vec_map = await asyncio.to_thread(
            _fetch_bucket_vectors_sync, all_paper_ids
        )
        logger.info("Step 3 fetched %d paper vectors", len(paper_vec_map))
    except Exception as exc:
        logger.warning("Step 3 Qdrant fetch failed (%s).", exc)

    # ── 4. Embed claim (Phase 4 prep) ──────────────────────────────────────
    claim_vector: Optional[np.ndarray] = None
    try:
        claim_vector = await asyncio.to_thread(_embed_claim_sync, reformulated_claim)
        logger.info("Step 4 claim_vector shape=%s", claim_vector.shape)
    except Exception as exc:
        logger.warning("Step 4 claim embedding failed (%s).", exc)

    # ── 5. Geometric verdict (Phase 4) ─────────────────────────────────────
    verdict = "no_stance"
    margin = 0.0
    distances: dict[str, float] = {"support": float("inf"), "refute": float("inf"), "no_stance": float("inf")}
    is_ambiguous = True
    low_sample_warning = False

    if claim_vector is not None and paper_vec_map:
        try:
            from app.services.geometry import fit_geometry  # late import

            bucket_vectors: dict[str, np.ndarray] = {}
            for lbl in ("support", "refute", "no_stance"):
                pids = buckets.get(lbl, [])
                vecs = [paper_vec_map[pid] for pid in pids if pid in paper_vec_map]
                if vecs:
                    bucket_vectors[lbl] = np.array(vecs, dtype=np.float32)

            geo_result = await asyncio.to_thread(
                fit_geometry, bucket_vectors, claim_vector
            )
            verdict          = geo_result["winning_label"]
            margin           = geo_result["margin"]
            distances        = geo_result["distances"]
            is_ambiguous     = geo_result["is_ambiguous"]
            low_sample_warning = geo_result["low_sample_warning"]
            logger.info(
                "Step 5 verdict=%r margin=%.4f ambiguous=%s",
                verdict, margin, is_ambiguous,
            )
        except Exception as exc:
            logger.warning("Step 5 fit_geometry failed (%s) — default verdict.", exc)

    # ── 6. Calibrate confidence (Phase 5) ──────────────────────────────────
    confidence = 0.5
    try:
        from app.services.calibrate import calibrate  # late import
        confidence = await asyncio.to_thread(calibrate, margin)
        logger.info("Step 6 confidence=%.4f", confidence)
    except Exception as exc:
        logger.warning("Step 6 calibrate failed (%s) — using 0.5.", exc)

    # ── 7. Extract grounding sentences (Phase 5) ───────────────────────────
    evidence: list[dict[str, Any]] = []
    if claim_vector is not None:
        try:
            from app.services.calibrate import extract_grounding  # late import

            winning_paper_ids = buckets.get(verdict, [])
            winning_chunks = await asyncio.to_thread(
                _fetch_winning_chunks_sync, winning_paper_ids
            )
            evidence = await asyncio.to_thread(
                extract_grounding, claim_vector, winning_chunks, 3
            )
            logger.info("Step 7 grounding sentences=%d", len(evidence))
        except Exception as exc:
            logger.warning("Step 7 extract_grounding failed (%s).", exc)

    # ── 8. Contradiction check (Phase 6) ───────────────────────────────────
    try:
        from app.services.contradiction import check_contradictions  # late import
        await asyncio.to_thread(check_contradictions, topic_id, None)
        logger.info("Step 8 contradiction check complete")
    except Exception as exc:
        logger.warning("Step 8 check_contradictions failed (%s).", exc)

    # ── 9. Contradiction map JSON (Phase 6) ────────────────────────────────
    contradiction_map_json = _empty_plotly_json("Contradiction map unavailable")
    try:
        from app.services.contradiction import generate_contradiction_map  # late import
        paper_titles_map = await _fetch_paper_titles(all_paper_ids)
        contradiction_map_json = await asyncio.to_thread(
            generate_contradiction_map,
            topic_id,
            all_paper_ids,       # paper_ids override
            paper_titles_map,    # paper_titles override
        )
        logger.info("Step 9 contradiction map generated")
    except Exception as exc:
        logger.warning("Step 9 generate_contradiction_map failed (%s).", exc)

    # ── 10. Project all vectors to 3D (Phase 7) ────────────────────────────
    plot_points: list[dict[str, Any]] = []
    claim_plot_point: Optional[dict[str, Any]] = None

    try:
        from app.services.visualize import project_3d  # late import

        # Build combined matrix: paper vectors + claim vector
        present_ids = [pid for pid in all_paper_ids if pid in paper_vec_map]
        present_labels = [
            paper_labels[all_paper_ids.index(pid)] for pid in present_ids
        ]
        present_titles_list = [
            paper_titles_map.get(pid, pid) for pid in present_ids
        ] if "paper_titles_map" in dir() else present_ids

        if claim_vector is not None and present_ids:
            matrix = np.vstack(
                [paper_vec_map[pid] for pid in present_ids]
                + [claim_vector.reshape(1, -1)]
            )
            all_labels = present_labels + ["claim"]
            all_ids = present_ids + ["__claim__"]
            all_titles = present_titles_list + [reformulated_claim[:60]]
        elif present_ids:
            matrix = np.vstack([paper_vec_map[pid] for pid in present_ids])
            all_labels = present_labels
            all_ids = present_ids
            all_titles = present_titles_list
        else:
            matrix = None

        if matrix is not None and len(matrix) > 0:
            all_points = await asyncio.to_thread(
                project_3d, matrix, all_labels, all_ids, all_titles
            )
            plot_points = [p for p in all_points if p["label"] != "claim"]
            claim_plot_point = next(
                (p for p in all_points if p["label"] == "claim"), None
            )
            logger.info("Step 10 projected %d points", len(all_points))
    except Exception as exc:
        logger.warning("Step 10 project_3d failed (%s).", exc)

    # ── 11. 3D Scatter JSON (Phase 7) ──────────────────────────────────────
    scatter_3d_json = _empty_plotly_json("3D scatter unavailable")
    try:
        from app.services.visualize import generate_scatter_3d  # late import
        scatter_3d_json = await asyncio.to_thread(
            generate_scatter_3d, plot_points, claim_plot_point
        )
        logger.info("Step 11 scatter_3d generated")
    except Exception as exc:
        logger.warning("Step 11 generate_scatter_3d failed (%s).", exc)

    # ── 12. Confidence bar chart JSON (Phase 5) ────────────────────────────
    confidence_chart_json = _empty_plotly_json("Confidence chart unavailable")
    try:
        from app.services.calibrate import generate_confidence_chart  # late import
        confidence_chart_json = await asyncio.to_thread(
            generate_confidence_chart, distances, confidence
        )
        logger.info("Step 12 confidence_chart generated")
    except Exception as exc:
        logger.warning("Step 12 generate_confidence_chart failed (%s).", exc)

    # ── Assemble response ──────────────────────────────────────────────────
    logger.info(
        "POST /verify complete — verdict=%r confidence=%.3f evidence=%d",
        verdict, confidence, len(evidence),
    )

    return VerifyResponse(
        verdict=verdict,
        confidence=confidence,
        reformulated_claim=reformulated_claim,
        evidence=evidence,
        scatter_3d=scatter_3d_json,
        confidence_chart=confidence_chart_json,
        contradiction_map=contradiction_map_json,
        low_sample_warning=low_sample_warning,
    )
