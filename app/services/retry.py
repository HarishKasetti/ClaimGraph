"""
app/services/retry.py
---------------------
Ambiguity retry loop for ClaimGraph claim verification.

When geometric verification returns is_ambiguous=True, this module attempts
to resolve the ambiguity by fetching additional papers with a narrowed query
(topic_id + claim text), ingesting them, re-classifying stance, and
re-running geometry.

Public API
----------
verify_claim(claim, topic_id, *, collection, db_client, qdrant_url)
    -> GeometryResult dict
    Thin orchestrator: Qdrant scroll -> per-bucket mean vectors -> embed
    claim -> fit_geometry -> return verdict.

handle_ambiguous(claim, topic_id, *, attempt, max_attempts, ...)
    -> GeometryResult | InsufficientEvidenceResult
    Recursive retry loop. On each ambiguous result, fetches new papers
    with a narrowed query, ingests and classifies them, then calls
    verify_claim again. After max_attempts exhausted, returns
    {"verdict": "insufficient_evidence", "confidence": 0,
     "reason": "max_retries_reached"}.

Design notes
------------
- verify_claim is deliberately thin — it delegates all math to geometry.py
  and all I/O to stance.py / store.py / embed.py.
- The narrowed query is: f"{topic_id} {claim}" (appends claim text to the
  original topic string, nudging all four APIs toward more specific results).
- New papers fetched on retry are merged into existing buckets *in MongoDB*
  via classify_paper_set (which does an $set upsert), not by overwriting.
- Recursion depth is bounded by max_attempts (default 2), so the loop can
  never run more than max_attempts+1 total calls to verify_claim.
- InsufficientEvidenceResult is returned, never raised, so callers always
  get a dict they can safely JSON-serialise.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any, Optional

import numpy as np
from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017/claimgraph")
_DB_NAME: str = _MONGO_URI.rstrip("/").split("/")[-1]
_QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
_COLLECTION: str = "papers"

INSUFFICIENT_EVIDENCE: dict[str, Any] = {
    "verdict": "insufficient_evidence",
    "confidence": 0,
    "reason": "max_retries_reached",
}


# ---------------------------------------------------------------------------
# Internal: per-paper mean vector from Qdrant
# ---------------------------------------------------------------------------

def _fetch_bucket_vectors(
    buckets: dict[str, list[str]],
    *,
    qdrant_url: Optional[str] = None,
    collection: str = _COLLECTION,
) -> dict[str, np.ndarray]:
    """
    For each stance label, retrieve all Qdrant points belonging to the
    paper_ids in that bucket and return a dict of per-label mean vectors.

    Returns
    -------
    dict[str, np.ndarray]
        Keys: "support", "refute", "no_stance".
        Values: float32 arrays of shape (n_papers_in_bucket, embedding_dim).
        Empty buckets return shape (0, 768).
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchAny

    url = qdrant_url or _QDRANT_URL
    client = QdrantClient(url=url)

    result: dict[str, np.ndarray] = {}
    for label, paper_ids in buckets.items():
        if not paper_ids:
            result[label] = np.zeros((0, 768), dtype=np.float32)
            continue

        flt = Filter(
            must=[FieldCondition(key="paper_id", match=MatchAny(any=paper_ids))]
        )
        points, _ = client.scroll(
            collection_name=collection,
            scroll_filter=flt,
            limit=10_000,
            with_payload=True,
            with_vectors=True,
        )

        # Group by paper_id, then take mean embedding per paper
        paper_vecs: dict[str, list[list[float]]] = defaultdict(list)
        for p in points:
            pid = (p.payload or {}).get("paper_id", "")
            if pid and p.vector:
                paper_vecs[pid].append(p.vector)

        means = [
            np.mean(paper_vecs[pid], axis=0)
            for pid in paper_ids
            if pid in paper_vecs
        ]
        result[label] = (
            np.array(means, dtype=np.float32)
            if means
            else np.zeros((0, 768), dtype=np.float32)
        )

    return result


def _embed_claim(claim: str) -> np.ndarray:
    """Embed the claim string with SPECTER2 (lazy singleton from embed.py)."""
    from app.services.embed import get_model
    model = get_model()
    vec = model.encode([claim], convert_to_numpy=True, show_progress_bar=False)
    return vec[0].astype(np.float32)


# ---------------------------------------------------------------------------
# Public: verify_claim
# ---------------------------------------------------------------------------

async def verify_claim(
    claim: str,
    topic_id: str,
    *,
    qdrant_url: Optional[str] = None,
    collection: str = _COLLECTION,
    db_client: Optional[AsyncIOMotorClient] = None,
) -> dict[str, Any]:
    """
    Run full geometric verification for a claim against a topic's stored papers.

    Steps
    -----
    1. Load persisted stance buckets from MongoDB.
    2. Retrieve per-bucket mean embeddings from Qdrant.
    3. Embed the claim with SPECTER2.
    4. Call fit_geometry() and return the verdict dict.

    Returns
    -------
    dict
        GeometryResult from fit_geometry, guaranteed to contain:
        "distances", "winning_label", "margin", "is_ambiguous",
        "low_sample_warning", "pca_n_components".
        If no papers exist in any bucket, returns INSUFFICIENT_EVIDENCE.
    """
    # --- Load stance buckets from MongoDB ----------------------------------
    own_client = db_client is None
    client = db_client or AsyncIOMotorClient(_MONGO_URI)
    try:
        db = client[_DB_NAME]
        doc = await db["topics"].find_one({"_id": topic_id})
    finally:
        if own_client:
            client.close()

    if doc and "stance_buckets" in doc:
        buckets: dict[str, list[str]] = doc["stance_buckets"]
    else:
        buckets = {"support": [], "refute": [], "no_stance": []}

    total = sum(len(v) for v in buckets.values())
    if total == 0:
        logger.warning(
            "verify_claim: no papers in any bucket for topic_id=%r", topic_id
        )
        return dict(INSUFFICIENT_EVIDENCE, reason="no_papers_ingested")

    # --- Fetch per-bucket mean vectors from Qdrant -------------------------
    bucket_vectors = _fetch_bucket_vectors(
        buckets, qdrant_url=qdrant_url, collection=collection
    )

    # --- Embed claim -------------------------------------------------------
    claim_vector = _embed_claim(claim)

    # --- Run geometry ------------------------------------------------------
    from app.services.geometry import fit_geometry

    try:
        result = fit_geometry(bucket_vectors, claim_vector)
    except ValueError as exc:
        logger.error("verify_claim: fit_geometry failed: %s", exc)
        return dict(INSUFFICIENT_EVIDENCE, reason=str(exc))

    logger.info(
        "verify_claim: topic_id=%r → winning=%r margin=%.4f ambiguous=%s",
        topic_id,
        result["winning_label"],
        result["margin"],
        result["is_ambiguous"],
    )
    return result


# ---------------------------------------------------------------------------
# Public: handle_ambiguous
# ---------------------------------------------------------------------------

async def handle_ambiguous(
    claim: str,
    topic_id: str,
    *,
    attempt: int = 0,
    max_attempts: int = 2,
    qdrant_url: Optional[str] = None,
    collection: str = _COLLECTION,
    db_client: Optional[AsyncIOMotorClient] = None,
) -> dict[str, Any]:
    """
    Resolve ambiguous geometric verdicts by fetching additional papers.

    Algorithm
    ---------
    1. Call verify_claim to get the current verdict.
    2. If NOT ambiguous → return the verdict immediately.
    3. If ambiguous AND attempt >= max_attempts → return INSUFFICIENT_EVIDENCE.
    4. Otherwise:
       a. Build narrowed query: f"{topic_id} {claim}"
       b. fetch_papers(narrowed_query) → new raw papers
       c. For each new paper: ingest_paper (GROBID → embed → Qdrant/Mongo).
       d. classify_paper_set with the original topic_id to update MongoDB buckets.
       e. Recurse with attempt+1.

    Parameters
    ----------
    claim : str
        The scientific claim to verify.
    topic_id : str
        The topic identifier (used for Qdrant/MongoDB lookup).
    attempt : int
        Current retry attempt number (0-indexed). Do not set manually.
    max_attempts : int
        Maximum number of retry attempts (default 2).
        After exhausting all attempts, returns INSUFFICIENT_EVIDENCE.
    qdrant_url : str, optional
        Override Qdrant REST URL.
    collection : str
        Qdrant collection name.
    db_client : AsyncIOMotorClient, optional
        Shared MongoDB client.

    Returns
    -------
    dict
        Either a GeometryResult (if verdict is non-ambiguous) or
        INSUFFICIENT_EVIDENCE = {"verdict": "insufficient_evidence",
                                  "confidence": 0,
                                  "reason": "max_retries_reached"}.
    """
    logger.info(
        "handle_ambiguous: attempt=%d/%d topic_id=%r claim=%r",
        attempt, max_attempts, topic_id, claim[:60],
    )

    # --- Step 1: Run verification ------------------------------------------
    verdict = await verify_claim(
        claim,
        topic_id,
        qdrant_url=qdrant_url,
        collection=collection,
        db_client=db_client,
    )

    # --- Step 2: If not ambiguous, return immediately ----------------------
    if not verdict.get("is_ambiguous", False):
        logger.info(
            "handle_ambiguous: verdict non-ambiguous at attempt=%d "
            "(winning=%r margin=%.4f)",
            attempt,
            verdict.get("winning_label"),
            verdict.get("margin", 0.0),
        )
        return verdict

    # --- Step 3: Max attempts exhausted ------------------------------------
    if attempt >= max_attempts:
        logger.warning(
            "handle_ambiguous: max_attempts=%d exhausted for topic_id=%r "
            "— returning insufficient_evidence.",
            max_attempts, topic_id,
        )
        return dict(INSUFFICIENT_EVIDENCE)

    # --- Step 4: Fetch additional papers with narrowed query ---------------
    narrowed_query = f"{topic_id} {claim}"
    logger.info(
        "handle_ambiguous: attempt=%d ambiguous — fetching with narrowed "
        "query %r",
        attempt, narrowed_query[:80],
    )

    try:
        from app.services.fetch import fetch_papers, filter_papers
        raw_papers = await fetch_papers(narrowed_query)
        new_papers = filter_papers(raw_papers)
        logger.info(
            "handle_ambiguous: fetched %d new papers for narrowed query",
            len(new_papers),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "handle_ambiguous: fetch failed on attempt=%d: %s — "
            "proceeding to next attempt with existing papers.",
            attempt, exc,
        )
        new_papers = []

    # --- Step 4b: Ingest new papers ----------------------------------------
    ingested = 0
    for paper in new_papers:
        pdf_url = paper.get("pdf_url")
        doi = paper.get("doi", "")
        if not pdf_url or not doi:
            continue
        try:
            import httpx as _httpx
            async with _httpx.AsyncClient(timeout=30.0) as hc:
                resp = await hc.get(pdf_url)
                resp.raise_for_status()
                pdf_bytes = resp.content

            paper_id = f"retry_{attempt}_{doi.replace('/', '_')}"
            from app.services.store import ingest_paper
            n, _ = await ingest_paper(
                paper_id,
                pdf_bytes,
                paper,
                db_client=db_client,
                qdrant_url=qdrant_url,
                collection=collection,
            )
            if n > 0:
                ingested += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "handle_ambiguous: failed to ingest paper doi=%r: %s",
                doi, exc,
            )

    logger.info(
        "handle_ambiguous: ingested %d new papers on attempt=%d",
        ingested, attempt,
    )

    # --- Step 4c: Re-classify stance for the topic -------------------------
    try:
        from app.services.stance import classify_paper_set
        await classify_paper_set(
            claim,
            topic_id,
            qdrant_url=qdrant_url,
            collection=collection,
            db_client=db_client,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "handle_ambiguous: classify_paper_set failed on attempt=%d: %s",
            attempt, exc,
        )

    # --- Step 5: Recurse ---------------------------------------------------
    return await handle_ambiguous(
        claim,
        topic_id,
        attempt=attempt + 1,
        max_attempts=max_attempts,
        qdrant_url=qdrant_url,
        collection=collection,
        db_client=db_client,
    )
