"""
app/services/stance.py
----------------------
Stance classification service for ClaimGraph.

Uses MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli -- an NLI model fine-tuned on
MNLI, FEVER, and ANLI -- via the HuggingFace ``transformers`` zero-shot
classification pipeline.

Public API
----------
- ``classify_stance(claim, passage)``
      -> ``{"label": "support"|"refute"|"no_stance", "score": float}``

- ``classify_paper_set(claim, topic_id)``
      -> ``{"support": [paper_ids], "refute": [paper_ids], "no_stance": [paper_ids]}``
      Retrieves every Qdrant chunk tagged with ``topic_id``, classifies each
      chunk, assigns each paper its strongest-chunk label, and persists the
      bucket assignment to MongoDB.

Design notes
------------
- Lazy singleton: the pipeline is loaded once per process (same pattern as
  embed.py's get_model()), keeping import time zero.
- Passages are hard-capped at 512 whitespace tokens before classification to
  stay within DeBERTa's context window.
- ``classify_paper_set`` uses Qdrant ``scroll`` (not ``search``) -- no query
  vector is needed; we iterate all chunks for a given topic.
- If a chunk's payload lacks ``topic_id``, the fallback scrolls the entire
  collection and logs a warning (backward-compatible with pre-Phase-3 data).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from motor.motor_asyncio import AsyncIOMotorClient
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
_LABELS = ["support", "refute", "no_stance"]
_MAX_PASSAGE_TOKENS = 512  # whitespace-token cap before calling the model

_MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017/claimgraph")
_DB_NAME: str = _MONGO_URI.rstrip("/").split("/")[-1]
_QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
_COLLECTION: str = "papers"

# ---------------------------------------------------------------------------
# Lazy singleton: HuggingFace zero-shot classification pipeline
# ---------------------------------------------------------------------------
_pipeline = None  # type: ignore[assignment]


def get_pipeline():
    """
    Return the DeBERTa-v3 zero-shot classification pipeline.

    Downloads from HuggingFace on first call; cached as a module-level
    singleton for all subsequent calls.
    """
    global _pipeline  # noqa: PLW0603
    if _pipeline is None:
        from transformers import pipeline as hf_pipeline  # late import

        logger.info("Loading stance model %r ...", _MODEL_NAME)
        _pipeline = hf_pipeline(
            "zero-shot-classification",
            model=_MODEL_NAME,
        )
        logger.info("Stance model %r loaded.", _MODEL_NAME)
    return _pipeline


# ---------------------------------------------------------------------------
# Public: classify_stance
# ---------------------------------------------------------------------------

def classify_stance(claim: str, passage: str) -> dict[str, Any]:
    """
    Classify a single (claim, passage) pair.

    Parameters
    ----------
    claim : str
        The scientific claim being verified (e.g. post-reformulation).
    passage : str
        A text chunk retrieved from a paper.

    Returns
    -------
    dict
        ``{"label": "support"|"refute"|"no_stance", "score": float}``
        where ``score`` is the model confidence for the top label.
    """
    # Cap passage length to stay within model context window
    tokens = passage.split()
    if len(tokens) > _MAX_PASSAGE_TOKENS:
        passage = " ".join(tokens[:_MAX_PASSAGE_TOKENS])

    pipe = get_pipeline()
    result = pipe(
        passage,
        candidate_labels=_LABELS,
        hypothesis_template="This text {} the claim: " + claim,
        multi_label=False,
    )

    top_label: str = result["labels"][0]
    top_score: float = float(result["scores"][0])

    logger.debug(
        "classify_stance -> label=%r score=%.4f | claim=%r | passage_start=%r",
        top_label,
        top_score,
        claim[:60],
        passage[:60],
    )
    return {"label": top_label, "score": top_score}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _qdrant_scroll(
    topic_id: str,
    *,
    qdrant_url: Optional[str] = None,
    collection: str = _COLLECTION,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    """
    Retrieve all Qdrant points associated with ``topic_id``.

    First tries a filtered scroll by ``topic_id`` payload field; if no points
    are found (data ingested before Phase 3 tagging was added), falls back to
    returning all points in the collection.
    """
    url = qdrant_url or _QDRANT_URL
    client = QdrantClient(url=url)

    from qdrant_client.models import Filter, FieldCondition, MatchValue

    scroll_filter = Filter(
        must=[
            FieldCondition(
                key="topic_id",
                match=MatchValue(value=topic_id),
            )
        ]
    )

    points: list[Any] = []
    offset = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(batch)
        if next_offset is None:
            break
        offset = next_offset

    if not points:
        # Fallback: no topic_id tagging -- fetch all points
        logger.warning(
            "_qdrant_scroll: no points found for topic_id=%r; "
            "falling back to full collection scroll (pre-Phase-3 data).",
            topic_id,
        )
        offset = None
        while True:
            batch, next_offset = client.scroll(
                collection_name=collection,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points.extend(batch)
            if next_offset is None:
                break
            offset = next_offset

    payloads = []
    for p in points:
        payload = dict(p.payload or {})
        payload.setdefault("paper_id", "unknown")
        payload.setdefault("text", "")
        payload.setdefault("section", "")
        payloads.append(payload)

    logger.info(
        "_qdrant_scroll: retrieved %d chunks for topic_id=%r",
        len(payloads),
        topic_id,
    )
    return payloads


async def _mongo_persist(
    topic_id: str,
    claim: str,
    buckets: dict[str, list[str]],
    *,
    db_client: Optional[AsyncIOMotorClient] = None,
) -> None:
    """
    Persist stance bucket results to MongoDB ``topics`` collection.
    """
    own_client = db_client is None
    client = db_client or AsyncIOMotorClient(_MONGO_URI)
    try:
        db = client[_DB_NAME]
        await db["topics"].update_one(
            {"_id": topic_id},
            {
                "$set": {
                    "stance_buckets": buckets,
                    "stance_claim": claim,
                    "stance_updated_at": datetime.now(timezone.utc).isoformat(),
                }
            },
            upsert=True,
        )
        logger.info(
            "_mongo_persist: persisted stance buckets for topic_id=%r "
            "(support=%d, refute=%d, no_stance=%d)",
            topic_id,
            len(buckets["support"]),
            len(buckets["refute"]),
            len(buckets["no_stance"]),
        )
    finally:
        if own_client:
            client.close()


# ---------------------------------------------------------------------------
# Public: classify_paper_set
# ---------------------------------------------------------------------------

async def classify_paper_set(
    claim: str,
    topic_id: str,
    *,
    qdrant_url: Optional[str] = None,
    collection: str = _COLLECTION,
    db_client: Optional[AsyncIOMotorClient] = None,
) -> dict[str, list[str]]:
    """
    Classify every stored chunk for a topic against a claim, bucket papers by
    their strongest-chunk label, and persist results to MongoDB.

    Parameters
    ----------
    claim : str
        The scientific claim to verify (post-reformulation string).
    topic_id : str
        Identifier for the research topic; used to filter Qdrant chunks.
    qdrant_url : str, optional
        Override Qdrant REST URL; defaults to ``QDRANT_URL`` env var.
    collection : str
        Qdrant collection name (default: ``"papers"``).
    db_client : AsyncIOMotorClient, optional
        Optional shared Motor client for MongoDB.

    Returns
    -------
    dict
        ``{"support": [...paper_ids], "refute": [...paper_ids],
           "no_stance": [...paper_ids]}``
        Each paper_id appears in exactly one bucket.
    """
    # --- Retrieve chunks ---------------------------------------------------
    payloads = _qdrant_scroll(
        topic_id, qdrant_url=qdrant_url, collection=collection
    )

    if not payloads:
        logger.warning(
            "classify_paper_set: no chunks found for topic_id=%r -- "
            "returning empty buckets.",
            topic_id,
        )
        empty: dict[str, list[str]] = {
            "support": [],
            "refute": [],
            "no_stance": [],
        }
        await _mongo_persist(topic_id, claim, empty, db_client=db_client)
        return empty

    # --- Classify each chunk (strongest-chunk-wins per paper) --------------
    best: dict[str, dict[str, Any]] = {}

    for payload in payloads:
        paper_id: str = payload["paper_id"]
        text: str = payload["text"]
        if not text.strip():
            continue

        result = classify_stance(claim, text)

        if paper_id not in best or result["score"] > best[paper_id]["score"]:
            best[paper_id] = result

    # --- Aggregate into buckets --------------------------------------------
    buckets: dict[str, list[str]] = {
        "support": [],
        "refute": [],
        "no_stance": [],
    }
    for paper_id, result in best.items():
        label = result["label"]
        buckets.setdefault(label, []).append(paper_id)

    # Ensure all three keys always present
    for key in ("support", "refute", "no_stance"):
        buckets.setdefault(key, [])

    logger.info(
        "classify_paper_set: topic_id=%r -> support=%d refute=%d no_stance=%d",
        topic_id,
        len(buckets["support"]),
        len(buckets["refute"]),
        len(buckets["no_stance"]),
    )

    # --- Persist to MongoDB ------------------------------------------------
    await _mongo_persist(topic_id, claim, buckets, db_client=db_client)

    return buckets
