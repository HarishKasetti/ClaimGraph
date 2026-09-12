"""
app/api/routes/topics.py
-------------------------
Phase 8 — Topic management endpoints.

POST /api/v1/topics
    Kick off background ingestion for a new topic.
    Returns immediately (<500ms) with {topic_id, status, job_id}.

GET /api/v1/topics/{topic_id}/status
    Poll Celery result backend + MongoDB for current ingestion status.
    Returns {status, paper_count, contradiction_count}.

GET /api/v1/topics/{topic_id}/suggestions
    After status == "ready", return 3-4 AI-generated claim suggestions
    derived from the topic's paper abstracts (Phase 0b: suggest_claims).
    Returns 409 Conflict if status is not "ready".
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/topics", tags=["Topics"])

# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------
_MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017/claimgraph")
_DB_NAME: str = _MONGO_URI.rstrip("/").split("/")[-1]


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class TopicRequest(BaseModel):
    topic: str


class TopicCreatedResponse(BaseModel):
    topic_id: str
    status: str
    job_id: str


class TopicStatusResponse(BaseModel):
    status: str
    paper_count: int
    contradiction_count: int


class TopicSuggestionsResponse(BaseModel):
    topic_id: str
    suggestions: list[str]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _upsert_topic(topic_id: str, doc: dict[str, Any]) -> None:
    """Create or update a topic document in MongoDB."""
    from motor.motor_asyncio import AsyncIOMotorClient
    client = AsyncIOMotorClient(_MONGO_URI)
    try:
        db = client[_DB_NAME]
        await db["topics"].update_one(
            {"_id": topic_id},
            {"$set": doc},
            upsert=True,
        )
    finally:
        client.close()


async def _get_topic(topic_id: str) -> dict[str, Any]:
    """Fetch a topic document from MongoDB. Returns {} if not found."""
    from motor.motor_asyncio import AsyncIOMotorClient
    client = AsyncIOMotorClient(_MONGO_URI)
    try:
        db = client[_DB_NAME]
        doc = await db["topics"].find_one({"_id": topic_id})
        return dict(doc) if doc else {}
    finally:
        client.close()


# ---------------------------------------------------------------------------
# POST /topics
# ---------------------------------------------------------------------------

@router.post("", response_model=TopicCreatedResponse, status_code=202)
async def create_topic(req: TopicRequest) -> TopicCreatedResponse:
    """
    Kick off background ingestion for a new topic.

    Returns immediately — the heavy work (fetching 10 papers, embedding,
    contradiction check) runs in a Celery worker.

    The returned ``topic_id`` is used to poll ``/topics/{id}/status`` and
    later call ``/topics/{id}/suggestions`` or ``POST /verify``.
    """
    topic_text = req.topic.strip()
    if not topic_text:
        raise HTTPException(status_code=422, detail="topic must not be empty.")

    # Generate stable topic_id from text (deterministic per topic)
    topic_id = str(uuid.uuid5(uuid.NAMESPACE_URL, topic_text))

    # Create initial document — upsert so re-submitting same topic is safe
    now = datetime.now(timezone.utc).isoformat()
    await _upsert_topic(topic_id, {
        "topic_text":         topic_text,
        "status":             "pending",
        "paper_count":        0,
        "contradiction_count": 0,
        "created_at":         now,
        "updated_at":         now,
    })

    # Fire Celery task — non-blocking
    from workers.tasks import ingest_topic  # late import (avoids circular deps)
    result = ingest_topic.delay(topic_id, topic_text)
    job_id = result.id

    # Store job_id for status polling
    await _upsert_topic(topic_id, {"job_id": job_id})

    logger.info(
        "POST /topics  topic_id=%r  job_id=%r  topic=%r",
        topic_id, job_id, topic_text[:60],
    )

    return TopicCreatedResponse(
        topic_id=topic_id,
        status="pending",
        job_id=job_id,
    )


# ---------------------------------------------------------------------------
# GET /topics/{topic_id}/status
# ---------------------------------------------------------------------------

@router.get("/{topic_id}/status", response_model=TopicStatusResponse)
async def get_topic_status(topic_id: str) -> TopicStatusResponse:
    """
    Poll ingestion status for a topic.

    Status lifecycle:
      pending   → task queued, worker hasn't picked it up yet
      ingesting → worker is actively fetching / embedding papers
      ready     → all phases complete; /suggestions and /verify are available
      failed    → all 3 retries exhausted; inspect worker logs

    Reads from both:
    - MongoDB (authoritative for ready/failed states written by the task)
    - Celery result backend (fallback for pending / in-progress states)
    """
    doc = await _get_topic(topic_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Topic {topic_id!r} not found.")

    mongo_status: str = doc.get("status", "pending")

    # If MongoDB already says ready or failed, trust it — no need to poll Celery
    if mongo_status in ("ready", "failed"):
        return TopicStatusResponse(
            status=mongo_status,
            paper_count=doc.get("paper_count", 0),
            contradiction_count=doc.get("contradiction_count", 0),
        )

    # For pending/ingesting, also check the Celery result backend
    celery_status = mongo_status
    job_id: str | None = doc.get("job_id")
    if job_id:
        try:
            from celery.result import AsyncResult
            from workers.celery_app import celery_app

            async_result = AsyncResult(job_id, app=celery_app)
            celery_state = async_result.state  # PENDING, STARTED, SUCCESS, FAILURE

            if celery_state == "SUCCESS":
                # Task finished but MongoDB wasn't updated yet (race condition)
                # Pull counts from Celery result
                celery_result = async_result.result or {}
                await _upsert_topic(topic_id, {
                    "status": "ready",
                    "paper_count": celery_result.get("paper_count", doc.get("paper_count", 0)),
                    "contradiction_count": celery_result.get("contradiction_count", 0),
                })
                celery_status = "ready"

            elif celery_state == "FAILURE":
                await _upsert_topic(topic_id, {"status": "failed"})
                celery_status = "failed"

            elif celery_state in ("STARTED", "RETRY"):
                celery_status = "ingesting"

            else:
                celery_status = "pending"

        except Exception as exc:
            logger.warning(
                "GET /topics/%s/status: Celery backend poll failed (%s) "
                "— using MongoDB status.",
                topic_id, exc,
            )
            celery_status = mongo_status

    # Re-fetch in case we just updated
    if celery_status in ("ready", "failed"):
        doc = await _get_topic(topic_id)

    return TopicStatusResponse(
        status=celery_status,
        paper_count=doc.get("paper_count", 0),
        contradiction_count=doc.get("contradiction_count", 0),
    )


# ---------------------------------------------------------------------------
# GET /topics/{topic_id}/suggestions
# ---------------------------------------------------------------------------

@router.get("/{topic_id}/suggestions", response_model=TopicSuggestionsResponse)
async def get_topic_suggestions(topic_id: str) -> TopicSuggestionsResponse:
    """
    Return 3-4 AI-generated claim suggestions for the topic.

    Only available when ``status == "ready"``.  Returns 409 Conflict otherwise
    so clients can distinguish "not ready yet" from "topic not found".

    Uses ``suggest_claims()`` (Phase 0b) fed with the abstracts stored during
    ingestion.
    """
    doc = await _get_topic(topic_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Topic {topic_id!r} not found.")

    status = doc.get("status", "pending")
    if status != "ready":
        raise HTTPException(
            status_code=409,
            detail=(
                f"Topic {topic_id!r} is not ready yet (status={status!r}). "
                "Poll /topics/{id}/status until status=='ready'."
            ),
        )

    abstracts: list[str] = doc.get("abstracts", [])
    if not abstracts:
        # Fallback: fetch abstracts directly from global_papers
        from motor.motor_asyncio import AsyncIOMotorClient
        client = AsyncIOMotorClient(_MONGO_URI)
        try:
            db = client[_DB_NAME]
            paper_ids: list[str] = doc.get("paper_ids", [])
            async for paper_doc in db["global_papers"].find(
                {"paper_id": {"$in": paper_ids}},
                {"abstract": 1},
            ):
                ab = paper_doc.get("abstract") or ""
                if ab:
                    abstracts.append(ab)
        finally:
            client.close()

    if not abstracts:
        logger.warning(
            "GET /topics/%s/suggestions: no abstracts found — returning empty list.",
            topic_id,
        )
        return TopicSuggestionsResponse(topic_id=topic_id, suggestions=[])

    import asyncio
    from app.services.reformulate import suggest_claims  # late import

    try:
        suggestions = await asyncio.to_thread(suggest_claims, abstracts[:5])
    except Exception as exc:
        logger.warning(
            "GET /topics/%s/suggestions: suggest_claims failed (%s).", topic_id, exc
        )
        suggestions = []

    logger.info(
        "GET /topics/%s/suggestions: returning %d suggestion(s)",
        topic_id, len(suggestions),
    )

    return TopicSuggestionsResponse(
        topic_id=topic_id,
        suggestions=suggestions,
    )
