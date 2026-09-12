"""
workers/tasks.py
-----------------
Phase 8 — Celery background tasks for ClaimGraph.

Task
----
ingest_topic(topic_id, topic_text)
    Background job that runs Phases 1-2 (fetch + ingest) and Phase 6
    (contradiction check) for a given topic.

    Steps:
      1. Mark topic status="ingesting" in MongoDB
      2. fetch_all(topic_text) → list of paper dicts (Phase 2 fetch)
      3. For each paper with a pdf_url: download PDF → ingest_paper()
      4. Persist paper stubs (no PDF) for papers without a pdf_url
      5. Run Phase 6: check_contradictions(topic_id, paper_ids)
      6. Mark topic status="ready" with paper_count + contradiction_count

    Failure handling:
      - Any unhandled exception triggers self.retry() up to 3 times
        with a 5-second delay between attempts.
      - If all retries are exhausted, status is set to "failed" in MongoDB.

Usage
-----
    from workers.tasks import ingest_topic
    result = ingest_topic.delay(topic_id, topic_text)
    # result.id is the Celery task ID

Start worker
------------
    celery -A workers.tasks worker --loglevel=info
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx

from workers.celery_app import celery_app

# Module-level imports for services that need to be patchable in tests.
# Late imports (inside task body) work fine in production but cannot be
# intercepted by unittest.mock.patch targeting "workers.tasks.<name>".
from app.services.contradiction import check_contradictions  # noqa: E402
from app.services.stance import classify_paper_set  # noqa: E402


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------
_MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017/claimgraph")
_DB_NAME: str = _MONGO_URI.rstrip("/").split("/")[-1]

_PDF_DOWNLOAD_TIMEOUT: float = 30.0   # seconds per PDF download
_MAX_PDF_PAPERS: int = 10              # max papers to fully ingest (download + parse)


# ---------------------------------------------------------------------------
# Helpers: MongoDB sync wrappers
# (Celery tasks are sync; use asyncio.run for motor calls)
# ---------------------------------------------------------------------------

def _mongo_update_status(topic_id: str, update: dict[str, Any]) -> None:
    """Synchronously update a topic document in MongoDB."""
    async def _do():
        from motor.motor_asyncio import AsyncIOMotorClient
        client = AsyncIOMotorClient(_MONGO_URI)
        try:
            db = client[_DB_NAME]
            await db["topics"].update_one(
                {"_id": topic_id},
                {"$set": {**update, "updated_at": datetime.now(timezone.utc).isoformat()}},
                upsert=True,
            )
        finally:
            client.close()

    asyncio.run(_do())


def _mongo_get_topic(topic_id: str) -> dict[str, Any]:
    """Synchronously fetch a topic document from MongoDB."""
    async def _do():
        from motor.motor_asyncio import AsyncIOMotorClient
        client = AsyncIOMotorClient(_MONGO_URI)
        try:
            db = client[_DB_NAME]
            doc = await db["topics"].find_one({"_id": topic_id})
            return dict(doc) if doc else {}
        finally:
            client.close()

    return asyncio.run(_do())


def _download_pdf(url: str) -> bytes | None:
    """Download a PDF from url, returning bytes or None on failure."""
    try:
        with httpx.Client(timeout=_PDF_DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "pdf" in content_type or resp.content[:4] == b"%PDF":
                return resp.content
            logger.debug("_download_pdf: non-PDF response for %s (%s)", url, content_type)
            return None
    except Exception as exc:
        logger.warning("_download_pdf: failed for %s (%s)", url, exc)
        return None


def _make_paper_id(doi: str | None, title: str) -> str:
    """Deterministic paper_id from DOI or title hash."""
    key = doi if doi else title
    return hashlib.sha256(key.encode()).hexdigest()[:20]


def _ingest_papers_sync(papers: list[dict[str, Any]], topic_id: str) -> list[str]:
    """
    Ingest a list of paper dicts (from fetch_all) into Qdrant + MongoDB.

    Returns the list of paper_ids that were successfully processed.
    """
    from app.services.store import ingest_paper, save_to_mongo

    paper_ids: list[str] = []

    for paper in papers:
        doi   = paper.get("doi")
        title = paper.get("title") or "Untitled"
        paper_id = _make_paper_id(doi, title)

        metadata = {
            "doi":          doi,
            "title":        title,
            "abstract":     paper.get("abstract") or "",
            "venue":        paper.get("venue") or "",
            "citation_count": paper.get("citation_count") or 0,
            "pdf_url":      paper.get("pdf_url") or "",
            "source":       paper.get("source") or "",
            "topic_id":     topic_id,
        }

        pdf_url = paper.get("pdf_url")
        pdf_bytes: bytes | None = None

        if pdf_url and len(paper_ids) < _MAX_PDF_PAPERS:
            pdf_bytes = _download_pdf(pdf_url)

        try:
            if pdf_bytes:
                # Full ingest: parse + embed + Qdrant + Mongo
                asyncio.run(ingest_paper(paper_id, pdf_bytes, metadata))
                logger.info("ingest_papers: full ingest paper_id=%r", paper_id)
            else:
                # Metadata-only upsert (no PDF available)
                asyncio.run(save_to_mongo(paper_id, metadata))
                logger.info("ingest_papers: metadata-only paper_id=%r", paper_id)

            paper_ids.append(paper_id)
        except Exception as exc:
            logger.warning(
                "ingest_papers: failed for paper_id=%r (%s) — skipping.", paper_id, exc
            )

    return paper_ids


# ---------------------------------------------------------------------------
# Celery task
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    name="workers.tasks.ingest_topic",
    max_retries=3,
    default_retry_delay=5,
    acks_late=True,
)
def ingest_topic(self, topic_id: str, topic_text: str) -> dict[str, Any]:
    """
    Background task: ingest a topic's papers (Phases 1-2) and run
    contradiction detection (Phase 6).

    Parameters
    ----------
    topic_id : str
        Unique identifier for the topic (also MongoDB _id).
    topic_text : str
        Raw topic query string (e.g. "intermittent fasting insulin sensitivity").

    Returns
    -------
    dict
        ``{"paper_count": int, "contradiction_count": int}``
        Stored as the Celery task result in Redis.
    """
    logger.info(
        "ingest_topic START  task_id=%s  topic_id=%r",
        self.request.id, topic_id,
    )

    try:
        # ── Step 1: mark status = ingesting ───────────────────────────────
        _mongo_update_status(topic_id, {"status": "ingesting"})

        # ── Step 2: fetch papers from all sources (Phase 2) ───────────────
        logger.info("ingest_topic: fetching papers for %r …", topic_text)
        from app.services.fetch import fetch_all  # late import

        try:
            papers = asyncio.run(fetch_all(topic_text))
        except Exception as exc:
            logger.warning("ingest_topic: fetch_all failed (%s) — using empty list.", exc)
            papers = []

        logger.info("ingest_topic: fetched %d papers", len(papers))

        # ── Step 3: ingest each paper into Qdrant + MongoDB ───────────────
        paper_ids = _ingest_papers_sync(papers, topic_id)
        paper_count = len(paper_ids)
        logger.info("ingest_topic: ingested %d papers", paper_count)

        # Persist paper_ids + abstracts on the topic doc for /suggestions
        abstracts = [p.get("abstract") or "" for p in papers if p.get("abstract")]
        _mongo_update_status(topic_id, {
            "paper_ids":  paper_ids,
            "paper_count": paper_count,
            "abstracts":  abstracts[:10],  # top-10 for suggest_claims
        })

        # ── Step 4: persist stance buckets (Phase 3) — best-effort ────────
        # classify_paper_set needs a claim; here we use the topic_text as proxy
        # to get initial stance distribution so /verify can skip it later.
        # Failures are non-fatal — /verify will re-classify with the real claim.
        try:
            buckets = asyncio.run(classify_paper_set(topic_text, topic_id))
            logger.info(
                "ingest_topic: stance — support=%d refute=%d no_stance=%d",
                len(buckets["support"]), len(buckets["refute"]), len(buckets["no_stance"]),
            )
            _mongo_update_status(topic_id, {"stance_buckets": buckets})
        except Exception as exc:
            logger.warning("ingest_topic: stance classification failed (%s).", exc)

        # ── Step 5: contradiction check (Phase 6) ─────────────────────────
        contradiction_count = 0
        try:
            pairs = check_contradictions(topic_id, paper_ids=paper_ids)
            contradiction_count = len(pairs)
            logger.info(
                "ingest_topic: %d contradiction(s) found", contradiction_count
            )
        except Exception as exc:
            logger.warning("ingest_topic: contradiction check failed (%s).", exc)

        # ── Step 6: mark ready ─────────────────────────────────────────────
        _mongo_update_status(topic_id, {
            "status":              "ready",
            "contradiction_count": contradiction_count,
        })

        result = {
            "paper_count":        paper_count,
            "contradiction_count": contradiction_count,
        }
        logger.info("ingest_topic DONE  topic_id=%r  result=%s", topic_id, result)
        return result

    except Exception as exc:
        logger.error(
            "ingest_topic ERROR  task_id=%s  topic_id=%r: %s",
            self.request.id, topic_id, exc,
        )
        # On final retry exhaustion, mark failed
        if self.request.retries >= self.max_retries:
            _mongo_update_status(topic_id, {"status": "failed", "error": str(exc)})
            raise  # re-raise so Celery marks task as FAILURE

        # Otherwise retry
        raise self.retry(exc=exc)
