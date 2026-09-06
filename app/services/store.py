"""
app/services/store.py
---------------------
Storage service for ClaimGraph.

Provides:
  - save_to_qdrant(paper_id, chunks, vectors) – upserts chunk points into Qdrant
  - save_to_mongo(paper_id, metadata)         – upserts paper metadata into MongoDB
  - ingest_paper(paper_id, pdf_bytes, metadata) – full ingestion orchestrator

Both storage functions are DOI-keyed and skip if the paper is already cached
in MongoDB's global_papers collection (via check_global_cache from fetch.py).
Qdrant point IDs are deterministic (uuid5) so upserts are truly idempotent.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Optional

import numpy as np
from motor.motor_asyncio import AsyncIOMotorClient
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.services.fetch import check_global_cache
from app.services.parse import parse_pdf, chunk_text
from app.services.embed import embed_chunks

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / env
# ---------------------------------------------------------------------------
_MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017/claimgraph")
_DB_NAME: str = _MONGO_URI.rstrip("/").split("/")[-1]
_QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")

_COLLECTION: str = "papers"
_VECTOR_DIM: int = 768
_UPSERT_BATCH: int = 100

# UUID namespace for deterministic point IDs
_UUID_NS = uuid.NAMESPACE_URL


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------

def _ensure_qdrant_collection(
    client: QdrantClient,
    collection_name: str = _COLLECTION,
    dim: int = _VECTOR_DIM,
) -> None:
    """
    Create the Qdrant collection if it does not already exist.
    Safe to call on every run — idempotent.
    """
    try:
        client.get_collection(collection_name)
        logger.debug("Qdrant collection %r already exists", collection_name)
    except (UnexpectedResponse, Exception) as exc:  # noqa: BLE001
        # Any error → try to create (most likely the collection doesn't exist)
        logger.info(
            "Creating Qdrant collection %r (reason: %s)", collection_name, exc
        )
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        logger.info("Qdrant collection %r created", collection_name)


def _point_id(paper_id: str, index: int) -> str:
    """
    Generate a deterministic UUID for a Qdrant point.

    Using uuid5 ensures that re-ingesting the same paper produces the same
    point IDs, so ``client.upsert`` is truly idempotent.
    """
    return str(uuid.uuid5(_UUID_NS, f"{paper_id}:{index}"))


# ---------------------------------------------------------------------------
# Public: save_to_qdrant
# ---------------------------------------------------------------------------

def save_to_qdrant(
    paper_id: str,
    chunks: list[dict[str, Any]],
    vectors: np.ndarray,
    *,
    qdrant_url: Optional[str] = None,
    collection: str = _COLLECTION,
) -> int:
    """
    Upsert chunk embeddings into Qdrant.

    Parameters
    ----------
    paper_id:
        Unique string identifier for the paper (e.g. MongoDB ``_id``).
    chunks:
        List of ``{text, section, paper_id}`` dicts (from ``chunk_text``).
    vectors:
        Float32 numpy array of shape ``(len(chunks), 768)``.
    qdrant_url:
        Override for the Qdrant REST URL; defaults to ``QDRANT_URL`` env var.
    collection:
        Qdrant collection name.

    Returns
    -------
    int
        Number of points upserted.

    Raises
    ------
    ValueError
        If ``vectors.shape[1] != 768`` or ``len(chunks) != len(vectors)``.
    """
    if vectors.ndim != 2 or vectors.shape[1] != _VECTOR_DIM:
        raise ValueError(
            f"Vector dimensionality must be {_VECTOR_DIM}, "
            f"got shape {vectors.shape}."
        )
    if len(chunks) != len(vectors):
        raise ValueError(
            f"chunks ({len(chunks)}) and vectors ({len(vectors)}) must have the same length."
        )

    url = qdrant_url or _QDRANT_URL
    client = QdrantClient(url=url)
    _ensure_qdrant_collection(client, collection)

    points: list[PointStruct] = [
        PointStruct(
            id=_point_id(paper_id, i),
            vector=vectors[i].tolist(),
            payload={
                "text": chunk.get("text", ""),
                "section": chunk.get("section", ""),
                "paper_id": paper_id,
            },
        )
        for i, chunk in enumerate(chunks)
    ]

    # Upsert in batches
    total = 0
    for start in range(0, len(points), _UPSERT_BATCH):
        batch = points[start : start + _UPSERT_BATCH]
        client.upsert(collection_name=collection, points=batch)
        total += len(batch)
        logger.debug(
            "Upserted batch [%d:%d] for paper_id=%r",
            start,
            start + len(batch),
            paper_id,
        )

    logger.info(
        "Saved %d Qdrant points for paper_id=%r in collection %r",
        total,
        paper_id,
        collection,
    )
    return total


# ---------------------------------------------------------------------------
# Public: save_to_mongo
# ---------------------------------------------------------------------------

async def save_to_mongo(
    paper_id: str,
    metadata: dict[str, Any],
    *,
    db_client: Optional[AsyncIOMotorClient] = None,
) -> str:
    """
    Upsert paper metadata into MongoDB's ``global_papers`` collection.

    Uses ``update_one`` with ``$setOnInsert`` + ``upsert=True`` so calling
    this function multiple times with the same DOI is safe and idempotent.

    Parameters
    ----------
    paper_id:
        The application-level paper identifier (added to the stored doc).
    metadata:
        Dict containing at minimum ``"doi"``.  All keys are stored as-is.
    db_client:
        Optional existing motor client (reused connection pool).

    Returns
    -------
    str
        The string representation of the MongoDB ``_id`` of the upserted
        or matched document.

    Raises
    ------
    ValueError
        If ``metadata`` does not contain a ``"doi"`` key.
    """
    doi = metadata.get("doi")
    if not doi:
        raise ValueError("metadata must contain a 'doi' key for save_to_mongo.")

    _own_client = db_client is None
    client = db_client or AsyncIOMotorClient(_MONGO_URI)
    try:
        db = client[_DB_NAME]
        doc = {**metadata, "paper_id": paper_id}
        result = await db["global_papers"].update_one(
            {"doi": doi},
            {"$setOnInsert": doc},
            upsert=True,
        )
        if result.upserted_id is not None:
            mongo_id = str(result.upserted_id)
            logger.info(
                "Inserted new paper into MongoDB: doi=%r _id=%s", doi, mongo_id
            )
        else:
            # Document already existed — fetch its _id
            existing = await db["global_papers"].find_one({"doi": doi}, {"_id": 1})
            mongo_id = str(existing["_id"]) if existing else ""
            logger.debug("Paper already in MongoDB: doi=%r _id=%s", doi, mongo_id)
        return mongo_id
    finally:
        if _own_client:
            client.close()


# ---------------------------------------------------------------------------
# Public: ingest_paper (full orchestrator)
# ---------------------------------------------------------------------------

async def ingest_paper(
    paper_id: str,
    pdf_bytes: bytes,
    metadata: dict[str, Any],
    *,
    db_client: Optional[AsyncIOMotorClient] = None,
    qdrant_url: Optional[str] = None,
    collection: str = _COLLECTION,
    max_tokens: int = 300,
) -> tuple[int, str]:
    """
    Full ingestion pipeline for a single paper.

    Steps
    -----
    1. Check MongoDB global cache — if DOI already present, return immediately.
    2. ``parse_pdf`` → TEI-XML string.
    3. ``chunk_text`` → list of chunk dicts.
    4. ``embed_chunks`` → np.ndarray (N, 768).
    5. ``save_to_qdrant`` → upsert points.
    6. ``save_to_mongo`` → upsert metadata.

    Parameters
    ----------
    paper_id:
        Unique identifier for this paper (e.g. a UUID or slug).
    pdf_bytes:
        Raw PDF file bytes.
    metadata:
        Dict containing at minimum ``"doi"``.
    db_client:
        Optional shared motor client.
    qdrant_url:
        Override Qdrant URL.
    collection:
        Qdrant collection name.
    max_tokens:
        Passed to ``chunk_text``.

    Returns
    -------
    tuple[int, str]
        ``(n_chunks_stored, mongo_id)``
        If the paper was already cached, returns ``(0, cached_mongo_id)``.
    """
    doi = metadata.get("doi", "")

    # --- Step 1: Global cache check ---------------------------------------
    if doi:
        cached_id = await check_global_cache(doi, db_client=db_client)
        if cached_id:
            logger.info(
                "ingest_paper: DOI %r already in cache (mongo_id=%s) — skipping",
                doi,
                cached_id,
            )
            return 0, cached_id

    # --- Step 2: Parse PDF ------------------------------------------------
    tei_xml = await parse_pdf(pdf_bytes, paper_id=paper_id)

    # --- Step 3: Chunk ----------------------------------------------------
    chunks = chunk_text(tei_xml, max_tokens=max_tokens, paper_id=paper_id)
    if not chunks:
        logger.warning(
            "ingest_paper: no chunks produced for paper_id=%r doi=%r", paper_id, doi
        )
        return 0, ""

    # --- Step 4: Embed ----------------------------------------------------
    vectors = embed_chunks(chunks)

    # --- Step 5: Store in Qdrant ------------------------------------------
    n_stored = save_to_qdrant(
        paper_id,
        chunks,
        vectors,
        qdrant_url=qdrant_url,
        collection=collection,
    )

    # --- Step 6: Store metadata in MongoDB --------------------------------
    mongo_id = await save_to_mongo(paper_id, metadata, db_client=db_client)

    return n_stored, mongo_id
