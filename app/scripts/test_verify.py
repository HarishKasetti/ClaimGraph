"""
app/scripts/test_verify.py
--------------------------
CLI end-to-end verifier for Phase 4 geometric verification.

Pulls chunk embeddings from Qdrant, retrieves stance buckets from MongoDB,
embeds the claim via SPECTER2, then calls fit_geometry() and prints the
full verdict for manual inspection.

Usage
-----
    python -m app.scripts.test_verify \\
        --claim "Intermittent fasting improves insulin sensitivity in healthy adults" \\
        --topic-id <topic_id>

Options
-------
    --claim TEXT        Claim to verify (required)
    --topic-id TEXT     Topic identifier (required)
    --collection TEXT   Qdrant collection name (default: papers)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from collections import defaultdict

import numpy as np
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("test_verify")


async def _get_stance_buckets(topic_id: str) -> dict[str, list[str]]:
    """Fetch persisted stance buckets from MongoDB topics collection."""
    from motor.motor_asyncio import AsyncIOMotorClient

    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/claimgraph")
    db_name = mongo_uri.rstrip("/").split("/")[-1]
    client = AsyncIOMotorClient(mongo_uri)
    try:
        db = client[db_name]
        doc = await db["topics"].find_one({"_id": topic_id})
        if doc and "stance_buckets" in doc:
            return doc["stance_buckets"]
        return {"support": [], "refute": [], "no_stance": []}
    finally:
        client.close()


def _get_paper_vectors(
    paper_ids: list[str],
    collection: str,
) -> np.ndarray:
    """
    Retrieve the mean embedding for each paper_id from Qdrant.
    Returns array of shape (len(paper_ids), dim) or (0, 768) if none found.
    """
    if not paper_ids:
        return np.zeros((0, 768), dtype=np.float32)

    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchAny

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=qdrant_url)

    scroll_filter = Filter(
        must=[
            FieldCondition(
                key="paper_id",
                match=MatchAny(any=paper_ids),
            )
        ]
    )

    points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=scroll_filter,
        limit=5000,
        with_payload=True,
        with_vectors=True,
    )

    # Group vectors by paper_id and take mean
    paper_vecs: dict[str, list[list[float]]] = defaultdict(list)
    for p in points:
        pid = (p.payload or {}).get("paper_id", "")
        if pid and p.vector:
            paper_vecs[pid].append(p.vector)

    means = []
    for pid in paper_ids:
        if pid in paper_vecs:
            means.append(np.mean(paper_vecs[pid], axis=0))
        else:
            logger.warning("No vectors found for paper_id=%r -- skipping.", pid)

    if not means:
        return np.zeros((0, 768), dtype=np.float32)
    return np.array(means, dtype=np.float32)


def _embed_claim(claim: str) -> np.ndarray:
    """Embed the claim string using SPECTER2."""
    from app.services.embed import get_model
    model = get_model()
    vec = model.encode([claim], convert_to_numpy=True, show_progress_bar=False)
    return vec[0].astype(np.float32)


async def _run(claim: str, topic_id: str, collection: str) -> None:
    from app.services.geometry import fit_geometry

    print("\n" + "=" * 64)
    print(f"CLAIM    : {claim}")
    print(f"TOPIC-ID : {topic_id}")
    print("=" * 64)

    # --- Fetch stance buckets from MongoDB ----------------------------------
    print("\n[1/4] Loading stance buckets from MongoDB ...")
    buckets = await _get_stance_buckets(topic_id)
    for lbl, ids in buckets.items():
        print(f"      {lbl}: {len(ids)} paper(s) -> {ids[:3]}")

    total_papers = sum(len(v) for v in buckets.values())
    if total_papers == 0:
        print(
            "\n[WARN] No papers in any bucket. Run classify_paper_set first:\n"
            "       python -m app.scripts.test_stance --claim <...> "
            f"--topic-id {topic_id}"
        )

    # --- Fetch paper vectors from Qdrant ------------------------------------
    print("\n[2/4] Fetching paper vectors from Qdrant ...")
    bucket_vectors: dict[str, np.ndarray] = {}
    for label, paper_ids in buckets.items():
        vecs = _get_paper_vectors(paper_ids, collection)
        bucket_vectors[label] = vecs
        print(f"      {label}: {vecs.shape}")

    non_empty_count = sum(1 for v in bucket_vectors.values() if len(v) > 0)
    if non_empty_count == 0:
        print(
            "\n[WARN] No vectors retrieved. Ingest papers and run stance "
            "classification first."
        )
        return

    # --- Embed the claim ---------------------------------------------------
    print("\n[3/4] Embedding claim via SPECTER2 ...")
    claim_vector = _embed_claim(claim)
    print(f"      Claim vector shape: {claim_vector.shape}")

    # --- Run geometry ------------------------------------------------------
    print("\n[4/4] Running geometric verification ...")
    try:
        result = fit_geometry(bucket_vectors, claim_vector)
    except ValueError as exc:
        print(f"\n[ERROR] fit_geometry failed: {exc}")
        return

    # --- Print verdict -----------------------------------------------------
    print("\n" + "=" * 64)
    print("GEOMETRIC VERDICT")
    print("=" * 64)
    print(f"  Winning label     : {result['winning_label'].upper()}")
    print(f"  Margin            : {result['margin']:.4f}")
    print(f"  Is ambiguous      : {result['is_ambiguous']}")
    print(f"  Low sample warning: {result['low_sample_warning']}")
    print(f"  PCA components    : {result['pca_n_components']}")
    print(f"\n  Distances:")
    for lbl, dist in result["distances"].items():
        tag = "<-- winner" if lbl == result["winning_label"] else ""
        dist_str = f"{dist:.4f}" if not (dist == float("inf")) else "inf (empty bucket)"
        print(f"    {lbl:<12}: {dist_str}  {tag}")

    if result["is_ambiguous"]:
        print("\n  [WARN] Result is AMBIGUOUS -- retry or ingest more papers.")
    if result["low_sample_warning"]:
        print("\n  [WARN] LOW SAMPLE WARNING -- cosine fallback used for small bucket.")
    print("\n" + "=" * 64 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4 geometric verification CLI."
    )
    parser.add_argument("--claim", required=True, help="Claim to verify")
    parser.add_argument("--topic-id", required=True, dest="topic_id")
    parser.add_argument("--collection", default="papers")
    args = parser.parse_args()

    asyncio.run(_run(args.claim, args.topic_id, args.collection))


if __name__ == "__main__":
    main()
