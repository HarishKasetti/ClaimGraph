"""
app/scripts/test_stance.py
--------------------------
CLI end-to-end verifier for the stance classification pipeline.

Retrieves all stored chunks for a topic from Qdrant, runs classify_paper_set
against a user-supplied claim, prints bucket sizes and sample passages for
manual review.

Usage
-----
    python -m app.scripts.test_stance \
        --claim "Intermittent fasting improves insulin sensitivity in healthy adults" \
        --topic-id <topic_id>

Options
-------
    --claim TEXT        Claim to verify against stored chunks (required)
    --topic-id TEXT     Topic identifier used to filter Qdrant chunks (required)
    --collection TEXT   Qdrant collection name (default: papers)
    --samples INT       Number of sample passages to print per bucket (default: 2)
    --no-persist        Skip MongoDB persistence (dry-run mode)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

# Ensure UTF-8 output on Windows regardless of terminal codepage
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("test_stance")


async def _run(
    claim: str,
    topic_id: str,
    collection: str,
    n_samples: int,
    no_persist: bool,
) -> None:
    from app.services.stance import classify_paper_set, _qdrant_scroll

    print("\n" + "=" * 64)
    print(f"CLAIM   : {claim}")
    print(f"TOPIC   : {topic_id}")
    print("=" * 64)

    # --- Retrieve raw chunks once so we can show sample passages later ------
    print("\n[1/2] Fetching chunks from Qdrant ...")
    payloads = _qdrant_scroll(topic_id, collection=collection)
    print(f"      {len(payloads)} chunks retrieved from collection {collection!r}.")

    if not payloads:
        print("\n[WARN] No chunks found. Ingest some papers first.")
        print("       Run:  python -m app.scripts.ingest_test_paper --pdf tests/sample.pdf")
        sys.exit(0)

    # --- Run classify_paper_set ---------------------------------------------
    print("\n[2/2] Running stance classification ...")
    if no_persist:
        # Dry-run: patch out MongoDB persistence
        from unittest.mock import AsyncMock, patch
        with patch("app.services.stance._mongo_persist", new_callable=AsyncMock):
            buckets = await classify_paper_set(
                claim, topic_id, collection=collection
            )
    else:
        buckets = await classify_paper_set(
            claim, topic_id, collection=collection
        )

    # --- Print summary -------------------------------------------------------
    print("\n" + "=" * 64)
    print("STANCE BUCKET RESULTS")
    print("=" * 64)
    total_papers = sum(len(v) for v in buckets.values())
    print(f"  Total papers classified : {total_papers}")
    print(f"  Support   : {len(buckets['support'])}")
    print(f"  Refute    : {len(buckets['refute'])}")
    print(f"  No-stance : {len(buckets['no_stance'])}")

    # --- Calibration warning -------------------------------------------------
    if not buckets["no_stance"] and total_papers > 0:
        print(
            "\n  [WARN] no_stance bucket is EMPTY -- possible miscalibration."
            "\n         Check that unrelated passages exist in the collection."
        )

    # --- Print sample passages per bucket -----------------------------------
    # Build lookup: paper_id -> [passages]
    paper_chunks: dict[str, list[str]] = {}
    for p in payloads:
        pid = p.get("paper_id", "unknown")
        text = p.get("text", "")
        paper_chunks.setdefault(pid, []).append(text)

    for bucket_name in ("support", "refute", "no_stance"):
        paper_ids = buckets[bucket_name]
        print(f"\n{'-'*64}")
        print(f"  Bucket: {bucket_name.upper()} ({len(paper_ids)} paper(s))")
        print(f"{'-'*64}")
        for paper_id in paper_ids[:n_samples]:
            print(f"\n  paper_id: {paper_id}")
            passages = paper_chunks.get(paper_id, [])
            if passages:
                # Show highest-token passage as the representative excerpt
                best_passage = max(passages, key=lambda t: len(t.split()))
                words = best_passage.split()
                preview = " ".join(words[:60])
                print(f"  Excerpt : {preview!r}")
                if len(words) > 60:
                    print(f"            ...({len(words)} words total)")
            else:
                print("  (no passage text available)")
        if len(paper_ids) > n_samples:
            print(f"  ... and {len(paper_ids) - n_samples} more paper(s).")

    if not no_persist:
        print(f"\n[OK] Bucket results persisted to MongoDB (topic_id={topic_id!r}).")
    print("\n" + "=" * 64 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end stance classification CLI verifier."
    )
    parser.add_argument(
        "--claim", required=True, help="Scientific claim to verify"
    )
    parser.add_argument(
        "--topic-id", required=True, dest="topic_id",
        help="Topic identifier used to filter Qdrant chunks"
    )
    parser.add_argument(
        "--collection", default="papers",
        help="Qdrant collection name (default: papers)"
    )
    parser.add_argument(
        "--samples", type=int, default=2, dest="n_samples",
        help="Number of sample passages to print per bucket"
    )
    parser.add_argument(
        "--no-persist", action="store_true", dest="no_persist",
        help="Skip MongoDB persistence (dry-run mode)"
    )
    args = parser.parse_args()

    asyncio.run(
        _run(
            claim=args.claim,
            topic_id=args.topic_id,
            collection=args.collection,
            n_samples=args.n_samples,
            no_persist=args.no_persist,
        )
    )


if __name__ == "__main__":
    main()

