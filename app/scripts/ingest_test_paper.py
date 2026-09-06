"""
app/scripts/ingest_test_paper.py
---------------------------------
CLI script for end-to-end ingestion of a single PDF into the ClaimGraph
pipeline: GROBID parse → chunk → SPECTER2 embed → Qdrant + MongoDB store.

Usage
-----
    python -m app.scripts.ingest_test_paper --pdf tests/sample.pdf [OPTIONS]

Options
-------
    --pdf PATH          Path to the PDF file to ingest (required)
    --doi TEXT          Override DOI (default: 10.0000/test.<stem>)
    --title TEXT        Override title (default: derived from filename)
    --collection TEXT   Qdrant collection name (default: papers)
    --max-tokens INT    Chunk token limit (default: 300)
    --print-chunks INT  Number of sample chunks to print (default: 3)
    --second-run        Run ingest a second time to verify cache skip
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("ingest_test_paper")


async def _run(
    pdf_path: Path,
    doi: str,
    title: str,
    collection: str,
    max_tokens: int,
    print_chunks: int,
    second_run: bool,
) -> None:
    from app.services.parse import parse_pdf, chunk_text
    from app.services.embed import embed_chunks
    from app.services.store import save_to_qdrant, save_to_mongo, ingest_paper
    from app.services.fetch import check_global_cache

    pdf_bytes = pdf_path.read_bytes()
    paper_id = f"test_{pdf_path.stem}"

    metadata = {
        "doi": doi,
        "title": title,
        "abstract": "Test paper abstract for ingestion verification.",
        "venue": "Test Venue",
        "citation_count": 0,
        "pdf_url": None,
        "source": "test",
    }

    # ── First ingestion ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"INGEST RUN 1  |  paper_id={paper_id!r}")
    print("=" * 60)

    n_chunks, mongo_id = await ingest_paper(
        paper_id,
        pdf_bytes,
        metadata,
        collection=collection,
        max_tokens=max_tokens,
    )

    print(f"\n  Chunks stored in Qdrant : {n_chunks}")
    print(f"  MongoDB _id             : {mongo_id}")

    # ── Print sample chunks ──────────────────────────────────────────────────
    if n_chunks > 0:
        tei_xml = await parse_pdf(pdf_bytes, paper_id=paper_id)
        chunks = chunk_text(tei_xml, max_tokens=max_tokens, paper_id=paper_id)
        print(f"\n  --- {min(print_chunks, len(chunks))} of {len(chunks)} chunks ---")
        for i, c in enumerate(chunks[:print_chunks], 1):
            words = c["text"].split()
            print(f"\n  Chunk {i}  |  section={c['section']!r}  |  words={len(words)}")
            # Show first and last sentence to verify no mid-sentence break
            preview = " ".join(words[:20])
            tail    = " ".join(words[-10:]) if len(words) > 20 else ""
            print(f"    START: {preview!r}")
            if tail:
                print(f"    END  : ...{tail!r}")
    else:
        print("  (0 chunks — paper was already cached or PDF was empty)")

    # ── Second run (cache check) ─────────────────────────────────────────────
    if second_run:
        print("\n" + "=" * 60)
        print(f"INGEST RUN 2  |  Expect cache hit, 0 chunks stored")
        print("=" * 60)

        n2, mongo_id2 = await ingest_paper(
            paper_id,
            pdf_bytes,
            metadata,
            collection=collection,
            max_tokens=max_tokens,
        )

        print(f"\n  Chunks stored : {n2}")
        print(f"  MongoDB _id   : {mongo_id2}")

        if n2 == 0 and mongo_id2:
            print("\n  [PASS] Cache hit confirmed — second run skipped ingestion.")
        else:
            print(f"\n  [FAIL] Expected (0, cached_id), got ({n2}, {mongo_id2!r})")
            sys.exit(1)

    # ── Qdrant collection info ────────────────────────────────────────────────
    try:
        from qdrant_client import QdrantClient
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qc = QdrantClient(url=qdrant_url)
        info = qc.get_collection(collection)
        vcount = getattr(info, "points_count", getattr(info, "vectors_count", 0))
        # Navigate the config structure defensively
        try:
            vsize = info.config.params.vectors.size
        except AttributeError:
            vsize = getattr(info.config.params.vectors, "size", "unknown")

        print(f"\n  Qdrant collection : {collection!r}")
        print(f"  Vector count      : {vcount}")
        print(f"  Vector size       : {vsize}")

        if vsize != 768:
            print(f"\n  [FAIL] Vector size {vsize} != 768 — CRITICAL dimension mismatch!")
            sys.exit(1)
        else:
            print("\n  [PASS] Vector size == 768")
    except Exception as exc:  # noqa: BLE001
        print(f"\n  [WARN] Could not query Qdrant: {exc}")

    print("\n" + "=" * 60)
    print("Ingestion complete.")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a PDF into ClaimGraph (parse -> chunk -> embed -> store)."
    )
    parser.add_argument("--pdf", required=True, type=Path, help="Path to PDF file")
    parser.add_argument(
        "--doi",
        default=None,
        help="DOI for the paper (default: 10.0000/test.<stem>)",
    )
    parser.add_argument("--title", default=None, help="Paper title (default: filename)")
    parser.add_argument(
        "--collection", default="papers", help="Qdrant collection name"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=300, help="Chunk token limit"
    )
    parser.add_argument(
        "--print-chunks",
        type=int,
        default=3,
        help="Number of sample chunks to display",
    )
    parser.add_argument(
        "--second-run",
        action="store_true",
        help="Run ingest twice to verify DOI cache hit on second run",
    )
    args = parser.parse_args()

    pdf_path: Path = args.pdf
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        sys.exit(1)

    doi   = args.doi   or f"10.0000/test.{pdf_path.stem}"
    title = args.title or f"Test Paper: {pdf_path.stem}"

    asyncio.run(
        _run(
            pdf_path=pdf_path,
            doi=doi,
            title=title,
            collection=args.collection,
            max_tokens=args.max_tokens,
            print_chunks=args.print_chunks,
            second_run=args.second_run,
        )
    )


if __name__ == "__main__":
    main()
