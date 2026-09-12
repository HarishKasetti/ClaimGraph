"""
app/scripts/test_contradictions.py
------------------------------------
Phase 6 Stage 6 — Contradiction Check CLI.

Fetches all papers for a topic, runs check_contradictions(), and prints
flagged pairs for manual cross-verification.

Usage
-----
    python -m app.scripts.test_contradictions --topic-id <id>

Options
-------
    --topic-id TEXT     Topic identifier (required)
    --threshold FLOAT   Override DeBERTa refute score threshold (default: 0.70)
    --no-persist        Skip MongoDB persistence of flagged pairs

Cross-verify checklist
----------------------
  1. Read the "Finding A" and "Finding B" for each flagged pair.
     Do they actually disagree, or are they flagged for unrelated reasons?
  2. Check the all-grey case: if no contradictions, the script prints
     "No contradictions detected" — this is a valid output, not a bug.
  3. Check short paper labels in the heatmap JSON match human-readable titles.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

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
logger = logging.getLogger("test_contradictions")


async def _get_paper_titles(paper_ids: list[str]) -> dict[str, str]:
    """Fetch paper titles from MongoDB for display."""
    from motor.motor_asyncio import AsyncIOMotorClient

    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/claimgraph")
    db_name = mongo_uri.rstrip("/").split("/")[-1]
    client = AsyncIOMotorClient(mongo_uri)
    try:
        db = client[db_name]
        titles: dict[str, str] = {}
        async for doc in db["global_papers"].find(
            {"paper_id": {"$in": paper_ids}},
            {"paper_id": 1, "title": 1},
        ):
            titles[doc["paper_id"]] = doc.get("title", doc["paper_id"])
        return titles
    finally:
        client.close()


def _run(topic_id: str, threshold: float, persist: bool) -> None:
    from app.services.contradiction import (
        check_contradictions,
        generate_contradiction_map,
        _fetch_topic_paper_ids,
        _CONTRADICTION_THRESHOLD,
    )
    import app.services.contradiction as cmod

    # Temporarily override threshold if requested
    original_threshold = cmod._CONTRADICTION_THRESHOLD
    cmod._CONTRADICTION_THRESHOLD = threshold

    try:
        print("\n" + "=" * 70)
        print(f"CONTRADICTION CHECK  |  topic_id = {topic_id!r}")
        print(f"Threshold            : refute score > {threshold:.2f}")
        print("=" * 70)

        # --- Fetch paper IDs ------------------------------------------------
        print("\n[1/4] Fetching paper IDs from MongoDB …")
        try:
            paper_ids = asyncio.run(_fetch_topic_paper_ids(topic_id))
        except Exception as exc:
            print(f"      [ERROR] Could not fetch paper_ids: {exc}")
            return

        if not paper_ids:
            print(
                "      [WARN] No papers found for this topic.\n"
                "      Run stance classification first:\n"
                f"      python -m app.scripts.test_stance --topic-id {topic_id}"
            )
            return

        print(f"      Found {len(paper_ids)} paper(s):")
        try:
            titles = asyncio.run(_get_paper_titles(paper_ids))
        except Exception:
            titles = {}
        for i, pid in enumerate(paper_ids, 1):
            title = titles.get(pid, "—")
            print(f"      [{i:2d}] {pid}  |  {title[:60]}")

        # --- Extract main findings ------------------------------------------
        print(f"\n[2/4] Extracting main findings via Ollama llama3 …")
        from app.services.contradiction import extract_main_finding

        findings: dict[str, str] = {}
        for pid in paper_ids:
            f = extract_main_finding(pid)
            findings[pid] = f
            short_title = titles.get(pid, pid)[:40]
            print(f"      {short_title:<40}  →  {f[:80]!r}")

        # --- Check contradictions ------------------------------------------
        print(f"\n[3/4] Classifying {len(paper_ids) * (len(paper_ids)-1) // 2}"
              f" pairs with DeBERTa …")
        contradictions = check_contradictions(
            topic_id, paper_ids=paper_ids
        )

        if not contradictions:
            print("\n  ✅  No contradictions detected above threshold.")
            print("     All-grey heatmap is expected and correct.")
        else:
            print(f"\n  ⚠️   {len(contradictions)} contradiction(s) flagged:\n")
            for i, pair in enumerate(contradictions, 1):
                a_title = titles.get(pair["paper_a_id"], pair["paper_a_id"])[:50]
                b_title = titles.get(pair["paper_b_id"], pair["paper_b_id"])[:50]
                print(f"  [{i}] Confidence: {pair['confidence']:.3f}")
                print(f"       Paper A : {a_title}")
                print(f"       Finding A: {pair['finding_a'][:120]}")
                print(f"       Paper B : {b_title}")
                print(f"       Finding B: {pair['finding_b'][:120]}")
                print()

        # --- Generate heatmap -----------------------------------------------
        print("[4/4] Generating contradiction map …")
        try:
            chart_json = generate_contradiction_map(
                topic_id,
                paper_ids=paper_ids,
                paper_titles=titles,
            )
            parsed = json.loads(chart_json)
            n_traces = len(parsed.get("data", []))
            print(f"      Plotly JSON generated: {len(chart_json)} chars, {n_traces} trace(s)")
            print("      Paste into https://chart-studio.plotly.com/create to render")

            # Save for inspection
            out_path = Path(f"contradiction_map_{topic_id[:20]}.json")
            out_path.write_text(chart_json, encoding="utf-8")
            print(f"      Saved → {out_path}")
        except Exception as exc:
            print(f"      [ERROR] Chart generation failed: {exc}")

    finally:
        cmod._CONTRADICTION_THRESHOLD = original_threshold

    print("\n" + "=" * 70)
    print("Cross-verify checklist:")
    print("  1. Read the flagged findings above — do they actually disagree?")
    print("  2. Zero contradictions = all-grey heatmap (valid output).")
    print("  3. Check paper labels in JSON are short titles, not raw IDs.")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6 — Contradiction Check CLI"
    )
    parser.add_argument("--topic-id", required=True, dest="topic_id",
                        help="Topic identifier")
    parser.add_argument("--threshold", type=float, default=0.70,
                        help="Minimum refute score to flag (default: 0.70)")
    parser.add_argument("--no-persist", action="store_true",
                        help="Skip MongoDB persistence of flagged pairs")
    args = parser.parse_args()
    _run(args.topic_id, args.threshold, not args.no_persist)


if __name__ == "__main__":
    main()
