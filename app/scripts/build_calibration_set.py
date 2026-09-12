"""
app/scripts/build_calibration_set.py
-------------------------------------
Phase 5 Stage 5 — Build Calibration Set.

Downloads the SciFact dev set from HuggingFace datasets and runs the full
Phase 3+4 pipeline (stance classification → geometric verification) on each
claim/abstract pair, producing (margin, correct) calibration pairs saved to
``data/calibration_set.jsonl``.

Design
------
For each SciFact dev example:
  1. Extract the claim text and the first cited abstract.
  2. Sentence-tokenise the abstract into chunks.
  3. Run DeBERTa zero-shot classification on each sentence to assign it to a
     stance bucket (support / refute / no_stance).
  4. Embed all sentences + the claim with SPECTER2.
  5. Build bucket_vectors dict (one vector per sentence, keyed by its bucket).
  6. Call fit_geometry() to get the geometric verdict and margin.
  7. Map ground-truth label to our label space:
       SUPPORTS        -> support
       CONTRADICTS     -> refute
       NOT ENOUGH INFO -> no_stance
  8. correct = int(winning_label == ground_truth_label)
  9. Append {"margin": float, "correct": int, "claim": str} to JSONL.

Skips examples where:
  - No cited abstract is available.
  - fit_geometry raises ValueError (all buckets empty).

Usage
-----
    python -m app.scripts.build_calibration_set

Output
------
    data/calibration_set.jsonl
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("build_calibration_set")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_OUTPUT_PATH = Path("data/calibration_set.jsonl")
_GT_LABEL_MAP: dict[str, str] = {
    "SUPPORTS": "support",
    "CONTRADICTS": "refute",
    "NOT ENOUGH INFO": "no_stance",
}
_LABELS = ["support", "refute", "no_stance"]

# Sentence boundary: split on ". " to avoid mid-sentence chunks
_MIN_SENTENCE_CHARS = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sentence_split(text: str) -> list[str]:
    """Naive sentence splitter: split on '. ' and '? ' and '! '."""
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) >= _MIN_SENTENCE_CHARS]


def _classify_sentence(pipe, claim: str, sentence: str) -> str:
    """Zero-shot classify a single sentence against the claim."""
    tokens = sentence.split()
    if len(tokens) > 512:
        sentence = " ".join(tokens[:512])
    result = pipe(
        sentence,
        candidate_labels=_LABELS,
        hypothesis_template="This text {} the claim: " + claim,
        multi_label=False,
    )
    return result["labels"][0]


def _embed_texts(model, texts: list[str]) -> np.ndarray:
    """Embed a list of texts with SPECTER2. Returns (N, 768) float32 array."""
    return model.encode(
        texts,
        batch_size=16,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)


def _map_ground_truth(label: str) -> str:
    return _GT_LABEL_MAP.get(label, "no_stance")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_calibration_set(max_examples: int | None = None) -> None:
    """
    Run the pipeline over SciFact dev set and write calibration JSONL.

    Parameters
    ----------
    max_examples : int or None
        Cap the number of examples processed (useful for quick testing).
        None = process all.
    """
    # Late imports: heavy ML libs loaded only when script actually runs
    from datasets import load_dataset  # type: ignore[import]
    from app.services.embed import get_model as get_embed_model
    from app.services.geometry import fit_geometry
    from app.services.stance import get_pipeline

    logger.info("Loading SciFact dataset from HuggingFace …")
    ds = load_dataset("allenai/scifact", "claims", trust_remote_code=True)
    dev_split = ds["validation"]

    # Also load the corpus to resolve abstract text from doc_id
    corpus_ds = load_dataset("allenai/scifact", "corpus", trust_remote_code=True)
    corpus: dict[int, str] = {}
    for row in corpus_ds["train"]:
        doc_id = int(row["doc_id"])
        # Flatten sentences list -> paragraph
        abstract_sents = row.get("abstract", [])
        if isinstance(abstract_sents, list):
            abstract = " ".join(abstract_sents)
        else:
            abstract = str(abstract_sents)
        corpus[doc_id] = abstract

    logger.info("Corpus loaded: %d documents", len(corpus))
    logger.info("Loading SPECTER2 model …")
    embed_model = get_embed_model()
    logger.info("Loading stance pipeline …")
    stance_pipe = get_pipeline()

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    with _OUTPUT_PATH.open("w", encoding="utf-8") as fout:
        for idx, row in enumerate(dev_split):
            if max_examples is not None and idx >= max_examples:
                break

            claim: str = row.get("claim", "")
            # SciFact claims has cited_doc_ids and evidence labels
            cited_doc_ids = row.get("cited_doc_ids", [])
            evidence = row.get("evidence", {})  # dict: doc_id -> list of evidence objs

            if not cited_doc_ids:
                logger.debug("Row %d: no cited_doc_ids, skipping.", idx)
                skipped += 1
                continue

            # Pick the first cited document with an abstract
            abstract_text: str = ""
            gt_label: str = "no_stance"
            chosen_doc_id: int | None = None

            for doc_id in cited_doc_ids:
                doc_id_int = int(doc_id)
                if doc_id_int in corpus:
                    abstract_text = corpus[doc_id_int]
                    # Determine ground-truth label from evidence
                    doc_id_str = str(doc_id_int)
                    if doc_id_str in evidence and evidence[doc_id_str]:
                        ev_list = evidence[doc_id_str]
                        # Take first evidence object's label
                        raw_label = ev_list[0].get("label", "NOT ENOUGH INFO")
                        gt_label = _map_ground_truth(raw_label)
                    else:
                        gt_label = "no_stance"
                    chosen_doc_id = doc_id_int
                    break

            if not abstract_text:
                logger.debug("Row %d: no abstract found, skipping.", idx)
                skipped += 1
                continue

            # --- Sentence split ---
            sentences = _sentence_split(abstract_text)
            if len(sentences) < 2:
                logger.debug("Row %d: too few sentences (%d), skipping.", idx, len(sentences))
                skipped += 1
                continue

            # --- Classify each sentence into a stance bucket ---
            buckets_texts: dict[str, list[str]] = {
                "support": [], "refute": [], "no_stance": []
            }
            for sent in sentences:
                lbl = _classify_sentence(stance_pipe, claim, sent)
                buckets_texts[lbl].append(sent)

            # Ensure at least one non-empty bucket for fit_geometry
            non_empty = [k for k, v in buckets_texts.items() if v]
            if not non_empty:
                skipped += 1
                continue

            # --- Embed all texts ---
            all_texts = sentences + [claim]
            try:
                all_vecs = _embed_texts(embed_model, all_texts)
            except Exception as exc:
                logger.warning("Row %d: embedding failed: %s", idx, exc)
                skipped += 1
                continue

            sent_vecs = all_vecs[: len(sentences)]
            claim_vec = all_vecs[-1]

            # Build bucket_vectors dict
            bucket_vectors: dict[str, np.ndarray] = {}
            offset = 0
            sent_bucket_map: list[str] = []
            for sent in sentences:
                lbl = _classify_sentence(stance_pipe, claim, sent)
                sent_bucket_map.append(lbl)

            # Collect per-bucket vector arrays
            for lbl in _LABELS:
                idxs = [i for i, b in enumerate(sent_bucket_map) if b == lbl]
                if idxs:
                    bucket_vectors[lbl] = sent_vecs[idxs]

            # --- Geometric verdict ---
            try:
                verdict = fit_geometry(bucket_vectors, claim_vec)
            except ValueError as exc:
                logger.warning("Row %d: fit_geometry failed: %s", idx, exc)
                skipped += 1
                continue

            margin: float = verdict["margin"]
            winning: str = verdict["winning_label"]
            correct: int = int(winning == gt_label)

            record = {
                "margin": float(margin) if np.isfinite(margin) else 10.0,
                "correct": correct,
                "claim": claim,
                "winning_label": winning,
                "gt_label": gt_label,
                "doc_id": chosen_doc_id,
            }
            fout.write(json.dumps(record) + "\n")
            written += 1

            if written % 10 == 0:
                logger.info(
                    "Progress: %d written, %d skipped (acc so far: %.1f%%)",
                    written,
                    skipped,
                    100.0 * sum(
                        1 for _ in [record] if record["correct"]
                    ) / max(written, 1),
                )

    logger.info(
        "Done. Wrote %d calibration pairs (%d skipped) → %s",
        written,
        skipped,
        _OUTPUT_PATH,
    )
    if written == 0:
        logger.error(
            "No pairs written! Check that Qdrant / stance model are accessible."
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build calibration set from SciFact dev.")
    parser.add_argument(
        "--max-examples", type=int, default=None,
        help="Cap number of dev examples to process (default: all)"
    )
    args = parser.parse_args()
    build_calibration_set(max_examples=args.max_examples)
