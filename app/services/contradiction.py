"""
app/services/contradiction.py
------------------------------
Phase 6 Stage 6 — Contradiction Check Service.

Runs independently of any claim — fires right after Phase 2 (paper fetch)
completes. Pairwise-compares the main findings of all topic papers and
produces a contradiction map for the frontend.

Public API
----------
extract_main_finding(paper_id) -> str
    Retrieves the paper's abstract from MongoDB and calls Ollama llama3 to
    condense it into exactly one sentence describing the main finding.
    Results are in-process cached to avoid redundant LLM calls across pairs.

check_contradictions(topic_id) -> list[dict]
    Fetches all paper_ids for the topic, extracts main findings, loops over
    all C(n,2) pairs, classifies each pair with DeBERTa, flags pairs where
    label == "refute" AND score > 0.7, and persists flagged pairs to MongoDB.
    Returns list of contradiction dicts (may be empty).

generate_contradiction_map(topic_id) -> str
    Builds an NxN Plotly annotated heatmap where:
      Green  = papers agree (support label)
      Red    = papers contradict (refute label)
      Grey   = no-stance or diagonal (paper vs itself)
    Returns a self-contained Plotly JSON string.

Design notes
------------
- Ollama calls use the same _call_ollama pattern from reformulate.py; the
  function is imported to avoid code duplication.
- The finding cache (module-level dict) is keyed by paper_id and lives for
  the process lifetime — suitable for a single topic run per worker.
- DeBERTa zero-shot is reused via stance.get_pipeline() — same lazy singleton.
- The contradiction threshold is score > 0.7 on the "refute" label.  This
  deliberately avoids flagging weak, uncertain refutations.
- MongoDB persistence uses a deterministic pair-hash _id so upserts are safe
  to call repeatedly.
- generate_contradiction_map works correctly with zero contradictions — it
  produces an all-grey heatmap.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import logging
import math
import os
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / env
# ---------------------------------------------------------------------------
_MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017/claimgraph")
_DB_NAME: str = _MONGO_URI.rstrip("/").split("/")[-1]

_CONTRADICTION_THRESHOLD: float = 0.70   # min DeBERTa refute score to flag
_LABELS: list[str] = ["support", "refute", "no_stance"]

# Short-title word count for heatmap axis labels
_TITLE_WORDS: int = 4

# ---------------------------------------------------------------------------
# Module-level import of get_pipeline so tests can patch
# app.services.contradiction.get_pipeline without needing to know the
# late-import location inside function bodies.
# ---------------------------------------------------------------------------
from app.services.stance import get_pipeline  # noqa: E402

# ---------------------------------------------------------------------------
# In-process finding cache  {paper_id: one_sentence_finding}
# ---------------------------------------------------------------------------
_finding_cache: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Internal: Ollama helper (reuse from reformulate to avoid duplication)
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str) -> str:
    """
    POST to the local Ollama /api/generate endpoint.
    Delegates to reformulate._call_ollama so there is a single implementation.

    Falls back to raising on error — callers must handle gracefully.
    """
    from app.services.reformulate import _call_ollama as _base_call  # late import
    return _base_call(prompt)


# ---------------------------------------------------------------------------
# Internal: MongoDB helpers
# ---------------------------------------------------------------------------

async def _fetch_paper_doc(
    paper_id: str,
    *,
    db_client=None,
) -> dict[str, Any]:
    """
    Retrieve a single paper document from MongoDB global_papers.

    Looks up by the ``paper_id`` field first; if not found, tries ``_id``.
    Returns an empty dict if not found.
    """
    from motor.motor_asyncio import AsyncIOMotorClient  # late import

    own_client = db_client is None
    client = db_client or AsyncIOMotorClient(_MONGO_URI)
    try:
        db = client[_DB_NAME]
        doc = await db["global_papers"].find_one({"paper_id": paper_id})
        if not doc:
            doc = await db["global_papers"].find_one({"_id": paper_id})
        return dict(doc) if doc else {}
    finally:
        if own_client:
            client.close()


async def _fetch_topic_paper_ids(
    topic_id: str,
    *,
    db_client=None,
) -> list[str]:
    """
    Retrieve all paper_ids associated with a topic from MongoDB.

    Reads ``stance_buckets`` from the ``topics`` collection and flattens all
    bucket lists into a deduplicated list of paper_ids.
    """
    from motor.motor_asyncio import AsyncIOMotorClient  # late import

    own_client = db_client is None
    client = db_client or AsyncIOMotorClient(_MONGO_URI)
    try:
        db = client[_DB_NAME]
        doc = await db["topics"].find_one({"_id": topic_id})
        if not doc:
            logger.warning(
                "_fetch_topic_paper_ids: topic_id=%r not found in MongoDB.", topic_id
            )
            return []
        buckets: dict[str, list[str]] = doc.get("stance_buckets", {})
        seen: set[str] = set()
        ids: list[str] = []
        for bucket_ids in buckets.values():
            for pid in bucket_ids:
                if pid not in seen:
                    seen.add(pid)
                    ids.append(pid)
        logger.info(
            "_fetch_topic_paper_ids: topic_id=%r -> %d unique paper_ids",
            topic_id, len(ids),
        )
        return ids
    finally:
        if own_client:
            client.close()


async def _persist_contradiction(pair: dict[str, Any], *, db_client=None) -> None:
    """Upsert a contradiction pair into MongoDB ``contradictions`` collection."""
    from motor.motor_asyncio import AsyncIOMotorClient  # late import

    # Deterministic pair_id — order-independent hash
    a, b = sorted([pair["paper_a_id"], pair["paper_b_id"]])
    raw = f"{a}:{b}"
    pair_hash = hashlib.sha256(raw.encode()).hexdigest()[:20]
    doc = {
        "_id": pair_hash,
        **pair,
        "persisted_at": datetime.now(timezone.utc).isoformat(),
    }

    own_client = db_client is None
    client = db_client or AsyncIOMotorClient(_MONGO_URI)
    try:
        db = client[_DB_NAME]
        await db["contradictions"].replace_one(
            {"_id": pair_hash}, doc, upsert=True
        )
        logger.debug(
            "_persist_contradiction: upserted pair %r <-> %r (hash=%s)",
            pair["paper_a_id"], pair["paper_b_id"], pair_hash,
        )
    finally:
        if own_client:
            client.close()


def _short_title(title: str, n_words: int = _TITLE_WORDS) -> str:
    """Return the first n_words of title, ellipsis-appended if truncated."""
    words = title.strip().split()
    if not words:
        return "Unknown"
    if len(words) <= n_words:
        return " ".join(words)
    return " ".join(words[:n_words]) + "…"


# ---------------------------------------------------------------------------
# Public: extract_main_finding
# ---------------------------------------------------------------------------

def extract_main_finding(
    paper_id: str,
    abstract: Optional[str] = None,
    *,
    db_client=None,
) -> str:
    """
    Retrieve a paper's abstract and condense it into a single finding sentence.

    Parameters
    ----------
    paper_id : str
        The paper identifier used in MongoDB global_papers.
    abstract : str, optional
        If provided, skip the MongoDB lookup and use this text directly.
        Useful for testing and batch calls where the abstract is already known.
    db_client : optional
        Motor client (injected for testing).

    Returns
    -------
    str
        One-sentence string describing the main finding.
        Falls back to the first 200 characters of the abstract on Ollama failure.
    """
    # In-process cache hit
    if paper_id in _finding_cache:
        logger.debug("extract_main_finding: cache hit for paper_id=%r", paper_id)
        return _finding_cache[paper_id]

    # Resolve abstract if not provided
    if abstract is None:
        try:
            doc = asyncio.run(_fetch_paper_doc(paper_id, db_client=db_client))
        except RuntimeError:
            # Already inside a running event loop
            doc = {}
        abstract = doc.get("abstract") or doc.get("summary") or ""

    if not abstract:
        logger.warning(
            "extract_main_finding: no abstract for paper_id=%r — returning empty.", paper_id
        )
        _finding_cache[paper_id] = ""
        return ""

    # Build Ollama prompt
    prompt = (
        "Read the following scientific abstract and write exactly ONE sentence "
        "describing its main finding. Be specific. Do not include background, "
        "methods, or conclusions — only the primary result. "
        "Return only the single sentence, no preamble.\n\n"
        f"Abstract:\n{abstract[:2000]}"
    )

    try:
        finding = _call_ollama(prompt)
        # Clean: strip leading numbering, quotes, or 'Finding:' prefixes
        import re
        finding = re.sub(r"^[\d]+[.)]\s*", "", finding).strip()
        finding = finding.strip('"').strip("'").strip()
        # If model returned multiple sentences, take only the first
        sentences = re.split(r"(?<=[.!?])\s+", finding)
        finding = sentences[0].strip() if sentences else finding
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "extract_main_finding: Ollama failed for paper_id=%r (%s) "
            "— falling back to abstract truncation.",
            paper_id, exc,
        )
        finding = abstract[:200].strip()
        if not finding.endswith("."):
            finding = finding.rstrip() + "…"

    logger.info(
        "extract_main_finding: paper_id=%r -> %r", paper_id, finding[:80]
    )
    _finding_cache[paper_id] = finding
    return finding


# ---------------------------------------------------------------------------
# Public: check_contradictions
# ---------------------------------------------------------------------------

def check_contradictions(
    topic_id: str,
    *,
    paper_ids: Optional[list[str]] = None,
    db_client=None,
) -> list[dict[str, Any]]:
    """
    Pairwise-compare all topic papers' main findings and flag contradictions.

    Parameters
    ----------
    topic_id : str
        The topic identifier — used to fetch paper_ids from MongoDB if
        ``paper_ids`` is not provided.
    paper_ids : list[str], optional
        Override the list of papers (skips MongoDB lookup).  Useful for
        testing and CLI invocation.
    db_client : optional
        Motor client for dependency injection.

    Returns
    -------
    list of dict
        Each dict has:
        ``{"paper_a_id", "paper_b_id", "finding_a", "finding_b",
           "type": "contradiction", "confidence": float, "topic_id": str}``
        May be empty if no contradictions exceed the threshold.
    """
    # get_pipeline is imported at module level; reference it directly so the
    # module-level patch in tests takes effect.

    # --- Step 1: resolve paper_ids ----------------------------------------
    if paper_ids is None:
        try:
            paper_ids = asyncio.run(
                _fetch_topic_paper_ids(topic_id, db_client=db_client)
            )
        except RuntimeError:
            paper_ids = []

    if not paper_ids:
        logger.warning(
            "check_contradictions: no paper_ids for topic_id=%r — "
            "returning empty.", topic_id
        )
        return []

    logger.info(
        "check_contradictions: topic_id=%r — %d papers, %d pairs to check",
        topic_id, len(paper_ids),
        len(paper_ids) * (len(paper_ids) - 1) // 2,
    )

    # --- Step 2: extract main finding for each paper ----------------------
    findings: dict[str, str] = {}
    for pid in paper_ids:
        findings[pid] = extract_main_finding(pid, db_client=db_client)

    # --- Step 3: classify all pairs ---------------------------------------
    pipe = get_pipeline()
    flagged: list[dict[str, Any]] = []

    for pid_a, pid_b in itertools.combinations(paper_ids, 2):
        finding_a = findings.get(pid_a, "")
        finding_b = findings.get(pid_b, "")

        if not finding_a or not finding_b:
            logger.debug(
                "check_contradictions: skipping pair (%r, %r) — empty finding.",
                pid_a, pid_b,
            )
            continue

        # Classify: does finding_a support/refute/not-address finding_b?
        try:
            tokens_a = finding_a.split()
            passage = " ".join(tokens_a[:512])
            result = pipe(
                passage,
                candidate_labels=_LABELS,
                hypothesis_template="This text {} the claim: " + finding_b,
                multi_label=False,
            )
            top_label: str = result["labels"][0]
            top_score: float = float(result["scores"][0])
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "check_contradictions: DeBERTa failed for pair (%r, %r): %s",
                pid_a, pid_b, exc,
            )
            continue

        logger.debug(
            "check_contradictions: (%r, %r) -> label=%r score=%.3f",
            pid_a, pid_b, top_label, top_score,
        )

        if top_label == "refute" and top_score > _CONTRADICTION_THRESHOLD:
            pair: dict[str, Any] = {
                "paper_a_id": pid_a,
                "paper_b_id": pid_b,
                "finding_a": finding_a,
                "finding_b": finding_b,
                "type": "contradiction",
                "confidence": round(top_score, 4),
                "topic_id": topic_id,
            }
            flagged.append(pair)
            logger.info(
                "check_contradictions: CONTRADICTION flagged — "
                "(%r, %r) score=%.3f",
                pid_a, pid_b, top_score,
            )

            # Persist to MongoDB
            try:
                asyncio.run(_persist_contradiction(pair, db_client=db_client))
            except RuntimeError:
                pass  # already inside event loop — caller is responsible

    logger.info(
        "check_contradictions: topic_id=%r — %d contradiction(s) flagged "
        "out of %d pairs.",
        topic_id,
        len(flagged),
        len(paper_ids) * (len(paper_ids) - 1) // 2,
    )
    return flagged


# ---------------------------------------------------------------------------
# Public: generate_contradiction_map
# ---------------------------------------------------------------------------

def generate_contradiction_map(
    topic_id: str,
    *,
    paper_ids: Optional[list[str]] = None,
    paper_titles: Optional[dict[str, str]] = None,
    db_client=None,
) -> str:
    """
    Generate a Plotly heatmap showing pairwise agreement/contradiction between
    all topic papers.

    Parameters
    ----------
    topic_id : str
        Topic identifier; used to resolve paper_ids and titles from MongoDB if
        overrides are not provided.
    paper_ids : list[str], optional
        Override paper_ids (skips MongoDB lookup).
    paper_titles : dict[str, str], optional
        Override ``{paper_id: title}`` map (skips MongoDB lookup).
    db_client : optional
        Motor client for dependency injection.

    Returns
    -------
    str
        Self-contained Plotly figure JSON string.
        - Green  cells: support (papers agree)
        - Red    cells: refute (papers contradict)
        - Grey   cells: no_stance or diagonal
        Paste at https://chart-studio.plotly.com/create to render.
    """
    import plotly.graph_objects as go  # late import
    # get_pipeline is imported at module level above

    # --- Resolve paper_ids ------------------------------------------------
    if paper_ids is None:
        try:
            paper_ids = asyncio.run(
                _fetch_topic_paper_ids(topic_id, db_client=db_client)
            )
        except RuntimeError:
            paper_ids = []

    if not paper_ids:
        logger.warning(
            "generate_contradiction_map: no papers for topic_id=%r — "
            "returning empty figure.", topic_id
        )
        fig = go.Figure()
        fig.update_layout(
            title="No papers found for this topic",
            paper_bgcolor="#1a1a2e",
            font=dict(color="#ecf0f1"),
        )
        return fig.to_json()

    n = len(paper_ids)

    # --- Resolve short titles ---------------------------------------------
    if paper_titles is None:
        # Attempt MongoDB lookup; gracefully fall back to paper_id on failure
        resolved: dict[str, str] = {}
        try:
            async def _fetch_titles() -> dict[str, str]:
                from motor.motor_asyncio import AsyncIOMotorClient
                client = AsyncIOMotorClient(_MONGO_URI)
                try:
                    db = client[_DB_NAME]
                    out: dict[str, str] = {}
                    async for doc in db["global_papers"].find(
                        {"paper_id": {"$in": paper_ids}},
                        {"paper_id": 1, "title": 1},
                    ):
                        out[doc["paper_id"]] = doc.get("title", doc["paper_id"])
                    return out
                finally:
                    client.close()

            resolved = asyncio.run(_fetch_titles())
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "generate_contradiction_map: title fetch failed (%s) — "
                "using paper_ids as labels.", exc
            )
        paper_titles = resolved

    labels: list[str] = [
        _short_title(paper_titles.get(pid, pid)) for pid in paper_ids
    ]

    # --- Classify all pairs -----------------------------------------------
    pipe = get_pipeline()
    findings: dict[str, str] = {}
    for pid in paper_ids:
        findings[pid] = extract_main_finding(pid, db_client=db_client)

    # score_matrix[i][j]: +1 support, -1 refute, 0 no_stance / diagonal
    score_matrix: list[list[float]] = [
        [0.0] * n for _ in range(n)
    ]
    # raw label matrix for hover text
    label_matrix: list[list[str]] = [
        ["—"] * n for _ in range(n)
    ]
    # confidence matrix for annotations
    conf_matrix: list[list[float]] = [
        [0.0] * n for _ in range(n)
    ]

    label_to_score = {"support": 1.0, "refute": -1.0, "no_stance": 0.0}

    for i, pid_a in enumerate(paper_ids):
        for j, pid_b in enumerate(paper_ids):
            if i == j:
                score_matrix[i][j] = 0.0
                label_matrix[i][j] = "same"
                conf_matrix[i][j] = 0.0
                continue
            if j < i:
                # Mirror: use symmetric result
                score_matrix[i][j] = score_matrix[j][i]
                label_matrix[i][j] = label_matrix[j][i]
                conf_matrix[i][j] = conf_matrix[j][i]
                continue

            fa = findings.get(pid_a, "")
            fb = findings.get(pid_b, "")
            if not fa or not fb:
                continue

            try:
                result = pipe(
                    " ".join(fa.split()[:512]),
                    candidate_labels=_LABELS,
                    hypothesis_template="This text {} the claim: " + fb,
                    multi_label=False,
                )
                lbl = result["labels"][0]
                sc = float(result["scores"][0])
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "generate_contradiction_map: DeBERTa failed (%s)", exc
                )
                lbl, sc = "no_stance", 0.0

            score_matrix[i][j] = label_to_score.get(lbl, 0.0)
            label_matrix[i][j] = lbl
            conf_matrix[i][j] = round(sc, 3)

    # --- Build Plotly heatmap ---------------------------------------------
    # Colour scale: -1 (red) → 0 (grey) → +1 (green)
    colorscale = [
        [0.0,  "#c0392b"],   # -1  strong red (contradict)
        [0.4,  "#c0392b"],   # -0.2 red zone
        [0.5,  "#7f8c8d"],   # 0   grey (no_stance / diagonal)
        [0.6,  "#27ae60"],   # +0.2 green zone
        [1.0,  "#27ae60"],   # +1  strong green (support)
    ]

    # Hover text: "Label (conf)"
    hover_text: list[list[str]] = []
    annotation_text: list[list[str]] = []
    for i in range(n):
        row_hover: list[str] = []
        row_ann: list[str] = []
        for j in range(n):
            if i == j:
                row_hover.append(labels[i])
                row_ann.append("")
            else:
                lbl = label_matrix[i][j]
                cf = conf_matrix[i][j]
                row_hover.append(f"{lbl} ({cf:.2f})")
                row_ann.append(f"{cf:.2f}")
        hover_text.append(row_hover)
        annotation_text.append(row_ann)

    fig = go.Figure(
        data=go.Heatmap(
            z=score_matrix,
            x=labels,
            y=labels,
            colorscale=colorscale,
            zmin=-1,
            zmax=1,
            text=annotation_text,
            texttemplate="%{text}",
            hovertext=hover_text,
            hovertemplate=(
                "<b>%{y}</b> vs <b>%{x}</b><br>"
                "%{hovertext}<extra></extra>"
            ),
            showscale=True,
            colorbar=dict(
                title=dict(
                    text="Stance",
                    font=dict(color="#ecf0f1", size=12),
                ),
                tickvals=[-1, 0, 1],
                ticktext=["Contradict", "Neutral", "Support"],
                tickfont=dict(color="#ecf0f1", size=11),
                bgcolor="#2c3e50",
                bordercolor="#34495e",
            ),
            xgap=2,
            ygap=2,
        )
    )

    n_contradictions = sum(
        1 for i in range(n) for j in range(i + 1, n)
        if label_matrix[i][j] == "refute"
        and conf_matrix[i][j] > _CONTRADICTION_THRESHOLD
    )
    subtitle = (
        f"{n_contradictions} contradiction(s) detected above {_CONTRADICTION_THRESHOLD:.0%} threshold"
        if n_contradictions > 0
        else "No contradictions detected — all papers agree or are neutral"
    )

    fig.update_layout(
        title=dict(
            text=(
                f"<b>Contradiction Map — {topic_id}</b><br>"
                f"<span style='font-size:13px;color:#95a5a6'>{subtitle}</span>"
            ),
            font=dict(size=17, color="#ecf0f1"),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title="Paper",
            title_font=dict(size=13, color="#bdc3c7"),
            tickfont=dict(size=10, color="#bdc3c7"),
            tickangle=-35,
            gridcolor="#2c3e50",
            side="bottom",
        ),
        yaxis=dict(
            title="Paper",
            title_font=dict(size=13, color="#bdc3c7"),
            tickfont=dict(size=10, color="#bdc3c7"),
            gridcolor="#2c3e50",
            autorange="reversed",   # top-left = first paper
        ),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(family="Inter, sans-serif", color="#ecf0f1"),
        margin=dict(t=100, b=120, l=120, r=40),
        height=max(400, 80 * n),
        width=max(500, 90 * n),
    )

    logger.info(
        "generate_contradiction_map: topic_id=%r — %dx%d matrix, "
        "%d contradiction(s)",
        topic_id, n, n, n_contradictions,
    )
    return fig.to_json()
