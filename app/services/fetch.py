"""
app/services/fetch.py
---------------------
Async paper-fetching service for ClaimGraph.

Provides:
  - fetch_semantic_scholar  – Semantic Scholar Graph API
  - fetch_openalex          – OpenAlex /works
  - fetch_arxiv             – arXiv Atom/XML API
  - fetch_crossref          – CrossRef /works
  - check_global_cache      – MongoDB dedup check before download
  - filter_papers           – dedup + filter + rank pipeline
  - fetch_papers            – top-level orchestrator (runs all 4 concurrently)
  - fetch_all               – public alias for fetch_papers
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import xml.etree.ElementTree as ET
from typing import Any, Optional

import httpx
from Levenshtein import ratio as lev_ratio
from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias for a normalised paper record
# ---------------------------------------------------------------------------
Paper = dict[str, Any]

# ---------------------------------------------------------------------------
# Constants / env
# ---------------------------------------------------------------------------
_SEMANTIC_SCHOLAR_KEY: str = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
_OPENALEX_EMAIL: str = os.getenv("OPENALEX_EMAIL", "")
_MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017/claimgraph")
_DB_NAME: str = _MONGO_URI.rstrip("/").split("/")[-1]

_SS_BASE = "https://api.semanticscholar.org/graph/v1"
_OA_BASE = "https://api.openalex.org"
_ARXIV_BASE = "https://export.arxiv.org/api"
_CR_BASE = "https://api.crossref.org"

_ARXIV_NS = "http://www.w3.org/2005/Atom"

VENUE_WHITELIST_DEFAULT: list[str] = []  # empty → accept all


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

async def _with_retry(
    fn,
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    **kwargs,
) -> Any:
    """
    Call *fn(*args, **kwargs)* up to *max_retries* times with exponential
    back-off + jitter on 429 / 5xx / network errors.  Returns the result on
    success, or an empty list if all retries are exhausted.
    """
    for attempt in range(max_retries):
        try:
            return await fn(*args, **kwargs)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in (429, 500, 502, 503, 504):
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning(
                    "HTTP %s from %s – retrying in %.1fs (attempt %d/%d)",
                    status,
                    exc.request.url,
                    delay,
                    attempt + 1,
                    max_retries,
                )
                await asyncio.sleep(delay)
            else:
                logger.error("Non-retryable HTTP %s: %s", status, exc.request.url)
                return []
        except httpx.RequestError as exc:
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            logger.warning(
                "Request error (%s) – retrying in %.1fs (attempt %d/%d)",
                exc,
                delay,
                attempt + 1,
                max_retries,
            )
            await asyncio.sleep(delay)

    logger.error("All %d retries exhausted", max_retries)
    return []


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _norm(value: Any, default: Any = None) -> Any:
    """Return *value* if truthy, else *default*."""
    return value if value else default


def _paper(
    *,
    title: str,
    abstract: Optional[str],
    doi: Optional[str],
    venue: Optional[str],
    citation_count: int,
    pdf_url: Optional[str],
    source: str,
) -> Paper:
    return {
        "title": title,
        "abstract": abstract,
        "doi": doi,
        "venue": venue,
        "citation_count": citation_count,
        "pdf_url": pdf_url,
        "source": source,
    }


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------

async def _do_fetch_semantic_scholar(topic: str) -> list[Paper]:
    headers: dict[str, str] = {}
    if _SEMANTIC_SCHOLAR_KEY:
        headers["x-api-key"] = _SEMANTIC_SCHOLAR_KEY

    params = {
        "query": topic,
        "limit": 50,
        "fields": "title,abstract,externalIds,venue,citationCount,openAccessPdf",
    }

    async with httpx.AsyncClient(timeout=20) as client:
        resp = client.get(f"{_SS_BASE}/paper/search", params=params, headers=headers)
        resp = await resp
        resp.raise_for_status()
        data = resp.json()

    papers: list[Paper] = []
    for item in data.get("data", []):
        doi = _norm(
            (item.get("externalIds") or {}).get("DOI"), None
        )
        pdf = _norm(
            (item.get("openAccessPdf") or {}).get("url"), None
        )
        papers.append(
            _paper(
                title=_norm(item.get("title"), ""),
                abstract=_norm(item.get("abstract")),
                doi=doi,
                venue=_norm(item.get("venue")),
                citation_count=_norm(item.get("citationCount"), 0),
                pdf_url=pdf,
                source="semantic_scholar",
            )
        )
    return papers


async def fetch_semantic_scholar(topic: str) -> list[Paper]:
    """Fetch papers from Semantic Scholar Graph API."""
    return await _with_retry(_do_fetch_semantic_scholar, topic)


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------

async def _do_fetch_openalex(topic: str) -> list[Paper]:
    params: dict[str, Any] = {
        "search": topic,
        "per-page": 50,
        "select": "title,abstract_inverted_index,doi,primary_location,cited_by_count,open_access",
    }
    if _OPENALEX_EMAIL:
        params["mailto"] = _OPENALEX_EMAIL

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(f"{_OA_BASE}/works", params=params)
        resp.raise_for_status()
        data = resp.json()

    papers: list[Paper] = []
    for item in data.get("results", []):
        # Reconstruct abstract from inverted index
        abstract: Optional[str] = None
        inv = item.get("abstract_inverted_index")
        if inv:
            positions: list[tuple[int, str]] = []
            for word, idxs in inv.items():
                for pos in idxs:
                    positions.append((pos, word))
            positions.sort()
            abstract = " ".join(w for _, w in positions)

        raw_doi: Optional[str] = item.get("doi")
        doi = raw_doi.replace("https://doi.org/", "") if raw_doi else None

        loc = item.get("primary_location") or {}
        venue = _norm((loc.get("source") or {}).get("display_name"))
        pdf_url = _norm(
            (item.get("open_access") or {}).get("oa_url")
        )

        papers.append(
            _paper(
                title=_norm(item.get("title"), ""),
                abstract=abstract,
                doi=doi,
                venue=venue,
                citation_count=_norm(item.get("cited_by_count"), 0),
                pdf_url=pdf_url,
                source="openalex",
            )
        )
    return papers


async def fetch_openalex(topic: str) -> list[Paper]:
    """Fetch papers from OpenAlex /works."""
    return await _with_retry(_do_fetch_openalex, topic)


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------

def _arxiv_text(entry: ET.Element, tag: str) -> Optional[str]:
    el = entry.find(f"{{{_ARXIV_NS}}}{tag}")
    return el.text.strip() if el is not None and el.text else None


async def _do_fetch_arxiv(topic: str) -> list[Paper]:
    params = {
        "search_query": f"all:{topic}",
        "start": 0,
        "max_results": 50,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{_ARXIV_BASE}/query", params=params)
        resp.raise_for_status()
        xml_text = resp.text

    root = ET.fromstring(xml_text)
    papers: list[Paper] = []

    for entry in root.findall(f"{{{_ARXIV_NS}}}entry"):
        title = _arxiv_text(entry, "title")
        if not title:
            continue
        abstract = _arxiv_text(entry, "summary")

        # arXiv DOI link (may be absent); fall back to arXiv ID
        doi: Optional[str] = None
        doi_el = entry.find(f"{{{_ARXIV_NS}}}doi")
        if doi_el is not None and doi_el.text:
            doi = doi_el.text.strip()
        else:
            # Derive a DOI-style identifier from the arXiv ID
            arxiv_id_el = entry.find(f"{{{_ARXIV_NS}}}id")
            if arxiv_id_el is not None and arxiv_id_el.text:
                raw_id = arxiv_id_el.text.strip()
                # e.g. http://arxiv.org/abs/2301.00001v1 → 2301.00001
                arx_id = raw_id.split("/abs/")[-1].split("v")[0]
                doi = f"10.48550/arXiv.{arx_id}"

        # PDF link
        pdf_url: Optional[str] = None
        for link_el in entry.findall(f"{{{_ARXIV_NS}}}link"):
            if link_el.get("title") == "pdf":
                pdf_url = link_el.get("href")
                break

        # Journal ref as venue
        journal_ref = entry.find(f"{{{_ARXIV_NS}}}journal_ref")
        venue = journal_ref.text.strip() if journal_ref is not None and journal_ref.text else None

        papers.append(
            _paper(
                title=title,
                abstract=abstract,
                doi=doi,
                venue=venue,
                citation_count=0,  # arXiv API does not return citation counts
                pdf_url=pdf_url,
                source="arxiv",
            )
        )
    return papers


async def fetch_arxiv(topic: str) -> list[Paper]:
    """Fetch papers from arXiv Atom API."""
    return await _with_retry(_do_fetch_arxiv, topic)


# ---------------------------------------------------------------------------
# CrossRef
# ---------------------------------------------------------------------------

async def _do_fetch_crossref(topic: str) -> list[Paper]:
    params: dict[str, Any] = {
        "query": topic,
        "rows": 50,
        "select": "title,abstract,DOI,container-title,is-referenced-by-count,link",
    }

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(f"{_CR_BASE}/works", params=params)
        resp.raise_for_status()
        data = resp.json()

    papers: list[Paper] = []
    for item in (data.get("message") or {}).get("items", []):
        title_list = item.get("title") or []
        title = title_list[0] if title_list else None
        if not title:
            continue

        doi = _norm(item.get("DOI"))

        venue_list = item.get("container-title") or []
        venue = venue_list[0] if venue_list else None

        # Best open-access PDF link
        pdf_url: Optional[str] = None
        for link in item.get("link") or []:
            if link.get("content-type") == "application/pdf":
                pdf_url = link.get("URL")
                break

        papers.append(
            _paper(
                title=title,
                abstract=_norm(item.get("abstract")),
                doi=doi,
                venue=venue,
                citation_count=_norm(item.get("is-referenced-by-count"), 0),
                pdf_url=pdf_url,
                source="crossref",
            )
        )
    return papers


async def fetch_crossref(topic: str) -> list[Paper]:
    """Fetch papers from CrossRef /works."""
    return await _with_retry(_do_fetch_crossref, topic)


# ---------------------------------------------------------------------------
# MongoDB global cache check
# ---------------------------------------------------------------------------

async def check_global_cache(
    doi: str,
    *,
    db_client: Optional[AsyncIOMotorClient] = None,
) -> Optional[str]:
    """
    Check whether a paper with the given *doi* already exists in the MongoDB
    ``global_papers`` collection.

    Parameters
    ----------
    doi:
        The DOI string to look up (e.g. ``"10.1038/nature12373"``).
    db_client:
        An optional pre-existing ``motor`` client.  If *None*, a new client
        is created from ``MONGO_URI`` env var for this call.

    Returns
    -------
    str
        The ``_id`` of the cached document as a string, if found.
    None
        If the paper is not in the cache.
    """
    _own_client = db_client is None
    client = db_client or AsyncIOMotorClient(_MONGO_URI)
    try:
        db = client[_DB_NAME]
        doc = await db["global_papers"].find_one({"doi": doi}, {"_id": 1})
        return str(doc["_id"]) if doc else None
    finally:
        if _own_client:
            client.close()


# ---------------------------------------------------------------------------
# Filter + dedup pipeline
# ---------------------------------------------------------------------------

def filter_papers(
    papers: list[Paper],
    min_citations: int = 5,
    venue_whitelist: Optional[list[str]] = None,
) -> list[Paper]:
    """
    Deduplicate and filter *papers*.

    Steps
    -----
    1. Dedupe by DOI (first occurrence wins).
    2. Dedupe by title Levenshtein similarity > 0.85 (first occurrence wins).
    3. Drop papers that have no DOI.
    4. Apply optional *venue_whitelist* (case-insensitive substring match).
    5. Drop papers with ``citation_count < min_citations``.
    6. Sort descending by ``citation_count``.
    7. Return top 10.

    Parameters
    ----------
    papers:
        Raw list of paper dicts (as returned by the fetch_* functions).
    min_citations:
        Minimum citation count to keep a paper.
    venue_whitelist:
        If non-empty, only papers whose ``venue`` contains at least one
        whitelisted string (case-insensitive) are kept.  Pass ``None`` or
        ``[]`` to accept all venues.

    Returns
    -------
    list[Paper]
        Filtered, deduplicated, sorted top-10 list.
    """
    # --- 1. Dedupe by DOI ------------------------------------------------
    seen_dois: set[str] = set()
    doi_deduped: list[Paper] = []
    for p in papers:
        doi = p.get("doi")
        if doi:
            if doi in seen_dois:
                continue
            seen_dois.add(doi)
        doi_deduped.append(p)

    # --- 2. Dedupe by title similarity -----------------------------------
    title_deduped: list[Paper] = []
    seen_titles: list[str] = []
    for p in doi_deduped:
        title = (p.get("title") or "").strip().lower()
        if not title:
            title_deduped.append(p)
            continue
        is_dup = any(lev_ratio(title, t) > 0.85 for t in seen_titles)
        if not is_dup:
            seen_titles.append(title)
            title_deduped.append(p)

    # --- 3. Drop papers with no DOI -------------------------------------
    has_doi = [p for p in title_deduped if p.get("doi")]

    # --- 4. Venue whitelist ---------------------------------------------
    wl = [v.lower() for v in (venue_whitelist or [])]
    if wl:
        def _venue_ok(p: Paper) -> bool:
            venue = (p.get("venue") or "").lower()
            return any(w in venue for w in wl)
        venue_filtered = [p for p in has_doi if _venue_ok(p)]
    else:
        venue_filtered = has_doi

    # --- 5. Min citations -----------------------------------------------
    citation_filtered = [
        p for p in venue_filtered if (p.get("citation_count") or 0) >= min_citations
    ]

    # --- 6 & 7. Sort + top 10 -------------------------------------------
    citation_filtered.sort(key=lambda p: p.get("citation_count") or 0, reverse=True)
    return citation_filtered[:10]


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

async def fetch_papers(
    topic: str,
    min_citations: int = 5,
    venue_whitelist: Optional[list[str]] = None,
    *,
    db_client: Optional[AsyncIOMotorClient] = None,
) -> list[Paper]:
    """
    Fetch papers on *topic* from all four sources concurrently, check the
    MongoDB global cache to avoid re-downloading already-known papers, then
    run the dedup/filter pipeline.

    Parameters
    ----------
    topic:
        Free-text topic / query string.
    min_citations:
        Passed through to :func:`filter_papers`.
    venue_whitelist:
        Passed through to :func:`filter_papers`.
    db_client:
        Optional shared motor client; forwarded to :func:`check_global_cache`.

    Returns
    -------
    list[Paper]
        Filtered, deduplicated list of up to 10 papers.
    """
    # Run all four fetchers concurrently
    results = await asyncio.gather(
        fetch_semantic_scholar(topic),
        fetch_openalex(topic),
        fetch_arxiv(topic),
        fetch_crossref(topic),
        return_exceptions=True,
    )

    all_papers: list[Paper] = []
    for result in results:
        if isinstance(result, Exception):
            logger.error("Fetcher raised unexpectedly: %s", result)
        else:
            all_papers.extend(result)

    # Check global cache — skip papers already stored
    cache_checked: list[Paper] = []
    for paper in all_papers:
        doi = paper.get("doi")
        if doi:
            cached_id = await check_global_cache(doi, db_client=db_client)
            if cached_id:
                logger.debug("Cache hit for DOI %s (paper_id=%s) – skipping", doi, cached_id)
                continue
        cache_checked.append(paper)

    return filter_papers(
        cache_checked,
        min_citations=min_citations,
        venue_whitelist=venue_whitelist,
    )


# ---------------------------------------------------------------------------
# Public alias – fetch_all is the canonical entry-point exposed to callers
# ---------------------------------------------------------------------------
#: Alias of :func:`fetch_papers` for convenience.
fetch_all = fetch_papers
