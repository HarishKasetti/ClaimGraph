"""
tests/test_fetch.py
-------------------
Pytest suite for app/services/fetch.py.

Uses:
  - respx   – httpx transport-level mocking (intercepts AsyncClient calls)
  - pytest-asyncio – async test support
  - unittest.mock  – AsyncMock for MongoDB motor calls
"""

from __future__ import annotations

import json
import textwrap
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import respx
from httpx import Response

from app.services.fetch import (
    check_global_cache,
    fetch_arxiv,
    fetch_crossref,
    fetch_openalex,
    fetch_semantic_scholar,
    filter_papers,
)

# ---------------------------------------------------------------------------
# Helpers – canned API payloads
# ---------------------------------------------------------------------------

_SS_PAYLOAD: dict[str, Any] = {
    "data": [
        {
            "title": "Attention Is All You Need",
            "abstract": "Transformer architecture paper.",
            "externalIds": {"DOI": "10.5555/3295222.3295349"},
            "venue": "NeurIPS",
            "citationCount": 80000,
            "openAccessPdf": {"url": "https://arxiv.org/pdf/1706.03762"},
        }
    ]
}

_OA_PAYLOAD: dict[str, Any] = {
    "results": [
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "abstract_inverted_index": {"BERT": [0], "model": [1]},
            "doi": "https://doi.org/10.18653/v1/N19-1423",
            "primary_location": {
                "source": {"display_name": "NAACL"}
            },
            "cited_by_count": 50000,
            "open_access": {"oa_url": "https://aclanthology.org/N19-1423.pdf"},
        }
    ]
}

_ARXIV_XML: str = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2301.00001v1</id>
        <title>A Survey on Large Language Models</title>
        <summary>This paper surveys LLMs.</summary>
        <link title="pdf" href="https://arxiv.org/pdf/2301.00001"/>
        <journal_ref>arXiv 2023</journal_ref>
      </entry>
    </feed>
""")

_CR_PAYLOAD: dict[str, Any] = {
    "message": {
        "items": [
            {
                "title": ["Deep Residual Learning for Image Recognition"],
                "abstract": "ResNet paper.",
                "DOI": "10.1109/CVPR.2016.90",
                "container-title": ["CVPR"],
                "is-referenced-by-count": 120000,
                "link": [{"content-type": "application/pdf", "URL": "https://example.com/resnet.pdf"}],
            }
        ]
    }
}


# ---------------------------------------------------------------------------
# fetch_semantic_scholar
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@respx.mock
async def test_fetch_semantic_scholar_success():
    """Mocked 200 response is normalised correctly."""
    respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
        return_value=Response(200, json=_SS_PAYLOAD)
    )

    papers = await fetch_semantic_scholar("transformers")

    assert len(papers) == 1
    p = papers[0]
    assert p["title"] == "Attention Is All You Need"
    assert p["doi"] == "10.5555/3295222.3295349"
    assert p["venue"] == "NeurIPS"
    assert p["citation_count"] == 80000
    assert p["pdf_url"] == "https://arxiv.org/pdf/1706.03762"
    assert p["source"] == "semantic_scholar"


# ---------------------------------------------------------------------------
# fetch_openalex
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@respx.mock
async def test_fetch_openalex_success():
    """OpenAlex abstract reconstruction and DOI stripping work correctly."""
    respx.get("https://api.openalex.org/works").mock(
        return_value=Response(200, json=_OA_PAYLOAD)
    )

    papers = await fetch_openalex("BERT")

    assert len(papers) == 1
    p = papers[0]
    assert "BERT" in p["title"]
    assert p["doi"] == "10.18653/v1/N19-1423"
    assert p["venue"] == "NAACL"
    assert p["citation_count"] == 50000
    assert p["source"] == "openalex"
    # Abstract reconstructed from inverted index
    assert p["abstract"] is not None
    assert "BERT" in p["abstract"]


# ---------------------------------------------------------------------------
# fetch_arxiv
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@respx.mock
async def test_fetch_arxiv_success():
    """arXiv XML is parsed and a synthetic DOI is derived from the arXiv ID."""
    respx.get("https://export.arxiv.org/api/query").mock(
        return_value=Response(200, text=_ARXIV_XML)
    )

    papers = await fetch_arxiv("large language models")

    assert len(papers) == 1
    p = papers[0]
    assert p["title"] == "A Survey on Large Language Models"
    assert p["doi"] == "10.48550/arXiv.2301.00001"
    assert p["pdf_url"] == "https://arxiv.org/pdf/2301.00001"
    assert p["source"] == "arxiv"
    assert p["venue"] == "arXiv 2023"


# ---------------------------------------------------------------------------
# fetch_crossref
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@respx.mock
async def test_fetch_crossref_success():
    """CrossRef items are normalised including the PDF link extraction."""
    respx.get("https://api.crossref.org/works").mock(
        return_value=Response(200, json=_CR_PAYLOAD)
    )

    papers = await fetch_crossref("ResNet")

    assert len(papers) == 1
    p = papers[0]
    assert "Residual Learning" in p["title"]
    assert p["doi"] == "10.1109/CVPR.2016.90"
    assert p["venue"] == "CVPR"
    assert p["citation_count"] == 120000
    assert p["pdf_url"] == "https://example.com/resnet.pdf"
    assert p["source"] == "crossref"


# ---------------------------------------------------------------------------
# Retry on 429
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@respx.mock
async def test_retry_on_rate_limit(monkeypatch):
    """A 429 response is retried and a subsequent 200 succeeds."""
    # Patch sleep so the test doesn't actually wait
    monkeypatch.setattr("app.services.fetch.asyncio.sleep", AsyncMock())

    route = respx.get("https://api.semanticscholar.org/graph/v1/paper/search")
    route.side_effect = [
        Response(429, json={"message": "Too Many Requests"}),
        Response(200, json=_SS_PAYLOAD),
    ]

    papers = await fetch_semantic_scholar("transformers")
    assert len(papers) == 1
    assert route.call_count == 2


# ---------------------------------------------------------------------------
# filter_papers – dedup by DOI
# ---------------------------------------------------------------------------

def _make_paper(title: str, doi: str, citation_count: int = 10, venue: str = "Nature") -> dict:
    return {
        "title": title,
        "abstract": "Test abstract.",
        "doi": doi,
        "venue": venue,
        "citation_count": citation_count,
        "pdf_url": None,
        "source": "test",
    }


def test_filter_dedup_by_doi():
    """Two papers with the same DOI: only the first is kept."""
    papers = [
        _make_paper("Paper Alpha", "10.1000/xyz123"),
        _make_paper("Paper Alpha (duplicate)", "10.1000/xyz123"),
    ]
    result = filter_papers(papers, min_citations=0)
    assert len(result) == 1
    assert result[0]["title"] == "Paper Alpha"


# ---------------------------------------------------------------------------
# filter_papers – dedup by title similarity
# ---------------------------------------------------------------------------

def test_filter_dedup_by_title_similarity():
    """Two papers with very similar titles (ratio > 0.85) and different DOIs: only the first kept."""
    papers = [
        _make_paper("Attention Is All You Need", "10.1000/aaa"),
        _make_paper("Attention Is All You Need.", "10.1000/bbb"),  # almost identical title
    ]
    result = filter_papers(papers, min_citations=0)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# filter_papers – drops paper without DOI
# ---------------------------------------------------------------------------

def test_filter_drops_no_doi():
    """Papers without a DOI are removed from the result."""
    papers = [
        {
            "title": "No DOI Paper",
            "abstract": None,
            "doi": None,
            "venue": "Nature",
            "citation_count": 100,
            "pdf_url": None,
            "source": "test",
        },
        _make_paper("Real Paper", "10.1000/real"),
    ]
    result = filter_papers(papers, min_citations=0)
    titles = [p["title"] for p in result]
    assert "No DOI Paper" not in titles
    assert "Real Paper" in titles


# ---------------------------------------------------------------------------
# filter_papers – venue whitelist
# ---------------------------------------------------------------------------

def test_filter_venue_whitelist():
    """Papers not matching the venue whitelist are excluded."""
    papers = [
        _make_paper("Nature Paper", "10.1000/nat", venue="Nature"),
        _make_paper("Science Paper", "10.1000/sci", venue="Science"),
        _make_paper("Blog Post", "10.1000/blg", venue="Medium Blog"),
    ]
    result = filter_papers(papers, min_citations=0, venue_whitelist=["Nature", "Science"])
    venues = {p["venue"] for p in result}
    assert "Medium Blog" not in venues
    assert "Nature" in venues
    assert "Science" in venues


# ---------------------------------------------------------------------------
# filter_papers – min_citations
# ---------------------------------------------------------------------------

def test_filter_min_citations():
    """Papers below min_citations threshold are dropped."""
    papers = [
        _make_paper("High Citations", "10.1000/high", citation_count=100),
        _make_paper("Low Citations", "10.1000/low", citation_count=2),
    ]
    result = filter_papers(papers, min_citations=5)
    titles = [p["title"] for p in result]
    assert "High Citations" in titles
    assert "Low Citations" not in titles


# ---------------------------------------------------------------------------
# filter_papers – top 10 cap and sort order
# ---------------------------------------------------------------------------

def test_filter_top10_sorted():
    """Result is capped at 10 papers and sorted by citation_count descending."""
    # Use clearly distinct titles so they are NOT collapsed by title dedup
    distinct_titles = [
        "Quantum Entanglement in Neural Networks",
        "Graph-Based Claim Verification Methods",
        "Zero-Shot Learning with Language Priors",
        "Federated Learning for Medical Imaging",
        "Contrastive Self-Supervised Pretraining",
        "Diffusion Models Beat Generative Adversarial Networks",
        "Reinforcement Learning from Human Feedback",
        "Cross-Lingual Transfer in Multilingual Transformers",
        "Knowledge Graph Completion via Embedding",
        "Sparse Mixture of Experts Scaling Laws",
        "Vision Transformers for Dense Prediction",
        "Neural Architecture Search with Proximal Policy",
        "Long-Range Dependencies in Protein Folding",
        "Continual Learning Without Catastrophic Forgetting",
        "Causal Inference from Observational Data",
        "Semantic Segmentation Using Deformable Convolutions",
        "Energy-Based Models and Contrastive Divergence",
        "Adversarial Robustness via Certified Defenses",
        "Automatic Speech Recognition with Conformer",
    ]
    papers = [
        _make_paper(distinct_titles[i], f"10.1000/{i:04d}", citation_count=i + 1)
        for i in range(len(distinct_titles))
    ]
    result = filter_papers(papers, min_citations=0)
    assert len(result) == 10
    # Sorted descending by citation_count
    counts = [p["citation_count"] for p in result]
    assert counts == sorted(counts, reverse=True)
    # Top result has the highest citation count (19)
    assert result[0]["citation_count"] == len(distinct_titles)


# ---------------------------------------------------------------------------
# check_global_cache – cache hit
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_check_global_cache_hit():
    """When MongoDB returns a matching document, the _id string is returned."""
    from bson import ObjectId

    fake_id = ObjectId()
    mock_collection = MagicMock()
    mock_collection.find_one = AsyncMock(return_value={"_id": fake_id})

    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=mock_collection)

    mock_client = MagicMock()
    mock_client.__getitem__ = MagicMock(return_value=mock_db)
    mock_client.close = MagicMock()

    result = await check_global_cache("10.1000/test", db_client=mock_client)

    assert result == str(fake_id)
    mock_collection.find_one.assert_awaited_once_with({"doi": "10.1000/test"}, {"_id": 1})


# ---------------------------------------------------------------------------
# check_global_cache – cache miss
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_check_global_cache_miss():
    """When MongoDB returns None, check_global_cache returns None."""
    mock_collection = MagicMock()
    mock_collection.find_one = AsyncMock(return_value=None)

    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=mock_collection)

    mock_client = MagicMock()
    mock_client.__getitem__ = MagicMock(return_value=mock_db)
    mock_client.close = MagicMock()

    result = await check_global_cache("10.1000/nonexistent", db_client=mock_client)

    assert result is None
