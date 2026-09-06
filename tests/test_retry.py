"""
tests/test_retry.py
--------------------
Unit tests for app/services/retry.py (Ambiguity retry loop).

Tests:
1. test_verify_claim_no_papers: topic doc empty/missing -> returns insufficient_evidence
2. test_verify_claim_success: mock qdrant/embed/geometry -> returns GeometryResult
3. test_handle_ambiguous_immediate_success: not ambiguous at attempt 0 -> returns verdict directly
4. test_handle_ambiguous_retry_and_succeed: ambiguous on attempt 0, non-ambiguous on attempt 1 -> returns resolved verdict
5. test_handle_ambiguous_exhaust_retries: ambiguous on attempt 0, 1, and 2 -> returns insufficient_evidence with reason=max_retries_reached after exactly 2 retries
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.retry import verify_claim, handle_ambiguous, INSUFFICIENT_EVIDENCE


def _make_mock_db(doc=None):
    mock_topics = MagicMock()
    mock_topics.find_one = AsyncMock(return_value=doc)
    mock_db = MagicMock()
    # Handle db[_DB_NAME]["topics"] indexing
    mock_db.__getitem__.return_value.__getitem__.return_value = mock_topics
    return mock_db


@pytest.mark.asyncio
async def test_verify_claim_no_papers():
    """When topic has no papers, verify_claim returns INSUFFICIENT_EVIDENCE."""
    mock_db = _make_mock_db(doc=None)

    result = await verify_claim("claim text", "topic_empty", db_client=mock_db)

    assert result["verdict"] == "insufficient_evidence"
    assert result["reason"] == "no_papers_ingested"


@pytest.mark.asyncio
async def test_verify_claim_success():
    """verify_claim retrieves vectors and runs geometry cleanly."""
    doc = {
        "_id": "topic_001",
        "stance_buckets": {
            "support": ["paper1", "paper2"],
            "refute": ["paper3"],
            "no_stance": []
        }
    }
    mock_db = _make_mock_db(doc=doc)

    expected_verdict = {
        "distances": {"support": 0.5, "refute": 2.1, "no_stance": float("inf")},
        "winning_label": "support",
        "margin": 1.6,
        "is_ambiguous": False,
        "low_sample_warning": False,
        "pca_n_components": 2,
    }

    mock_vectors = {
        "support": np.ones((2, 768), dtype=np.float32),
        "refute": np.zeros((1, 768), dtype=np.float32),
        "no_stance": np.zeros((0, 768), dtype=np.float32),
    }

    with patch("app.services.retry._fetch_bucket_vectors", return_value=mock_vectors), \
         patch("app.services.retry._embed_claim", return_value=np.ones(768, dtype=np.float32)), \
         patch("app.services.geometry.fit_geometry", return_value=expected_verdict):

        result = await verify_claim("claim text", "topic_001", db_client=mock_db)

        assert result == expected_verdict
        assert result["winning_label"] == "support"
        assert result["is_ambiguous"] is False


@pytest.mark.asyncio
async def test_handle_ambiguous_immediate_success():
    """If initial claim verification is not ambiguous, return immediately."""
    non_ambiguous_verdict = {
        "winning_label": "support",
        "margin": 1.5,
        "is_ambiguous": False,
        "low_sample_warning": False,
    }

    with patch("app.services.retry.verify_claim", new_callable=AsyncMock) as mock_verify:
        mock_verify.return_value = non_ambiguous_verdict

        result = await handle_ambiguous("claim", "topic_1", max_attempts=2)

        assert result == non_ambiguous_verdict
        assert mock_verify.call_count == 1


@pytest.mark.asyncio
async def test_handle_ambiguous_retry_and_succeed():
    """Ambiguous on attempt 0, succeeds (non-ambiguous) on attempt 1."""
    ambiguous_verdict = {
        "winning_label": "support",
        "margin": 0.1,
        "is_ambiguous": True,
        "low_sample_warning": False,
    }
    resolved_verdict = {
        "winning_label": "support",
        "margin": 0.8,
        "is_ambiguous": False,
        "low_sample_warning": False,
    }

    with patch("app.services.retry.verify_claim", new_callable=AsyncMock) as mock_verify, \
         patch("app.services.fetch.fetch_papers", new_callable=AsyncMock, return_value=[{"doi": "10.1/a", "pdf_url": "http://a.pdf"}]), \
         patch("app.services.fetch.filter_papers", return_value=[{"doi": "10.1/a", "pdf_url": "http://a.pdf"}]), \
         patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_http, \
         patch("app.services.store.ingest_paper", new_callable=AsyncMock, return_value=(3, ["c1", "c2", "c3"])), \
         patch("app.services.stance.classify_paper_set", new_callable=AsyncMock):

        mock_http.return_value.status_code = 200
        mock_http.return_value.content = b"%PDF-mock"
        mock_http.return_value.raise_for_status = MagicMock()

        mock_verify.side_effect = [ambiguous_verdict, resolved_verdict]

        result = await handle_ambiguous("my claim", "topic_1", max_attempts=2)

        assert result == resolved_verdict
        assert mock_verify.call_count == 2


@pytest.mark.asyncio
async def test_handle_ambiguous_exhaust_retries():
    """Ambiguous across attempts -> stops at attempt 2 and returns insufficient_evidence."""
    ambiguous_verdict = {
        "winning_label": "support",
        "margin": 0.05,
        "is_ambiguous": True,
        "low_sample_warning": False,
    }

    verify_calls = []

    async def mock_verify_impl(claim, topic_id, **kwargs):
        verify_calls.append((claim, topic_id))
        return ambiguous_verdict

    with patch("app.services.retry.verify_claim", side_effect=mock_verify_impl), \
         patch("app.services.fetch.fetch_papers", new_callable=AsyncMock, return_value=[]), \
         patch("app.services.fetch.filter_papers", return_value=[]), \
         patch("app.services.stance.classify_paper_set", new_callable=AsyncMock):

        result = await handle_ambiguous("ambiguous claim", "topic_ambig", max_attempts=2)

        # Confirm exactly 2 retries (3 calls total: attempt 0, attempt 1, attempt 2)
        assert len(verify_calls) == 3
        assert result["verdict"] == "insufficient_evidence"
        assert result["confidence"] == 0
        assert result["reason"] == "max_retries_reached"
