"""
tests/cross_verify_fetch.py
----------------------------
Manual cross-verification script for app/services/fetch.py.

Checks:
  1. <=10 papers returned, every paper has a non-null DOI
  2. DOI-resolve spot-check (streaming GET to https://doi.org/<doi>)
  3. Two consecutive runs produce near-identical results (stability)
  4. Pipeline completes without crash when Semantic Scholar is unreachable
     (_do_fetch_semantic_scholar monkey-patched to raise ConnectError)

Run with:
  .venv\\Scripts\\python -m pytest tests/cross_verify_fetch.py -v --asyncio-mode=auto -s
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock as AM, MagicMock, patch

import httpx
import pytest

# Force UTF-8 output so emoji / arrow characters never cause cp1252 errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_db_client():
    """Return a Motor-like async client that always reports cache-miss."""
    mock_coll = MagicMock()
    mock_coll.find_one = AM(return_value=None)
    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=mock_coll)
    mock_client = MagicMock()
    mock_client.__getitem__ = MagicMock(return_value=mock_db)
    mock_client.close = MagicMock()
    return mock_client


async def _run_pipeline(topic: str = "intermittent fasting insulin sensitivity") -> list[dict]:
    from app.services.fetch import fetch_all
    return await fetch_all(topic, db_client=_mock_db_client())


# ---------------------------------------------------------------------------
# Check 1 - count & non-null DOI
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_check1_count_and_doi():
    """CROSS-VERIFY 1: pipeline returns <=10 papers, every paper has a DOI."""
    papers = await _run_pipeline()

    print(f"\n[CHECK 1] Got {len(papers)} papers")
    assert len(papers) <= 10, f"Expected <=10 papers, got {len(papers)}"
    assert len(papers) > 0,   "Expected at least one paper"

    for i, p in enumerate(papers, 1):
        doi = p.get("doi")
        title = (p["title"] or "")[:70]
        print(f"  {i:2d}. {title!r:74s} | doi={doi} | cit={p['citation_count']}")
        assert doi, f"Paper #{i} has null DOI: {p['title']!r}"

    print("[CHECK 1] PASSED")


# ---------------------------------------------------------------------------
# Check 2 - DOI resolves on doi.org
# ---------------------------------------------------------------------------

_SPOT_CHECK_COUNT = 3


@pytest.mark.asyncio
async def test_check2_doi_resolves():
    """
    CROSS-VERIFY 2: top 3 DOIs return non-404/410 when streaming GET via doi.org.

    Many publisher endpoints return 403/405 for bots but still confirm the DOI
    exists; only 404 and 410 mean the DOI is genuinely unresolvable.
    """
    papers = await _run_pipeline()
    to_check = [p for p in papers if p.get("doi")][:_SPOT_CHECK_COUNT]

    print(f"\n[CHECK 2] Resolving {len(to_check)} DOIs via doi.org ...")

    async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
        for p in to_check:
            doi = p["doi"]
            url = f"https://doi.org/{doi}"
            try:
                async with client.stream("GET", url) as resp:
                    async for _ in resp.aiter_bytes(1):
                        break
                final_url = str(resp.url)
                print(
                    f"  DOI {doi}\n"
                    f"       -> {final_url[:80]}\n"
                    f"       status={resp.status_code}"
                )
                # 404 / 410 = genuinely missing; anything else proves DOI is registered
                assert resp.status_code not in (404, 410), (
                    f"DOI {doi!r} resolved to HTTP {resp.status_code} (not found / gone)"
                )
            except httpx.RequestError as exc:
                pytest.skip(f"Network error resolving {doi}: {exc}")

    print("[CHECK 2] PASSED")


# ---------------------------------------------------------------------------
# Check 3 - two runs produce near-identical results
# ---------------------------------------------------------------------------

_OVERLAP_THRESHOLD = 0.70  # at least 70% of DOIs must appear in both runs


@pytest.mark.asyncio
async def test_check3_stability():
    """
    CROSS-VERIFY 3: same topic twice gives >=70% DOI overlap.
    """
    topic = "intermittent fasting insulin sensitivity"

    print("\n[CHECK 3] Run 1 ...")
    run1 = await _run_pipeline(topic)
    dois1 = {p["doi"] for p in run1 if p.get("doi")}

    print("[CHECK 3] Run 2 ...")
    run2 = await _run_pipeline(topic)
    dois2 = {p["doi"] for p in run2 if p.get("doi")}

    overlap = dois1 & dois2
    union   = dois1 | dois2
    ratio   = len(overlap) / len(union) if union else 0.0

    print(
        f"  Run 1 DOIs : {len(dois1)}\n"
        f"  Run 2 DOIs : {len(dois2)}\n"
        f"  Overlap    : {len(overlap)} ({ratio:.0%})\n"
        f"  Only run 1 : {dois1 - dois2}\n"
        f"  Only run 2 : {dois2 - dois1}"
    )

    assert ratio >= _OVERLAP_THRESHOLD, (
        f"Stability too low: {ratio:.0%} overlap (threshold {_OVERLAP_THRESHOLD:.0%})"
    )
    print("[CHECK 3] PASSED")


# ---------------------------------------------------------------------------
# Check 4 - pipeline survives when Semantic Scholar is unreachable
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_check4_ss_down_graceful():
    """
    CROSS-VERIFY 4: monkey-patch _do_fetch_semantic_scholar to raise
    ConnectError on every call (equivalent to /etc/hosts block without
    needing admin rights) and verify the pipeline still completes.
    """
    from app.services.fetch import fetch_all

    async def _blocked(*args, **kwargs):
        raise httpx.ConnectError("Simulated: host blocked in /etc/hosts")

    print("\n[CHECK 4] Patching _do_fetch_semantic_scholar -> ConnectError ...")
    with patch("app.services.fetch._do_fetch_semantic_scholar", side_effect=_blocked):
        papers = await fetch_all(
            "intermittent fasting insulin sensitivity",
            db_client=_mock_db_client(),
        )

    print(f"  Pipeline returned {len(papers)} papers with SS blocked:")
    for p in papers:
        print(f"    [{p['source']:<18s}] {(p['title'] or '')[:65]!r}")

    assert len(papers) > 0, "Pipeline returned nothing when SS was blocked!"

    ss_papers = [p for p in papers if p.get("source") == "semantic_scholar"]
    assert not ss_papers, (
        f"Unexpected SS papers when SS was blocked: {ss_papers}"
    )

    for p in papers:
        assert p.get("doi"), f"Paper without DOI slipped through: {p['title']!r}"

    print("[CHECK 4] PASSED -- pipeline is resilient to SS being unreachable")
