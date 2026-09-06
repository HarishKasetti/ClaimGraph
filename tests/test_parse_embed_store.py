"""
tests/test_parse_embed_store.py
--------------------------------
Pytest suite for app/services/parse.py, embed.py, and store.py.

All external I/O is mocked:
  - GROBID / httpx  → respx
  - pdfplumber      → unittest.mock.patch
  - SentenceTransformer → MagicMock / patch
  - Qdrant          → MagicMock
  - MongoDB motor   → AsyncMock
"""

from __future__ import annotations

import textwrap
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import respx
from httpx import Response

# ---------------------------------------------------------------------------
# TEI-XML fixtures
# ---------------------------------------------------------------------------

_INTRO_TEXT = (
    "This paper investigates the effects of intermittent fasting on insulin "
    "sensitivity in adult humans. We recruited 100 participants."
)
_METHOD_TEXT = (
    "Participants followed a 16:8 fasting protocol for 12 weeks. "
    "Fasting glucose and insulin levels were measured at baseline and endpoint."
)

_GOOD_TEI = textwrap.dedent(f"""\
    <?xml version="1.0" encoding="UTF-8"?>
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <text>
        <body>
          <div>
            <head>Introduction</head>
            <p>{_INTRO_TEXT}</p>
          </div>
          <div>
            <head>Methods</head>
            <p>{_METHOD_TEXT}</p>
          </div>
          <div>
            <head>References</head>
            <p>Smith et al. 2020. Some journal.</p>
          </div>
        </body>
      </text>
    </TEI>
""")

_BIBLREF_TEI = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <text>
        <body>
          <div>
            <head>Introduction</head>
            <p>Real content here.</p>
          </div>
          <listBibl>
            <div>
              <head>References</head>
              <p>Jones 1999. Nature.</p>
            </div>
          </listBibl>
        </body>
      </text>
    </TEI>
""")

_LONG_PARA = " ".join([f"word{i}" for i in range(500)])
_LONG_TEI = textwrap.dedent(f"""\
    <?xml version="1.0" encoding="UTF-8"?>
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <text><body>
        <div>
          <head>BigSection</head>
          <p>{_LONG_PARA}</p>
        </div>
      </body></text>
    </TEI>
""")

_MALFORMED_TEI = "this is not xml <<< broken"


# ===========================================================================
# parse.py tests
# ===========================================================================

class TestChunkText:

    def test_section_labels_preserved(self):
        """chunk_text extracts the <head> label for each section."""
        from app.services.parse import chunk_text

        chunks = chunk_text(_GOOD_TEI, max_tokens=300, paper_id="p1")
        sections = {c["section"] for c in chunks}
        assert "Introduction" in sections
        assert "Methods" in sections

    def test_reference_section_skipped(self):
        """Sections with 'Reference' in the head label are excluded."""
        from app.services.parse import chunk_text

        chunks = chunk_text(_GOOD_TEI, max_tokens=300, paper_id="p1")
        sections = [c["section"] for c in chunks]
        assert not any("ref" in s.lower() or "bibliograph" in s.lower()
                        for s in sections)

    def test_listbibl_skipped(self):
        """<div> elements inside <listBibl> are never chunked."""
        from app.services.parse import chunk_text

        chunks = chunk_text(_BIBLREF_TEI, max_tokens=300, paper_id="p2")
        texts = " ".join(c["text"] for c in chunks)
        assert "Jones 1999" not in texts
        assert "Real content" in texts

    def test_max_tokens_not_exceeded(self):
        """No chunk contains more than max_tokens words."""
        from app.services.parse import chunk_text

        chunks = chunk_text(_LONG_TEI, max_tokens=300, paper_id="p3")
        assert chunks, "Expected at least one chunk from a 500-word paragraph"
        for c in chunks:
            assert len(c["text"].split()) <= 300, (
                f"Chunk exceeded 300 tokens: {len(c['text'].split())} words"
            )

    def test_paper_id_propagated(self):
        """paper_id is set on every chunk."""
        from app.services.parse import chunk_text

        chunks = chunk_text(_GOOD_TEI, paper_id="my_paper_42")
        for c in chunks:
            assert c["paper_id"] == "my_paper_42"

    def test_malformed_xml_returns_empty(self):
        """Malformed XML returns [] without raising."""
        from app.services.parse import chunk_text

        result = chunk_text(_MALFORMED_TEI, paper_id="bad")
        assert result == []

    def test_empty_string_returns_empty(self):
        """Empty input returns [] without raising."""
        from app.services.parse import chunk_text

        assert chunk_text("") == []
        assert chunk_text("   ") == []


class TestParsePdf:

    @pytest.mark.asyncio
    @respx.mock
    async def test_grobid_success_returns_tei(self):
        """200 from GROBID → returns the TEI-XML string."""
        from app.services.parse import parse_pdf

        respx.post("http://localhost:8070/api/processFulltextDocument").mock(
            return_value=Response(200, text=_GOOD_TEI)
        )
        result = await parse_pdf(b"%PDF-fake", paper_id="test_paper")
        assert "<TEI" in result
        assert "Introduction" in result

    @pytest.mark.asyncio
    @respx.mock
    async def test_grobid_timeout_uses_pdfplumber(self):
        """GROBID timeout triggers pdfplumber fallback."""
        import httpx
        from app.services.parse import parse_pdf

        respx.post("http://localhost:8070/api/processFulltextDocument").mock(
            side_effect=httpx.TimeoutException("timeout")
        )
        with patch("app.services.parse._pdfplumber_fallback") as mock_fb:
            mock_fb.return_value = _GOOD_TEI
            result = await parse_pdf(b"%PDF-fake", paper_id="timeout_paper")

        mock_fb.assert_called_once()
        assert result == _GOOD_TEI

    @pytest.mark.asyncio
    @respx.mock
    async def test_grobid_error_status_uses_pdfplumber(self):
        """GROBID HTTP 500 triggers pdfplumber fallback."""
        from app.services.parse import parse_pdf

        respx.post("http://localhost:8070/api/processFulltextDocument").mock(
            return_value=Response(500, text="Internal Server Error")
        )
        with patch("app.services.parse._pdfplumber_fallback") as mock_fb:
            mock_fb.return_value = "<TEI><text><body><div><head>Body</head><p>text</p></div></body></text></TEI>"
            result = await parse_pdf(b"%PDF-broken", paper_id="error_paper")

        mock_fb.assert_called_once()
        assert "<TEI" in result


# ===========================================================================
# embed.py tests
# ===========================================================================

class TestEmbedChunks:

    def _sample_chunks(self, n: int = 3) -> list[dict]:
        return [{"text": f"Sample text {i}", "section": "Intro", "paper_id": "p"} for i in range(n)]

    def test_embed_chunks_shape(self):
        """embed_chunks returns an array of shape (N, 768)."""
        from app.services import embed as embed_mod

        fake_vectors = np.random.rand(3, 768).astype(np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_vectors

        # new= makes the replacement itself callable and returns mock_model when called
        with patch.object(embed_mod, "get_model", new=MagicMock(return_value=mock_model)):
            result = embed_mod.embed_chunks(self._sample_chunks(3), batch_size=16)

        assert result.shape == (3, 768)
        assert result.dtype == np.float32

    def test_embed_chunks_wrong_dim_raises(self):
        """If the model returns the wrong dimension, ValueError is raised."""
        from app.services import embed as embed_mod

        fake_512 = np.random.rand(3, 512).astype(np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_512

        with patch.object(embed_mod, "get_model", new=MagicMock(return_value=mock_model)):
            with pytest.raises(ValueError, match="Unexpected embedding shape"):
                embed_mod.embed_chunks(self._sample_chunks(3))

    def test_embed_chunks_empty_raises(self):
        """Empty chunk list raises ValueError before get_model is ever called."""
        from app.services.embed import embed_chunks

        with pytest.raises(ValueError, match="empty"):
            embed_chunks([])


# ===========================================================================
# store.py tests
# ===========================================================================

class TestSaveToQdrant:

    def _make_client(self):
        client = MagicMock()
        client.get_collection.return_value = MagicMock()  # collection exists
        client.upsert.return_value = MagicMock()
        return client

    def test_dim_guard_raises_on_wrong_dim(self):
        """Vectors with dim != 768 raise ValueError before any Qdrant call."""
        from app.services.store import save_to_qdrant

        bad_vectors = np.zeros((3, 512), dtype=np.float32)
        chunks = [{"text": "t", "section": "s", "paper_id": "p"}] * 3

        with patch("app.services.store.QdrantClient", return_value=self._make_client()):
            with pytest.raises(ValueError, match="768"):
                save_to_qdrant("paper1", chunks, bad_vectors)

    def test_correct_dim_upserts(self):
        """768-dim vectors are upserted without error."""
        from app.services.store import save_to_qdrant

        vectors = np.zeros((2, 768), dtype=np.float32)
        chunks = [{"text": f"t{i}", "section": "s", "paper_id": "p"} for i in range(2)]
        mock_client = self._make_client()

        with patch("app.services.store.QdrantClient", return_value=mock_client):
            count = save_to_qdrant("paper1", chunks, vectors)

        assert count == 2
        mock_client.upsert.assert_called_once()

    def test_deterministic_point_ids(self):
        """Same paper_id + index always produces the same point ID."""
        from app.services.store import _point_id

        id_a = _point_id("paper_abc", 0)
        id_b = _point_id("paper_abc", 0)
        id_c = _point_id("paper_abc", 1)

        assert id_a == id_b           # deterministic
        assert id_a != id_c           # unique per chunk index


class TestSaveToMongo:

    @pytest.mark.asyncio
    async def test_upsert_by_doi(self):
        """save_to_mongo calls update_one with the DOI filter."""
        from app.services.store import save_to_mongo

        result_mock = MagicMock()
        result_mock.upserted_id = "fake_id_123"

        mock_coll = MagicMock()
        mock_coll.update_one = AsyncMock(return_value=result_mock)

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_coll)
        mock_client = MagicMock()
        mock_client.__getitem__ = MagicMock(return_value=mock_db)
        mock_client.close = MagicMock()

        metadata = {"doi": "10.1234/test", "title": "Test Paper"}
        mongo_id = await save_to_mongo("paper_x", metadata, db_client=mock_client)

        mock_coll.update_one.assert_awaited_once()
        call_args = mock_coll.update_one.call_args
        assert call_args[0][0] == {"doi": "10.1234/test"}   # filter
        assert mongo_id == "fake_id_123"

    @pytest.mark.asyncio
    async def test_save_to_mongo_missing_doi_raises(self):
        """Metadata without 'doi' raises ValueError."""
        from app.services.store import save_to_mongo

        with pytest.raises(ValueError, match="doi"):
            await save_to_mongo("paper_x", {"title": "No DOI here"})


class TestIngestPaper:

    @pytest.mark.asyncio
    async def test_cache_hit_returns_early(self):
        """
        If check_global_cache returns a mongo_id the full parse/embed/store
        pipeline is never called.
        """
        from app.services import store as store_mod

        with patch("app.services.store.check_global_cache", new=AsyncMock(return_value="cached_mongo_id")):
            with patch("app.services.store.parse_pdf", new=AsyncMock()) as mock_parse:
                n, mongo_id = await store_mod.ingest_paper(
                    "paper1",
                    b"%PDF-fake",
                    {"doi": "10.1234/cached"},
                )

        assert n == 0
        assert mongo_id == "cached_mongo_id"
        mock_parse.assert_not_awaited()
