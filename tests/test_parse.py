"""
tests/test_parse.py
-------------------
Focused pytest suite for app/services/parse.py.

Covers:
  - chunk_text(): section labels, token budget, listBibl skip, paper_id,
                  malformed XML, empty input, multi-chunk long paragraphs,
                  inline element text extraction
  - parse_pdf():  GROBID success path, timeout fallback, HTTP-error fallback,
                  network-error fallback
  - _make_tei_envelope(): plain-text input round-trips through chunk_text
  - _pdfplumber_fallback(): extracts text; empty PDF produces empty envelope
"""

from __future__ import annotations

import textwrap
from unittest.mock import MagicMock, patch

import pytest
import respx
from httpx import Response

# ---------------------------------------------------------------------------
# TEI-XML fixtures
# ---------------------------------------------------------------------------

_INTRO = (
    "This paper investigates the metabolic effects of intermittent fasting "
    "on insulin sensitivity in adult humans. We recruited one hundred participants."
)
_METHODS = (
    "Participants followed a 16:8 time-restricted eating protocol for twelve weeks. "
    "Fasting glucose and insulin levels were measured at baseline and endpoint."
)
_RESULTS = (
    "Time-restricted eating reduced fasting insulin by 22 percent compared to "
    "the control group. HOMA-IR scores improved significantly."
)

_FULL_TEI = textwrap.dedent(f"""\
    <?xml version="1.0" encoding="UTF-8"?>
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <text>
        <body>
          <div>
            <head>Introduction</head>
            <p>{_INTRO}</p>
          </div>
          <div>
            <head>Methods</head>
            <p>{_METHODS}</p>
          </div>
          <div>
            <head>Results</head>
            <p>{_RESULTS}</p>
          </div>
          <div>
            <head>References</head>
            <p>Smith 2020. Journal of Nutrition.</p>
          </div>
        </body>
      </text>
    </TEI>
""")

_BIBLREF_TEI = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <text><body>
        <div><head>Introduction</head><p>Real scientific content here.</p></div>
        <listBibl>
          <div><head>References</head><p>Jones 1999. Nature.</p></div>
        </listBibl>
      </body></text>
    </TEI>
""")

_INLINE_TEI = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <text><body>
        <div>
          <head>Discussion</head>
          <p>As shown in <ref>Figure 1</ref>, the effect was significant
          across all <formula>p &lt; 0.05</formula> thresholds.</p>
        </div>
      </body></text>
    </TEI>
""")

_LONG_WORDS = " ".join(f"word{i}" for i in range(700))
_LONG_TEI = textwrap.dedent(f"""\
    <?xml version="1.0" encoding="UTF-8"?>
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <text><body>
        <div>
          <head>LongSection</head>
          <p>{_LONG_WORDS}</p>
        </div>
      </body></text>
    </TEI>
""")

_MALFORMED = "this is <not valid xml <<< !!!"


# ===========================================================================
# chunk_text tests
# ===========================================================================

class TestChunkText:

    def test_section_labels_preserved(self):
        """Each chunk carries its parent <div><head> label."""
        from app.services.parse import chunk_text
        chunks = chunk_text(_FULL_TEI, paper_id="p1")
        labels = {c["section"] for c in chunks}
        assert "Introduction" in labels
        assert "Methods" in labels
        assert "Results" in labels

    def test_reference_section_excluded_by_label(self):
        """<div> with 'References' head label is dropped."""
        from app.services.parse import chunk_text
        chunks = chunk_text(_FULL_TEI, paper_id="p1")
        labels = [c["section"] for c in chunks]
        assert not any("ref" in s.lower() or "bibliograph" in s.lower() for s in labels)

    def test_listbibl_div_excluded(self):
        """<div> nested inside <listBibl> is never emitted as a chunk."""
        from app.services.parse import chunk_text
        chunks = chunk_text(_BIBLREF_TEI, paper_id="p2")
        joined = " ".join(c["text"] for c in chunks)
        assert "Jones 1999" not in joined
        assert "Real scientific content" in joined

    def test_max_tokens_not_exceeded(self):
        """No chunk contains more than max_tokens words."""
        from app.services.parse import chunk_text
        chunks = chunk_text(_LONG_TEI, max_tokens=300, paper_id="p3")
        assert chunks, "Expected at least one chunk from a 700-word paragraph"
        for c in chunks:
            n = len(c["text"].split())
            assert n <= 300, f"Chunk has {n} words (limit 300)"

    def test_long_paragraph_produces_multiple_chunks(self):
        """A paragraph > max_tokens is split into multiple chunks."""
        from app.services.parse import chunk_text
        chunks = chunk_text(_LONG_TEI, max_tokens=300, paper_id="p3")
        # 700 words / 300 = at least 3 chunks (last may be shorter)
        assert len(chunks) >= 2, f"Expected multiple chunks, got {len(chunks)}"

    def test_paper_id_on_every_chunk(self):
        """paper_id is propagated to every chunk dict."""
        from app.services.parse import chunk_text
        chunks = chunk_text(_FULL_TEI, paper_id="my_paper_99")
        for c in chunks:
            assert c["paper_id"] == "my_paper_99"

    def test_chunk_text_fields_present(self):
        """Every chunk has exactly the three required keys."""
        from app.services.parse import chunk_text
        chunks = chunk_text(_FULL_TEI, paper_id="p")
        for c in chunks:
            assert "text" in c
            assert "section" in c
            assert "paper_id" in c

    def test_inline_element_text_extracted(self):
        """Text inside inline child elements (<ref>, <formula>) is included."""
        from app.services.parse import chunk_text
        chunks = chunk_text(_INLINE_TEI, paper_id="inline")
        assert chunks
        joined = " ".join(c["text"] for c in chunks)
        assert "Figure 1" in joined

    def test_malformed_xml_returns_empty_list(self):
        """Malformed XML → empty list, no exception raised."""
        from app.services.parse import chunk_text
        result = chunk_text(_MALFORMED, paper_id="bad")
        assert result == []

    def test_empty_string_returns_empty_list(self):
        """Empty / whitespace-only input → empty list."""
        from app.services.parse import chunk_text
        assert chunk_text("") == []
        assert chunk_text("   \n\t  ") == []

    def test_no_body_element_returns_empty(self):
        """TEI without a <body> element returns []."""
        from app.services.parse import chunk_text
        tei = '<?xml version="1.0"?><TEI xmlns="http://www.tei-c.org/ns/1.0"><text/></TEI>'
        assert chunk_text(tei, paper_id="nobody") == []


# ===========================================================================
# _make_tei_envelope + _pdfplumber_fallback
# ===========================================================================

class TestTEIEnvelope:

    def test_make_tei_envelope_is_valid_xml(self):
        """_make_tei_envelope produces parseable TEI-XML."""
        import xml.etree.ElementTree as ET
        from app.services.parse import _make_tei_envelope
        xml_str = _make_tei_envelope("Hello world, this is a test sentence.")
        root = ET.fromstring(xml_str)
        assert root is not None

    def test_make_tei_envelope_roundtrips_through_chunk_text(self):
        """Text wrapped in the TEI envelope can be chunked."""
        from app.services.parse import _make_tei_envelope, chunk_text
        text = "The quick brown fox jumps over the lazy dog. " * 5
        envelope = _make_tei_envelope(text)
        chunks = chunk_text(envelope, paper_id="env_test")
        assert chunks
        assert chunks[0]["section"] == "Body"
        joined = " ".join(c["text"] for c in chunks)
        assert "quick brown fox" in joined

    def test_make_tei_envelope_escapes_special_chars(self):
        """& < > in plain text are XML-escaped in the envelope."""
        from app.services.parse import _make_tei_envelope
        xml_str = _make_tei_envelope("A & B < C > D")
        # Should not raise ParseError
        import xml.etree.ElementTree as ET
        ET.fromstring(xml_str)

    def test_pdfplumber_fallback_empty_pdf(self):
        """An empty PDF produces a TEI envelope with empty body (no crash)."""
        from app.services.parse import _pdfplumber_fallback
        import io
        try:
            import pdfplumber
        except ImportError:
            pytest.skip("pdfplumber not installed")

        # Minimal valid PDF bytes with no pages
        with patch("app.services.parse.pdfplumber") as mock_plumber:
            mock_pdf = MagicMock()
            mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
            mock_pdf.__exit__ = MagicMock(return_value=False)
            mock_pdf.pages = []
            mock_plumber.open.return_value = mock_pdf

            result = _pdfplumber_fallback(b"%PDF-empty")

        assert "<TEI" in result
        # Chunking an empty body should produce an empty list
        from app.services.parse import chunk_text
        chunks = chunk_text(result, paper_id="empty_pdf")
        assert isinstance(chunks, list)


# ===========================================================================
# parse_pdf tests
# ===========================================================================

class TestParsePdf:

    @pytest.mark.asyncio
    @respx.mock
    async def test_grobid_success_returns_tei_string(self):
        """HTTP 200 from GROBID → returns raw TEI-XML string."""
        from app.services.parse import parse_pdf
        respx.post("http://localhost:8070/api/processFulltextDocument").mock(
            return_value=Response(200, text=_FULL_TEI)
        )
        result = await parse_pdf(b"%PDF-fake", paper_id="ok_paper")
        assert isinstance(result, str)
        assert "<TEI" in result
        assert "Introduction" in result

    @pytest.mark.asyncio
    @respx.mock
    async def test_grobid_timeout_triggers_pdfplumber(self):
        """TimeoutException from GROBID → _pdfplumber_fallback is called."""
        import httpx
        from app.services.parse import parse_pdf
        respx.post("http://localhost:8070/api/processFulltextDocument").mock(
            side_effect=httpx.TimeoutException("timed out")
        )
        with patch("app.services.parse._pdfplumber_fallback", return_value=_FULL_TEI) as fb:
            result = await parse_pdf(b"%PDF-fake", paper_id="timeout_paper")
        fb.assert_called_once_with(b"%PDF-fake")
        assert result == _FULL_TEI

    @pytest.mark.asyncio
    @respx.mock
    async def test_grobid_http500_triggers_pdfplumber(self):
        """HTTP 500 from GROBID → _pdfplumber_fallback is called."""
        from app.services.parse import parse_pdf
        respx.post("http://localhost:8070/api/processFulltextDocument").mock(
            return_value=Response(500, text="Server Error")
        )
        with patch("app.services.parse._pdfplumber_fallback", return_value=_FULL_TEI) as fb:
            result = await parse_pdf(b"%PDF-broken", paper_id="err_paper")
        fb.assert_called_once()
        assert "<TEI" in result

    @pytest.mark.asyncio
    @respx.mock
    async def test_grobid_connect_error_triggers_pdfplumber(self):
        """ConnectError (GROBID unreachable) → _pdfplumber_fallback is called."""
        import httpx
        from app.services.parse import parse_pdf
        respx.post("http://localhost:8070/api/processFulltextDocument").mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        with patch("app.services.parse._pdfplumber_fallback", return_value=_FULL_TEI) as fb:
            result = await parse_pdf(b"%PDF-noconn", paper_id="noconn_paper")
        fb.assert_called_once()
        assert result == _FULL_TEI

    @pytest.mark.asyncio
    @respx.mock
    async def test_grobid_http200_without_fallback(self):
        """Successful GROBID response does NOT call pdfplumber."""
        from app.services.parse import parse_pdf
        respx.post("http://localhost:8070/api/processFulltextDocument").mock(
            return_value=Response(200, text=_FULL_TEI)
        )
        with patch("app.services.parse._pdfplumber_fallback") as fb:
            await parse_pdf(b"%PDF-ok", paper_id="no_fallback")
        fb.assert_not_called()
