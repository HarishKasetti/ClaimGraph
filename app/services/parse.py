"""
app/services/parse.py
---------------------
PDF parsing service for ClaimGraph.

Provides:
  - parse_pdf(pdf_bytes, paper_id)  -> TEI-XML string
      Calls GROBID /api/processFulltextDocument (timeout=30s).
      Falls back to pdfplumber on timeout or non-200 response.
  - chunk_text(tei_xml, max_tokens, paper_id) -> list[{text, section, paper_id}]
      Chunks TEI-XML by <div> section elements, capping each chunk at
      max_tokens words. Reference sections (<listBibl>) are skipped.
"""

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Any, Optional

import httpx
import pdfplumber

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_GROBID_URL: str = os.getenv("GROBID_URL", "http://localhost:8070")
_GROBID_ENDPOINT: str = f"{_GROBID_URL}/api/processFulltextDocument"
_GROBID_TIMEOUT: float = 30.0

_TEI_NS: str = "http://www.tei-c.org/ns/1.0"
_NS: dict[str, str] = {"tei": _TEI_NS}

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Chunk = dict[str, Any]


# ---------------------------------------------------------------------------
# PDF → TEI-XML  (GROBID primary, pdfplumber fallback)
# ---------------------------------------------------------------------------

def _make_tei_envelope(body_text: str) -> str:
    """
    Wrap plain text in a minimal TEI-XML envelope so that chunk_text()
    always receives a consistent XML structure regardless of which parser ran.
    """
    escaped = (
        body_text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<TEI xmlns="{_TEI_NS}">'
        "<text><body>"
        "<div>"
        "<head>Body</head>"
        f"<p>{escaped}</p>"
        "</div>"
        "</body></text>"
        "</TEI>"
    )


def _pdfplumber_fallback(pdf_bytes: bytes) -> str:
    """
    Extract plain text from PDF bytes with pdfplumber and return it
    wrapped in a minimal TEI-XML envelope.
    """
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            pages: list[str] = []
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    pages.append(txt.strip())
        full_text = "\n\n".join(pages)
        logger.info("pdfplumber extracted %d characters from PDF", len(full_text))
    except Exception as exc:  # noqa: BLE001
        logger.warning("pdfplumber also failed: %s", exc)
        full_text = ""
    return _make_tei_envelope(full_text)


async def parse_pdf(pdf_bytes: bytes, paper_id: str = "") -> str:
    """
    Convert PDF bytes to a TEI-XML string.

    Primary path: POST pdf_bytes to GROBID /api/processFulltextDocument.
    Fallback: pdfplumber (triggered by timeout or HTTP error).

    Parameters
    ----------
    pdf_bytes:
        Raw PDF file content.
    paper_id:
        Optional identifier used for log messages.

    Returns
    -------
    str
        TEI-XML string (always valid, even if pdfplumber had to be used).
    """
    try:
        async with httpx.AsyncClient(timeout=_GROBID_TIMEOUT) as client:
            resp = await client.post(
                _GROBID_ENDPOINT,
                files={"input": ("paper.pdf", pdf_bytes, "application/pdf")},
                data={"consolidateHeader": "1"},
            )

        if resp.status_code == 200:
            logger.info(
                "GROBID processed paper_id=%r (%d bytes TEI)",
                paper_id,
                len(resp.text),
            )
            return resp.text

        logger.warning(
            "GROBID returned HTTP %d for paper_id=%r — using pdfplumber fallback",
            resp.status_code,
            paper_id,
        )

    except httpx.TimeoutException:
        logger.warning(
            "GROBID timed out (>%.0fs) for paper_id=%r — using pdfplumber fallback",
            _GROBID_TIMEOUT,
            paper_id,
        )
    except httpx.RequestError as exc:
        logger.warning(
            "GROBID request error for paper_id=%r (%s) — using pdfplumber fallback",
            paper_id,
            exc,
        )

    return _pdfplumber_fallback(pdf_bytes)


# ---------------------------------------------------------------------------
# TEI-XML → chunks
# ---------------------------------------------------------------------------

def _tag(local: str) -> str:
    """Return a Clark-notation tag name for the TEI namespace."""
    return f"{{{_TEI_NS}}}{local}"


def _words(text: str) -> list[str]:
    """Split text into words (whitespace-delimited, strip empty strings)."""
    return [w for w in text.split() if w]


def chunk_text(
    tei_xml: str,
    max_tokens: int = 300,
    paper_id: str = "",
) -> list[Chunk]:
    """
    Chunk a TEI-XML document by section, honouring a token (word) budget.

    Strategy
    --------
    - Iterates ``<div>`` elements directly under ``<body>``.
    - The section label is taken from the first ``<head>`` child of the div.
    - ``<p>`` elements within the div are processed in order.  Their words
      are accumulated into a growing buffer; when the buffer would exceed
      *max_tokens*, the current buffer is emitted as a chunk and a new
      buffer starts with the current paragraph's overflow.
    - Any div whose parent tag is ``<listBibl>`` (reference list) is skipped.
    - Returns an empty list on malformed XML (logs a warning, never raises).

    Parameters
    ----------
    tei_xml:
        Raw TEI-XML string (from GROBID or the pdfplumber envelope).
    max_tokens:
        Maximum number of whitespace-delimited tokens per chunk.
    paper_id:
        Passed through into every chunk dict for downstream traceability.

    Returns
    -------
    list[dict]
        Each element: ``{"text": str, "section": str, "paper_id": str}``.
    """
    if not tei_xml or not tei_xml.strip():
        return []

    try:
        root = ET.fromstring(tei_xml)
    except ET.ParseError as exc:
        logger.warning("chunk_text: malformed TEI-XML (paper_id=%r): %s", paper_id, exc)
        return []

    # Locate <body> — may be nested under <text>
    body = root.find(f".//{_tag('body')}")
    if body is None:
        logger.warning("chunk_text: no <body> element found (paper_id=%r)", paper_id)
        return []

    chunks: list[Chunk] = []

    def _flush(buf: list[str], section: str) -> None:
        text = " ".join(buf).strip()
        if text:
            chunks.append({"text": text, "section": section, "paper_id": paper_id})

    for div in body.iter(_tag("div")):
        # Skip reference sections
        parent = _get_parent(root, div)
        if parent is not None and parent.tag == _tag("listBibl"):
            continue
        # Also skip if this div is itself a listBibl child
        head_el = div.find(_tag("head"))
        section_label = (
            (head_el.text or "").strip() if head_el is not None else "Unknown"
        ) or "Unknown"

        if "ref" in section_label.lower() or "bibliograph" in section_label.lower():
            continue  # Skip reference/bibliography sections by label too

        buf: list[str] = []

        for p in div.findall(_tag("p")):
            # Gather all text nodes within the paragraph (including tail text
            # from inline elements like <ref>, <formula>, etc.)
            para_text = _extract_text(p)
            para_words = _words(para_text)

            if not para_words:
                continue

            # If adding this paragraph would NOT exceed the budget, append it
            if len(buf) + len(para_words) <= max_tokens:
                buf.extend(para_words)
            else:
                # Flush current buffer first (if non-empty)
                if buf:
                    _flush(buf, section_label)
                    buf = []

                # Then handle the paragraph itself — it may be > max_tokens alone
                i = 0
                while i < len(para_words):
                    slice_ = para_words[i : i + max_tokens]
                    if len(slice_) == max_tokens:
                        _flush(slice_, section_label)
                    else:
                        buf = slice_  # carry the remainder into the next iteration
                    i += max_tokens

        # Flush any remaining buffer for this section
        if buf:
            _flush(buf, section_label)

    return chunks


# ---------------------------------------------------------------------------
# Helper: find parent element (ET has no parent pointer)
# ---------------------------------------------------------------------------

def _get_parent(
    root: ET.Element,
    target: ET.Element,
) -> Optional[ET.Element]:
    """Return the direct parent of *target* within *root*, or None."""
    for parent in root.iter():
        for child in parent:
            if child is target:
                return parent
    return None


def _extract_text(element: ET.Element) -> str:
    """
    Recursively extract all text content from an element, including
    text tails from inline child elements (e.g. <ref>, <formula>).
    """
    parts: list[str] = []
    if element.text:
        parts.append(element.text)
    for child in element:
        parts.append(_extract_text(child))
        if child.tail:
            parts.append(child.tail)
    return " ".join(parts)
