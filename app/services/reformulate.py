"""
app/services/reformulate.py
---------------------------
Utilities for turning user inputs (questions or raw text) into verifiable
scientific assertions, and for suggesting candidate claims from paper abstracts.

All LLM calls are made to a locally-running Ollama instance:
    http://localhost:11434/api/generate  (model: llama3)
"""

from __future__ import annotations

import re
import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"
OLLAMA_TIMEOUT = 60.0  # seconds – generation can be slow on CPU

# Question-detection heuristics
_QUESTION_PREFIXES = re.compile(
    r"^\s*(does|is|are|can|do|will|would|could|should|has|have|was|were|did)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_question(text: str) -> bool:
    """Return True if *text* looks like a question rather than an assertion.

    Detection logic (either condition is sufficient):
      1. The stripped text ends with a literal `?`.
      2. It starts with one of the common question-starter verbs.
    """
    stripped = text.strip()
    if stripped.endswith("?"):
        return True
    if _QUESTION_PREFIXES.match(stripped):
        return True
    return False


def _call_ollama(prompt: str) -> str:
    """POST to the local Ollama /api/generate endpoint and return the response.

    Uses ``stream=False`` so the full response arrives in a single JSON object.

    Raises:
        httpx.HTTPError: on network / HTTP-level failures.
        KeyError: if the expected `response` key is absent from the payload.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    with httpx.Client(timeout=OLLAMA_TIMEOUT) as client:
        resp = client.post(OLLAMA_URL, json=payload)
        resp.raise_for_status()
    return resp.json()["response"].strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reformulate_to_claim(user_input: str) -> dict:
    """Convert *user_input* into a verifiable factual assertion.

    Behaviour
    ---------
    * If *user_input* is already an assertion (not a question), it is returned
      unchanged with ``reformulated=False``.
    * If *user_input* is a question, the local Ollama model rewrites it as a
      single factual assertion that can be verified against scientific
      literature, and the result is returned with ``reformulated=True``.

    Parameters
    ----------
    user_input:
        Raw string entered by the user -- either a question or an assertion.

    Returns
    -------
    dict
        For assertions::

            {"claim": <original text>, "reformulated": False}

        For questions::

            {
                "claim":        <reformulated assertion>,
                "original":     <original question>,
                "reformulated": True,
            }
    """
    if not _is_question(user_input):
        return {"claim": user_input, "reformulated": False}

    prompt = (
        "Rewrite the following question as a single factual assertion that can "
        "be verified as true or false against scientific literature. "
        "Return only the assertion, no explanation.\n"
        f"Question: {user_input}"
    )

    reformulated_text = _call_ollama(prompt)

    return {
        "claim": reformulated_text,
        "original": user_input,
        "reformulated": True,
    }


def suggest_claims(abstract_texts: list[str]) -> list[str]:
    """Generate 3-4 candidate checkable claims from a list of paper abstracts.

    The claims are suitable as starting-point suggestions shown to the user
    before they type their own input.

    Parameters
    ----------
    abstract_texts:
        A list of raw abstract strings fetched from academic sources.

    Returns
    -------
    list[str]
        A list of 3-4 short, self-contained, verifiable assertions extracted
        from the supplied abstracts. Returns an empty list if *abstract_texts*
        is empty.
    """
    if not abstract_texts:
        return []

    combined = "\n\n---\n\n".join(abstract_texts)

    prompt = (
        "You are a scientific fact-checker assistant. "
        "Read the following research abstracts and generate exactly 3 to 4 "
        "short, self-contained, verifiable assertions that could be checked "
        "against scientific literature. "
        "Each assertion must be on its own line, starting with a dash (-). "
        "Do not include any explanation or numbering -- only the assertions.\n\n"
        f"Abstracts:\n{combined}"
    )

    raw = _call_ollama(prompt)

    # Parse bullet lines that start with '-' or '*', stripping markers
    claims: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith(("-", "*")):
            claim = line.lstrip("-* ").strip()
            if claim:
                claims.append(claim)

    # Fallback: if the model did not use list markers, take non-empty lines
    if not claims:
        claims = [ln.strip() for ln in raw.splitlines() if ln.strip()]

    return claims[:4]  # cap at 4 suggestions
