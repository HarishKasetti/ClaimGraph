"""
app/services/embed.py
---------------------
Embedding service for ClaimGraph using allenai/specter2_base.

Provides:
  - get_model()          -> SentenceTransformer  (lazy singleton)
  - embed_chunks(chunks) -> np.ndarray of shape (N, 768)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MODEL_NAME: str = "allenai/specter2_base"
_EXPECTED_DIM: int = 768

# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """
    Return the SPECTER2 model, downloading from HuggingFace on first call.

    The model is cached as a module-level singleton so it is only loaded
    once per process lifetime.
    """
    global _model  # noqa: PLW0603
    if _model is None:
        logger.info("Loading embedding model %r ...", _MODEL_NAME)
        _model = SentenceTransformer(_MODEL_NAME)
        # Verify dimension immediately so failures are caught at load time
        sample = _model.encode(["warmup"], convert_to_numpy=True)
        actual_dim = sample.shape[1]
        if actual_dim != _EXPECTED_DIM:
            _model = None
            raise ValueError(
                f"Model {_MODEL_NAME!r} returned dimension {actual_dim}, "
                f"expected {_EXPECTED_DIM}."
            )
        logger.info(
            "Model %r loaded — embedding dimension: %d", _MODEL_NAME, actual_dim
        )
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_chunks(
    chunks: list[dict[str, Any]],
    batch_size: int = 16,
) -> np.ndarray:
    """
    Embed a list of chunk dicts using SPECTER2.

    Parameters
    ----------
    chunks:
        List of dicts as returned by ``parse.chunk_text()``.
        Each dict must contain at minimum a ``"text"`` key.
    batch_size:
        Number of texts to encode per forward pass.  16 is a safe default
        for a CPU machine; increase to 32–64 on GPU.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(len(chunks), 768)``.

    Raises
    ------
    ValueError
        If ``chunks`` is empty, or if the model returns an unexpected
        embedding dimension (should not happen with specter2_base).
    """
    if not chunks:
        raise ValueError("embed_chunks received an empty chunk list.")

    texts: list[str] = [c.get("text", "") for c in chunks]
    model = get_model()

    logger.info(
        "Embedding %d chunks (batch_size=%d) ...", len(texts), batch_size
    )
    vectors: np.ndarray = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    if vectors.ndim != 2 or vectors.shape[1] != _EXPECTED_DIM:
        raise ValueError(
            f"Unexpected embedding shape {vectors.shape}; "
            f"expected (N, {_EXPECTED_DIM})."
        )

    logger.info("Produced embeddings: shape=%s dtype=%s", vectors.shape, vectors.dtype)
    return vectors.astype(np.float32)
