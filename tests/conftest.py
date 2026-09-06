"""
tests/conftest.py
-----------------
Session-scoped stubs for heavy ML and infrastructure dependencies.

Stubs sentence_transformers and qdrant_client so that importing
embed.py or store.py in tests never triggers model downloads or
connections to external services.

These stubs are installed into sys.modules BEFORE any test module
is collected, which is why conftest.py is the right place for them.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Stub: sentence_transformers
# ---------------------------------------------------------------------------
# Prevent the real SentenceTransformer from being imported at module level in
# embed.py — which would trigger a HuggingFace download on first import.

_st_mock = MagicMock()
_st_mock.SentenceTransformer = MagicMock  # class-level mock
sys.modules.setdefault("sentence_transformers", _st_mock)

# ---------------------------------------------------------------------------
# Stub: qdrant_client sub-modules (only if not already imported)
# ---------------------------------------------------------------------------
# store.py imports from qdrant_client.models and qdrant_client.http.exceptions.
# Stub them so tests that don't need a live Qdrant don't fail on import.

def _stub_qdrant() -> None:
    qc = MagicMock()
    qc.QdrantClient = MagicMock
    sys.modules.setdefault("qdrant_client", qc)

    models = MagicMock()
    # Provide real-ish names that store.py unpacks
    models.Distance = MagicMock()
    models.Distance.COSINE = "Cosine"
    models.PointStruct = MagicMock(side_effect=lambda **kw: kw)
    models.VectorParams = MagicMock(side_effect=lambda **kw: kw)
    sys.modules.setdefault("qdrant_client.models", models)

    exc_mod = MagicMock()
    exc_mod.UnexpectedResponse = Exception  # make it catchable
    sys.modules.setdefault("qdrant_client.http", MagicMock())
    sys.modules.setdefault("qdrant_client.http.exceptions", exc_mod)


_stub_qdrant()
