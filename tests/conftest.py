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
from unittest.mock import MagicMock, patch

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

# ---------------------------------------------------------------------------
# Stub: transformers (pipeline used by stance.py)
# ---------------------------------------------------------------------------
# Prevent the real DeBERTa model from loading at import time.
# Individual tests that need a controlled pipeline will patch get_pipeline()
# directly with a deterministic callable.

def _stub_transformers() -> None:
    tf_mock = MagicMock()
    # pipeline(task, model=...) returns a MagicMock by default;
    # tests will override get_pipeline() themselves.
    tf_mock.pipeline = MagicMock(return_value=MagicMock())
    sys.modules.setdefault("transformers", tf_mock)


_stub_transformers()

# ---------------------------------------------------------------------------
# Stub: datasets (HuggingFace) — only needed if imported at module level
# ---------------------------------------------------------------------------
# calibrate.py does NOT import datasets at module level, so this is a safety
# net for any future changes.  build_calibration_set.py uses a late import.

def _stub_datasets() -> None:
    ds_mock = MagicMock()
    ds_mock.load_dataset = MagicMock(return_value=MagicMock())
    sys.modules.setdefault("datasets", ds_mock)


_stub_datasets()

# ---------------------------------------------------------------------------
# Stub: motor.motor_asyncio — prevent real MongoDB connections in tests
# ---------------------------------------------------------------------------
# test_calibrate.py patches motor inline; this stub is a belt-and-suspenders
# guard so that importing calibrate.py never opens a real connection.

def _stub_motor() -> None:
    motor_mock = MagicMock()
    motor_mock.AsyncIOMotorClient = MagicMock()
    sys.modules.setdefault("motor", MagicMock())
    sys.modules.setdefault("motor.motor_asyncio", motor_mock)


_stub_motor()

# ---------------------------------------------------------------------------
# Stub: redis — prevent connection attempts when importing celery_app
# ---------------------------------------------------------------------------

def _stub_redis() -> None:
    redis_mock = MagicMock()
    sys.modules.setdefault("redis", redis_mock)
    sys.modules.setdefault("redis.client", MagicMock())
    sys.modules.setdefault("redis.connection", MagicMock())


_stub_redis()

# ---------------------------------------------------------------------------
# Stub: celery + kombu — allow importing workers.celery_app without a broker
# ---------------------------------------------------------------------------

def _stub_celery() -> None:
    # Only stub if celery isn't importable (CI without Redis).
    # In a real env celery IS installed, so we leave it but stub the
    # connection-making parts used at import time.
    try:
        import celery  # noqa: F401 — real celery installed
    except ImportError:
        celery_mock = MagicMock()
        celery_mock.Celery = MagicMock(return_value=MagicMock())
        sys.modules.setdefault("celery", celery_mock)
        sys.modules.setdefault("celery.result", MagicMock())


_stub_celery()
