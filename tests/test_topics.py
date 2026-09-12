"""
tests/test_topics.py
---------------------
Unit tests for Phase 8 — Topic endpoints and Celery task.

All tests are fully isolated:
  - No real Redis/Celery connections (workers.tasks and celery.result patched)
  - No real MongoDB (motor patched via conftest + inline patches)
  - FastAPI TestClient used for HTTP route tests

Test Groups
-----------
TestPostTopics          -- POST /api/v1/topics behaviour
TestGetTopicStatus      -- GET /api/v1/topics/{id}/status lifecycle
TestGetTopicSuggestions -- GET /api/v1/topics/{id}/suggestions
TestIngestTopicTask     -- workers.tasks.ingest_topic Celery task logic
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# App client fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Return a synchronous TestClient for the FastAPI app."""
    from app.main import app
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_TOPIC_TEXT = "intermittent fasting insulin sensitivity"
_TOPIC_ID   = "a1b2c3d4e5f6a1b2c3d4"   # fake deterministic ID


def _pending_doc():
    return {
        "_id":               _TOPIC_ID,
        "topic_text":        _TOPIC_TEXT,
        "status":            "pending",
        "paper_count":       0,
        "contradiction_count": 0,
        "job_id":            "celery-job-001",
        "abstracts":         [],
    }


def _ready_doc():
    return {
        **_pending_doc(),
        "status":            "ready",
        "paper_count":       8,
        "contradiction_count": 2,
        "abstracts":         [
            "Study A found intermittent fasting reduces insulin resistance.",
            "Study B found no significant effect on insulin sensitivity.",
        ],
        "paper_ids": ["p001", "p002"],
    }


# ---------------------------------------------------------------------------
# TestPostTopics
# ---------------------------------------------------------------------------

class TestPostTopics:
    """POST /api/v1/topics — fires background task, returns immediately."""

    def _post(self, client, topic=_TOPIC_TEXT):
        mock_result = MagicMock()
        mock_result.id = "celery-job-001"

        with patch("app.api.routes.topics._upsert_topic", new_callable=AsyncMock), \
             patch("app.api.routes.topics._get_topic", new_callable=AsyncMock,
                   return_value=_pending_doc()), \
             patch("workers.tasks.ingest_topic") as mock_task:
            mock_task.delay.return_value = mock_result
            response = client.post("/api/v1/topics", json={"topic": topic})

        return response

    def test_returns_202(self, client) -> None:
        response = self._post(client)
        assert response.status_code == 202, response.text

    def test_response_has_required_keys(self, client) -> None:
        response = self._post(client)
        data = response.json()
        assert "topic_id" in data
        assert "status" in data
        assert "job_id" in data

    def test_status_is_pending(self, client) -> None:
        response = self._post(client)
        assert response.json()["status"] == "pending"

    def test_job_id_is_string(self, client) -> None:
        response = self._post(client)
        assert isinstance(response.json()["job_id"], str)
        assert response.json()["job_id"]  # not empty

    def test_topic_id_is_string(self, client) -> None:
        response = self._post(client)
        assert isinstance(response.json()["topic_id"], str)
        assert response.json()["topic_id"]  # not empty

    def test_same_topic_produces_same_topic_id(self, client) -> None:
        """Submitting the same topic text twice must return the same topic_id."""
        r1 = self._post(client, topic=_TOPIC_TEXT)
        r2 = self._post(client, topic=_TOPIC_TEXT)
        assert r1.json()["topic_id"] == r2.json()["topic_id"]

    def test_empty_topic_returns_422(self, client) -> None:
        response = client.post("/api/v1/topics", json={"topic": ""})
        assert response.status_code == 422

    def test_celery_delay_called_once(self, client) -> None:
        """ingest_topic.delay() must be called exactly once per request."""
        mock_result = MagicMock()
        mock_result.id = "celery-job-002"

        with patch("app.api.routes.topics._upsert_topic", new_callable=AsyncMock), \
             patch("app.api.routes.topics._get_topic", new_callable=AsyncMock,
                   return_value=_pending_doc()), \
             patch("workers.tasks.ingest_topic") as mock_task:
            mock_task.delay.return_value = mock_result
            client.post("/api/v1/topics", json={"topic": "CRISPR gene editing"})
            mock_task.delay.assert_called_once()


# ---------------------------------------------------------------------------
# TestGetTopicStatus
# ---------------------------------------------------------------------------

class TestGetTopicStatus:
    """GET /api/v1/topics/{id}/status — status lifecycle."""

    def test_returns_200_for_existing_topic(self, client) -> None:
        with patch("app.api.routes.topics._get_topic", new_callable=AsyncMock,
                   return_value=_ready_doc()):
            response = client.get(f"/api/v1/topics/{_TOPIC_ID}/status")
        assert response.status_code == 200

    def test_returns_404_for_unknown_topic(self, client) -> None:
        with patch("app.api.routes.topics._get_topic", new_callable=AsyncMock,
                   return_value={}):
            response = client.get("/api/v1/topics/nonexistent/status")
        assert response.status_code == 404

    def test_ready_status_has_required_keys(self, client) -> None:
        with patch("app.api.routes.topics._get_topic", new_callable=AsyncMock,
                   return_value=_ready_doc()):
            response = client.get(f"/api/v1/topics/{_TOPIC_ID}/status")
        data = response.json()
        assert "status" in data
        assert "paper_count" in data
        assert "contradiction_count" in data

    def test_ready_status_is_ready(self, client) -> None:
        with patch("app.api.routes.topics._get_topic", new_callable=AsyncMock,
                   return_value=_ready_doc()):
            response = client.get(f"/api/v1/topics/{_TOPIC_ID}/status")
        assert response.json()["status"] == "ready"

    def test_paper_count_matches_mongo(self, client) -> None:
        with patch("app.api.routes.topics._get_topic", new_callable=AsyncMock,
                   return_value=_ready_doc()):
            response = client.get(f"/api/v1/topics/{_TOPIC_ID}/status")
        assert response.json()["paper_count"] == 8

    def test_contradiction_count_matches_mongo(self, client) -> None:
        with patch("app.api.routes.topics._get_topic", new_callable=AsyncMock,
                   return_value=_ready_doc()):
            response = client.get(f"/api/v1/topics/{_TOPIC_ID}/status")
        assert response.json()["contradiction_count"] == 2

    def test_pending_status_polls_celery(self, client) -> None:
        """When MongoDB says pending, Celery backend is consulted."""
        mock_async_result = MagicMock()
        mock_async_result.state = "STARTED"

        with patch("app.api.routes.topics._get_topic", new_callable=AsyncMock,
                   return_value=_pending_doc()), \
             patch("app.api.routes.topics._upsert_topic", new_callable=AsyncMock), \
             patch("celery.result.AsyncResult", return_value=mock_async_result):
            response = client.get(f"/api/v1/topics/{_TOPIC_ID}/status")

        assert response.json()["status"] == "ingesting"

    def test_celery_success_updates_status_to_ready(self, client) -> None:
        """If Celery says SUCCESS but Mongo says pending, route updates to ready."""
        mock_async_result = MagicMock()
        mock_async_result.state = "SUCCESS"
        mock_async_result.result = {"paper_count": 5, "contradiction_count": 1}

        updated = {}

        async def _mock_upsert(tid, doc):
            updated.update(doc)

        with patch("app.api.routes.topics._get_topic", new_callable=AsyncMock,
                   return_value=_pending_doc()), \
             patch("app.api.routes.topics._upsert_topic", side_effect=_mock_upsert), \
             patch("celery.result.AsyncResult", return_value=mock_async_result):
            response = client.get(f"/api/v1/topics/{_TOPIC_ID}/status")

        assert updated.get("status") == "ready"

    def test_failed_status_returned_directly_from_mongo(self, client) -> None:
        doc = {**_pending_doc(), "status": "failed"}
        with patch("app.api.routes.topics._get_topic", new_callable=AsyncMock,
                   return_value=doc):
            response = client.get(f"/api/v1/topics/{_TOPIC_ID}/status")
        assert response.json()["status"] == "failed"


# ---------------------------------------------------------------------------
# TestGetTopicSuggestions
# ---------------------------------------------------------------------------

class TestGetTopicSuggestions:
    """GET /api/v1/topics/{id}/suggestions."""

    def test_returns_200_when_ready(self, client) -> None:
        with patch("app.api.routes.topics._get_topic", new_callable=AsyncMock,
                   return_value=_ready_doc()), \
             patch("app.services.reformulate.suggest_claims",
                   return_value=["Claim A.", "Claim B.", "Claim C."]):
            response = client.get(f"/api/v1/topics/{_TOPIC_ID}/suggestions")
        assert response.status_code == 200

    def test_returns_409_when_pending(self, client) -> None:
        with patch("app.api.routes.topics._get_topic", new_callable=AsyncMock,
                   return_value=_pending_doc()):
            response = client.get(f"/api/v1/topics/{_TOPIC_ID}/suggestions")
        assert response.status_code == 409

    def test_returns_409_when_ingesting(self, client) -> None:
        doc = {**_pending_doc(), "status": "ingesting"}
        with patch("app.api.routes.topics._get_topic", new_callable=AsyncMock,
                   return_value=doc):
            response = client.get(f"/api/v1/topics/{_TOPIC_ID}/suggestions")
        assert response.status_code == 409

    def test_returns_409_when_failed(self, client) -> None:
        doc = {**_pending_doc(), "status": "failed"}
        with patch("app.api.routes.topics._get_topic", new_callable=AsyncMock,
                   return_value=doc):
            response = client.get(f"/api/v1/topics/{_TOPIC_ID}/suggestions")
        assert response.status_code == 409

    def test_returns_404_for_missing_topic(self, client) -> None:
        with patch("app.api.routes.topics._get_topic", new_callable=AsyncMock,
                   return_value={}):
            response = client.get(f"/api/v1/topics/doesnotexist/suggestions")
        assert response.status_code == 404

    def test_suggestions_is_list(self, client) -> None:
        with patch("app.api.routes.topics._get_topic", new_callable=AsyncMock,
                   return_value=_ready_doc()), \
             patch("app.services.reformulate.suggest_claims",
                   return_value=["Claim A.", "Claim B.", "Claim C."]):
            response = client.get(f"/api/v1/topics/{_TOPIC_ID}/suggestions")
        assert isinstance(response.json()["suggestions"], list)

    def test_suggestions_count_is_3_to_4(self, client) -> None:
        with patch("app.api.routes.topics._get_topic", new_callable=AsyncMock,
                   return_value=_ready_doc()), \
             patch("app.services.reformulate.suggest_claims",
                   return_value=["Claim A.", "Claim B.", "Claim C.", "Claim D."]):
            response = client.get(f"/api/v1/topics/{_TOPIC_ID}/suggestions")
        suggestions = response.json()["suggestions"]
        assert 3 <= len(suggestions) <= 4

    def test_suggestions_are_strings(self, client) -> None:
        with patch("app.api.routes.topics._get_topic", new_callable=AsyncMock,
                   return_value=_ready_doc()), \
             patch("app.services.reformulate.suggest_claims",
                   return_value=["Intermittent fasting reduces insulin resistance.",
                                  "Time-restricted eating improves metabolic markers.",
                                  "Fasting protocols decrease HbA1c in type-2 diabetes."]):
            response = client.get(f"/api/v1/topics/{_TOPIC_ID}/suggestions")
        for s in response.json()["suggestions"]:
            assert isinstance(s, str) and s.strip()

    def test_topic_id_echoed_in_response(self, client) -> None:
        with patch("app.api.routes.topics._get_topic", new_callable=AsyncMock,
                   return_value=_ready_doc()), \
             patch("app.services.reformulate.suggest_claims",
                   return_value=["Claim A."]):
            response = client.get(f"/api/v1/topics/{_TOPIC_ID}/suggestions")
        assert response.json()["topic_id"] == _TOPIC_ID


# ---------------------------------------------------------------------------
# TestIngestTopicTask
# ---------------------------------------------------------------------------

class TestIngestTopicTask:
    """workers.tasks.ingest_topic — Celery task logic (no live services)."""

    def _make_self(self, retries: int = 0) -> MagicMock:
        mock_self = MagicMock()
        mock_self.request.id = "test-task-id"
        mock_self.request.retries = retries
        mock_self.max_retries = 3
        return mock_self

    def _run(self, mock_self, papers=None, contradict_count=0):
        """
        Call ingest_topic.run() directly (bypasses Celery dispatch machinery).
        All external I/O is patched.
        """
        if papers is None:
            papers = [
                {"doi": "10.1/a", "title": "Paper A", "abstract": "Abstract A.",
                 "pdf_url": None, "source": "test", "venue": "J", "citation_count": 10},
                {"doi": "10.1/b", "title": "Paper B", "abstract": "Abstract B.",
                 "pdf_url": None, "source": "test", "venue": "J", "citation_count": 5},
            ]

        mongo_calls: list[dict] = []

        def _capture_update(tid, doc):
            mongo_calls.append(doc.copy())

        mock_pairs = [MagicMock()] * contradict_count

        with patch("workers.tasks._mongo_update_status", side_effect=_capture_update), \
             patch("workers.tasks._ingest_papers_sync", return_value=["p1", "p2"]), \
             patch("workers.tasks.asyncio.run", return_value=papers), \
             patch("workers.tasks.check_contradictions", return_value=mock_pairs), \
             patch("workers.tasks.classify_paper_set",
                   return_value={"support": [], "refute": [], "no_stance": []}):

            from workers.tasks import ingest_topic
            result = ingest_topic.__wrapped__.__func__(mock_self, _TOPIC_ID, _TOPIC_TEXT)

        return result, mongo_calls

    def test_task_returns_dict_with_paper_count(self) -> None:
        result, _ = self._run(self._make_self())
        assert "paper_count" in result

    def test_task_returns_dict_with_contradiction_count(self) -> None:
        result, _ = self._run(self._make_self())
        assert "contradiction_count" in result

    def test_paper_count_is_non_negative_int(self) -> None:
        result, _ = self._run(self._make_self())
        assert isinstance(result["paper_count"], int)
        assert result["paper_count"] >= 0

    def test_contradiction_count_is_non_negative_int(self) -> None:
        result, _ = self._run(self._make_self())
        assert isinstance(result["contradiction_count"], int)
        assert result["contradiction_count"] >= 0

    def test_status_set_to_ingesting_first(self) -> None:
        _, calls = self._run(self._make_self())
        statuses = [c.get("status") for c in calls if "status" in c]
        assert "ingesting" in statuses
        # ingesting appears before ready
        assert statuses.index("ingesting") < statuses.index("ready")

    def test_status_set_to_ready_last(self) -> None:
        _, calls = self._run(self._make_self())
        statuses = [c.get("status") for c in calls if "status" in c]
        assert statuses[-1] == "ready"

    def test_retry_called_on_exception(self) -> None:
        """When an unhandled exception occurs, self.retry() must be called."""
        mock_self = self._make_self(retries=0)
        mock_self.retry.side_effect = Exception("retry triggered")

        with patch("workers.tasks._mongo_update_status",
                   side_effect=RuntimeError("mongo down")):
            from workers.tasks import ingest_topic
            with pytest.raises(Exception):
                ingest_topic.__wrapped__.__func__(mock_self, _TOPIC_ID, _TOPIC_TEXT)

        mock_self.retry.assert_called_once()

    def test_status_set_to_failed_on_final_retry(self) -> None:
        """On final retry (retries==max_retries), status must become 'failed'."""
        mongo_calls: list[dict] = []

        mock_self = self._make_self(retries=3)   # == max_retries
        mock_self.retry.side_effect = RuntimeError("boom")

        call_count = {"n": 0}

        def _side_effect(tid, doc):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First call (status=ingesting) succeeds
                mongo_calls.append(doc.copy())
            else:
                # Subsequent calls also captured, then raise to simulate DB issue
                mongo_calls.append(doc.copy())

        first_call = True

        def _patched_update(tid, doc):
            nonlocal first_call
            mongo_calls.append(doc.copy())
            if doc.get("status") == "ingesting":
                first_call = False
                raise RuntimeError("forced failure after ingesting")

        with patch("workers.tasks._mongo_update_status",
                   side_effect=_patched_update):
            from workers.tasks import ingest_topic
            with pytest.raises(RuntimeError):
                ingest_topic.__wrapped__.__func__(mock_self, _TOPIC_ID, _TOPIC_TEXT)

        statuses = [c.get("status") for c in mongo_calls if "status" in c]
        assert "failed" in statuses
