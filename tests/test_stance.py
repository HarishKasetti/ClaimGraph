"""
tests/test_stance.py
---------------------
Unit tests for app/services/stance.py.

All tests are fully isolated -- no model is downloaded, no Qdrant or MongoDB
connection is made.  The three hard-coded NLI test pairs use manually-verified,
unambiguous examples:

  Pair 1: strong supporting passage  -> expected "support"
  Pair 2: strong refuting passage    -> expected "refute"
  Pair 3: historically neutral passage -> expected "no_stance"

The classify_paper_set tests mock _qdrant_scroll and _mongo_persist so only the
bucketing / aggregation logic is exercised.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Hard-coded NLI test pairs
# Pre-labelled BEFORE running any model. The expected labels are based on
# straightforward entailment / contradiction / neutrality.
# ---------------------------------------------------------------------------

CLAIM = "Aspirin reduces the risk of major cardiovascular events"

# Pair 1: Explicit meta-analytic confirmation -- clearly SUPPORT
PASSAGE_SUPPORT = (
    "A meta-analysis of 12 randomised controlled trials involving 97,456 "
    "participants confirmed that low-dose aspirin therapy significantly reduces "
    "the incidence of myocardial infarction and stroke by 22% in high-risk "
    "adults (RR 0.78, 95% CI 0.71-0.86, p<0.001), supporting its use as "
    "primary prevention in selected populations."
)

# Pair 2: Large RCT showing no benefit and increased harm -- clearly REFUTE
PASSAGE_REFUTE = (
    "The ASCEND trial (n=15,480) demonstrated that aspirin did not significantly "
    "reduce the composite of nonfatal MI, nonfatal stroke, or vascular death "
    "compared to placebo (8.5% vs 9.6%, RR 0.88 but 95% CI crossed 1.0), while "
    "major bleeding events were significantly higher in the aspirin group "
    "(4.1% vs 3.2%), indicating net harm in primary prevention."
)

# Pair 3: Historical / synthesis fact unrelated to the claim -- clearly NO_STANCE
PASSAGE_NO_STANCE = (
    "Aspirin (acetylsalicylic acid) was first synthesised in its pure, stable "
    "form by Felix Hoffmann at Bayer in 1897.  The compound was named after "
    "Spiraea ulmaria, a plant whose extracts had long been used in folk medicine "
    "for pain relief, and it was commercially introduced in 1899."
)

# Expected labels for the three pairs
_HAND_LABELED: list[tuple[str, str]] = [
    (PASSAGE_SUPPORT,    "support"),
    (PASSAGE_REFUTE,     "refute"),
    (PASSAGE_NO_STANCE,  "no_stance"),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_pipe(label: str, score: float = 0.92):
    """Return a deterministic mock pipeline callable."""
    def _pipe(passage, candidate_labels, hypothesis_template, multi_label):  # noqa: ANN001
        # Return the pre-set label with the highest score
        other_labels = [l for l in candidate_labels if l != label]
        remaining = (1.0 - score) / max(len(other_labels), 1)
        scores = [score] + [remaining] * len(other_labels)
        labels = [label] + other_labels
        return {"labels": labels, "scores": scores}
    return _pipe


# ---------------------------------------------------------------------------
# TestClassifyStance – three hard-coded pairs
# ---------------------------------------------------------------------------

class TestClassifyStance:
    """
    Each test patches get_pipeline() with a deterministic callable that returns
    the expected label with high confidence.  This validates that classify_stance
    correctly extracts label and score from the pipeline output -- independent of
    any real model.
    """

    @pytest.mark.parametrize("passage,expected_label", _HAND_LABELED)
    def test_hard_labeled_pair(self, passage: str, expected_label: str) -> None:
        """
        The three manually-verified NLI pairs must return the correct label.
        The mock pipeline is pre-configured to return the expected label, which
        tests classify_stance's routing logic without the real model.
        """
        mock_pipe = _make_pipe(expected_label, score=0.92)

        with patch("app.services.stance.get_pipeline", return_value=mock_pipe):
            from app.services.stance import classify_stance

            result = classify_stance(CLAIM, passage)

        assert result["label"] == expected_label, (
            f"Expected {expected_label!r} for passage starting with "
            f"{passage[:80]!r}, got {result['label']!r}"
        )
        assert 0.0 <= result["score"] <= 1.0

    def test_classify_stance_returns_required_keys(self) -> None:
        """Result dict must always contain 'label' and 'score' keys."""
        mock_pipe = _make_pipe("support")
        with patch("app.services.stance.get_pipeline", return_value=mock_pipe):
            from app.services.stance import classify_stance
            result = classify_stance(CLAIM, PASSAGE_SUPPORT)

        assert "label" in result
        assert "score" in result

    def test_classify_stance_label_is_valid(self) -> None:
        """Label must be one of the three valid values."""
        valid_labels = {"support", "refute", "no_stance"}
        mock_pipe = _make_pipe("refute")
        with patch("app.services.stance.get_pipeline", return_value=mock_pipe):
            from app.services.stance import classify_stance
            result = classify_stance(CLAIM, PASSAGE_REFUTE)

        assert result["label"] in valid_labels

    def test_passage_capped_at_512_tokens(self) -> None:
        """
        Very long passages must be silently truncated to 512 whitespace-tokens
        before being handed to the pipeline.
        """
        long_passage = " ".join(["word"] * 1000)
        captured: list[str] = []

        def _capturing_pipe(passage, **_kwargs):  # noqa: ANN001
            captured.append(passage)
            return {"labels": ["no_stance", "support", "refute"],
                    "scores": [0.8, 0.1, 0.1]}

        with patch("app.services.stance.get_pipeline",
                   return_value=_capturing_pipe):
            from app.services.stance import classify_stance
            classify_stance(CLAIM, long_passage)

        assert len(captured) == 1
        actual_token_count = len(captured[0].split())
        assert actual_token_count <= 512, (
            f"Passage was not truncated: {actual_token_count} tokens sent to pipeline"
        )


# ---------------------------------------------------------------------------
# TestClassifyPaperSet – aggregation and bucketing logic
# ---------------------------------------------------------------------------

# Fake chunk payloads representing two papers × two chunks each
_FAKE_PAYLOADS = [
    {"paper_id": "paper_A", "text": PASSAGE_SUPPORT,   "section": "Results"},
    {"paper_id": "paper_A", "text": "A brief historical note about aspirin synthesis.", "section": "Intro"},
    {"paper_id": "paper_B", "text": PASSAGE_REFUTE,    "section": "Discussion"},
    {"paper_id": "paper_B", "text": "Aspirin was patented in 1900.",                   "section": "Intro"},
    {"paper_id": "paper_C", "text": PASSAGE_NO_STANCE, "section": "Background"},
]


class TestClassifyPaperSet:

    def _mock_classify(self, label_map: dict[str, str]):
        """
        Return a mock for classify_stance that deterministically returns a label
        based on the passage content mapped via label_map.
        """
        score_map = {"support": 0.95, "refute": 0.93, "no_stance": 0.85}

        def _classify(claim, passage):  # noqa: ANN001
            for key_text, lbl in label_map.items():
                if key_text in passage:
                    return {"label": lbl, "score": score_map[lbl]}
            return {"label": "no_stance", "score": 0.6}

        return _classify

    @pytest.mark.asyncio(loop_scope="function")
    async def test_buckets_contain_expected_papers(self) -> None:
        """
        With three papers classified as support / refute / no_stance respectively,
        classify_paper_set must place each paper in the correct bucket.
        """
        label_map = {
            PASSAGE_SUPPORT[:30]:   "support",
            PASSAGE_REFUTE[:30]:    "refute",
            PASSAGE_NO_STANCE[:30]: "no_stance",
        }

        with patch("app.services.stance._qdrant_scroll",
                   return_value=_FAKE_PAYLOADS), \
             patch("app.services.stance._mongo_persist",
                   new_callable=AsyncMock) as mock_persist, \
             patch("app.services.stance.classify_stance",
                   side_effect=self._mock_classify(label_map)):

            from app.services.stance import classify_paper_set
            buckets = await classify_paper_set(CLAIM, "topic_test_001")

        assert "paper_A" in buckets["support"]
        assert "paper_B" in buckets["refute"]
        assert "paper_C" in buckets["no_stance"]
        mock_persist.assert_awaited_once()

    @pytest.mark.asyncio(loop_scope="function")
    async def test_strongest_chunk_wins(self) -> None:
        """
        When paper_A has one support chunk (score 0.95) and one refute chunk
        (score 0.93), it must be placed in 'support' (highest score wins).
        """
        call_count = {"n": 0}

        def _alternating(claim, passage):  # noqa: ANN001
            call_count["n"] += 1
            # First call for paper_A -> support (0.95), second -> refute (0.93)
            if call_count["n"] == 1:
                return {"label": "support", "score": 0.95}
            return {"label": "refute", "score": 0.93}

        two_chunk_payloads = [
            {"paper_id": "paper_A", "text": "chunk one", "section": "Abstract"},
            {"paper_id": "paper_A", "text": "chunk two", "section": "Results"},
        ]

        with patch("app.services.stance._qdrant_scroll",
                   return_value=two_chunk_payloads), \
             patch("app.services.stance._mongo_persist",
                   new_callable=AsyncMock), \
             patch("app.services.stance.classify_stance",
                   side_effect=_alternating):

            from app.services.stance import classify_paper_set
            buckets = await classify_paper_set(CLAIM, "topic_strongest")

        assert "paper_A" in buckets["support"], (
            "Strongest chunk (support, 0.95) should win over refute (0.93)"
        )
        assert "paper_A" not in buckets["refute"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_empty_collection_returns_empty_buckets(self) -> None:
        """If Qdrant returns no chunks, all buckets must be empty lists."""
        with patch("app.services.stance._qdrant_scroll", return_value=[]), \
             patch("app.services.stance._mongo_persist", new_callable=AsyncMock):

            from app.services.stance import classify_paper_set
            buckets = await classify_paper_set(CLAIM, "topic_empty")

        assert buckets == {"support": [], "refute": [], "no_stance": []}

    @pytest.mark.asyncio(loop_scope="function")
    async def test_all_three_bucket_keys_always_present(self) -> None:
        """Result dict must always contain all three keys, even if some are empty."""
        payloads = [
            {"paper_id": "paper_X", "text": "supports the claim firmly", "section": "Results"},
        ]

        def _always_support(claim, passage):  # noqa: ANN001
            return {"label": "support", "score": 0.98}

        with patch("app.services.stance._qdrant_scroll", return_value=payloads), \
             patch("app.services.stance._mongo_persist", new_callable=AsyncMock), \
             patch("app.services.stance.classify_stance",
                   side_effect=_always_support):

            from app.services.stance import classify_paper_set
            buckets = await classify_paper_set(CLAIM, "topic_single")

        for key in ("support", "refute", "no_stance"):
            assert key in buckets, f"Missing bucket key: {key!r}"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_mongo_persist_called_with_correct_args(self) -> None:
        """_mongo_persist must receive the topic_id, claim, and buckets dict."""
        payloads = [
            {"paper_id": "pX", "text": "strong support passage here", "section": "Results"},
        ]

        def _support(claim, passage):  # noqa: ANN001
            return {"label": "support", "score": 0.99}

        with patch("app.services.stance._qdrant_scroll", return_value=payloads), \
             patch("app.services.stance._mongo_persist",
                   new_callable=AsyncMock) as mock_persist, \
             patch("app.services.stance.classify_stance", side_effect=_support):

            from app.services.stance import classify_paper_set
            await classify_paper_set(CLAIM, "topic_mongo_check")

        mock_persist.assert_awaited_once()
        call_kwargs = mock_persist.call_args
        # positional: topic_id, claim, buckets
        assert call_kwargs.args[0] == "topic_mongo_check"
        assert call_kwargs.args[1] == CLAIM
        assert isinstance(call_kwargs.args[2], dict)
