"""Tests for bbox_verify.py — 3-pass word_id cluster verifier.

All tests use mocked vertex_client (no live API calls).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from docile.dataset import BBox

from beat_docile.bbox_verify import (
    BboxVerification,
    build_candidate_clusters,
    verify_bbox,
)
from beat_docile.data import WordBox

# ── Helpers ────────────────────────────────────────────────────────────────────


def make_word(id: int, text: str, left: float, top: float) -> WordBox:
    return WordBox(id=id, text=text, bbox=(left, top, left + 0.05, top + 0.02), page=0)


def make_page(texts: list[str], row_y: float = 0.1) -> list[WordBox]:
    """Single-row page with words evenly spaced."""
    return [make_word(i, text, 0.06 * i, row_y) for i, text in enumerate(texts)]


def make_mock_client(reply: str) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=reply)]
    client = MagicMock()
    client.messages.create.return_value = msg
    return client


# ── build_candidate_clusters ───────────────────────────────────────────────────


class TestBuildCandidateClusters:
    def test_proposed_is_first(self):
        words = make_page([f"w{i}" for i in range(10)])
        proposed = [3, 4]
        candidates = build_candidate_clusters(proposed, words)
        assert candidates[0] == proposed

    def test_no_duplicates(self):
        words = make_page([f"w{i}" for i in range(20)])
        proposed = [5, 6]
        candidates = build_candidate_clusters(proposed, words, expansion_radius=3)
        keys = [tuple(c) for c in candidates]
        assert len(keys) == len(set(keys)), "Duplicate clusters detected"

    def test_reasonable_count_on_20_word_page(self):
        words = make_page([f"w{i}" for i in range(20)])
        proposed = [5, 6]
        candidates = build_candidate_clusters(proposed, words, expansion_radius=3)
        # Proposed + expanded + row + sliding windows; deduplicated
        assert len(candidates) >= 5
        assert len(candidates) <= 50

    def test_invalid_proposed_ids_filtered(self):
        words = make_page([f"w{i}" for i in range(5)])
        proposed = [2, 99]  # 99 is not a valid id
        candidates = build_candidate_clusters(proposed, words)
        assert len(candidates) >= 1
        # 99 must not appear in any cluster
        for c in candidates:
            assert 99 not in c

    def test_empty_words_returns_proposed(self):
        candidates = build_candidate_clusters([0, 1], [], expansion_radius=3)
        assert candidates == [[0, 1]]

    def test_empty_proposed_returns_empty(self):
        words = make_page(["a", "b", "c"])
        candidates = build_candidate_clusters([], words)
        assert candidates == []

    def test_same_row_cluster_included(self):
        # Row 0 (y=0.1): words 0-4; Row 1 (y=0.5): words 5-9
        row0 = [make_word(i, f"r0w{i}", 0.1 * i, 0.10) for i in range(5)]
        row1 = [make_word(i + 5, f"r1w{i}", 0.1 * i, 0.50) for i in range(5)]
        words = row0 + row1
        proposed = [2]  # row 0
        candidates = build_candidate_clusters(proposed, words, expansion_radius=1)
        row0_ids = {0, 1, 2, 3, 4}
        # At least one candidate should contain all row-0 words
        assert any(row0_ids.issubset(set(c)) for c in candidates)

    def test_single_word_proposed(self):
        words = make_page([f"w{i}" for i in range(5)])
        proposed = [2]
        candidates = build_candidate_clusters(proposed, words, expansion_radius=2)
        assert candidates[0] == [2]
        assert len(candidates) > 1

    def test_all_proposed_invalid_returns_empty(self):
        words = make_page(["a", "b"])
        candidates = build_candidate_clusters([99, 100], words)
        assert candidates == []


# ── verify_bbox ────────────────────────────────────────────────────────────────


class TestVerifyBbox:
    def test_correct_proposed_not_corrected(self):
        words = make_page(["Invoice", "No:", "INV-001", "Date:", "2024-01-15"])
        verification = verify_bbox(
            "date_issue", [4], "2024-01-15", words,
            vertex_client=None, use_llm_fallback=False,
        )
        assert not verification.corrected
        assert verification.confidence > 0.3
        assert 4 in verification.word_ids

    def test_wrong_proposed_corrected(self):
        words = make_page(["Invoice", "No:", "INV-001", "Date:", "2024-01-15"])
        # Proposed is "Invoice No:" but the value is "2024-01-15"
        verification = verify_bbox(
            "date_issue", [0, 1], "2024-01-15", words,
            vertex_client=None, use_llm_fallback=False,
        )
        assert verification.corrected
        assert 4 in verification.word_ids  # "2024-01-15" is word id 4

    def test_all_poor_returns_original_unchanged(self):
        words = make_page(["cat", "dog", "bird", "fish"])
        proposed = [0, 1]
        verification = verify_bbox(
            "document_id", proposed, "XXXXXX", words,
            vertex_client=None, use_llm_fallback=False,
        )
        assert not verification.corrected
        assert verification.word_ids == proposed

    def test_llm_fallback_called_when_all_scores_low(self):
        # Words that don't lexically match the extracted text at all
        words = make_page(["cat", "dog", "bird", "fish", "whale"])
        mock_client = make_mock_client("2")

        verify_bbox(
            "document_id", [0], "XXXXXX", words,
            vertex_client=mock_client, use_llm_fallback=True,
        )

        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args.kwargs
        messages = call_kwargs["messages"]
        prompt_text = messages[0]["content"]
        assert "document_id" in prompt_text
        assert "XXXXXX" in prompt_text

    def test_llm_fallback_prompt_lists_top5_clusters(self):
        words = make_page([f"w{i}" for i in range(10)])
        mock_client = make_mock_client("0")

        verify_bbox(
            "vendor_name", [0], "XXXXXX", words,
            vertex_client=mock_client, use_llm_fallback=True,
        )

        prompt_text = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
        # At least cluster "0:" must appear in the prompt
        assert "0:" in prompt_text

    def test_llm_fallback_not_called_when_scores_good(self):
        words = make_page(["Invoice", "No:", "INV-001", "Date:", "2024-01-15"])
        mock_client = make_mock_client("0")

        verify_bbox(
            "date_issue", [4], "2024-01-15", words,
            vertex_client=mock_client, use_llm_fallback=True,
        )

        mock_client.messages.create.assert_not_called()

    def test_llm_exception_handled_gracefully(self):
        words = make_page(["cat", "dog", "bird"])
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("API failure")

        verification = verify_bbox(
            "document_id", [0, 1], "XXXXXX", words,
            vertex_client=mock_client, use_llm_fallback=True,
        )

        # LLM failed but scores are all poor → defensive path → original unchanged
        assert isinstance(verification, BboxVerification)
        assert isinstance(verification.bbox, BBox)
        assert not verification.corrected
        assert verification.word_ids == [0, 1]

    def test_returns_valid_bbox_coords(self):
        words = make_page(["Invoice", "2024-01-15", "EUR", "100.00"])
        verification = verify_bbox(
            "date_issue", [1], "2024-01-15", words,
            vertex_client=None, use_llm_fallback=False,
        )
        bbox = verification.bbox
        assert bbox.left <= bbox.right
        assert bbox.top <= bbox.bottom

    def test_empty_proposed_returns_fallback(self):
        words = make_page(["a", "b", "c"])
        verification = verify_bbox(
            "document_id", [], "test", words,
            vertex_client=None, use_llm_fallback=False,
        )
        assert isinstance(verification, BboxVerification)
        assert not verification.corrected

    def test_empty_extracted_text_returns_fallback(self):
        words = make_page(["a", "b", "c"])
        verification = verify_bbox(
            "document_id", [0], "   ", words,
            vertex_client=None, use_llm_fallback=False,
        )
        assert isinstance(verification, BboxVerification)
        assert not verification.corrected

    def test_outer_exception_returns_proposed_unchanged(self):
        mock_client = MagicMock()

        # Corrupt the words list mid-call via a broken WordBox
        bad_word = WordBox(id=0, text="a", bbox=None, page=0)  # type: ignore[arg-type]
        try:
            verification = verify_bbox(
                "document_id", [0], "a", [bad_word],
                vertex_client=mock_client, use_llm_fallback=False,
            )
            # If it doesn't raise, result must be a valid BboxVerification
            assert isinstance(verification, BboxVerification)
        except Exception:
            pytest.fail("verify_bbox raised instead of returning fallback")
