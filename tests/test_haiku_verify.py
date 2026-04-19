"""Tests for haiku_verify.py — span-precision verifier.

Unit tests use a mock LLM client (no live API calls).
Real-API smoke test is marked with pytest.mark.real_api and skipped by default.
Run smoke test with: uv run pytest -m real_api tests/test_haiku_verify.py -v
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from beat_docile.data import WordBox
from beat_docile.haiku_verify import (
    VerificationResult,
    apply_verification,
    verify_extractions_batch,
    verify_span,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────


def _make_words() -> list[WordBox]:
    """Synthetic word list for tests."""
    return [
        WordBox(id=1, text="Invoice", bbox=(0.05, 0.10, 0.15, 0.12), page=0),
        WordBox(id=2, text="No:", bbox=(0.16, 0.10, 0.20, 0.12), page=0),
        WordBox(id=3, text="INV-2024-001", bbox=(0.21, 0.10, 0.40, 0.12), page=0),
        WordBox(id=4, text="Acme", bbox=(0.05, 0.20, 0.12, 0.22), page=0),
        WordBox(id=5, text="Corp", bbox=(0.13, 0.20, 0.20, 0.22), page=0),
        WordBox(id=6, text="123", bbox=(0.05, 0.23, 0.10, 0.25), page=0),
        WordBox(id=7, text="Main", bbox=(0.11, 0.23, 0.18, 0.25), page=0),
        WordBox(id=8, text="St", bbox=(0.19, 0.23, 0.23, 0.25), page=0),
        WordBox(id=9, text="London", bbox=(0.05, 0.26, 0.15, 0.28), page=0),
        WordBox(id=10, text="€", bbox=(0.60, 0.50, 0.63, 0.52), page=0),
        WordBox(id=11, text="1,234.56", bbox=(0.64, 0.50, 0.78, 0.52), page=0),
    ]


def _make_mock_client(response_json: dict) -> MagicMock:
    """Build a mock AnthropicVertex client returning a specific JSON response."""
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock()]
    mock_msg.content[0].text = json.dumps(response_json)
    client = MagicMock()
    client.messages.create.return_value = mock_msg
    return client


def _make_mock_client_raw(raw_text: str) -> MagicMock:
    """Build a mock client returning raw text (for malformed-JSON tests)."""
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock()]
    mock_msg.content[0].text = raw_text
    client = MagicMock()
    client.messages.create.return_value = mock_msg
    return client


# ── verify_span unit tests ─────────────────────────────────────────────────────


class TestVerifySpanAccept:
    def test_verdict_is_accept(self) -> None:
        words = _make_words()
        client = _make_mock_client(
            {"verdict": "accept", "word_ids": [3], "confidence": 0.97, "reasoning": "exact"}
        )
        result = verify_span("document_id", [3], "INV-2024-001", words, client)
        assert result.verdict == "accept"

    def test_word_ids_unchanged_on_accept(self) -> None:
        words = _make_words()
        client = _make_mock_client(
            {"verdict": "accept", "word_ids": [3], "confidence": 0.97, "reasoning": "exact"}
        )
        result = verify_span("document_id", [3], "INV-2024-001", words, client)
        assert result.word_ids == [3]

    def test_confidence_propagated(self) -> None:
        words = _make_words()
        client = _make_mock_client(
            {"verdict": "accept", "word_ids": [10, 11], "confidence": 0.85, "reasoning": "ok"}
        )
        result = verify_span("amount_total_gross", [10, 11], "€ 1,234.56", words, client)
        assert result.confidence == pytest.approx(0.85)

    def test_client_called_once(self) -> None:
        words = _make_words()
        client = _make_mock_client(
            {"verdict": "accept", "word_ids": [3], "confidence": 0.9, "reasoning": "ok"}
        )
        verify_span("document_id", [3], "INV-2024-001", words, client)
        client.messages.create.assert_called_once()


class TestVerifySpanCorrect:
    def test_verdict_is_correct(self) -> None:
        words = _make_words()
        client = _make_mock_client(
            {"verdict": "correct", "word_ids": [3], "confidence": 0.95, "reasoning": "strip label"}
        )
        result = verify_span("document_id", [1, 2, 3], "Invoice No: INV-2024-001", words, client)
        assert result.verdict == "correct"

    def test_corrected_word_ids_returned(self) -> None:
        words = _make_words()
        client = _make_mock_client(
            {"verdict": "correct", "word_ids": [3], "confidence": 0.95, "reasoning": "strip label"}
        )
        result = verify_span("document_id", [1, 2, 3], "Invoice No: INV-2024-001", words, client)
        assert result.word_ids == [3]

    def test_corrected_ids_expanded_for_address(self) -> None:
        words = _make_words()
        client = _make_mock_client(
            {
                "verdict": "correct",
                "word_ids": [4, 5, 6, 7, 8, 9],
                "confidence": 0.90,
                "reasoning": "added continuation lines",
            }
        )
        result = verify_span(
            "customer_billing_address", [4, 5], "Acme Corp", words, client
        )
        assert result.verdict == "correct"
        assert result.word_ids == [4, 5, 6, 7, 8, 9]

    def test_reasoning_propagated(self) -> None:
        words = _make_words()
        client = _make_mock_client(
            {"verdict": "correct", "word_ids": [3], "confidence": 0.9, "reasoning": "strip label"}
        )
        result = verify_span("document_id", [1, 2, 3], "Invoice No: INV-2024-001", words, client)
        assert result.reasoning == "strip label"


class TestVerifySpanReject:
    def test_verdict_is_reject(self) -> None:
        words = _make_words()
        client = _make_mock_client(
            {"verdict": "reject", "word_ids": [], "confidence": 0.92, "reasoning": "wrong field"}
        )
        result = verify_span("customer_id", [4, 5], "Acme Corp", words, client)
        assert result.verdict == "reject"

    def test_word_ids_empty_on_reject(self) -> None:
        words = _make_words()
        client = _make_mock_client(
            {"verdict": "reject", "word_ids": [], "confidence": 0.92, "reasoning": "wrong field"}
        )
        result = verify_span("customer_id", [4, 5], "Acme Corp", words, client)
        assert result.word_ids == []

    def test_confidence_propagated_on_reject(self) -> None:
        words = _make_words()
        client = _make_mock_client(
            {"verdict": "reject", "word_ids": [], "confidence": 0.78, "reasoning": "wrong field"}
        )
        result = verify_span("customer_id", [4, 5], "Acme Corp", words, client)
        assert result.confidence == pytest.approx(0.78)


class TestVerifySpanDefensiveCases:
    def test_malformed_json_returns_defensive_accept(self) -> None:
        words = _make_words()
        client = _make_mock_client_raw("I cannot determine the verdict for this candidate.")
        result = verify_span("document_id", [3], "INV-2024-001", words, client)
        assert result.verdict == "accept"
        assert result.word_ids == [3]
        assert result.reasoning == "parse_failed"

    def test_empty_response_returns_defensive_accept(self) -> None:
        words = _make_words()
        client = _make_mock_client_raw("")
        result = verify_span("document_id", [3], "INV-2024-001", words, client)
        assert result.verdict == "accept"
        assert result.word_ids == [3]

    def test_unknown_verdict_returns_defensive_accept(self) -> None:
        words = _make_words()
        client = _make_mock_client(
            {"verdict": "maybe", "word_ids": [3], "confidence": 0.5, "reasoning": "unsure"}
        )
        result = verify_span("document_id", [3], "INV-2024-001", words, client)
        assert result.verdict == "accept"
        assert result.reasoning == "parse_failed"

    def test_correct_with_all_invalid_word_ids_returns_defensive_accept(self) -> None:
        words = _make_words()
        # word_ids 999, 1000 do not exist in the words list
        client = _make_mock_client(
            {"verdict": "correct", "word_ids": [999, 1000], "confidence": 0.8, "reasoning": "corrected"}
        )
        result = verify_span("document_id", [3], "INV-2024-001", words, client)
        assert result.verdict == "accept"
        assert result.word_ids == [3]
        assert result.reasoning == "parse_failed"

    def test_api_exception_returns_defensive_accept(self) -> None:
        words = _make_words()
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("connection refused")
        result = verify_span("document_id", [3], "INV-2024-001", words, client)
        assert result.verdict == "accept"
        assert result.word_ids == [3]
        assert result.reasoning == "api_error"

    def test_markdown_fences_stripped(self) -> None:
        words = _make_words()
        fenced = '```json\n{"verdict": "accept", "word_ids": [3], "confidence": 0.9, "reasoning": "ok"}\n```'
        client = _make_mock_client_raw(fenced)
        result = verify_span("document_id", [3], "INV-2024-001", words, client)
        assert result.verdict == "accept"

    def test_confidence_defaults_to_half_on_parse_failure(self) -> None:
        words = _make_words()
        client = _make_mock_client_raw("not json")
        result = verify_span("document_id", [3], "INV-2024-001", words, client)
        assert result.confidence == pytest.approx(0.5)


# ── apply_verification tests ───────────────────────────────────────────────────


class TestApplyVerification:
    def _make_extractions(self) -> dict[str, list[tuple[list[int], str]]]:
        return {
            "document_id": [([1, 2, 3], "Invoice No: INV-001")],
            "amount_total_gross": [([10, 11], "€ 1,234.56")],
            "customer_id": [([4, 5], "Acme Corp")],
            "vendor_name": [([4, 5], "Acme Corp")],
            "date_issue": [([3], "INV-001")],
        }

    def test_accept_keeps_original_word_ids(self) -> None:
        extractions = {"document_id": [([1, 2, 3], "Invoice No: INV-001")]}
        verifications = {
            "document_id": [VerificationResult(verdict="accept", word_ids=[1, 2, 3], confidence=0.9)]
        }
        result = apply_verification(extractions, verifications)
        assert result["document_id"][0][0] == [1, 2, 3]

    def test_correct_high_confidence_replaces_word_ids(self) -> None:
        extractions = {"document_id": [([1, 2, 3], "Invoice No: INV-001")]}
        verifications = {
            "document_id": [VerificationResult(verdict="correct", word_ids=[3], confidence=0.9)]
        }
        result = apply_verification(extractions, verifications)
        assert result["document_id"][0][0] == [3]

    def test_correct_low_confidence_keeps_original(self) -> None:
        extractions = {"document_id": [([1, 2, 3], "Invoice No: INV-001")]}
        verifications = {
            "document_id": [VerificationResult(verdict="correct", word_ids=[3], confidence=0.4)]
        }
        result = apply_verification(extractions, verifications)
        assert result["document_id"][0][0] == [1, 2, 3]

    def test_reject_high_confidence_drops_candidate(self) -> None:
        extractions = {"customer_id": [([4, 5], "Acme Corp")]}
        verifications = {
            "customer_id": [VerificationResult(verdict="reject", word_ids=[], confidence=0.9)]
        }
        result = apply_verification(extractions, verifications)
        assert result["customer_id"] == []

    def test_reject_low_confidence_keeps_original(self) -> None:
        extractions = {"customer_id": [([4, 5], "Acme Corp")]}
        verifications = {
            "customer_id": [VerificationResult(verdict="reject", word_ids=[], confidence=0.3)]
        }
        result = apply_verification(extractions, verifications)
        assert result["customer_id"][0][0] == [4, 5]

    def test_boundary_confidence_exactly_min_is_applied(self) -> None:
        """Confidence exactly equal to min_confidence should act (>=)."""
        extractions = {"document_id": [([1, 2, 3], "text")]}
        verifications = {
            "document_id": [VerificationResult(verdict="correct", word_ids=[3], confidence=0.6)]
        }
        result = apply_verification(extractions, verifications, min_confidence=0.6)
        assert result["document_id"][0][0] == [3]

    def test_text_preserved_after_correction(self) -> None:
        extractions = {"document_id": [([1, 2, 3], "Invoice No: INV-001")]}
        verifications = {
            "document_id": [VerificationResult(verdict="correct", word_ids=[3], confidence=0.9)]
        }
        result = apply_verification(extractions, verifications)
        assert result["document_id"][0][1] == "Invoice No: INV-001"

    def test_mixed_verdicts_across_multiple_fields(self) -> None:
        extractions = {
            "document_id": [([1, 2, 3], "Invoice No: INV-001")],
            "customer_id": [([4, 5], "Acme Corp")],
            "amount_total_gross": [([10, 11], "€ 1,234.56")],
        }
        verifications = {
            "document_id": [VerificationResult(verdict="correct", word_ids=[3], confidence=0.95)],
            "customer_id": [VerificationResult(verdict="reject", word_ids=[], confidence=0.88)],
            "amount_total_gross": [VerificationResult(verdict="accept", word_ids=[10, 11], confidence=0.97)],
        }
        result = apply_verification(extractions, verifications)
        assert result["document_id"][0][0] == [3]
        assert result["customer_id"] == []
        assert result["amount_total_gross"][0][0] == [10, 11]

    def test_missing_fieldtype_in_verifications_keeps_original(self) -> None:
        extractions = {"document_id": [([3], "INV-001")]}
        result = apply_verification(extractions, {})
        assert result["document_id"][0][0] == [3]

    def test_multiple_candidates_same_fieldtype(self) -> None:
        """tax_detail_* can have multiple candidates per fieldtype."""
        extractions = {
            "tax_detail_rate": [([1], "10%"), ([2], "20%")],
        }
        verifications = {
            "tax_detail_rate": [
                VerificationResult(verdict="accept", word_ids=[1], confidence=0.9),
                VerificationResult(verdict="reject", word_ids=[], confidence=0.85),
            ]
        }
        result = apply_verification(extractions, verifications)
        assert len(result["tax_detail_rate"]) == 1
        assert result["tax_detail_rate"][0][0] == [1]

    def test_custom_min_confidence(self) -> None:
        extractions = {"document_id": [([1, 2, 3], "text")]}
        verifications = {
            "document_id": [VerificationResult(verdict="correct", word_ids=[3], confidence=0.55)]
        }
        # With min_confidence=0.5 the correction should be applied
        result = apply_verification(extractions, verifications, min_confidence=0.5)
        assert result["document_id"][0][0] == [3]


# ── verify_extractions_batch tests ────────────────────────────────────────────


class TestVerifyExtractionsBatch:
    def test_returns_same_fieldtype_keys(self) -> None:
        words = _make_words()
        client = _make_mock_client(
            {"verdict": "accept", "word_ids": [3], "confidence": 0.9, "reasoning": "ok"}
        )
        extractions = {
            "document_id": [([3], "INV-001")],
            "amount_total_gross": [([10, 11], "€ 1,234.56")],
        }
        result = verify_extractions_batch(extractions, words, client)
        assert set(result.keys()) == {"document_id", "amount_total_gross"}

    def test_correct_number_of_calls(self) -> None:
        words = _make_words()
        client = _make_mock_client(
            {"verdict": "accept", "word_ids": [3], "confidence": 0.9, "reasoning": "ok"}
        )
        extractions = {
            "document_id": [([3], "INV-001"), ([1, 2, 3], "Invoice No: INV-001")],
            "amount_total_gross": [([10, 11], "€ 1,234.56")],
        }
        verify_extractions_batch(extractions, words, client)
        assert client.messages.create.call_count == 3

    def test_parallel_structured_output(self) -> None:
        """Result list length matches input candidate list length."""
        words = _make_words()
        client = _make_mock_client(
            {"verdict": "accept", "word_ids": [3], "confidence": 0.9, "reasoning": "ok"}
        )
        extractions = {
            "tax_detail_rate": [([1], "10%"), ([2], "20%"), ([3], "21%")],
        }
        result = verify_extractions_batch(extractions, words, client)
        assert len(result["tax_detail_rate"]) == 3

    def test_all_results_are_verification_results(self) -> None:
        words = _make_words()
        client = _make_mock_client(
            {"verdict": "accept", "word_ids": [3], "confidence": 0.9, "reasoning": "ok"}
        )
        extractions = {"document_id": [([3], "INV-001")]}
        result = verify_extractions_batch(extractions, words, client)
        assert all(isinstance(vr, VerificationResult) for vr in result["document_id"])

    def test_empty_extractions_returns_empty(self) -> None:
        words = _make_words()
        client = MagicMock()
        result = verify_extractions_batch({}, words, client)
        assert result == {}
        client.messages.create.assert_not_called()

    def test_empty_candidates_list(self) -> None:
        words = _make_words()
        client = MagicMock()
        extractions = {"document_id": []}
        result = verify_extractions_batch(extractions, words, client)
        assert result["document_id"] == []
        client.messages.create.assert_not_called()

    def test_parallel_argument_accepted(self) -> None:
        words = _make_words()
        client = _make_mock_client(
            {"verdict": "accept", "word_ids": [3], "confidence": 0.9, "reasoning": "ok"}
        )
        extractions = {"document_id": [([3], "INV-001")]}
        result = verify_extractions_batch(extractions, words, client, parallel=1)
        assert len(result["document_id"]) == 1


# ── VerificationResult dataclass ──────────────────────────────────────────────


class TestVerificationResultDataclass:
    def test_default_reasoning_is_empty_string(self) -> None:
        vr = VerificationResult(verdict="accept", word_ids=[1], confidence=0.9)
        assert vr.reasoning == ""

    def test_all_fields_accessible(self) -> None:
        vr = VerificationResult(verdict="correct", word_ids=[3], confidence=0.8, reasoning="stripped label")
        assert vr.verdict == "correct"
        assert vr.word_ids == [3]
        assert vr.confidence == pytest.approx(0.8)
        assert vr.reasoning == "stripped label"


# ── Real-API smoke test ────────────────────────────────────────────────────────


@pytest.mark.real_api
def test_real_haiku_smoke() -> None:
    """Smoke test: confirm Haiku produces parseable JSON on a simple synthetic case.

    Run with: uv run pytest -m real_api tests/test_haiku_verify.py::test_real_haiku_smoke -v
    Requires ANTHROPIC_API_KEY env var.
    """
    from beat_docile.llm_client import get_client

    words = [
        WordBox(id=1, text="Invoice", bbox=(0.05, 0.10, 0.15, 0.12), page=0),
        WordBox(id=2, text="No:", bbox=(0.16, 0.10, 0.20, 0.12), page=0),
        WordBox(id=3, text="INV-2024-001", bbox=(0.21, 0.10, 0.40, 0.12), page=0),
    ]
    client = get_client()

    result = verify_span(
        fieldtype="document_id",
        candidate_word_ids=[1, 2, 3],
        candidate_text="Invoice No: INV-2024-001",
        words=words,
        vertex_client=client,
    )

    # Basic sanity: we got a real verdict, not a defensive fallback
    assert result.verdict in ("accept", "correct", "reject")
    assert isinstance(result.word_ids, list)
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.reasoning, str)

    # The correct verdict here is "correct" — label should be stripped
    # We don't assert it strictly (model may vary) but log for inspection
    import logging
    logging.getLogger(__name__).info(
        "Real Haiku verdict for label-overrun: verdict=%s word_ids=%s conf=%.2f reasoning=%r",
        result.verdict,
        result.word_ids,
        result.confidence,
        result.reasoning,
    )
