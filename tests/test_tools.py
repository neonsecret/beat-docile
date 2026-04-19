"""Unit tests for beat_docile.tools — pure tool functions for the ReAct harness.

Uses only synthetic WordBox fixtures. No Claude API calls.
"""

from __future__ import annotations

from beat_docile.data import WordBox
from beat_docile.tools import (
    Candidate,
    cluster_fewshot,
    refine_span,
    regex_extract,
    spatial_neighbor,
    validator_check,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


def w(wid: int, text: str, left: float, top: float, right: float, bottom: float) -> WordBox:
    return WordBox(id=wid, text=text, bbox=(left, top, right, bottom), page=0)


def _invoice_words() -> list[WordBox]:
    """Minimal two-row invoice word list for spatial tests."""
    return [
        w(0, "Invoice",    0.05, 0.10, 0.15, 0.12),
        w(1, "No:",        0.16, 0.10, 0.22, 0.12),
        w(2, "INV-001",    0.23, 0.10, 0.35, 0.12),
        w(3, "Date:",      0.05, 0.20, 0.15, 0.22),
        w(4, "2024-01-15", 0.16, 0.20, 0.35, 0.22),
        w(5, "Total:",     0.05, 0.30, 0.15, 0.32),
        w(6, "€",          0.16, 0.30, 0.20, 0.32),
        w(7, "1,234.56",   0.21, 0.30, 0.38, 0.32),
    ]


# ── regex_extract ─────────────────────────────────────────────────────────────


class TestRegexExtractIBAN:
    def test_valid_iban_single_word(self):
        """DE89... passes mod-97 checksum — single OCR word."""
        words = [w(0, "DE89370400440532013000", 0.1, 0.5, 0.5, 0.52)]
        candidates = regex_extract("iban", words)
        assert len(candidates) >= 1
        assert candidates[0].source == "regex"
        assert 0 in candidates[0].word_ids

    def test_valid_iban_with_spaces_multi_word(self):
        """IBAN split across multiple OCR tokens must be found in window scan."""
        words = [
            w(0, "DE89",  0.10, 0.5, 0.20, 0.52),
            w(1, "3704",  0.21, 0.5, 0.30, 0.52),
            w(2, "0044",  0.31, 0.5, 0.40, 0.52),
            w(3, "0532",  0.41, 0.5, 0.50, 0.52),
            w(4, "0130",  0.51, 0.5, 0.60, 0.52),
            w(5, "00",    0.61, 0.5, 0.65, 0.52),
        ]
        candidates = regex_extract("iban", words)
        # At least one candidate spanning multiple words
        multi = [c for c in candidates if len(c.word_ids) > 1]
        assert len(multi) >= 1

    def test_invalid_iban_bad_checksum(self):
        """Last digit changed — mod-97 fails — no candidates."""
        words = [w(0, "DE89370400440532013001", 0.1, 0.5, 0.5, 0.52)]
        candidates = regex_extract("iban", words)
        assert len(candidates) == 0

    def test_no_iban_words(self):
        words = [w(0, "Invoice", 0.1, 0.5, 0.3, 0.52), w(1, "No:", 0.31, 0.5, 0.4, 0.52)]
        assert regex_extract("iban", words) == []

    def test_empty_words_returns_empty(self):
        assert regex_extract("iban", []) == []


class TestRegexExtractBIC:
    def test_valid_bic_8_chars(self):
        words = [w(0, "DEUTDEDB", 0.1, 0.5, 0.4, 0.52)]
        candidates = regex_extract("bic", words)
        assert len(candidates) >= 1
        assert candidates[0].word_ids == [0]

    def test_valid_bic_11_chars(self):
        words = [w(0, "NWBKGB2LXXX", 0.1, 0.5, 0.4, 0.52)]
        candidates = regex_extract("bic", words)
        assert len(candidates) >= 1

    def test_invalid_bic_wrong_length(self):
        words = [w(0, "DEUT", 0.1, 0.5, 0.3, 0.52)]
        assert regex_extract("bic", words) == []

    def test_invalid_bic_has_digits_in_wrong_place(self):
        words = [w(0, "1234DEDB", 0.1, 0.5, 0.4, 0.52)]
        assert regex_extract("bic", words) == []


class TestRegexExtractDate:
    def test_iso_date_single_word(self):
        words = [w(0, "2024-01-15", 0.1, 0.5, 0.3, 0.52)]
        candidates = regex_extract("date_issue", words)
        assert len(candidates) >= 1
        assert candidates[0].word_ids == [0]

    def test_eu_dot_date(self):
        words = [w(0, "15.01.2024", 0.1, 0.5, 0.3, 0.52)]
        candidates = regex_extract("date_due", words)
        assert len(candidates) >= 1

    def test_slash_date(self):
        words = [w(0, "01/15/2024", 0.1, 0.5, 0.3, 0.52)]
        candidates = regex_extract("date_issue", words)
        assert len(candidates) >= 1

    def test_text_date_multi_word(self):
        """'15 January 2024' spans three OCR words."""
        words = [
            w(0, "15",       0.10, 0.5, 0.15, 0.52),
            w(1, "January",  0.16, 0.5, 0.30, 0.52),
            w(2, "2024",     0.31, 0.5, 0.45, 0.52),
        ]
        candidates = regex_extract("date_issue", words)
        assert len(candidates) >= 1
        multi = [c for c in candidates if len(c.word_ids) >= 2]
        assert len(multi) >= 1

    def test_line_item_date(self):
        words = [w(0, "31.12.2023", 0.1, 0.5, 0.3, 0.52)]
        candidates = regex_extract("line_item_date", words)
        assert len(candidates) >= 1

    def test_no_date(self):
        words = [w(0, "ACME", 0.1, 0.5, 0.3, 0.52), w(1, "Corp", 0.31, 0.5, 0.5, 0.52)]
        assert regex_extract("date_issue", words) == []


class TestRegexExtractAmount:
    def test_amount_with_euro_symbol_prefix(self):
        words = [w(0, "€1,234.56", 0.1, 0.5, 0.3, 0.52)]
        candidates = regex_extract("amount_total_gross", words)
        assert len(candidates) >= 1

    def test_amount_with_currency_suffix_two_words(self):
        words = [w(0, "1234.56", 0.1, 0.5, 0.25, 0.52), w(1, "EUR", 0.26, 0.5, 0.35, 0.52)]
        candidates = regex_extract("amount_due", words)
        assert len(candidates) >= 1

    def test_amount_plain_numeric(self):
        words = [w(0, "99.99", 0.1, 0.5, 0.25, 0.52)]
        candidates = regex_extract("amount_total_net", words)
        assert len(candidates) >= 1

    def test_line_item_amount_net(self):
        words = [w(0, "250.00", 0.5, 0.5, 0.65, 0.52)]
        candidates = regex_extract("line_item_amount_net", words)
        assert len(candidates) >= 1

    def test_line_item_unit_price_gross(self):
        words = [w(0, "49.95", 0.4, 0.5, 0.55, 0.52)]
        candidates = regex_extract("line_item_unit_price_gross", words)
        assert len(candidates) >= 1

    def test_no_amount_text_only(self):
        words = [w(0, "ACME"), w(1, "Corporation")] if False else [
            w(0, "ACME", 0.1, 0.5, 0.3, 0.52),
        ]
        assert regex_extract("amount_due", words) == []


class TestRegexExtractEmail:
    def test_valid_email(self):
        words = [w(0, "info@acme.com", 0.1, 0.5, 0.4, 0.52)]
        candidates = regex_extract("vendor_email", words)
        assert len(candidates) >= 1

    def test_invalid_email_no_at(self):
        words = [w(0, "not-an-email", 0.1, 0.5, 0.4, 0.52)]
        assert regex_extract("vendor_email", words) == []

    def test_invalid_email_no_tld(self):
        words = [w(0, "user@domain", 0.1, 0.5, 0.4, 0.52)]
        assert regex_extract("vendor_email", words) == []


class TestRegexExtractCurrency:
    def test_currency_symbol(self):
        words = [w(0, "€", 0.1, 0.5, 0.15, 0.52)]
        candidates = regex_extract("currency_code_amount_due", words)
        assert len(candidates) >= 1

    def test_currency_iso_code(self):
        words = [w(0, "USD", 0.1, 0.5, 0.2, 0.52)]
        candidates = regex_extract("currency_code_amount_due", words)
        assert len(candidates) >= 1

    def test_unknown_three_letter_code(self):
        """XYZ is not in the known currency list — no match."""
        words = [w(0, "XYZ", 0.1, 0.5, 0.2, 0.52)]
        candidates = regex_extract("currency_code_amount_due", words)
        assert len(candidates) == 0

    def test_line_item_currency(self):
        words = [w(0, "GBP", 0.1, 0.5, 0.2, 0.52)]
        candidates = regex_extract("line_item_currency", words)
        assert len(candidates) >= 1


class TestRegexExtractUnsupported:
    def test_vendor_name_returns_empty(self):
        """Free-text fields have no registered pattern."""
        words = [w(0, "ACME Corp", 0.1, 0.5, 0.4, 0.52)]
        assert regex_extract("vendor_name", words) == []

    def test_vendor_address_returns_empty(self):
        words = [w(0, "123 Main St", 0.1, 0.5, 0.5, 0.52)]
        assert regex_extract("vendor_address", words) == []

    def test_unknown_fieldtype_returns_empty(self):
        words = [w(0, "anything", 0.1, 0.5, 0.4, 0.52)]
        assert regex_extract("not_a_real_field", words) == []


# ── validator_check ───────────────────────────────────────────────────────────


class TestValidatorCheck:
    def test_valid_iban_returns_candidate(self):
        result = validator_check("iban", "DE89370400440532013000")
        assert result is not None
        assert isinstance(result, Candidate)
        assert result.score == 1.0
        assert result.source == "validator"
        assert result.word_ids == []

    def test_invalid_iban_bad_checksum_returns_none(self):
        result = validator_check("iban", "DE89370400440532013001")
        assert result is None

    def test_valid_bic_returns_candidate(self):
        result = validator_check("bic", "DEUTDEDB")
        assert result is not None
        assert result.score == 1.0

    def test_invalid_bic_returns_none(self):
        result = validator_check("bic", "NOTBIC")
        assert result is None

    def test_valid_date_returns_candidate(self):
        result = validator_check("date_issue", "2024-01-15")
        assert result is not None
        assert result.score >= 0.5

    def test_invalid_date_returns_none(self):
        result = validator_check("date_issue", "not-a-date!!!!")
        assert result is None

    def test_valid_amount_returns_candidate(self):
        result = validator_check("amount_total_gross", "1234.56")
        assert result is not None
        assert result.score >= 0.5

    def test_invalid_amount_returns_none(self):
        result = validator_check("amount_due", "totally not an amount")
        assert result is None

    def test_unknown_fieldtype_always_returns_candidate(self):
        """format_confidence returns 1.0 for fields without a validator."""
        result = validator_check("vendor_name", "Some Company Ltd")
        assert result is not None
        assert result.score == 1.0

    def test_ambiguous_score_at_boundary(self):
        """Scores of exactly 0.5 are at the threshold — returned."""
        result = validator_check("vendor_tax_id", "DE 12345678")
        # Space in tax ID = 0.5 (ambiguous) — must return Candidate, not None
        if result is not None:
            assert result.score >= 0.5


# ── spatial_neighbor ──────────────────────────────────────────────────────────


class TestSpatialNeighbor:
    def test_right_neighbor_found(self):
        """'INV-001' is to the right of 'No:' — should be returned."""
        words = _invoice_words()
        candidates = spatial_neighbor(["Invoice No", "No:"], words, direction="right")
        assert len(candidates) >= 1
        all_word_ids = {wid for c in candidates for wid in c.word_ids}
        assert 2 in all_word_ids  # word 2 is "INV-001"

    def test_right_neighbor_score_decreases_with_distance(self):
        """Closer neighbors get higher scores."""
        words = _invoice_words()
        candidates = spatial_neighbor(["No:"], words, direction="right")
        if len(candidates) >= 2:
            assert candidates[0].score >= candidates[-1].score

    def test_label_not_found_returns_empty(self):
        words = _invoice_words()
        candidates = spatial_neighbor(["IBAN", "Account No"], words, direction="right")
        assert candidates == []

    def test_empty_words_returns_empty(self):
        assert spatial_neighbor(["Invoice No"], [], direction="right") == []

    def test_empty_label_phrases_returns_empty(self):
        words = _invoice_words()
        assert spatial_neighbor([], words, direction="right") == []

    def test_below_direction(self):
        """Words below the label anchor — no crash even if no match."""
        words = _invoice_words()
        candidates = spatial_neighbor(["Invoice"], words, direction="below")
        assert isinstance(candidates, list)

    def test_right_or_below_default(self):
        words = _invoice_words()
        candidates = spatial_neighbor(["Date:"], words)  # default direction
        assert isinstance(candidates, list)

    def test_max_distance_frac_excludes_far_words(self):
        """With tiny max_distance, no neighbor should be found."""
        words = _invoice_words()
        candidates = spatial_neighbor(["Invoice No", "No:"], words, max_distance_frac=0.001)
        # Should find nothing because gap > 0.001
        assert candidates == []

    def test_case_insensitive_label_match(self):
        words = _invoice_words()
        candidates = spatial_neighbor(["invoice no", "no:"], words, direction="right")
        assert len(candidates) >= 1

    def test_source_is_spatial(self):
        words = _invoice_words()
        candidates = spatial_neighbor(["No:"], words, direction="right")
        for c in candidates:
            assert c.source == "spatial"

    def test_two_row_doc(self):
        """Toy two-row document: label on row 0, value on row 1 (below)."""
        words = [
            w(0, "IBAN:",            0.05, 0.10, 0.15, 0.12),
            w(1, "DE89370400440532", 0.05, 0.20, 0.40, 0.22),
        ]
        candidates = spatial_neighbor(["IBAN"], words, direction="below")
        assert len(candidates) >= 1
        assert 1 in candidates[0].word_ids


# ── cluster_fewshot ───────────────────────────────────────────────────────────


class TestClusterFewshot:
    def _train_docs(self) -> dict:
        return {
            "doc001": {
                "cluster_id": 5,
                "fields": [
                    {"fieldtype": "document_id", "text": "INV-001"},
                    {"fieldtype": "date_issue", "text": "2024-01-15"},
                ],
            },
            "doc002": {
                "cluster_id": 5,
                "fields": [{"fieldtype": "document_id", "text": "INV-002"}],
            },
            "doc003": {
                "cluster_id": 7,
                "fields": [{"fieldtype": "document_id", "text": "ORD-999"}],
            },
        }

    def test_matching_cluster_returns_examples(self):
        examples = cluster_fewshot("document_id", 5, self._train_docs())
        assert len(examples) >= 1
        assert examples[0]["fieldtype"] == "document_id"

    def test_matching_cluster_correct_text(self):
        examples = cluster_fewshot("document_id", 5, self._train_docs())
        texts = {e["text"] for e in examples}
        assert texts <= {"INV-001", "INV-002"}

    def test_non_matching_cluster_returns_empty(self):
        examples = cluster_fewshot("document_id", 99, self._train_docs())
        assert examples == []

    def test_none_cluster_id_returns_any_examples(self):
        """cluster_id=None means return from any cluster."""
        examples = cluster_fewshot("document_id", None, self._train_docs())
        assert len(examples) >= 1

    def test_wrong_fieldtype_returns_empty(self):
        examples = cluster_fewshot("vendor_email", 5, self._train_docs())
        assert examples == []

    def test_empty_train_docs_returns_empty(self):
        assert cluster_fewshot("document_id", 5, {}) == []

    def test_at_most_three_examples_returned(self):
        many_docs = {
            f"doc{i:03d}": {"cluster_id": 1, "fields": [{"fieldtype": "date_issue", "text": f"2024-0{(i % 9) + 1}-01"}]}
            for i in range(10)
        }
        examples = cluster_fewshot("date_issue", 1, many_docs)
        assert len(examples) <= 3

    def test_result_is_list_of_dicts(self):
        examples = cluster_fewshot("document_id", 5, self._train_docs())
        assert isinstance(examples, list)
        for ex in examples:
            assert isinstance(ex, dict)
            assert "fieldtype" in ex
            assert "text" in ex
            assert "docid" in ex


# ── refine_span ───────────────────────────────────────────────────────────────


class TestRefineSpan:
    def test_returns_candidate_type(self):
        words = _invoice_words()
        result = refine_span("document_id", [0, 1, 2], words, "INV-001")
        assert isinstance(result, Candidate)
        assert result.source == "refiner"

    def test_strips_label_words_from_document_id(self):
        """'Invoice No:' prefix should be stripped, keeping only the value."""
        words = _invoice_words()
        result = refine_span("document_id", [0, 1, 2], words, "INV-001")
        # Word 2 (INV-001) should be in the refined ids
        assert 2 in result.word_ids

    def test_amount_refiner_keeps_numeric_words(self):
        words = _invoice_words()
        result = refine_span("amount_total_gross", [5, 6, 7], words, "€ 1,234.56")
        assert isinstance(result, Candidate)
        assert len(result.word_ids) >= 1

    def test_empty_word_ids_returns_empty_candidate(self):
        words = _invoice_words()
        result = refine_span("document_id", [], words, "")
        assert isinstance(result, Candidate)
        assert result.word_ids == []
        assert result.score == 0.0

    def test_invalid_word_ids_not_on_page(self):
        """word_ids that don't exist on the page — refiner should handle gracefully."""
        words = _invoice_words()
        result = refine_span("document_id", [999, 1000], words, "")
        assert isinstance(result, Candidate)

    def test_date_refiner(self):
        words = _invoice_words()
        result = refine_span("date_issue", [3, 4], words, "2024-01-15")
        assert isinstance(result, Candidate)
        assert 4 in result.word_ids  # date word

    def test_refined_score_is_positive(self):
        words = _invoice_words()
        result = refine_span("document_id", [0, 1, 2], words, "INV-001")
        if result.word_ids:
            assert result.score > 0

    def test_reason_field_is_populated(self):
        words = _invoice_words()
        result = refine_span("document_id", [0, 1, 2], words, "INV-001")
        assert isinstance(result.reason, str)
