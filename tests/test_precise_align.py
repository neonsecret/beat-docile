"""Unit tests for precise_align.py — character-level OCR alignment.

All tests use synthetic WordBox lists so no real documents are needed.
Tests cover: exact, exact_norm, fuzzy, format, multi-line, failed, amount, date, code fields.
"""

from __future__ import annotations

from beat_docile.data import WordBox
from beat_docile.precise_align import (
    _norm_compact,
    _norm_digits_only,
    _norm_nfkc,
    _reading_order,
    align_fields_to_words,
    align_text_to_words,
)

# ─────────────────────────────────────────────────────────────────────────────
# Test helpers
# ─────────────────────────────────────────────────────────────────────────────

def _w(wid: int, text: str, *, top: float = 0.1, left: float = 0.0, width: float = 0.1) -> WordBox:
    """Create a WordBox with simple geometry."""
    return WordBox(id=wid, text=text, bbox=(left, top, left + width, top + 0.02), page=0)


def _row(*texts, top: float = 0.1, start_id: int = 0) -> list[WordBox]:
    """Create a row of words with sequential IDs and evenly spaced left coords."""
    words = []
    for i, text in enumerate(texts):
        words.append(_w(start_id + i, text, top=top, left=i * 0.15))
    return words


# ─────────────────────────────────────────────────────────────────────────────
# Normalization helpers
# ─────────────────────────────────────────────────────────────────────────────

def test_norm_compact():
    assert _norm_compact("DE89 3704 0044") == "DE8937040044"
    assert _norm_compact("hello world") == "helloworld"


def test_norm_digits_only():
    assert _norm_digits_only("1.234,56 EUR") == "123456"
    assert _norm_digits_only("2024-01-15") == "20240115"
    assert _norm_digits_only("21%") == "21"


def test_norm_nfkc():
    # NFKC normalizes compatibility equivalents + lowercases, but keeps precomposed accents
    assert _norm_nfkc("HELLO  WORLD") == "hello world"
    assert _norm_nfkc("  SPACES  ") == "spaces"
    # Ligatures decomposed
    assert _norm_nfkc("ﬁle") == "file"


# ─────────────────────────────────────────────────────────────────────────────
# Reading order
# ─────────────────────────────────────────────────────────────────────────────

def test_reading_order_top_to_bottom():
    words = [
        _w(0, "B", top=0.2),
        _w(1, "A", top=0.1),
        _w(2, "C", top=0.3),
    ]
    sorted_words = _reading_order(words)
    assert [w.text for w in sorted_words] == ["A", "B", "C"]


def test_reading_order_left_to_right_same_row():
    words = [
        _w(0, "C", top=0.1, left=0.4),
        _w(1, "A", top=0.1, left=0.0),
        _w(2, "B", top=0.1, left=0.2),
    ]
    sorted_words = _reading_order(words)
    assert [w.text for w in sorted_words] == ["A", "B", "C"]


# ─────────────────────────────────────────────────────────────────────────────
# Exact match
# ─────────────────────────────────────────────────────────────────────────────

def test_exact_match_single_word():
    words = _row("Invoice", "No:", "INV-2024-001", top=0.1, start_id=0)
    result = align_text_to_words("INV-2024-001", words)
    assert result.method == "exact"
    assert result.confidence == 1.0
    assert result.word_ids == [2]  # only the ID token


def test_exact_match_multi_word():
    words = _row("Total", "Due:", "1", "234.56", top=0.1, start_id=0)
    result = align_text_to_words("1 234.56", words)
    assert result.method == "exact"
    assert result.confidence == 1.0
    assert 2 in result.word_ids
    assert 3 in result.word_ids


def test_exact_match_compact_iban():
    """IBAN 'DE89 3704 0044' should match words ['DE89', '3704', '0044'] via compact."""
    words = [
        _w(0, "IBAN:", top=0.1, left=0.0),
        _w(1, "DE89", top=0.1, left=0.15),
        _w(2, "3704", top=0.1, left=0.30),
        _w(3, "0044", top=0.1, left=0.45),
    ]
    result = align_text_to_words("DE89 3704 0044", words)
    assert result.method == "exact"
    assert set(result.word_ids) == {1, 2, 3}


def test_exact_match_full_iban():
    words = [
        _w(0, "DE89", top=0.1, left=0.0),
        _w(1, "3704", top=0.1, left=0.1),
        _w(2, "0044", top=0.1, left=0.2),
        _w(3, "0532", top=0.1, left=0.3),
        _w(4, "0130", top=0.1, left=0.4),
        _w(5, "00", top=0.1, left=0.5),
    ]
    result = align_text_to_words("DE89 3704 0044 0532 0130 00", words)
    assert result.method == "exact"
    assert sorted(result.word_ids) == [0, 1, 2, 3, 4, 5]


def test_exact_match_whitespace_collapse():
    """Extra spaces in query are collapsed."""
    words = _row("2024-01-15", top=0.1, start_id=0)
    result = align_text_to_words("  2024-01-15  ", words)
    assert result.method == "exact"
    assert result.word_ids == [0]


# ─────────────────────────────────────────────────────────────────────────────
# Exact normalized match
# ─────────────────────────────────────────────────────────────────────────────

def test_exact_norm_lowercase():
    """Case difference should resolve with NFKC normalization."""
    words = _row("VENDOR", "GMBH", top=0.1, start_id=0)
    result = align_text_to_words("Vendor GmbH", words)
    # Should find match via exact_norm or fuzzy
    assert result.confidence > 0.8
    assert set(result.word_ids) == {0, 1}


def test_exact_norm_accents():
    """Accented characters normalized by NFKC."""
    words = _row("Straße", top=0.1, start_id=0)
    result = align_text_to_words("Straße", words)
    assert result.confidence >= 0.95
    assert result.word_ids == [0]


def test_exact_norm_ocr_code_substitution():
    """OCR common substitutions: 'l' ↔ '1', 'O' ↔ '0'."""
    words = _row("DElOO", top=0.1, start_id=0)  # OCR error: 'O' instead of '0'
    result = align_text_to_words("DE100", words)
    # Should match via OCR substitution or fuzzy
    assert result.confidence > 0.7
    assert result.word_ids == [0]


# ─────────────────────────────────────────────────────────────────────────────
# Fuzzy match
# ─────────────────────────────────────────────────────────────────────────────

def test_fuzzy_match_single_typo():
    """One character difference should still fuzzy-match."""
    words = _row("15.01.2O24", top=0.1, start_id=0)  # OCR '2O24' instead of '2024'
    result = align_text_to_words("15.01.2024", words)
    assert result.method in ("fuzzy", "exact_norm")
    assert result.confidence >= 0.75
    assert result.word_ids == [0]


def test_fuzzy_match_amount_format_difference():
    """European decimal format vs query: fuzzy should still find the right word."""
    words = _row("1.234,56", top=0.1, start_id=0)
    result = align_text_to_words("1,234.56", words, fieldtype="amount_due")
    # The digit normalization strategy should handle this
    assert result.confidence > 0.7
    assert result.word_ids == [0]


def test_fuzzy_no_match_below_threshold():
    """Completely unrelated text should return failed."""
    words = _row("Hello", "World", top=0.1, start_id=0)
    result = align_text_to_words("XYZ123456789", words)
    assert result.method == "failed"
    assert result.confidence == 0.0
    assert result.word_ids == []


# ─────────────────────────────────────────────────────────────────────────────
# Format-constrained match
# ─────────────────────────────────────────────────────────────────────────────

def test_format_constrained_iban():
    """FORMAT strategy for IBAN field: should find the IBAN even with slight variations."""
    words = [
        _w(0, "Account:", top=0.1, left=0.0),
        _w(1, "DE89370400440532013000", top=0.1, left=0.2),  # compact IBAN
        _w(2, "BIC:", top=0.1, left=0.5),
        _w(3, "DEUTDEDB", top=0.1, left=0.6),
    ]
    result = align_text_to_words("DE89 3704 0044 0532 0130 00", words, fieldtype="iban")
    assert result.confidence > 0.0
    assert 1 in result.word_ids  # should find the IBAN word


def test_format_constrained_bic():
    """BIC format validator should help find BIC code."""
    words = [
        _w(0, "Bank:", top=0.1, left=0.0),
        _w(1, "DEUTDEDB", top=0.1, left=0.2),
        _w(2, "Account", top=0.1, left=0.4),
    ]
    result = align_text_to_words("DEUTDEDB", words, fieldtype="bic")
    assert result.confidence >= 0.9
    assert result.word_ids == [1]


def test_format_constrained_date():
    """Date field should find correct date token."""
    words = [
        _w(0, "Invoice", top=0.1, left=0.0),
        _w(1, "Date:", top=0.1, left=0.15),
        _w(2, "15.01.2024", top=0.1, left=0.30),
        _w(3, "Due:", top=0.1, left=0.55),
        _w(4, "30.01.2024", top=0.1, left=0.70),
    ]
    result = align_text_to_words("15.01.2024", words, fieldtype="date_issue")
    assert result.confidence > 0.8
    assert result.word_ids == [2]


# ─────────────────────────────────────────────────────────────────────────────
# Multi-line addresses
# ─────────────────────────────────────────────────────────────────────────────

def test_multiline_address_exact():
    """Multi-line address with \\n separator should align each line."""
    words = [
        # Row 1 (company name)
        _w(0, "Vendor", top=0.10, left=0.0),
        _w(1, "GmbH", top=0.10, left=0.15),
        # Row 2 (street)
        _w(2, "Hauptstrasse", top=0.15, left=0.0),
        _w(3, "42", top=0.15, left=0.25),
        # Row 3 (city)
        _w(4, "12345", top=0.20, left=0.0),
        _w(5, "Berlin", top=0.20, left=0.15),
    ]
    query = "Vendor GmbH\nHauptstrasse 42\n12345 Berlin"
    result = align_text_to_words(query, words, fieldtype="vendor_address")
    assert result.method.startswith("multiline_")
    assert result.confidence > 0.8
    assert set(result.word_ids) == {0, 1, 2, 3, 4, 5}


def test_multiline_address_partial_match():
    """Missing line in OCR should still return partial match."""
    words = [
        _w(0, "Vendor", top=0.10, left=0.0),
        _w(1, "GmbH", top=0.10, left=0.15),
        # 'Hauptstrasse 42' is missing from OCR (e.g., image glitch)
        _w(2, "12345", top=0.20, left=0.0),
        _w(3, "Berlin", top=0.20, left=0.15),
    ]
    query = "Vendor GmbH\nHauptstrasse 42\n12345 Berlin"
    result = align_text_to_words(query, words, fieldtype="vendor_address")
    assert result.method.startswith("multiline_")
    assert result.confidence > 0.0
    assert set(result.word_ids).issubset({0, 1, 2, 3})


def test_multiline_single_line_no_newline():
    """Text without \\n for non-address field should not use multiline strategy."""
    words = _row("42", top=0.1, start_id=0)
    result = align_text_to_words("42", words, fieldtype="line_item_position")
    assert result.method == "exact"
    assert result.word_ids == [0]


# ─────────────────────────────────────────────────────────────────────────────
# Amount and date field-specific handling
# ─────────────────────────────────────────────────────────────────────────────

def test_amount_exact_match():
    words = [
        _w(0, "Total:", top=0.1, left=0.0),
        _w(1, "1,234.56", top=0.1, left=0.2),
        _w(2, "EUR", top=0.1, left=0.4),
    ]
    result = align_text_to_words("1,234.56", words, fieldtype="amount_total_gross")
    assert result.confidence >= 0.9
    assert 1 in result.word_ids


def test_amount_digit_normalization():
    """European format 1.234,56 should match query 1,234.56 via digit normalization."""
    words = [_w(0, "1.234,56", top=0.1, left=0.0)]
    result = align_text_to_words("1,234.56", words, fieldtype="amount_due")
    assert result.confidence > 0.7
    assert result.word_ids == [0]


def test_date_exact_match():
    words = [
        _w(0, "Date:", top=0.1, left=0.0),
        _w(1, "15.01.2024", top=0.1, left=0.2),
    ]
    result = align_text_to_words("15.01.2024", words, fieldtype="date_issue")
    assert result.method == "exact"
    assert result.word_ids == [1]


def test_date_digit_normalization():
    """Different separators should still match via digit comparison."""
    words = [_w(0, "15-01-2024", top=0.1, left=0.0)]
    result = align_text_to_words("15.01.2024", words, fieldtype="date_issue")
    assert result.confidence > 0.7
    assert result.word_ids == [0]


# ─────────────────────────────────────────────────────────────────────────────
# Failed cases
# ─────────────────────────────────────────────────────────────────────────────

def test_empty_text_returns_failed():
    words = _row("Something", top=0.1, start_id=0)
    result = align_text_to_words("", words)
    assert result.method == "failed"


def test_empty_words_returns_failed():
    result = align_text_to_words("some text", [])
    assert result.method == "failed"


def test_completely_unrelated_returns_failed():
    words = _row("Apple", "Banana", "Cherry", top=0.1, start_id=0)
    result = align_text_to_words("XZ99999999", words)
    assert result.method == "failed"
    assert result.confidence == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# align_fields_to_words pipeline helper
# ─────────────────────────────────────────────────────────────────────────────

def test_align_fields_to_words_kile():
    words = [
        _w(0, "Invoice:", top=0.1, left=0.0),
        _w(1, "INV-2024-001", top=0.1, left=0.2),
        _w(2, "Date:", top=0.2, left=0.0),
        _w(3, "15.01.2024", top=0.2, left=0.15),
    ]
    extracted = [
        {"fieldtype": "document_id", "text": "INV-2024-001", "score": 0.95, "line_item_id": None},
        {"fieldtype": "date_issue", "text": "15.01.2024", "score": 0.9, "line_item_id": None},
    ]
    kile, lir = align_fields_to_words(extracted, words, page_idx=0)
    assert len(kile) == 2
    assert len(lir) == 0
    ft_map = {f.fieldtype: f for f in kile}
    assert "document_id" in ft_map
    assert "date_issue" in ft_map
    # Scores should be reduced by alignment confidence
    assert ft_map["document_id"].score <= 0.95
    assert ft_map["date_issue"].score <= 0.9


def test_align_fields_to_words_lir():
    words = [
        _w(0, "Widget", top=0.3, left=0.0),
        _w(1, "A", top=0.3, left=0.15),
        _w(2, "10.00", top=0.3, left=0.3),
        _w(3, "3", top=0.3, left=0.45),
        _w(4, "30.00", top=0.3, left=0.6),
    ]
    extracted = [
        {"fieldtype": "line_item_description", "text": "Widget A", "score": 0.9, "line_item_id": 1},
        {"fieldtype": "line_item_unit_price_net", "text": "10.00", "score": 0.85, "line_item_id": 1},
        {"fieldtype": "line_item_quantity", "text": "3", "score": 0.95, "line_item_id": 1},
    ]
    kile, lir = align_fields_to_words(extracted, words, page_idx=0)
    assert len(kile) == 0
    assert len(lir) == 3
    assert all(f.line_item_id == 1 for f in lir)


def test_align_fields_low_confidence_filtered():
    """Fields with alignment confidence below min_confidence should be dropped."""
    words = [_w(0, "Hello", top=0.1, left=0.0)]
    extracted = [
        {"fieldtype": "document_id", "text": "XZ999999999", "score": 0.9, "line_item_id": None},
    ]
    kile, _lir = align_fields_to_words(extracted, words, page_idx=0, min_confidence=0.5)
    assert len(kile) == 0  # should be filtered out (no match)


def test_align_fields_unknown_fieldtype_ignored():
    words = [_w(0, "something", top=0.1, left=0.0)]
    extracted = [
        {"fieldtype": "nonexistent_field_type", "text": "something", "score": 0.9, "line_item_id": None},
    ]
    kile, lir = align_fields_to_words(extracted, words, page_idx=0)
    assert len(kile) == 0
    assert len(lir) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────

def test_align_prefers_shorter_window():
    """Should return the shortest window that matches, not a longer superset."""
    words = [
        _w(0, "Some", top=0.1, left=0.00),
        _w(1, "INV-001", top=0.1, left=0.15),
        _w(2, "Other", top=0.1, left=0.30),
    ]
    result = align_text_to_words("INV-001", words)
    assert result.method == "exact"
    assert result.word_ids == [1]


def test_align_currency_symbol_in_amount():
    """Currency symbol may or may not be part of the aligned span."""
    words = [
        _w(0, "€", top=0.1, left=0.0),
        _w(1, "1,234.56", top=0.1, left=0.05),
    ]
    # Query includes currency symbol
    result = align_text_to_words("€1,234.56", words, fieldtype="amount_total_gross")
    assert result.confidence > 0.0


def test_align_tax_detail_multiple_occurrences():
    """Each distinct text value should be independently alignable."""
    words = [
        _w(0, "19%", top=0.30, left=0.0),
        _w(1, "100.00", top=0.30, left=0.15),
        _w(2, "190.00", top=0.30, left=0.30),
        _w(3, "7%", top=0.35, left=0.0),
        _w(4, "50.00", top=0.35, left=0.15),
        _w(5, "3.50", top=0.35, left=0.30),
    ]
    r1 = align_text_to_words("19%", words, fieldtype="tax_detail_rate")
    r2 = align_text_to_words("7%", words, fieldtype="tax_detail_rate")
    assert r1.confidence > 0.8
    assert r2.confidence > 0.8
    assert r1.word_ids != r2.word_ids
