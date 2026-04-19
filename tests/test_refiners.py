"""Unit tests for refiners.py — no real Claude calls, pure synthetic word_ids.

Ref: PLAN_V2.md §Phase 2.
"""

from __future__ import annotations

import pytest

from beat_docile.data import WordBox
from beat_docile.refiners import (
    _bbox_from_words,
    _largest_contiguous_run,
    _strip_label_prefix,
    _to_rows,
    refine_field,
)

# ── Synthetic word builder ────────────────────────────────────────────────────


def w(wid: int, text: str, left: float, top: float, right: float, bottom: float) -> WordBox:
    return WordBox(id=wid, text=text, bbox=(left, top, right, bottom), page=0)


# ── Helper tests ──────────────────────────────────────────────────────────────


def test_largest_contiguous_run_basic():
    assert _largest_contiguous_run([1, 2, 5, 6, 7, 9]) == [5, 6, 7]


def test_largest_contiguous_run_single():
    assert _largest_contiguous_run([3]) == [3]


def test_largest_contiguous_run_all_contiguous():
    assert _largest_contiguous_run([4, 5, 6]) == [4, 5, 6]


def test_largest_contiguous_run_empty():
    assert _largest_contiguous_run([]) == []


def test_to_rows_groups_by_top():
    words = [
        w(0, "A", 0.0, 0.10, 0.1, 0.12),
        w(1, "B", 0.1, 0.10, 0.2, 0.12),
        w(2, "C", 0.0, 0.20, 0.1, 0.22),
    ]
    rows = _to_rows([0, 1, 2], words)
    assert len(rows) == 2
    assert set(rows[0]) == {0, 1}
    assert rows[1] == [2]


def test_to_rows_single_row():
    words = [
        w(0, "X", 0.0, 0.10, 0.1, 0.12),
        w(1, "Y", 0.2, 0.105, 0.3, 0.12),
    ]
    rows = _to_rows([0, 1], words)
    assert len(rows) == 1


def test_strip_label_prefix_strips_label():
    words = [
        w(0, "Bill", 0.0, 0.1, 0.1, 0.12),
        w(1, "To:", 0.1, 0.1, 0.2, 0.12),
        w(2, "Acme", 0.2, 0.1, 0.3, 0.12),
        w(3, "Corp", 0.3, 0.1, 0.4, 0.12),
    ]
    result = _strip_label_prefix([0, 1, 2, 3], words, {"bill", "to:"})
    assert result == [2, 3]


def test_strip_label_prefix_no_label():
    words = [
        w(0, "Acme", 0.0, 0.1, 0.1, 0.12),
        w(1, "Corp", 0.1, 0.1, 0.2, 0.12),
    ]
    result = _strip_label_prefix([0, 1], words, {"bill", "to:"})
    assert result == [0, 1]


def test_bbox_from_words():
    words = [
        w(0, "A", 0.1, 0.2, 0.3, 0.4),
        w(1, "B", 0.5, 0.1, 0.7, 0.3),
    ]
    bbox = _bbox_from_words([0, 1], words)
    assert bbox is not None
    assert bbox.left == pytest.approx(0.1)
    assert bbox.top == pytest.approx(0.1)
    assert bbox.right == pytest.approx(0.7)
    assert bbox.bottom == pytest.approx(0.4)


# ── refine_field: address ─────────────────────────────────────────────────────


def test_refine_address_strips_label_and_keeps_block():
    words = [
        w(0, "Bill",      0.0,  0.10, 0.05, 0.12),
        w(1, "To:",       0.06, 0.10, 0.10, 0.12),
        w(2, "Acme",      0.0,  0.15, 0.06, 0.17),
        w(3, "Corp",      0.07, 0.15, 0.13, 0.17),
        w(4, "123",       0.0,  0.19, 0.04, 0.21),
        w(5, "Main",      0.05, 0.19, 0.11, 0.21),
        w(6, "St",        0.12, 0.19, 0.16, 0.21),
        # Far-away row (different address block)
        w(7, "Elsewhere", 0.0,  0.60, 0.10, 0.62),
    ]
    ids, bbox = refine_field("customer_billing_address", [0, 1, 2, 3, 4, 5, 6, 7], words)
    # Label words stripped, far-away word excluded
    assert 0 not in ids
    assert 1 not in ids
    assert 7 not in ids
    assert all(i in ids for i in [2, 3, 4, 5, 6])
    assert bbox is not None


def test_refine_address_no_label():
    words = [
        w(0, "123",  0.0,  0.10, 0.05, 0.12),
        w(1, "Main", 0.06, 0.10, 0.12, 0.12),
        w(2, "City", 0.0,  0.15, 0.06, 0.17),
    ]
    ids, bbox = refine_field("vendor_address", [0, 1, 2], words)
    assert set(ids) == {0, 1, 2}
    assert bbox is not None


# ── refine_field: names ───────────────────────────────────────────────────────


def test_refine_name_multi_row_keeps_best():
    words = [
        w(0, "Vendor:", 0.0,  0.10, 0.08, 0.12),
        w(1, "Acme",   0.0,  0.18, 0.06, 0.20),
        w(2, "Corp",   0.07, 0.18, 0.14, 0.20),
        w(3, "Ltd",    0.15, 0.18, 0.20, 0.20),
        # Stray word far below
        w(4, "Junk",   0.0,  0.50, 0.05, 0.52),
    ]
    ids, _bbox = refine_field("vendor_name", [0, 1, 2, 3, 4], words)
    assert 0 not in ids  # label stripped
    assert all(i in ids for i in [1, 2, 3])
    assert 4 not in ids


def test_refine_name_single_row():
    words = [
        w(0, "John", 0.0, 0.10, 0.05, 0.12),
        w(1, "Doe",  0.06, 0.10, 0.12, 0.12),
    ]
    ids, _bbox = refine_field("customer_billing_name", [0, 1], words)
    assert set(ids) == {0, 1}


# ── refine_field: single-value (IDs) ─────────────────────────────────────────


def test_refine_document_id_strips_label():
    words = [
        w(0, "Invoice", 0.0,  0.10, 0.08, 0.12),
        w(1, "No:",     0.09, 0.10, 0.14, 0.12),
        w(2, "INV-001", 0.15, 0.10, 0.25, 0.12),
    ]
    ids, bbox = refine_field("document_id", [0, 1, 2], words)
    assert 0 not in ids
    assert 1 not in ids
    assert 2 in ids
    assert bbox is not None


def test_refine_document_id_too_many_tokens_keeps_run():
    words = [w(i, f"tok{i}", i * 0.05, 0.10, (i + 1) * 0.05, 0.12) for i in range(8)]
    # word_ids is a contiguous block — keep the longest run
    ids, _ = refine_field("document_id", list(range(8)), words)
    # Should not crash; length should be ≤ original
    assert len(ids) <= 8


# ── refine_field: financial codes ────────────────────────────────────────────


def test_refine_iban_strips_label():
    words = [
        w(0, "IBAN:",          0.0,  0.10, 0.06, 0.12),
        w(1, "GB29",           0.07, 0.10, 0.14, 0.12),
        w(2, "NWBK",          0.15, 0.10, 0.22, 0.12),
        w(3, "6016",          0.23, 0.10, 0.30, 0.12),
    ]
    ids, _bbox = refine_field("iban", [0, 1, 2, 3], words)
    assert 0 not in ids
    assert 1 in ids and 2 in ids and 3 in ids


def test_refine_bic_single_token():
    words = [
        w(0, "BIC:",      0.0,  0.10, 0.05, 0.12),
        w(1, "DEUTDEDB", 0.06, 0.10, 0.18, 0.12),
    ]
    ids, _bbox = refine_field("bic", [0, 1], words)
    assert 0 not in ids
    assert 1 in ids


# ── refine_field: amounts ─────────────────────────────────────────────────────


def test_refine_amount_keeps_currency_and_value():
    words = [
        w(0, "Total:",  0.0,  0.10, 0.07, 0.12),
        w(1, "$",       0.08, 0.10, 0.10, 0.12),
        w(2, "1,234.56", 0.11, 0.10, 0.22, 0.12),
    ]
    ids, _bbox = refine_field("amount_due", [0, 1, 2], words)
    assert 0 not in ids
    assert 1 in ids
    assert 2 in ids


def test_refine_amount_single_token():
    words = [w(0, "100.00", 0.5, 0.10, 0.65, 0.12)]
    ids, bbox = refine_field("amount_total_net", [0], words)
    assert ids == [0]
    assert bbox is not None


# ── refine_field: dates ───────────────────────────────────────────────────────


def test_refine_date_strips_label():
    words = [
        w(0, "Date:",      0.0,  0.10, 0.06, 0.12),
        w(1, "01/15/2024", 0.07, 0.10, 0.18, 0.12),
    ]
    ids, _bbox = refine_field("date_issue", [0, 1], words)
    assert 0 not in ids
    assert 1 in ids


def test_refine_date_multiword_format():
    words = [
        w(0, "15", 0.0,  0.10, 0.04, 0.12),
        w(1, "Jan", 0.05, 0.10, 0.10, 0.12),
        w(2, "2024", 0.11, 0.10, 0.19, 0.12),
    ]
    ids, bbox = refine_field("date_due", [0, 1, 2], words)
    assert set(ids) == {0, 1, 2}
    assert bbox is not None


# ── refine_field: generic + edge cases ───────────────────────────────────────


def test_refine_generic_contiguous_run():
    words = [w(i, f"w{i}", i * 0.05, 0.10, (i + 1) * 0.05, 0.12) for i in range(10)]
    # Non-contiguous ids; longest run [3,4,5,6]
    _ids, bbox = refine_field("payment_terms", [0, 3, 4, 5, 6, 9], words)
    assert bbox is not None


def test_refine_field_invalid_word_ids_returns_fallback():
    words = [w(0, "abc", 0.1, 0.1, 0.2, 0.2)]
    # word_ids 99 and 100 don't exist in words
    ids, bbox = refine_field("document_id", [99, 100], words)
    # Should return empty (all invalid)
    assert ids == []
    assert bbox is None


def test_refine_field_empty_word_ids():
    words = [w(0, "abc", 0.1, 0.1, 0.2, 0.2)]
    ids, bbox = refine_field("vendor_name", [], words)
    assert ids == []
    assert bbox is None


def test_refine_field_unknown_fieldtype_falls_back_gracefully():
    words = [
        w(0, "foo", 0.0, 0.1, 0.1, 0.2),
        w(1, "bar", 0.1, 0.1, 0.2, 0.2),
    ]
    ids, bbox = refine_field("vendor_email", [0, 1], words)
    assert bbox is not None
    assert len(ids) >= 1
