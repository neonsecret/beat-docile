"""Tests for cls_candidates.py — sliding-window classifier candidate generation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from beat_docile.cls_candidates import (
    _ADDRESS_FIELDTYPES,
    _LIR_FIELDTYPES,
    _SKIP_FOR_RECALL,
    CandidateSpan,
    _deduplicate_candidates,
    _default_max_span,
    _sort_words_reading_order,
    _span_bbox,
    _span_text,
    generate_candidates,
    generate_doc_candidates,
    score_bbox_span,
)
from beat_docile.data import WordBox

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "classifiers"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_word(wid: int, text: str, left: float, t: float, r: float, b: float, page: int = 0) -> WordBox:
    return WordBox(id=wid, text=text, bbox=(left, t, r, b), page=page)


@pytest.fixture
def simple_page() -> list[WordBox]:
    """Six words on two rows."""
    return [
        _make_word(0, "Invoice",    0.0,  0.0,  0.1,  0.02),
        _make_word(1, "No:",        0.1,  0.0,  0.2,  0.02),
        _make_word(2, "12345",      0.2,  0.0,  0.3,  0.02),
        _make_word(3, "Date:",      0.0,  0.05, 0.1,  0.07),
        _make_word(4, "2024-01-15", 0.1,  0.05, 0.3,  0.07),
        _make_word(5, "Amount",     0.0,  0.10, 0.1,  0.12),
    ]


@pytest.fixture
def address_page() -> list[WordBox]:
    """Multi-line address block."""
    return [
        _make_word(0, "Acme",   0.0,  0.0,  0.1,  0.02),
        _make_word(1, "Corp",   0.1,  0.0,  0.2,  0.02),
        _make_word(2, "123",    0.0,  0.05, 0.05, 0.07),
        _make_word(3, "Main",   0.05, 0.05, 0.15, 0.07),
        _make_word(4, "Street", 0.15, 0.05, 0.3,  0.07),
        _make_word(5, "London", 0.0,  0.10, 0.15, 0.12),
        _make_word(6, "UK",     0.15, 0.10, 0.25, 0.12),
    ]


# ---------------------------------------------------------------------------
# _span_text / _span_bbox
# ---------------------------------------------------------------------------


def test_span_text_basic(simple_page):
    id_map = {w.id: w for w in simple_page}
    assert _span_text([0, 1, 2], id_map) == "Invoice No: 12345"


def test_span_text_missing_ids(simple_page):
    id_map = {w.id: w for w in simple_page}
    assert _span_text([0, 99, 2], id_map) == "Invoice 12345"


def test_span_bbox_basic(simple_page):
    id_map = {w.id: w for w in simple_page}
    bbox = _span_bbox([0, 1, 2], id_map)
    assert bbox == (0.0, 0.0, 0.3, 0.02)


def test_span_bbox_empty_ids():
    assert _span_bbox([], {}) == (0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# _deduplicate_candidates
# ---------------------------------------------------------------------------


def test_deduplicate_keeps_highest_score():
    c1 = CandidateSpan("document_id", [0, 1], (0, 0, 0.3, 0.02), 0, 0.9, "Inv 12345")
    c2 = CandidateSpan("document_id", [1, 0], (0, 0, 0.3, 0.02), 0, 0.7, "12345 Inv")
    result = _deduplicate_candidates([c1, c2])
    assert len(result) == 1
    assert result[0].score == 0.9


def test_deduplicate_different_word_sets():
    c1 = CandidateSpan("document_id", [0, 1], (0, 0, 0.2, 0.02), 0, 0.9, "a")
    c2 = CandidateSpan("document_id", [2, 3], (0.2, 0, 0.4, 0.02), 0, 0.8, "b")
    result = _deduplicate_candidates([c1, c2])
    assert len(result) == 2


def test_deduplicate_page_isolation():
    c1 = CandidateSpan("date_issue", [0], (0, 0, 0.1, 0.02), 0, 0.9, "a")
    c2 = CandidateSpan("date_issue", [0], (0, 0, 0.1, 0.02), 1, 0.85, "a")  # different page
    result = _deduplicate_candidates([c1, c2])
    assert len(result) == 2


# ---------------------------------------------------------------------------
# _sort_words_reading_order
# ---------------------------------------------------------------------------


def test_sort_reading_order_row_then_x(simple_page):
    shuffled = list(reversed(simple_page))
    sorted_words = _sort_words_reading_order(shuffled)
    texts = [w.text for w in sorted_words]
    # Row 0: Invoice, No:, 12345 — then row 1: Date:, 2024-01-15 — then row 2: Amount
    assert texts[0] == "Invoice"
    assert texts[1] == "No:"
    assert texts[2] == "12345"
    assert texts[3] == "Date:"


# ---------------------------------------------------------------------------
# _default_max_span
# ---------------------------------------------------------------------------


def test_default_max_span_address():
    for ft in _ADDRESS_FIELDTYPES:
        assert _default_max_span(ft) == 20


def test_default_max_span_non_address():
    assert _default_max_span("document_id") == 4
    assert _default_max_span("date_issue") == 4
    assert _default_max_span("amount_due") == 4


# ---------------------------------------------------------------------------
# Fieldtype sets
# ---------------------------------------------------------------------------


def test_lir_fieldtypes_in_skip_for_recall():
    for ft in _LIR_FIELDTYPES:
        assert ft in _SKIP_FOR_RECALL, f"{ft} should be in _SKIP_FOR_RECALL"


def test_address_fieldtypes_not_in_skip_for_recall():
    for ft in _ADDRESS_FIELDTYPES:
        assert ft not in _SKIP_FOR_RECALL, f"{ft} should not be skipped for recall"


def test_skip_for_recall_has_known_weak_classifiers():
    for ft in ("bic", "customer_tax_id", "tax_detail_rate", "vendor_registration_id"):
        assert ft in _SKIP_FOR_RECALL


# ---------------------------------------------------------------------------
# generate_candidates — mock pipeline
# ---------------------------------------------------------------------------


def _make_pipeline(prob: float = 0.9):
    """Return a mock sklearn Pipeline that always predicts `prob`."""
    pipeline = MagicMock()
    pipeline.predict_proba = lambda x: np.full((x.shape[0], 2), [[1.0 - prob, prob]])
    return pipeline


@patch("beat_docile.cls_candidates.load_classifier")
def test_generate_candidates_returns_above_threshold(mock_load, simple_page):
    mock_load.return_value = _make_pipeline(prob=0.95)
    candidates = generate_candidates(
        fieldtype="document_id",
        words=simple_page,
        page_w=1.0,
        page_h=1.0,
        model_dir=Path("models/classifiers"),
        score_threshold=0.7,
    )
    assert len(candidates) > 0
    for c in candidates:
        assert c.score >= 0.7
        assert c.fieldtype == "document_id"
        assert c.page == 0


@patch("beat_docile.cls_candidates.load_classifier")
def test_generate_candidates_returns_empty_below_threshold(mock_load, simple_page):
    mock_load.return_value = _make_pipeline(prob=0.3)
    candidates = generate_candidates(
        fieldtype="document_id",
        words=simple_page,
        page_w=1.0,
        page_h=1.0,
        model_dir=Path("models/classifiers"),
        score_threshold=0.7,
    )
    assert len(candidates) == 0


@patch("beat_docile.cls_candidates.load_classifier")
def test_generate_candidates_no_model_returns_empty(mock_load, simple_page):
    mock_load.return_value = None
    candidates = generate_candidates(
        fieldtype="iban",
        words=simple_page,
        page_w=1.0,
        page_h=1.0,
        model_dir=Path("models/classifiers"),
    )
    assert candidates == []


@patch("beat_docile.cls_candidates.load_classifier")
def test_generate_candidates_empty_words(mock_load):
    mock_load.return_value = _make_pipeline()
    candidates = generate_candidates(
        fieldtype="document_id",
        words=[],
        page_w=1.0,
        page_h=1.0,
        model_dir=Path("models/classifiers"),
    )
    assert candidates == []


@patch("beat_docile.cls_candidates.load_classifier")
def test_generate_candidates_address_allows_multirow(mock_load, address_page):
    """Address fieldtype should include multi-row spans."""
    mock_load.return_value = _make_pipeline(prob=0.95)
    candidates = generate_candidates(
        fieldtype="vendor_address",
        words=address_page,
        page_w=1.0,
        page_h=1.0,
        model_dir=Path("models/classifiers"),
        score_threshold=0.5,
    )
    multi_row = [c for c in candidates if len(c.word_ids) > 2]
    assert len(multi_row) > 0


@patch("beat_docile.cls_candidates.load_classifier")
def test_generate_candidates_single_line_stays_in_row(mock_load, simple_page):
    """Non-address fieldtype should not span across row boundaries."""
    mock_load.return_value = _make_pipeline(prob=0.95)
    candidates = generate_candidates(
        fieldtype="document_id",
        words=simple_page,
        page_w=1.0,
        page_h=1.0,
        model_dir=Path("models/classifiers"),
        score_threshold=0.5,
    )
    row0_ids = {0, 1, 2}
    row1_ids = {3, 4}
    for c in candidates:
        span_set = set(c.word_ids)
        assert not (span_set & row0_ids and span_set & row1_ids), (
            f"Cross-row span found: {c.word_ids}"
        )


@patch("beat_docile.cls_candidates.load_classifier")
def test_generate_candidates_deduplication(mock_load, simple_page):
    """Output should have no duplicate (page, word_ids) pairs."""
    mock_load.return_value = _make_pipeline(prob=0.95)
    candidates = generate_candidates(
        fieldtype="date_issue",
        words=simple_page,
        page_w=1.0,
        page_h=1.0,
        model_dir=Path("models/classifiers"),
        score_threshold=0.5,
    )
    keys = [(c.page, tuple(sorted(c.word_ids))) for c in candidates]
    assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# generate_doc_candidates
# ---------------------------------------------------------------------------


@patch("beat_docile.cls_candidates.generate_candidates")
def test_generate_doc_candidates_calls_all_pages(mock_gen):
    mock_gen.return_value = []
    page0 = [_make_word(0, "foo", 0, 0, 0.1, 0.02, page=0)]
    page1 = [_make_word(1, "bar", 0, 0, 0.1, 0.02, page=1)]
    words_by_page = {0: page0, 1: page1}
    fieldtypes = ["document_id"]
    generate_doc_candidates(words_by_page, fieldtypes=fieldtypes)
    assert mock_gen.call_count == 2  # once per page


@patch("beat_docile.cls_candidates.generate_candidates")
def test_generate_doc_candidates_returns_per_fieldtype_dict(mock_gen):
    mock_gen.return_value = []
    words_by_page = {0: [_make_word(0, "test", 0, 0, 0.1, 0.02)]}
    fieldtypes = ["document_id", "date_issue"]
    result = generate_doc_candidates(words_by_page, fieldtypes=fieldtypes)
    assert set(result.keys()) == {"document_id", "date_issue"}


@patch("beat_docile.cls_candidates.generate_candidates")
def test_generate_doc_candidates_aggregates_pages(mock_gen):
    c1 = CandidateSpan("document_id", [0], (0, 0, 0.1, 0.02), 0, 0.95, "A")
    c2 = CandidateSpan("document_id", [1], (0, 0, 0.1, 0.02), 1, 0.85, "B")
    mock_gen.side_effect = [[c1], [c2]]
    words_by_page = {
        0: [_make_word(0, "A", 0, 0, 0.1, 0.02, page=0)],
        1: [_make_word(1, "B", 0, 0, 0.1, 0.02, page=1)],
    }
    result = generate_doc_candidates(words_by_page, fieldtypes=["document_id"])
    assert len(result["document_id"]) == 2
    assert result["document_id"][0].score == 0.95  # sorted descending


# ---------------------------------------------------------------------------
# score_bbox_span
# ---------------------------------------------------------------------------


@patch("beat_docile.cls_candidates.load_classifier")
def test_score_bbox_span_covers_words(mock_load, simple_page):
    mock_load.return_value = _make_pipeline(prob=0.88)
    # bbox covering words 0 and 1 ("Invoice No:")
    score = score_bbox_span(
        fieldtype="document_id",
        pred_bbox=(0.0, 0.0, 0.21, 0.025),
        page_words=simple_page,
        model_dir=Path("models/classifiers"),
    )
    assert abs(score - 0.88) < 1e-4


@patch("beat_docile.cls_candidates.load_classifier")
def test_score_bbox_span_no_words_in_bbox(mock_load, simple_page):
    mock_load.return_value = _make_pipeline(prob=0.88)
    score = score_bbox_span(
        fieldtype="document_id",
        pred_bbox=(0.9, 0.9, 1.0, 1.0),
        page_words=simple_page,
        model_dir=Path("models/classifiers"),
    )
    assert score == 0.5  # neutral fallback


@patch("beat_docile.cls_candidates.load_classifier")
def test_score_bbox_span_no_model(mock_load, simple_page):
    mock_load.return_value = None
    score = score_bbox_span(
        fieldtype="iban",
        pred_bbox=(0.0, 0.0, 0.5, 0.5),
        page_words=simple_page,
        model_dir=Path("models/classifiers"),
    )
    assert score == 0.5


def test_score_bbox_span_with_margin(simple_page):
    """Tight bbox that misses word center without margin should succeed with margin=0.005."""
    # Word 2 ("12345") has centre x=0.25, y=0.01; right edge at 0.249 would miss without margin
    with patch("beat_docile.cls_candidates.load_classifier") as mock_load:
        mock_load.return_value = _make_pipeline(prob=0.75)
        score = score_bbox_span(
            fieldtype="document_id",
            pred_bbox=(0.2, 0.0, 0.249, 0.02),
            page_words=simple_page,
            model_dir=Path("models/classifiers"),
            margin=0.005,
        )
        # margin expands right to 0.254 → covers centre 0.25 → non-neutral score
        assert score != 0.5


# ---------------------------------------------------------------------------
# Integration test with real classifiers (if model dir exists)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (MODEL_DIR / "document_id.joblib").exists(),
    reason="Real classifier models not present",
)
def test_generate_candidates_real_model_smoke(simple_page):
    """Smoke test with real document_id classifier — verifies pipeline runs end-to-end."""
    candidates = generate_candidates(
        fieldtype="document_id",
        words=simple_page,
        page_w=1.0,
        page_h=1.0,
        model_dir=MODEL_DIR,
        score_threshold=0.0,
    )
    assert isinstance(candidates, list)
    for c in candidates:
        assert 0.0 <= c.score <= 1.0


@pytest.mark.skipif(
    not (MODEL_DIR / "vendor_address.joblib").exists(),
    reason="Real classifier models not present",
)
def test_generate_candidates_address_real_model(address_page):
    """Smoke test address classifier with multi-row span generation."""
    candidates = generate_candidates(
        fieldtype="vendor_address",
        words=address_page,
        page_w=1.0,
        page_h=1.0,
        model_dir=MODEL_DIR,
        score_threshold=0.0,
    )
    assert isinstance(candidates, list)
    for c in candidates:
        assert 0.0 <= c.score <= 1.0
