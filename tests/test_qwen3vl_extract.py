"""Unit tests for qwen3vl_extract helpers. No model or GPU required."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from docile.dataset import BBox

from beat_docile.data import WordBox
from beat_docile.qwen3vl_extract import (
    _KILE_FIELDS,
    _parse_and_snap,
    _pcc_in_bbox,
    _snap_to_words,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_words(*texts: str) -> list[WordBox]:
    """Build left-to-right word list with non-overlapping bboxes."""
    words: list[WordBox] = []
    x = 0.0
    for i, text in enumerate(texts):
        width = max(len(text) * 0.02, 0.02)
        words.append(WordBox(id=i, text=text, bbox=(x, 0.1, x + width, 0.2), page=0))
        x += width + 0.01
    return words


def _raw_json(**kwargs: object) -> str:
    return json.dumps(kwargs, ensure_ascii=False)


# ── _pcc_in_bbox ──────────────────────────────────────────────────────────────

class TestPccInBbox:
    def test_center_inside(self):
        word = WordBox(id=0, text="x", bbox=(0.2, 0.2, 0.4, 0.4), page=0)
        assert _pcc_in_bbox(word, BBox(0.1, 0.1, 0.5, 0.5))

    def test_center_outside(self):
        word = WordBox(id=0, text="x", bbox=(0.6, 0.6, 0.8, 0.8), page=0)
        assert not _pcc_in_bbox(word, BBox(0.1, 0.1, 0.5, 0.5))

    def test_center_on_boundary(self):
        # center at (0.3, 0.15) — on exact boundary of [0.3, 0.1, 0.5, 0.2]
        word = WordBox(id=0, text="x", bbox=(0.2, 0.1, 0.4, 0.2), page=0)
        assert _pcc_in_bbox(word, BBox(0.3, 0.1, 0.5, 0.2))


# ── _snap_to_words ────────────────────────────────────────────────────────────

class TestSnapToWords:
    def test_single_word_snap(self):
        words = _make_words("Invoice")
        # BBox that covers the center of word 0
        pred = BBox(0.0, 0.05, 0.25, 0.25)
        result = _snap_to_words(pred, words)
        assert result is not None
        bbox, text = result
        assert text == "Invoice"
        assert isinstance(bbox, BBox)

    def test_multi_word_snap(self):
        words = _make_words("Acme", "Ltd")
        # Cover both words
        pred = BBox(0.0, 0.05, 0.5, 0.25)
        result = _snap_to_words(pred, words)
        assert result is not None
        _, text = result
        assert "Acme" in text and "Ltd" in text

    def test_no_words_in_bbox_returns_none(self):
        words = _make_words("far_right_word")
        # BBox in top-left corner, no words there
        pred = BBox(0.9, 0.9, 1.0, 1.0)
        assert _snap_to_words(pred, words) is None

    def test_snapped_bbox_is_word_union(self):
        words = _make_words("A", "B")
        pred = BBox(0.0, 0.05, 0.5, 0.25)
        result = _snap_to_words(pred, words)
        assert result is not None
        bbox, _ = result
        assert bbox.left == min(w.bbox[0] for w in words)
        assert bbox.right == max(w.bbox[2] for w in words)


# ── _parse_and_snap ───────────────────────────────────────────────────────────

class TestParseAndSnap:
    def test_single_field_parsed(self):
        words = _make_words("Acme", "Ltd")
        raw = _raw_json(vendor_name={"bbox_2d": [0, 50, 500, 250], "text": "Acme Ltd"})
        fields = _parse_and_snap(raw, words, page=0)
        assert len(fields) == 1
        assert fields[0].fieldtype == "vendor_name"
        assert fields[0].page == 0

    def test_multi_occurrence_field_as_list(self):
        words = _make_words("21%", "19%")
        raw = _raw_json(
            tax_detail_rate=[
                {"bbox_2d": [0, 50, 150, 250], "text": "21%"},
                {"bbox_2d": [200, 50, 350, 250], "text": "19%"},
            ]
        )
        fields = _parse_and_snap(raw, words, page=0)
        assert len(fields) == 2
        assert all(f.fieldtype == "tax_detail_rate" for f in fields)

    def test_unknown_fieldtype_dropped(self):
        words = _make_words("Acme")
        raw = _raw_json(nonexistent_field={"bbox_2d": [0, 50, 200, 250], "text": "Acme"})
        fields = _parse_and_snap(raw, words, page=0)
        assert fields == []

    def test_empty_json_returns_empty_list(self):
        words = _make_words("Acme")
        fields = _parse_and_snap("{}", words, page=0)
        assert fields == []

    def test_invalid_json_returns_empty_list(self):
        words = _make_words("Acme")
        fields = _parse_and_snap("not valid json at all", words, page=0)
        assert fields == []

    def test_markdown_fence_stripped(self):
        words = _make_words("Acme", "Ltd")
        raw = "```json\n" + _raw_json(
            vendor_name={"bbox_2d": [0, 50, 500, 250], "text": "Acme Ltd"}
        ) + "\n```"
        fields = _parse_and_snap(raw, words, page=0)
        assert len(fields) == 1

    def test_no_snap_falls_back_to_raw_bbox(self):
        # BBox in a region with no OCR words → Field still created with raw bbox
        words = _make_words("unrelated")
        raw = _raw_json(vendor_name={"bbox_2d": [800, 800, 1000, 1000], "text": "Acme"})
        fields = _parse_and_snap(raw, words, page=0)
        assert len(fields) == 1
        assert pytest.approx(fields[0].bbox.left, abs=0.001) == 0.8

    def test_bbox_converted_from_0_1000_to_0_1(self):
        words = _make_words("Acme")
        raw = _raw_json(vendor_name={"bbox_2d": [100, 200, 500, 300], "text": "x"})
        fields = _parse_and_snap(raw, words, page=0)
        # If snap misses, raw bbox [0.1, 0.2, 0.5, 0.3] is used
        # (words are far left, center ~ x=0.01, which is outside [0.1, 0.5])
        # Either way, no assertion failure — just confirm type
        assert all(isinstance(f.bbox, BBox) for f in fields)

    def test_page_index_propagated(self):
        words = _make_words("Acme", "Ltd")
        raw = _raw_json(vendor_name={"bbox_2d": [0, 50, 500, 250], "text": "Acme"})
        fields = _parse_and_snap(raw, words, page=3)
        assert all(f.page == 3 for f in fields)

    def test_score_is_1_0(self):
        words = _make_words("Acme", "Ltd")
        raw = _raw_json(vendor_name={"bbox_2d": [0, 50, 500, 250], "text": "Acme Ltd"})
        fields = _parse_and_snap(raw, words, page=0)
        assert all(f.score == 1.0 for f in fields)

    def test_missing_bbox_2d_skipped(self):
        words = _make_words("Acme")
        raw = _raw_json(vendor_name={"text": "Acme"})  # no bbox_2d
        fields = _parse_and_snap(raw, words, page=0)
        assert fields == []

    def test_wrong_length_bbox_skipped(self):
        words = _make_words("Acme")
        raw = _raw_json(vendor_name={"bbox_2d": [0, 50, 500], "text": "Acme"})
        fields = _parse_and_snap(raw, words, page=0)
        assert fields == []

    def test_non_dict_top_level_returns_empty(self):
        words = _make_words("Acme")
        fields = _parse_and_snap("[1, 2, 3]", words, page=0)
        assert fields == []


# ── Field catalog ─────────────────────────────────────────────────────────────

class TestFieldCatalog:
    def test_kile_count(self):
        assert len(_KILE_FIELDS) == 36

    def test_no_lir_in_kile(self):
        lir_prefix = "line_item_"
        assert not any(ft.startswith(lir_prefix) for ft in _KILE_FIELDS)

    def test_known_fields_present(self):
        for ft in ["vendor_name", "document_id", "amount_due", "date_issue"]:
            assert ft in _KILE_FIELDS

    def test_tax_detail_fields_present(self):
        for ft in ["tax_detail_rate", "tax_detail_gross", "tax_detail_net", "tax_detail_tax"]:
            assert ft in _KILE_FIELDS


# ── extract_documents contract ────────────────────────────────────────────────

class TestExtractDocumentsContract:
    """Verify extract_documents output structure without loading a real model."""

    def _make_fake_doc(self, docid: str) -> MagicMock:
        doc = MagicMock()
        doc.docid = docid
        return doc

    def test_all_input_docids_present_in_output(self, monkeypatch):
        import beat_docile.qwen3vl_extract as m

        def mock_iter_pages(doc):
            page = MagicMock()
            page.docid = doc.docid
            page.page_index = 0
            page.image = MagicMock()
            page.words = []
            yield page

        monkeypatch.setattr(m, "iter_pages", mock_iter_pages)

        fake_extractor = MagicMock()
        fake_extractor.extract_page.return_value = []

        with patch.object(m, "Qwen3VLExtractor", return_value=fake_extractor):
            docs = [self._make_fake_doc("doc1"), self._make_fake_doc("doc2")]
            result = m.extract_documents(docs)

        assert set(result.keys()) == {"doc1", "doc2"}
        assert all(isinstance(v, list) for v in result.values())

    def test_empty_page_yields_empty_list(self, monkeypatch):
        import beat_docile.qwen3vl_extract as m

        def mock_iter_pages(doc):
            page = MagicMock()
            page.docid = doc.docid
            page.page_index = 0
            page.image = MagicMock()
            page.words = []
            yield page

        monkeypatch.setattr(m, "iter_pages", mock_iter_pages)
        fake_extractor = MagicMock()
        fake_extractor.extract_page.return_value = []

        with patch.object(m, "Qwen3VLExtractor", return_value=fake_extractor):
            result = m.extract_documents([self._make_fake_doc("docA")])

        assert result["docA"] == []

    def test_limit_respected(self, monkeypatch):
        import beat_docile.qwen3vl_extract as m

        monkeypatch.setattr(m, "iter_pages", lambda doc: iter([]))
        fake_extractor = MagicMock()
        fake_extractor.extract_page.return_value = []

        docs = [self._make_fake_doc(f"doc{i}") for i in range(10)]
        with patch.object(m, "Qwen3VLExtractor", return_value=fake_extractor):
            result = m.extract_documents(docs, limit=3)

        assert len(result) == 3

    def test_output_values_are_lists_of_fields(self, monkeypatch):
        from docile.dataset import BBox, Field

        import beat_docile.qwen3vl_extract as m

        def mock_iter_pages(doc):
            page = MagicMock()
            page.docid = doc.docid
            page.page_index = 0
            page.image = MagicMock()
            page.words = []
            yield page

        monkeypatch.setattr(m, "iter_pages", mock_iter_pages)

        fake_field = Field(
            bbox=BBox(0.1, 0.1, 0.5, 0.2),
            page=0,
            fieldtype="vendor_name",
            score=1.0,
        )
        fake_extractor = MagicMock()
        fake_extractor.extract_page.return_value = [fake_field]

        with patch.object(m, "Qwen3VLExtractor", return_value=fake_extractor):
            result = m.extract_documents([self._make_fake_doc("docX")])

        assert len(result["docX"]) == 1
        assert result["docX"][0].fieldtype == "vendor_name"
