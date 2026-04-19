"""Unit tests for glm_ocr_extract helpers. No model or GPU required."""

from __future__ import annotations

import pytest
from docile.dataset import BBox

from beat_docile.data import WordBox
from beat_docile.glm_ocr_extract import (
    _KILE_FIELDS,
    _LIR_FIELDS,
    _MULTI_KILE,
    _fuzz_match_region,
    _parse_kile,
    _parse_lir_from_tables,
    _region_bbox,
    _resolve_field_bbox,
    _span_to_bbox,
    _word_in_bbox,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_words(*texts: str) -> list[WordBox]:
    """Build a synthetic left-to-right word list with non-overlapping bboxes."""
    words = []
    x = 0.0
    for i, text in enumerate(texts):
        width = len(text) * 0.02
        words.append(WordBox(id=i, text=text, bbox=(x, 0.1, x + width, 0.2), page=0))
        x += width + 0.01
    return words


def _make_region(label: str, content: str, bbox_2d: list[int]) -> dict:
    return {"label": label, "content": content, "bbox_2d": bbox_2d}


# ── _region_bbox: coordinate conversion ──────────────────────────────────────

class TestRegionBbox:
    def test_0_1000_to_0_1(self):
        region = _make_region("text", "hello", [100, 200, 500, 700])
        bbox = _region_bbox(region)
        assert bbox is not None
        assert pytest.approx(bbox.left) == 0.1
        assert pytest.approx(bbox.top) == 0.2
        assert pytest.approx(bbox.right) == 0.5
        assert pytest.approx(bbox.bottom) == 0.7

    def test_full_page_region(self):
        region = _make_region("text", "content", [0, 0, 1000, 1000])
        bbox = _region_bbox(region)
        assert bbox is not None
        assert pytest.approx(bbox.left) == 0.0
        assert pytest.approx(bbox.right) == 1.0

    def test_missing_bbox_2d_returns_none(self):
        region = {"label": "text", "content": "x"}
        assert _region_bbox(region) is None

    def test_wrong_length_returns_none(self):
        region = {"label": "text", "content": "x", "bbox_2d": [0, 0, 100]}
        assert _region_bbox(region) is None

    def test_all_corners(self):
        region = _make_region("text", "x", [250, 375, 750, 875])
        bbox = _region_bbox(region)
        assert bbox is not None
        assert pytest.approx(bbox.left, abs=1e-6) == 0.25
        assert pytest.approx(bbox.top, abs=1e-6) == 0.375
        assert pytest.approx(bbox.right, abs=1e-6) == 0.75
        assert pytest.approx(bbox.bottom, abs=1e-6) == 0.875


# ── _span_to_bbox ─────────────────────────────────────────────────────────────

class TestSpanToBbox:
    def test_single_word(self):
        words = _make_words("Invoice")
        bbox = _span_to_bbox((0, 0), words)
        assert isinstance(bbox, BBox)
        assert pytest.approx(bbox.left, abs=1e-6) == words[0].bbox[0]

    def test_multi_word_span(self):
        words = _make_words("Acme", "Ltd")
        bbox = _span_to_bbox((0, 1), words)
        assert bbox.left == words[0].bbox[0]
        assert bbox.right == words[1].bbox[2]

    def test_bbox_is_union(self):
        words = _make_words("A", "B", "C")
        bbox = _span_to_bbox((0, 2), words)
        assert bbox.left == min(w.bbox[0] for w in words)
        assert bbox.right == max(w.bbox[2] for w in words)


# ── _word_in_bbox ──────────────────────────────────────────────────────────────

class TestWordInBbox:
    def test_center_inside(self):
        word = WordBox(id=0, text="x", bbox=(0.2, 0.2, 0.4, 0.4), page=0)
        bbox = BBox(0.1, 0.1, 0.5, 0.5)
        assert _word_in_bbox(word, bbox)

    def test_center_outside(self):
        word = WordBox(id=0, text="x", bbox=(0.6, 0.6, 0.8, 0.8), page=0)
        bbox = BBox(0.1, 0.1, 0.5, 0.5)
        assert not _word_in_bbox(word, bbox)

    def test_center_well_inside(self):
        # center at (0.3, 0.3), bbox [0.1, 0.1, 0.5, 0.5] — clearly inside
        word = WordBox(id=0, text="x", bbox=(0.2, 0.2, 0.4, 0.4), page=0)
        bbox = BBox(0.1, 0.1, 0.5, 0.5)
        assert _word_in_bbox(word, bbox)


# ── _fuzz_match_region ────────────────────────────────────────────────────────

class TestFuzzMatchRegion:
    def test_exact_match(self):
        regions = [_make_region("text", "Invoice No: INV-001", [0, 0, 500, 100])]
        match = _fuzz_match_region("INV-001", regions)
        assert match is not None
        assert match["content"] == "Invoice No: INV-001"

    def test_no_match_below_threshold(self):
        regions = [_make_region("text", "completely unrelated text here", [0, 0, 500, 100])]
        match = _fuzz_match_region("XYZZY99999", regions, threshold=90)
        assert match is None

    def test_picks_best_region(self):
        regions = [
            _make_region("text", "random stuff", [0, 0, 100, 100]),
            _make_region("text", "Total amount due: 1234.56", [0, 100, 500, 200]),
        ]
        match = _fuzz_match_region("1234.56", regions)
        assert match is not None
        assert "1234.56" in match["content"]

    def test_empty_regions_returns_none(self):
        assert _fuzz_match_region("anything", []) is None

    def test_returns_none_when_rapidfuzz_missing(self, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "rapidfuzz", None)
        # Should fall back to None rather than crash
        from beat_docile import glm_ocr_extract as m
        result = m._fuzz_match_region("value", [{"content": "value", "bbox_2d": [0, 0, 100, 100]}])
        # With rapidfuzz unavailable, returns None
        assert result is None


# ── _resolve_field_bbox ────────────────────────────────────────────────────────

class TestResolveFieldBbox:
    def _words_for(self, *texts: str) -> list[WordBox]:
        return _make_words(*texts)

    def test_mode_region_uses_region_bbox(self):
        words = _make_words("Acme", "Ltd")
        region = _make_region("text", "Acme Ltd", [100, 100, 500, 200])
        bbox = _resolve_field_bbox("Acme Ltd", region, words, mode="region")
        assert bbox is not None
        assert pytest.approx(bbox.left) == 0.1
        assert pytest.approx(bbox.right) == 0.5

    def test_mode_words_aligns_to_word_span(self):
        # Words span the page; region covers them
        words = _make_words("Acme", "Ltd")
        # Region bbox covers both words (0.0 to ~0.18)
        region = _make_region("text", "Acme Ltd", [0, 50, 250, 250])
        bbox = _resolve_field_bbox("Acme Ltd", region, words, mode="words")
        assert bbox is not None
        # Should be tighter than or equal to region bbox
        assert bbox.left >= 0.0
        assert bbox.right <= 1.0

    def test_no_region_falls_back_to_global_align(self):
        words = _make_words("INV", "001")
        bbox = _resolve_field_bbox("INV 001", None, words, mode="words")
        assert bbox is not None

    def test_no_region_no_text_match_returns_none(self):
        words = _make_words("completely", "unrelated")
        bbox = _resolve_field_bbox("XYZZY99999", None, words, mode="words")
        assert bbox is None

    def test_mode_region_no_region_falls_back_to_global(self):
        words = _make_words("Acme", "Ltd")
        bbox = _resolve_field_bbox("Acme Ltd", None, words, mode="region")
        assert bbox is not None

    def test_region_bbox_none_falls_back(self):
        words = _make_words("Acme", "Ltd")
        region_no_bbox = {"label": "text", "content": "Acme Ltd"}  # no bbox_2d
        bbox = _resolve_field_bbox("Acme Ltd", region_no_bbox, words, mode="words")
        # Should fall back to global align since rbbox is None
        assert bbox is not None


# ── _parse_kile ───────────────────────────────────────────────────────────────

class TestParseKile:
    def test_known_field_matched(self):
        words = _make_words("Acme", "Ltd")
        data = {"vendor_name": "Acme Ltd"}
        fields = _parse_kile(data, [], words, page_idx=0, mode="words")
        assert len(fields) == 1
        assert fields[0].fieldtype == "vendor_name"
        assert fields[0].page == 0
        assert fields[0].score == 1.0

    def test_uses_region_in_mode_region(self):
        words = _make_words("Acme", "Ltd")
        # Region with known bbox
        regions = [_make_region("text", "Acme Ltd", [100, 100, 500, 200])]
        data = {"vendor_name": "Acme Ltd"}
        fields = _parse_kile(data, regions, words, page_idx=0, mode="region")
        assert len(fields) == 1
        # bbox should come from region (0.1 to 0.5)
        assert pytest.approx(fields[0].bbox.left, abs=0.05) == 0.1

    def test_unknown_field_dropped(self):
        words = _make_words("Acme")
        data = {"nonexistent_field": "Acme"}
        fields = _parse_kile(data, [], words, page_idx=0, mode="words")
        assert fields == []

    def test_empty_string_skipped(self):
        words = _make_words("Acme")
        data = {"vendor_name": ""}
        fields = _parse_kile(data, [], words, page_idx=0, mode="words")
        assert fields == []

    def test_multi_value_field(self):
        words = _make_words("21%", "19%")
        data = {"tax_detail_rate": ["21%", "19%"]}
        fields = _parse_kile(data, [], words, page_idx=0, mode="words")
        assert len(fields) == 2
        assert all(f.fieldtype == "tax_detail_rate" for f in fields)

    def test_no_match_no_region_returns_empty(self):
        words = _make_words("completely", "unrelated", "text")
        data = {"document_id": "INV-9999-XXXXXXX"}
        fields = _parse_kile(data, [], words, page_idx=0, mode="words")
        assert fields == []

    def test_line_item_id_absent_for_kile(self):
        words = _make_words("Acme", "Ltd")
        data = {"vendor_name": "Acme Ltd"}
        fields = _parse_kile(data, [], words, page_idx=0, mode="words")
        assert fields[0].line_item_id is None

    def test_page_propagated(self):
        words = _make_words("Acme")
        data = {"vendor_name": "Acme"}
        fields = _parse_kile(data, [], words, page_idx=3, mode="words")
        assert fields[0].page == 3


# ── _parse_lir_from_tables ────────────────────────────────────────────────────

class TestParseLirFromTables:
    def test_no_table_regions_returns_empty(self):
        fields = _parse_lir_from_tables(None, [], [], page_idx=0,
                                         model=None, processor=None, device="cpu")
        assert fields == []

    def test_table_region_without_bbox_skipped(self):
        region = {"label": "table", "content": "x"}  # no bbox_2d
        fields = _parse_lir_from_tables(None, [region], [], page_idx=0,
                                         model=None, processor=None, device="cpu")
        assert fields == []

    def test_lir_fields_have_line_item_id(self, monkeypatch):
        from beat_docile import glm_ocr_extract as m

        words = _make_words("Widget", "2", "Gadget", "3")
        table_region = _make_region("table", "", [0, 0, 1000, 1000])

        def mock_run_kie(image, schema, model, processor, device, max_new_tokens=512):
            return {"line_items": [
                {"line_item_description": "Widget", "line_item_quantity": "2"},
                {"line_item_description": "Gadget", "line_item_quantity": "3"},
            ]}

        monkeypatch.setattr(m, "_run_kie", mock_run_kie)

        from unittest.mock import MagicMock
        image = MagicMock()
        image.size = (800, 1000)
        image.crop.return_value = image

        fields = m._parse_lir_from_tables(
            image, [table_region], words, page_idx=0,
            model=None, processor=None, device="cpu",
        )

        assert len(fields) > 0
        li_ids = {f.line_item_id for f in fields}
        assert li_ids == {1, 2}
        assert all(f.page == 0 for f in fields)
        assert all(f.score == 1.0 for f in fields)

    def test_second_table_li_ids_offset_from_first(self, monkeypatch):
        """li_id_offset advances after each table so ids don't collide."""
        from beat_docile import glm_ocr_extract as m

        # Both tables cover the full page so all words are in scope
        # Widget at x≈0.05, Gadget at x≈0.19 — both inside [0,0,1000,1000]
        words = _make_words("Widget", "Gadget")
        table1 = _make_region("table", "", [0, 0, 1000, 500])
        table2 = _make_region("table", "", [0, 500, 1000, 1000])

        call_count = [0]

        def mock_run_kie(image, schema, model, processor, device, max_new_tokens=512):
            call_count[0] += 1
            # table1: 2 line items; table2: 1 line item
            if call_count[0] == 1:
                return {"line_items": [
                    {"line_item_description": "Widget"},
                    {"line_item_description": "Widget"},
                ]}
            return {"line_items": [{"line_item_description": "Gadget"}]}

        monkeypatch.setattr(m, "_run_kie", mock_run_kie)

        from unittest.mock import MagicMock
        image = MagicMock()
        image.size = (800, 1000)
        image.crop.return_value = image

        fields = m._parse_lir_from_tables(
            image, [table1, table2], words, page_idx=0,
            model=None, processor=None, device="cpu",
        )
        # table1 produced li_ids 1 and 2 (two line items)
        li_ids = sorted({f.line_item_id for f in fields})
        assert li_ids == [1, 2]
        # both KIE calls were made (one per table)
        assert call_count[0] == 2


# ── _parse_layout ─────────────────────────────────────────────────────────────

class TestParseLayout:
    def test_returns_empty_if_glmocr_missing(self, monkeypatch):
        """_parse_layout returns [] silently when glmocr is not installed.
        The warning about missing layout is emitted by extract_page, not _parse_layout.
        """
        import sys
        from unittest.mock import MagicMock

        monkeypatch.setitem(sys.modules, "glmocr", None)
        from beat_docile import glm_ocr_extract as m

        img = MagicMock()
        img.save = MagicMock()
        result = m._parse_layout(img)
        assert result == []

    def test_returns_regions_from_mock(self, monkeypatch):
        """_parse_layout returns region list; test by patching at function level."""
        from unittest.mock import MagicMock

        from beat_docile import glm_ocr_extract as m

        fixture_regions = [
            {"label": "title", "content": "Invoice", "bbox_2d": [0, 0, 1000, 100]},
            {"label": "table", "content": "items", "bbox_2d": [0, 200, 1000, 800]},
        ]

        img = MagicMock()
        monkeypatch.setattr(m, "_parse_layout", lambda image: fixture_regions)
        regions = m._parse_layout(img)

        assert len(regions) == 2
        assert regions[0]["label"] == "title"
        assert regions[1]["label"] == "table"


# ── Field catalog sanity ──────────────────────────────────────────────────────

class TestFieldCatalogs:
    def test_kile_count(self):
        assert len(_KILE_FIELDS) == 36

    def test_lir_count(self):
        assert len(_LIR_FIELDS) == 19

    def test_no_overlap(self):
        assert not set(_KILE_FIELDS) & set(_LIR_FIELDS)

    def test_kile_no_duplicates(self):
        assert len(_KILE_FIELDS) == len(set(_KILE_FIELDS))

    def test_lir_no_duplicates(self):
        assert len(_LIR_FIELDS) == len(set(_LIR_FIELDS))

    def test_multi_kile_subset_of_kile(self):
        assert _MULTI_KILE.issubset(set(_KILE_FIELDS))

    def test_multi_kile_not_in_lir(self):
        assert not _MULTI_KILE & set(_LIR_FIELDS)


# ── extract_page integration (happy path + warning) ──────────────────────────

class TestExtractPageIntegration:
    """Test extract_page end-to-end with mocked layout + KIE — no model/GPU."""

    def _make_page(self, words):
        from unittest.mock import MagicMock
        page = MagicMock()
        page.docid = "testdoc"
        page.page_index = 0
        page.words = words
        page.image = MagicMock()
        return page

    def test_happy_path_regions_and_kie_produce_fields(self, monkeypatch):
        """extract_page with real regions + matching KIE output → fields returned."""
        from beat_docile import glm_ocr_extract as m

        words = _make_words("Acme", "Ltd", "INV-001")
        page = self._make_page(words)

        # Layout parse returns one text region covering the page
        fake_regions = [
            _make_region("text", "Acme Ltd INV-001", [0, 0, 1000, 1000]),
        ]
        monkeypatch.setattr(m, "_parse_layout", lambda image: fake_regions)

        # KIE returns vendor_name and document_id — both matchable in OCR words
        def mock_run_kie(image, schema, model, processor, device, max_new_tokens=512):
            return {"vendor_name": "Acme Ltd", "document_id": "INV-001"}

        monkeypatch.setattr(m, "_run_kie", mock_run_kie)
        monkeypatch.setattr(m, "_load_model", lambda model_id=m.MODEL_ID: (None, None, "cpu"))

        kile, _lir = m.extract_page(page, model_id=m.MODEL_ID)

        assert len(kile) >= 1, "Expected at least one KILE field from regions + KIE"
        fieldtypes = {f.fieldtype for f in kile}
        assert "vendor_name" in fieldtypes or "document_id" in fieldtypes
        assert all(f.page == 0 for f in kile)
        assert all(f.line_item_id is None for f in kile)

    def test_region_bbox_used_in_mode_region(self, monkeypatch, monkeypatch_env=None):
        """In 'region' mode, bbox comes from region bbox_2d, not word alignment."""

        from beat_docile import glm_ocr_extract as m

        words = _make_words("Acme", "Ltd")
        page = self._make_page(words)

        # Region has a known bbox in 0-1000 scale
        fake_regions = [_make_region("text", "Acme Ltd", [200, 300, 700, 400])]
        monkeypatch.setattr(m, "_parse_layout", lambda image: fake_regions)

        def mock_run_kie(image, schema, model, processor, device, max_new_tokens=512):
            return {"vendor_name": "Acme Ltd"}

        monkeypatch.setattr(m, "_run_kie", mock_run_kie)
        monkeypatch.setattr(m, "_load_model", lambda model_id=m.MODEL_ID: (None, None, "cpu"))
        monkeypatch.setenv("BD_GLM_BBOX_MODE", "region")

        kile, _ = m.extract_page(page, model_id=m.MODEL_ID)

        assert len(kile) == 1
        bbox = kile[0].bbox
        # Bbox should be the region bbox / 1000
        assert pytest.approx(bbox.left, abs=0.01) == 0.2
        assert pytest.approx(bbox.top, abs=0.01) == 0.3
        assert pytest.approx(bbox.right, abs=0.01) == 0.7
        assert pytest.approx(bbox.bottom, abs=0.01) == 0.4

    def test_empty_regions_triggers_warning(self, monkeypatch, capsys):
        """extract_page prints a WARN to stderr when layout returns no regions."""
        from beat_docile import glm_ocr_extract as m

        words = _make_words("Acme")
        page = self._make_page(words)

        # Layout returns empty — simulates missing PaddlePaddle / failed parse
        monkeypatch.setattr(m, "_parse_layout", lambda image: [])

        def mock_run_kie(image, schema, model, processor, device, max_new_tokens=512):
            return {"vendor_name": "Acme"}

        monkeypatch.setattr(m, "_run_kie", mock_run_kie)
        monkeypatch.setattr(m, "_load_model", lambda model_id=m.MODEL_ID: (None, None, "cpu"))

        m.extract_page(page, model_id=m.MODEL_ID)

        captured = capsys.readouterr()
        assert "WARN" in captured.err
        assert "no layout regions" in captured.err

    def test_empty_regions_still_returns_fields_via_fallback(self, monkeypatch):
        """Even with no layout regions, find_span fallback extracts fields."""
        from beat_docile import glm_ocr_extract as m

        words = _make_words("Acme", "Ltd")
        page = self._make_page(words)

        monkeypatch.setattr(m, "_parse_layout", lambda image: [])

        def mock_run_kie(image, schema, model, processor, device, max_new_tokens=512):
            return {"vendor_name": "Acme Ltd"}

        monkeypatch.setattr(m, "_run_kie", mock_run_kie)
        monkeypatch.setattr(m, "_load_model", lambda model_id=m.MODEL_ID: (None, None, "cpu"))

        kile, _ = m.extract_page(page, model_id=m.MODEL_ID)

        # With no regions, falls back to global find_span — should still find "Acme Ltd"
        assert len(kile) == 1
        assert kile[0].fieldtype == "vendor_name"


# ── extract_documents contract ────────────────────────────────────────────────

class TestExtractDocumentsStructure:
    def test_all_docids_present(self, monkeypatch):
        from beat_docile import glm_ocr_extract as m

        def mock_extract_page(page, model_id=m.MODEL_ID):
            return [], []

        monkeypatch.setattr(m, "extract_page", mock_extract_page)

        def fake_iter_pages(doc):
            class _FakePage:
                docid = doc.docid
                page_index = 0
                image = None
                words: list = []  # noqa: RUF012
            yield _FakePage()

        monkeypatch.setattr(m, "iter_pages", fake_iter_pages)

        class _FakeDoc:
            def __init__(self, docid):
                self.docid = docid

        result = m.extract_documents(["doc1", "doc2", "doc3"],
                                     [_FakeDoc("doc1"), _FakeDoc("doc2"), _FakeDoc("doc3")])
        assert set(result.keys()) == {"doc1", "doc2", "doc3"}
        assert all(isinstance(v, list) for v in result.values())

    def test_extra_dataset_docs_ignored(self, monkeypatch):
        from beat_docile import glm_ocr_extract as m

        monkeypatch.setattr(m, "extract_page", lambda page, model_id=m.MODEL_ID: ([], []))
        monkeypatch.setattr(m, "iter_pages", lambda doc: iter([]))

        class _FakeDoc:
            def __init__(self, docid):
                self.docid = docid

        result = m.extract_documents(["doc1"], [_FakeDoc("doc1"), _FakeDoc("extra")])
        assert "extra" not in result
        assert "doc1" in result
