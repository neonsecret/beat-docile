"""Basic unit tests for donut_extract helpers. No model/GPU required."""

from __future__ import annotations

from beat_docile.donut_extract import (
    _KILE_FIELDS,
    _LIR_FIELDS,
    _normalize_kile,
    _normalize_line_items,
    _parse_json_output,
)

# ── _parse_json_output ────────────────────────────────────────────────────────

class TestParseJsonOutput:
    def test_clean_json(self):
        raw = '{"kile": {"vendor_name": "Acme"}, "line_items": []}'
        result = _parse_json_output(raw)
        assert result == {"kile": {"vendor_name": "Acme"}, "line_items": []}

    def test_markdown_code_fence(self):
        raw = '```json\n{"kile": {"document_id": "INV-1"}, "line_items": []}\n```'
        result = _parse_json_output(raw)
        assert result["kile"]["document_id"] == "INV-1"

    def test_markdown_code_fence_no_lang(self):
        raw = '```\n{"kile": {}, "line_items": []}\n```'
        result = _parse_json_output(raw)
        assert result == {"kile": {}, "line_items": []}

    def test_json_embedded_in_preamble(self):
        raw = 'Here is the extraction:\n{"kile": {"vendor_name": "X"}, "line_items": []}'
        result = _parse_json_output(raw)
        assert result["kile"]["vendor_name"] == "X"

    def test_invalid_returns_empty(self):
        result = _parse_json_output("not json at all")
        assert result == {}

    def test_empty_string(self):
        result = _parse_json_output("")
        assert result == {}

    def test_partial_json_fallback(self):
        # Truncated output — JSON object still parseable if it closed
        raw = '{"kile": {"vendor_name": "Acme Ltd"}, "line_items": []}'
        result = _parse_json_output(raw)
        assert result["kile"]["vendor_name"] == "Acme Ltd"


# ── _normalize_kile ───────────────────────────────────────────────────────────

class TestNormalizeKile:
    def test_known_fields_pass_through(self):
        raw = {"vendor_name": "Acme", "document_id": "001"}
        result = _normalize_kile(raw)
        assert result == {"vendor_name": "Acme", "document_id": "001"}

    def test_unknown_fields_dropped(self):
        raw = {"vendor_name": "Acme", "nonexistent_field": "x"}
        result = _normalize_kile(raw)
        assert "nonexistent_field" not in result
        assert "vendor_name" in result

    def test_list_values_kept(self):
        raw = {"tax_detail_rate": ["21%", "15%"]}
        result = _normalize_kile(raw)
        assert result["tax_detail_rate"] == ["21%", "15%"]

    def test_int_coerced_to_str(self):
        raw = {"document_id": 1234}
        result = _normalize_kile(raw)
        assert result["document_id"] == "1234"

    def test_float_coerced_to_str(self):
        raw = {"amount_total_gross": 1210.0}
        result = _normalize_kile(raw)
        assert result["amount_total_gross"] == "1210.0"

    def test_empty_string_dropped(self):
        raw = {"vendor_name": "", "document_id": "001"}
        result = _normalize_kile(raw)
        assert "vendor_name" not in result

    def test_none_dropped(self):
        raw = {"vendor_name": None, "document_id": "001"}
        result = _normalize_kile(raw)
        assert "vendor_name" not in result

    def test_empty_list_dropped(self):
        raw = {"tax_detail_rate": []}
        result = _normalize_kile(raw)
        assert "tax_detail_rate" not in result

    def test_list_with_empty_strings_dropped(self):
        raw = {"tax_detail_rate": ["", " "]}
        result = _normalize_kile(raw)
        assert "tax_detail_rate" not in result

    def test_all_kile_fields_recognized(self):
        raw = {ft: "value" for ft in _KILE_FIELDS}
        result = _normalize_kile(raw)
        assert set(result.keys()) == set(_KILE_FIELDS)


# ── _normalize_line_items ─────────────────────────────────────────────────────

class TestNormalizeLineItems:
    def test_basic_item(self):
        items = [{"line_item_description": "Widget", "line_item_quantity": "2"}]
        result = _normalize_line_items(items)
        assert len(result) == 1
        assert result[0] == {"line_item_description": "Widget", "line_item_quantity": "2"}

    def test_unknown_keys_dropped(self):
        items = [{"line_item_description": "Widget", "unknown_key": "x"}]
        result = _normalize_line_items(items)
        assert "unknown_key" not in result[0]

    def test_empty_values_dropped(self):
        items = [{"line_item_description": "Widget", "line_item_quantity": ""}]
        result = _normalize_line_items(items)
        assert "line_item_quantity" not in result[0]

    def test_non_dict_items_skipped(self):
        result = _normalize_line_items(["not a dict", 42, None])
        assert result == []

    def test_empty_item_skipped(self):
        items = [{"unknown": "x"}]  # all keys unknown → empty after filter
        result = _normalize_line_items(items)
        assert result == []

    def test_int_coerced_to_str(self):
        items = [{"line_item_position": 1, "line_item_quantity": 2}]
        result = _normalize_line_items(items)
        assert result[0]["line_item_position"] == "1"
        assert result[0]["line_item_quantity"] == "2"

    def test_multiple_items(self):
        items = [
            {"line_item_description": "A", "line_item_quantity": "1"},
            {"line_item_description": "B", "line_item_quantity": "3"},
        ]
        result = _normalize_line_items(items)
        assert len(result) == 2

    def test_all_lir_fields_recognized(self):
        item = {ft: "value" for ft in _LIR_FIELDS}
        result = _normalize_line_items([item])
        assert set(result[0].keys()) == set(_LIR_FIELDS)


# ── Field catalog sanity checks ───────────────────────────────────────────────

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
