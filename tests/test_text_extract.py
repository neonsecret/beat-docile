"""Unit tests for text_extract.py — text-only Claude extraction parser.

No API calls are made. Tests cover _parse_text_response only.
"""

from __future__ import annotations

import json

from beat_docile.text_extract import _parse_text_response

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_json(fields=None, line_items=None) -> str:
    payload = {}
    if fields is not None:
        payload["fields"] = fields
    if line_items is not None:
        payload["line_items"] = line_items
    return json.dumps(payload)


# ─────────────────────────────────────────────────────────────────────────────
# Basic parsing
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_empty_response():
    result = _parse_text_response("")
    assert result == []


def test_parse_invalid_json():
    result = _parse_text_response("not json at all {{{")
    assert result == []


def test_parse_valid_single_field():
    raw = _make_json(fields=[{"fieldtype": "document_id", "text": "INV-2024-001", "score": 0.95}])
    result = _parse_text_response(raw)
    assert len(result) == 1
    f = result[0]
    assert f.fieldtype == "document_id"
    assert f.text == "INV-2024-001"
    assert f.score == 0.95
    assert f.line_item_id is None


def test_parse_score_defaults_to_0_8():
    raw = _make_json(fields=[{"fieldtype": "date_issue", "text": "2024-01-15"}])
    result = _parse_text_response(raw)
    assert len(result) == 1
    assert result[0].score == 0.8


def test_parse_unknown_fieldtype_ignored():
    raw = _make_json(fields=[
        {"fieldtype": "unknown_field", "text": "something", "score": 0.9},
        {"fieldtype": "document_id", "text": "INV-001", "score": 0.9},
    ])
    result = _parse_text_response(raw)
    # Only known KILE type should survive
    assert len(result) == 1
    assert result[0].fieldtype == "document_id"


def test_parse_empty_text_ignored():
    raw = _make_json(fields=[
        {"fieldtype": "document_id", "text": "", "score": 0.9},
        {"fieldtype": "date_issue", "text": "2024-01-15", "score": 0.9},
    ])
    result = _parse_text_response(raw)
    assert len(result) == 1
    assert result[0].fieldtype == "date_issue"


def test_parse_whitespace_only_text_ignored():
    raw = _make_json(fields=[{"fieldtype": "document_id", "text": "   ", "score": 0.9}])
    result = _parse_text_response(raw)
    assert len(result) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Markdown fence stripping
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_strips_markdown_fence():
    raw = '```json\n{"fields":[{"fieldtype":"document_id","text":"INV-001","score":0.9}],"line_items":[]}\n```'
    result = _parse_text_response(raw)
    assert len(result) == 1
    assert result[0].text == "INV-001"


def test_parse_strips_plain_fence():
    raw = '```\n{"fields":[{"fieldtype":"date_issue","text":"2024-01-15","score":0.85}],"line_items":[]}\n```'
    result = _parse_text_response(raw)
    assert len(result) == 1
    assert result[0].fieldtype == "date_issue"


# ─────────────────────────────────────────────────────────────────────────────
# Multiple KILE fields
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_multiple_kile_fields():
    raw = _make_json(fields=[
        {"fieldtype": "document_id", "text": "INV-2024-001", "score": 0.95},
        {"fieldtype": "date_issue", "text": "15.01.2024", "score": 0.9},
        {"fieldtype": "vendor_name", "text": "Acme GmbH", "score": 0.85},
        {"fieldtype": "amount_total_gross", "text": "1,234.56", "score": 0.8},
    ])
    result = _parse_text_response(raw)
    assert len(result) == 4
    ft_set = {f.fieldtype for f in result}
    assert ft_set == {"document_id", "date_issue", "vendor_name", "amount_total_gross"}


def test_parse_multiple_tax_detail_rows():
    """Multiple entries of same fieldtype (tax_detail_*) should all be kept."""
    raw = _make_json(fields=[
        {"fieldtype": "tax_detail_rate", "text": "19%", "score": 0.9},
        {"fieldtype": "tax_detail_rate", "text": "7%", "score": 0.9},
        {"fieldtype": "tax_detail_gross", "text": "1,190.00", "score": 0.85},
        {"fieldtype": "tax_detail_gross", "text": "107.00", "score": 0.85},
    ])
    result = _parse_text_response(raw)
    rates = [f for f in result if f.fieldtype == "tax_detail_rate"]
    grosses = [f for f in result if f.fieldtype == "tax_detail_gross"]
    assert len(rates) == 2
    assert len(grosses) == 2
    assert {r.text for r in rates} == {"19%", "7%"}


# ─────────────────────────────────────────────────────────────────────────────
# Multi-line addresses
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_multiline_address():
    """Address with embedded \\n should be preserved in the text."""
    address = "Vendor GmbH\nHauptstrasse 42\n12345 Berlin\nGermany"
    raw = _make_json(fields=[{"fieldtype": "vendor_address", "text": address, "score": 0.9}])
    result = _parse_text_response(raw)
    assert len(result) == 1
    assert result[0].text == address


def test_parse_address_with_actual_newlines():
    """JSON with actual newline character in text string."""
    payload = {"fields": [{"fieldtype": "customer_billing_address", "text": "Street 1\nCity 12345", "score": 0.85}], "line_items": []}
    result = _parse_text_response(json.dumps(payload))
    assert len(result) == 1
    assert "\n" in result[0].text


# ─────────────────────────────────────────────────────────────────────────────
# LIR line items
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_single_line_item():
    raw = _make_json(line_items=[{
        "line_item_id": 1,
        "fields": [
            {"fieldtype": "line_item_description", "text": "Widget A", "score": 0.9},
            {"fieldtype": "line_item_quantity", "text": "5", "score": 0.95},
            {"fieldtype": "line_item_amount_gross", "text": "50.00", "score": 0.9},
        ],
    }])
    result = _parse_text_response(raw)
    lir_fields = [f for f in result if f.line_item_id is not None]
    assert len(lir_fields) == 3
    assert all(f.line_item_id == 1 for f in lir_fields)
    ft_set = {f.fieldtype for f in lir_fields}
    assert ft_set == {"line_item_description", "line_item_quantity", "line_item_amount_gross"}


def test_parse_multiple_line_items():
    raw = _make_json(line_items=[
        {
            "line_item_id": 1,
            "fields": [{"fieldtype": "line_item_amount_gross", "text": "100.00", "score": 0.9}],
        },
        {
            "line_item_id": 2,
            "fields": [{"fieldtype": "line_item_amount_gross", "text": "200.00", "score": 0.9}],
        },
        {
            "line_item_id": 3,
            "fields": [{"fieldtype": "line_item_amount_gross", "text": "50.00", "score": 0.9}],
        },
    ])
    result = _parse_text_response(raw)
    lir = {f.line_item_id: f.text for f in result if f.line_item_id is not None}
    assert lir == {1: "100.00", 2: "200.00", 3: "50.00"}


def test_parse_unknown_lir_fieldtype_ignored():
    raw = _make_json(line_items=[{
        "line_item_id": 1,
        "fields": [
            {"fieldtype": "line_item_amount_gross", "text": "100.00", "score": 0.9},
            {"fieldtype": "nonexistent_lir_field", "text": "???", "score": 0.5},
        ],
    }])
    result = _parse_text_response(raw)
    lir = [f for f in result if f.line_item_id is not None]
    assert len(lir) == 1
    assert lir[0].fieldtype == "line_item_amount_gross"


# ─────────────────────────────────────────────────────────────────────────────
# Mixed KILE + LIR
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_mixed_kile_and_lir():
    raw = _make_json(
        fields=[
            {"fieldtype": "document_id", "text": "INV-001", "score": 0.95},
            {"fieldtype": "vendor_name", "text": "Acme", "score": 0.9},
        ],
        line_items=[{
            "line_item_id": 1,
            "fields": [
                {"fieldtype": "line_item_quantity", "text": "3", "score": 0.9},
                {"fieldtype": "line_item_unit_price_net", "text": "10.00", "score": 0.85},
            ],
        }],
    )
    result = _parse_text_response(raw)
    kile = [f for f in result if f.line_item_id is None]
    lir = [f for f in result if f.line_item_id is not None]
    assert len(kile) == 2
    assert len(lir) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Robustness
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_missing_fields_key():
    raw = json.dumps({"line_items": []})
    result = _parse_text_response(raw)
    assert result == []


def test_parse_missing_line_items_key():
    raw = json.dumps({"fields": [{"fieldtype": "document_id", "text": "INV-001", "score": 0.9}]})
    result = _parse_text_response(raw)
    assert len(result) == 1


def test_parse_score_clamped_to_float():
    raw = _make_json(fields=[{"fieldtype": "document_id", "text": "X", "score": "0.9"}])
    result = _parse_text_response(raw)
    assert result[0].score == 0.9


def test_parse_non_string_text_coerced():
    raw = _make_json(fields=[{"fieldtype": "document_id", "text": 12345, "score": 0.9}])
    result = _parse_text_response(raw)
    assert result[0].text == "12345"


def test_parse_iban_with_spaces():
    raw = _make_json(fields=[{"fieldtype": "iban", "text": "DE89 3704 0044 0532 0130 00", "score": 0.95}])
    result = _parse_text_response(raw)
    assert len(result) == 1
    assert result[0].text == "DE89 3704 0044 0532 0130 00"
