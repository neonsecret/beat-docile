"""Integration tests for react_extract — mock vertex client, no live API calls.

Verifies the tool-use dispatch loop, tool result injection, max_steps cap,
error recovery, and Field object construction.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from beat_docile.data import WordBox
from beat_docile.react_extract import (
    PER_FIELD_SYSTEM,
    TRIAGE_SYSTEM,
    VERIFIER_SYSTEM,
    _bbox_from_word_ids,
    _execute_tool,
    _parse_candidates_from_response,
    _serialize_content_block,
    extract_field_react,
    extract_page_react,
    triage_fields,
    verify_extractions,
)
from beat_docile.tools import Candidate

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _word(wid: int, text: str, left: float, top: float, right: float, bottom: float) -> WordBox:
    return WordBox(id=wid, text=text, bbox=(left, top, right, bottom), page=0)


def _mock_image() -> MagicMock:
    img = MagicMock()
    import io

    from PIL import Image
    real_img = Image.new("RGB", (100, 100), color="white")
    buf = io.BytesIO()
    real_img.save(buf, format="PNG")
    img.save = lambda b, format: b.write(buf.getvalue())
    return img


def _text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _tool_use_block(tool_id: str, name: str, input_dict: dict) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = input_dict
    return block


def _response(stop_reason: str, content_blocks: list) -> MagicMock:
    resp = MagicMock()
    resp.stop_reason = stop_reason
    resp.content = content_blocks
    return resp


def _invoice_words() -> list[WordBox]:
    return [
        _word(0, "Invoice",    0.05, 0.10, 0.15, 0.12),
        _word(1, "No:",        0.16, 0.10, 0.22, 0.12),
        _word(2, "INV-001",   0.23, 0.10, 0.35, 0.12),
        _word(3, "Date:",      0.05, 0.20, 0.15, 0.22),
        _word(4, "2024-01-15", 0.16, 0.20, 0.35, 0.22),
    ]


def _mock_client() -> MagicMock:
    return MagicMock()


# ── System prompt smoke tests ────────────────────────────────────────────────


def test_triage_system_prompt_non_empty():
    assert len(TRIAGE_SYSTEM) > 100
    assert "present_fields" in TRIAGE_SYSTEM


def test_per_field_system_template_has_placeholders():
    assert "{fieldtype}" in PER_FIELD_SYSTEM
    assert "{field_description}" in PER_FIELD_SYSTEM


def test_verifier_system_non_empty():
    assert len(VERIFIER_SYSTEM) > 50
    assert "corrections" in VERIFIER_SYSTEM


# ── Helper: _serialize_content_block ────────────────────────────────────────


class TestSerializeContentBlock:
    def test_text_block(self):
        block = _text_block("hello")
        result = _serialize_content_block(block)
        assert result == {"type": "text", "text": "hello"}

    def test_tool_use_block(self):
        block = _tool_use_block("tu_1", "regex_extract", {"fieldtype": "iban"})
        result = _serialize_content_block(block)
        assert result["type"] == "tool_use"
        assert result["id"] == "tu_1"
        assert result["name"] == "regex_extract"
        assert result["input"] == {"fieldtype": "iban"}


# ── Helper: _parse_candidates_from_response ──────────────────────────────────


class TestParseCandidates:
    def _wrap(self, json_str: str) -> MagicMock:
        return _response("end_turn", [_text_block(json_str)])

    def test_valid_candidates(self):
        resp = self._wrap('{"candidates": [{"word_ids": [2], "text": "INV-001", "score": 0.9, "reason": "ok"}]}')
        result = _parse_candidates_from_response(resp)
        assert len(result) == 1
        assert result[0].word_ids == [2]
        assert result[0].text == "INV-001"
        assert result[0].score == 0.9
        assert result[0].source == "react"

    def test_empty_candidates(self):
        resp = self._wrap('{"candidates": []}')
        assert _parse_candidates_from_response(resp) == []

    def test_malformed_json_returns_empty(self):
        resp = self._wrap("not valid json at all")
        assert _parse_candidates_from_response(resp) == []

    def test_json_without_candidates_key_returns_empty(self):
        resp = self._wrap('{"fields": []}')
        assert _parse_candidates_from_response(resp) == []

    def test_markdown_fenced_json_is_stripped(self):
        resp = self._wrap('```json\n{"candidates": [{"word_ids": [0], "text": "x", "score": 0.5, "reason": ""}]}\n```')
        result = _parse_candidates_from_response(resp)
        assert len(result) == 1

    def test_multiple_candidates(self):
        resp = self._wrap(
            '{"candidates": ['
            '{"word_ids": [1], "text": "a", "score": 0.8, "reason": ""},'
            '{"word_ids": [2], "text": "b", "score": 0.7, "reason": ""}'
            ']}'
        )
        result = _parse_candidates_from_response(resp)
        assert len(result) == 2


# ── Helper: _execute_tool ────────────────────────────────────────────────────


class TestExecuteTool:
    def _words(self) -> list[WordBox]:
        return [
            _word(0, "DE89370400440532013000", 0.1, 0.5, 0.6, 0.52),
            _word(1, "2024-01-15", 0.1, 0.6, 0.4, 0.62),
        ]

    def test_regex_extract_dispatch(self):
        result = _execute_tool("regex_extract", {"fieldtype": "iban"}, self._words(), None, {})
        assert isinstance(result, list)

    def test_validator_check_dispatch_valid(self):
        result = _execute_tool("validator_check", {"fieldtype": "iban", "text": "DE89370400440532013000"}, [], None, {})
        assert result is not None
        assert "score" in result

    def test_validator_check_dispatch_invalid(self):
        result = _execute_tool("validator_check", {"fieldtype": "iban", "text": "bad"}, [], None, {})
        assert result is None

    def test_spatial_neighbor_dispatch(self):
        words = _invoice_words()
        result = _execute_tool(
            "spatial_neighbor",
            {"label_phrases": ["Invoice No", "No:"], "direction": "right"},
            words,
            None,
            {},
        )
        assert isinstance(result, list)

    def test_cluster_fewshot_dispatch(self):
        train_docs = {
            "doc1": {"cluster_id": 5, "fields": [{"fieldtype": "document_id", "text": "INV-001"}]},
        }
        result = _execute_tool(
            "cluster_fewshot",
            {"fieldtype": "document_id", "cluster_id": 5},
            [],
            5,
            train_docs,
        )
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_refine_span_dispatch(self):
        words = _invoice_words()
        result = _execute_tool(
            "refine_span",
            {"fieldtype": "document_id", "word_ids": [0, 1, 2], "text": "INV-001"},
            words,
            None,
            {},
        )
        assert isinstance(result, dict)
        assert "word_ids" in result
        assert "source" in result

    def test_unknown_tool_raises(self):
        with pytest.raises(ValueError, match="Unknown tool"):
            _execute_tool("totally_unknown_tool", {}, [], None, {})


# ── Helper: _bbox_from_word_ids ──────────────────────────────────────────────


class TestBboxFromWordIds:
    def test_single_word(self):
        words = [_word(0, "hello", 0.1, 0.2, 0.3, 0.4)]
        bbox = _bbox_from_word_ids([0], words)
        assert bbox is not None
        assert abs(bbox.left - 0.1) < 1e-9
        assert abs(bbox.top - 0.2) < 1e-9
        assert abs(bbox.right - 0.3) < 1e-9
        assert abs(bbox.bottom - 0.4) < 1e-9

    def test_multi_word_bbox_is_union(self):
        words = [
            _word(0, "A", 0.1, 0.2, 0.3, 0.4),
            _word(1, "B", 0.4, 0.2, 0.6, 0.4),
        ]
        bbox = _bbox_from_word_ids([0, 1], words)
        assert bbox is not None
        assert abs(bbox.left - 0.1) < 1e-9
        assert abs(bbox.right - 0.6) < 1e-9

    def test_unknown_word_ids_returns_none(self):
        words = [_word(0, "A", 0.1, 0.2, 0.3, 0.4)]
        bbox = _bbox_from_word_ids([999, 1000], words)
        assert bbox is None

    def test_empty_word_ids_returns_none(self):
        words = [_word(0, "A", 0.1, 0.2, 0.3, 0.4)]
        bbox = _bbox_from_word_ids([], words)
        assert bbox is None


# ── triage_fields ─────────────────────────────────────────────────────────────


class TestTriageFields:
    def test_returns_list_of_fieldtypes(self):
        client = _mock_client()
        client.messages.create.return_value = _response(
            "end_turn",
            [_text_block('{"present_fields": ["vendor_name", "date_issue", "amount_due"]}')],
        )
        words = _invoice_words()
        result = triage_fields(words, _mock_image(), client)
        assert "vendor_name" in result
        assert "date_issue" in result

    def test_malformed_json_falls_back_to_all_fields(self):
        client = _mock_client()
        client.messages.create.return_value = _response(
            "end_turn", [_text_block("not valid json")]
        )
        result = triage_fields(_invoice_words(), _mock_image(), client)
        assert len(result) > 5  # falls back to all known field types

    def test_api_exception_falls_back_to_all_fields(self):
        client = _mock_client()
        client.messages.create.side_effect = RuntimeError("API down")
        result = triage_fields(_invoice_words(), _mock_image(), client)
        assert len(result) > 5

    def test_returns_only_string_elements(self):
        client = _mock_client()
        client.messages.create.return_value = _response(
            "end_turn",
            [_text_block('{"present_fields": ["vendor_name", null, "date_issue"]}')],
        )
        result = triage_fields(_invoice_words(), _mock_image(), client)
        assert all(isinstance(f, str) for f in result)

    def test_calls_api_once(self):
        client = _mock_client()
        client.messages.create.return_value = _response(
            "end_turn", [_text_block('{"present_fields": ["vendor_name"]}')]
        )
        triage_fields(_invoice_words(), _mock_image(), client)
        assert client.messages.create.call_count == 1


# ── extract_field_react ───────────────────────────────────────────────────────


class TestExtractFieldReact:
    def test_end_turn_immediately_returns_candidates(self):
        client = _mock_client()
        client.messages.create.return_value = _response(
            "end_turn",
            [_text_block('{"candidates": [{"word_ids": [2], "text": "INV-001", "score": 0.9, "reason": "found"}]}')],
        )
        result = extract_field_react("document_id", _invoice_words(), _mock_image(), None, {}, client)
        assert len(result) == 1
        assert result[0].word_ids == [2]
        assert result[0].text == "INV-001"
        assert result[0].score == 0.9

    def test_tool_use_then_end_turn_uses_two_api_calls(self):
        client = _mock_client()
        tool_resp = _response("tool_use", [_tool_use_block("tu_1", "regex_extract", {"fieldtype": "document_id"})])
        final_resp = _response(
            "end_turn",
            [_text_block('{"candidates": [{"word_ids": [2], "text": "INV-001", "score": 0.8, "reason": "regex"}]}')],
        )
        client.messages.create.side_effect = [tool_resp, final_resp]
        result = extract_field_react("document_id", _invoice_words(), _mock_image(), None, {}, client)
        assert client.messages.create.call_count == 2
        assert len(result) == 1

    def test_tool_result_injected_into_second_call(self):
        client = _mock_client()
        tool_resp = _response("tool_use", [_tool_use_block("tu_1", "regex_extract", {"fieldtype": "date_issue"})])
        final_resp = _response("end_turn", [_text_block('{"candidates": []}')])
        client.messages.create.side_effect = [tool_resp, final_resp]

        words = [_word(0, "2024-01-15", 0.1, 0.5, 0.3, 0.52)]
        extract_field_react("date_issue", words, _mock_image(), None, {}, client)

        second_call_kwargs = client.messages.create.call_args_list[1].kwargs
        messages = second_call_kwargs["messages"]
        last_msg = messages[-1]
        assert last_msg["role"] == "user"
        content = last_msg["content"]
        assert len(content) >= 1
        assert content[0]["type"] == "tool_result"
        assert content[0]["tool_use_id"] == "tu_1"

    def test_max_steps_cap_stops_loop(self):
        client = _mock_client()
        # Always return tool_use — loop should terminate at max_steps
        tool_block = _tool_use_block("tu_1", "regex_extract", {"fieldtype": "document_id"})
        client.messages.create.return_value = _response("tool_use", [tool_block])
        result = extract_field_react("document_id", _invoice_words(), _mock_image(), None, {}, client, max_steps=3)
        assert client.messages.create.call_count <= 3
        assert isinstance(result, list)

    def test_unknown_tool_name_returns_is_error_result(self):
        client = _mock_client()
        bad_tool = _response("tool_use", [_tool_use_block("tu_bad", "totally_unknown_tool", {})])
        final_resp = _response("end_turn", [_text_block('{"candidates": []}')])
        client.messages.create.side_effect = [bad_tool, final_resp]

        result = extract_field_react("document_id", _invoice_words(), _mock_image(), None, {}, client)
        assert isinstance(result, list)
        # Second call must have been made with is_error tool_result
        second_call_kwargs = client.messages.create.call_args_list[1].kwargs
        messages = second_call_kwargs["messages"]
        last_msg = messages[-1]
        content = last_msg["content"]
        error_blocks = [c for c in content if c.get("is_error")]
        assert len(error_blocks) >= 1

    def test_malformed_final_json_returns_empty_list(self):
        client = _mock_client()
        client.messages.create.return_value = _response(
            "end_turn", [_text_block("I could not find the field.")]
        )
        result = extract_field_react("iban", _invoice_words(), _mock_image(), None, {}, client)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_api_exception_returns_empty_list(self):
        client = _mock_client()
        client.messages.create.side_effect = RuntimeError("API error")
        result = extract_field_react("document_id", _invoice_words(), _mock_image(), None, {}, client)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_multiple_tool_calls_in_one_response(self):
        client = _mock_client()
        # Response with two tool_use blocks in one turn
        tool_resp = _response("tool_use", [
            _tool_use_block("tu_1", "regex_extract", {"fieldtype": "document_id"}),
            _tool_use_block("tu_2", "spatial_neighbor", {"label_phrases": ["Invoice No"]}),
        ])
        final_resp = _response(
            "end_turn",
            [_text_block('{"candidates": [{"word_ids": [2], "text": "INV-001", "score": 0.85, "reason": ""}]}')],
        )
        client.messages.create.side_effect = [tool_resp, final_resp]

        extract_field_react("document_id", _invoice_words(), _mock_image(), None, {}, client)
        # Both tool results should be sent in the next user message
        second_call_kwargs = client.messages.create.call_args_list[1].kwargs
        last_msg = second_call_kwargs["messages"][-1]
        assert len(last_msg["content"]) == 2

    def test_tools_passed_to_api(self):
        client = _mock_client()
        client.messages.create.return_value = _response(
            "end_turn", [_text_block('{"candidates": []}')]
        )
        extract_field_react("document_id", _invoice_words(), _mock_image(), None, {}, client)
        call_kwargs = client.messages.create.call_args.kwargs
        tools = call_kwargs["tools"]
        tool_names = {t["name"] for t in tools}
        assert "regex_extract" in tool_names
        assert "validator_check" in tool_names
        assert "spatial_neighbor" in tool_names
        assert "cluster_fewshot" in tool_names
        assert "refine_span" in tool_names


# ── verify_extractions ────────────────────────────────────────────────────────


class TestVerifyExtractions:
    def test_no_corrections_returns_input_unchanged(self):
        client = _mock_client()
        client.messages.create.return_value = _response(
            "end_turn", [_text_block('{"corrections": []}')]
        )
        extractions = {
            "vendor_name": [Candidate(word_ids=[0], text="ACME", score=0.9, source="react")],
        }
        result = verify_extractions(extractions, _invoice_words(), _mock_image(), client)
        assert "vendor_name" in result
        assert len(result["vendor_name"]) == 1

    def test_correction_action_remove_clears_field(self):
        client = _mock_client()
        client.messages.create.return_value = _response(
            "end_turn",
            [_text_block('{"corrections": [{"fieldtype": "amount_total_net", "action": "remove", "reason": "same as gross"}]}')],
        )
        extractions = {
            "amount_total_gross": [Candidate(word_ids=[5], text="100.00", score=0.9, source="react")],
            "amount_total_net": [Candidate(word_ids=[5], text="100.00", score=0.8, source="react")],
        }
        result = verify_extractions(extractions, _invoice_words(), _mock_image(), client)
        assert result["amount_total_net"] == []
        assert len(result["amount_total_gross"]) == 1

    def test_api_exception_returns_input_unchanged(self):
        client = _mock_client()
        client.messages.create.side_effect = RuntimeError("API down")
        extractions = {
            "vendor_name": [Candidate(word_ids=[0], text="ACME", score=0.9, source="react")],
        }
        result = verify_extractions(extractions, _invoice_words(), _mock_image(), client)
        assert result == extractions

    def test_empty_extractions_returns_empty(self):
        client = _mock_client()
        result = verify_extractions({}, _invoice_words(), _mock_image(), client)
        assert result == {}
        # API should NOT be called for empty extractions
        client.messages.create.assert_not_called()

    def test_malformed_verifier_response_returns_input_unchanged(self):
        client = _mock_client()
        client.messages.create.return_value = _response(
            "end_turn", [_text_block("I found no issues.")]
        )
        extractions = {
            "vendor_name": [Candidate(word_ids=[0], text="ACME", score=0.9, source="react")],
        }
        result = verify_extractions(extractions, _invoice_words(), _mock_image(), client)
        assert result == extractions


# ── extract_page_react ────────────────────────────────────────────────────


class TestExtractDocumentReact:
    def _setup_client(self) -> MagicMock:
        """Client that returns vendor_name and date_issue after triage."""
        client = _mock_client()

        triage_resp = _response(
            "end_turn",
            [_text_block('{"present_fields": ["vendor_name", "date_issue"]}')],
        )
        vendor_resp = _response(
            "end_turn",
            [_text_block('{"candidates": [{"word_ids": [0], "text": "ACME Corp", "score": 0.95, "reason": ""}]}')],
        )
        date_resp = _response(
            "end_turn",
            [_text_block('{"candidates": [{"word_ids": [4], "text": "2024-01-15", "score": 0.9, "reason": ""}]}')],
        )
        verifier_resp = _response("end_turn", [_text_block('{"corrections": []}')])

        client.messages.create.side_effect = [
            triage_resp,
            vendor_resp,
            date_resp,
            verifier_resp,
        ]
        return client

    def test_returns_tuple_of_two_lists(self):
        client = self._setup_client()
        words = _invoice_words()
        kile, lir = extract_page_react(words, _mock_image(), None, {}, client)
        assert isinstance(kile, list)
        assert isinstance(lir, list)

    def test_kile_fields_have_valid_bbox(self):
        client = self._setup_client()
        words = _invoice_words()
        kile, _ = extract_page_react(words, _mock_image(), None, {}, client)
        for field in kile:
            assert field.bbox is not None
            assert 0.0 <= field.bbox.left <= 1.0
            assert 0.0 <= field.bbox.top <= 1.0

    def test_kile_fields_have_scores(self):
        client = self._setup_client()
        words = _invoice_words()
        kile, _ = extract_page_react(words, _mock_image(), None, {}, client)
        for field in kile:
            assert field.score is not None
            assert 0.0 <= field.score <= 1.0

    def test_lir_fields_have_line_item_ids(self):
        client = _mock_client()
        triage_resp = _response(
            "end_turn",
            [_text_block('{"present_fields": ["line_item_description"]}')],
        )
        lir_resp = _response(
            "end_turn",
            [_text_block(
                '{"candidates": ['
                '{"word_ids": [0], "text": "Widget A", "score": 0.9, "reason": ""},'
                '{"word_ids": [1], "text": "Widget B", "score": 0.85, "reason": ""}'
                ']}'
            )],
        )
        verifier_resp = _response("end_turn", [_text_block('{"corrections": []}')])
        client.messages.create.side_effect = [triage_resp, lir_resp, verifier_resp]

        words = [
            _word(0, "Widget A", 0.05, 0.30, 0.25, 0.32),
            _word(1, "Widget B", 0.05, 0.40, 0.25, 0.42),
        ]
        _, lir = extract_page_react(words, _mock_image(), None, {}, client)
        for field in lir:
            assert field.line_item_id is not None
            assert field.line_item_id >= 1

    def test_model_routing_triage_haiku_react_sonnet_verifier_haiku(self):
        """triage + verifier use Haiku; per-field ReAct uses Sonnet."""
        client = self._setup_client()
        extract_page_react(_invoice_words(), _mock_image(), None, {}, client)
        calls = client.messages.create.call_args_list
        # call 0 = triage, call 1 = vendor_name react, call 2 = date_issue react, call 3 = verifier
        assert calls[0].kwargs["model"] == "claude-haiku-4-5"   # triage
        assert calls[1].kwargs["model"] == "claude-sonnet-4-6"  # react
        assert calls[-1].kwargs["model"] == "claude-haiku-4-5"  # verifier

    def test_empty_words_returns_empty_fields(self):
        client = _mock_client()
        client.messages.create.return_value = _response(
            "end_turn", [_text_block('{"present_fields": []}')]
        )
        kile, lir = extract_page_react([], _mock_image(), None, {}, client)
        assert kile == []
        assert lir == []

    def test_triage_unknown_fieldtype_is_skipped(self):
        """Fieldtypes not in KILE or LIR catalogs are silently skipped."""
        client = _mock_client()
        triage_resp = _response(
            "end_turn",
            [_text_block('{"present_fields": ["totally_fake_field", "vendor_name"]}')],
        )
        vendor_resp = _response(
            "end_turn",
            [_text_block('{"candidates": [{"word_ids": [0], "text": "ACME", "score": 0.9, "reason": ""}]}')],
        )
        verifier_resp = _response("end_turn", [_text_block('{"corrections": []}')])
        client.messages.create.side_effect = [triage_resp, vendor_resp, verifier_resp]

        words = _invoice_words()
        kile, _lir = extract_page_react(words, _mock_image(), None, {}, client)
        # Only vendor_name should be extracted — fake field skipped
        fieldtypes = {f.fieldtype for f in kile}
        assert "totally_fake_field" not in fieldtypes
