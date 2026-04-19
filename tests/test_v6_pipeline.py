"""Unit tests for v6_pipeline — mock vertex client, no live API calls.

Tests verify:
  (a) all pages iterated in extract_document_v6
  (b) haiku verifier called when use_haiku_verify=True
  (c) classifier tool registered in _TOOL_SCHEMAS (6 tools total)
  (d) output schema matches v5b_50.json shape
  (e) run_v6_on_docids iterates docs and writes JSON
  (f) evaluate_v6 returns correct metric keys
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from beat_docile.data import WordBox
from beat_docile.react_extract import _TOOL_SCHEMAS
from beat_docile.tools import Candidate
from beat_docile.v6_pipeline import (
    _build_haiku_verifier,
    _field_to_dict,
    _load_train_docs,
    evaluate_v6,
    extract_document_v6,
    run_v6_on_docids,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _word(wid: int, text: str, left: float, top: float, right: float, bottom: float) -> WordBox:
    return WordBox(id=wid, text=text, bbox=(left, top, right, bottom), page=0)


def _invoice_words() -> list[WordBox]:
    return [
        _word(0, "Invoice", 0.05, 0.10, 0.15, 0.12),
        _word(1, "No:",     0.16, 0.10, 0.22, 0.12),
        _word(2, "INV-001", 0.23, 0.10, 0.35, 0.12),
    ]


def _mock_page(words: list[WordBox] | None = None) -> MagicMock:
    page = MagicMock()
    page.words = words or _invoice_words()
    page.image = MagicMock()
    page.image.save = MagicMock()
    return page


def _mock_doc(docid: str = "test_doc", cluster_id: int = 1) -> MagicMock:
    doc = MagicMock()
    doc.docid = docid
    doc.annotation.cluster_id = cluster_id
    return doc


def _make_field(fieldtype: str, li_id: int | None = None):
    """Return a real docile Field for output schema tests."""
    from docile.dataset import BBox
    from docile.dataset import Field as DocileField
    bbox = BBox(0.1, 0.2, 0.3, 0.4)
    return DocileField(
        bbox=bbox,
        page=0,
        fieldtype=fieldtype,
        score=0.9,
        line_item_id=li_id,
        text="test_text",
    )


# ── (c) Classifier tool registered ───────────────────────────────────────────


def test_classifier_tool_registered_in_schemas():
    """_TOOL_SCHEMAS must contain exactly 6 tools with classifier_score as the 6th."""
    tool_names = [t["name"] for t in _TOOL_SCHEMAS]
    assert "classifier_score" in tool_names, "classifier_score missing from _TOOL_SCHEMAS"
    assert len(_TOOL_SCHEMAS) == 6, f"Expected 6 tools, got {len(_TOOL_SCHEMAS)}"


def test_classifier_tool_schema_shape():
    """classifier_score schema must have the required input field."""
    schema = next(t for t in _TOOL_SCHEMAS if t["name"] == "classifier_score")
    assert "word_ids" in schema["input_schema"]["properties"]
    assert schema["input_schema"]["required"] == ["word_ids"]


# ── (b) Haiku verifier integration ───────────────────────────────────────────


def test_build_haiku_verifier_accept():
    """Haiku verifier preserves candidate on 'accept' verdict."""
    words = _invoice_words()
    vertex_client = MagicMock()

    vr = MagicMock()
    vr.verdict = "accept"
    vr.word_ids = [2]
    vr.confidence = 0.95
    vr.reasoning = "ok"

    with patch("beat_docile.v6_pipeline.verify_extractions_batch") as mock_batch, \
         patch("beat_docile.v6_pipeline.apply_verification") as mock_apply:
        mock_batch.return_value = {"vendor_name": [vr]}
        mock_apply.return_value = {"vendor_name": [([2], "INV-001")]}

        verifier = _build_haiku_verifier(words, vertex_client)
        candidates = [Candidate(word_ids=[2], text="INV-001", score=0.9, source="react", reason="")]
        result = verifier("vendor_name", candidates)

    mock_batch.assert_called_once()
    mock_apply.assert_called_once()
    assert len(result) == 1
    assert result[0].word_ids == [2]
    assert result[0].score == 0.9


def test_build_haiku_verifier_correct_high_confidence():
    """Haiku verifier updates word_ids on 'correct' with confidence >= 0.6."""
    words = _invoice_words()
    vertex_client = MagicMock()

    vr = MagicMock()
    vr.verdict = "correct"
    vr.word_ids = [2]  # corrected: strip label words
    vr.confidence = 0.85

    with patch("beat_docile.v6_pipeline.verify_extractions_batch") as mock_batch, \
         patch("beat_docile.v6_pipeline.apply_verification") as mock_apply:
        mock_batch.return_value = {"document_id": [vr]}
        mock_apply.return_value = {"document_id": [([2], "INV-001")]}

        verifier = _build_haiku_verifier(words, vertex_client)
        candidates = [Candidate(word_ids=[0, 1, 2], text="Invoice No: INV-001",
                                score=0.7, source="react", reason="")]
        result = verifier("document_id", candidates)

    assert len(result) == 1
    assert result[0].word_ids == [2]  # corrected by Haiku
    assert result[0].score == 0.7    # score preserved from original


def test_build_haiku_verifier_reject_high_confidence():
    """Haiku verifier drops candidate on 'reject' with confidence >= 0.6."""
    words = _invoice_words()
    vertex_client = MagicMock()

    vr = MagicMock()
    vr.verdict = "reject"
    vr.word_ids = []
    vr.confidence = 0.90

    with patch("beat_docile.v6_pipeline.verify_extractions_batch") as mock_batch, \
         patch("beat_docile.v6_pipeline.apply_verification") as mock_apply:
        mock_batch.return_value = {"customer_id": [vr]}
        mock_apply.return_value = {"customer_id": []}

        verifier = _build_haiku_verifier(words, vertex_client)
        candidates = [Candidate(word_ids=[0, 1], text="Invoice No", score=0.5, source="react", reason="")]
        result = verifier("customer_id", candidates)

    assert result == []  # rejected and dropped


def test_build_haiku_verifier_empty_candidates():
    """Haiku verifier returns empty list unchanged without API call."""
    words = _invoice_words()
    vertex_client = MagicMock()

    with patch("beat_docile.v6_pipeline.verify_extractions_batch") as mock_batch, \
         patch("beat_docile.v6_pipeline.apply_verification"):
        verifier = _build_haiku_verifier(words, vertex_client)
        result = verifier("vendor_name", [])

    mock_batch.assert_not_called()
    assert result == []


# ── (a) All pages iterated ────────────────────────────────────────────────────


@patch("beat_docile.v6_pipeline.iter_pages")
@patch("beat_docile.v6_pipeline.extract_page_react")
def test_all_pages_iterated(mock_extract_page, mock_iter_pages):
    """extract_document_v6 must call extract_page_react once per page."""
    pages = [_mock_page(), _mock_page()]
    mock_iter_pages.return_value = iter(pages)
    mock_extract_page.return_value = ([], [])

    doc = _mock_doc()
    result = extract_document_v6(
        docid="test_doc",
        doc=doc,
        train_docs={},
        vertex_client=MagicMock(),
        use_haiku_verify=False,
        use_classifier_tool=False,
    )

    assert mock_extract_page.call_count == 2
    assert result["docid"] == "test_doc"
    assert isinstance(result["fields"], list)


@patch("beat_docile.v6_pipeline.iter_pages")
@patch("beat_docile.v6_pipeline.extract_page_react")
def test_single_page_doc(mock_extract_page, mock_iter_pages):
    """Single-page document yields exactly one extract_page_react call."""
    mock_iter_pages.return_value = iter([_mock_page()])
    mock_extract_page.return_value = ([], [])

    result = extract_document_v6(
        docid="single",
        doc=_mock_doc(),
        train_docs={},
        vertex_client=MagicMock(),
        use_haiku_verify=False,
        use_classifier_tool=False,
    )

    assert mock_extract_page.call_count == 1
    assert result["docid"] == "single"


# ── (d) Output schema ─────────────────────────────────────────────────────────


@patch("beat_docile.v6_pipeline.iter_pages")
@patch("beat_docile.v6_pipeline.extract_page_react")
def test_output_schema_kile_field(mock_extract_page, mock_iter_pages):
    """KILE field output has all required keys and correct types."""
    kile_field = _make_field("vendor_name")
    mock_iter_pages.return_value = iter([_mock_page()])
    mock_extract_page.return_value = ([kile_field], [])

    result = extract_document_v6(
        docid="abc",
        doc=_mock_doc(),
        train_docs={},
        vertex_client=MagicMock(),
        use_haiku_verify=False,
        use_classifier_tool=False,
    )

    assert len(result["fields"]) == 1
    fd = result["fields"][0]
    required = {"bbox", "page", "score", "text", "fieldtype", "line_item_id", "use_only_for_ap"}
    assert required.issubset(set(fd.keys()))
    assert isinstance(fd["bbox"], list)
    assert len(fd["bbox"]) == 4
    assert all(isinstance(v, float) for v in fd["bbox"])
    assert fd["fieldtype"] == "vendor_name"
    assert fd["line_item_id"] is None
    assert fd["use_only_for_ap"] is False


@patch("beat_docile.v6_pipeline.iter_pages")
@patch("beat_docile.v6_pipeline.extract_page_react")
def test_output_schema_lir_field(mock_extract_page, mock_iter_pages):
    """LIR field has non-None line_item_id in output."""
    lir_field = _make_field("line_item_quantity", li_id=1)
    mock_iter_pages.return_value = iter([_mock_page()])
    mock_extract_page.return_value = ([], [lir_field])

    result = extract_document_v6(
        docid="abc",
        doc=_mock_doc(),
        train_docs={},
        vertex_client=MagicMock(),
        use_haiku_verify=False,
        use_classifier_tool=False,
    )

    assert len(result["fields"]) == 1
    fd = result["fields"][0]
    assert fd["line_item_id"] == 1
    assert fd["fieldtype"] == "line_item_quantity"


@patch("beat_docile.v6_pipeline.iter_pages")
@patch("beat_docile.v6_pipeline.extract_page_react")
def test_output_schema_matches_v5b_keys(mock_extract_page, mock_iter_pages):
    """Output field dict keys exactly match those in predictions/v5b_50.json."""
    v5b_keys = {"bbox", "page", "score", "text", "fieldtype", "line_item_id", "use_only_for_ap"}
    kile_field = _make_field("date_issue")
    mock_iter_pages.return_value = iter([_mock_page()])
    mock_extract_page.return_value = ([kile_field], [])

    result = extract_document_v6(
        docid="xyz",
        doc=_mock_doc(),
        train_docs={},
        vertex_client=MagicMock(),
        use_haiku_verify=False,
        use_classifier_tool=False,
    )

    fd_keys = set(result["fields"][0].keys())
    assert fd_keys == v5b_keys


# ── (e) run_v6_on_docids ─────────────────────────────────────────────────────


@patch("beat_docile.v6_pipeline.get_client")
@patch("beat_docile.v6_pipeline._load_train_docs")
@patch("beat_docile.v6_pipeline.Dataset")
@patch("beat_docile.v6_pipeline.extract_document_v6")
def test_run_v6_writes_json(mock_extract_doc, mock_dataset_cls,
                             mock_load_train, mock_get_client, tmp_path):
    """run_v6_on_docids iterates docs and writes valid JSON."""
    mock_doc = MagicMock()
    mock_doc.docid = "doc1"
    mock_dataset_cls.return_value = [mock_doc]
    mock_load_train.return_value = {}
    mock_get_client.return_value = MagicMock()
    mock_extract_doc.return_value = {
        "docid": "doc1",
        "fields": [{"fieldtype": "vendor_name", "bbox": [0.1, 0.1, 0.3, 0.2],
                    "page": 0, "score": 0.9, "text": None,
                    "line_item_id": None, "use_only_for_ap": False}],
    }

    out_path = tmp_path / "v6_test.json"
    result = run_v6_on_docids(
        docids=["doc1"],
        output_path=out_path,
        use_haiku_verify=False,
        use_classifier_tool=False,
        progress=False,
    )

    assert out_path.exists()
    written = json.loads(out_path.read_text())
    assert "doc1" in written
    assert len(written["doc1"]) == 1
    assert written["doc1"][0]["fieldtype"] == "vendor_name"
    assert mock_extract_doc.call_count == 1
    assert result is written or result["doc1"] == written["doc1"]


@patch("beat_docile.v6_pipeline.get_client")
@patch("beat_docile.v6_pipeline._load_train_docs")
@patch("beat_docile.v6_pipeline.Dataset")
@patch("beat_docile.v6_pipeline.extract_document_v6")
def test_run_v6_handles_extraction_error(mock_extract_doc, mock_dataset_cls,
                                          mock_load_train, mock_get_client, tmp_path):
    """run_v6_on_docids writes empty list for docs that fail extraction."""
    mock_doc = MagicMock()
    mock_doc.docid = "fail_doc"
    mock_dataset_cls.return_value = [mock_doc]
    mock_load_train.return_value = {}
    mock_get_client.return_value = MagicMock()
    mock_extract_doc.side_effect = RuntimeError("API timeout")

    out_path = tmp_path / "v6_err.json"
    run_v6_on_docids(
        docids=["fail_doc"],
        output_path=out_path,
        progress=False,
    )

    assert out_path.exists()
    written = json.loads(out_path.read_text())
    assert written["fail_doc"] == []


@patch("beat_docile.v6_pipeline.get_client")
@patch("beat_docile.v6_pipeline._load_train_docs")
@patch("beat_docile.v6_pipeline.Dataset")
@patch("beat_docile.v6_pipeline.extract_document_v6")
def test_run_v6_iterates_multiple_docs(mock_extract_doc, mock_dataset_cls,
                                        mock_load_train, mock_get_client, tmp_path):
    """run_v6_on_docids calls extract_document_v6 once per document."""
    docs = [MagicMock(docid=f"doc{i}") for i in range(3)]
    mock_dataset_cls.return_value = docs
    mock_load_train.return_value = {}
    mock_get_client.return_value = MagicMock()
    mock_extract_doc.side_effect = [
        {"docid": f"doc{i}", "fields": []} for i in range(3)
    ]

    run_v6_on_docids(
        docids=["doc0", "doc1", "doc2"],
        output_path=tmp_path / "out.json",
        progress=False,
    )

    assert mock_extract_doc.call_count == 3


# ── (f) evaluate_v6 ──────────────────────────────────────────────────────────


@patch("beat_docile.v6_pipeline.Dataset")
@patch("beat_docile.v6_pipeline.evaluate_dataset")
def test_evaluate_v6_returns_metric_keys(mock_eval_ds, mock_dataset_cls, tmp_path):
    """evaluate_v6 returns dict with all required metric keys."""
    predictions = {
        "doc1": [
            {"bbox": [0.1, 0.1, 0.3, 0.2], "page": 0, "score": 0.9, "text": "ACME",
             "fieldtype": "vendor_name", "line_item_id": None, "use_only_for_ap": False},
        ]
    }
    pred_path = tmp_path / "preds.json"
    pred_path.write_text(json.dumps(predictions))

    mock_doc = MagicMock()
    mock_doc.docid = "doc1"
    mock_dataset_cls.return_value = [mock_doc]

    mock_result = MagicMock()
    mock_result.get_metrics.side_effect = lambda task: (
        {"AP": 0.42, "precision": 0.5, "recall": 0.35}
        if task == "kile"
        else {"f1": 0.52, "precision": 0.6, "recall": 0.45}
    )
    mock_eval_ds.return_value = mock_result

    metrics = evaluate_v6(pred_path)

    assert set(metrics.keys()) == {"kile_ap", "kile_p", "kile_r", "lir_f1", "lir_p", "lir_r"}
    assert metrics["kile_ap"] == pytest.approx(0.42)
    assert metrics["lir_f1"] == pytest.approx(0.52)


@patch("beat_docile.v6_pipeline.Dataset")
@patch("beat_docile.v6_pipeline.evaluate_dataset")
def test_evaluate_v6_splits_kile_lir(mock_eval_ds, mock_dataset_cls, tmp_path):
    """evaluate_v6 routes LIR fields (line_item_id != None) to lir_preds."""
    predictions = {
        "doc1": [
            {"bbox": [0.1, 0.1, 0.3, 0.2], "page": 0, "score": 0.9, "text": "ACME",
             "fieldtype": "vendor_name", "line_item_id": None, "use_only_for_ap": False},
            {"bbox": [0.5, 0.5, 0.7, 0.6], "page": 0, "score": 0.8, "text": "10",
             "fieldtype": "line_item_quantity", "line_item_id": 1, "use_only_for_ap": False},
        ]
    }
    pred_path = tmp_path / "preds2.json"
    pred_path.write_text(json.dumps(predictions))

    mock_dataset_cls.return_value = [MagicMock(docid="doc1")]

    kile_captured: dict = {}
    lir_captured: dict = {}

    def _capture(ds, kile, lir):
        kile_captured.update(kile)
        lir_captured.update(lir)
        mock_r = MagicMock()
        mock_r.get_metrics.return_value = {"AP": 0.0, "precision": 0.0,
                                            "recall": 0.0, "f1": 0.0}
        return mock_r

    mock_eval_ds.side_effect = _capture
    evaluate_v6(pred_path)

    assert len(kile_captured.get("doc1", [])) == 1
    assert len(lir_captured.get("doc1", [])) == 1


# ── _field_to_dict ────────────────────────────────────────────────────────────


def test_field_to_dict_kile():
    """_field_to_dict serializes KILE field with None line_item_id."""
    f = _make_field("document_id")
    d = _field_to_dict(f)
    assert d["fieldtype"] == "document_id"
    assert d["line_item_id"] is None
    assert d["use_only_for_ap"] is False
    assert d["bbox"] == [0.1, 0.2, 0.3, 0.4]
    assert d["page"] == 0


def test_field_to_dict_lir():
    """_field_to_dict serializes LIR field with integer line_item_id."""
    f = _make_field("line_item_code", li_id=3)
    d = _field_to_dict(f)
    assert d["line_item_id"] == 3


# ── _load_train_docs ──────────────────────────────────────────────────────────


def test_load_train_docs_missing_train_json(tmp_path):
    """Returns empty dict when train.json is absent."""
    result = _load_train_docs(tmp_path)
    assert result == {}


def test_load_train_docs_parses_annotations(tmp_path):
    """Parses annotation files correctly, including cluster_id from metadata."""
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()
    ann = {
        "metadata": {"cluster_id": 7},
        "field_extractions": [
            {"fieldtype": "vendor_name", "text": "Acme Corp", "bbox": [0, 0, 0, 0], "page": 0}
        ],
        "line_item_extractions": [
            {"fieldtype": "line_item_quantity", "text": "5", "bbox": [0, 0, 0, 0], "page": 0}
        ],
    }
    (ann_dir / "abc123.json").write_text(json.dumps(ann))
    (tmp_path / "train.json").write_text(json.dumps(["abc123"]))

    result = _load_train_docs(tmp_path)

    assert "abc123" in result
    assert result["abc123"]["cluster_id"] == 7
    fields = result["abc123"]["fields"]
    ftypes = [f["fieldtype"] for f in fields]
    assert "vendor_name" in ftypes
    assert "line_item_quantity" in ftypes


def test_load_train_docs_skips_missing_annotation(tmp_path):
    """Docids without annotation files are silently skipped."""
    (tmp_path / "annotations").mkdir()
    (tmp_path / "train.json").write_text(json.dumps(["missing_doc"]))

    result = _load_train_docs(tmp_path)
    assert result == {}
