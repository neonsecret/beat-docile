"""Tests for src/beat_docile/classifiers.py."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from beat_docile.classifiers import (
    _FEATURE_DIM,
    _LABEL_VOCAB,
    DocRecord,
    build_training_set,
    classifier_score,
    extract_features,
    featurize_for_sklearn,
    load_classifier,
    load_doc_records,
    train_all_fields,
    train_classifier,
)
from beat_docile.data import WordBox

# ── Synthetic document fixtures ───────────────────────────────────────────────


def _make_words(n: int = 10) -> list[WordBox]:
    """Create n synthetic WordBox objects arranged in two rows."""
    words = []
    texts = [
        "Invoice", "No.", "12345", "Date", "2024-01-15",
        "Total", "Amount", "100.00", "EUR", "Due",
    ]
    for i in range(n):
        col = i % 5
        row = i // 5
        left = col * 0.18 + 0.02
        top = row * 0.15 + 0.05
        right = left + 0.15
        bottom = top + 0.04
        words.append(WordBox(
            id=i,
            text=texts[i] if i < len(texts) else f"word{i}",
            bbox=(left, top, right, bottom),
            page=0,
        ))
    return words


def _make_doc_record(
    fieldtype: str = "document_id",
    n_words: int = 10,
    n_annotation_fields: int = 1,
) -> DocRecord:
    """Create a minimal synthetic DocRecord with one annotation field."""
    words = _make_words(n_words)
    ann_word = words[2]
    wl, wt, wr, wb = ann_word.bbox
    ann_field = {
        "bbox": [wl, wt, wr, wb],
        "fieldtype": fieldtype,
        "page": 0,
    }
    kile_fields = [ann_field] * n_annotation_fields
    return DocRecord(
        docid="synthetic_doc_001",
        pages=[words],
        kile_fields=kile_fields,
        lir_fields=[],
    )


def _make_many_docs(
    fieldtype: str,
    n_docs: int,
    n_pos_per_doc: int = 1,
) -> list[DocRecord]:
    """Create n_docs synthetic DocRecords each with n_pos_per_doc annotations."""
    docs = []
    rng = random.Random(99)
    for i in range(n_docs):
        n_words = rng.randint(8, 15)
        words = _make_words(n_words)
        ann_word = words[rng.randint(0, n_words - 1)]
        wl, wt, wr, wb = ann_word.bbox
        kile_fields = [
            {"bbox": [wl, wt, wr, wb], "fieldtype": fieldtype, "page": 0}
            for _ in range(n_pos_per_doc)
        ]
        docs.append(DocRecord(
            docid=f"synthetic_{i:04d}",
            pages=[words],
            kile_fields=kile_fields,
            lir_fields=[],
        ))
    return docs


# ── extract_features tests ────────────────────────────────────────────────────


class TestExtractFeatures:
    def test_all_features_populated(self) -> None:
        words = _make_words(10)
        span_ids = [2, 3]  # "12345", "Date"
        feats = extract_features(span_ids, words, 1.0, 1.0)

        assert feats.text == "12345 Date"
        assert feats.char_count == len("12345 Date")
        assert feats.word_count == 2
        assert feats.has_digits is True
        assert 0.0 <= feats.digit_ratio <= 1.0
        assert feats.has_letters is True
        assert isinstance(feats.has_currency_symbol, bool)
        assert isinstance(feats.matches_iban_pattern, bool)
        assert isinstance(feats.matches_date_pattern, bool)
        assert isinstance(feats.matches_amount_pattern, bool)
        assert 0.0 <= feats.bbox_left_frac <= 1.0
        assert 0.0 <= feats.bbox_top_frac <= 1.0
        assert feats.bbox_width_frac > 0.0
        assert feats.bbox_height_frac > 0.0
        assert isinstance(feats.left_neighbor_text, str)
        assert isinstance(feats.right_neighbor_text, str)
        assert isinstance(feats.above_neighbor_text, str)
        assert isinstance(feats.below_neighbor_text, str)
        assert isinstance(feats.nearest_label_phrase, str)
        assert feats.nearest_label_distance_frac >= 0.0

    def test_currency_symbol_detected(self) -> None:
        words = [WordBox(id=0, text="€100", bbox=(0.1, 0.1, 0.2, 0.15), page=0)]
        feats = extract_features([0], words, 1.0, 1.0)
        assert feats.has_currency_symbol is True

    def test_iban_pattern_detected(self) -> None:
        words = [WordBox(id=0, text="GB29NWBK60161331926819", bbox=(0.1, 0.1, 0.4, 0.15), page=0)]
        feats = extract_features([0], words, 1.0, 1.0)
        assert feats.matches_iban_pattern is True

    def test_date_pattern_detected(self) -> None:
        words = [WordBox(id=0, text="2024-01-15", bbox=(0.1, 0.1, 0.3, 0.15), page=0)]
        feats = extract_features([0], words, 1.0, 1.0)
        assert feats.matches_date_pattern is True

    def test_amount_pattern_detected(self) -> None:
        words = [WordBox(id=0, text="1,234.56", bbox=(0.1, 0.1, 0.3, 0.15), page=0)]
        feats = extract_features([0], words, 1.0, 1.0)
        assert feats.matches_amount_pattern is True

    def test_empty_span_returns_defaults(self) -> None:
        words = _make_words(10)
        feats = extract_features([], words, 1.0, 1.0)
        assert feats.text == ""
        assert feats.char_count == 0
        assert feats.word_count == 0

    def test_nearest_label_found(self) -> None:
        words = [
            WordBox(id=0, text="Invoice", bbox=(0.0, 0.0, 0.1, 0.05), page=0),
            WordBox(id=1, text="12345", bbox=(0.15, 0.0, 0.3, 0.05), page=0),
        ]
        feats = extract_features([1], words, 1.0, 1.0)
        assert feats.nearest_label_phrase == "invoice"
        assert feats.nearest_label_distance_frac < 1.0

    def test_neighbour_left_found(self) -> None:
        words = [
            WordBox(id=0, text="Total", bbox=(0.0, 0.1, 0.1, 0.15), page=0),
            WordBox(id=1, text="100.00", bbox=(0.15, 0.1, 0.3, 0.15), page=0),
        ]
        feats = extract_features([1], words, 1.0, 1.0)
        assert feats.left_neighbor_text == "Total"


# ── featurize_for_sklearn tests ───────────────────────────────────────────────


class TestFeaturizeForSklearn:
    def test_output_shape(self) -> None:
        words = _make_words(10)
        feats = extract_features([2, 3], words, 1.0, 1.0)
        vec = featurize_for_sklearn(feats)
        assert vec.shape == (_FEATURE_DIM,)

    def test_deterministic(self) -> None:
        words = _make_words(10)
        feats = extract_features([2, 3], words, 1.0, 1.0)
        vec1 = featurize_for_sklearn(feats)
        vec2 = featurize_for_sklearn(feats)
        np.testing.assert_array_equal(vec1, vec2)

    def test_values_in_expected_range(self) -> None:
        words = _make_words(10)
        feats = extract_features([2, 3], words, 1.0, 1.0)
        vec = featurize_for_sklearn(feats)
        assert np.all(vec[:15] >= 0.0)
        assert np.all(vec[:15] <= 1.0)
        assert set(vec[31:71].tolist()).issubset({0.0, 1.0})

    def test_label_vocab_one_hot_set(self) -> None:
        words = [
            WordBox(id=0, text="Invoice", bbox=(0.0, 0.0, 0.1, 0.05), page=0),
            WordBox(id=1, text="12345", bbox=(0.15, 0.0, 0.3, 0.05), page=0),
        ]
        feats = extract_features([1], words, 1.0, 1.0)
        vec = featurize_for_sklearn(feats)
        idx = _LABEL_VOCAB.index("invoice")
        assert vec[31 + idx] == 1.0

    def test_dtype_float32(self) -> None:
        words = _make_words(5)
        feats = extract_features([0], words, 1.0, 1.0)
        vec = featurize_for_sklearn(feats)
        assert vec.dtype == np.float32


# ── build_training_set tests ──────────────────────────────────────────────────


class TestBuildTrainingSet:
    def test_positive_count_matches_annotations(self) -> None:
        doc = _make_doc_record(fieldtype="document_id", n_annotation_fields=1)
        _x, y = build_training_set("document_id", [doc], n_negatives_per_doc=5)
        assert int(y.sum()) == 1

    def test_negative_count_matches_n_negatives_per_doc(self) -> None:
        doc = _make_doc_record(fieldtype="document_id", n_annotation_fields=1)
        n_neg_requested = 5
        _x, y = build_training_set("document_id", [doc], n_negatives_per_doc=n_neg_requested)
        n_neg = int((y == 0).sum())
        assert n_neg <= n_neg_requested

    def test_output_shapes_consistent(self) -> None:
        doc = _make_doc_record(fieldtype="document_id")
        x_mat, y = build_training_set("document_id", [doc])
        assert x_mat.shape[0] == y.shape[0]
        assert x_mat.shape[1] == _FEATURE_DIM

    def test_no_annotations_yields_only_negatives(self) -> None:
        # Docs with no annotations for the target fieldtype still contribute negatives
        # (spans from those pages are valid "not this fieldtype" examples).
        doc = DocRecord(docid="empty", pages=[_make_words(5)], kile_fields=[], lir_fields=[])
        x_mat, y = build_training_set("document_id", [doc])
        assert x_mat.shape[1] == _FEATURE_DIM
        assert x_mat.shape[0] == y.shape[0]
        assert int(y.sum()) == 0

    def test_multiple_docs_accumulate(self) -> None:
        docs = [_make_doc_record("document_id") for _ in range(3)]
        _x, y = build_training_set("document_id", docs)
        assert y.sum() == 3

    def test_label_values_binary(self) -> None:
        docs = _make_many_docs("document_id", n_docs=5)
        _x, y = build_training_set("document_id", docs)
        assert set(y.tolist()).issubset({0, 1})


# ── train_classifier and classifier_score tests ───────────────────────────────


class TestTrainClassifier:
    def test_model_saves_and_loads(self, tmp_path: Path) -> None:
        docs = _make_many_docs("document_id", n_docs=60, n_pos_per_doc=1)
        train_classifier("document_id", docs, tmp_path)
        assert (tmp_path / "document_id.joblib").exists()
        pipeline = load_classifier("document_id", tmp_path)
        assert pipeline is not None

    def test_metrics_dict_structure(self, tmp_path: Path) -> None:
        docs = _make_many_docs("document_id", n_docs=60)
        metrics = train_classifier("document_id", docs, tmp_path)
        assert metrics["fieldtype"] == "document_id"
        assert isinstance(metrics["n_pos"], int)
        assert isinstance(metrics["n_neg"], int)
        assert metrics["val_f1"] is not None
        assert 0.0 <= metrics["val_f1"] <= 1.0
        assert metrics["val_auc"] is not None
        assert 0.0 <= metrics["val_auc"] <= 1.0

    def test_classifier_score_valid_probability(self, tmp_path: Path) -> None:
        docs = _make_many_docs("document_id", n_docs=60)
        train_classifier("document_id", docs, tmp_path)
        words = _make_words(10)
        score = classifier_score("document_id", [2], words, 1.0, 1.0, tmp_path)
        assert 0.0 <= score <= 1.0

    def test_classifier_score_fallback_no_model(self, tmp_path: Path) -> None:
        words = _make_words(5)
        score = classifier_score("nonexistent_field", [0], words, 1.0, 1.0, tmp_path)
        assert score == 0.5

    def test_low_positive_fieldtype_skipped(self, tmp_path: Path) -> None:
        docs = _make_many_docs("document_id", n_docs=3)
        metrics = train_classifier("document_id", docs, tmp_path)
        assert metrics["val_f1"] is None
        assert not (tmp_path / "document_id.joblib").exists()


# ── train_all_fields tests ────────────────────────────────────────────────────


class TestTrainAllFields:
    def test_skips_low_positive_fieldtypes(self, tmp_path: Path) -> None:
        docs = _make_many_docs("document_id", n_docs=60)
        results = train_all_fields(docs, tmp_path, fieldtypes=["document_id", "vendor_name"])
        assert results["document_id"]["val_f1"] is not None
        assert results["vendor_name"]["val_f1"] is None

    def test_returns_dict_keyed_by_fieldtype(self, tmp_path: Path) -> None:
        docs = _make_many_docs("document_id", n_docs=60)
        fieldtypes = ["document_id", "date_issue"]
        results = train_all_fields(docs, tmp_path, fieldtypes=fieldtypes)
        assert set(results.keys()) == set(fieldtypes)

    def test_trained_model_files_exist(self, tmp_path: Path) -> None:
        docs = _make_many_docs("document_id", n_docs=60)
        train_all_fields(docs, tmp_path, fieldtypes=["document_id"])
        assert (tmp_path / "document_id.joblib").exists()


# ── load_doc_records tests ────────────────────────────────────────────────────


class TestLoadDocRecords:
    def test_missing_files_skipped(self, tmp_path: Path) -> None:
        records = load_doc_records(["nonexistent_doc_abc123"], tmp_path)
        assert records == []

    def test_loads_valid_json_files(self, tmp_path: Path) -> None:
        ocr_dir = tmp_path / "ocr"
        ann_dir = tmp_path / "annotations"
        ocr_dir.mkdir()
        ann_dir.mkdir()

        docid = "testdoc001"
        ocr_data = {
            "pages": [{
                "page_idx": 0,
                "dimensions": [1000, 1400],
                "orientation": 0,
                "language": "en",
                "blocks": [{
                    "lines": [{
                        "words": [
                            {
                                "value": "Invoice",
                                "confidence": 0.99,
                                "geometry": [[0.1, 0.05], [0.25, 0.08]],
                                "snapped_geometry": [[0.1, 0.05], [0.25, 0.08]],
                            },
                            {
                                "value": "12345",
                                "confidence": 0.99,
                                "geometry": [[0.3, 0.05], [0.45, 0.08]],
                                "snapped_geometry": [[0.3, 0.05], [0.45, 0.08]],
                            },
                        ]
                    }]
                }]
            }]
        }
        ann_data = {
            "field_extractions": [
                {"bbox": [0.3, 0.04, 0.45, 0.09], "fieldtype": "document_id", "page": 0, "text": "12345"}
            ],
            "line_item_extractions": [],
            "metadata": {"cluster_id": 1, "page_count": 1},
        }

        (ocr_dir / f"{docid}.json").write_text(json.dumps(ocr_data))
        (ann_dir / f"{docid}.json").write_text(json.dumps(ann_data))

        records = load_doc_records([docid], tmp_path)
        assert len(records) == 1
        rec = records[0]
        assert rec.docid == docid
        assert len(rec.pages) == 1
        assert len(rec.pages[0]) == 2
        assert len(rec.kile_fields) == 1
        assert rec.kile_fields[0]["fieldtype"] == "document_id"
