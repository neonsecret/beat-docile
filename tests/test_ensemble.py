"""Tests for beat_docile.ensemble.merge_predictions.

Covers the 4 scenarios from PLAN_V2 §ensembler:
  1. Identical predictions across sources collapse to one merged field.
  2. Disjoint bboxes both survive.
  3. LIR predictions with same fieldtype but different line_item_id stay separate.
  4. Sanity: real ensemble of v2_preds vs v5_baseline on 50-doc overlap.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from docile.dataset import BBox, Field

from beat_docile.ensemble import (
    _iou,
    load_predictions,
    merge_predictions,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def kf(fieldtype: str, bbox: tuple, score: float = 0.9, page: int = 0) -> Field:
    return Field(bbox=BBox(*bbox), page=page, fieldtype=fieldtype, score=score)


def lf(fieldtype: str, bbox: tuple, li_id: int, score: float = 0.9, page: int = 0) -> Field:
    return Field(
        bbox=BBox(*bbox), page=page, fieldtype=fieldtype, score=score, line_item_id=li_id
    )


# ── _iou sanity ───────────────────────────────────────────────────────────────


def test_iou_identical_is_one():
    b = BBox(0.1, 0.1, 0.2, 0.2)
    assert _iou(b, b) == pytest.approx(1.0)


def test_iou_disjoint_is_zero():
    a = BBox(0.0, 0.0, 0.1, 0.1)
    b = BBox(0.5, 0.5, 0.6, 0.6)
    assert _iou(a, b) == 0.0


def test_iou_partial_overlap():
    a = BBox(0.0, 0.0, 0.2, 0.2)
    b = BBox(0.1, 0.1, 0.3, 0.3)
    # intersection 0.01, union 0.04 + 0.04 - 0.01 = 0.07
    assert _iou(a, b) == pytest.approx(0.01 / 0.07)


# ── Scenario 1: identical predictions collapse ─────────────────────────────────


def test_identical_predictions_collapse_to_one():
    bbox = (0.1, 0.1, 0.3, 0.2)
    src_a = {"doc1": [kf("vendor_name", bbox, score=0.8)]}
    src_b = {"doc1": [kf("vendor_name", bbox, score=0.9)]}

    merged = merge_predictions([src_a, src_b], weights=[1.0, 1.0])
    assert "doc1" in merged
    assert len(merged["doc1"]) == 1
    f = merged["doc1"][0]
    assert f.fieldtype == "vendor_name"
    # weighted_max default → max(0.8*1, 0.9*1) = 0.9
    assert f.score == pytest.approx(0.9)


def test_identical_predictions_weighted_mean():
    bbox = (0.1, 0.1, 0.3, 0.2)
    src_a = {"d": [kf("vendor_name", bbox, score=0.8)]}
    src_b = {"d": [kf("vendor_name", bbox, score=0.6)]}

    merged = merge_predictions(
        [src_a, src_b], weights=[0.7, 0.3], score_combine="weighted_mean"
    )
    f = merged["d"][0]
    # (0.8*0.7 + 0.6*0.3) / (0.7+0.3) = 0.74
    assert f.score == pytest.approx(0.74)


# ── Scenario 2: disjoint bboxes both survive ─────────────────────────────────


def test_disjoint_bboxes_both_kept():
    src_a = {"doc1": [kf("vendor_name", (0.1, 0.1, 0.2, 0.2), score=0.8)]}
    src_b = {"doc1": [kf("vendor_name", (0.6, 0.6, 0.8, 0.8), score=0.9)]}

    merged = merge_predictions([src_a, src_b], weights=[1.0, 1.0])
    assert len(merged["doc1"]) == 2
    types = {f.fieldtype for f in merged["doc1"]}
    assert types == {"vendor_name"}


def test_singleton_score_scaled_by_weight():
    src_a = {"d": [kf("date_issue", (0.0, 0.0, 0.1, 0.05), score=0.9)]}
    src_b: dict = {"d": []}

    merged = merge_predictions(
        [src_a, src_b], weights=[0.5, 0.5], score_combine="weighted_max"
    )
    assert len(merged["d"]) == 1
    # weighted_max of single source: 0.9 * 0.5 = 0.45
    assert merged["d"][0].score == pytest.approx(0.45)


# ── Scenario 3: LIR — line_item_id keeps things separate ─────────────────────


def test_lir_different_line_item_ids_stay_separate():
    bbox = (0.1, 0.1, 0.3, 0.15)
    src_a = {"d": [lf("line_item_quantity", bbox, li_id=1, score=0.8)]}
    src_b = {"d": [lf("line_item_quantity", bbox, li_id=2, score=0.8)]}

    merged = merge_predictions([src_a, src_b], weights=[1.0, 1.0])
    assert len(merged["d"]) == 2
    li_ids = sorted(f.line_item_id for f in merged["d"])
    assert li_ids == [1, 2]


def test_lir_same_line_item_id_merges():
    bbox = (0.1, 0.1, 0.3, 0.15)
    src_a = {"d": [lf("line_item_description", bbox, li_id=3, score=0.7)]}
    src_b = {"d": [lf("line_item_description", bbox, li_id=3, score=0.85)]}

    merged = merge_predictions([src_a, src_b], weights=[1.0, 1.0])
    assert len(merged["d"]) == 1
    assert merged["d"][0].line_item_id == 3


# ── Cross-page never collapses ───────────────────────────────────────────────


def test_same_bbox_different_pages_not_merged():
    bbox = (0.1, 0.1, 0.3, 0.2)
    src_a = {"d": [kf("document_id", bbox, page=0)]}
    src_b = {"d": [kf("document_id", bbox, page=1)]}

    merged = merge_predictions([src_a, src_b], weights=[1.0, 1.0])
    assert len(merged["d"]) == 2
    pages = sorted(f.page for f in merged["d"])
    assert pages == [0, 1]


# ── Different fieldtypes don't merge even if overlapping ─────────────────────


def test_different_fieldtypes_not_merged():
    bbox = (0.1, 0.1, 0.3, 0.2)
    src_a = {"d": [kf("vendor_name", bbox)]}
    src_b = {"d": [kf("customer_billing_name", bbox)]}

    merged = merge_predictions([src_a, src_b], weights=[1.0, 1.0])
    assert len(merged["d"]) == 2


# ── Vote score_combine ───────────────────────────────────────────────────────


def test_vote_combine_three_sources_two_agree():
    bbox = (0.1, 0.1, 0.3, 0.2)
    src_a = {"d": [kf("vendor_name", bbox, score=0.5)]}
    src_b = {"d": [kf("vendor_name", bbox, score=0.5)]}
    src_c: dict = {"d": []}

    merged = merge_predictions(
        [src_a, src_b, src_c], weights=[1.0, 1.0, 1.0], score_combine="vote"
    )
    assert len(merged["d"]) == 1
    # 2 of 3 sources voted → 2/3
    assert merged["d"][0].score == pytest.approx(2 / 3)


# ── Empty inputs ─────────────────────────────────────────────────────────────


def test_empty_sources_returns_empty():
    assert merge_predictions([]) == {}


def test_single_source_passes_through():
    src = {"d": [kf("vendor_name", (0.0, 0.0, 0.1, 0.1), score=0.7)]}
    merged = merge_predictions([src], weights=[1.0])
    assert len(merged["d"]) == 1
    assert merged["d"][0].score == pytest.approx(0.7)


def test_invalid_weights_length_raises():
    with pytest.raises(ValueError):
        merge_predictions([{"d": []}], weights=[1.0, 1.0])


def test_invalid_combine_mode_raises():
    with pytest.raises(ValueError):
        merge_predictions(
            [{"d": [kf("vendor_name", (0, 0, 0.1, 0.1))]}],
            score_combine="bogus",
        )


# ── Scenario 4: real predictions sanity (round-trip + bucket counts) ─────────

PRED_DIR = Path(__file__).resolve().parent.parent / "predictions"


@pytest.mark.skipif(
    not (PRED_DIR / "v2_preds.json").exists() or not (PRED_DIR / "v5_baseline_50.json").exists(),
    reason="prediction fixtures not present",
)
def test_real_predictions_merge_does_not_explode():
    v2 = load_predictions(PRED_DIR / "v2_preds.json")
    v5 = load_predictions(PRED_DIR / "v5_baseline_50.json")
    common = set(v2) & set(v5)
    assert len(common) == 50

    v2_subset = {d: v2[d] for d in common}
    merged = merge_predictions(
        [v2_subset, v5], weights=[0.5, 0.5], iou_threshold=0.5, score_combine="weighted_max"
    )
    assert set(merged.keys()) == common

    # Merged total should never exceed sum of inputs
    total_in = sum(len(v) for v in v2_subset.values()) + sum(len(v) for v in v5.values())
    total_out = sum(len(v) for v in merged.values())
    assert total_out <= total_in
    # And should be at least the max of either single source (no field deletion)
    assert total_out >= max(
        sum(len(v) for v in v2_subset.values()), sum(len(v) for v in v5.values())
    )
