"""[RESEARCH-BURIED] Sliding-window classifier candidate generator for DocILE.

Status: RESEARCH-BURIED — Option A (augment) produced -19pp regression due to
~300 FPs per doc. See KNOWLEDGE_BASE.md §6.12 for details. Classifiers were
trained with random negatives and don't track doc-level field presence.
Option B (rerank) also buried at 250-doc (-2.5pp KILE); see §6.6 / §5.1.
LIR-only reranking +1.6pp is the one positive finding (see §8.9).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .classifiers import (
    _ALL_FIELDTYPES,
    _FEATURE_DIM,
    _ROW_GAP_FRAC,
    extract_features,
    featurize_for_sklearn,
    load_classifier,
)
from .data import WordBox

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fieldtype classification
# ---------------------------------------------------------------------------

_ADDRESS_FIELDTYPES: frozenset[str] = frozenset(
    {
        "customer_billing_address",
        "customer_delivery_address",
        "customer_other_address",
        "vendor_address",
    }
)

_LIR_FIELDTYPES: frozenset[str] = frozenset(
    {
        "line_item_amount_gross",
        "line_item_amount_net",
        "line_item_code",
        "line_item_currency",
        "line_item_date",
        "line_item_description",
        "line_item_discount_amount",
        "line_item_discount_rate",
        "line_item_hts_number",
        "line_item_order_id",
        "line_item_person_name",
        "line_item_position",
        "line_item_quantity",
        "line_item_tax",
        "line_item_tax_rate",
        "line_item_unit_price_gross",
        "line_item_unit_price_net",
        "line_item_units_of_measure",
        "line_item_weight",
    }
)

# Fieldtypes to skip for Option A recall augmentation.
# Includes: classifiers with val_f1 < 0.5, null metrics (< 10 positives),
# and all LIR types (grouping complexity).
_SKIP_FOR_RECALL: frozenset[str] = (
    frozenset(
        {
            "bic",  # val_f1 = 0.0
            "customer_delivery_name",  # val_f1 = 0.37
            "customer_registration_id",  # null (3 positives)
            "customer_tax_id",  # val_f1 = 0.0
            "iban",  # null (3 positives)
            "line_item_discount_amount",  # null (6 positives)
            "line_item_discount_rate",  # null (3 positives)
            "line_item_person_name",  # val_f1 = 0.47
            "line_item_weight",  # val_f1 = 0.22
            "tax_detail_rate",  # val_f1 = 0.0
            "vendor_registration_id",  # val_f1 = 0.0
        }
    )
    | _LIR_FIELDTYPES
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CandidateSpan:
    fieldtype: str
    word_ids: list[int]
    bbox: tuple[float, float, float, float]  # (left, top, right, bottom) in [0,1]
    page: int
    score: float  # classifier p(positive)
    text: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _span_text(word_ids: list[int], id_map: dict[int, WordBox]) -> str:
    return " ".join(id_map[wid].text for wid in word_ids if wid in id_map)


def _span_bbox(
    word_ids: list[int], id_map: dict[int, WordBox]
) -> tuple[float, float, float, float]:
    bboxes = [id_map[wid].bbox for wid in word_ids if wid in id_map]
    if not bboxes:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    )


def _deduplicate_candidates(candidates: list[CandidateSpan]) -> list[CandidateSpan]:
    """Keep highest-scoring span for each unique (page, frozenset(word_ids)) key."""
    seen: dict[tuple, CandidateSpan] = {}
    for c in candidates:
        key = (c.page, tuple(sorted(c.word_ids)))
        if key not in seen or c.score > seen[key].score:
            seen[key] = c
    return sorted(seen.values(), key=lambda x: -x.score)


def _sort_words_reading_order(words: list[WordBox]) -> list[WordBox]:
    """Row-first reading order: bucket by _ROW_GAP_FRAC, then sort left-to-right."""
    return sorted(words, key=lambda w: (round(w.bbox[1] / _ROW_GAP_FRAC), w.bbox[0]))


def _default_max_span(fieldtype: str) -> int:
    """Return sensible default max window size for the given fieldtype."""
    return 20 if fieldtype in _ADDRESS_FIELDTYPES else 4


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_candidates(
    fieldtype: str,
    words: list[WordBox],
    page_w: float,
    page_h: float,
    model_dir: Path,
    min_span_words: int = 1,
    max_span_words: int = 8,
    score_threshold: float = 0.7,
) -> list[CandidateSpan]:
    """Sliding window over words, score each (fieldtype, span) with the classifier.
    Return all spans with classifier score > threshold, deduplicated.

    For multi-line fields (addresses): allow span up to max_span_words=20 with row-tolerance.
    For single-line fields (iban, dates, amounts): max_span_words=4, single-row only.
    """
    if not words:
        return []

    pipeline = load_classifier(fieldtype, model_dir)
    if pipeline is None:
        return []

    is_address = fieldtype in _ADDRESS_FIELDTYPES
    # When caller leaves default (8), apply the fieldtype-appropriate window size.
    if max_span_words == 8:
        max_span_words = _default_max_span(fieldtype)

    sorted_words = _sort_words_reading_order(words)
    id_map = {w.id: w for w in sorted_words}
    n = len(sorted_words)

    # Build all candidate spans using a sliding window in reading order.
    span_ids_list: list[list[int]] = []
    for start in range(n):
        start_y = sorted_words[start].bbox[1]
        for end in range(start + min_span_words - 1, min(start + max_span_words, n)):
            end_word = sorted_words[end]
            # For single-value fields, stop extending once we cross a row boundary.
            if not is_address and abs(end_word.bbox[1] - start_y) > _ROW_GAP_FRAC * 3:
                break
            span_ids = [sorted_words[i].id for i in range(start, end + 1)]
            span_ids_list.append(span_ids)

    if not span_ids_list:
        return []

    # Batch featurize to avoid per-span overhead from repeated sklearn calls.
    feat_matrix = np.zeros((len(span_ids_list), _FEATURE_DIM), dtype=np.float32)
    for i, span_ids in enumerate(span_ids_list):
        feats = extract_features(span_ids, sorted_words, page_w, page_h)
        feat_matrix[i] = featurize_for_sklearn(feats)

    probs: np.ndarray = pipeline.predict_proba(feat_matrix)[:, 1]

    page = words[0].page
    results: list[CandidateSpan] = []
    for i, span_ids in enumerate(span_ids_list):
        score = float(probs[i])
        if score >= score_threshold:
            bbox = _span_bbox(span_ids, id_map)
            text = _span_text(span_ids, id_map)
            results.append(
                CandidateSpan(
                    fieldtype=fieldtype,
                    word_ids=span_ids,
                    bbox=bbox,
                    page=page,
                    score=score,
                    text=text,
                )
            )

    return _deduplicate_candidates(results)


def generate_doc_candidates(
    words_by_page: dict[int, list[WordBox]],
    fieldtypes: list[str] | None = None,
    model_dir: Path = Path("models/classifiers"),
    score_threshold: float = 0.7,
) -> dict[str, list[CandidateSpan]]:
    """Run candidate generation for all field types across all pages of a doc.
    Returns {fieldtype: [CandidateSpan, ...]}.
    """
    if fieldtypes is None:
        fieldtypes = _ALL_FIELDTYPES

    results: dict[str, list[CandidateSpan]] = {ft: [] for ft in fieldtypes}

    for page_words in words_by_page.values():
        if not page_words:
            continue
        for ft in fieldtypes:
            candidates = generate_candidates(
                fieldtype=ft,
                words=page_words,
                page_w=1.0,
                page_h=1.0,
                model_dir=model_dir,
                max_span_words=_default_max_span(ft),
                score_threshold=score_threshold,
            )
            results[ft].extend(candidates)

    for ft in results:
        results[ft].sort(key=lambda x: -x.score)

    return results


def score_bbox_span(
    fieldtype: str,
    pred_bbox: tuple[float, float, float, float],
    page_words: list[WordBox],
    model_dir: Path,
    margin: float = 0.005,
) -> float:
    """Return classifier score for a bbox-defined span (for Option B reranking).

    Finds OCR words whose centre falls within pred_bbox (with small margin),
    extracts features, and returns p(fieldtype).  Returns 0.5 (neutral) if
    no words are found (bbox too tight) or no model is available.
    """
    pred_l, pred_t, pred_r, pred_b = pred_bbox
    covered: list[int] = []
    for w in page_words:
        wl, wt, wr, wb = w.bbox
        cx = (wl + wr) / 2.0
        cy = (wt + wb) / 2.0
        if (pred_l - margin) <= cx <= (pred_r + margin) and (pred_t - margin) <= cy <= (
            pred_b + margin
        ):
            covered.append(w.id)

    if not covered:
        return 0.5

    pipeline = load_classifier(fieldtype, model_dir)
    if pipeline is None:
        return 0.5

    feats = extract_features(covered, page_words, 1.0, 1.0)
    vec = featurize_for_sklearn(feats).reshape(1, -1)
    return float(pipeline.predict_proba(vec)[0, 1])
