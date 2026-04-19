"""[RESEARCH-BURIED] GLiNER-based post-extraction conflict resolver.

Status: RESEARCH-BURIED — ±0pp net effect. See KNOWLEDGE_BASE.md §6.18 for details.
Root cause: GLiNER is designed to choose between competing candidates; Sonnet
emits a single categorical pick per field, so there are no competing candidates
to disambiguate. Might compose with classifier reranking (§5.1) where multiple
candidates exist per field.

Post-processing pass over KILE field predictions:
  1. Find conflict groups: 2+ fields with overlapping bboxes and close scores
  2. For ambiguous pairs, run GLiNER with description-based labels
  3. If GLiNER gap > RESOLVE_GAP_THRESHOLD: drop loser; else abstain
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import TYPE_CHECKING

from docile.dataset import BBox, Field

from .data import WordBox

if TYPE_CHECKING:
    from gliner import GLiNER as _GLiNERType

_log = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────

CONFLICT_GAP_THRESHOLD: float = 0.20
RESOLVE_GAP_THRESHOLD: float = 0.15
BBOX_OVERLAP_THRESHOLD: float = 0.30

# ── Ambiguous field-pair taxonomy (ordered by expected KILE impact) ───────────

AMBIGUOUS_PAIRS: set[frozenset[str]] = {
    frozenset({"vendor_tax_id", "customer_tax_id"}),
    frozenset({"vendor_registration_id", "customer_registration_id"}),
    frozenset({"date_issue", "date_due"}),
    frozenset({"amount_total_gross", "amount_total_net"}),
    frozenset({"document_id", "payment_reference"}),
    # NOTE: amount_due + amount_total_gross intentionally excluded —
    # DocILE allows both to be annotated at the same bbox (same value on simple invoices).
    frozenset({"vendor_name", "customer_billing_name"}),
    frozenset({"order_id", "customer_order_id"}),
    frozenset({"order_id", "vendor_order_id"}),
    frozenset({"account_num", "bank_num"}),
}

# ── Description-based NER labels for GLiNER ──────────────────────────────────
# Longer, more specific descriptions outperform label names alone (ZeroNER ACL 2025).

FIELDTYPE_TO_LABEL: dict[str, str] = {
    "vendor_tax_id": "vendor VAT or tax identification number",
    "customer_tax_id": "customer VAT or tax identification number",
    "vendor_registration_id": "vendor company registration number",
    "customer_registration_id": "customer company registration number",
    "date_issue": "invoice issue date or invoice creation date",
    "date_due": "payment due date or pay-by date",
    "amount_total_gross": "total gross amount including tax",
    "amount_total_net": "total net amount before tax",
    "amount_due": "amount due for payment or balance due",
    "document_id": "invoice number or document identifier",
    "payment_reference": "payment reference code or remittance reference",
    "vendor_name": "vendor or supplier company name",
    "customer_billing_name": "customer or buyer company name",
    "order_id": "generic order number",
    "customer_order_id": "customer purchase order number",
    "vendor_order_id": "vendor order number",
    "account_num": "bank account number",
    "bank_num": "bank routing or sort code",
}

# ── Lazy model loader (CPU-only) ──────────────────────────────────────────────

_MODEL: _GLiNERType | None = None
MODEL_ID = "knowledgator/gliner-multitask-large-v0.5"


def _get_model() -> _GLiNERType:
    global _MODEL
    if _MODEL is None:
        from gliner import GLiNER

        _log.info("Loading GLiNER model %s (CPU)...", MODEL_ID)
        _MODEL = GLiNER.from_pretrained(MODEL_ID)
        _MODEL.eval()
    return _MODEL


# ── Geometry helpers ──────────────────────────────────────────────────────────


def _bbox_overlap(b1: BBox, b2: BBox, threshold: float = BBOX_OVERLAP_THRESHOLD) -> bool:
    ix1 = max(b1.left, b2.left)
    iy1 = max(b1.top, b2.top)
    ix2 = min(b1.right, b2.right)
    iy2 = min(b1.bottom, b2.bottom)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return False
    area1 = (b1.right - b1.left) * (b1.bottom - b1.top)
    area2 = (b2.right - b2.left) * (b2.bottom - b2.top)
    min_area = min(area1, area2)
    return (inter_area / min_area) >= threshold if min_area > 0 else False


def _build_context(f: Field, words: list[WordBox], radius: int = 5) -> str:
    """Build context: ±radius words around the field bbox in reading order."""
    sorted_words = sorted(words, key=lambda w: (round(w.bbox[1] * 50), w.bbox[0]))

    def word_bbox(w: WordBox) -> BBox:
        return BBox(w.bbox[0], w.bbox[1], w.bbox[2], w.bbox[3])

    matched_indices = [
        i for i, w in enumerate(sorted_words) if _bbox_overlap(word_bbox(w), f.bbox, threshold=0.1)
    ]

    if not matched_indices:
        return f.text or ""

    lo = max(0, min(matched_indices) - radius)
    hi = min(len(sorted_words), max(matched_indices) + radius + 1)
    return " ".join(w.text for w in sorted_words[lo:hi])


# ── Stats ─────────────────────────────────────────────────────────────────────


@dataclass
class DisambigStats:
    n_conflicts: int = 0
    n_resolved: int = 0
    n_abstained: int = 0
    # "fieldtype_a|fieldtype_b" (sorted) → count
    pair_counts: dict[str, int] = dc_field(default_factory=dict)
    pair_resolved: dict[str, int] = dc_field(default_factory=dict)


# ── Public API ────────────────────────────────────────────────────────────────


def resolve_conflicts(
    fields: list[Field],
    words_by_page: dict[int, list[WordBox]],
    conflict_gap: float = CONFLICT_GAP_THRESHOLD,
    resolve_gap: float = RESOLVE_GAP_THRESHOLD,
    overlap_threshold: float = BBOX_OVERLAP_THRESHOLD,
) -> tuple[list[Field], DisambigStats]:
    """Resolve KILE field conflicts for one document using GLiNER.

    Only resolves pairs in AMBIGUOUS_PAIRS where both V5b scores are within
    conflict_gap of each other. GLiNER must achieve gap > resolve_gap to override.
    LIR fields (line_item_id is not None) pass through untouched.

    Args:
        fields: All KILE+LIR fields for one doc (V5b predictions).
        words_by_page: {page_idx: [WordBox, ...]} from iter_pages.

    Returns:
        (resolved_fields, stats)
    """
    stats = DisambigStats()

    kile_fields = [f for f in fields if f.line_item_id is None]
    lir_fields = [f for f in fields if f.line_item_id is not None]

    n = len(kile_fields)
    conflict_pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = kile_fields[i], kile_fields[j]
            if a.page != b.page:
                continue
            pair_key = frozenset({a.fieldtype, b.fieldtype})
            if pair_key not in AMBIGUOUS_PAIRS:
                continue
            if not _bbox_overlap(a.bbox, b.bbox, threshold=overlap_threshold):
                continue
            if abs(a.score - b.score) >= conflict_gap:
                continue
            conflict_pairs.append((i, j))
            stats.n_conflicts += 1
            label = "|".join(sorted([a.fieldtype, b.fieldtype]))
            stats.pair_counts[label] = stats.pair_counts.get(label, 0) + 1

    if not conflict_pairs:
        return fields, stats

    model = _get_model()
    drop_indices: set[int] = set()

    for i, j in conflict_pairs:
        a, b = kile_fields[i], kile_fields[j]
        page_words = words_by_page.get(a.page, [])

        label_a = FIELDTYPE_TO_LABEL.get(a.fieldtype, a.fieldtype)
        label_b = FIELDTYPE_TO_LABEL.get(b.fieldtype, b.fieldtype)

        context = _build_context(a, page_words)
        if not context.strip():
            context = _build_context(b, page_words)
        if not context.strip():
            stats.n_abstained += 1
            continue

        try:
            entities = model.predict_entities(
                context,
                [label_a, label_b],
                threshold=0.05,  # low: we want scores, not hard pass/fail
            )
        except Exception:
            _log.exception(
                "GLiNER failed: %s vs %s | context: %.80s",
                a.fieldtype,
                b.fieldtype,
                context,
            )
            stats.n_abstained += 1
            continue

        score_a = max((e["score"] for e in entities if e["label"] == label_a), default=0.0)
        score_b = max((e["score"] for e in entities if e["label"] == label_b), default=0.0)
        gliner_gap = abs(score_a - score_b)

        if gliner_gap < resolve_gap:
            stats.n_abstained += 1
            continue

        loser_idx = j if score_a >= score_b else i
        if loser_idx not in drop_indices:
            drop_indices.add(loser_idx)
            stats.n_resolved += 1
            label = "|".join(sorted([a.fieldtype, b.fieldtype]))
            stats.pair_resolved[label] = stats.pair_resolved.get(label, 0) + 1

    resolved_kile = [f for k, f in enumerate(kile_fields) if k not in drop_indices]
    return resolved_kile + lir_fields, stats
