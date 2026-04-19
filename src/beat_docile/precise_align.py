"""[ARCHIVED] Character-precise text-to-OCR-word alignment for DocILE.

Status: ARCHIVED — alignment cascade for the text-aligner approach (30-34% KILE).
See KNOWLEDGE_BASE.md §6.11. Strategy cascade: exact → NFKC → fuzzy → format.
Still below v2 by 7-12pp on every iteration. Kept for code-archaeology only.

Original design: given Claude's extracted text, find the word_id sequence whose
concatenated text best matches via cascade — EXACT, EXACT_NORM (NFKC+lower), FUZZY
(SequenceMatcher 0.85), FORMAT (validator candidates). Multi-line addresses split
and aligned per-line; amount/date fields use digit-only normalization.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from .data import WordBox

if TYPE_CHECKING:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AlignResult:
    word_ids: list[int]
    matched_text: str
    confidence: float  # 0-1; 0 = failed
    method: str  # "exact" / "exact_norm" / "fuzzy" / "format" / "multiline_*" / "failed"


_FAILED = AlignResult(word_ids=[], matched_text="", confidence=0.0, method="failed")


# ─────────────────────────────────────────────────────────────────────────────
# Field-type sets for routing
# ─────────────────────────────────────────────────────────────────────────────

_AMOUNT_FIELDS = frozenset(
    {
        "amount_due",
        "amount_paid",
        "amount_total_gross",
        "amount_total_net",
        "amount_total_tax",
        "tax_detail_gross",
        "tax_detail_net",
        "tax_detail_tax",
        "line_item_amount_gross",
        "line_item_amount_net",
        "line_item_unit_price_gross",
        "line_item_unit_price_net",
        "line_item_tax",
        "line_item_discount_amount",
    }
)

_DATE_FIELDS = frozenset({"date_due", "date_issue", "line_item_date"})

_ADDRESS_FIELDS = frozenset(
    {
        "customer_billing_address",
        "customer_delivery_address",
        "customer_other_address",
        "vendor_address",
    }
)

_CODE_FIELDS = frozenset(
    {
        "iban",
        "bic",
        "account_num",
        "bank_num",
        "vendor_tax_id",
        "customer_tax_id",
        "vendor_registration_id",
        "customer_registration_id",
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# Reading-order sort
# ─────────────────────────────────────────────────────────────────────────────


def _reading_order(words: list[WordBox]) -> list[WordBox]:
    """Sort words top-to-bottom, left-to-right (reading order)."""
    return sorted(words, key=lambda w: (round(w.bbox[1] * 100), w.bbox[0]))


# ─────────────────────────────────────────────────────────────────────────────
# Normalization helpers
# ─────────────────────────────────────────────────────────────────────────────


def _norm_ws(s: str) -> str:
    """Collapse whitespace only."""
    return re.sub(r"\s+", " ", s).strip()


def _norm_compact(s: str) -> str:
    """Remove all whitespace (for codes/IBANs split across OCR tokens)."""
    return re.sub(r"\s+", "", s)


def _norm_nfkc(s: str) -> str:
    """NFKC + lowercase + whitespace collapse."""
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", " ", s.lower()).strip()


def _norm_ocr_code(s: str) -> str:
    """NFKC + lowercase + remove spaces + common OCR char substitutions."""
    s = _norm_nfkc(s).replace(" ", "")
    return s.replace("o", "0").replace("l", "1")


def _norm_digits_only(s: str) -> str:
    """Keep only digit characters (for amount/date numeric comparison)."""
    return re.sub(r"\D", "", s)


# ─────────────────────────────────────────────────────────────────────────────
# Window construction helpers
# ─────────────────────────────────────────────────────────────────────────────


def _window_texts(words: list[WordBox]) -> tuple[str, str]:
    """Return (spaced_text, compact_text) for a word window."""
    texts = [w.text for w in words]
    return " ".join(texts), "".join(texts)


def _fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


# ─────────────────────────────────────────────────────────────────────────────
# Sliding-window matchers
# ─────────────────────────────────────────────────────────────────────────────


def _exact_match(
    query: str,
    words_sorted: list[WordBox],
    max_window: int = 20,
) -> AlignResult | None:
    """Exact match: whitespace-normalized or compact (no spaces)."""
    norm_q_ws = _norm_ws(query)
    norm_q_compact = _norm_compact(query)
    n = len(words_sorted)

    for start in range(n):
        parts: list[str] = []
        for end in range(start, min(start + max_window, n)):
            parts.append(words_sorted[end].text)
            spaced = " ".join(parts)
            compact = "".join(parts)

            if _norm_ws(spaced) == norm_q_ws or compact == norm_q_compact:
                window = words_sorted[start : end + 1]
                return AlignResult(
                    word_ids=[w.id for w in window],
                    matched_text=spaced,
                    confidence=1.0,
                    method="exact",
                )
    return None


def _exact_norm_match(
    query: str,
    words_sorted: list[WordBox],
    max_window: int = 20,
) -> AlignResult | None:
    """NFKC + lowercase normalized match; also tries OCR code substitutions."""
    norm_q = _norm_nfkc(query)
    norm_q_compact = _norm_nfkc(_norm_compact(query))
    code_q = _norm_ocr_code(query)
    n = len(words_sorted)

    for start in range(n):
        parts: list[str] = []
        for end in range(start, min(start + max_window, n)):
            parts.append(words_sorted[end].text)
            spaced = " ".join(parts)
            compact = "".join(parts)

            if (
                _norm_nfkc(spaced) == norm_q
                or _norm_nfkc(compact) == norm_q_compact
                or _norm_ocr_code(spaced) == code_q
                or _norm_ocr_code(compact) == code_q
            ):
                window = words_sorted[start : end + 1]
                return AlignResult(
                    word_ids=[w.id for w in window],
                    matched_text=spaced,
                    confidence=0.95,
                    method="exact_norm",
                )
    return None


def _digits_match(
    query: str,
    words_sorted: list[WordBox],
    max_window: int = 20,
) -> AlignResult | None:
    """Digit-only match for amounts and dates: strips all non-digit chars."""
    query_digits = _norm_digits_only(query)
    if not query_digits:
        return None

    n = len(words_sorted)
    best_ratio = 0.0
    best_window: list[WordBox] = []

    for start in range(n):
        parts: list[str] = []
        for end in range(start, min(start + max_window, n)):
            parts.append(words_sorted[end].text)
            spaced = " ".join(parts)
            compact = "".join(parts)

            for text in (spaced, compact):
                text_digits = _norm_digits_only(text)
                if text_digits == query_digits:
                    window = words_sorted[start : end + 1]
                    return AlignResult(
                        word_ids=[w.id for w in window],
                        matched_text=spaced,
                        confidence=0.9,
                        method="exact_norm",
                    )
                ratio = _fuzzy_ratio(text_digits, query_digits)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_window = words_sorted[start : end + 1]

    if best_ratio >= 0.85 and best_window:
        spaced = " ".join(w.text for w in best_window)
        return AlignResult(
            word_ids=[w.id for w in best_window],
            matched_text=spaced,
            confidence=best_ratio * 0.88,
            method="fuzzy",
        )
    return None


def _fuzzy_match(
    query: str,
    words_sorted: list[WordBox],
    max_window: int = 20,
    threshold: float = 0.85,
) -> AlignResult | None:
    """Fuzzy sliding-window match using SequenceMatcher."""
    norm_q = _norm_nfkc(query)
    n = len(words_sorted)

    best_ratio = 0.0
    best_window: list[WordBox] = []

    for start in range(n):
        parts: list[str] = []
        for end in range(start, min(start + max_window, n)):
            parts.append(words_sorted[end].text)
            spaced = " ".join(parts)
            compact = "".join(parts)

            ratio = max(
                _fuzzy_ratio(_norm_nfkc(spaced), norm_q),
                _fuzzy_ratio(_norm_nfkc(compact), norm_q),
            )
            if ratio > best_ratio:
                best_ratio = ratio
                best_window = words_sorted[start : end + 1]

    if best_ratio >= threshold and best_window:
        spaced = " ".join(w.text for w in best_window)
        return AlignResult(
            word_ids=[w.id for w in best_window],
            matched_text=spaced,
            confidence=best_ratio,
            method="fuzzy",
        )
    return None


def _format_constrained_match(
    query: str,
    words_sorted: list[WordBox],
    fieldtype: str,
    max_window: int = 20,
    format_min: float = 0.8,
    fuzzy_min: float = 0.7,
) -> AlignResult | None:
    """Find windows passing the format validator, then pick best fuzzy match."""
    from .validators import format_confidence

    norm_q = _norm_nfkc(query)
    n = len(words_sorted)

    best_combined = 0.0
    best_window: list[WordBox] = []

    for start in range(n):
        parts: list[str] = []
        for end in range(start, min(start + max_window, n)):
            parts.append(words_sorted[end].text)
            spaced = " ".join(parts)
            compact = "".join(parts)

            fc = max(
                format_confidence(fieldtype, spaced),
                format_confidence(fieldtype, compact),
            )
            if fc < format_min:
                continue

            ratio = max(
                _fuzzy_ratio(_norm_nfkc(spaced), norm_q),
                _fuzzy_ratio(_norm_nfkc(compact), norm_q),
            )
            combined = ratio * fc
            if combined > best_combined:
                best_combined = combined
                best_window = words_sorted[start : end + 1]

    if best_combined >= fuzzy_min and best_window:
        spaced = " ".join(w.text for w in best_window)
        return AlignResult(
            word_ids=[w.id for w in best_window],
            matched_text=spaced,
            confidence=best_combined,
            method="format",
        )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Multi-line alignment (addresses)
# ─────────────────────────────────────────────────────────────────────────────


def _align_single_line(
    text: str,
    words_sorted: list[WordBox],
    fieldtype: str | None = None,
    max_window: int = 20,
) -> AlignResult:
    """Align a single line of text using the strategy cascade."""
    if not text.strip() or not words_sorted:
        return _FAILED

    # Digit-only shortcut for amounts and dates (handles format variations)
    if fieldtype in _AMOUNT_FIELDS or fieldtype in _DATE_FIELDS:
        result = _digits_match(text, words_sorted, max_window)
        if result:
            # Still prefer exact if found (digits match may be less precise)
            exact = _exact_match(text, words_sorted, max_window)
            if exact:
                return exact
            return result

    # Strategy 1: EXACT
    result = _exact_match(text, words_sorted, max_window)
    if result:
        return result

    # Strategy 2: EXACT_NORM (NFKC + OCR substitutions)
    result = _exact_norm_match(text, words_sorted, max_window)
    if result:
        return result

    # Strategy 3: FORMAT-CONSTRAINED (only when fieldtype has a meaningful validator)
    if fieldtype:
        result = _format_constrained_match(text, words_sorted, fieldtype, max_window)
        if result:
            return result

    # Strategy 4: FUZZY
    result = _fuzzy_match(text, words_sorted, max_window)
    if result:
        return result

    return _FAILED


def _align_multiline(
    text: str,
    words_sorted: list[WordBox],
    fieldtype: str | None = None,
) -> AlignResult:
    """Align multi-line text (addresses) by aligning each line independently."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) <= 1:
        return _align_single_line(text, words_sorted, fieldtype)

    all_word_ids: list[int] = []
    all_texts: list[str] = []
    confidences: list[float] = []
    methods: list[str] = []

    for line in lines:
        r = _align_single_line(line, words_sorted, fieldtype)
        if r.method == "failed":
            continue  # some address lines may not be in OCR — skip
        all_word_ids.extend(r.word_ids)
        all_texts.append(r.matched_text)
        confidences.append(r.confidence)
        methods.append(r.method)

    if not all_word_ids:
        return _FAILED

    return AlignResult(
        word_ids=all_word_ids,
        matched_text="\n".join(all_texts),
        confidence=min(confidences) * 0.95,  # small penalty for multi-line
        method="multiline_" + "+".join(sorted(set(methods))),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def align_text_to_words(
    text: str,
    words: list[WordBox],
    fieldtype: str | None = None,
) -> AlignResult:
    """Find the word_id sequence whose concatenated text best matches `text`.

    Strategy cascade: exact → exact_norm → fuzzy → format_constrained.
    For address fields, splits on \\n and aligns each line independently.

    Args:
        text: Extracted field text (from Claude text-only extraction).
        words: All OCR words on the page (unsorted; will be sorted internally).
        fieldtype: DocILE field type string. Enables field-type-aware strategies.

    Returns:
        AlignResult. confidence=0.0 and method="failed" if no good match found.
    """
    if not text or not text.strip() or not words:
        return _FAILED

    words_sorted = _reading_order(words)

    if "\n" in text or fieldtype in _ADDRESS_FIELDS:
        return _align_multiline(text, words_sorted, fieldtype)

    return _align_single_line(text, words_sorted, fieldtype)


def _find_all_occurrences(
    text: str,
    words: list[WordBox],
    fieldtype: str | None = None,
    min_confidence: float = 0.5,
    max_occ: int = 30,
) -> list[AlignResult]:
    """Find all non-overlapping occurrences of text in words, in reading order.

    For multi-line/address text: returns at most one occurrence using the full
    align_text_to_words (which handles \\n splits). For single-line text: finds
    multiple occurrences by removing matched words after each hit.
    """
    # Multi-line or address fields: at most one occurrence, use full aligner
    if "\n" in text or fieldtype in _ADDRESS_FIELDS:
        align = align_text_to_words(text, words, fieldtype)
        if align.method != "failed" and align.confidence >= min_confidence:
            return [align]
        return []

    words_sorted = _reading_order(words)
    remaining = list(words_sorted)
    occurrences: list[AlignResult] = []
    seen: set[tuple[int, ...]] = set()

    for _ in range(max_occ):
        align = _align_single_line(text, remaining, fieldtype)
        if align.method == "failed" or align.confidence < min_confidence:
            break
        key = tuple(sorted(align.word_ids))
        if key in seen:
            break
        seen.add(key)
        occurrences.append(align)
        matched = set(align.word_ids)
        remaining = [w for w in remaining if w.id not in matched]

    return occurrences  # already in reading order (top-to-bottom scan)


def _make_field(
    align: AlignResult,
    ft: str,
    score: float,
    page_idx: int,
    li_id: int | None,
    id_to_word: dict,
    use_validator: bool,
) -> object | None:
    """Build a docile Field from an AlignResult, or None if invalid."""
    from docile.dataset import BBox, Field

    bboxes = [id_to_word[wid].bbox for wid in align.word_ids if wid in id_to_word]
    if not bboxes:
        return None
    bbox = BBox(
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    )
    adjusted = score * align.confidence
    if use_validator:
        # Need the original text — stored in matched_text (may differ from query)
        # Use align.matched_text as proxy; format_confidence needs the query text
        # We don't have it here, so skip validator (caller should handle)
        pass
    return Field(bbox=bbox, page=page_idx, fieldtype=ft, score=adjusted, line_item_id=li_id)


def align_fields_to_words(
    extracted: list[dict],
    words: list[WordBox],
    page_idx: int,
    *,
    min_confidence: float = 0.5,
) -> tuple[list, list]:
    """Convert text-extracted fields to docile Field objects via sequential alignment.

    Uses sequential occurrence assignment: when a (fieldtype, text) value appears
    N times in predictions, finds N OCR occurrences and assigns them 1:1 in
    reading order. This prevents multiple predictions from collapsing onto the
    same word when the same value (e.g., date, amount) repeats across line items.

    For LIR: sorts line items by line_item_id (Claude assigns in reading order),
    then assigns occurrences sequentially across line items that share a value.

    Args:
        extracted: List of dicts with keys: fieldtype, text, score, [line_item_id].
        words: OCR words for this page.
        page_idx: Page index (0-based).
        min_confidence: Minimum alignment confidence to include a field.

    Returns:
        (kile_fields, lir_fields) as lists of docile Field objects.
    """
    import os
    from collections import defaultdict

    from .extract import _KILE_TYPES, _LIR_TYPES
    from .validators import format_confidence

    use_validator = os.environ.get("BD_USE_VALIDATOR", "1") == "1"
    id_to_word = {w.id: w for w in words}

    def _field_from_align(align: AlignResult, ft: str, text: str, score: float, li_id: int | None):
        from docile.dataset import BBox, Field

        bboxes = [id_to_word[wid].bbox for wid in align.word_ids if wid in id_to_word]
        if not bboxes:
            return None
        bbox = BBox(
            min(b[0] for b in bboxes),
            min(b[1] for b in bboxes),
            max(b[2] for b in bboxes),
            max(b[3] for b in bboxes),
        )
        adjusted = score * align.confidence
        if use_validator:
            adjusted *= format_confidence(ft, text)
        return Field(bbox=bbox, page=page_idx, fieldtype=ft, score=adjusted, line_item_id=li_id)

    # ── Separate KILE and LIR items ───────────────────────────────────────────
    kile_items = []
    lir_items = []
    for item in extracted:
        ft = item.get("fieldtype", "")
        text = str(item.get("text", "")).strip()
        if not ft or not text:
            continue
        if ft in _KILE_TYPES and item.get("line_item_id") is None:
            kile_items.append(item)
        elif ft in _LIR_TYPES and item.get("line_item_id") is not None:
            lir_items.append(item)

    # ── KILE: sequential occurrence assignment ────────────────────────────────
    # Group by (fieldtype, text); find N occurrences; assign 1:1.
    kile_groups: dict[tuple, list[dict]] = defaultdict(list)
    for item in kile_items:
        key = (item["fieldtype"], str(item.get("text", "")).strip())
        kile_groups[key].append(item)

    kile: list = []
    for (ft, text), group in kile_groups.items():
        occurrences = _find_all_occurrences(
            text, words, fieldtype=ft, min_confidence=min_confidence
        )
        for item, align in zip(group, occurrences, strict=False):
            score = float(item.get("score", 0.8))
            f = _field_from_align(align, ft, text, score, None)
            if f is not None:
                kile.append(f)

    # ── LIR: group by li_id, sort by id, assign occurrences sequentially ─────
    # Sort line item groups by li_id (Claude assigns in document reading order).
    lir_groups: dict[int, list[dict]] = defaultdict(list)
    for item in lir_items:
        lir_groups[item["line_item_id"]].append(item)

    sorted_li_ids = sorted(lir_groups.keys())

    # For each (ft, text) pair, collect li_ids that have it (in sorted order).
    ft_text_li_ids: dict[tuple, list[int]] = defaultdict(list)
    for li_id in sorted_li_ids:
        for item in lir_groups[li_id]:
            key = (item["fieldtype"], str(item.get("text", "")).strip())
            ft_text_li_ids[key].append(li_id)

    # Find all occurrences per (ft, text) — do this once per unique pair.
    ft_text_occurrences: dict[tuple, list[AlignResult]] = {}
    for key, _li_ids in ft_text_li_ids.items():
        ft, text = key
        ft_text_occurrences[key] = _find_all_occurrences(
            text, words, fieldtype=ft, min_confidence=min_confidence
        )

    # Build (li_id, ft, text) → AlignResult map.
    assignment: dict[tuple, AlignResult] = {}
    for key, li_ids in ft_text_li_ids.items():
        occurrences = ft_text_occurrences[key]
        for li_id, align in zip(li_ids, occurrences, strict=False):
            assignment[(li_id, key[0], key[1])] = align

    # Build output LIR fields.
    lir: list = []
    for li_id in sorted_li_ids:
        for item in lir_groups[li_id]:
            ft = item["fieldtype"]
            text = str(item.get("text", "")).strip()
            score = float(item.get("score", 0.8))
            align = assignment.get((li_id, ft, text))
            if align is None:
                continue
            f = _field_from_align(align, ft, text, score, li_id)
            if f is not None:
                lir.append(f)

    return kile, lir
