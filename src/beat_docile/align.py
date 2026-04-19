"""[ACTIVE] Text-first OCR alignment — fuzzy span matching for field localization.

Status: ACTIVE — used in current best (v2_ensemble) via extract.py fallback path.
See KNOWLEDGE_BASE.md §3 for the architecture map.

Claude extracts WHAT the field value is; this module finds WHERE it is.
Uses sliding window + SequenceMatcher to match against snapped OCR words,
avoiding the word_id hallucination problem (PCC-IoU=1.0 punishes bad spans).
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

from docile.dataset import BBox, Field

from .data import WordBox


def _normalize(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace, remove punctuation."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _word_tokens(words: list[WordBox]) -> list[str]:
    return [_normalize(w.text) for w in words]


def _text_tokens(text: str) -> list[str]:
    return [t for t in _normalize(text).split() if t]


def find_span(
    query_text: str,
    words: list[WordBox],
    *,
    min_ratio: float = 0.75,
) -> tuple[int, int] | None:
    """Find the best contiguous span of words matching query_text.

    Returns (start_idx, end_idx) inclusive in words list, or None if no good match.
    Uses sliding window + SequenceMatcher for fuzzy matching.
    """
    if not query_text or not words:
        return None

    q_tokens = _text_tokens(query_text)
    w_tokens = _word_tokens(words)
    if not q_tokens:
        return None

    n_q = len(q_tokens)
    n_w = len(w_tokens)

    best_ratio = 0.0
    best_span: tuple[int, int] | None = None

    # Try spans of length n_q ± 2 (allow slight over/under by OCR tokenization)
    for span_len in range(max(1, n_q - 2), n_q + 3):
        if span_len > n_w:
            break
        for start in range(n_w - span_len + 1):
            window = " ".join(w_tokens[start : start + span_len])
            ratio = SequenceMatcher(None, " ".join(q_tokens), window).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_span = (start, start + span_len - 1)

    if best_ratio >= min_ratio and best_span is not None:
        return best_span
    return None


def text_fields_to_docile(
    text_predictions: list[dict],
    words: list[WordBox],
    page_idx: int,
    *,
    min_ratio: float = 0.75,
) -> tuple[list[Field], list[Field]]:
    """Convert text-based predictions to Field objects via OCR alignment.

    text_predictions: list of dicts with keys: fieldtype, text, score, [line_item_id]
    words: snapped OCR words for this page
    page_idx: page index

    Returns (kile_fields, lir_fields).
    """
    from .extract import _KILE_TYPES, _LIR_TYPES

    kile: list[Field] = []
    lir: list[Field] = []

    for pred in text_predictions:
        ft = pred.get("fieldtype", "")
        if ft not in _KILE_TYPES and ft not in _LIR_TYPES:
            continue

        value_text = str(pred.get("text", ""))
        score = float(pred.get("score", 0.8))
        li_id = pred.get("line_item_id")

        span = find_span(value_text, words, min_ratio=min_ratio)
        if span is None:
            continue

        start, end = span
        span_words = words[start : end + 1]
        left = min(w.bbox[0] for w in span_words)
        top = min(w.bbox[1] for w in span_words)
        right = max(w.bbox[2] for w in span_words)
        bottom = max(w.bbox[3] for w in span_words)
        bbox = BBox(left, top, right, bottom)

        field = Field(bbox=bbox, page=page_idx, fieldtype=ft, score=score, line_item_id=li_id)
        if li_id is not None:
            lir.append(field)
        else:
            kile.append(field)

    return kile, lir
