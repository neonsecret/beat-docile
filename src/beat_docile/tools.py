"""[ARCHIVED] Tool functions exposed to Claude via the native tool-use API (V6 ReAct).

Status: ARCHIVED — tool suite for the buried V6 ReAct pipeline.
See KNOWLEDGE_BASE.md §6.7. Kept for code-archaeology only.

Original design: pure, deterministic functions returning Candidate objects,
invoked by the ReAct loop in react_extract.py when Claude calls a tool.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from .data import WordBox
from .validators import format_confidence

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    """A single extraction candidate produced by a tool."""

    word_ids: list[int]
    text: str
    score: float  # 0..1 confidence from this tool
    source: str  # "regex" / "spatial" / "validator" / "refiner" / "cluster"
    reason: str = field(default="")
    line_item_id: int | None = field(default=None)  # LIR only


# ── Regex patterns ────────────────────────────────────────────────────────────

_IBAN_BODY_RE = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}$")
_BIC_RE = re.compile(r"^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$")
_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")
_PAYMENT_REF_RE = re.compile(r"^[A-Z0-9\-/]{6,30}$", re.IGNORECASE)
_TAX_ID_RE = re.compile(r"^[A-Z]{2}[\w\-./]{5,}$", re.IGNORECASE)
_QUANTITY_RE = re.compile(r"^-?\d+([.,]\d+)?$")
_RATE_RE = re.compile(r"^\d+(\.\d+)?\s*%?$")
_AMOUNT_RE = re.compile(
    r"^(?:[$€£¥₹₽¢₩₪₦₴₺₱฿₸]|[A-Z]{3}\s*)?[\d][\d,.\s]*(?:\s*(?:[A-Z]{3}|[$€£¥₹₽¢₩₪₦₴₺₱฿₸]))?$",
    re.IGNORECASE,
)

_DATE_PATS: list[re.Pattern] = [
    re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$"),
    re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$"),
    re.compile(r"^\d{1,2}\.\d{1,2}\.\d{2,4}$"),
    re.compile(r"^\d{1,2}-\d{1,2}-\d{2,4}$"),
    re.compile(
        r"^\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*,?\s*\d{2,4}$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4}$",
        re.IGNORECASE,
    ),
]

_CURRENCY_CODES = frozenset(
    {
        "USD",
        "EUR",
        "GBP",
        "JPY",
        "CHF",
        "CAD",
        "AUD",
        "NZD",
        "CNY",
        "HKD",
        "SEK",
        "NOK",
        "DKK",
        "SGD",
        "MXN",
        "BRL",
        "INR",
        "KRW",
        "ZAR",
        "TRY",
        "RUB",
        "PLN",
        "CZK",
        "HUF",
        "RON",
        "BGN",
        "HRK",
        "ISK",
        "ILS",
        "SAR",
        "AED",
        "THB",
        "MYR",
        "IDR",
        "PHP",
        "TWD",
        "PKR",
        "EGP",
        "UAH",
        "CLP",
    }
)
_CURRENCY_SYMBOLS = frozenset("$€£¥₹₽¢₩₪₦₴₺₱฿₸")


# ── Field matchers ────────────────────────────────────────────────────────────


def _is_iban(text: str) -> bool:
    """Return True if text passes IBAN format and mod-97 checksum."""
    normalised = text.replace(" ", "").upper()
    if not (15 <= len(normalised) <= 34):
        return False
    if not _IBAN_BODY_RE.match(normalised):
        return False
    rearranged = normalised[4:] + normalised[:4]
    numeric = "".join(str(ord(ch) - 55) if ch.isalpha() else ch for ch in rearranged)
    try:
        return int(numeric) % 97 == 1
    except ValueError:
        return False


def _is_date(text: str) -> bool:
    """Return True if text matches any common date format."""
    stripped = text.strip()
    return any(p.match(stripped) for p in _DATE_PATS)


def _is_amount(text: str) -> bool:
    """Return True if text looks like a monetary amount."""
    return bool(_AMOUNT_RE.match(text.strip()))


def _is_currency(text: str) -> bool:
    """Return True if text is a currency symbol or ISO 4217 code."""
    stripped = text.strip()
    return stripped in _CURRENCY_SYMBOLS or stripped.upper() in _CURRENCY_CODES


_AMOUNT_FIELDS = frozenset(
    {
        "amount_total_gross",
        "amount_total_net",
        "amount_total_tax",
        "amount_due",
        "amount_paid",
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

# Fields that may span multiple OCR words
_MULTI_WORD_FIELDS = frozenset({"iban"}) | _DATE_FIELDS | _AMOUNT_FIELDS

# Single-word exact matchers
_SINGLE_WORD_MATCHERS: dict[str, object] = {
    "iban": _is_iban,
    "bic": lambda t: bool(_BIC_RE.match(t.strip().upper())) and len(t.strip()) in (8, 11),
    "vendor_tax_id": lambda t: bool(_TAX_ID_RE.match(t.strip())),
    "customer_tax_id": lambda t: bool(_TAX_ID_RE.match(t.strip())),
    "vendor_email": lambda t: bool(_EMAIL_RE.match(t.strip())),
    "currency_code_amount_due": _is_currency,
    "line_item_currency": _is_currency,
    "payment_reference": lambda t: bool(_PAYMENT_REF_RE.match(t.strip())),
    "line_item_quantity": lambda t: bool(_QUANTITY_RE.match(t.strip())),
    "tax_detail_rate": lambda t: bool(_RATE_RE.match(t.strip())),
    "line_item_tax_rate": lambda t: bool(_RATE_RE.match(t.strip())),
    "line_item_discount_rate": lambda t: bool(_RATE_RE.match(t.strip())),
    "line_item_position": lambda t: t.strip().isdigit(),
}


def regex_extract(fieldtype: str, words: list[WordBox]) -> list[Candidate]:
    """Run a fieldtype-specific regex over the doc text, return candidate spans.

    Scans individual words and small consecutive windows (2-4 words) when the
    field value may span multiple OCR tokens (e.g., IBAN with spaces, date).

    Args:
        fieldtype: DocILE field type string to run extraction for.
        words: All OCR words on the page, sorted top-to-bottom, left-to-right.

    Returns:
        List of Candidate objects with source="regex". Empty if fieldtype has no
        registered pattern or no matches are found.
    """
    if not words:
        return []

    if fieldtype in _AMOUNT_FIELDS:
        single_matcher = _is_amount
    elif fieldtype in _DATE_FIELDS:
        single_matcher = _is_date
    else:
        single_matcher = _SINGLE_WORD_MATCHERS.get(fieldtype)  # type: ignore[assignment]

    if single_matcher is None:
        return []

    candidates: list[Candidate] = []
    seen: set[tuple[int, ...]] = set()

    for word in words:
        if single_matcher(word.text):
            key = (word.id,)
            if key not in seen:
                seen.add(key)
                candidates.append(
                    Candidate(
                        word_ids=[word.id],
                        text=word.text,
                        score=0.85,
                        source="regex",
                        reason=f"word '{word.text}' matches {fieldtype} pattern",
                    )
                )

    if fieldtype in _MULTI_WORD_FIELDS:
        # IBANs can span up to 8 tokens when printed with spaces (e.g. GB29 NWBK 6016 1331 9268 19)
        max_window = 8 if fieldtype == "iban" else 4
        for window_size in range(2, min(max_window + 1, len(words) + 1)):
            for i in range(len(words) - window_size + 1):
                window = words[i : i + window_size]
                joined = " ".join(w.text for w in window)
                texts = [joined]
                if fieldtype == "iban":
                    texts.append("".join(w.text for w in window))
                for text in texts:
                    if single_matcher(text):
                        key = tuple(w.id for w in window)
                        if key not in seen:
                            seen.add(key)
                            candidates.append(
                                Candidate(
                                    word_ids=[w.id for w in window],
                                    text=joined,
                                    score=0.80,
                                    source="regex",
                                    reason=f"{window_size}-word window matches {fieldtype} pattern",
                                )
                            )
                        break

    return candidates


def validator_check(fieldtype: str, text: str) -> Candidate | None:
    """Wrap validators.format_confidence into a Candidate-style return.

    Args:
        fieldtype: DocILE field type to validate against.
        text: Candidate text to validate.

    Returns:
        Candidate(score=conf, source='validator') if conf >= 0.5, else None.
    """
    conf = format_confidence(fieldtype, text)
    if conf < 0.5:
        return None
    return Candidate(
        word_ids=[],
        text=text,
        score=conf,
        source="validator",
        reason=f"{fieldtype} format confidence={conf:.2f}",
    )


def spatial_neighbor(
    label_phrases: list[str],
    words: list[WordBox],
    direction: str = "right_or_below",
    max_distance_frac: float = 0.20,
) -> list[Candidate]:
    """Find words adjacent to label keywords and return them as candidates.

    Matches label words by checking whether the word text (lowercased, stripped
    of trailing punctuation) is a substring of any label_phrase or vice-versa.

    Args:
        label_phrases: Phrases to search for as label anchors, e.g. ["Invoice No", "Inv #"].
        words: All OCR words on the page.
        direction: Where to look for the value: "right", "below", or "right_or_below".
        max_distance_frac: Max gap as fraction of page dimension (0-1).

    Returns:
        List of Candidates pointing at value words near each matched label word.
    """
    if not words or not label_phrases:
        return []

    phrases_lower = [p.lower() for p in label_phrases]

    def _is_label(word: WordBox) -> bool:
        clean = word.text.lower().rstrip(":.,-;")
        # Require minimum length to avoid stop-word false positives ("no", "to", "of")
        if len(clean) < 3:
            return False
        return any(clean in ph or ph in clean for ph in phrases_lower)

    label_ids = {w.id for w in words if _is_label(w)}
    if not label_ids:
        return []

    candidates: list[Candidate] = []
    row_tol = 0.03  # ~3% page height to consider same row

    for label in words:
        if label.id not in label_ids:
            continue
        lb = label.bbox  # (left, top, right, bottom)

        nearby: list[tuple[float, WordBox, str]] = []
        for value in words:
            if value.id in label_ids:
                continue
            vb = value.bbox

            if direction in ("right", "right_or_below"):
                x_gap = vb[0] - lb[2]
                if 0 <= x_gap <= max_distance_frac and abs(vb[1] - lb[1]) <= row_tol:
                    nearby.append((x_gap, value, "right"))

            if direction in ("below", "right_or_below"):
                y_gap = vb[1] - lb[3]
                if 0 <= y_gap <= max_distance_frac and abs(vb[0] - lb[0]) <= 0.30:
                    nearby.append((y_gap, value, "below"))

        nearby.sort(key=lambda x: x[0])
        for dist, value_word, found_dir in nearby[:3]:
            candidates.append(
                Candidate(
                    word_ids=[value_word.id],
                    text=value_word.text,
                    score=max(0.1, 0.75 - dist * 2.0),
                    source="spatial",
                    reason=(
                        f"word '{value_word.text}' is {found_dir} of label '{label.text}'"
                        f" (gap={dist:.3f})"
                    ),
                )
            )

    return candidates


def cluster_fewshot(
    fieldtype: str,
    cluster_id: int | None,
    train_docs: dict,
) -> list[dict]:
    """Return up to 3 few-shot examples from train_docs matching cluster_id and fieldtype.

    Args:
        fieldtype: DocILE field type to search for in training documents.
        cluster_id: Cluster ID to match; None means return examples from any cluster.
        train_docs: Pre-loaded dict {docid: {fields: [...], words: [...], cluster_id: int}}.

    Returns:
        List of dicts with keys: docid, fieldtype, text. At most 3 entries.
    """
    examples: list[dict] = []

    for docid, doc_data in train_docs.items():
        if cluster_id is not None and doc_data.get("cluster_id") != cluster_id:
            continue
        for f in doc_data.get("fields", []):
            if f.get("fieldtype") == fieldtype:
                examples.append(
                    {
                        "docid": docid,
                        "fieldtype": fieldtype,
                        "text": f.get("text", ""),
                        "cluster_id": doc_data.get("cluster_id"),
                    }
                )
                break
        if len(examples) >= 3:
            break

    return examples


def refine_span(
    fieldtype: str,
    word_ids: list[int],
    words: list[WordBox],
    text: str,
) -> Candidate:
    """Wrap refiners.refine_field to produce a cleaned-up Candidate.

    Args:
        fieldtype: DocILE field type.
        word_ids: Raw word IDs to refine (may include label words, be multi-row, etc.).
        words: Full page word list.
        text: Claude's extracted text (used as hint by the refiner).

    Returns:
        Candidate with refined word_ids and source="refiner". Returns an empty
        Candidate (word_ids=[]) if word_ids is empty or refinement fails.
    """
    if not word_ids:
        return Candidate(
            word_ids=[],
            text=text,
            score=0.0,
            source="refiner",
            reason="empty input word_ids",
        )

    from .refiners import refine_field

    refined_ids, _ = refine_field(fieldtype, word_ids, words, text)

    id_to_word = {w.id: w for w in words}
    refined_text = " ".join(id_to_word[wid].text for wid in refined_ids if wid in id_to_word)

    return Candidate(
        word_ids=refined_ids,
        text=refined_text or text,
        score=0.90,
        source="refiner",
        reason=f"refined {len(word_ids)} → {len(refined_ids)} word_ids",
    )


def classifier_score_tool(
    fieldtype: str,
    word_ids: list[int],
    words: list[WordBox],
    page_w: float,
    page_h: float,
    model_dir: Path,
) -> float:
    """Wraps classifiers.classifier_score. Returns p(positive) for the span being
    an instance of fieldtype. Returns 0.5 if no model trained for this fieldtype.
    """
    from .classifiers import classifier_score

    return classifier_score(fieldtype, word_ids, words, page_w, page_h, model_dir)
