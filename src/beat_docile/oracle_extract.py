"""[RESEARCH-BURIED] Deterministic regex+checksum extractor for structured KILE fields.

Status: RESEARCH-BURIED — both post-pass and pre-pass integration shapes buried
at 500-doc scale. See KNOWLEDGE_BASE.md §6.5 for details.

Post-pass: 0.00pp delta (v2 already captures these fields).
Pre-pass: -3.97pp to -6.28pp KILE (hint injection narrows attention on other fields).
Best remaining use: as oracle presence signal for AOL recall augmentation (§5.3)
or as confidence input in an auditable-confidence ensemble (§5.5).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from .data import WordBox
from .validators import _BIC_RE, _IBAN_RE, _iban_mod97

_log = logging.getLogger(__name__)

SUPPORTED_FIELDTYPES: set[str] = {
    "iban",
    "bic",
    "account_num",
    "bank_num",
    "customer_tax_id",
    "vendor_tax_id",
    "vendor_registration_id",
    "customer_registration_id",
    "payment_reference",
    "document_id",
}


@dataclass
class OracleMatch:
    word_ids: list[int]
    text: str
    score: float  # 1.0 = checksum-verified or label-confirmed; 0.7 = pattern-only
    fieldtype: str


# ── Label context phrases (lowercase) ────────────────────────────────────────

_LABELS: dict[str, list[str]] = {
    "iban": ["iban", "international bank", "bankverbindung", "compte bancaire"],
    "bic": ["bic", "swift", "bic/swift", "swift/bic", "bank identifier", "bank code"],
    "account_num": [
        "account",
        "konto",
        "kontonummer",
        "compte",
        "conto",
        "account no",
        "account number",
        "bank account",
        "číslo účtu",
        "account:",
        "acc.",
        "acc no",
    ],
    "bank_num": [
        "sort code",
        "routing",
        "blz",
        "bankleitzahl",
        "aba",
        "bank code",
        "transit",
        "routing number",
        "bank num",
    ],
    "customer_tax_id": [
        "vat",
        "tax id",
        "tax no",
        "mwst",
        "tva",
        "nif",
        "btw",
        "dič",
        "customer vat",
        "customer tax",
        "umsatzsteuer",
        "ust-id",
        "tax number",
        "your vat",
    ],
    "vendor_tax_id": [
        "vat",
        "tax id",
        "tax no",
        "mwst",
        "tva",
        "nif",
        "btw",
        "dič",
        "vendor vat",
        "our vat",
        "umsatzsteuer",
        "ust-id",
        "tax number",
        "our tax",
    ],
    "customer_registration_id": [
        "reg",
        "registration",
        "kvk",
        "hrb",
        "ičo",
        "company no",
        "firmenbuch",
        "registered",
        "customer reg",
        "company reg",
    ],
    "vendor_registration_id": [
        "reg",
        "registration",
        "kvk",
        "hrb",
        "ičo",
        "company no",
        "firmenbuch",
        "registered",
        "abn",
        "vendor reg",
    ],
    "payment_reference": [
        "ref",
        "reference",
        "payment ref",
        "zahlungsreferenz",
        "payment reference",
        "remittance",
        "use reference",
        "ref.",
        "our ref",
        "your ref",
    ],
    "document_id": [
        "invoice",
        "rechnung",
        "facture",
        "faktura",
        "invoice no",
        "invoice number",
        "bill no",
        "document no",
        "doc no",
        "inv",
        "inv.",
        "inv no",
        "invoice #",
        "nr.",
        "no.",
        "document id",
        "bill number",
    ],
}

# ── Regex patterns ────────────────────────────────────────────────────────────

# Quick pre-filter: IBAN always starts with 2 letters + 2 digits
_IBAN_START_RE = re.compile(r"^[A-Z]{2}[0-9]{2}", re.IGNORECASE)

# Tax ID: country prefix (2 uppercase letters) + alphanumeric body, OR digit-only sequence
_TAX_ID_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{5,12}$|^\d{5,15}$", re.IGNORECASE)

# Registration ID: alphanumeric 6-15 chars with optional separators
_REG_ID_RE = re.compile(r"^[A-Z0-9][\w\-./]{5,14}$", re.IGNORECASE)

# Account / bank routing number: 8-18 digits (spaces/dashes allowed between groups)
_ACCOUNT_DIGITS_RE = re.compile(r"^\d{8,18}$")

# Payment reference: alphanumeric 5-20 chars, may contain - / .
_PAYMENT_REF_RE = re.compile(r"^[A-Z0-9][\w\-./]{4,19}$", re.IGNORECASE)

# Document ID: alphanumeric 3-20 chars with at least one digit, may contain - / _ .
_DOC_ID_RE = re.compile(r"^[A-Z0-9][\w\-./]{2,19}$", re.IGNORECASE)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _sort_by_position(words: list[WordBox]) -> list[WordBox]:
    """Sort words into reading order: top-to-bottom, left-to-right."""
    return sorted(words, key=lambda w: (round(w.bbox[1] * 50), w.bbox[0]))


def _has_label(
    pos: int,
    span_len: int,
    sorted_words: list[WordBox],
    labels: list[str],
    radius: int = 3,
) -> bool:
    """Return True if any word within ±radius of the matched span contains a label phrase."""
    lo = max(0, pos - radius)
    hi = min(len(sorted_words), pos + span_len + radius)
    context = sorted_words[lo:pos] + sorted_words[pos + span_len : hi]
    ctx = " ".join(w.text.lower() for w in context)
    return any(lbl in ctx for lbl in labels)


def _join(words: list[WordBox], sep: str = " ") -> str:
    return sep.join(w.text for w in words)


# ── Per-fieldtype extractors ──────────────────────────────────────────────────


def _extract_iban(sorted_words: list[WordBox]) -> list[OracleMatch]:
    """Scan multi-word windows for IBANs; validate with ISO 13616 mod-97."""
    matches: list[OracleMatch] = []
    seen: set[str] = set()
    n = len(sorted_words)
    i = 0
    while i < n:
        # Quick pre-filter: first word must start with 2 letters + 2 digits
        if not _IBAN_START_RE.match(sorted_words[i].text):
            i += 1
            continue
        found = False
        for size in range(1, 9):
            if i + size > n:
                break
            window = sorted_words[i : i + size]
            candidate = "".join(w.text for w in window).upper().replace(" ", "")
            if len(candidate) > 34:
                break  # Adding more words can only make it longer
            if len(candidate) < 15:
                continue
            if _IBAN_RE.match(candidate) and _iban_mod97(candidate):
                if candidate not in seen:
                    seen.add(candidate)
                    # Checksum-verified IBANs always score 1.0 regardless of label
                    matches.append(
                        OracleMatch(
                            word_ids=[w.id for w in window],
                            text=_join(window),
                            score=1.0,
                            fieldtype="iban",
                        )
                    )
                i += size
                found = True
                break
        if not found:
            i += 1
    return matches


def _extract_bic(sorted_words: list[WordBox]) -> list[OracleMatch]:
    """Extract BIC/SWIFT codes (exactly 8 or 11 uppercase chars)."""
    matches: list[OracleMatch] = []
    seen: set[str] = set()
    n = len(sorted_words)
    for i in range(n):
        for size in (1, 2):
            if i + size > n:
                break
            window = sorted_words[i : i + size]
            candidate = "".join(w.text for w in window).upper().replace(" ", "")
            if len(candidate) not in (8, 11):
                continue
            if candidate in seen:
                break
            if _BIC_RE.match(candidate):
                seen.add(candidate)
                has_lbl = _has_label(i, size, sorted_words, _LABELS["bic"])
                matches.append(
                    OracleMatch(
                        word_ids=[w.id for w in window],
                        text=_join(window),
                        score=1.0 if has_lbl else 0.7,
                        fieldtype="bic",
                    )
                )
                break
    return matches


def _extract_account_num(sorted_words: list[WordBox], fieldtype: str) -> list[OracleMatch]:
    """Extract bank account or routing numbers: 8-18 contiguous digits."""
    labels = _LABELS[fieldtype]
    matches: list[OracleMatch] = []
    seen: set[str] = set()
    n = len(sorted_words)
    for i in range(n):
        for size in range(1, 5):
            if i + size > n:
                break
            window = sorted_words[i : i + size]
            raw = " ".join(w.text for w in window)
            digits_only = re.sub(r"[\s\-]", "", raw)
            if not digits_only.isdigit():
                break  # Non-digit content — stop expanding
            if not _ACCOUNT_DIGITS_RE.match(digits_only):
                continue
            if digits_only in seen:
                break
            seen.add(digits_only)
            has_lbl = _has_label(i, size, sorted_words, labels)
            matches.append(
                OracleMatch(
                    word_ids=[w.id for w in window],
                    text=raw,
                    score=1.0 if has_lbl else 0.7,
                    fieldtype=fieldtype,
                )
            )
            break
    return matches


def _extract_tax_id(sorted_words: list[WordBox], fieldtype: str) -> list[OracleMatch]:
    """Extract VAT/tax IDs: country-prefix + alphanumeric, or digit-only sequences."""
    labels = _LABELS[fieldtype]
    matches: list[OracleMatch] = []
    seen: set[str] = set()
    n = len(sorted_words)
    for i in range(n):
        for size in range(1, 3):
            if i + size > n:
                break
            window = sorted_words[i : i + size]
            candidate = "".join(w.text for w in window).strip()
            norm = candidate.upper().replace(" ", "")
            if norm in seen:
                break
            if _TAX_ID_RE.match(norm):
                seen.add(norm)
                has_lbl = _has_label(i, size, sorted_words, labels)
                matches.append(
                    OracleMatch(
                        word_ids=[w.id for w in window],
                        text=_join(window),
                        score=1.0 if has_lbl else 0.7,
                        fieldtype=fieldtype,
                    )
                )
                break
    return matches


def _extract_registration_id(sorted_words: list[WordBox], fieldtype: str) -> list[OracleMatch]:
    """Extract company registration numbers near label context."""
    labels = _LABELS[fieldtype]
    matches: list[OracleMatch] = []
    seen: set[str] = set()
    n = len(sorted_words)
    for i in range(n):
        for size in range(1, 3):
            if i + size > n:
                break
            window = sorted_words[i : i + size]
            candidate = " ".join(w.text for w in window).strip()
            norm = re.sub(r"[\s]", "", candidate)
            if norm in seen:
                break
            if _REG_ID_RE.match(norm):
                seen.add(norm)
                has_lbl = _has_label(i, size, sorted_words, labels)
                matches.append(
                    OracleMatch(
                        word_ids=[w.id for w in window],
                        text=candidate,
                        score=1.0 if has_lbl else 0.7,
                        fieldtype=fieldtype,
                    )
                )
                break
    return matches


def _extract_payment_reference(sorted_words: list[WordBox]) -> list[OracleMatch]:
    """Extract payment references: alphanumeric 5-20 chars near reference labels."""
    labels = _LABELS["payment_reference"]
    matches: list[OracleMatch] = []
    seen: set[str] = set()
    n = len(sorted_words)
    for i in range(n):
        for size in range(1, 3):
            if i + size > n:
                break
            window = sorted_words[i : i + size]
            candidate = " ".join(w.text for w in window).strip()
            norm = candidate.replace(" ", "")
            if norm in seen:
                break
            if _PAYMENT_REF_RE.match(norm):
                seen.add(norm)
                has_lbl = _has_label(i, size, sorted_words, labels)
                matches.append(
                    OracleMatch(
                        word_ids=[w.id for w in window],
                        text=candidate,
                        score=1.0 if has_lbl else 0.7,
                        fieldtype="payment_reference",
                    )
                )
                break
    return matches


def _extract_document_id(sorted_words: list[WordBox]) -> list[OracleMatch]:
    """Extract invoice/document numbers near invoice-related labels."""
    labels = _LABELS["document_id"]
    matches: list[OracleMatch] = []
    seen: set[str] = set()
    n = len(sorted_words)
    for i in range(n):
        for size in range(1, 3):
            if i + size > n:
                break
            window = sorted_words[i : i + size]
            candidate = " ".join(w.text for w in window).strip()
            norm = candidate.replace(" ", "")
            if norm in seen:
                break
            if not _DOC_ID_RE.match(norm):
                continue
            # Require at least one digit to filter out pure label words
            if not any(c.isdigit() for c in norm):
                continue
            seen.add(norm)
            has_lbl = _has_label(i, size, sorted_words, labels)
            matches.append(
                OracleMatch(
                    word_ids=[w.id for w in window],
                    text=candidate,
                    score=1.0 if has_lbl else 0.7,
                    fieldtype="document_id",
                )
            )
            break
    return matches


# ── Public API ────────────────────────────────────────────────────────────────


def oracle_extract_field(
    fieldtype: str,
    words: list[WordBox],
    page_idx: int,
) -> list[OracleMatch]:
    """Run deterministic extraction for a single field type on a page's words.

    For each supported fieldtype, applies a regex strategy with label-proximity
    scoring. Returns [] for unsupported fieldtypes or when no match is found.

    All returned word_ids point to words in the `words` list.
    Score: 1.0 = checksum-verified (IBAN) or label-confirmed; 0.7 = pattern-only.
    """
    if fieldtype not in SUPPORTED_FIELDTYPES:
        return []

    sw = _sort_by_position(words)

    if fieldtype == "iban":
        return _extract_iban(sw)
    if fieldtype == "bic":
        return _extract_bic(sw)
    if fieldtype in ("account_num", "bank_num"):
        return _extract_account_num(sw, fieldtype)
    if fieldtype in ("customer_tax_id", "vendor_tax_id"):
        return _extract_tax_id(sw, fieldtype)
    if fieldtype in ("vendor_registration_id", "customer_registration_id"):
        return _extract_registration_id(sw, fieldtype)
    if fieldtype == "payment_reference":
        return _extract_payment_reference(sw)
    if fieldtype == "document_id":
        return _extract_document_id(sw)

    return []  # unreachable


def oracle_extract_doc(
    words_by_page: dict[int, list[WordBox]],
    fieldtypes: set[str] | None = None,
) -> list[OracleMatch]:
    """Run oracle extraction across all pages for all supported fieldtypes.

    If fieldtypes is None, run all supported types.
    Deduplicate: if the same text appears on multiple pages, keep the highest-score match.
    """
    active = (fieldtypes & SUPPORTED_FIELDTYPES) if fieldtypes is not None else SUPPORTED_FIELDTYPES

    all_matches: list[OracleMatch] = []
    for page_idx, words in sorted(words_by_page.items()):
        for ft in active:
            try:
                all_matches.extend(oracle_extract_field(ft, words, page_idx))
            except Exception:
                _log.exception("oracle_extract_field failed fieldtype=%s page=%d", ft, page_idx)

    # Deduplicate by (fieldtype, normalized_text) — keep highest score
    best: dict[tuple[str, str], OracleMatch] = {}
    for m in all_matches:
        key = (m.fieldtype, m.text.upper().replace(" ", ""))
        if key not in best or m.score > best[key].score:
            best[key] = m

    return list(best.values())
