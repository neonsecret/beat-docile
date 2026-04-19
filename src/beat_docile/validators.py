"""[EXPERIMENTAL] Format-confidence validators for DocILE field types.

Status: EXPERIMENTAL — built and tested; was neutral on AP when last measured.
See KNOWLEDGE_BASE.md §3.3 for details. Currently OFF in the standing best.

Each validator returns a float in [0.0, 1.0]:
  1.0 — text matches expected format
  0.5 — ambiguous / uncertain
  0.0 — clearly wrong format

Use as a score multiplier:
    field.score *= format_confidence(field.fieldtype, field.text)

Design principle: BE CONSERVATIVE. Return 1.0 when in doubt.
"""

from __future__ import annotations

import re
from datetime import datetime

# ── IBAN ─────────────────────────────────────────────────────────────────────

_IBAN_RE = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}$")


def _iban_mod97(iban: str) -> bool:
    """Return True if the IBAN passes the ISO 13616 mod-97 checksum."""
    # Move first 4 chars to end
    rearranged = iban[4:] + iban[:4]
    # Convert letters to digits: A=10, B=11, ...
    numeric = "".join(str(ord(ch) - 55) if ch.isalpha() else ch for ch in rearranged)
    try:
        return int(numeric) % 97 == 1
    except ValueError:
        return False


def _validate_iban(text: str) -> float:
    normalised = text.replace(" ", "").upper()
    if not (15 <= len(normalised) <= 34):
        return 0.0
    if not _IBAN_RE.match(normalised):
        return 0.0
    if not _iban_mod97(normalised):
        return 0.0
    return 1.0


# ── BIC / SWIFT ───────────────────────────────────────────────────────────────

_BIC_RE = re.compile(r"^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$")


def _validate_bic(text: str) -> float:
    normalised = text.strip().upper()
    if len(normalised) not in (8, 11):
        return 0.0
    if not _BIC_RE.match(normalised):
        return 0.0
    return 1.0


# ── VAT / Tax IDs ─────────────────────────────────────────────────────────────

# Plausible tax-ID pattern: starts with 2 letters OR has 5+ digits, may contain
# hyphens, slashes, dots as separators.  Spaces make it ambiguous (0.5).
_TAX_ID_STRICT_RE = re.compile(r"^[A-Z]{2}[\w\-./]{5,}$", re.IGNORECASE)
_TAX_ID_DIGITS_RE = re.compile(r"^\d[\d\-./]{4,}$")
_TAX_ID_HAS_SPACE = re.compile(r"\s")
_TAX_ID_GARBAGE_RE = re.compile(r"[^\w\s\-./]")


def _validate_tax_id(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 0.0
    if _TAX_ID_GARBAGE_RE.search(stripped):
        return 0.0
    if _TAX_ID_HAS_SPACE.search(stripped):
        # A space in a tax ID is suspicious — ambiguous
        return 0.5
    if _TAX_ID_STRICT_RE.match(stripped):
        return 1.0
    if _TAX_ID_DIGITS_RE.match(stripped):
        return 1.0
    # Has letters or digits but doesn't fit known patterns
    return 0.5


# ── Registration IDs ──────────────────────────────────────────────────────────

# Loose: alphanumeric core, 5-15 chars, optional separators
_REG_ID_RE = re.compile(r"^[A-Z0-9][\w\-./\s]{3,14}$", re.IGNORECASE)
_REG_ID_GARBAGE_RE = re.compile(r"[^\w\s\-./]")


def _validate_registration_id(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 0.0
    if _REG_ID_GARBAGE_RE.search(stripped):
        return 0.5
    if _REG_ID_RE.match(stripped):
        return 1.0
    # Very long strings are probably not reg IDs
    if len(stripped) > 20:
        return 0.5
    return 0.5


# ── Monetary amounts ──────────────────────────────────────────────────────────

# Matches: optional currency prefix/suffix, digits with optional , . separators
_AMOUNT_RE = re.compile(
    r"^"
    r"(?:[$€£¥₹₽¢₩₪₦₴₺₱฿₸]|[A-Z]{3}\s*)?"  # optional leading currency
    r"[\d][\d,.\s]*"  # digits with separators
    r"(?:\s*(?:[A-Z]{3}|[$€£¥₹₽¢₩₪₦₴₺₱฿₸]))?"  # optional trailing currency
    r"$",
    re.IGNORECASE,
)
_AMOUNT_NON_NUMERIC_RE = re.compile(r"[^\d,.\s$€£¥₹₽¢₩₪₦₴₺₱฿₸A-Z%-]", re.IGNORECASE)


def _validate_amount(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 0.0
    if _AMOUNT_NON_NUMERIC_RE.search(stripped):
        return 0.0
    if _AMOUNT_RE.match(stripped):
        return 1.0
    return 0.0


# ── Tax / discount rates ──────────────────────────────────────────────────────

_RATE_RE = re.compile(r"^\d+(\.\d+)?\s*%?$")


def _validate_rate(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 0.0
    if _RATE_RE.match(stripped):
        try:
            val = float(stripped.rstrip("%").strip())
            if 0 <= val <= 100:
                return 1.0
        except ValueError:
            pass
    return 0.0


# ── Dates ─────────────────────────────────────────────────────────────────────

# List of common date regexes (fast path before dateutil).
_DATE_PATTERNS = [
    # ISO: 2024-01-15
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    # US: 01/15/2024 or 1/15/2024
    re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$"),
    # EU: 15.01.2024
    re.compile(r"^\d{1,2}\.\d{1,2}\.\d{2,4}$"),
    # DD-MM-YYYY
    re.compile(r"^\d{1,2}-\d{1,2}-\d{2,4}$"),
    # Text: "15 Jan 2024" or "January 15, 2024" or "15 January 2024"
    re.compile(
        r"^\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*,?\s*\d{2,4}$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4}$",
        re.IGNORECASE,
    ),
]

# strptime formats to try as fallback
_DATE_FMTS = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d.%m.%Y",
    "%d-%m-%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%d/%m/%y",
    "%m/%d/%y",
    "%d.%m.%y",
]


def _validate_date(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 0.0
    # Fast path: check regex patterns
    for pat in _DATE_PATTERNS:
        if pat.match(stripped):
            return 1.0
    # Slow path: try strptime formats
    for fmt in _DATE_FMTS:
        try:
            datetime.strptime(stripped, fmt)
            return 1.0
        except ValueError:
            continue
    return 0.0


# ── Currency codes ─────────────────────────────────────────────────────────────

# Common ISO 4217 3-letter codes — non-exhaustive but covers 95%+ of invoices
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
        "ARS",
        "COP",
        "PEN",
        "VND",
        "NGN",
        "KES",
        "GHS",
        "MAD",
        "QAR",
        "KWD",
        "BHD",
        "OMR",
        "JOD",
        "LBP",
        "DZD",
        "TND",
        "LYD",
        "ETB",
        "TZS",
        "UGX",
    }
)
_CURRENCY_SYMBOLS = frozenset(
    {"$", "€", "£", "¥", "₹", "₽", "¢", "₩", "₪", "₦", "₴", "₺", "₱", "฿", "₸"}
)
_CURRENCY_CODE_RE = re.compile(r"^[A-Z]{3}$")


def _validate_currency_code(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 0.0
    if stripped in _CURRENCY_SYMBOLS:
        return 1.0
    upper = stripped.upper()
    if upper in _CURRENCY_CODES:
        return 1.0
    # Unknown 3-letter uppercase code: ambiguous but possible (some rare currencies)
    if _CURRENCY_CODE_RE.match(upper):
        return 0.5
    return 0.0


# ── Email ─────────────────────────────────────────────────────────────────────

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")


def _validate_email(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 0.0
    return 1.0 if _EMAIL_RE.match(stripped) else 0.0


# ── Quantity ──────────────────────────────────────────────────────────────────

_QUANTITY_RE = re.compile(r"^-?\d+([.,]\d+)?$")


def _validate_quantity(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 0.0
    return 1.0 if _QUANTITY_RE.match(stripped) else 0.0


# ── Position (line item row number) ──────────────────────────────────────────

_POSITION_RE = re.compile(r"^\d+$")


def _validate_position(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 0.0
    return 1.0 if _POSITION_RE.match(stripped) else 0.0


# ── Dispatch table ────────────────────────────────────────────────────────────

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

_RATE_FIELDS = frozenset(
    {
        "tax_detail_rate",
        "line_item_tax_rate",
        "line_item_discount_rate",
    }
)

_DATE_FIELDS = frozenset(
    {
        "date_due",
        "date_issue",
        "line_item_date",
    }
)

_TAX_ID_FIELDS = frozenset(
    {
        "vendor_tax_id",
        "customer_tax_id",
    }
)

_REG_ID_FIELDS = frozenset(
    {
        "vendor_registration_id",
        "customer_registration_id",
    }
)

_CURRENCY_FIELDS = frozenset(
    {
        "currency_code_amount_due",
        "line_item_currency",
    }
)

_VALIDATORS: dict[str, callable] = {
    "iban": _validate_iban,
    "bic": _validate_bic,
    "vendor_email": _validate_email,
    "line_item_quantity": _validate_quantity,
    "line_item_position": _validate_position,
}

# Add grouped fields
for _ft in _AMOUNT_FIELDS:
    _VALIDATORS[_ft] = _validate_amount
for _ft in _RATE_FIELDS:
    _VALIDATORS[_ft] = _validate_rate
for _ft in _DATE_FIELDS:
    _VALIDATORS[_ft] = _validate_date
for _ft in _TAX_ID_FIELDS:
    _VALIDATORS[_ft] = _validate_tax_id
for _ft in _REG_ID_FIELDS:
    _VALIDATORS[_ft] = _validate_registration_id
for _ft in _CURRENCY_FIELDS:
    _VALIDATORS[_ft] = _validate_currency_code


# ── Public API ────────────────────────────────────────────────────────────────


def format_confidence(fieldtype: str, text: str) -> float:
    """Return format-match confidence for a predicted field value.

    Returns 1.0 if text matches expected format, 0.5 if ambiguous,
    0.0 if clearly wrong.

    Use as a score multiplier:
        field.score *= format_confidence(field.fieldtype, field.text)

    Unknown fieldtypes always return 1.0 (no penalty for free-text fields).
    """
    validator = _VALIDATORS.get(fieldtype)
    if validator is None:
        return 1.0
    try:
        result = validator(text)
        # Clamp to [0.0, 1.0] defensively
        return float(max(0.0, min(1.0, result)))
    except Exception:
        # Never crash the pipeline — default to no penalty
        return 1.0
