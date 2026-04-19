"""[RESEARCH-BURIED] Per-field-type deterministic span refiners for DocILE.

Status: RESEARCH-BURIED — net negative at 500-doc scale across every
configuration tested. See KNOWLEDGE_BASE.md §6.2 for details.

Helps ~6 fields (address blocks, vendor names), hurts ~6 others (exact-value
codes, standard amounts). Selective-refiner gating (§8.8) is the remaining
opportunity. Guard mode (BD_USE_REFINER_GUARD=1) reduces but does not
eliminate the regression.
"""

from __future__ import annotations

import itertools
import os
import re

from docile.dataset import BBox

from .data import WordBox

# Guard mode: only allow refiner to remove words that are in the field's label set.
# Blocks aggressive spatial tightening that nukes legitimate multi-word fields.
_USE_GUARD = os.environ.get("BD_USE_REFINER_GUARD", "0") == "1"

# ── Label prefixes to strip ───────────────────────────────────────────────────

_ADDRESS_LABELS = {
    "bill",
    "ship",
    "to:",
    "address:",
    "from:",
    "vendor:",
    "customer:",
    "billing",
    "delivery",
    "sold",
    "invoice",
    "attn:",
    "attention:",
    "recipient:",
    "consignee:",
    "shipper:",
    "sender:",
}

_NAME_LABELS = {
    "vendor:",
    "customer:",
    "bill",
    "ship",
    "sold",
    "name:",
    "client:",
    "to:",
    "from:",
    "buyer:",
    "seller:",
}

_ID_LABELS = {
    "invoice",
    "invoice:",
    "no.",
    "no:",
    "number:",
    "#",
    "id:",
    "order:",
    "ref:",
    "reference:",
    "document:",
    "doc:",
    "po:",
    "p.o.:",
    "contract:",
    "code:",
    "num:",
    "nr:",
    "nr",
    "no",
    "number",
    "document",
    "order",
}

_FINANCIAL_LABELS = {
    "account:",
    "acct:",
    "account",
    "acct",
    "bank:",
    "bank",
    "bic:",
    "bic",
    "swift:",
    "swift",
    "iban:",
    "iban",
    "routing:",
    "sort",
    "sort:",
    "code:",
    "tax:",
    "vat:",
    "vat",
    "tax",
    "id:",
    "no:",
    "number:",
    "reg:",
    "reg",
    "registration:",
    "kvk:",
    "hrb:",
    "iňo:",
    "ico:",
    "abn:",
    "nif:",
    "btw:",
}

_AMOUNT_LABELS = {
    "total:",
    "total",
    "amount:",
    "amount",
    "due:",
    "due",
    "paid:",
    "paid",
    "gross:",
    "gross",
    "net:",
    "net",
    "tax:",
    "tax",
    "subtotal:",
    "subtotal",
    "balance:",
    "balance",
    "sum:",
    "sum",
}

_DATE_LABELS = {
    "date:",
    "date",
    "due:",
    "due",
    "issued:",
    "issued",
    "invoice",
    "payment",
    "valid:",
    "expiry:",
    "expires:",
    "from:",
    "to:",
    "period:",
}

# ── Currency symbols / codes ──────────────────────────────────────────────────

_CURRENCY_RE = re.compile(
    r"^[\$€£¥₹₩₽฿₺₴₪]$|^(USD|EUR|GBP|CHF|JPY|CAD|AUD|NZD|SEK|NOK|DKK|"
    r"PLN|CZK|HUF|RON|BGN|HRK|RUB|UAH|TRY|CNY|INR|BRL|MXN|SGD|HKD|KRW|"
    r"ZAR|AED|SAR|THB|MYR|IDR|PHP|VND|kr|Kč|zł|Ft|lei|kn)$",
    re.IGNORECASE,
)

# ── Numeric pattern helpers ───────────────────────────────────────────────────

_DIGIT_RE = re.compile(r"\d")
_AMOUNT_RE = re.compile(r"^[\d,.\s]+$")
_ID_LIKE_RE = re.compile(r"[A-Za-z0-9]")


# ─────────────────────────────────────────────────────────────────────────────
# Core utility helpers
# ─────────────────────────────────────────────────────────────────────────────


def _to_rows(
    word_ids: list[int],
    words: list[WordBox],
    row_tol: float = 0.012,
) -> list[list[int]]:
    """Group word_ids into visual rows by top-y proximity.

    Returns list of rows (each row is a list of word_ids), sorted top-to-bottom,
    within each row sorted left-to-right.
    """
    id_to_word = {w.id: w for w in words}
    valid = [wid for wid in word_ids if wid in id_to_word]
    if not valid:
        return []

    # Sort all words by top-y then left-x
    sorted_ids = sorted(valid, key=lambda wid: (id_to_word[wid].bbox[1], id_to_word[wid].bbox[0]))

    rows: list[list[int]] = []
    current_row: list[int] = []
    current_top: float | None = None

    for wid in sorted_ids:
        top = id_to_word[wid].bbox[1]
        if current_top is None or abs(top - current_top) <= row_tol:
            current_row.append(wid)
            if current_top is None:
                current_top = top
        else:
            rows.append(current_row)
            current_row = [wid]
            current_top = top

    if current_row:
        rows.append(current_row)

    return rows


def _largest_contiguous_run(word_ids: list[int]) -> list[int]:
    """Return longest run of consecutive integers in word_ids (preserving order).

    E.g. [1,2,5,6,7,9] → [5,6,7].
    """
    if not word_ids:
        return []
    sorted_ids = sorted(set(word_ids))
    best_run: list[int] = []
    current_run = [sorted_ids[0]]
    for prev, curr in itertools.pairwise(sorted_ids):
        if curr == prev + 1:
            current_run.append(curr)
        else:
            if len(current_run) > len(best_run):
                best_run = current_run
            current_run = [curr]
    if len(current_run) > len(best_run):
        best_run = current_run
    # Preserve original ordering (not necessarily sorted)
    best_set = set(best_run)
    return [wid for wid in word_ids if wid in best_set]


def _strip_label_prefix(
    word_ids: list[int],
    words: list[WordBox],
    label_words: set[str],
) -> list[int]:
    """Drop leading word_ids whose text matches any label_words (case-insensitive).

    Also strips a lone trailing colon word if it follows a stripped label.
    """
    id_to_word = {w.id: w for w in words}
    result = list(word_ids)
    while result:
        wid = result[0]
        if wid not in id_to_word:
            result.pop(0)
            continue
        tok = id_to_word[wid].text.lower().strip()
        if tok in label_words or tok.rstrip(":") in label_words:
            result.pop(0)
        else:
            break
    return result


def _bbox_from_words(word_ids: list[int], words: list[WordBox]) -> BBox | None:
    """Compute tight bbox via min/max over selected word bboxes."""
    id_to_word = {w.id: w for w in words}
    bboxes = [id_to_word[wid].bbox for wid in word_ids if wid in id_to_word]
    if not bboxes:
        return None
    return BBox(
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    )


def _row_top(row: list[int], id_to_word: dict[int, WordBox]) -> float:
    """Average top-y of all words in a row."""
    return sum(id_to_word[wid].bbox[1] for wid in row) / len(row)


def _rows_are_consecutive(
    rows: list[list[int]], id_to_word: dict[int, WordBox], max_gap: float = 0.04
) -> list[bool]:
    """For each adjacent pair of rows, True if gap between them is < max_gap."""
    if len(rows) < 2:
        return []
    result = []
    for r1, r2 in itertools.pairwise(rows):
        # bottom of r1 vs top of r2
        bottom1 = max(id_to_word[wid].bbox[3] for wid in r1)
        top2 = min(id_to_word[wid].bbox[1] for wid in r2)
        result.append((top2 - bottom1) < max_gap)
    return result


def _largest_consecutive_row_block(
    rows: list[list[int]],
    id_to_word: dict[int, WordBox],
    max_gap: float = 0.04,
) -> list[list[int]]:
    """Return the largest contiguous block of rows where each pair is visually adjacent."""
    if not rows:
        return []
    consec = _rows_are_consecutive(rows, id_to_word, max_gap)
    # Build blocks
    blocks: list[list[list[int]]] = []
    current_block = [rows[0]]
    for i, adjacent in enumerate(consec):
        if adjacent:
            current_block.append(rows[i + 1])
        else:
            blocks.append(current_block)
            current_block = [rows[i + 1]]
    blocks.append(current_block)
    # Return block with most words
    return max(blocks, key=lambda blk: sum(len(r) for r in blk))


# ─────────────────────────────────────────────────────────────────────────────
# Per-field-type refiners
# ─────────────────────────────────────────────────────────────────────────────


def _refine_address(
    word_ids: list[int],
    words: list[WordBox],
) -> tuple[list[int], BBox | None]:
    """Address: sort by (row, col), strip label prefix, keep largest contiguous row block."""
    if not word_ids:
        return word_ids, _bbox_from_words(word_ids, words)

    # Strip label prefix first
    word_ids = _strip_label_prefix(word_ids, words, _ADDRESS_LABELS)
    if not word_ids:
        return [], None

    id_to_word = {w.id: w for w in words}

    # Group into rows (tight row tolerance for addresses)
    rows = _to_rows(word_ids, words, row_tol=0.012)
    if not rows:
        return word_ids, _bbox_from_words(word_ids, words)

    # Strip label rows from the front (rows whose words are all label words)
    while rows:
        row_texts = [id_to_word[wid].text.lower().strip() for wid in rows[0] if wid in id_to_word]
        row_texts_clean = [t.rstrip(":") for t in row_texts]
        if all(
            t in _ADDRESS_LABELS or tc in _ADDRESS_LABELS
            for t, tc in zip(row_texts, row_texts_clean, strict=False)
        ):
            rows.pop(0)
        else:
            break

    if not rows:
        return word_ids, _bbox_from_words(word_ids, words)

    # Keep largest consecutive row block
    best_block = _largest_consecutive_row_block(rows, id_to_word, max_gap=0.04)
    refined_ids = [wid for row in best_block for wid in row]

    if not refined_ids:
        return word_ids, _bbox_from_words(word_ids, words)

    return refined_ids, _bbox_from_words(refined_ids, words)


def _refine_name(
    word_ids: list[int],
    words: list[WordBox],
) -> tuple[list[int], BBox | None]:
    """Name: almost always single row. Keep the row with the most words."""
    if not word_ids:
        return word_ids, _bbox_from_words(word_ids, words)

    word_ids = _strip_label_prefix(word_ids, words, _NAME_LABELS)
    if not word_ids:
        return [], None

    rows = _to_rows(word_ids, words, row_tol=0.012)
    if not rows:
        return word_ids, _bbox_from_words(word_ids, words)

    # Keep row with most words (the name line)
    best_row = max(rows, key=len)
    return best_row, _bbox_from_words(best_row, words)


def _refine_single_value(
    word_ids: list[int],
    words: list[WordBox],
) -> tuple[list[int], BBox | None]:
    """ID fields: 1-3 contiguous tokens, alphanumeric. Strip label prefix."""
    if not word_ids:
        return word_ids, _bbox_from_words(word_ids, words)

    word_ids = _strip_label_prefix(word_ids, words, _ID_LABELS)
    if not word_ids:
        return [], None

    id_to_word = {w.id: w for w in words}

    # Stay on single row
    rows = _to_rows(word_ids, words, row_tol=0.012)
    if rows:
        # Pick row with most alphanumeric-looking word with digits
        def row_score(row: list[int]) -> int:
            texts = [id_to_word[wid].text for wid in row if wid in id_to_word]
            # Prefer rows that contain digits
            has_digit = any(_DIGIT_RE.search(t) for t in texts)
            return (2 if has_digit else 0) + len(row)

        best_row = max(rows, key=row_score)
        word_ids = best_row

    # If > 5 tokens, keep only the best contiguous run
    if len(word_ids) > 5:
        run = _largest_contiguous_run(word_ids)
        if run:
            word_ids = run

    return word_ids, _bbox_from_words(word_ids, words)


def _refine_financial_code(
    word_ids: list[int],
    words: list[WordBox],
) -> tuple[list[int], BBox | None]:
    """Financial codes (IBAN, BIC, account_num, etc.): single row, contiguous run, strip labels."""
    if not word_ids:
        return word_ids, _bbox_from_words(word_ids, words)

    word_ids = _strip_label_prefix(word_ids, words, _FINANCIAL_LABELS)
    if not word_ids:
        return [], None

    # Keep only words on the same row as the first word
    rows = _to_rows(word_ids, words, row_tol=0.015)
    if rows:
        word_ids = rows[0]  # take first (topmost) row — that's where the value is

    return word_ids, _bbox_from_words(word_ids, words)


def _refine_amount(
    word_ids: list[int],
    words: list[WordBox],
) -> tuple[list[int], BBox | None]:
    """Amounts: usually 1-2 tokens. Strip labels, allow currency symbols at start."""
    if not word_ids:
        return word_ids, _bbox_from_words(word_ids, words)

    word_ids = _strip_label_prefix(word_ids, words, _AMOUNT_LABELS)
    if not word_ids:
        return [], None

    id_to_word = {w.id: w for w in words}

    # Single row only
    rows = _to_rows(word_ids, words, row_tol=0.012)
    if rows:
        # Pick row that has numeric content
        def amount_row_score(row: list[int]) -> int:
            texts = [id_to_word[wid].text for wid in row if wid in id_to_word]
            has_digit = any(_DIGIT_RE.search(t) for t in texts)
            return (3 if has_digit else 0) + len(row)

        word_ids = max(rows, key=amount_row_score)

    # If > 3 tokens, keep the currency symbol (if any) + numeric run
    if len(word_ids) > 3:
        # Find the numeric/currency subset
        kept: list[int] = []
        found_numeric = False
        for wid in word_ids:
            if wid not in id_to_word:
                continue
            tok = id_to_word[wid].text.strip()
            is_currency = bool(_CURRENCY_RE.match(tok))
            is_numeric = bool(_DIGIT_RE.search(tok)) and bool(
                _AMOUNT_RE.match(tok.replace(" ", ""))
            )
            if is_currency and not found_numeric:
                kept.append(wid)
            elif is_numeric:
                found_numeric = True
                kept.append(wid)
            elif is_currency and found_numeric:
                # Trailing currency symbol (e.g. "1,234.56 EUR") — keep it
                kept.append(wid)
                break
            elif found_numeric:
                # Non-currency, non-numeric after numeric content — stop
                break
        if kept:
            word_ids = kept

    return word_ids, _bbox_from_words(word_ids, words)


def _refine_date(
    word_ids: list[int],
    words: list[WordBox],
) -> tuple[list[int], BBox | None]:
    """Dates: 1-3 tokens, single row. Strip date label prefixes."""
    if not word_ids:
        return word_ids, _bbox_from_words(word_ids, words)

    word_ids = _strip_label_prefix(word_ids, words, _DATE_LABELS)
    if not word_ids:
        return [], None

    id_to_word = {w.id: w for w in words}

    # Single row: take the row with date-looking content
    rows = _to_rows(word_ids, words, row_tol=0.012)
    if rows:

        def date_row_score(row: list[int]) -> int:
            texts = [id_to_word[wid].text for wid in row if wid in id_to_word]
            # Dates contain digits and separators; prefer short rows with digits
            has_digit = any(_DIGIT_RE.search(t) for t in texts)
            return (3 if has_digit else 0) + (2 if len(row) <= 3 else 0) + len(row)

        word_ids = max(rows, key=date_row_score)

    # If > 3 tokens, keep longest contiguous run
    if len(word_ids) > 3:
        run = _largest_contiguous_run(word_ids)
        if run and len(run) <= 3:
            word_ids = run
        elif run:
            # Take up to 3 from the run
            word_ids = run[:3]

    return word_ids, _bbox_from_words(word_ids, words)


def _refine_line_item_field(
    fieldtype: str,
    word_ids: list[int],
    words: list[WordBox],
    text: str,
) -> tuple[list[int], BBox | None]:
    """LIR fields: generally single-row, contiguous. Delegate to KILE equivalents where possible."""
    # Map LIR subtypes to KILE-style refiners
    if any(x in fieldtype for x in ("amount", "price", "discount_amount", "tax")):
        return _refine_amount(word_ids, words)
    if "date" in fieldtype:
        return _refine_date(word_ids, words)
    if "name" in fieldtype or "person" in fieldtype:
        return _refine_name(word_ids, words)
    if any(x in fieldtype for x in ("code", "order_id", "position", "hts")):
        return _refine_single_value(word_ids, words)
    if any(x in fieldtype for x in ("rate", "quantity", "weight", "units")):
        return _refine_generic(word_ids, words)
    # Default: single row, contiguous
    return _refine_generic(word_ids, words)


def _refine_generic(
    word_ids: list[int],
    words: list[WordBox],
) -> tuple[list[int], BBox | None]:
    """Generic fallback: sort word_ids, find longest contiguous run, return its bbox."""
    if not word_ids:
        return word_ids, _bbox_from_words(word_ids, words)

    # Try keeping single row first
    rows = _to_rows(word_ids, words, row_tol=0.012)
    if len(rows) == 1:
        # Already single row — just take longest contiguous run
        run = _largest_contiguous_run(word_ids)
        if run and len(run) >= len(word_ids) // 2:
            return run, _bbox_from_words(run, words)
        return word_ids, _bbox_from_words(word_ids, words)

    # Multi-row: find longest contiguous run overall
    run = _largest_contiguous_run(word_ids)
    if run and len(run) >= 1:
        return run, _bbox_from_words(run, words)

    return word_ids, _bbox_from_words(word_ids, words)


# ─────────────────────────────────────────────────────────────────────────────
# Field type routing tables
# ─────────────────────────────────────────────────────────────────────────────

_ADDRESS_TYPES = frozenset(
    {
        "customer_billing_address",
        "customer_delivery_address",
        "customer_other_address",
        "vendor_address",
    }
)

_NAME_TYPES = frozenset(
    {
        "customer_billing_name",
        "customer_delivery_name",
        "vendor_name",
        "customer_other_name",
        "line_item_person_name",
    }
)

_SINGLE_VALUE_TYPES = frozenset(
    {
        "document_id",
        "order_id",
        "customer_id",
        "customer_order_id",
        "vendor_order_id",
        "customer_registration_id",
        "vendor_registration_id",
        "payment_reference",
        "line_item_code",
        "line_item_order_id",
        "line_item_position",
        "line_item_hts_number",
    }
)

_FINANCIAL_CODE_TYPES = frozenset(
    {
        "account_num",
        "bank_num",
        "bic",
        "iban",
        "vendor_tax_id",
        "customer_tax_id",
    }
)

_AMOUNT_TYPES = frozenset(
    {
        "amount_due",
        "amount_paid",
        "amount_total_gross",
        "amount_total_net",
        "amount_total_tax",
        "tax_detail_gross",
        "tax_detail_net",
        "tax_detail_tax",
        "tax_detail_rate",
        "currency_code_amount_due",
        "line_item_amount_gross",
        "line_item_amount_net",
        "line_item_discount_amount",
        "line_item_discount_rate",
        "line_item_tax",
        "line_item_tax_rate",
        "line_item_unit_price_gross",
        "line_item_unit_price_net",
        "line_item_quantity",
        "line_item_weight",
        "line_item_units_of_measure",
        "line_item_currency",
    }
)

_DATE_TYPES = frozenset(
    {
        "date_due",
        "date_issue",
        "line_item_date",
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# Guard mode — per-field label sets
# ─────────────────────────────────────────────────────────────────────────────

# Maps each field type to the set of label words that are safe to strip.
# Any other word removed by the refiner triggers the guard.
_FIELD_LABEL_SETS: dict[str, frozenset] = {}
for _ft in _ADDRESS_TYPES:
    _FIELD_LABEL_SETS[_ft] = _ADDRESS_LABELS
for _ft in _NAME_TYPES:
    _FIELD_LABEL_SETS[_ft] = _NAME_LABELS
for _ft in _AMOUNT_TYPES:
    _FIELD_LABEL_SETS[_ft] = _AMOUNT_LABELS
for _ft in _DATE_TYPES:
    _FIELD_LABEL_SETS[_ft] = _DATE_LABELS
for _ft in _SINGLE_VALUE_TYPES:
    _FIELD_LABEL_SETS[_ft] = _ID_LABELS
for _ft in _FINANCIAL_CODE_TYPES:
    _FIELD_LABEL_SETS[_ft] = _FINANCIAL_LABELS


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def refine_field(
    fieldtype: str,
    word_ids: list[int],
    words: list[WordBox],
    text: str = "",
) -> tuple[list[int], BBox | None]:
    """Return (refined_word_ids, refined_bbox). May return ([], None) to drop the prediction.

    Defensive: if the refiner cannot improve, returns the original word_ids unchanged.

    Args:
        fieldtype: DocILE field type string.
        word_ids: Indices into `words` as returned by Claude.
        words: Full page word list (list[WordBox] from data.py).
        text: Claude's extracted text, optional hint.

    Returns:
        (refined_word_ids, refined_bbox) — refined_bbox is computed from refined_word_ids.
    """
    if not word_ids:
        return [], None

    # Validate word_ids against the page word list
    valid_ids = {w.id for w in words}
    word_ids = [wid for wid in word_ids if wid in valid_ids]
    if not word_ids:
        return [], None

    original_word_ids = list(word_ids)

    try:
        if fieldtype in _ADDRESS_TYPES:
            refined_ids, bbox = _refine_address(word_ids, words)
        elif fieldtype in _NAME_TYPES:
            refined_ids, bbox = _refine_name(word_ids, words)
        elif fieldtype in _SINGLE_VALUE_TYPES:
            refined_ids, bbox = _refine_single_value(word_ids, words)
        elif fieldtype in _FINANCIAL_CODE_TYPES:
            refined_ids, bbox = _refine_financial_code(word_ids, words)
        elif fieldtype in _AMOUNT_TYPES:
            refined_ids, bbox = _refine_amount(word_ids, words)
        elif fieldtype in _DATE_TYPES:
            refined_ids, bbox = _refine_date(word_ids, words)
        elif fieldtype.startswith("line_item_"):
            refined_ids, bbox = _refine_line_item_field(fieldtype, word_ids, words, text)
        else:
            refined_ids, bbox = _refine_generic(word_ids, words)
    except Exception:
        # Never silently drop predictions due to refiner bugs
        return word_ids, _bbox_from_words(word_ids, words)

    # Defensive fallback: if refiner returned empty, keep original
    if not refined_ids:
        return word_ids, _bbox_from_words(word_ids, words)

    # ── Guard mode ─────────────────────────────────────────────────────────────
    # If BD_USE_REFINER_GUARD=1: when the refiner shrinks the span, verify that
    # ONLY known label words were removed. If any non-label content was dropped
    # (e.g. address continuation lines, trailing currency, multi-word name parts),
    # revert to a label-prefix-stripped version of the original instead.
    # Skip guard for line_item_* fields: they NEED row-isolation spatial tightening.
    if (
        _USE_GUARD
        and len(refined_ids) < len(original_word_ids)
        and not fieldtype.startswith("line_item_")
    ):
        id_to_word_g = {w.id: w for w in words}
        label_set = _FIELD_LABEL_SETS.get(fieldtype, frozenset())
        removed = [wid for wid in original_word_ids if wid not in set(refined_ids)]
        non_label_removed = [
            wid
            for wid in removed
            if wid in id_to_word_g
            and id_to_word_g[wid].text.lower().strip().rstrip(":") not in label_set
        ]
        if non_label_removed:
            # Guard triggered: fall back to just stripping the label prefix
            stripped = _strip_label_prefix(original_word_ids, words, label_set)
            if stripped:
                return stripped, _bbox_from_words(stripped, words)
            return original_word_ids, _bbox_from_words(original_word_ids, words)

    return refined_ids, bbox
