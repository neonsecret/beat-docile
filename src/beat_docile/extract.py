"""[ACTIVE] Claude Sonnet extractor — zero-shot + few-shot + targeted second pass.

Status: ACTIVE — used in current best (v2_ensemble). PRODUCTION PATH — edit carefully.
See KNOWLEDGE_BASE.md §3 for the architecture map.

Bbox format: [left, top, right, bottom] normalized [0,1] — NEVER pixel coords.
Ref: EVAL_SPEC §1 (field catalog), §2 (prediction format), §5 (gotchas),
§7 (snapped OCR coords mandatory for predictions).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
from collections.abc import Sequence

from docile.dataset import BBox, Field

from .data import PageContext, WordBox, iter_pages
from .llm_client import complete

_TEMPERATURE = float(os.environ.get("BD_TEMPERATURE", "1.0"))
_ALT_PROMPT = os.environ.get("BD_ALT_PROMPT", "0") == "1"
_USE_ORACLE_PREPASS = os.environ.get("BD_USE_ORACLE_PREPASS", "0") == "1"

# ── System prompt (cached — identical across all docs) ──────────────────────────────────────────

_SYSTEM = """You are a document information extraction assistant for the DocILE benchmark.

## KILE Field Types (36) — extract at most one per page unless noted:
account_num: Bank account number (e.g., account no, konto nr, compte no)
amount_due: Amount due for payment
amount_paid: Amount already paid
amount_total_gross: Total gross amount (with tax)
amount_total_net: Total net amount (before tax)
amount_total_tax: Total tax amount
bank_num: Bank routing/sort code (e.g., sort code, BLZ, routing number, ABA)
bic: BIC/SWIFT code (8-11 chars, format: AAAABBCCXXX)
currency_code_amount_due: Currency symbol/code for amount due
customer_billing_address: Customer billing address block (include ALL address lines; select word_ids for ALL words in the full address block)
customer_billing_name: Customer billing name
customer_delivery_address: Customer delivery address block (include ALL address lines; select word_ids for ALL words in the full address block)
customer_delivery_name: Customer delivery name
customer_id: Customer identifier
customer_order_id: Order ID issued by customer
customer_other_address: Other customer address (include ALL address lines; select word_ids for ALL words in the full address block)
customer_other_name: Other customer name
customer_registration_id: Customer company registration/business ID (e.g., reg no, KvK, HRB, IČO, RN)
customer_tax_id: Customer VAT/tax ID (e.g., VAT no, MwSt-IdNr, DIČ, TVA)
date_due: Payment due date
date_issue: Invoice issue date
document_id: Document/invoice number
iban: IBAN (starts with 2-letter country code, e.g., GB29NWBK...)
order_id: Order identifier
payment_reference: Payment reference string
payment_terms: Payment terms text
tax_detail_gross: Per-tax-rate gross amount (multiple allowed, one per tax rate row)
tax_detail_net: Per-tax-rate net amount (multiple allowed, one per tax rate row)
tax_detail_rate: Tax rate percentage (multiple allowed, one per tax rate row)
tax_detail_tax: Per-tax-rate tax amount (multiple allowed, one per tax rate row)
vendor_address: Vendor address block (include ALL address lines; select word_ids for ALL words in the full address block)
vendor_email: Vendor email
vendor_name: Vendor name
vendor_order_id: Order ID issued by vendor
vendor_registration_id: Vendor company registration/business ID (e.g., reg no, KvK, HRB, IČO, RN)
vendor_tax_id: Vendor VAT/tax ID (e.g., VAT no, MwSt-IdNr, DIČ, TVA)

## LIR Field Types (19) — one set per line item row:
line_item_amount_gross: Line item gross amount
line_item_amount_net: Line item net amount
line_item_code: Product/SKU code
line_item_currency: Currency for this line item
line_item_date: Date on line item
line_item_description: Product/service description
line_item_discount_amount: Discount amount
line_item_discount_rate: Discount rate
line_item_hts_number: Harmonized tariff schedule number
line_item_order_id: Order ID for this line item
line_item_person_name: Person name on line item
line_item_position: Row/position number
line_item_quantity: Quantity
line_item_tax: Tax amount
line_item_tax_rate: Tax rate
line_item_unit_price_gross: Unit price gross
line_item_unit_price_net: Unit price net
line_item_units_of_measure: Units of measure
line_item_weight: Weight

## Output Format (JSON only, no markdown fences):
{
  "fields": [
    {"fieldtype": "<KILE field type>", "word_ids": [<id>, ...], "text": "<exact text>", "score": <0.0-1.0>}
  ],
  "line_items": [
    {
      "line_item_id": <int, 1-based, same id = same row>,
      "fields": [
        {"fieldtype": "<LIR field type>", "word_ids": [<id>, ...], "text": "<exact text>", "score": <0.0-1.0>}
      ]
    }
  ]
}
Note: "text" is the exact verbatim text of the field value as it appears on the document.

Rules:
- word_ids must reference ids from the provided word list (format "id:text")
- Include ONLY the words that ARE the field value — not surrounding labels or colons
  Example: for "Invoice No: 12345", word_ids should point to "12345" only, not "Invoice No:"
- For address blocks, include all address words but not the label "Bill To:" or "Address:"
- score: your confidence 0.0-1.0 for this specific extraction
- line_item_id groups fields belonging to the same table row (same id = same row)
- Only output fields you are confident about; omit fields not clearly present
- Return valid JSON only, no explanation, no markdown
- For address fields (customer_billing_address, vendor_address, customer_delivery_address, etc.): include ALL words across all address lines — street, city, postal code, country. Do not miss continuation lines.
- For tax_detail_* fields: extract one set per tax rate. If there are 3 tax rates, you must output 3 separate tax_detail_rate entries (and corresponding tax_detail_net, tax_detail_gross, tax_detail_tax entries for each row).
- MUTUAL EXCLUSION: customer_billing_address, customer_delivery_address, customer_other_address, vendor_address must each have DISJOINT word_id sets. A word that appears in one address field must NOT appear in another.
- MUTUAL EXCLUSION: amount_total_net, amount_total_gross, amount_total_tax must each have DISJOINT word_id sets. They are different values on the document.
- CONTIGUOUS WORDS: Every field's word_ids must reference words that form a single visually cohesive block (contiguous reading order, no gaps spanning multiple unrelated words). If a value appears in multiple disconnected places (e.g., reprinted at top and bottom), pick the most prominent occurrence.
"""

if os.environ.get("BD_USE_FIELD_INSTRUCTIONS", "0") == "1":
    from .field_instructions import ALL_FIELD_GUIDANCE

    _SYSTEM += (
        "\n\n## Per-Field Extraction Guide:\n"
        "IMPORTANT: amount_total_net and amount_total_gross CAN BOTH appear on the same "
        "invoice — they are always different numeric values. Extract BOTH when present.\n"
        "Similarly, tax_detail_net and tax_detail_gross are per-row values that always coexist.\n\n"
        + ALL_FIELD_GUIDANCE
    )


def _words_to_prompt(words: list[WordBox]) -> str:
    """Group words into visual rows (similar top-y) for spatial grounding.

    Format: "Row{i}(y≈{top:.2f}): {id}:{text}  {id}:{text} ..."
    This gives Claude row context so it can navigate layout without
    confusing adjacent-row words as part of the same field value.
    """
    if not words:
        return ""

    row_gap = 0.012  # ~2% page height; words within this are same row
    rows: list[list[WordBox]] = []
    current_row: list[WordBox] = []
    sorted_words = sorted(words, key=lambda w: (w.bbox[1], w.bbox[0]))

    for w in sorted_words:
        if not current_row or abs(w.bbox[1] - current_row[0].bbox[1]) <= row_gap:
            current_row.append(w)
        else:
            rows.append(sorted(current_row, key=lambda x: x.bbox[0]))
            current_row = [w]
    if current_row:
        rows.append(sorted(current_row, key=lambda x: x.bbox[0]))

    lines = []
    for i, row in enumerate(rows):
        row_y = row[0].bbox[1]
        tokens = "  ".join(f"{w.id}:{w.text}" for w in row)
        lines.append(f"R{i}(y≈{row_y:.3f}): {tokens}")
    return "\n".join(lines)


def _format_oracle_hints(matches: list) -> str:
    """Format OracleMatch list as a hint block for the Claude prompt."""
    matches = [m for m in matches if m.score == 1.0]  # strict: checksum/label-confirmed only
    if not matches:
        return ""
    lines = [
        "ORACLE CANDIDATES (high-confidence regex/checksum matches; "
        "verify and use only if correct):"
    ]
    for m in matches:
        if m.fieldtype == "iban":
            reason = "mod-97 valid"
        elif m.score == 1.0:
            reason = "label-confirmed"
        else:
            reason = "regex match, verify"
        lines.append(f'  - {m.fieldtype}: word_ids {m.word_ids} → "{m.text}" ({reason})')
    return "\n".join(lines)


def _image_to_b64(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def _merge_bboxes(word_ids: list[int], words: list[WordBox]) -> BBox | None:
    """Bbox enclosing all referenced words (simple min/max over valid ids)."""
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


_KILE_TYPES = {
    "account_num",
    "amount_due",
    "amount_paid",
    "amount_total_gross",
    "amount_total_net",
    "amount_total_tax",
    "bank_num",
    "bic",
    "currency_code_amount_due",
    "customer_billing_address",
    "customer_billing_name",
    "customer_delivery_address",
    "customer_delivery_name",
    "customer_id",
    "customer_order_id",
    "customer_other_address",
    "customer_other_name",
    "customer_registration_id",
    "customer_tax_id",
    "date_due",
    "date_issue",
    "document_id",
    "iban",
    "order_id",
    "payment_reference",
    "payment_terms",
    "tax_detail_gross",
    "tax_detail_net",
    "tax_detail_rate",
    "tax_detail_tax",
    "vendor_address",
    "vendor_email",
    "vendor_name",
    "vendor_order_id",
    "vendor_registration_id",
    "vendor_tax_id",
}

_LIR_TYPES = {
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


def _parse_response(
    raw: str, words: list[WordBox], page_idx: int
) -> tuple[list[Field], list[Field]]:
    """Parse Claude JSON → (kile_fields, lir_fields). Returns empty lists on parse error."""
    # Strip markdown fences if Claude adds them despite instructions
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [], []

    kile: list[Field] = []
    lir: list[Field] = []

    use_refiner = os.environ.get("BD_USE_REFINER", "1") == "1"
    use_validator = os.environ.get("BD_USE_VALIDATOR", "1") == "1"
    use_bbox_verify = os.environ.get("BD_USE_BBOX_VERIFY", "0") == "1"

    from .refiners import refine_field
    from .validators import format_confidence

    def _resolve_bbox(word_ids: list[int], value_text: str, fieldtype: str) -> BBox | None:
        """Refine word_ids per field-type rules → bbox; fall back to text alignment."""
        refined_word_ids = word_ids
        if use_refiner:
            refined_word_ids, bbox = refine_field(fieldtype, word_ids, words, value_text)
            if bbox is not None:
                if use_bbox_verify:
                    from .bbox_verify import verify_bbox
                    from .llm_client import get_client

                    verification = verify_bbox(
                        fieldtype, refined_word_ids, value_text, words, get_client()
                    )
                    return verification.bbox
                return bbox
        # Fallback path: simple merge in case refiner returns None or disabled
        bbox = _merge_bboxes(refined_word_ids, words)
        if bbox is not None:
            if use_bbox_verify:
                from .bbox_verify import verify_bbox
                from .llm_client import get_client

                verification = verify_bbox(
                    fieldtype, refined_word_ids, value_text, words, get_client()
                )
                return verification.bbox
            return bbox
        if value_text:
            from .align import find_span

            span = find_span(value_text, words, min_ratio=0.7)
            if span:
                sw = words[span[0] : span[1] + 1]
                return BBox(
                    min(w.bbox[0] for w in sw),
                    min(w.bbox[1] for w in sw),
                    max(w.bbox[2] for w in sw),
                    max(w.bbox[3] for w in sw),
                )
        return None

    for item in data.get("fields", []):
        ft = item.get("fieldtype", "")
        if ft not in _KILE_TYPES:
            continue
        word_ids = item.get("word_ids", [])
        value_text = str(item.get("text", ""))
        bbox = _resolve_bbox(word_ids, value_text, ft)
        if bbox is None:
            continue
        score = float(item.get("score", 0.8))
        if use_validator:
            score *= format_confidence(ft, value_text)
        kile.append(Field(bbox=bbox, page=page_idx, fieldtype=ft, score=score))

    for li in data.get("line_items", []):
        li_id = int(li.get("line_item_id", 0))
        for item in li.get("fields", []):
            ft = item.get("fieldtype", "")
            if ft not in _LIR_TYPES:
                continue
            word_ids = item.get("word_ids", [])
            value_text = str(item.get("text", ""))
            bbox = _resolve_bbox(word_ids, value_text, ft)
            if bbox is None:
                continue
            score = float(item.get("score", 0.8))
            if use_validator:
                score *= format_confidence(ft, value_text)
            lir.append(
                Field(bbox=bbox, page=page_idx, fieldtype=ft, line_item_id=li_id, score=score)
            )

    return kile, lir


def _bbox_overlap(b1: BBox, b2: BBox, threshold: float = 0.3) -> bool:
    """True if two bboxes overlap by at least threshold fraction of the smaller one."""
    ix1, iy1 = max(b1.left, b2.left), max(b1.top, b2.top)
    ix2, iy2 = min(b1.right, b2.right), min(b1.bottom, b2.bottom)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return False
    area1 = (b1.right - b1.left) * (b1.bottom - b1.top)
    area2 = (b2.right - b2.left) * (b2.bottom - b2.top)
    min_area = min(area1, area2)
    return (inter_area / min_area) >= threshold if min_area > 0 else False


def _vote_predictions(
    runs: list[tuple[list[Field], list[Field]]],
    min_votes: int = 2,
) -> tuple[list[Field], list[Field]]:
    """Vote across N extraction runs. Keep predictions with >= min_votes agreeing runs.

    Agreement = same fieldtype AND overlapping bbox (>=30% overlap by smaller).
    Score of surviving prediction = (agreeing_run_count / total_runs).
    For multi-occurrence fields (tax_detail_*, line items), match greedily.
    """
    n = len(runs)
    all_kile = [r[0] for r in runs]
    all_lir = [r[1] for r in runs]

    def vote_list(per_run: list[list[Field]]) -> list[Field]:
        # Collect all predictions from run 0 as candidates
        if not per_run or not per_run[0]:
            return []
        candidates = list(per_run[0])
        kept = []
        for cand in candidates:
            votes = 1
            for run_i in range(1, n):
                for f in per_run[run_i]:
                    if f.fieldtype == cand.fieldtype and _bbox_overlap(cand.bbox, f.bbox):
                        votes += 1
                        break
            if votes >= min_votes:
                # Use highest score among agreeing runs, but cap by vote fraction
                best_score = max(
                    f.score
                    for run in per_run
                    for f in run
                    if f.fieldtype == cand.fieldtype and _bbox_overlap(cand.bbox, f.bbox)
                )
                kept.append(
                    Field(
                        bbox=cand.bbox,
                        page=cand.page,
                        fieldtype=cand.fieldtype,
                        score=min(votes / n, best_score),
                        line_item_id=cand.line_item_id,
                        text=cand.text,
                    )
                )
        return kept

    return vote_list(all_kile), vote_list(all_lir)


async def extract_page_sc(
    page: PageContext,
    model: str,
    n_samples: int = 3,
    few_shot_messages: list[dict] | None = None,
) -> tuple[list[Field], list[Field]]:
    """Self-consistency: run extract_page n_samples times, vote on results."""
    tasks = [
        extract_page(page, model, few_shot_messages=few_shot_messages) for _ in range(n_samples)
    ]
    runs = list(await asyncio.gather(*tasks))
    return _vote_predictions(runs, min_votes=max(1, (n_samples + 1) // 2))


async def extract_page(
    page: PageContext,
    model: str,
    few_shot_messages: list[dict] | None = None,
) -> tuple[list[Field], list[Field]]:
    """Extract KILE + LIR fields from one page. Returns (kile_fields, lir_fields).

    If few_shot_messages is provided, they are prepended before the query user message.
    The messages must alternate user→assistant→...→user (query last).
    """
    if not page.words:
        return [], []

    img_b64 = _image_to_b64(page.image)
    words_layout = _words_to_prompt(page.words)

    oracle_prefix = ""
    if _USE_ORACLE_PREPASS:
        from .oracle_extract import oracle_extract_doc

        page_matches = oracle_extract_doc({page.page_index: page.words})
        oracle_prefix = _format_oracle_hints(page_matches)

    task_text = (
        "For each field present in this document, select the word_ids that "
        "exactly cover the field value — no more, no less. Return JSON only."
        if _ALT_PROMPT
        else "Extract all fields from this invoice page. "
        "Use the word ids shown above. Return JSON only."
    )
    user_text_parts = []
    if oracle_prefix:
        user_text_parts.append(oracle_prefix)
    user_text_parts.append(f"[Document words]\n{words_layout}")
    user_text_parts.append(f"[Task]\n{task_text}")

    user_content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": img_b64,
            },
        },
        {
            "type": "text",
            "text": "\n\n".join(user_text_parts),
        },
    ]

    messages = []
    if few_shot_messages:
        messages.extend(few_shot_messages)
    messages.append({"role": "user", "content": user_content})

    msg = await complete(
        model=model,
        system=_SYSTEM,
        messages=messages,
        max_tokens=4096,
        cache_system=True,
        temperature=_TEMPERATURE,
    )

    raw = msg.content[0].text if msg.content else ""
    return _parse_response(raw, page.words, page.page_index)


# ── Targeted second pass for financial / registration fields ──────────────────────────────────────

_TARGETED_FIELDS = {
    "account_num",
    "bank_num",
    "bic",
    "iban",
    "vendor_registration_id",
    "customer_registration_id",
    "vendor_tax_id",
    "customer_tax_id",
    "tax_detail_gross",
    "tax_detail_net",
    "tax_detail_rate",
    "tax_detail_tax",
}

_SYSTEM_TARGETED = """You extract ONLY financial and registration fields from invoices.

Fields to look for (ignore all others):
account_num: Bank account number (label: account no, konto, compte, conto, číslo účtu)
bank_num: Bank routing/sort code (label: sort code, routing no, BLZ, ABA, bankleitzahl)
bic: BIC/SWIFT code — 8-11 uppercase letters (e.g., DEUTDEDB, NWBKGB2L)
iban: IBAN — starts with 2-letter country code then digits (e.g., GB29NWBK..., DE89370...)
vendor_registration_id: Vendor company reg number (label: reg no, KvK, HRB, IČO, RN, ABN)
customer_registration_id: Customer company reg number (same label formats as vendor)
vendor_tax_id: Vendor VAT/tax ID (label: VAT no, MwSt-IdNr, DIČ, TVA, NIF, ΑΦΜ, BTW)
customer_tax_id: Customer VAT/tax ID (same formats)
tax_detail_gross: Gross amount for one tax rate (one per rate row)
tax_detail_net: Net amount for one tax rate (one per rate row)
tax_detail_rate: Tax rate % (e.g., 21%, 19%, 0%) — one per rate row
tax_detail_tax: Tax amount for one rate row

Output (JSON, no fences):
{"fields":[{"fieldtype":"...","word_ids":[...],"score":0.9}],"line_items":[]}

Rules:
- word_ids must be ids from the provided word list
- Include ONLY the exact value words — not surrounding labels or colons
- tax_detail_* fields: one set per rate row (2 rates = 2 tax_detail_rate entries etc.)
- If none present: {"fields":[],"line_items":[]}
- Bank/payment details are often in a separate block at bottom of invoice
"""


async def extract_page_targeted(page: PageContext, model: str) -> list[Field]:
    """Second-pass extraction hunting only for financial/registration fields."""
    if not page.words:
        return []

    img_b64 = _image_to_b64(page.image)
    words_layout = _words_to_prompt(page.words)
    user_content = [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": img_b64},
        },
        {
            "type": "text",
            "text": (
                f"[Document words]\n{words_layout}\n\n"
                "[Task]\n"
                "Look carefully for any banking, registration, or tax-rate detail fields. "
                "Return JSON only."
            ),
        },
    ]

    msg = await complete(
        model=model,
        system=_SYSTEM_TARGETED,
        messages=[{"role": "user", "content": user_content}],
        max_tokens=1024,
        cache_system=True,
    )

    raw = msg.content[0].text if msg.content else ""
    kile, lir = _parse_response(raw, page.words, page.page_index)
    # Only keep fields in the targeted set
    return [f for f in kile + lir if f.fieldtype in _TARGETED_FIELDS]


async def extract_documents(
    docs: Sequence,
    model: str,
    train_index: dict[int, list[str]] | None = None,
    targeted_pass: bool = True,
    self_consistency: bool = False,
    cluster_override: dict[str, int] | None = None,
) -> tuple[dict[str, list[Field]], dict[str, list[Field]]]:
    """Extract fields for a sequence of Document objects.

    Returns (kile_preds, lir_preds) both as {docid: [Field, ...]}

    If train_index is provided, cluster-based few-shot examples are loaded.
    If targeted_pass is True, a second focused pass extracts financial/registration fields.
    If self_consistency is True, each page is extracted 3 times and predictions are voted.
    If cluster_override is provided ({docid: cluster_id}), it takes precedence over
    doc.annotation.cluster_id — use this for test docs that have no annotated cluster_id.
    """
    from .fewshot import build_few_shot_messages, load_few_shot_examples

    def _cluster_id(doc) -> int | None:
        if cluster_override is not None and doc.docid in cluster_override:
            return cluster_override[doc.docid]
        try:
            return doc.annotation.cluster_id
        except Exception:
            return None

    kile_preds: dict[str, list[Field]] = {}
    lir_preds: dict[str, list[Field]] = {}

    # Build few-shot cache keyed by cluster_id if train_index is provided
    few_shot_cache: dict[int, list[dict]] = {}
    if train_index is not None:
        cluster_ids = []
        for doc in docs:
            cid = _cluster_id(doc)
            if cid is not None:
                cluster_ids.append(cid)
        unique_cluster_ids = list(set(cluster_ids))
        if unique_cluster_ids:
            examples_by_cluster = load_few_shot_examples(
                unique_cluster_ids, train_index, max_per_cluster=1
            )
            for cid, examples in examples_by_cluster.items():
                few_shot_cache[cid] = build_few_shot_messages(examples)

    async def process_doc(doc) -> None:
        kile_preds[doc.docid] = []
        lir_preds[doc.docid] = []

        fs_messages: list[dict] | None = None
        if train_index is not None:
            cid = _cluster_id(doc)
            if cid is not None and cid in few_shot_cache:
                fs_messages = few_shot_cache[cid]

        pages = list(iter_pages(doc))

        # Main pass (self-consistency or standard)
        if self_consistency:
            main_tasks = [
                extract_page_sc(page, model, few_shot_messages=fs_messages) for page in pages
            ]
        else:
            main_tasks = [
                extract_page(page, model, few_shot_messages=fs_messages) for page in pages
            ]
        # Targeted pass (concurrent with nothing extra — runs after main to avoid burst)
        targeted_tasks = (
            [extract_page_targeted(page, model) for page in pages] if targeted_pass else []
        )

        all_results = await asyncio.gather(*main_tasks, *targeted_tasks)
        n = len(pages)
        main_results = all_results[:n]
        targeted_results = all_results[n:] if targeted_pass else []

        for kile, lir in main_results:
            kile_preds[doc.docid].extend(kile)
            lir_preds[doc.docid].extend(lir)

        # Merge targeted pass — add only fields in _TARGETED_FIELDS not already covered
        for fields in targeted_results:
            for f in fields:
                if f.line_item_id is not None:
                    lir_preds[doc.docid].append(f)
                else:
                    kile_preds[doc.docid].append(f)

    await asyncio.gather(*[process_doc(doc) for doc in docs])
    return kile_preds, lir_preds
