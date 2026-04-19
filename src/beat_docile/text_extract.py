"""[ARCHIVED] Text-only Claude extraction — no word_ids in the output.

Status: ARCHIVED — 30-34% KILE across three alignment-fix iterations (vs 44.6% v2).
See KNOWLEDGE_BASE.md §6.11. PCC-IoU=1.0 requires spatial grounding; text→align
is structurally inferior to direct word_id selection. Kept for code-archaeology.

Original design: Claude extracts WHAT (semantic accuracy); precise_align.py finds
WHERE via character-level alignment. Decoupled semantic/spatial to fix V5b word_id
hallucination — but structural inferior to direct word_id selection.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from docile.dataset import Field

from .data import PageContext, iter_pages
from .extract import _words_to_prompt  # row-grouped layout formatter
from .vertex import complete

log = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────
# Same field catalog as V5b but output format has no word_ids.
# Address fields: multi-line output with \n.

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
customer_billing_address: Customer billing address block
customer_billing_name: Customer billing name
customer_delivery_address: Customer delivery address block
customer_delivery_name: Customer delivery name
customer_id: Customer identifier
customer_order_id: Order ID issued by customer
customer_other_address: Other customer address
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
vendor_address: Vendor address block
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
    {"fieldtype": "<KILE field type>", "text": "<exact verbatim text>", "score": <0.0-1.0>}
  ],
  "line_items": [
    {
      "line_item_id": <int, 1-based, same id = same row>,
      "fields": [
        {"fieldtype": "<LIR field type>", "text": "<exact verbatim text>", "score": <0.0-1.0>}
      ]
    }
  ]
}

Rules:
- "text" is the EXACT VERBATIM text as it appears on the document — copy characters exactly
- Do NOT normalize or translate: copy "1.234,56" not "1,234.56", copy "15.01.2024" not "Jan 15"
- For address fields: join all address lines with \\n (e.g., "Vendor GmbH\\nStreet 1\\n12345 City")
  Include ALL lines: company name, street, city, postal code, country. Do NOT include label "Bill To:"
- For IBAN: copy with exact spacing as shown (e.g., "DE89 3704 0044 0532 0130 00")
- For tax_detail_* fields: one set per tax rate row (2 rates = 2 entries each)
- MUTUAL EXCLUSION: address fields must each refer to DIFFERENT addresses on the document
- MUTUAL EXCLUSION: amount_total_net, amount_total_gross, amount_total_tax are separate values
- score: your confidence 0.0-1.0 for this specific extraction
- Only output fields you are confident about; omit fields not clearly present
- Return valid JSON only, no explanation, no markdown"""

# ─────────────────────────────────────────────────────────────────────────────
# Field type sets
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Extracted field dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ExtractedTextField:
    """A single field extracted by Claude (text-only, no spatial info yet)."""

    fieldtype: str
    text: str
    score: float
    line_item_id: int | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Response parser
# ─────────────────────────────────────────────────────────────────────────────


def _parse_text_response(raw: str) -> list[ExtractedTextField]:
    """Parse Claude's text-only JSON response into ExtractedTextField list.

    Returns empty list on parse error — never raises.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        log.warning("text_extract: JSON parse error")
        return []

    results: list[ExtractedTextField] = []

    for item in data.get("fields", []):
        ft = item.get("fieldtype", "")
        if ft not in _KILE_TYPES:
            continue
        field_text = str(item.get("text", "")).strip()
        if not field_text:
            continue
        score = float(item.get("score", 0.8))
        results.append(ExtractedTextField(fieldtype=ft, text=field_text, score=score))

    for li in data.get("line_items", []):
        li_id = int(li.get("line_item_id", 0))
        for item in li.get("fields", []):
            ft = item.get("fieldtype", "")
            if ft not in _LIR_TYPES:
                continue
            field_text = str(item.get("text", "")).strip()
            if not field_text:
                continue
            score = float(item.get("score", 0.8))
            results.append(
                ExtractedTextField(fieldtype=ft, text=field_text, score=score, line_item_id=li_id)
            )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Image helper (same as extract.py)
# ─────────────────────────────────────────────────────────────────────────────


def _image_to_b64(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# Page-level extraction
# ─────────────────────────────────────────────────────────────────────────────


async def extract_page_text(
    page: PageContext,
    model: str,
    few_shot_messages: list[dict] | None = None,
) -> list[ExtractedTextField]:
    """Extract field text values from one page (no word_ids).

    Uses same image+words prompt layout as V5b extract_page(), but asks Claude
    to output only text values — no word_ids. Alignment is done separately.

    Returns list of ExtractedTextField (one per extracted field, possibly
    multiple entries for the same fieldtype, e.g. tax_detail_*).
    """
    if not page.words:
        return []

    img_b64 = _image_to_b64(page.image)
    words_layout = _words_to_prompt(page.words)

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
            "text": (
                f"[Document words]\n{words_layout}\n\n"
                "[Task]\n"
                "Extract all field values from this invoice page. "
                "Output the exact verbatim text for each field. Return JSON only."
            ),
        },
    ]

    messages: list[dict] = []
    if few_shot_messages:
        messages.extend(few_shot_messages)
    messages.append({"role": "user", "content": user_content})

    msg = await complete(
        model=model,
        system=_SYSTEM,
        messages=messages,
        max_tokens=4096,
        cache_system=True,
    )

    raw = msg.content[0].text if msg.content else ""
    return _parse_text_response(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Page-level pipeline: extraction + alignment → Field objects
# ─────────────────────────────────────────────────────────────────────────────


async def extract_page_with_alignment(
    page: PageContext,
    model: str,
    few_shot_messages: list[dict] | None = None,
    min_align_confidence: float = 0.5,
) -> tuple[list[Field], list[Field]]:
    """Full text-align pipeline for one page: Claude extracts text, aligner maps to words.

    Returns (kile_fields, lir_fields) as docile Field objects with real bboxes.
    """
    from .precise_align import align_fields_to_words

    extracted = await extract_page_text(page, model, few_shot_messages=few_shot_messages)
    if not extracted:
        return [], []

    # Convert to dict list for align_fields_to_words
    items = [
        {
            "fieldtype": e.fieldtype,
            "text": e.text,
            "score": e.score,
            "line_item_id": e.line_item_id,
        }
        for e in extracted
    ]

    kile, lir = align_fields_to_words(
        items, page.words, page.page_index, min_confidence=min_align_confidence
    )
    return kile, lir


# ─────────────────────────────────────────────────────────────────────────────
# Document-level orchestration
# ─────────────────────────────────────────────────────────────────────────────


async def extract_documents_text(
    docs: Sequence,
    model: str,
    train_index: dict[int, list[str]] | None = None,
    cluster_override: dict[str, int] | None = None,
    min_align_confidence: float = 0.5,
    _few_shot_cache: dict[int, list[dict]] | None = None,
) -> tuple[dict[str, list[Field]], dict[str, list[Field]]]:
    """Extract fields for a sequence of documents using text-only Claude + alignment.

    Returns (kile_preds, lir_preds) both as {docid: [Field, ...]}.

    Args:
        docs: Sequence of Document objects.
        model: Claude model ID.
        train_index: {cluster_id: [docid, ...]} from fewshot._build_cluster_index.
        cluster_override: {docid: cluster_id} override for test docs without cluster_id.
        min_align_confidence: Skip alignments below this threshold.
        _few_shot_cache: Pre-built {cluster_id: few_shot_messages} — if provided,
            skips the internal load_few_shot_examples call. Use to avoid repeated
            train-dataset loading when calling this function in a loop.
    """
    from .fewshot import build_few_shot_messages, load_few_shot_examples

    def _cluster_id(doc) -> int | None:
        if cluster_override and doc.docid in cluster_override:
            return cluster_override[doc.docid]
        try:
            return doc.annotation.cluster_id
        except Exception:
            return None

    kile_preds: dict[str, list[Field]] = {}
    lir_preds: dict[str, list[Field]] = {}

    # Use caller-provided cache or build from train_index
    if _few_shot_cache is not None:
        few_shot_cache = _few_shot_cache
    else:
        few_shot_cache: dict[int, list[dict]] = {}
        if train_index is not None:
            unique_cluster_ids = list(
                {_cluster_id(doc) for doc in docs if _cluster_id(doc) is not None}
            )
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

        tasks = [
            extract_page_with_alignment(
                page,
                model,
                few_shot_messages=fs_messages,
                min_align_confidence=min_align_confidence,
            )
            for page in pages
        ]
        results = await asyncio.gather(*tasks)

        for kile, lir in results:
            kile_preds[doc.docid].extend(kile)
            lir_preds[doc.docid].extend(lir)

    await asyncio.gather(*[process_doc(doc) for doc in docs])
    return kile_preds, lir_preds
