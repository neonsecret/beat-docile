"""[ARCHIVED] Per-field ReAct extraction loop using Claude's native tool-use API.

Status: ARCHIVED — part of V6 pipeline (22.7% KILE). See KNOWLEDGE_BASE.md §6.7.
Triage gate semantics were delete-skewed; cross-field verifier deleted whole arrays.
Kept for code-archaeology only.

Original design: triage → per-field ReAct → cross-field verify pipeline producing
(kile_fields, lir_fields) tuple. Both gates proved delete-skewed in practice.
"""

from __future__ import annotations

import base64
import dataclasses
import io
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from docile.dataset import BBox, Field

from .data import WordBox
from .extract import _KILE_TYPES, _LIR_TYPES, _words_to_prompt
from .tools import (
    Candidate,
    cluster_fewshot,
    refine_span,
    regex_extract,
    spatial_neighbor,
    validator_check,
)

logger = logging.getLogger(__name__)

_MODEL_SONNET = "claude-sonnet-4-6"
_MODEL_HAIKU = "claude-haiku-4-5"

# ── System prompts ────────────────────────────────────────────────────────────

TRIAGE_SYSTEM = """You are analyzing an invoice document.
Look at the page image and word list, then determine which KILE field types are likely present.

Available KILE field types:
account_num, amount_due, amount_paid, amount_total_gross, amount_total_net, amount_total_tax,
bank_num, bic, currency_code_amount_due, customer_billing_address, customer_billing_name,
customer_delivery_address, customer_delivery_name, customer_id, customer_order_id,
customer_other_address, customer_other_name, customer_registration_id, customer_tax_id,
date_due, date_issue, document_id, iban, order_id, payment_reference, payment_terms,
tax_detail_gross, tax_detail_net, tax_detail_rate, tax_detail_tax,
vendor_address, vendor_email, vendor_name, vendor_order_id, vendor_registration_id, vendor_tax_id

Available LIR field types (only if the document has a line-items table):
line_item_amount_gross, line_item_amount_net, line_item_code, line_item_currency,
line_item_date, line_item_description, line_item_discount_amount, line_item_discount_rate,
line_item_hts_number, line_item_order_id, line_item_person_name, line_item_position,
line_item_quantity, line_item_tax, line_item_tax_rate, line_item_unit_price_gross,
line_item_unit_price_net, line_item_units_of_measure, line_item_weight

Return ONLY valid JSON, no markdown fences, no explanation:
{"present_fields": ["vendor_name", "date_issue", ...]}

Be inclusive — prefer to include a field that might be present over missing it.
"""

PER_FIELD_SYSTEM = """You are extracting the field "{fieldtype}" from an invoice document.

## Field description
{field_description}

## Workflow
Use the available tools to gather evidence, then emit your final answer.

Suggested steps:
1. Call regex_extract("{fieldtype}") to find pattern-matching candidates.
2. Call spatial_neighbor with common label phrases for this field.
3. Optionally call cluster_fewshot("{fieldtype}", cluster_id) to see similar examples.
4. For your best candidate(s), call validator_check("{fieldtype}", text) to confirm format.
5. Call refine_span to clean up the word_ids of your best candidate.
6. When confident, emit the final JSON answer.

## Final answer format
When done, output ONLY this JSON (no markdown, no explanation, no extra text):
{{"candidates": [{{"word_ids": [1, 2], "text": "...", "score": 0.95, "reason": "..."}}]}}

Use an empty list if the field is not present: {{"candidates": []}}
For multi-occurrence fields (tax_detail_*, line_item_*), return one candidate per occurrence.

For LIR (line_item_*) fields only: include "line_item_id" (1-based integer) in each candidate
to group fields belonging to the same table row. All fields in the same row share the same id.
Example for 3 line items:
{{"candidates": [
  {{"word_ids": [10], "text": "Widget A", "score": 0.9, "line_item_id": 1, "reason": "row 1"}},
  {{"word_ids": [20], "text": "Widget B", "score": 0.9, "line_item_id": 2, "reason": "row 2"}},
  {{"word_ids": [30], "text": "Widget C", "score": 0.9, "line_item_id": 3, "reason": "row 3"}}
]}}
"""

VERIFIER_SYSTEM = """You are verifying extracted invoice fields for consistency.

Check for the following problems:
1. MUTUAL EXCLUSION — addresses (billing/delivery/other/vendor) must have DISJOINT word_id sets.
2. MUTUAL EXCLUSION — amounts (amount_total_net, amount_total_gross, amount_total_tax) must be
   distinct values with different word_id sets.
3. DUPLICATE WORD_IDS — no two KILE fields should share word_ids unless they are intentionally
   at the same location (e.g., a currency code that is also part of an amount field).
4. SANITY — amount_total_gross >= amount_total_net when both are present (gross includes tax).

Return ONLY valid JSON listing corrections to apply (empty list if everything is fine):
{"corrections": [
  {"fieldtype": "amount_total_net", "action": "remove", "reason": "same word_ids as amount_total_gross"}
]}
Supported actions: "remove".
"""

# ── Field descriptions for the per-field system prompt ───────────────────────

_FIELD_DESCRIPTIONS: dict[str, str] = {
    "account_num": "Bank account number (label: account no, konto, compte, conto).",
    "amount_due": "Amount due for payment — numeric value, possibly with currency symbol.",
    "amount_paid": "Amount already paid — numeric value.",
    "amount_total_gross": "Total gross amount including tax — the largest total on the invoice.",
    "amount_total_net": "Total net amount before tax — smaller than gross.",
    "amount_total_tax": "Total tax amount — the difference between gross and net.",
    "bank_num": "Bank routing/sort code (label: sort code, BLZ, routing no, ABA).",
    "bic": "BIC/SWIFT code — 8 or 11 uppercase letters (e.g. DEUTDEDB, NWBKGB2LXXX).",
    "currency_code_amount_due": "Currency symbol or ISO code for the amount due (e.g. EUR, $, £).",
    "customer_billing_address": "Customer billing address block — all address lines.",
    "customer_billing_name": "Customer billing company or person name.",
    "customer_delivery_address": "Customer delivery/shipping address block — all address lines.",
    "customer_delivery_name": "Customer delivery name.",
    "customer_id": "Customer identifier assigned by the vendor.",
    "customer_order_id": "Order ID issued by the customer (purchase order number).",
    "customer_other_address": "Other customer address block not classified as billing or delivery.",
    "customer_other_name": "Other customer name (not billing or delivery).",
    "customer_registration_id": "Customer company registration ID (e.g. KvK, HRB, IČO, RN).",
    "customer_tax_id": "Customer VAT/tax ID (label: VAT no, MwSt-IdNr, DIČ, TVA, NIF).",
    "date_due": "Payment due date — date by which payment is expected.",
    "date_issue": "Invoice issue date — the date the invoice was created.",
    "document_id": "Document/invoice number — unique identifier for this invoice.",
    "iban": "IBAN — starts with 2-letter country code (e.g. GB29NWBK..., DE89370...).",
    "order_id": "Order identifier (not a PO number — the vendor's internal order number).",
    "payment_reference": "Payment reference string used when making the bank transfer.",
    "payment_terms": "Payment terms text (e.g. 'Net 30', '2/10 net 30').",
    "tax_detail_gross": "Gross amount for one specific tax rate row.",
    "tax_detail_net": "Net amount for one specific tax rate row.",
    "tax_detail_rate": "Tax rate percentage for one row (e.g. 21%, 0%, 19%).",
    "tax_detail_tax": "Tax amount for one specific tax rate row.",
    "vendor_address": "Vendor address block — all address lines.",
    "vendor_email": "Vendor email address.",
    "vendor_name": "Vendor company or person name (seller/issuer of the invoice).",
    "vendor_order_id": "Order ID issued by the vendor.",
    "vendor_registration_id": "Vendor company registration ID (e.g. KvK, HRB, IČO, RN, ABN).",
    "vendor_tax_id": "Vendor VAT/tax ID (label: VAT no, MwSt-IdNr, DIČ, TVA, NIF, BTW).",
    "line_item_amount_gross": "Line item gross amount (price including tax).",
    "line_item_amount_net": "Line item net amount (price before tax).",
    "line_item_code": "Product or SKU code for this line item.",
    "line_item_currency": "Currency for this specific line item.",
    "line_item_date": "Date associated with this line item.",
    "line_item_description": "Product or service description for this line item.",
    "line_item_discount_amount": "Discount amount applied to this line item.",
    "line_item_discount_rate": "Discount rate (percentage) for this line item.",
    "line_item_hts_number": "Harmonized Tariff Schedule number.",
    "line_item_order_id": "Order ID for this specific line item.",
    "line_item_person_name": "Person name associated with this line item.",
    "line_item_position": "Row or position number of this line item in the table.",
    "line_item_quantity": "Quantity of items in this row.",
    "line_item_tax": "Tax amount for this line item.",
    "line_item_tax_rate": "Tax rate for this line item.",
    "line_item_unit_price_gross": "Unit price gross (per item, including tax).",
    "line_item_unit_price_net": "Unit price net (per item, before tax).",
    "line_item_units_of_measure": "Unit of measure (e.g. pcs, kg, hrs).",
    "line_item_weight": "Weight of items in this row.",
}

# ── Tool schemas for Claude's native tool-use API ────────────────────────────

_TOOL_SCHEMAS: list[dict] = [
    {
        "name": "regex_extract",
        "description": (
            "Run a fieldtype-specific regex over all OCR words to find candidate spans. "
            "Returns a list of candidates with word_ids, text, and score. "
            "Use this first for fields with structured formats: IBAN, BIC, dates, amounts, "
            "email, currency codes, tax IDs, payment references."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fieldtype": {
                    "type": "string",
                    "description": "DocILE field type to run regex extraction for.",
                },
            },
            "required": ["fieldtype"],
        },
    },
    {
        "name": "validator_check",
        "description": (
            "Validate whether a text string matches the expected format for a fieldtype. "
            "Returns a confidence score 0.0-1.0. Use to confirm your chosen candidate "
            "before finalizing, especially for structured fields (IBAN, BIC, date, amount)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fieldtype": {
                    "type": "string",
                    "description": "DocILE field type.",
                },
                "text": {
                    "type": "string",
                    "description": "Candidate text value to validate.",
                },
            },
            "required": ["fieldtype", "text"],
        },
    },
    {
        "name": "spatial_neighbor",
        "description": (
            "Find words that appear directly to the right of or below common label keywords. "
            "Use for label-value fields like 'Invoice No: INV-001', 'Due Date: 15.01.2024'. "
            "Returns candidates with word_ids and confidence scores."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "label_phrases": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Label keyword phrases to anchor the search. "
                        "Case-insensitive substring match. "
                        "E.g. ['Invoice No', 'Inv #', 'Invoice Number']."
                    ),
                },
                "direction": {
                    "type": "string",
                    "enum": ["right_or_below", "right", "below"],
                    "description": "Direction to look for the value. Default: right_or_below.",
                },
                "max_distance_frac": {
                    "type": "number",
                    "description": (
                        "Maximum gap as a fraction of page dimension (0.0-1.0). Default: 0.20."
                    ),
                },
            },
            "required": ["label_phrases"],
        },
    },
    {
        "name": "cluster_fewshot",
        "description": (
            "Retrieve few-shot examples from training documents with the same invoice template "
            "(cluster_id). Shows what the field value looks like in matching templates. "
            "Useful when you are unsure what format the field uses in this document."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fieldtype": {
                    "type": "string",
                    "description": "DocILE field type to retrieve examples for.",
                },
                "cluster_id": {
                    "type": ["integer", "null"],
                    "description": "Cluster ID from the current document. Pass null if unknown.",
                },
            },
            "required": ["fieldtype", "cluster_id"],
        },
    },
    {
        "name": "refine_span",
        "description": (
            "Clean up a candidate word span using field-type-specific rules. "
            "Strips label prefixes (e.g. 'Invoice No:'), enforces row contiguity, "
            "and removes non-value words. Call this last, after selecting word_ids, "
            "to get the final refined span."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fieldtype": {
                    "type": "string",
                    "description": "DocILE field type.",
                },
                "word_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Word IDs of the raw candidate span to refine.",
                },
                "text": {
                    "type": "string",
                    "description": "Text of the candidate span (hint for refinement).",
                },
            },
            "required": ["fieldtype", "word_ids", "text"],
        },
    },
    {
        "name": "classifier_score",
        "description": (
            "Score how likely a candidate span is the requested fieldtype, using a "
            "per-fieldtype binary classifier trained on 2000 docs. Returns p(is_field) "
            "in [0,1]. Use to validate uncertain candidates from regex/spatial."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "word_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Word IDs of the candidate span to score.",
                },
            },
            "required": ["word_ids"],
        },
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _image_to_b64(image: Any) -> str:
    """Encode a PIL image to base64 PNG string."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def _execute_tool(
    name: str,
    args: dict,
    words: list[WordBox],
    cluster_id: int | None,
    train_docs: dict,
    fieldtype: str = "",
    model_dir: Path | None = None,
) -> Any:
    """Dispatch a tool call to the appropriate Python function.

    Args:
        name: Tool name as called by Claude.
        args: Tool arguments dict from Claude's tool_use block.
        words: Page OCR words (closure context for word-dependent tools).
        cluster_id: Document cluster ID for few-shot lookup.
        train_docs: Pre-loaded training data dict.
        fieldtype: Current field being extracted (used by classifier_score).
        model_dir: Path to trained classifier models (None → returns 0.5).

    Returns:
        JSON-serializable result to send back as tool_result content.

    Raises:
        ValueError: If name is not a known tool.
    """
    if name == "regex_extract":
        candidates = regex_extract(args["fieldtype"], words)
        return [dataclasses.asdict(c) for c in candidates]

    if name == "validator_check":
        result = validator_check(args["fieldtype"], args["text"])
        return dataclasses.asdict(result) if result is not None else None

    if name == "spatial_neighbor":
        candidates = spatial_neighbor(
            label_phrases=args["label_phrases"],
            words=words,
            direction=args.get("direction", "right_or_below"),
            max_distance_frac=float(args.get("max_distance_frac", 0.20)),
        )
        return [dataclasses.asdict(c) for c in candidates]

    if name == "cluster_fewshot":
        return cluster_fewshot(
            fieldtype=args["fieldtype"],
            cluster_id=args.get("cluster_id"),
            train_docs=train_docs,
        )

    if name == "refine_span":
        candidate = refine_span(
            fieldtype=args["fieldtype"],
            word_ids=args.get("word_ids", []),
            words=words,
            text=args.get("text", ""),
        )
        return dataclasses.asdict(candidate)

    if name == "classifier_score":
        if model_dir is None:
            return 0.5
        from .tools import classifier_score_tool

        return classifier_score_tool(
            fieldtype=fieldtype,
            word_ids=[int(x) for x in args.get("word_ids", [])],
            words=words,
            page_w=1.0,
            page_h=1.0,
            model_dir=model_dir,
        )

    raise ValueError(f"Unknown tool: {name!r}")


def _serialize_content_block(block: Any) -> dict:
    """Convert an SDK content block object to a plain dict for re-submission."""
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    # Passthrough for any other block types (e.g. thinking)
    return {"type": block.type}


def _parse_candidates_from_response(response: Any) -> list[Candidate]:
    """Extract Candidate list from Claude's end-turn response.

    Looks for a JSON object with a "candidates" key in text blocks.
    Claude often prepends explanatory text before the JSON — this handles that
    by searching for the first {"candidates" occurrence when full-text parse fails.
    Returns an empty list on any parse error.
    """
    for block in response.content:
        if block.type != "text":
            continue
        raw = block.text.strip()
        # Strip markdown fences if Claude adds them
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw).strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Claude often prepends reasoning text — find the embedded JSON
            for prefix in ('{"candidates"', '{ "candidates"'):
                idx = raw.find(prefix)
                if idx != -1:
                    try:
                        data = json.loads(raw[idx:])
                        break
                    except json.JSONDecodeError:
                        pass
            else:
                continue
        if not isinstance(data, dict) or "candidates" not in data:
            continue
        results: list[Candidate] = []
        for item in data.get("candidates", []):
            if not isinstance(item, dict):
                continue
            try:
                raw_li_id = item.get("line_item_id")
                li_id = int(raw_li_id) if raw_li_id is not None else None
                results.append(
                    Candidate(
                        word_ids=[int(x) for x in item.get("word_ids", [])],
                        text=str(item.get("text", "")),
                        score=float(item.get("score", 0.5)),
                        source="react",
                        reason=str(item.get("reason", "")),
                        line_item_id=li_id,
                    )
                )
            except (TypeError, ValueError) as exc:
                logger.warning("Skipping malformed candidate: %s — %s", item, exc)
        return results
    return []


def _bbox_from_word_ids(word_ids: list[int], words: list[WordBox]) -> BBox | None:
    """Compute bounding box enclosing all word_ids via min/max."""
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


# ── Public API ────────────────────────────────────────────────────────────────


def triage_fields(
    words: list[WordBox],
    image: Any,
    vertex_client: Any,
) -> list[str]:
    """Single Claude call to determine which field types are likely present.

    Args:
        words: Page OCR words.
        image: PIL image of the page.
        vertex_client: AnthropicVertex client instance.

    Returns:
        List of fieldtype strings likely present. Falls back to all known
        field types on parse failure so nothing is silently skipped.
    """
    words_layout = _words_to_prompt(words)
    img_b64 = _image_to_b64(image)

    user_content = [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": img_b64},
        },
        {
            "type": "text",
            "text": (
                f"Words grouped by visual row:\n{words_layout}\n\n"
                "Which field types are present in this invoice? "
                "Return JSON only."
            ),
        },
    ]

    try:
        response = vertex_client.messages.create(
            model=_MODEL_HAIKU,
            system=[
                {"type": "text", "text": TRIAGE_SYSTEM, "cache_control": {"type": "ephemeral"}}
            ],
            messages=[{"role": "user", "content": user_content}],
            max_tokens=512,
        )
    except Exception as exc:
        logger.warning("Triage call failed: %s — returning all field types", exc)
        return list(_KILE_TYPES | _LIR_TYPES)

    for block in response.content:
        if block.type != "text":
            continue
        raw = block.text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw).strip()
        try:
            data = json.loads(raw)
            fields = data.get("present_fields", [])
            if isinstance(fields, list) and fields:
                return [f for f in fields if isinstance(f, str)]
        except (json.JSONDecodeError, AttributeError):
            pass

    logger.warning("Triage: could not parse response — returning all field types")
    return list(_KILE_TYPES | _LIR_TYPES)


def extract_field_react(
    fieldtype: str,
    words: list[WordBox],
    image: Any,
    cluster_id: int | None,
    train_docs: dict,
    vertex_client: Any,
    max_steps: int = 8,
    model_dir: Path | None = None,
) -> list[Candidate]:
    """Per-field ReAct loop using Claude's native tool-use API.

    Runs a multi-turn conversation where Claude reasons, calls tools, observes
    results, and emits a final list of Candidates for the given fieldtype.

    Args:
        fieldtype: DocILE field type to extract (e.g. "document_id").
        words: All OCR words on the page.
        image: PIL image of the page.
        cluster_id: Document cluster ID for few-shot retrieval.
        train_docs: Pre-loaded training data dict for cluster_fewshot.
        vertex_client: AnthropicVertex client instance.
        max_steps: Hard cap on tool-use iterations. Returns partial results if hit.

    Returns:
        List of Candidates — may be empty if field not found. Multiple candidates
        for multi-occurrence fields (e.g. tax_detail_*, line_item_*).
    """
    field_desc = _FIELD_DESCRIPTIONS.get(fieldtype, f"DocILE field: {fieldtype}")
    system = PER_FIELD_SYSTEM.format(
        fieldtype=fieldtype,
        field_description=field_desc,
    )
    system_param = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]

    words_layout = _words_to_prompt(words)
    img_b64 = _image_to_b64(image)

    initial_user = {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": img_b64},
            },
            {
                "type": "text",
                "text": (
                    f"Words grouped by visual row:\n{words_layout}\n\n"
                    f"Extract field: {fieldtype}\n"
                    f"cluster_id={cluster_id}\n\n"
                    "Use tools to gather evidence, then emit your final JSON answer."
                ),
            },
        ],
    }

    messages: list[dict] = [initial_user]
    partial_candidates: list[Candidate] = []

    for step in range(max_steps):
        try:
            response = vertex_client.messages.create(
                model=_MODEL_SONNET,
                system=system_param,
                messages=messages,
                tools=_TOOL_SCHEMAS,
                max_tokens=2048,
            )
        except Exception as exc:
            logger.error("API call failed at step %d for %s: %s", step, fieldtype, exc)
            break

        if response.stop_reason == "end_turn":
            partial_candidates = _parse_candidates_from_response(response)
            break

        if response.stop_reason == "tool_use":
            # Serialize assistant message for re-submission
            content_dicts = [_serialize_content_block(b) for b in response.content]
            messages.append({"role": "assistant", "content": content_dicts})

            # Execute each tool call and collect results
            tool_results: list[dict] = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                try:
                    result = _execute_tool(
                        block.name,
                        block.input,
                        words,
                        cluster_id,
                        train_docs,
                        fieldtype=fieldtype,
                        model_dir=model_dir,
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        }
                    )
                except Exception as exc:
                    logger.warning("Tool %r failed: %s", block.name, exc)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Error executing tool: {exc}",
                            "is_error": True,
                        }
                    )

            messages.append({"role": "user", "content": tool_results})
            continue

        # Any other stop_reason (e.g. max_tokens) — try to parse what we have
        logger.warning(
            "Unexpected stop_reason=%r for %s at step %d",
            response.stop_reason,
            fieldtype,
            step,
        )
        partial_candidates = _parse_candidates_from_response(response)
        break

    return partial_candidates


def verify_extractions(
    extractions: dict[str, list[Candidate]],
    words: list[WordBox],
    image: Any,
    vertex_client: Any,
) -> dict[str, list[Candidate]]:
    """Final pass: single Claude call to cross-check extracted fields.

    Checks for mutual exclusion violations (shared word_ids across address or
    amount fields) and logical inconsistencies (gross < net). Applies corrections
    by removing flagged candidates.

    Args:
        extractions: {fieldtype: [Candidate, ...]} from extract_field_react calls.
        words: Page OCR words (for context).
        image: PIL image.
        vertex_client: AnthropicVertex client instance.

    Returns:
        Updated extractions dict with problematic candidates removed.
    """
    if not extractions:
        return extractions

    # Build a compact summary for Claude
    summary: dict[str, list[dict]] = {}
    for ft, candidates in extractions.items():
        if candidates:
            summary[ft] = [
                {"word_ids": c.word_ids, "text": c.text, "score": c.score} for c in candidates
            ]

    img_b64 = _image_to_b64(image)
    user_content = [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": img_b64},
        },
        {
            "type": "text",
            "text": (
                "Extracted fields:\n"
                + json.dumps(summary, indent=2)
                + "\n\nCross-check for violations and return corrections JSON only."
            ),
        },
    ]

    try:
        response = vertex_client.messages.create(
            model=_MODEL_HAIKU,
            system=[
                {"type": "text", "text": VERIFIER_SYSTEM, "cache_control": {"type": "ephemeral"}}
            ],
            messages=[{"role": "user", "content": user_content}],
            max_tokens=1024,
        )
    except Exception as exc:
        logger.warning("Verifier call failed: %s — returning extractions unchanged", exc)
        return extractions

    for block in response.content:
        if block.type != "text":
            continue
        raw = block.text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw).strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        corrections = data.get("corrections", [])
        if not isinstance(corrections, list):
            continue
        result = {ft: list(cs) for ft, cs in extractions.items()}
        for correction in corrections:
            ft = correction.get("fieldtype")
            action = correction.get("action")
            if action == "remove" and ft in result:
                logger.info("Verifier removing %s: %s", ft, correction.get("reason", ""))
                result[ft] = []
        return result

    return extractions


def extract_page_react(
    words: list[WordBox],
    image: Any,
    cluster_id: int | None,
    train_docs: dict,
    vertex_client: Any,
    candidate_verifier: Callable[[str, list[Candidate]], list[Candidate]] | None = None,
    model_dir: Path | None = None,
    max_steps_per_field: int = 6,
) -> tuple[list[Field], list[Field]]:
    """Single-page entry: triage → per-field ReAct → (optional verify) → cross-field verify → Fields.

    Single-page only. For multi-page documents call this per page via data.iter_pages
    and concatenate results — page indices are read from WordBox.page.

    The optional candidate_verifier callback is the integration seam for haiku_verify.py
    (bbox-precision verification). It is called per fieldtype with raw Candidates
    before bbox resolution and cross-field checking. Signature:
        (fieldtype: str, candidates: list[Candidate]) -> list[Candidate]

    Args:
        words: Snapped OCR words for ONE page (from data.iter_pages).
        image: PIL image of the same page.
        cluster_id: Document cluster ID for few-shot retrieval.
        train_docs: Pre-loaded training data dict {docid: {fields, words, cluster_id}}.
        vertex_client: AnthropicVertex client instance.
        candidate_verifier: Optional per-field candidate filter/corrector (Phase B seam).

    Returns:
        (kile_fields, lir_fields) as lists of docile.dataset.Field with resolved bboxes.
    """
    page_index = words[0].page if words else 0

    # Step 1: triage (Haiku)
    present_fieldtypes = triage_fields(words, image, vertex_client)
    logger.info(
        "Triage found %d field types: %s",
        len(present_fieldtypes),
        ", ".join(sorted(present_fieldtypes)[:10]),
    )

    # Step 2: per-field ReAct loops (Sonnet)
    extractions: dict[str, list[Candidate]] = {}
    for fieldtype in present_fieldtypes:
        if fieldtype not in _KILE_TYPES and fieldtype not in _LIR_TYPES:
            continue
        logger.info("Extracting field: %s", fieldtype)
        candidates = extract_field_react(
            fieldtype=fieldtype,
            words=words,
            image=image,
            cluster_id=cluster_id,
            train_docs=train_docs,
            vertex_client=vertex_client,
            max_steps=max_steps_per_field,
            model_dir=model_dir,
        )
        # Step 2b: optional bbox-precision verifier (haiku_verify seam)
        if candidates and candidate_verifier is not None:
            candidates = candidate_verifier(fieldtype, candidates)
        if candidates:
            extractions[fieldtype] = candidates

    # Step 3: cross-field verification (Haiku)
    if extractions:
        extractions = verify_extractions(extractions, words, image, vertex_client)

    # Step 4: convert Candidates → Field objects; use Claude-assigned line_item_id for LIR
    kile_fields: list[Field] = []
    lir_fields: list[Field] = []

    for fieldtype, candidates in extractions.items():
        is_lir = fieldtype in _LIR_TYPES
        for li_idx, candidate in enumerate(candidates):
            bbox = _bbox_from_word_ids(candidate.word_ids, words)
            if bbox is None:
                continue
            if is_lir:
                # Prefer Claude's assigned line_item_id; fall back to position order
                li_id = candidate.line_item_id if candidate.line_item_id is not None else li_idx + 1
                lir_fields.append(
                    Field(
                        bbox=bbox,
                        page=page_index,
                        fieldtype=fieldtype,
                        score=candidate.score,
                        line_item_id=li_id,
                        text=candidate.text or None,
                    )
                )
            else:
                kile_fields.append(
                    Field(
                        bbox=bbox,
                        page=page_index,
                        fieldtype=fieldtype,
                        score=candidate.score,
                        text=candidate.text or None,
                    )
                )

    logger.info(
        "extract_page_react: %d KILE + %d LIR fields",
        len(kile_fields),
        len(lir_fields),
    )
    return kile_fields, lir_fields
