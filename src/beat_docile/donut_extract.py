"""[RESEARCH-BURIED] Qwen3-VL-2B-Instruct wrapper for DocILE invoice field extraction.

Status: RESEARCH-BURIED — 0.29%-2.8% KILE on 50d. OCR-free generative models cannot
align outputs to DocILE's snapped OCR word grid; PCC-IoU=1.0 requires selecting from
existing DocTR words, not generating text. See KNOWLEDGE_BASE.md §6.14 for root cause.

Preserved as reference for OCR-free VLM approaches and the text-then-align failure mode.
Module name is historical (originally named for Donut-style extraction architecture).

Output contract: extract_donut_doc / extract_donut_images return
  {
    "kile": {fieldtype: str | list[str]},   # KILE predictions
    "line_items": [{fieldtype: str, ...}],  # LIR — one dict per row
  }
The eval driver (tools/run_donut_eval.py) handles alignment and Field conversion.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

_KILE_FIELDS: list[str] = [
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
]

_LIR_FIELDS: list[str] = [
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
]

_EXTRACTION_PROMPT = (
    "Extract fields from this invoice image. "
    "CRITICAL RULES:\n"
    "- Copy text EXACTLY as it appears in the image — do NOT guess, invent, or repeat digits\n"
    "- OMIT any field that is not clearly visible in the image\n"
    "- Keep each value short (under 200 characters)\n\n"
    "Return ONLY a JSON object (no markdown, no explanation) with exactly two keys:\n\n"
    '1. "kile": an object mapping field names to their text values (strings). '
    "Only include fields whose text is explicitly visible. "
    "For repeating fields (tax_detail_rate, tax_detail_gross, tax_detail_net, tax_detail_tax) "
    "use arrays of strings.\n"
    "Valid kile fields: " + ", ".join(_KILE_FIELDS) + "\n\n"
    '2. "line_items": an array of objects, one per invoice line item row. '
    "Each object maps field names to string values for that row.\n"
    "Valid line item fields: " + ", ".join(_LIR_FIELDS) + "\n\n"
    "Example:\n"
    '{"kile": {"vendor_name": "Acme Ltd", "document_id": "INV-001", '
    '"date_issue": "2024-01-15", "amount_total_gross": "1210.00", '
    '"tax_detail_rate": ["21%"], "vendor_address": "123 Main St, Prague"}, '
    '"line_items": [{"line_item_position": "1", "line_item_description": "Widget", '
    '"line_item_quantity": "2", "line_item_unit_price_net": "500.00", '
    '"line_item_amount_gross": "1210.00"}]}'
)


def load_model(
    model_id: str = MODEL_ID,
    device: str = "cuda",
) -> tuple[Any, Any]:
    """Load Qwen3-VL model and processor in BF16. Returns (model, processor)."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


def _extract_images_from_messages(messages: list[dict]) -> list[Any]:
    """Pull PIL images out of a messages list (replacement for qwen_vl_utils)."""
    images = []
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    images.append(item["image"])
    return images


def _run_single_page(
    image: Image.Image,
    model: Any,
    processor: Any,
    device: str = "cuda",
    max_new_tokens: int = 2048,
) -> str:
    """Run Qwen2-VL on a single page image. Returns raw text output."""
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": _EXTRACTION_PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Try qwen_vl_utils first; fall back to manual extraction
    try:
        from qwen_vl_utils import process_vision_info  # type: ignore[import]

        image_inputs, video_inputs = process_vision_info(messages)
    except ImportError:
        image_inputs = _extract_images_from_messages(messages)
        video_inputs = None

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy — prevents digit-repetition loops
            repetition_penalty=1.15,  # extra guard against token loops
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def _parse_json_output(raw: str) -> dict[str, Any]:
    """Extract JSON from model output, handling markdown code fences."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if match:
        raw = match.group(1)

    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # Fallback: find the outermost JSON object
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {}


def _normalize_kile(raw: dict) -> dict[str, str | list[str]]:
    """Filter to known KILE fields and coerce values to str / list[str]."""
    result: dict[str, str | list[str]] = {}
    for ft in _KILE_FIELDS:
        val = raw.get(ft)
        if val is None:
            continue
        if isinstance(val, list):
            strs = [str(v).strip() for v in val if str(v).strip()]
            if strs:
                result[ft] = strs
        else:
            s = str(val).strip()
            if s:
                result[ft] = s
    return result


def _normalize_line_items(raw: list) -> list[dict[str, str]]:
    """Filter each line item dict to known LIR fields, coerce to str."""
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        clean = {}
        for ft in _LIR_FIELDS:
            val = item.get(ft)
            if val is not None:
                s = str(val).strip()
                if s:
                    clean[ft] = s
        if clean:
            out.append(clean)
    return out


def extract_donut_images(
    images: list[Image.Image],
    model: Any,
    processor: Any,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run Qwen2-VL on a sequence of page images (one document).

    Returns {
        "kile": {fieldtype: str | list[str]},
        "line_items": [{fieldtype: str, ...}],  # one dict per LIR row
        "raw_outputs": [str, ...],              # raw model output per page
    }
    """
    all_kile: dict[str, str | list[str]] = {}
    all_line_items: list[dict[str, str]] = []
    raw_outputs: list[str] = []

    for image in images:
        raw = _run_single_page(image, model, processor, device)
        raw_outputs.append(raw)
        parsed = _parse_json_output(raw)

        kile = _normalize_kile(parsed.get("kile", {}))
        line_items = _normalize_line_items(parsed.get("line_items", []))

        # KILE: first-page values take priority; accumulate array fields
        for ft, val in kile.items():
            if ft not in all_kile:
                all_kile[ft] = val
            elif isinstance(val, list):
                existing = all_kile[ft]
                if isinstance(existing, list):
                    all_kile[ft] = existing + val
                else:
                    all_kile[ft] = [existing, *val]

        all_line_items.extend(line_items)

    return {
        "kile": all_kile,
        "line_items": all_line_items,
        "raw_outputs": raw_outputs,
    }


def extract_donut_doc(
    pdf_path: Path,
    model: Any,
    processor: Any,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run pre-trained doc-VLM on a single PDF page (or first page if multi-page).

    Returns {fieldtype: text} dict — convert to docile fields downstream.
    Concretely returns the richer structure from extract_donut_images with
    "kile" and "line_items" keys; callers use the full dict for alignment.
    """
    import fitz  # PyMuPDF — transitive dep of docile-benchmark

    doc = fitz.open(str(pdf_path))
    images: list[Image.Image] = []
    mat = fitz.Matrix(150 / 72, 150 / 72)  # 150 DPI, consistent with data.py
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    doc.close()

    return extract_donut_images(images, model, processor, device)
