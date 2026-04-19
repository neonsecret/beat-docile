"""[EXPERIMENTAL] GutenOCR-3B grounded OCR integration for DocILE KILE extraction.

Status: EXPERIMENTAL — built; never properly evaluated at scale. See KNOWLEDGE_BASE.md §2.4.

Model: rootsautomation/GutenOCR-3B (Qwen2.5-VL-3B FT on 30M+ business pages, cc-by-nc-4.0)
License note: cc-by-nc-4.0 — research use only.
VRAM: ~4.5GB at 4-bit NF4 on CUDA; ~6.5GB BF16 on MPS.

Two integration modes:

  Mode A — replace:
    GutenOCR locates all KILE fields directly (same prompt as vlm_extract.py).
    Returns tight bboxes from a model trained for grounded OCR.
    PCC-snap those bboxes to Doctr words → Field objects.
    Use as a standalone replacement for the VLM extraction step.

  Mode B — augment:
    Run Mode A on the same page to get GutenOCR field proposals.
    For each existing (V5b/Code-Factory) KILE field, check whether GutenOCR
    proposed the same fieldtype with sufficient IoU overlap. If yes, replace
    the field's bbox with GutenOCR's tighter bbox (re-snapped). LIR fields
    are passed through unchanged.

Usage:
    # spike test (5 docs)
    uv run python tools/run_gutenocr_50.py --limit 5 --mode a
    # 50-doc eval
    uv run python tools/run_gutenocr_50.py --limit 50 --mode a
    uv run python tools/run_gutenocr_50.py --limit 50 --mode b
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from docile.dataset import BBox, Field

from .data import PageContext, WordBox

GUTENOCR_MODEL_ID = "rootsautomation/GutenOCR-3B"

KILE_FIELD_TYPES = [
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
    "seller_address",
    "seller_ico",
    "seller_name",
    "seller_registration_id",
    "seller_tax_id",
    "seller_vat_id",
    "tax_detail_gross",
    "tax_detail_net",
    "tax_detail_rate",
    "tax_detail_tax",
]

_LOCATE_PROMPT = (
    "Locate all KILE invoice fields present on this page. "
    "For each detected field return its exact bounding box and text. "
    'Output ONLY a JSON object: {"fieldtype": {"bbox_2d": [x1, y1, x2, y2], "text": "..."}}. '
    "Coordinates are in 0-1000 scale (0=top-left corner, 1000=bottom-right corner). "
    "Omit fields not present on the page. "
    f"Valid fieldtypes: {', '.join(KILE_FIELD_TYPES)}."
)

# Minimum IoU for Mode B to accept GutenOCR's bbox as "same field"
_MODE_B_IOU_THRESHOLD = 0.3


# ── Geometry helpers ──────────────────────────────────────────────────────────


def _pcc_in_bbox(word: WordBox, bbox: BBox) -> bool:
    cx = (word.bbox[0] + word.bbox[2]) / 2
    cy = (word.bbox[1] + word.bbox[3]) / 2
    return bbox.left <= cx <= bbox.right and bbox.top <= cy <= bbox.bottom


def _snap_to_words(pred_bbox: BBox, words: list[WordBox]) -> tuple[BBox, str] | None:
    """PCC-snap: find words whose pseudo-character center falls in pred_bbox."""
    matched = [w for w in words if _pcc_in_bbox(w, pred_bbox)]
    if not matched:
        return None
    snapped = BBox(
        min(w.bbox[0] for w in matched),
        min(w.bbox[1] for w in matched),
        max(w.bbox[2] for w in matched),
        max(w.bbox[3] for w in matched),
    )
    return snapped, " ".join(w.text for w in matched)


def _bbox_iou(a: BBox, b: BBox) -> float:
    inter_l = max(a.left, b.left)
    inter_t = max(a.top, b.top)
    inter_r = min(a.right, b.right)
    inter_b = min(a.bottom, b.bottom)
    if inter_r <= inter_l or inter_b <= inter_t:
        return 0.0
    inter = (inter_r - inter_l) * (inter_b - inter_t)
    a_area = (a.right - a.left) * (a.bottom - a.top)
    b_area = (b.right - b.left) * (b.bottom - b.top)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


# ── Output parser ─────────────────────────────────────────────────────────────


def _parse_and_snap(raw: str, words: list[WordBox], page: int) -> list[Field]:
    """Parse GutenOCR JSON output; PCC-snap each bbox to Doctr words."""
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []

    fields: list[Field] = []
    for fieldtype, val in data.items():
        if fieldtype not in KILE_FIELD_TYPES:
            continue
        if not isinstance(val, dict):
            continue
        bbox_2d = val.get("bbox_2d")
        if not bbox_2d or len(bbox_2d) != 4:
            continue
        try:
            x1, y1, x2, y2 = bbox_2d
            pred_bbox = BBox(x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0)
        except (TypeError, ValueError):
            continue

        snap = _snap_to_words(pred_bbox, words)
        if snap:
            bbox, text = snap
        else:
            bbox = pred_bbox
            text = val.get("text", "")

        fields.append(
            Field(
                bbox=bbox,
                page=page,
                score=1.0,
                text=text,
                fieldtype=fieldtype,
            )
        )

    return fields


# ── Model wrapper ─────────────────────────────────────────────────────────────


class GutenOCRExtractor:
    """Runs GutenOCR-3B for KILE field extraction.

    On CUDA: 4-bit NF4 quantization (bitsandbytes required) — ~4.5GB VRAM.
    On MPS/CPU: bfloat16 — ~6.5GB unified memory; no bitsandbytes needed.
    """

    def __init__(
        self,
        model_id: str = GUTENOCR_MODEL_ID,
        cache_dir: Path | None = None,
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self._device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Loading GutenOCR-3B ({model_id}) on {self._device}...")

        load_kw: dict = {}
        if cache_dir:
            load_kw["cache_dir"] = str(cache_dir)

        if self._device == "cuda":
            try:
                from transformers import BitsAndBytesConfig

                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                model_kw = {"quantization_config": bnb_cfg, "device_map": "cuda"}
            except ImportError:
                print("  bitsandbytes not found — falling back to BF16 on CUDA")
                model_kw = {"torch_dtype": torch.bfloat16, "device_map": "cuda"}
        else:
            model_kw = {"torch_dtype": torch.bfloat16, "device_map": self._device}

        self._processor = AutoProcessor.from_pretrained(model_id, **load_kw)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, **model_kw, **load_kw
        )
        self._model.eval()
        print("GutenOCR-3B loaded.")

    def _run_inference(self, image, prompt: str) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        image_inputs, _ = process_vision_info(messages)
        text_prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text_prompt],
            images=image_inputs,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        return self._processor.decode(generated, skip_special_tokens=True).strip()

    def extract_page_mode_a(self, page: PageContext) -> list[Field]:
        """Mode A: GutenOCR locates all KILE fields; PCC-snap to Doctr words."""
        raw = self._run_inference(page.image, _LOCATE_PROMPT)
        return _parse_and_snap(raw, page.words, page.page_index)

    def augment_page_mode_b(
        self,
        page: PageContext,
        existing_fields: list[Field],
    ) -> list[Field]:
        """Mode B: Replace V5b KILE field bboxes with GutenOCR's tighter ones.

        For each existing KILE field on this page, GutenOCR is checked for a
        prediction of the same fieldtype with IoU >= _MODE_B_IOU_THRESHOLD.
        If found, the GutenOCR bbox (re-snapped) replaces the original.
        LIR fields (line_item_id is not None) are passed through unchanged.
        """
        guten_fields = self.extract_page_mode_a(page)

        guten_by_type: dict[str, list[Field]] = {}
        for f in guten_fields:
            guten_by_type.setdefault(f.fieldtype, []).append(f)

        result: list[Field] = []
        for field in existing_fields:
            if (
                field.line_item_id is None
                and field.page == page.page_index
                and field.fieldtype in guten_by_type
            ):
                candidates = guten_by_type[field.fieldtype]
                best = max(candidates, key=lambda g: _bbox_iou(field.bbox, g.bbox))
                if _bbox_iou(field.bbox, best.bbox) >= _MODE_B_IOU_THRESHOLD:
                    result.append(best)
                    continue
            result.append(field)

        return result


# ── Document-level helpers ────────────────────────────────────────────────────


def extract_document_mode_a(
    extractor: GutenOCRExtractor,
    doc,
) -> list[Field]:
    """Run Mode A on all pages of a document."""
    from .data import iter_pages

    fields: list[Field] = []
    for page in iter_pages(doc):
        fields.extend(extractor.extract_page_mode_a(page))
    return fields


def augment_document_mode_b(
    extractor: GutenOCRExtractor,
    doc,
    existing_fields: list[Field],
) -> list[Field]:
    """Run Mode B on all pages of a document against existing predictions."""
    from .data import iter_pages

    result: list[Field] = []
    for page in iter_pages(doc):
        page_existing = [f for f in existing_fields if f.page == page.page_index]
        page_other = [f for f in existing_fields if f.page != page.page_index]
        result.extend(extractor.augment_page_mode_b(page, page_existing))
        result.extend(page_other)
    return result
