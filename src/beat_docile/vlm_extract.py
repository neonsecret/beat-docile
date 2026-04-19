"""[ACTIVE] Qwen3-VL-2B-Instruct PCC-snap extraction utilities for DocILE KILE fields.

Status: ACTIVE — utility module imported by qwen8b_extract.py and cli.py.
KILE_FIELD_TYPES, _PROMPT, and _parse_and_snap are the shared grounding primitives.

VLMExtractor wraps a merged (non-adapter) Qwen3-VL-2B checkpoint at ~/qwen3vl_docile/.
qwen3vl_extract.py has its own local copy of PCC-snap helpers for the LoRA-adapter path.

Usage:
    uv run bd vlm-extract --split val --limit 50
    uv run python -m beat_docile.vlm_extract --split val --limit 50
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from docile.dataset import BBox, Field

from .data import PageContext, WordBox, iter_pages

DEFAULT_MODEL_DIR = Path.home() / "qwen3vl_docile"

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

_PROMPT = (
    "Locate all KILE invoice fields present on this page. "
    "For each detected field return its exact bounding box. "
    'Output ONLY a JSON object: {"fieldtype": {"bbox_2d": [x1, y1, x2, y2]}}. '
    "Coordinates are in 0-1000 scale (0=top-left corner, 1000=bottom-right corner). "
    "Omit fields not present on the page. "
    f"Possible fieldtypes: {', '.join(KILE_FIELD_TYPES)}."
)


# ── PCC-snap helpers ──────────────────────────────────────────────────────────


def _pcc_in_bbox(word: WordBox, bbox: BBox) -> bool:
    """Return True if the word's pseudo-character center falls inside bbox."""
    cx = (word.bbox[0] + word.bbox[2]) / 2
    cy = (word.bbox[1] + word.bbox[3]) / 2
    return bbox.left <= cx <= bbox.right and bbox.top <= cy <= bbox.bottom


def _snap_to_words(
    pred_bbox: BBox,
    words: list[WordBox],
) -> tuple[BBox, str] | None:
    """
    Find OCR words whose PCC falls inside pred_bbox.
    Returns (snapped_bbox, text) from the matched words, or None if no match.
    Snapped bbox is the union of matched word bboxes (already snapped by docile OCR).
    """
    matched = [w for w in words if _pcc_in_bbox(w, pred_bbox)]
    if not matched:
        return None
    snapped = BBox(
        min(w.bbox[0] for w in matched),
        min(w.bbox[1] for w in matched),
        max(w.bbox[2] for w in matched),
        max(w.bbox[3] for w in matched),
    )
    text = " ".join(w.text for w in matched)
    return snapped, text


# ── Model wrapper ─────────────────────────────────────────────────────────────


class VLMExtractor:
    """Load a merged Qwen3-VL-2B checkpoint; run per-page KILE extraction with PCC-snap."""

    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self._device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Loading VLM from {model_dir} on {self._device}...")
        self._processor = AutoProcessor.from_pretrained(str(model_dir))
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            device_map=self._device,
        )
        self._model.eval()
        print("VLM loaded.")

    def _run_inference(self, page_image) -> str:
        """Run one forward pass; return raw decoded string."""
        import torch
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": page_image},
                    {"type": "text", "text": _PROMPT},
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

    def extract_page(self, page: PageContext) -> list[Field]:
        """
        Inference + PCC-snap for one page.

        Flow:
          model → {fieldtype: {bbox_2d}} → convert to [0,1] →
          PCC-snap against OCR words → Field with snapped bbox
        """
        raw = self._run_inference(page.image)
        return _parse_and_snap(raw, page.words, page.page_index)


# ── Parse + snap ──────────────────────────────────────────────────────────────


def _parse_and_snap(raw: str, words: list[WordBox], page: int) -> list[Field]:
    """
    Parse model JSON output; PCC-snap each predicted bbox to OCR words.
    Returns Field objects with snapped bboxes (or raw model bbox if no words found).
    """
    # Strip markdown fences
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
            # No OCR words in region — use raw model bbox as fallback
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


# ── Document-level extraction ─────────────────────────────────────────────────


def extract_documents(
    split: str,
    model_dir: Path = DEFAULT_MODEL_DIR,
    limit: int | None = None,
) -> dict[str, list[Field]]:
    """Extract KILE fields for all docs in a split using the fine-tuned VLM."""
    from .data import load_split

    extractor = VLMExtractor(model_dir)
    dataset = load_split(split)
    if limit:
        dataset = dataset[:limit]

    predictions: dict[str, list[Field]] = {}
    for i, doc in enumerate(dataset):
        print(f"[{i + 1}/{len(dataset)}] {doc.docid}")
        doc_fields: list[Field] = []
        for page in iter_pages(doc):
            if page.page_index > 0:
                continue  # trained on page 0 only
            doc_fields.extend(extractor.extract_page(page))
        predictions[doc.docid] = doc_fields

    return predictions


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json as _json

    from .data import load_split
    from .eval import print_scores, run_eval

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    preds = extract_documents(args.split, args.model_dir, args.limit)
    if args.out:
        with open(args.out, "w") as f:
            _json.dump(
                {
                    k: [
                        {
                            "bbox": list(ff.bbox.to_tuple()),
                            "page": ff.page,
                            "text": ff.text,
                            "fieldtype": ff.fieldtype,
                            "score": ff.score,
                        }
                        for ff in v
                    ]
                    for k, v in preds.items()
                },
                f,
                indent=2,
            )
        print(f"Saved to {args.out}")

    dataset = load_split(args.split)
    if args.limit:
        dataset = dataset[: args.limit]
    result = run_eval(dataset, kile_preds=preds, lir_preds={})
    print_scores(result)
