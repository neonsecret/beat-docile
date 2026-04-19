"""[EXPERIMENTAL] Qwen3-VL-8B + LoRA adapter inference for DocILE KILE extraction (Mac-side).

Status: EXPERIMENTAL — pending path B (RunPod LoRA) training result.
See KNOWLEDGE_BASE.md §5.2 for recipe details and projected KILE range.

Fine-tuned Qwen3-VL-8B with r=32 LoRA adapter outputs bbox_2d in 0-1000 scale;
bboxes are PCC-snapped to DocTR snapped OCR words for PCC-IoU=1.0 alignment.
Checkpoint: ~/qwen3vl_lora_docile/ (adapter + processor rsynced from RunPod).

Usage:
    uv run bd qwen3vl-extract --split val --limit 50
    from beat_docile.qwen3vl_extract import extract_documents
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

from docile.dataset import BBox, Field

from .data import PageContext, WordBox, iter_pages

DEFAULT_ADAPTER_DIR = Path.home() / "qwen3vl_lora_docile"
BASE_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

_MAX_PIXELS = 1280 * 28 * 28
_MIN_PIXELS = 256 * 28 * 28

_KILE_FIELDS = frozenset(
    [
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
)

_KILE_PROMPT = (
    "Locate all KILE invoice fields present on this page. "
    "For each field output its bounding box (0-1000 scale) and extracted text. "
    "Return ONLY valid JSON. Omit fields not present on this page. "
    'Format: {"fieldtype": {"bbox_2d": [x1, y1, x2, y2], "text": "..."}} '
    "For multiple instances (tax_detail_*): "
    '{"fieldtype": [{"bbox_2d": [...], "text": "..."}, ...]} '
    "If no KILE fields are present, return {}. "
    f"Possible fieldtypes: {', '.join(sorted(_KILE_FIELDS))}."
)


# ── PCC-snap helpers (mirrors vlm_extract.py) ──────────────────────────────────


def _pcc_in_bbox(word: WordBox, bbox: BBox) -> bool:
    cx = (word.bbox[0] + word.bbox[2]) / 2
    cy = (word.bbox[1] + word.bbox[3]) / 2
    return bbox.left <= cx <= bbox.right and bbox.top <= cy <= bbox.bottom


def _snap_to_words(pred_bbox: BBox, words: list[WordBox]) -> tuple[BBox, str] | None:
    """PCC-snap: find words whose center falls inside pred_bbox → snapped union bbox."""
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


# ── Parse model output ────────────────────────────────────────────────────────


def _parse_and_snap(raw: str, words: list[WordBox], page: int) -> list[Field]:
    """Parse model JSON; PCC-snap each predicted bbox to OCR words → Field list."""
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []

    fields: list[Field] = []

    def _process_entry(fieldtype: str, entry: dict) -> None:
        if not isinstance(entry, dict):
            return
        bbox_2d = entry.get("bbox_2d")
        if not bbox_2d or len(bbox_2d) != 4:
            return
        try:
            x1, y1, x2, y2 = bbox_2d
            pred_bbox = BBox(x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0)
        except (TypeError, ValueError):
            return

        snap = _snap_to_words(pred_bbox, words)
        if snap:
            bbox, text = snap
        else:
            bbox = pred_bbox
            text = entry.get("text", "")

        fields.append(Field(bbox=bbox, page=page, score=1.0, text=text, fieldtype=fieldtype))

    for fieldtype, val in data.items():
        if fieldtype not in _KILE_FIELDS:
            continue
        if isinstance(val, list):
            for entry in val:
                _process_entry(fieldtype, entry)
        else:
            _process_entry(fieldtype, val)

    return fields


# ── Model wrapper ─────────────────────────────────────────────────────────────


class Qwen3VLExtractor:
    """Load Qwen3-VL-8B + LoRA adapter; run per-page KILE extraction."""

    def __init__(
        self,
        adapter_dir: Path = DEFAULT_ADAPTER_DIR,
        base_model_id: str = BASE_MODEL_ID,
        max_new_tokens: int = 1024,
    ) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self._device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self._max_new_tokens = max_new_tokens

        print(f"Loading Qwen3-VL-8B base from {base_model_id} on {self._device} ...")
        if self._device not in ("cuda",):
            print(
                f"[WARN] {self._device}: BF16 full-precision (~16 GB unified memory). "
                "Requires Mac with ≥24 GB RAM."
            )

        self._processor = AutoProcessor.from_pretrained(
            str(adapter_dir),  # processor was saved alongside the adapter
            min_pixels=_MIN_PIXELS,
            max_pixels=_MAX_PIXELS,
        )

        base = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map=self._device,
        )
        # Load and merge LoRA adapter for fastest inference (no adapter overhead)
        model = PeftModel.from_pretrained(base, str(adapter_dir))
        self._model = model.merge_and_unload()
        self._model.eval()
        print("Qwen3-VL-8B + LoRA adapter loaded and merged.")

    def _run_inference(self, page_image) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": page_image},
                    {"type": "text", "text": _KILE_PROMPT},
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
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        return self._processor.decode(generated, skip_special_tokens=True).strip()

    def extract_page(self, page: PageContext) -> list[Field]:
        """Run model inference on one page and PCC-snap predicted bboxes to OCR words."""
        raw = self._run_inference(page.image)
        return _parse_and_snap(raw, page.words, page.page_index)


# ── Document-level extraction ─────────────────────────────────────────────────


def extract_documents(
    docs: Sequence,
    adapter_dir: Path = DEFAULT_ADAPTER_DIR,
    base_model_id: str = BASE_MODEL_ID,
    limit: int | None = None,
) -> dict[str, list[Field]]:
    """Extract KILE fields for a sequence of Document objects using fine-tuned VLM.

    Returns {docid: [Field, ...]} — all docs appear even if extraction yields [].
    Mirrors the interface of vlm_extract.extract_documents.
    """
    extractor = Qwen3VLExtractor(adapter_dir=adapter_dir, base_model_id=base_model_id)
    if limit:
        docs = list(docs)[:limit]

    predictions: dict[str, list[Field]] = {}
    for i, doc in enumerate(docs):
        print(f"[{i + 1}/{len(docs) if hasattr(docs, '__len__') else '?'}] {doc.docid}")
        predictions[doc.docid] = []
        for page in iter_pages(doc):
            predictions[doc.docid].extend(extractor.extract_page(page))

    return predictions


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json as _json

    from .data import load_split
    from .eval import print_scores, run_eval

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--base-model", default=BASE_MODEL_ID)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    dataset = load_split(args.split)
    docs = list(dataset)[: args.limit] if args.limit else list(dataset)

    preds = extract_documents(docs, adapter_dir=args.adapter_dir, base_model_id=args.base_model)

    if args.out:
        with open(args.out, "w") as fh:
            _json.dump(
                {
                    k: [
                        {
                            "bbox": list(f.bbox.to_tuple()),
                            "page": f.page,
                            "text": f.text,
                            "fieldtype": f.fieldtype,
                            "score": f.score,
                        }
                        for f in v
                    ]
                    for k, v in preds.items()
                },
                fh,
                indent=2,
            )
        print(f"Saved to {args.out}")

    result = run_eval(
        dataset if not args.limit else dataset[: args.limit], kile_preds=preds, lir_preds={}
    )
    print_scores(result)
