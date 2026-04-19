"""[EXPERIMENTAL] Qwen3-VL-8B-Instruct zero-shot extractor for DocILE KILE extraction.

Status: EXPERIMENTAL — built but never run end-to-end. See KNOWLEDGE_BASE.md §2.4 for context.

Designed as an ensemble arm alongside V5b Sonnet, merged via ensemble.py weighted-max.
CUDA: 4-bit NF4 quantization (bitsandbytes) — fits 3070 8GB at ~6.1GB weights.
MPS/CPU: bfloat16 full-precision (testing only, slow).
Imports shared PCC-snap primitives from vlm_extract.py.

Usage:
    uv run python tools/run_qwen8b_50.py
"""

from __future__ import annotations

from docile.dataset import Field

from .data import PageContext, iter_pages
from .vlm_extract import (
    _PROMPT,
    _parse_and_snap,
)

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

# Pixel budget for 3070 8GB: weights ~6.1GB + activations ~1.5GB = ~7.6GB target.
# 768*28*28 ≈ 600K pixels (down from Qwen3-VL default 1280*28*28 ≈ 1M).
_MAX_PIXELS = 768 * 28 * 28
_MIN_PIXELS = 256 * 28 * 28


class Qwen8BExtractor:
    """Zero-shot Qwen3-VL-8B-Instruct extractor with 4-bit NF4 quantization for CUDA."""

    def __init__(self, model_id: str = MODEL_ID, max_new_tokens: int = 512) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if torch.cuda.is_available():
            self._device = "cuda"
        elif torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

        self._max_new_tokens = max_new_tokens

        print(f"Loading {model_id} on {self._device} ...")
        self._processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=_MIN_PIXELS,
            max_pixels=_MAX_PIXELS,
        )

        if self._device == "cuda":
            from transformers import BitsAndBytesConfig

            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=bnb_cfg,
                device_map="auto",
            )
        else:
            # MPS / CPU: no BnB support — load in bfloat16.
            # An 8B model at bf16 needs ~16GB; Mac with 24+ GB unified memory can manage.
            print(
                f"[WARN] {self._device}: bfloat16 full-precision (no 4-bit quantization). "
                "Expect 16+ GB RAM usage. For production use CUDA."
            )
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map=self._device,
            )

        self._model.eval()
        print("Qwen3-VL-8B loaded.")

    def _run_inference(self, page_image) -> str:
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
        # BnB device_map="auto" places model on CUDA; inputs need explicit placement.
        input_device = "cuda" if self._device == "cuda" else self._device
        inputs = self._processor(
            text=[text_prompt],
            images=image_inputs,
            return_tensors="pt",
        ).to(input_device)

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
        """Run inference on one page and PCC-snap predicted bboxes to OCR words."""
        raw = self._run_inference(page.image)
        return _parse_and_snap(raw, page.words, page.page_index)


def extract_documents(
    split: str,
    model_id: str = MODEL_ID,
    limit: int | None = None,
    all_pages: bool = True,
) -> dict[str, list[Field]]:
    """Extract KILE fields for all docs using Qwen3-VL-8B-Instruct zero-shot."""
    from .data import load_split

    extractor = Qwen8BExtractor(model_id)
    dataset = load_split(split)
    if limit:
        dataset = dataset[:limit]

    predictions: dict[str, list[Field]] = {}
    for i, doc in enumerate(dataset):
        print(f"[{i + 1}/{len(dataset)}] {doc.docid}")
        doc_fields: list[Field] = []
        for page in iter_pages(doc):
            if not all_pages and page.page_index > 0:
                continue
            doc_fields.extend(extractor.extract_page(page))
        predictions[doc.docid] = doc_fields

    return predictions
