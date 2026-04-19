#!/usr/bin/env python3
"""Qwen3-VL-8B LoRA fine-tune for DocILE KILE grounding.

Design:
  - PEFT LoRA r=32 on LM linear layers; vision tower frozen
  - BF16, no quantization (32 GB 5090 has headroom without QLoRA)
  - Gradient checkpointing to bound activation memory
  - Per-page training: one conversation per page image
  - ~6700 pages from 5180 train docs; 3 epochs; effective batch 16
  - Adapter saved to outputs/qwen3vl_lora_docile/

Usage (on RunPod after setup_qwen.sh):
    python qwen_vl_train.py \
        --data-root /workspace/docile_data \
        --model-dir /workspace/Qwen3-VL-8B-Instruct \
        --output-dir /workspace/outputs/qwen3vl_lora_docile \
        --cache-dir /workspace/dataset_cache
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

# ── Field catalog ──────────────────────────────────────────────────────────────

_KILE_FIELDS = [
    "account_num", "amount_due", "amount_paid", "amount_total_gross",
    "amount_total_net", "amount_total_tax", "bank_num", "bic",
    "currency_code_amount_due", "customer_billing_address", "customer_billing_name",
    "customer_delivery_address", "customer_delivery_name", "customer_id",
    "customer_order_id", "customer_other_address", "customer_other_name",
    "customer_registration_id", "customer_tax_id", "date_due", "date_issue",
    "document_id", "iban", "order_id", "payment_reference", "payment_terms",
    "tax_detail_gross", "tax_detail_net", "tax_detail_rate", "tax_detail_tax",
    "vendor_address", "vendor_email", "vendor_name", "vendor_order_id",
    "vendor_registration_id", "vendor_tax_id",
]

_MULTI_KILE = {"tax_detail_gross", "tax_detail_net", "tax_detail_rate", "tax_detail_tax"}

# Pixel budget: Qwen3-VL default (1280 patches x 28x28 pixels per patch).
# Matches the existing vlm_extract.py comment re: "Qwen3-VL default 1280*28*28 ~= 1M pixels."
_MAX_PIXELS = 1280 * 28 * 28
_MIN_PIXELS = 256 * 28 * 28

_KILE_PROMPT = (
    "Locate all KILE invoice fields present on this page. "
    "For each field output its bounding box (0-1000 scale) and extracted text. "
    "Return ONLY valid JSON. Omit fields not present on this page. "
    "Format: {\"fieldtype\": {\"bbox_2d\": [x1, y1, x2, y2], \"text\": \"...\"}} "
    "For multiple instances (tax_detail_*): "
    "{\"fieldtype\": [{\"bbox_2d\": [...], \"text\": \"...\"}, ...]} "
    "If no KILE fields are present, return {}. "
    f"Possible fieldtypes: {', '.join(_KILE_FIELDS)}."
)


# ── Annotation → training response ────────────────────────────────────────────

def _build_response(kile_fields: list[Any]) -> dict:
    """Convert docile Field list → JSON-serializable bbox dict."""
    from collections import defaultdict

    by_type: dict[str, list[dict]] = defaultdict(list)
    for f in kile_fields:
        x1, y1, x2, y2 = f.bbox.to_tuple()
        entry = {
            "bbox_2d": [int(x1 * 1000), int(y1 * 1000), int(x2 * 1000), int(y2 * 1000)],
            "text": f.text or "",
        }
        by_type[f.fieldtype].append(entry)

    result: dict[str, Any] = {}
    for ft, entries in by_type.items():
        result[ft] = entries if len(entries) > 1 else entries[0]
    return result  # {} → teach model to refuse hallucination when page has no fields


# ── Dataset preparation ────────────────────────────────────────────────────────

def prepare_dataset(data_root: str, cache_dir: str, split: str = "train") -> list[dict]:
    """Pre-render page images to disk and build annotation index.

    Returns list of {"image_path": str, "response": dict}.
    Idempotent: skips pages whose PNG already exists.
    """
    from docile.dataset import Dataset

    cache_path = Path(cache_dir) / f"{split}_index.jsonl"
    images_dir = Path(cache_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"Loading cached dataset index from {cache_path}")
        with open(cache_path) as fh:
            return [json.loads(line) for line in fh]

    print(f"Building {split} dataset from {data_root} ...")
    dataset = Dataset(split, dataset_path=data_root, load_annotations=True, load_ocr=False)
    items: list[dict] = []

    for doc in tqdm(dataset, desc=f"Preprocessing {split}"):
        with doc:
            for page_idx in range(doc.page_count):
                # KILE fields only (line_item_id=None distinguishes KILE from LIR)
                kile_fields = [
                    f for f in doc.annotation.fields
                    if f.page == page_idx and f.line_item_id is None
                ]
                response = _build_response(kile_fields)

                img_path = images_dir / f"{doc.docid}_p{page_idx}.png"
                if not img_path.exists():
                    w150, h150 = doc.page_image_size(page_idx, dpi=150)
                    img = doc.page_image(page_idx, image_size=(w150, h150))
                    img.save(str(img_path))

                items.append({
                    "image_path": str(img_path),
                    "response": response,
                })

    with open(cache_path, "w") as fh:
        for item in items:
            fh.write(json.dumps(item) + "\n")

    print(f"Saved {len(items)} samples → {cache_path}")
    return items


class DocILEDataset(torch.utils.data.Dataset):
    def __init__(self, items: list[dict]) -> None:
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        item = self._items[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        response_json = json.dumps(item["response"], ensure_ascii=False)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": _KILE_PROMPT},
                ],
            },
            {"role": "assistant", "content": response_json},
        ]
        return {"messages": messages}


# ── Data collator ──────────────────────────────────────────────────────────────

@dataclasses.dataclass
class QwenVLCollator:
    """Multimodal collator: processes images, tokenizes, masks prompt tokens in labels."""

    processor: Any
    max_length: int = 2048

    # Qwen3-VL chat template adds this string at the start of the assistant turn
    _response_prefix: str = "<|im_start|>assistant\n"

    def _find_response_start(self, input_ids: list[int]) -> int:
        """Return index of first token belonging to the assistant response."""
        prefix_ids = self.processor.tokenizer.encode(
            self._response_prefix, add_special_tokens=False
        )
        n, k = len(input_ids), len(prefix_ids)
        for i in range(n - k, -1, -1):
            if input_ids[i: i + k] == prefix_ids:
                return i + k
        return n  # fallback: mask entire sequence (shouldn't happen)

    def __call__(self, samples: list[dict]) -> dict:
        from qwen_vl_utils import process_vision_info

        texts: list[str] = []
        all_image_inputs: list[Any] = []

        for s in samples:
            text = self.processor.apply_chat_template(
                s["messages"], tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            image_inputs, _ = process_vision_info(s["messages"])
            all_image_inputs.extend(image_inputs)

        batch = self.processor(
            text=texts,
            images=all_image_inputs if all_image_inputs else None,
            return_tensors="pt",
            padding=True,
            padding_side="right",
            truncation=True,
            max_length=self.max_length,
        )

        # Build labels: -100 for prompt tokens, real IDs for response tokens
        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        pad_id = self.processor.tokenizer.pad_token_id

        for i, ids_row in enumerate(input_ids):
            ids_list = ids_row.tolist()
            resp_start = self._find_response_start(ids_list)
            labels[i, :resp_start] = -100
            if pad_id is not None:
                labels[i, ids_row == pad_id] = -100

        batch["labels"] = labels
        return batch


# ── Training ───────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoProcessor,
        Qwen3VLForConditionalGeneration,
        Trainer,
        TrainingArguments,
    )

    print("=== Preparing dataset ===")
    items = prepare_dataset(args.data_root, args.cache_dir, split="train")
    if args.limit_samples:
        items = items[:args.limit_samples]

    # Hold out last 100 pages for eval-during-training (catches overfitting before epoch 3)
    if len(items) > 200:
        val_items = items[-100:]
        train_items = items[:-100]
        val_ds: DocILEDataset | None = DocILEDataset(val_items)
        print(f"Train: {len(train_items)} samples, Val: {len(val_items)} samples")
    else:
        train_items = items
        val_ds = None
        print(f"Train: {len(train_items)} samples (too small to split val)")

    train_ds = DocILEDataset(train_items)

    print("=== Loading model & processor ===")
    processor = AutoProcessor.from_pretrained(
        args.model_dir,
        min_pixels=_MIN_PIXELS,
        max_pixels=_MAX_PIXELS,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Freeze vision tower — only fine-tune LM layers
    frozen = 0
    for name, param in model.named_parameters():
        if "visual" in name:
            param.requires_grad = False
            frozen += param.numel()
    print(f"Frozen vision tower parameters: {frozen:,}")

    # Enable gradient checkpointing before adding LoRA
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("=== Configuring Trainer ===")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,  # -1 = disabled; set >0 for smoke test
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        gradient_checkpointing=False,  # already enabled manually above
        logging_steps=25,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="none",
        optim="adamw_torch",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=125,
        per_device_eval_batch_size=1,
    )

    collator = QwenVLCollator(processor=processor, max_length=args.max_seq_len)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    print("=== Training ===")
    trainer.train()

    print("=== Saving adapter ===")
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    print(f"Adapter + processor saved to {output_dir}")


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen3-VL-8B LoRA fine-tune for DocILE KILE")
    p.add_argument("--data-root", required=True, help="DocILE dataset root dir")
    p.add_argument("--model-dir", default="/workspace/Qwen3-VL-8B-Instruct")
    p.add_argument("--output-dir", default="/workspace/outputs/qwen3vl_lora_docile")
    p.add_argument("--cache-dir", default="/workspace/dataset_cache")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--limit-samples", type=int, default=0,
                   help="Truncate dataset to first N samples (0 = no limit; use 8 for smoke test)")
    p.add_argument("--max-steps", type=int, default=-1,
                   help="Override num_train_epochs step count (-1 = disabled; use 1 for smoke test)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Python {sys.version}")
    print(f"PyTorch {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    train(args)
