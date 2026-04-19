"""Fine-tune Qwen3-VL-2B-Instruct on DocILE KILE extraction with Unsloth.

Run on RunPod 5090 (32GB VRAM):
    python3 /workspace/train_vlm_unsloth.py

Output:
    /workspace/qwen3vl_docile_lora/   — LoRA adapter
    /workspace/qwen3vl_docile_merged/ — merged 16-bit model (rsync to neon)
"""

from __future__ import annotations

import json
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID      = "Qwen/Qwen3-VL-2B-Instruct"
DATA_PATH     = "/workspace/data/vlm_training/train.jsonl"
OUTPUT_LORA   = "/workspace/qwen3vl_docile_lora"
OUTPUT_MERGED = "/workspace/qwen3vl_docile_merged"

MAX_SEQ_LEN  = 4096
LORA_RANK    = 64
LORA_ALPHA   = 128
LEARNING_RATE = 2e-4
BATCH_SIZE   = 2
GRAD_ACC     = 4
EPOCHS       = 3

# ── Load model ─────────────────────────────────────────────────────────────────
print("Loading Qwen3-VL-2B-Instruct via Unsloth...")
from unsloth import FastVisionModel  # noqa: E402

model, processor = FastVisionModel.from_pretrained(
    MODEL_ID,
    load_in_4bit=False,
    use_gradient_checkpointing="unsloth",
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    bias="none",
    random_state=42,
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

# ── Load records ───────────────────────────────────────────────────────────────
records = []
with open(DATA_PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))
print(f"Training examples: {len(records)}")

# ── Format function (called per-example by the data collator) ─────────────────
from PIL import Image  # noqa: E402

def formatting_func(example: dict) -> list[dict]:
    """Convert JSONL record → Qwen3-VL message list with embedded PIL image."""
    img = Image.open(example["image"]).convert("RGB")
    convs = example["conversations"]
    user_text = convs[0]["value"].replace("<image>\n", "").strip()
    assistant_text = convs[1]["value"]

    messages: list[dict] = []
    if example.get("system"):
        messages.append({"role": "system", "content": example["system"]})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text",  "text": user_text},
        ],
    })
    messages.append({"role": "assistant", "content": assistant_text})
    return messages

# ── Data collator ──────────────────────────────────────────────────────────────
from unsloth import UnslothVisionDataCollator  # noqa: E402

collator = UnslothVisionDataCollator(
    model=model,
    processor=processor,
    formatting_func=formatting_func,
    max_seq_length=MAX_SEQ_LEN,
    train_on_responses_only=True,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

# ── Trainer ────────────────────────────────────────────────────────────────────
from transformers import TrainingArguments  # noqa: E402
from trl import SFTTrainer               # noqa: E402

training_args = TrainingArguments(
    output_dir=OUTPUT_LORA,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    report_to="none",
    optim="adamw_8bit",
    remove_unused_columns=False,
    dataloader_num_workers=0,           # collator opens files — keep in main process
)

trainer = SFTTrainer(
    model=model,
    tokenizer=processor,
    train_dataset=records,              # plain list; collator handles formatting
    args=training_args,
    data_collator=collator,
    max_seq_length=MAX_SEQ_LEN,
    packing=False,
    dataset_text_field="",              # collator builds text, not SFTTrainer
)

# ── Train ──────────────────────────────────────────────────────────────────────
print("Starting training...")
trainer.train()

# ── Save LoRA ──────────────────────────────────────────────────────────────────
print(f"Saving LoRA adapter → {OUTPUT_LORA}")
model.save_pretrained(OUTPUT_LORA)
processor.save_pretrained(OUTPUT_LORA)

# ── Merge to 16-bit ────────────────────────────────────────────────────────────
print(f"Merging → {OUTPUT_MERGED} (this takes a few minutes)")
model.save_pretrained_merged(OUTPUT_MERGED, processor, save_method="merged_16bit")

print("Training complete. Ready to rsync to neon.")
