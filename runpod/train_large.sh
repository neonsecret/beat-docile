#!/bin/bash
# Train LayoutLMv3-LARGE on DocILE using the 5090 (32 GB VRAM).
# Effective batch = train_bs * gradient_accumulation_steps = 8 * 2 = 16.
# Lower lr (1e-5) than base-model training (2e-5) suits the larger model.
# Estimated wall time: ~8-10 h for 20 epochs on a single 5090.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

NER_DIR=/workspace/docile-repo/baselines/NER
export PYTHONPATH="${NER_DIR}:${PYTHONPATH:-}"

mkdir -p /workspace/logs /workspace/checkpoints

echo "=== Starting LayoutLMv3-LARGE fine-tuning ==="
echo "    Model : microsoft/layoutlmv3-large"
echo "    Epochs: 20   Batch: 8   GradAccum: 2   EffBatch: 16"
echo "    LR    : 1e-5"
echo "    Log   : /workspace/logs/train_large.log"
echo ""

python ${NER_DIR}/docile_train_NER_multilabel_layoutLMv3.py \
    --docile_path /workspace/data/ \
    --model_name microsoft/layoutlmv3-large \
    --output_dir /workspace/checkpoints/layoutlmv3_large_ft/ \
    --train_bs 8 \
    --test_bs 8 \
    --gradient_accumulation_steps 2 \
    --num_epochs 20 \
    --lr 1e-5 \
    --weight_decay 0.001 \
    --use_BIO_format \
    --tag_everything \
    --report_all_metrics \
    --save_total_limit 2 \
    2>&1 | tee /workspace/logs/train_large.log

echo ""
echo "Training complete. Best checkpoint in /workspace/checkpoints/layoutlmv3_large_ft/"
