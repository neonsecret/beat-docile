#!/bin/bash
# Run Track B (LayoutLMv3) inference on a val or test split.
# Usage: bash run_inference_layoutlmv3.sh [val|test] [checkpoint_dir]
# Checkpoint defaults to the latest in ~/beat_docile/checkpoints/layoutlmv3_base_ft/

set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

SPLIT=${1:-val}
CHECKPOINT_DIR=${2:-/home/neon/beat_docile/checkpoints/layoutlmv3_base_ft}

# Find the latest checkpoint subdirectory if not specified exactly
if [ -d "${CHECKPOINT_DIR}" ] && [ -z "$(ls -A ${CHECKPOINT_DIR}/checkpoint-* 2>/dev/null)" ]; then
    echo "No checkpoint-* dirs found in ${CHECKPOINT_DIR}"
    exit 1
fi

CHECKPOINT=$(ls -d ${CHECKPOINT_DIR}/checkpoint-* 2>/dev/null | sort -V | tail -1)
if [ -z "$CHECKPOINT" ]; then
    echo "No checkpoint found. Has training completed at least one save?"
    exit 1
fi

echo "Using checkpoint: ${CHECKPOINT}"

NER_DIR=/home/neon/beat_docile/references/docile/baselines/NER
export PYTHONPATH="${NER_DIR}:${PYTHONPATH:-}"

OUTPUT_DIR=/home/neon/beat_docile/predictions/layoutlmv3_${SPLIT}
mkdir -p ${OUTPUT_DIR}

/home/neon/beat_docile/.venv/bin/python ${NER_DIR}/docile_inference_NER_multilabel_layoutLMv3.py \
    --split ${SPLIT} \
    --docile_path /home/neon/beat_docile/data/ \
    --checkpoint "${CHECKPOINT}" \
    --output_dir ${OUTPUT_DIR} \
    --store_intermediate_results \
    --merge_strategy new \
    2>&1 | tee ${OUTPUT_DIR}/inference.log

echo "Inference done. Predictions in: ${OUTPUT_DIR}"
echo "Evaluate with: cd ~/beat_docile && .venv/bin/bd eval --split ${SPLIT} --predictions ${OUTPUT_DIR}/kile_predictions.json"
