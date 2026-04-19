#!/bin/bash
# Run inference with the trained LayoutLMv3-LARGE checkpoint.
# Usage: bash inference.sh [val|test]
# Defaults to val.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

SPLIT=${1:-val}
NER_DIR=/workspace/docile-repo/baselines/NER
export PYTHONPATH="${NER_DIR}:${PYTHONPATH:-}"

CHECKPOINT_BASE=/workspace/checkpoints/layoutlmv3_large_ft

# Pick the latest checkpoint subdir (e.g. checkpoint-1234)
CHECKPOINT=$(ls -d ${CHECKPOINT_BASE}/checkpoint-* 2>/dev/null | sort -V | tail -1)
if [ -z "${CHECKPOINT}" ]; then
    echo "ERROR: No checkpoint found in ${CHECKPOINT_BASE}. Has training finished?"
    exit 1
fi
echo "Using checkpoint: ${CHECKPOINT}"

OUTPUT_DIR=/workspace/predictions/large_${SPLIT}
mkdir -p "${OUTPUT_DIR}"

echo "=== Running inference on split: ${SPLIT} ==="
python ${NER_DIR}/docile_inference_NER_multilabel_layoutLMv3.py \
    --split "${SPLIT}" \
    --docile_path /workspace/data/ \
    --checkpoint "${CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR}" \
    --store_intermediate_results \
    --merge_strategy new \
    2>&1 | tee "${OUTPUT_DIR}/inference.log"

echo ""
echo "Inference done. Predictions in: ${OUTPUT_DIR}"
echo "Key files:"
ls "${OUTPUT_DIR}"/*.json 2>/dev/null || echo "(no .json files yet — check the log)"
