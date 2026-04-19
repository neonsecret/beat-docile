#!/bin/bash
# Setup RunPod RTX 5090 pod for Qwen3-VL-8B LoRA fine-tune.
# Idempotent — safe to run multiple times.
# Run time: ~15-25 min (dominated by model download, ~16 GB).
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/workspace/Qwen3-VL-8B-Instruct}"
DATA_ROOT="${DATA_ROOT:-/workspace/docile_data}"
CACHE_DIR="${CACHE_DIR:-/workspace/dataset_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs/qwen3vl_lora_docile}"

echo "=== [1/6] System packages ==="
apt-get update -qq
apt-get install -y git tmux htop rsync unzip poppler-utils

echo "=== [2/6] PyTorch cu128 fix for RTX 5090 (Blackwell SM_120 kernels) ==="
# Default RunPod image ships torch 2.4+cu124 — broken on 5090 in two ways:
#   1. torch 2.4 too old for current transformers
#   2. cu124 has no SM_120 kernels → CUDA error on first forward pass
# Fix: install torch 2.11+cu128 which includes SM_120 Blackwell kernels.
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128 \
    --quiet

# Verify SM_120 kernels are available
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available after reinstall'
print(f'torch {torch.__version__} | device: {torch.cuda.get_device_name(0)}')
x = torch.ones(4, device='cuda')
print(f'CUDA forward pass OK: {x.sum().item()}')
"

echo "=== [3/6] ML packages ==="
pip install --quiet \
    "transformers>=4.52" \
    "accelerate>=1.6" \
    "peft>=0.14" \
    "trl>=0.12" \
    "bitsandbytes>=0.45" \
    "qwen-vl-utils>=0.0.8" \
    "docile-benchmark>=0.3" \
    "pillow>=11" \
    "tqdm" \
    "huggingface_hub"

echo "=== [4/6] Download Qwen3-VL-8B-Instruct model weights ==="
if [ ! -d "${MODEL_DIR}" ] || [ -z "$(ls -A ${MODEL_DIR} 2>/dev/null)" ]; then
    echo "Downloading to ${MODEL_DIR} ..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Qwen/Qwen3-VL-8B-Instruct',
    local_dir='${MODEL_DIR}',
    ignore_patterns=['*.bin'],  # prefer safetensors
)
print('Model download complete.')
"
else
    echo "Model already present at ${MODEL_DIR}, skipping download."
fi

echo "=== [5/6] Verify DocILE dataset ==="
if [ ! -d "${DATA_ROOT}" ]; then
    echo "WARNING: DocILE data not found at ${DATA_ROOT}."
    echo "Run download_data.sh first, or set DATA_ROOT env var."
    echo "Continuing setup — dataset will be needed at training time."
else
    python3 -c "
from docile.dataset import Dataset
ds = Dataset('train', dataset_path='${DATA_ROOT}', load_annotations=True, load_ocr=False)
print(f'DocILE train split: {len(ds)} documents OK')
"
fi

echo "=== [6/6] Sanity check: model loads in BF16 ==="
python3 - <<'PYEOF'
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

model_dir = "/workspace/Qwen3-VL-8B-Instruct"
print(f"Loading processor from {model_dir} ...")
proc = AutoProcessor.from_pretrained(model_dir, min_pixels=256*28*28, max_pixels=1280*28*28)
print(f"Loading model in BF16 ...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype=torch.bfloat16, device_map="auto"
)
total = sum(p.numel() for p in model.parameters())
print(f"Model loaded: {total/1e9:.2f}B params")
vram = torch.cuda.memory_allocated() / 1e9
print(f"VRAM used after load: {vram:.2f} GB")
del model
torch.cuda.empty_cache()
print("Sanity check PASSED.")
PYEOF

mkdir -p "${CACHE_DIR}" "${OUTPUT_DIR}"

echo ""
echo "=== Setup complete ==="
echo "  Model:   ${MODEL_DIR}"
echo "  Data:    ${DATA_ROOT}"
echo "  Cache:   ${CACHE_DIR}"
echo "  Output:  ${OUTPUT_DIR}"
echo ""
echo "To launch training in tmux:"
echo "  tmux new-session -d -s train -x 220 -y 50"
echo "  tmux send-keys -t train 'python /workspace/qwen_vl_train.py \\"
echo "      --data-root ${DATA_ROOT} \\"
echo "      --model-dir ${MODEL_DIR} \\"
echo "      --output-dir ${OUTPUT_DIR} \\"
echo "      --cache-dir ${CACHE_DIR}' Enter"
echo "  tmux attach -t train"
