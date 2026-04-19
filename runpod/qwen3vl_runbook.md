# Qwen3-VL-8B LoRA Fine-Tune Runbook

DocILE KILE grounding via fine-tuned Qwen3-VL-8B-Instruct.  
Target: close the 26 pp gap to GraphDoc 71.25% KILE.

---

## Gate criteria

Before spending the budget, the fine-tuned checkpoint must clear:

- Val KILE AP ≥ 55% (≥10 pp above v2's 44.61%) on 500-doc val split
- Val LIR F1 ≥ 45% (optional; KILE is primary gate)

If the checkpoint underperforms 50% KILE AP, investigate training loss curve before spending on a second run.

---

## 1. Pod creation

1. Log in to RunPod → **Deploy** → GPU Cloud  
2. GPU: **RTX 5090** (32 GB VRAM, $0.69/hr)  
3. Template: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`  
   (any recent PyTorch base is fine — we reinstall torch in step 2)  
4. Container disk: **50 GB** (model 16 GB + dataset ~20 GB + outputs)  
5. Volume disk: **0** (no persistent volume needed)  
6. **NO credentials**, NO `.env`, NO API keys on the pod — see project policy

---

## 2. Initial SSH connection

```bash
# From RunPod dashboard, copy the SSH command, e.g.:
ssh -i ~/.ssh/id_ed25519 root@<pod-ip> -p <port>
```

---

## 3. PyTorch cu128 fix (RTX 5090 / Blackwell SM_120)

The default image ships `torch 2.4+cu124` — broken on 5090 in two ways:  
1. `torch 2.4` too old for current transformers  
2. `cu124` has no SM_120 kernels → `CUDA error: no kernel image` on first forward pass

**Fix is embedded in `setup_qwen.sh` step 2.** Manual command if needed:

```bash
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
```

Expected result: `torch 2.11.0+cu128`

---

## 4. Upload scripts to pod

From your Mac (adjust port):

```bash
RUNPOD_HOST="root@<pod-ip> -p <port>"

# Upload training script and setup script
scp -P <port> \
    runpod/setup_qwen.sh \
    runpod/qwen_vl_train.py \
    root@<pod-ip>:/workspace/
```

---

## 5. Run setup script

```bash
# On the pod:
chmod +x /workspace/setup_qwen.sh
bash /workspace/setup_qwen.sh 2>&1 | tee /workspace/setup.log
```

This does (idempotent):
- System packages (git, tmux, rsync)
- PyTorch cu128 fix
- ML packages (transformers, peft, trl, bitsandbytes, qwen-vl-utils, docile-benchmark)
- Download `Qwen/Qwen3-VL-8B-Instruct` (~16 GB) to `/workspace/Qwen3-VL-8B-Instruct/`
- Verify DocILE dataset at `/workspace/docile_data/`
- Sanity check: model loads in BF16, VRAM check

Expected output: `"Sanity check PASSED."` and VRAM ~16 GB after load.

---

## 6. Upload DocILE dataset (if not already on pod)

The dataset is ~20 GB. Two options:

**Option A — rsync from Mac** (if dataset is on your Mac):
```bash
rsync -avz --progress -e "ssh -p <port>" \
    /path/to/docile_data/ \
    root@<pod-ip>:/workspace/docile_data/
```

**Option B — download directly on pod** using the existing `download_data.sh` pattern.

---

## 7. Launch training in tmux

### 7a. SMOKE TEST FIRST (~5 min, ~$0.06)

**Run this before full training.** Catches OOM errors and NaN loss before committing the budget.

```bash
python /workspace/qwen_vl_train.py \
    --data-root /workspace/docile_data \
    --model-dir /workspace/Qwen3-VL-8B-Instruct \
    --output-dir /workspace/outputs/_smoke \
    --cache-dir /workspace/dataset_cache \
    --epochs 1 --batch-size 1 --grad-accum 1 \
    --limit-samples 8 --max-steps 1 \
    2>&1 | tee /workspace/smoke.log
```

**Gate:** Look for a line like `{'loss': 2.34, 'learning_rate': ...}` in the output.  
- Loss is a finite number → **smoke passed**, proceed to full training  
- OOM error → reduce `--batch-size 1` and add `--grad-accum 16` to the full run command  
- `nan` or `inf` loss → check transformers version (`pip install -U transformers`) and re-run

---

### 7b. Full training

```bash
# On the pod:
tmux new-session -d -s train -x 220 -y 50

tmux send-keys -t train \
'python /workspace/qwen_vl_train.py \
    --data-root /workspace/docile_data \
    --model-dir /workspace/Qwen3-VL-8B-Instruct \
    --output-dir /workspace/outputs/qwen3vl_lora_docile \
    --cache-dir /workspace/dataset_cache \
    --epochs 3 \
    --batch-size 2 \
    --grad-accum 8 \
    --lr 2e-4 \
    2>&1 | tee /workspace/train.log' Enter

tmux attach -t train
```

Detach with `Ctrl+B, D`. Reconnect: `tmux attach -t train`.

### What the training script does

1. **Dataset prep** (~10-15 min, one-time): renders all ~6700 train pages to PNG at 150 DPI, saves `dataset_cache/train_index.jsonl`
2. **Train/val split**: holds out last 100 pages as an in-training eval set
3. **Training**: 3 epochs × ~411 steps/epoch (batch=16) = ~1233 optimizer steps; eval runs every 125 steps
4. **Saves**: LoRA adapter + processor to `/workspace/outputs/qwen3vl_lora_docile/`

---

## 8. Training cost and wall-clock estimate

| Phase | Duration | Cost at $0.69/hr |
|---|---|---|
| Setup + model download | ~25 min | ~$0.29 |
| Dataset preprocessing | ~15 min | ~$0.17 |
| Training (3 epochs, ~6700 pages) | 4–8 hr | $2.76–$5.52 |
| **Total** | **~5–9 hr** | **~$3.50–$6.00** |

Budget remaining: **$7.66** — training fits comfortably with ~$1–4 to spare.

Monitor VRAM during training:
```bash
# In a second tmux window:
watch -n 5 nvidia-smi
```

Expected peak VRAM: ~20–24 GB (model 16 GB BF16 + LoRA optimizer + activations).

---

## 9. Monitor training progress

```bash
tail -f /workspace/train.log
```

Healthy indicators:
- Training loss starts ~2–4, decreases toward 0.5–1.5 over 3 epochs
- Steps/second: ~0.06–0.12 (8–15 s per step with batch=2 + grad_ckpt)
- No OOM errors (if OOM: reduce `--batch-size 1` or add `--grad-accum 16`)

If training crashes: check `train.log` for the error. Common issues:
- OOM → reduce batch size
- `CUDA error: no kernel image` → cu128 fix was not applied (re-run step 2)
- `all_tied_weights_keys` AttributeError → update transformers: `pip install -U transformers`

---

## 10. Download adapter to Mac

```bash
# From Mac (adapter is ~200 MB, fast download):
rsync -avz --progress -e "ssh -p <port>" \
    root@<pod-ip>:/workspace/outputs/qwen3vl_lora_docile/ \
    ~/qwen3vl_lora_docile/
```

The adapter directory contains:
- `adapter_config.json` + `adapter_model.safetensors` (~200 MB)
- `processor_config.json`, `tokenizer*.json`, etc.

---

## 11. Mac inference setup

Ensure `Qwen/Qwen3-VL-8B-Instruct` is in your HuggingFace cache (~16 GB):

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-VL-8B-Instruct')
print('Base model ready.')
"
```

Install inference deps if not already present:
```bash
uv add peft qwen-vl-utils
```

---

## 12. Run inference on val split

```bash
# 500 val docs (full eval):
uv run python -m beat_docile.qwen3vl_extract \
    --split val \
    --adapter-dir ~/qwen3vl_lora_docile \
    --out predictions/qwen3vl_lora_val.json

# Quick smoke test (10 docs):
uv run python -m beat_docile.qwen3vl_extract \
    --split val \
    --limit 10 \
    --adapter-dir ~/qwen3vl_lora_docile
```

---

## 13. Evaluation

The inference script prints KILE AP and LIR F1 at the end. Alternatively:

```bash
uv run python -c "
import json
from beat_docile.data import load_split
from beat_docile.eval import run_eval, print_scores
from docile.dataset import BBox, Field

with open('predictions/qwen3vl_lora_val.json') as f:
    raw = json.load(f)
preds = {
    k: [Field(bbox=BBox(*v2['bbox']), page=v2['page'], fieldtype=v2['fieldtype'], score=v2['score'])
        for v2 in v]
    for k, v in raw.items()
}
ds = load_split('val')
result = run_eval(ds, kile_preds=preds, lir_preds={})
print_scores(result)
"
```

Gate: KILE AP ≥ 55% → proceed to ensemble with v2 pipeline.  
If KILE AP < 50% → investigate loss curve + possible training bug before spending more.

---

## 14. Checkpoint directory layout

```
/workspace/outputs/qwen3vl_lora_docile/      ← on RunPod
~/qwen3vl_lora_docile/                        ← on Mac after rsync

├── adapter_config.json                        # LoRA config (r=32, alpha=64, ...)
├── adapter_model.safetensors                  # ~200 MB LoRA weights
├── preprocessor_config.json
├── processor_config.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
└── checkpoint-*/                              # intermediate checkpoints (safe to delete after eval)
```

---

## 15. Stopping the pod

**After adapter is downloaded**, stop the pod from the RunPod dashboard.  
Do NOT terminate before the rsync in step 10 completes.

```bash
# Verify rsync completed before stopping:
ls -lh ~/qwen3vl_lora_docile/adapter_model.safetensors
# Expected: ~150-250 MB file
```
