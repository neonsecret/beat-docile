# RunPod: LayoutLMv3-LARGE Training

Train `microsoft/layoutlmv3-large` (358M params) on DocILE using a RunPod RTX 5090 (32 GB VRAM).  
Estimated cost: ~$7. Estimated wall time: 8-10 h.

---

## 1. Create the Pod

In the RunPod console:

- GPU: **RTX 5090** (Community Cloud, $0.69/hr)
- Image: **RunPod PyTorch 2.x** (e.g. `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`)
- Disk: **100 GB minimum** (dataset + checkpoints + predictions)
- GPUs: **1**
- Expose port **22** (SSH)

Note the pod IP and SSH port after it starts.

---

## 2. Copy Scripts to the Pod

From your Mac (replace `<ip>` and `<port>`):

```bash
export POD_IP=<pod-ip>       # your pod's IP (from RunPod dashboard)
export POD_PORT=22           # or whatever port RunPod assigned for SSH

scp -P $POD_PORT -r runpod/ root@$POD_IP:/workspace/setup_scripts/
```

---

## 3. SSH In and Start a tmux Session

```bash
ssh -p $POD_PORT root@$POD_IP
tmux new -s train
```

All following commands run inside tmux so they survive SSH disconnects.

---

## 4. Run in Order

```bash
# Step 1: Install deps and apply compatibility patches (~5 min)
bash /workspace/setup_scripts/setup.sh

# Step 2: Download DocILE dataset (~5-10 min)
export DOCILE_TOKEN=<your-docile-token>  # request from DocILE benchmark organizers
bash /workspace/setup_scripts/download_data.sh

# Step 3: Train LayoutLMv3-LARGE (~8-10 h)
bash /workspace/setup_scripts/train_large.sh

# Step 4: Run inference on val (~30 min)
bash /workspace/setup_scripts/inference.sh val

# Optional: run inference on test too (costs ~30 min extra)
# bash /workspace/setup_scripts/inference.sh test
```

Detach from tmux at any time with `Ctrl-B d`. Reattach with `tmux attach -t train`.

---

## 5. Download Predictions to Mac

From your Mac (while the pod is still running):

```bash
bash runpod/fetch_results.sh root@$POD_IP:$POD_PORT
```

Predictions land in `runpod_results/` at the project root.

---

## 6. Evaluate

```bash
cd /path/to/beat_docile
.venv/bin/bd eval --split val --predictions runpod_results/kile_predictions.json
```

Target: KILE AP >= 55% standalone (before ensemble with base model).

---

## Cost Estimate

| Task | Time | Cost |
|---|---|---|
| Setup + download | ~15 min | ~$0.17 |
| Training (20 epochs) | ~9 h | ~$6.20 |
| Inference val | ~30 min | ~$0.35 |
| Inference test (optional) | ~30 min | ~$0.35 |
| Buffer | ~1 h | ~$0.69 |
| **Total** | **~11 h** | **~$7.76** |

Stop the pod as soon as predictions are downloaded. Remaining budget (~$5) covers the word-merge classifier (Phase 6) if pursued.

---

## Troubleshooting

- **OOM on 5090**: Unlikely with 32 GB, but if it happens reduce `--train_bs` to 4 in `train_large.sh`.
- **Checkpoint not found after training**: Training saves every epoch; check `/workspace/checkpoints/layoutlmv3_large_ft/` for `checkpoint-*` dirs.
- **pyarrow error at startup**: The `setup.sh` patches handle this; if you see it, re-run `setup.sh`.
- **tmux session lost**: Reconnect via `ssh` then `tmux attach -t train`; the process keeps running.
