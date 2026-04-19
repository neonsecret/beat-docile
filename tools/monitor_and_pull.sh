#!/bin/bash
# Polls pod training status, then pulls merged model to Mac and neon when done.
set -euo pipefail

# Configure these before running (set via env vars or edit here):
POD_IP="${POD_IP:?Set POD_IP to the RunPod IP address}"
POD_PORT="${POD_PORT:-22}"
NEON_HOST="${NEON_HOST:?Set NEON_HOST to user@neon-host}"

POD_SSH="ssh -p ${POD_PORT} -o StrictHostKeyChecking=no root@${POD_IP}"
TRAIN_PID=1772
LOCAL_DIR="$HOME/qwen3vl_docile"
NEON_DIR="${NEON_HOST}:~/qwen3vl_docile/"

echo "Monitoring training PID $TRAIN_PID on pod..."
mkdir -p "$LOCAL_DIR"

while true; do
    if ! $POD_SSH "kill -0 $TRAIN_PID 2>/dev/null"; then
        echo "[$(date)] Training finished!"
        break
    fi
    step=$($POD_SSH "grep -oP '\d+(?=/1935)' /tmp/train.log | tail -1" 2>/dev/null || echo "?")
    echo "[$(date)] Step: $step / 1935 — still running"
    sleep 300  # check every 5 minutes
done

# Wait for merge to complete (save_pretrained_merged runs after trainer.train())
echo "Waiting for merged model to appear..."
for i in $(seq 1 60); do
    if $POD_SSH "test -f /workspace/qwen3vl_docile_merged/config.json" 2>/dev/null; then
        echo "Merged model ready."
        break
    fi
    echo "  Waiting for merge... ($i)"
    sleep 30
done

echo "Pulling merged model from pod → Mac..."
rsync -avz --progress \
    -e "ssh -p ${POD_PORT} -o StrictHostKeyChecking=no" \
    root@${POD_IP}:/workspace/qwen3vl_docile_merged/ \
    "$LOCAL_DIR/"

echo "Rsyncing Mac → neon..."
rsync -avz --progress "$LOCAL_DIR/" "$NEON_DIR"

echo "All done. Stopping pod..."
$POD_SSH "halt" 2>/dev/null || true
echo "Pod stop signal sent. Total transfer complete at $(date)."
