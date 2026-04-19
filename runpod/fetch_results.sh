#!/bin/bash
# Pull large-model val predictions from a running RunPod pod to the local Mac.
# Run this FROM YOUR MAC (not from the pod).
#
# Usage:
#   bash runpod/fetch_results.sh root@<pod-ip>:<ssh-port>
#
# Example:
#   bash runpod/fetch_results.sh root@<pod-ip>:22
#   bash runpod/fetch_results.sh root@<pod-ip>:42000   # if RunPod mapped a non-standard port
set -euo pipefail

POD_HOST=${1:?Usage: fetch_results.sh user@pod-ip:port}

# Parse host and port from user@host:port
SSH_PORT="${POD_HOST##*:}"
SSH_TARGET="${POD_HOST%:*}"

DEST="$(cd "$(dirname "$0")/.." && pwd)/runpod_results"
mkdir -p "${DEST}"

echo "Pulling from ${SSH_TARGET} (port ${SSH_PORT}) -> ${DEST}/"
scp -P "${SSH_PORT}" \
    "${SSH_TARGET}:/workspace/predictions/large_val/*.json" \
    "${DEST}/"

echo ""
echo "Predictions saved to ${DEST}/"
ls "${DEST}/"
