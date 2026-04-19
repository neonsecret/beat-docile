#!/bin/bash
# Download DocILE dataset to /workspace/data/.
# Usage: bash download_data.sh [docile_token]
# Token can also be set via DOCILE_TOKEN env var.
set -euo pipefail

TOKEN="${1:-${DOCILE_TOKEN:?Set DOCILE_TOKEN env var or pass token as first argument}}"

cd /workspace
mkdir -p data

echo "=== Downloading annotated trainval split ==="
bash /workspace/docile-repo/download_dataset.sh "${TOKEN}" annotated-trainval data/ --unzip

echo "=== Downloading test split ==="
bash /workspace/docile-repo/download_dataset.sh "${TOKEN}" test data/ --unzip

echo "=== Verifying ==="
if [ -d data/annotations ]; then
    echo "Annotations present:"
    ls data/annotations/ | head -10
else
    echo "ERROR: data/annotations/ not found. Download may have failed."
    exit 1
fi

if [ -d data/pdfs ]; then
    echo "PDFs present."
else
    echo "ERROR: data/pdfs/ not found."
    exit 1
fi

echo ""
echo "Data ready."
