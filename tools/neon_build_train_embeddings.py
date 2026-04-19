#!/usr/bin/env python3
"""Build Qwen3-VL-Embedding-2B train embeddings on neon (RTX 3070, CUDA).

Run on neon:
    python3 ~/beat_docile/tools/neon_build_train_embeddings.py

Output: ~/beat_docile/models/qwen3vl_train_embeddings.npz

Rsync to Mac:
    scp <neon-user>@<neon-host>:~/beat_docile/models/qwen3vl_train_embeddings.npz \\
        ~/projects/beat_docile/models/qwen3vl_train_embeddings.npz
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("neon_embed")

DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path.home() / "beat_docile" / "data"))
OUTPUT_PATH = Path.home() / "beat_docile" / "models" / "qwen3vl_train_embeddings.npz"
MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
DEVICE = "cuda"

# Add src to path so beat_docile is importable
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))


def main() -> None:
    import numpy as np
    from PIL import ImageFile
    from sentence_transformers import SentenceTransformer

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    logger.info("DATA_ROOT: %s  device: %s", DATA_ROOT, DEVICE)
    logger.info("Loading %s ...", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=DEVICE)
    logger.info("Model loaded.")

    index_file = DATA_ROOT / "train.json"
    with open(index_file) as f:
        all_train_docids = json.load(f)
    logger.info("Train split: %d docs", len(all_train_docids))

    ann_dir = DATA_ROOT / "annotations"
    pdf_dir = DATA_ROOT / "pdfs"
    available = [
        d for d in all_train_docids
        if (ann_dir / f"{d}.json").exists() and (pdf_dir / f"{d}.pdf").exists()
    ]
    logger.info("Available (ann + pdf): %d", len(available))

    from docile.dataset import Dataset
    logger.info("Loading Dataset for all %d docs...", len(available))
    ds = Dataset("smoke_subset", DATA_ROOT, load_annotations=True,
                 load_ocr=False, docids=available)
    docs = list(ds)
    logger.info("Dataset loaded: %d docs", len(docs))

    embeddings: list[np.ndarray] = []
    cluster_ids: list[int] = []
    docids: list[str] = []
    n_failed = 0
    t0 = time.monotonic()

    for i, doc in enumerate(docs):
        try:
            with doc:
                cid = doc.annotation.cluster_id
                image = doc.page_image(0)

            emb = model.encode([image], convert_to_numpy=True, show_progress_bar=False)[0]
            emb = emb.astype(np.float32)
            emb /= (np.linalg.norm(emb) + 1e-8)

            embeddings.append(emb)
            cluster_ids.append(int(cid))
            docids.append(doc.docid)

        except Exception as exc:
            logger.warning("Failed %s: %s", getattr(doc, 'docid', '?'), exc)
            n_failed += 1

        if (i + 1) % 50 == 0:
            elapsed = time.monotonic() - t0
            rate = (i + 1) / elapsed
            eta = (len(docs) - i - 1) / max(rate, 0.001)
            logger.info("Progress: %d/%d (%.2f/s, ETA %.0fmin)",
                        i + 1, len(docs), rate, eta / 60)

    matrix = np.stack(embeddings).astype(np.float32)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT_PATH,
             embeddings=matrix,
             cluster_ids=np.array(cluster_ids, dtype=np.int32),
             docids=np.array(docids, dtype=object))
    elapsed = time.monotonic() - t0
    logger.info("Done. n=%d failed=%d dim=%d elapsed=%.0fs size=%.1fMB",
                len(embeddings), n_failed, matrix.shape[1], elapsed,
                OUTPUT_PATH.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
