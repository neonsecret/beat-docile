#!/usr/bin/env python
"""Re-run V5b extraction on test set using Qwen3-VL inferred cluster IDs.

Loads predictions/test_inferred_clusters_qwen3vl.json (or --cluster-map arg),
passes cluster_override to extract_documents, saves to
predictions/v5b_test_1000_qwen3vl.json.

Usage:
    DATA_ROOT=data uv run python tools/v5b_test_with_inferred_clusters.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
os.environ.setdefault("DATA_ROOT", str(DATA_DIR))
os.environ["BD_USE_REFINER"] = "1"
os.environ["BD_USE_VALIDATOR"] = "1"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from beat_docile.config import DEFAULT_MODEL
from beat_docile.data import load_split
from beat_docile.extract import extract_documents
from beat_docile.fewshot import _build_cluster_index

CLUSTER_MAP_PATH = PROJECT_ROOT / "predictions" / "test_inferred_clusters_qwen3vl.json"
OUT_PATH = PROJECT_ROOT / "predictions" / "v5b_test_1000_qwen3vl.json"
MODEL = DEFAULT_MODEL
CONCURRENCY = 8


async def main() -> None:
    if not CLUSTER_MAP_PATH.exists():
        print(f"ERROR: cluster map not found: {CLUSTER_MAP_PATH}")
        sys.exit(1)

    cluster_override: dict[str, int] = json.loads(CLUSTER_MAP_PATH.read_text())
    print(f"Loaded cluster map for {len(cluster_override)} docs from {CLUSTER_MAP_PATH}")

    print("Loading test split…")
    dataset = load_split("test")
    docs = list(dataset)
    print(f"Loaded {len(docs)} test docs")

    print("Building train cluster index for few-shot…")
    train_index = _build_cluster_index("train")
    print(f"Train index: {len(train_index)} clusters")

    # Resume support
    all_results: dict[str, list[dict]] = {}
    if OUT_PATH.exists():
        all_results = json.loads(OUT_PATH.read_text())
        print(f"Resuming — already done: {len(all_results)} docs")

    remaining = [d for d in docs if d.docid not in all_results]
    print(f"Remaining: {len(remaining)} docs\n")

    if not remaining:
        print("All docs already done.")
        return

    t0 = time.time()
    sem = asyncio.Semaphore(CONCURRENCY)
    completed = len(all_results)
    total = len(docs)
    errors: list[str] = []

    async def process_doc(doc) -> None:
        nonlocal completed
        async with sem:
            try:
                kile, lir = await extract_documents(
                    [doc],
                    MODEL,
                    train_index=train_index,
                    targeted_pass=True,
                    cluster_override=cluster_override,
                )
                fields = kile.get(doc.docid, []) + lir.get(doc.docid, [])
                all_results[doc.docid] = [f.to_dict() for f in fields]
            except Exception as e:
                errors.append(f"{doc.docid}: {e}")
                all_results[doc.docid] = []

            completed += 1
            if completed % 50 == 0 or completed == total:
                elapsed = time.time() - t0
                done_this_run = completed - (total - len(remaining))
                rate = done_this_run / elapsed if elapsed > 0 else 0
                eta = (total - completed) / rate if rate > 0 else float("inf")
                OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
                OUT_PATH.write_text(json.dumps(all_results, indent=2))
                print(
                    f"[{completed}/{total}] {elapsed:.0f}s elapsed, "
                    f"{rate:.2f} docs/s, ETA {eta:.0f}s — saved"
                )

    await asyncio.gather(*[process_doc(d) for d in remaining])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(all_results, indent=2))
    print(f"\nDone. {len(all_results)} docs → {OUT_PATH}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"  {e}")


if __name__ == "__main__":
    asyncio.run(main())
