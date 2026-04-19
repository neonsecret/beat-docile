#!/usr/bin/env python
"""V5b full-test batch driver: processes all 1000 test docs.

No eval (test labels held out). Saves to predictions/v5b_test_1000.json.
Concurrency: asyncio.Semaphore(8).
Progress: every 50 docs with intermediate saves.

Usage:
    DATA_ROOT=data uv run python tools/v5b_full_test.py [--out predictions/v5b_test_1000.json]
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# ── Path / env setup ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
os.environ.setdefault("DATA_ROOT", str(DATA_DIR))
os.environ["BD_USE_REFINER"] = "1"
os.environ["BD_USE_VALIDATOR"] = "1"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from beat_docile.config import DEFAULT_MODEL, DATA_ROOT
from beat_docile.data import load_split, iter_pages
from beat_docile.extract import extract_page, extract_page_targeted
from beat_docile.fewshot import _build_cluster_index, load_few_shot_examples, build_few_shot_messages
from docile.dataset import Field

MAX_WORKERS = 8
PROGRESS_INTERVAL = 50
MODEL = DEFAULT_MODEL

OUT_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "predictions" / "v5b_test_1000.json"


def _fields_to_dicts(fields: list[Field]) -> list[dict]:
    return [f.to_dict() for f in fields]


async def run_batch() -> None:
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"Model: {MODEL}")
    print(f"Output: {OUT_PATH}")
    print(f"Concurrency: {MAX_WORKERS}")

    print("\nLoading test split (1000 docs)...")
    dataset = load_split("test")
    docs = list(dataset)
    print(f"Loaded {len(docs)} docs")

    print("Building train cluster index for few-shot...")
    train_index = _build_cluster_index("train")
    print(f"Train index: {len(train_index)} clusters")

    def _cluster_id(doc):
        try:
            return doc.annotation.cluster_id
        except (KeyError, AttributeError):
            return None

    unique_cluster_ids = list({_cluster_id(doc) for doc in docs
                                if _cluster_id(doc) is not None})
    print(f"Unique cluster IDs in test: {len(unique_cluster_ids)}")
    examples_by_cluster = load_few_shot_examples(unique_cluster_ids, train_index, max_per_cluster=1)
    few_shot_cache: dict[int, list[dict]] = {
        cid: build_few_shot_messages(examples)
        for cid, examples in examples_by_cluster.items()
    }
    print(f"Few-shot cache built for {len(few_shot_cache)} clusters")

    all_results: dict[str, list[dict]] = {}
    if OUT_PATH.exists():
        all_results = json.loads(OUT_PATH.read_text())
        print(f"Resuming — already done: {len(all_results)} docs")

    remaining = [d for d in docs if d.docid not in all_results]
    total = len(docs)
    completed = len(all_results)
    print(f"Remaining: {len(remaining)} docs\n")

    if not remaining:
        print("All docs already done — skipping extraction.")
        return

    sem = asyncio.Semaphore(MAX_WORKERS)
    errors: list[str] = []
    start_t = time.time()

    async def process_doc(doc) -> None:
        nonlocal completed
        async with sem:
            try:
                fs_messages = few_shot_cache.get(_cluster_id(doc))
                pages = list(iter_pages(doc))

                main_tasks = [extract_page(p, MODEL, few_shot_messages=fs_messages) for p in pages]
                targeted_tasks = [extract_page_targeted(p, MODEL) for p in pages]

                all_page_results = await asyncio.gather(*main_tasks, *targeted_tasks)
                n = len(pages)
                main_results = all_page_results[:n]
                targeted_results = all_page_results[n:]

                kile: list[Field] = []
                lir: list[Field] = []
                for k, l in main_results:
                    kile.extend(k)
                    lir.extend(l)
                for fields in targeted_results:
                    for f in fields:
                        if f.line_item_id is not None:
                            lir.append(f)
                        else:
                            kile.append(f)

                all_results[doc.docid] = _fields_to_dicts(kile + lir)

            except Exception as e:
                errors.append(f"{doc.docid}: {e}")
                all_results[doc.docid] = []

            completed += 1
            if completed % PROGRESS_INTERVAL == 0 or completed == total:
                elapsed = time.time() - start_t
                rate = (completed - (total - len(remaining))) / elapsed if elapsed > 0 else 0
                eta = (total - completed) / rate if rate > 0 else float("inf")
                OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
                OUT_PATH.write_text(json.dumps(all_results, indent=2))
                print(f"[{completed}/{total}] done — {elapsed:.0f}s elapsed, "
                      f"{rate:.1f} docs/s, ETA {eta:.0f}s — saved to {OUT_PATH}")

    await asyncio.gather(*[process_doc(doc) for doc in remaining])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(all_results, indent=2))
    print(f"\nExtraction complete. {len(all_results)} docs saved to {OUT_PATH}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors:
            print(f"  {e}")

    print(f"\nTest set done. No eval (labels held out). Submit {OUT_PATH} to leaderboard.")


if __name__ == "__main__":
    asyncio.run(run_batch())
