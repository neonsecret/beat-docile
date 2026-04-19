#!/usr/bin/env python
"""V5b full-val batch driver: processes all 500 val docs with BD_USE_REFINER=1 BD_USE_VALIDATOR=1.

Concurrency: asyncio.Semaphore(8) — at most 8 docs in-flight at once.
Progress: reports every 50 docs; saves intermediate results for resumability.
Eval: runs KILE AP + LIR F1 at the end. Compare to V5b-50 baseline (41.86% / 52.36%).

Usage:
    DATA_ROOT=data uv run python tools/v5b_full_val.py [--out predictions/v5b_val_500.json]
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# ── Path / env setup (before beat_docile imports) ────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
os.environ.setdefault("DATA_ROOT", str(DATA_DIR))
os.environ["BD_USE_REFINER"] = "1"
os.environ["BD_USE_VALIDATOR"] = "1"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from beat_docile.config import DEFAULT_MODEL, DATA_ROOT
from beat_docile.data import load_split, iter_pages
from beat_docile.extract import extract_page, extract_page_targeted, _TARGETED_FIELDS
from beat_docile.fewshot import _build_cluster_index, load_few_shot_examples, build_few_shot_messages
from docile.dataset import Field

MAX_WORKERS = 8
PROGRESS_INTERVAL = 50
MODEL = DEFAULT_MODEL

# ── Args ─────────────────────────────────────────────────────────────────────
OUT_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "predictions" / "v5b_val_500.json"


def _fields_to_dicts(fields: list[Field]) -> list[dict]:
    return [f.to_dict() for f in fields]


async def run_batch() -> None:
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"Model: {MODEL}")
    print(f"Output: {OUT_PATH}")
    print(f"Concurrency: {MAX_WORKERS}")

    # ── Load dataset ─────────────────────────────────────────────────────────
    print("\nLoading val split (500 docs)...")
    dataset = load_split("val")
    docs = list(dataset)
    print(f"Loaded {len(docs)} docs")

    # ── Build few-shot cache ──────────────────────────────────────────────────
    print("Building train cluster index for few-shot...")
    train_index = _build_cluster_index("train")
    print(f"Train index: {len(train_index)} clusters")

    unique_cluster_ids = list({doc.annotation.cluster_id for doc in docs
                                if doc.annotation.cluster_id is not None})
    print(f"Unique cluster IDs in val: {len(unique_cluster_ids)}")
    examples_by_cluster = load_few_shot_examples(unique_cluster_ids, train_index, max_per_cluster=1)
    few_shot_cache: dict[int, list[dict]] = {
        cid: build_few_shot_messages(examples)
        for cid, examples in examples_by_cluster.items()
    }
    print(f"Few-shot cache built for {len(few_shot_cache)} clusters")

    # ── Load intermediate results for resumability ────────────────────────────
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

    # ── Async extraction with semaphore ───────────────────────────────────────
    sem = asyncio.Semaphore(MAX_WORKERS)
    errors: list[str] = []
    start_t = time.time()

    async def process_doc(doc) -> None:
        nonlocal completed
        async with sem:
            try:
                fs_messages = few_shot_cache.get(doc.annotation.cluster_id)
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

    # ── Final save ────────────────────────────────────────────────────────────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(all_results, indent=2))
    print(f"\nExtraction complete. {len(all_results)} docs saved to {OUT_PATH}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors:
            print(f"  {e}")

    # ── Eval ─────────────────────────────────────────────────────────────────
    print("\nRunning evaluation...")
    from beat_docile.eval import run_eval, print_scores
    from docile.dataset import Dataset

    kile_preds: dict[str, list[Field]] = {}
    lir_preds: dict[str, list[Field]] = {}
    for docid, fields in all_results.items():
        kile_preds[docid] = []
        lir_preds[docid] = []
        for fd in fields:
            f = Field.from_dict(fd)
            if f.line_item_id is not None:
                lir_preds[docid].append(f)
            else:
                kile_preds[docid].append(f)

    eval_dataset = Dataset(
        split_name="val",
        dataset_path=DATA_ROOT,
        load_annotations=True,
        load_ocr=False,
    )
    result = run_eval(eval_dataset, kile_preds, lir_preds)
    scores = print_scores(result)
    print(f"\nV5b-50 baseline: KILE 41.86% / LIR 52.36%")
    print(f"V2-500 baseline: KILE 44.60% / LIR 50.90%")
    print(f"Full-val result: KILE {scores.get('kile_AP', 0)*100:.2f}% / LIR {scores.get('lir_f1', 0)*100:.2f}%")


if __name__ == "__main__":
    asyncio.run(run_batch())
