#!/usr/bin/env python
"""Phase 1 eval: 300 DPI snap fix vs V5b baseline on same 50 val docs.

Runs identical pipeline to V5b (BD_USE_REFINER=1 BD_USE_VALIDATOR=1) but with
data.py patched to render at 300 DPI and skip the 150-DPI snap cache.

Saves to predictions/v5b_50_dpi300.json, then evals KILE AP / LIR F1.

Usage:
    DATA_ROOT=data uv run python tools/v5b_dpi300_50.py
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

from beat_docile.config import DEFAULT_MODEL, DATA_ROOT
from beat_docile.data import load_split, iter_pages
from beat_docile.extract import extract_page, extract_page_targeted
from beat_docile.fewshot import _build_cluster_index, load_few_shot_examples, build_few_shot_messages
from docile.dataset import Field

MAX_WORKERS = 8
MODEL = DEFAULT_MODEL

V5B_50_PATH = PROJECT_ROOT / "predictions" / "v5b_50.json"
OUT_PATH = PROJECT_ROOT / "predictions" / "v5b_50_dpi300.json"

V5B_KILE = 41.79
V5B_LIR = 49.90


def _fields_to_dicts(fields: list[Field]) -> list[dict]:
    return [f.to_dict() for f in fields]


async def run_batch() -> None:
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"Model: {MODEL}")
    print(f"Output: {OUT_PATH}")
    print(f"BD_USE_REFINER=1 BD_USE_VALIDATOR=1")
    print(f"data.py: 300 DPI snap, use_cached_snapping=False")

    target_docids = set(json.loads(V5B_50_PATH.read_text()).keys())
    print(f"\nTarget docids: {len(target_docids)} (from v5b_50.json)")

    print("Loading val split...")
    dataset = load_split("val")
    docs = [d for d in dataset if d.docid in target_docids]
    print(f"Matched {len(docs)} docs")

    print("Building train cluster index for few-shot...")
    train_index = _build_cluster_index("train")
    unique_cluster_ids = list({doc.annotation.cluster_id for doc in docs
                                if doc.annotation.cluster_id is not None})
    examples_by_cluster = load_few_shot_examples(unique_cluster_ids, train_index, max_per_cluster=1)
    few_shot_cache: dict[int, list[dict]] = {
        cid: build_few_shot_messages(examples)
        for cid, examples in examples_by_cluster.items()
    }
    print(f"Few-shot cache: {len(few_shot_cache)} clusters")

    all_results: dict[str, list[dict]] = {}
    if OUT_PATH.exists():
        all_results = json.loads(OUT_PATH.read_text())
        print(f"Resuming — already done: {len(all_results)} docs")

    remaining = [d for d in docs if d.docid not in all_results]
    total = len(docs)
    completed = len(all_results)
    print(f"Remaining: {len(remaining)} docs\n")

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
            if completed % 10 == 0 or completed == total:
                elapsed = time.time() - start_t
                n_done = completed - (total - len(remaining))
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (total - completed) / rate if rate > 0 else float("inf")
                OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
                OUT_PATH.write_text(json.dumps(all_results, indent=2))
                print(f"[{completed}/{total}] done — {elapsed:.0f}s elapsed, "
                      f"{rate:.1f} docs/s, ETA {eta:.0f}s")

    await asyncio.gather(*[process_doc(doc) for doc in remaining])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(all_results, indent=2))
    print(f"\nExtraction complete. {len(all_results)} docs saved to {OUT_PATH}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors:
            print(f"  {e}")

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
        split_name="dpi300_50",
        dataset_path=DATA_ROOT,
        load_annotations=True,
        load_ocr=False,
        docids=list(all_results.keys()),
    )
    result = run_eval(eval_dataset, kile_preds, lir_preds)
    scores = print_scores(result)

    kile_ap = scores.get("kile_AP", 0) * 100
    lir_f1 = scores.get("lir_f1", 0) * 100
    print(f"\n{'='*60}")
    print(f"V5b baseline (150 DPI):  KILE AP {V5B_KILE:.2f}% / LIR F1 {V5B_LIR:.2f}%")
    print(f"V5b + 300 DPI snap:      KILE AP {kile_ap:.2f}% / LIR F1 {lir_f1:.2f}%")
    print(f"Delta:                   KILE {kile_ap - V5B_KILE:+.2f}pp / LIR {lir_f1 - V5B_LIR:+.2f}pp")


if __name__ == "__main__":
    asyncio.run(run_batch())
