#!/usr/bin/env python
"""Gemini 3 Flash extraction — 250-doc gate run for ensemble diversity.

Runs gemini-3-flash-preview on the first 250 val docids (tools/val_250_docids.json).
Saves predictions/v2_gemini_250.json, then scores and prints results.

Usage:
    DATA_ROOT=data uv run python tools/run_gemini_250.py
    BD_GEMINI_WORKERS=4 DATA_ROOT=data uv run python tools/run_gemini_250.py  # if rate limited
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
os.environ["BD_USE_REFINER"] = "0"
os.environ["BD_USE_VALIDATOR"] = "0"
os.environ["BD_USE_BBOX_VERIFY"] = "0"
os.environ["BD_USE_REFINER_GUARD"] = "0"
os.environ["BD_TEMPERATURE"] = "1.0"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from beat_docile.config import DATA_ROOT
from beat_docile.data import load_split, iter_pages
from beat_docile.fewshot import _build_cluster_index, load_few_shot_examples
from beat_docile.gemini_extract import extract_page_gemini, extract_page_targeted_gemini, _GEMINI_MODEL
from docile.dataset import Field, Dataset

DOCIDS_PATH = PROJECT_ROOT / "tools" / "val_250_docids.json"
MAX_WORKERS = int(os.environ.get("BD_GEMINI_WORKERS", "6"))
PROGRESS_INTERVAL = 10
MODEL = _GEMINI_MODEL

OUT_PATH = PROJECT_ROOT / "predictions" / "v2_gemini_250.json"


def _fields_to_dicts(fields: list[Field]) -> list[dict]:
    return [f.to_dict() for f in fields]


async def run_batch() -> None:
    target_docids: list[str] = json.loads(DOCIDS_PATH.read_text())
    print(f"Target docids: {len(target_docids)}")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"Model: {MODEL}")
    print(f"Output: {OUT_PATH}")
    print(f"MAX_WORKERS: {MAX_WORKERS}")

    print("\nLoading val split...")
    dataset = load_split("val")
    target_set = set(target_docids)
    docs = [d for d in dataset if d.docid in target_set]
    docs.sort(key=lambda d: target_docids.index(d.docid))
    print(f"Loaded {len(docs)} target docs")

    print("Building train cluster index for few-shot...")
    train_index = _build_cluster_index("train")

    unique_cids = list({doc.annotation.cluster_id for doc in docs
                        if doc.annotation.cluster_id is not None})
    examples_by_cluster = load_few_shot_examples(unique_cids, train_index, max_per_cluster=1)
    few_shot_cache = dict(examples_by_cluster)
    print(f"Few-shot cache: {len(few_shot_cache)} clusters")

    all_results: dict[str, list[dict]] = {}
    if OUT_PATH.exists():
        all_results = json.loads(OUT_PATH.read_text())
        print(f"Resuming — already done: {len(all_results)} docs")

    remaining = [d for d in docs if d.docid not in all_results]
    total = len(docs)
    completed = len(all_results)
    print(f"Remaining: {len(remaining)} docs\n")

    if not remaining:
        print("All docs already done — running eval only.")
    else:
        sem = asyncio.Semaphore(MAX_WORKERS)
        errors: list[str] = []
        start_t = time.time()

        async def process_doc(doc) -> None:
            nonlocal completed
            async with sem:
                try:
                    fs_examples = few_shot_cache.get(doc.annotation.cluster_id)
                    pages = list(iter_pages(doc))
                    main_tasks = [
                        extract_page_gemini(page, model=MODEL, few_shot_examples=fs_examples)
                        for page in pages
                    ]
                    targeted_tasks = [
                        extract_page_targeted_gemini(page, model=MODEL) for page in pages
                    ]
                    all_page_results = await asyncio.gather(*main_tasks, *targeted_tasks)
                    n = len(pages)
                    kile: list[Field] = []
                    lir: list[Field] = []
                    for k, l in all_page_results[:n]:
                        kile.extend(k)
                        lir.extend(l)
                    for fields in all_page_results[n:]:
                        for f in fields:
                            if f.line_item_id is not None:
                                lir.append(f)
                            else:
                                kile.append(f)
                    all_results[doc.docid] = _fields_to_dicts(kile + lir)
                except Exception as e:
                    errors.append(f"{doc.docid}: {e}")
                    print(f"  ERROR {doc.docid}: {e}")
                    all_results[doc.docid] = []

                completed += 1
                if completed % PROGRESS_INTERVAL == 0 or completed == total:
                    elapsed = time.time() - start_t
                    n_done = completed - (total - len(remaining))
                    rate = n_done / elapsed if elapsed > 0 else 0
                    eta = (total - completed) / rate if rate > 0 else float("inf")
                    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
                    OUT_PATH.write_text(json.dumps(all_results, indent=2))
                    print(f"[{completed}/{total}] {elapsed:.0f}s elapsed, "
                          f"{rate:.2f} docs/s, ETA {eta:.0f}s — saved")

        await asyncio.gather(*[process_doc(doc) for doc in remaining])
        OUT_PATH.write_text(json.dumps(all_results, indent=2))
        if errors:
            print(f"\nErrors ({len(errors)}): {errors[:10]}")

    print("\nRunning evaluation (Gemini-only on 250 docs)...")
    from beat_docile.eval import run_eval, print_scores

    kile_preds: dict[str, list[Field]] = {}
    lir_preds: dict[str, list[Field]] = {}
    for docid, fields in all_results.items():
        parsed = [Field.from_dict(f) for f in fields]
        kile_preds[docid] = [f for f in parsed if f.line_item_id is None]
        lir_preds[docid] = [f for f in parsed if f.line_item_id is not None]

    eval_dataset = Dataset(
        split_name="v2_250_gate",
        dataset_path=DATA_ROOT,
        load_annotations=True,
        load_ocr=False,
        docids=target_docids,
    )
    result = run_eval(eval_dataset, kile_preds, lir_preds)
    scores = print_scores(result)
    kile_ap = scores.get("kile_AP", 0) * 100
    lir_f1 = scores.get("lir_f1", 0) * 100
    print(f"\nv2_gemini_250 (Gemini-only): KILE {kile_ap:.2f}% / LIR {lir_f1:.2f}%")


if __name__ == "__main__":
    asyncio.run(run_batch())
