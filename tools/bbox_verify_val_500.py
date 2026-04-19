#!/usr/bin/env python
"""500-doc bbox_verify run: BD_USE_BBOX_VERIFY=1 + BD_USE_REFINER=1 + BD_USE_VALIDATOR=1.

Tests whether bbox_verify lifts V5b to beat v2 baseline (44.61% / 50.89%) on full val set.
Monkey-patches verify_bbox to count corrections and LLM fallback triggers.
Saves to predictions/bbox_verify_val_500.json, then evals KILE AP / LIR F1.

Usage:
    DATA_ROOT=data uv run python tools/bbox_verify_val_500.py
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
os.environ["BD_USE_BBOX_VERIFY"] = "1"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Monkey-patch verify_bbox to count corrections BEFORE any extract import
import beat_docile.bbox_verify as bv_module

_corrected_count = 0
_total_verify_calls = 0
_llm_fallback_count = 0
_original_verify_bbox = bv_module.verify_bbox


def _patched_verify_bbox(*args, **kwargs):
    global _corrected_count, _total_verify_calls, _llm_fallback_count
    result = _original_verify_bbox(*args, **kwargs)
    _total_verify_calls += 1
    if result.corrected:
        _corrected_count += 1
    return result


bv_module.verify_bbox = _patched_verify_bbox

from beat_docile.config import DEFAULT_MODEL, DATA_ROOT
from beat_docile.data import load_split, iter_pages
from beat_docile.extract import extract_page, extract_page_targeted
from beat_docile.fewshot import _build_cluster_index, load_few_shot_examples, build_few_shot_messages
from docile.dataset import Field

MAX_WORKERS = 8
PROGRESS_INTERVAL = 50
MODEL = DEFAULT_MODEL

OUT_PATH = PROJECT_ROOT / "predictions" / "bbox_verify_val_500.json"

V5B_500_KILE = 41.79
V5B_500_LIR = 49.90
V2_500_KILE = 44.61
V2_500_LIR = 50.89


def _fields_to_dicts(fields: list[Field]) -> list[dict]:
    return [f.to_dict() for f in fields]


async def run_batch() -> None:
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"Model: {MODEL}")
    print(f"Output: {OUT_PATH}")
    print(f"Flags: BD_USE_BBOX_VERIFY=1 BD_USE_REFINER=1 BD_USE_VALIDATOR=1")
    print(f"Baselines: V5b-500 {V5B_500_KILE}% KILE / {V5B_500_LIR}% LIR | v2-500 {V2_500_KILE}% KILE / {V2_500_LIR}% LIR")

    print("\nLoading val split (500 docs)...")
    dataset = load_split("val")
    docs = list(dataset)
    print(f"Loaded {len(docs)} docs")

    print("Building train cluster index for few-shot...")
    train_index = _build_cluster_index("train")
    print(f"Train index: {len(train_index)} clusters")

    unique_cluster_ids = list({doc.annotation.cluster_id for doc in docs
                                if doc.annotation.cluster_id is not None})
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
        print("All docs already done — running eval only.")
    else:
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
                    n_done = completed - (total - len(remaining))
                    rate = n_done / elapsed if elapsed > 0 else 0
                    eta = (total - completed) / rate if rate > 0 else float("inf")
                    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
                    OUT_PATH.write_text(json.dumps(all_results, indent=2))
                    corr_pct = (100 * _corrected_count / _total_verify_calls) if _total_verify_calls else 0
                    print(f"[{completed}/{total}] {elapsed:.0f}s elapsed, {rate:.1f} docs/s, "
                          f"ETA {eta:.0f}s | verify: {_total_verify_calls} calls, "
                          f"{_corrected_count} corrected ({corr_pct:.1f}%)")

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
        split_name="val",
        dataset_path=DATA_ROOT,
        load_annotations=True,
        load_ocr=False,
    )
    result = run_eval(eval_dataset, kile_preds, lir_preds)
    scores = print_scores(result)

    kile_ap = scores.get("kile_AP", 0) * 100
    lir_f1 = scores.get("lir_f1", 0) * 100

    print(f"\n{'='*60}")
    print(f"V5b-500 baseline:      KILE {V5B_500_KILE:.2f}% / LIR {V5B_500_LIR:.2f}%")
    print(f"v2-500 baseline:       KILE {V2_500_KILE:.2f}% / LIR {V2_500_LIR:.2f}%")
    print(f"bbox_verify (this):    KILE {kile_ap:.2f}% / LIR {lir_f1:.2f}%")
    print(f"Delta vs V5b-500:      KILE {kile_ap - V5B_500_KILE:+.2f}pp / LIR {lir_f1 - V5B_500_LIR:+.2f}pp")
    print(f"Delta vs v2-500:       KILE {kile_ap - V2_500_KILE:+.2f}pp / LIR {lir_f1 - V2_500_LIR:+.2f}pp")
    corr_pct = (100 * _corrected_count / _total_verify_calls) if _total_verify_calls else 0
    print(f"verify_bbox stats:     {_total_verify_calls} calls, {_corrected_count} corrected ({corr_pct:.1f}%)")

    if kile_ap > V2_500_KILE:
        print(f"\n✅ BEATS v2 baseline by {kile_ap - V2_500_KILE:+.2f}pp KILE — APPLY")
    elif kile_ap > V5B_500_KILE:
        print(f"\n⚠️  Beats V5b but not v2 — partial win ({kile_ap - V5B_500_KILE:+.2f}pp vs V5b)")
    else:
        print(f"\n❌ Does not beat V5b-500 — BURY")


if __name__ == "__main__":
    asyncio.run(run_batch())
