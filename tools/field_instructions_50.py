#!/usr/bin/env python
"""A/B test: BD_USE_FIELD_INSTRUCTIONS=1 on same 50 val docs as v5b_50 baseline.

Injects ALL_FIELD_GUIDANCE (55 fields, FORMAT-only descriptions) into system prompt.
Key fix vs original -8.9pp run: amount_total_net/gross explicit coexistence rule added.

Compare to v5b_50 baseline (41.86% KILE / 52.36% LIR).

Usage:
    DATA_ROOT=data uv run python tools/field_instructions_50.py
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
os.environ["BD_USE_BBOX_VERIFY"] = "0"
os.environ["BD_USE_REFINER_GUARD"] = "1"
os.environ["BD_USE_FIELD_INSTRUCTIONS"] = "1"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from beat_docile.config import DEFAULT_MODEL, DATA_ROOT
from beat_docile.data import load_split, iter_pages
from beat_docile.extract import extract_page, extract_page_targeted
from beat_docile.fewshot import _build_cluster_index, load_few_shot_examples, build_few_shot_messages
from docile.dataset import Field

MAX_WORKERS = 8
MODEL = DEFAULT_MODEL

OUT_PATH = PROJECT_ROOT / "predictions" / "field_instructions_50.json"
V5B_50_PATH = PROJECT_ROOT / "predictions" / "v5b_50.json"

V5B_KILE = 41.86
V5B_LIR = 52.36
GUARD_KILE = 44.87  # refiner_guard_50 result
GUARD_LIR = 52.08


def _fields_to_dicts(fields: list[Field]) -> list[dict]:
    return [f.to_dict() for f in fields]


async def run_batch() -> None:
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"Model: {MODEL}")
    print(f"Output: {OUT_PATH}")
    print(f"Flags: BD_USE_REFINER=1 BD_USE_VALIDATOR=1 BD_USE_REFINER_GUARD=1 BD_USE_FIELD_INSTRUCTIONS=1")
    print(f"Baseline: V5b-50 {V5B_KILE}% KILE / {V5B_LIR}% LIR")
    print(f"Comparison: refiner_guard_50 {GUARD_KILE}% KILE / {GUARD_LIR}% LIR")

    target_docids = set(json.loads(V5B_50_PATH.read_text()).keys())
    print(f"\nTarget docids: {len(target_docids)} (same as v5b_50.json)")

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
                if completed % 10 == 0 or completed == total:
                    elapsed = time.time() - start_t
                    n_done = completed - (total - len(remaining))
                    rate = n_done / elapsed if elapsed > 0 else 0
                    eta = (total - completed) / rate if rate > 0 else float("inf")
                    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
                    OUT_PATH.write_text(json.dumps(all_results, indent=2))
                    print(f"[{completed}/{total}] {elapsed:.0f}s elapsed, "
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
        split_name="field_instructions_50",
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
    print(f"V5b-50 baseline:          KILE {V5B_KILE:.2f}% / LIR {V5B_LIR:.2f}%")
    print(f"refiner_guard_50:         KILE {GUARD_KILE:.2f}% / LIR {GUARD_LIR:.2f}%")
    print(f"field_instructions (50):  KILE {kile_ap:.2f}% / LIR {lir_f1:.2f}%")
    print(f"Delta vs V5b-50:          KILE {kile_ap - V5B_KILE:+.2f}pp / LIR {lir_f1 - V5B_LIR:+.2f}pp")
    print(f"Delta vs guard:           KILE {kile_ap - GUARD_KILE:+.2f}pp / LIR {lir_f1 - GUARD_LIR:+.2f}pp")

    if kile_ap > GUARD_KILE and lir_f1 > GUARD_LIR:
        print(f"\n✅ BEATS guard on BOTH — field instructions are additive")
    elif kile_ap > GUARD_KILE:
        print(f"\n⚠️  KILE improved over guard (+{kile_ap - GUARD_KILE:.2f}pp), LIR behind by {lir_f1 - GUARD_LIR:.2f}pp")
    elif kile_ap > V5B_KILE:
        print(f"\n⚠️  KILE beats V5b but not guard — field instructions not additive with guard")
    else:
        print(f"\n❌ KILE did not improve — instructions may be hurting via over-specification")


if __name__ == "__main__":
    asyncio.run(run_batch())
