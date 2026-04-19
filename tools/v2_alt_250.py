#!/usr/bin/env python
"""v2 with alt prompt — 250-doc gate run for ensemble diversity.

Uses the first 250 docids of v2_preds.json (tools/val_250_docids.json).
Alt prompt: "For each field present in this document, select the word_ids
that exactly cover the field value — no more, no less. Return JSON only."

T=1.0 (same as v2 baseline), BD_ALT_PROMPT=1.

Usage:
    DATA_ROOT=data uv run python tools/v2_alt_250.py
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
os.environ["BD_ALT_PROMPT"] = "1"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from beat_docile.config import DEFAULT_MODEL, DATA_ROOT
from beat_docile.data import load_split, iter_pages
from beat_docile.extract import extract_page
from beat_docile.fewshot import _build_cluster_index, load_few_shot_examples, build_few_shot_messages
from docile.dataset import Field, Dataset

DOCIDS_PATH = PROJECT_ROOT / "tools" / "val_250_docids.json"
MAX_WORKERS = 8
PROGRESS_INTERVAL = 25
MODEL = DEFAULT_MODEL

OUT_PATH = PROJECT_ROOT / "predictions" / "v2_alt_250.json"


def _fields_to_dicts(fields: list[Field]) -> list[dict]:
    return [f.to_dict() for f in fields]


async def run_batch() -> None:
    target_docids: list[str] = json.loads(DOCIDS_PATH.read_text())
    print(f"Target docids: {len(target_docids)}")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"Model: {MODEL}")
    print(f"Output: {OUT_PATH}")
    print(f"Flags: BD_USE_REFINER=0 BD_TEMPERATURE=1.0 BD_ALT_PROMPT=1")
    print(f"Alt prompt: 'select word_ids exactly cover the field value — no more, no less'")

    print("\nLoading val split...")
    dataset = load_split("val")
    target_set = set(target_docids)
    docs = [d for d in dataset if d.docid in target_set]
    docs.sort(key=lambda d: target_docids.index(d.docid))
    print(f"Loaded {len(docs)} target docs")

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
                    all_page_results = await asyncio.gather(*main_tasks)
                    kile: list[Field] = []
                    lir: list[Field] = []
                    for k, l in all_page_results:
                        kile.extend(k)
                        lir.extend(l)
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
                    print(f"[{completed}/{total}] {elapsed:.0f}s elapsed, "
                          f"{rate:.1f} docs/s, ETA {eta:.0f}s — saved")

        await asyncio.gather(*[process_doc(doc) for doc in remaining])
        OUT_PATH.write_text(json.dumps(all_results, indent=2))
        if errors:
            print(f"Errors ({len(errors)}): {errors[:5]}")

    print("\nRunning evaluation...")
    from beat_docile.eval import run_eval, print_scores

    kile_preds: dict[str, list[Field]] = {}
    lir_preds: dict[str, list[Field]] = {}
    for docid, fields in all_results.items():
        kile_preds[docid] = [Field.from_dict(f) for f in fields if Field.from_dict(f).line_item_id is None]
        lir_preds[docid] = [Field.from_dict(f) for f in fields if Field.from_dict(f).line_item_id is not None]

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
    print(f"\nv2_alt_250: KILE {kile_ap:.2f}% / LIR {lir_f1:.2f}%")


if __name__ == "__main__":
    asyncio.run(run_batch())
