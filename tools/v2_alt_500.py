#!/usr/bin/env python
"""v2 with alt prompt — 500-doc val run for ensemble diversity.

Identical to v2 (BD_USE_REFINER=0, T=1.0, few-shot) but with a rephrased
task instruction: "For each field present in this document, select the word_ids
that exactly cover the field value — no more, no less. Return JSON only."

Hypothesis: emphasizing "no more, no less" might improve precision on
multi-word fields where Claude sometimes includes neighboring words.

v2 baseline (T=1.0): KILE 44.61% / LIR 50.89%

Usage:
    DATA_ROOT=data uv run python tools/v2_alt_500.py
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
from docile.dataset import Field

MAX_WORKERS = 8
PROGRESS_INTERVAL = 50
MODEL = DEFAULT_MODEL

OUT_PATH = PROJECT_ROOT / "predictions" / "v2_alt_500.json"

V2_500_KILE = 44.61
V2_500_LIR = 50.89


def _fields_to_dicts(fields: list[Field]) -> list[dict]:
    return [f.to_dict() for f in fields]


async def run_batch() -> None:
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"Model: {MODEL}")
    print(f"Output: {OUT_PATH}")
    print(f"Flags: BD_USE_REFINER=0 BD_TEMPERATURE=1.0 BD_ALT_PROMPT=1")
    print(f"Alt prompt: 'select word_ids that exactly cover the field value — no more, no less'")
    print(f"Baseline: v2-500 ({V2_500_KILE}% KILE / {V2_500_LIR}% LIR)")

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
                          f"{rate:.1f} docs/s, ETA {eta:.0f}s — saved {OUT_PATH.name}")

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
    print(f"v2-500 baseline:     KILE {V2_500_KILE:.2f}% / LIR {V2_500_LIR:.2f}%")
    print(f"v2_alt_500 (this):   KILE {kile_ap:.2f}% / LIR {lir_f1:.2f}%")
    print(f"Delta vs v2:         KILE {kile_ap - V2_500_KILE:+.2f}pp / LIR {lir_f1 - V2_500_LIR:+.2f}pp")


if __name__ == "__main__":
    asyncio.run(run_batch())
