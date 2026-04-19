#!/usr/bin/env python
"""V5b eval on 50-doc val after cache rebuild. data.py is unchanged.

Run rebuild_snap_cache_200dpi.py first, then this script.

Usage:
    DATA_ROOT=data uv run python tools/v5b_resnapped_50.py [--snap-label margin3]
"""

from __future__ import annotations

import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument("--snap-label", default="resnapped", help="Label for output filename")
args, _ = parser.parse_known_args()

from beat_docile.config import DEFAULT_MODEL, DATA_ROOT
from beat_docile.data import load_split, iter_pages
from beat_docile.extract import extract_page, extract_page_targeted
from beat_docile.fewshot import _build_cluster_index, load_few_shot_examples, build_few_shot_messages
from docile.dataset import Field

MAX_WORKERS = 8
MODEL = DEFAULT_MODEL
V5B_50_PATH = PROJECT_ROOT / "predictions" / "v5b_50.json"
OUT_PATH = PROJECT_ROOT / "predictions" / f"v5b_50_{args.snap_label}.json"
V5B_KILE, V5B_LIR = 41.79, 49.90


def _fields_to_dicts(fields): return [f.to_dict() for f in fields]


async def run_batch():
    print(f"Output: {OUT_PATH}")
    target_docids = set(json.loads(V5B_50_PATH.read_text()).keys())
    dataset = load_split("val")
    docs = [d for d in dataset if d.docid in target_docids]
    print(f"Matched {len(docs)} docs")

    train_index = _build_cluster_index("train")
    unique_cids = list({d.annotation.cluster_id for d in docs if d.annotation.cluster_id is not None})
    examples = load_few_shot_examples(unique_cids, train_index, max_per_cluster=1)
    fs_cache = {cid: build_few_shot_messages(ex) for cid, ex in examples.items()}

    all_results: dict[str, list[dict]] = {}
    if OUT_PATH.exists():
        all_results = json.loads(OUT_PATH.read_text())
        print(f"Resuming — already done: {len(all_results)}")

    remaining = [d for d in docs if d.docid not in all_results]
    total = len(docs)
    completed = len(all_results)
    sem = asyncio.Semaphore(MAX_WORKERS)
    errors = []
    start_t = time.time()

    async def process_doc(doc):
        nonlocal completed
        async with sem:
            try:
                fs_messages = fs_cache.get(doc.annotation.cluster_id)
                pages = list(iter_pages(doc))
                main_tasks = [extract_page(p, MODEL, few_shot_messages=fs_messages) for p in pages]
                targeted_tasks = [extract_page_targeted(p, MODEL) for p in pages]
                all_page_results = await asyncio.gather(*main_tasks, *targeted_tasks)
                n = len(pages)
                kile, lir = [], []
                for k, l in all_page_results[:n]:
                    kile.extend(k); lir.extend(l)
                for fields in all_page_results[n:]:
                    for f in fields:
                        (lir if f.line_item_id is not None else kile).append(f)
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
                print(f"[{completed}/{total}] {elapsed:.0f}s, ETA {eta:.0f}s")

    await asyncio.gather(*[process_doc(d) for d in remaining])
    OUT_PATH.write_text(json.dumps(all_results, indent=2))
    if errors:
        print(f"Errors: {errors}")

    from beat_docile.eval import run_eval, print_scores
    from docile.dataset import Dataset
    kile_preds, lir_preds = {}, {}
    for docid, fields in all_results.items():
        kile_preds[docid] = []; lir_preds[docid] = []
        for fd in fields:
            f = Field.from_dict(fd)
            (lir_preds if f.line_item_id is not None else kile_preds)[docid].append(f)
    eval_dataset = Dataset(split_name=f"resnapped_50_{args.snap_label}", dataset_path=DATA_ROOT,
                           load_annotations=True, load_ocr=False, docids=list(all_results.keys()))
    result = run_eval(eval_dataset, kile_preds, lir_preds)
    scores = print_scores(result)
    kile_ap = scores.get("kile_AP", 0) * 100
    lir_f1 = scores.get("lir_f1", 0) * 100
    print(f"\n{'='*60}")
    print(f"V5b baseline (200 DPI margin=6): KILE {V5B_KILE:.2f}% / LIR {V5B_LIR:.2f}%")
    print(f"200 DPI {args.snap_label}:       KILE {kile_ap:.2f}% / LIR {lir_f1:.2f}%")
    print(f"Delta:                           KILE {kile_ap-V5B_KILE:+.2f}pp / LIR {lir_f1-V5B_LIR:+.2f}pp")


if __name__ == "__main__":
    asyncio.run(run_batch())
