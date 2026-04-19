#!/usr/bin/env python
"""LIR-only eval: compare RapidTable-based LIR extraction vs V5b baseline.

Runs on the same 50 val docs as v5b_50.json.
Keeps V5b KILE fields unchanged; replaces V5b LIR with RapidTable LIR.
Falls back to V5b LIR for docs where RapidTable finds no tables.

Usage:
    DATA_ROOT=data uv run python tools/eval_lir.py [--docs 5] [--output predictions/lir_tabled_50.json]
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

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from beat_docile.config import DEFAULT_MODEL, DATA_ROOT
from beat_docile.data import load_split, iter_pages
from beat_docile.tabled_lir import extract_lir_for_doc
from docile.dataset import Dataset, Field


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RapidTable LIR eval")
    p.add_argument("--docs", type=int, default=50, help="Number of docs to eval (default 50)")
    p.add_argument("--output", default="predictions/lir_tabled_50.json", help="Output JSON path")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Claude model for column classification")
    return p.parse_args()


async def run_eval(args: argparse.Namespace) -> None:
    # Load 50-doc val set docids from existing v5b baseline
    v5b_path = PROJECT_ROOT / "predictions" / "v5b_50.json"
    if not v5b_path.exists():
        print(f"ERROR: {v5b_path} not found — run v5b_bbox_50.py first")
        sys.exit(1)

    v5b_data: dict[str, list[dict]] = json.loads(v5b_path.read_text())
    target_docids = list(v5b_data.keys())[: args.docs]
    print(f"Target docs: {len(target_docids)}")

    # Split V5b fields into KILE and LIR per doc
    v5b_kile: dict[str, list[dict]] = {}
    v5b_lir: dict[str, list[dict]] = {}
    for docid, fields in v5b_data.items():
        v5b_kile[docid] = [f for f in fields if f.get("line_item_id") is None]
        v5b_lir[docid] = [f for f in fields if f.get("line_item_id") is not None]

    print("Loading val split...")
    dataset = load_split("val")
    docs = [d for d in dataset if d.docid in set(target_docids)]
    doc_map = {d.docid: d for d in docs}
    docs = [doc_map[did] for did in target_docids if did in doc_map]
    print(f"Matched {len(docs)} docs")

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resumability
    all_results: dict[str, list[dict]] = {}
    if out_path.exists():
        all_results = json.loads(out_path.read_text())
        print(f"Resuming — already done: {len(all_results)} docs")

    remaining = [d for d in docs if d.docid not in all_results]
    print(f"Remaining: {len(remaining)} docs\n")

    errors: list[str] = []
    start_t = time.time()
    stats = {"tabled_used": 0, "v5b_fallback": 0, "total_lir_fields": 0}

    for i, doc in enumerate(remaining):
        docid = doc.docid
        try:
            pages = list(iter_pages(doc))
            tabled_lir = await extract_lir_for_doc(pages, args.model)

            if tabled_lir:
                lir_fields = [f.to_dict() for f in tabled_lir]
                stats["tabled_used"] += 1
            else:
                # Fallback: use V5b LIR for this doc
                lir_fields = v5b_lir.get(docid, [])
                stats["v5b_fallback"] += 1

            stats["total_lir_fields"] += len(lir_fields)
            kile_fields = v5b_kile.get(docid, [])
            all_results[docid] = kile_fields + lir_fields

        except Exception as e:
            errors.append(f"{docid}: {e}")
            # On error: use full V5b for this doc
            all_results[docid] = v5b_data.get(docid, [])

        completed = len(all_results) - (len(docs) - len(remaining))
        if (i + 1) % 5 == 0 or (i + 1) == len(remaining):
            elapsed = time.time() - start_t
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(remaining) - i - 1) / rate if rate > 0 else 0
            out_path.write_text(json.dumps(all_results, indent=2))
            print(
                f"[{i+1}/{len(remaining)}] elapsed={elapsed:.0f}s rate={rate:.2f}/s eta={eta:.0f}s "
                f"| tabled={stats['tabled_used']} fallback={stats['v5b_fallback']}"
            )

    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nExtraction done. {len(all_results)} docs saved to {out_path}")
    print(f"Stats: tabled_used={stats['tabled_used']} v5b_fallback={stats['v5b_fallback']} "
          f"total_lir_fields={stats['total_lir_fields']}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors:
            print(f"  {e}")

    # Evaluate
    print("\nRunning evaluation...")
    from beat_docile.eval import run_eval as do_eval, print_scores

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
        split_name="lir_tabled_50",
        dataset_path=DATA_ROOT,
        load_annotations=True,
        load_ocr=False,
        docids=list(all_results.keys()),
    )
    result = do_eval(eval_dataset, kile_preds, lir_preds)
    scores = print_scores(result)

    kile_ap = scores.get("kile_AP", 0) * 100
    lir_f1 = scores.get("lir_f1", 0) * 100
    print(f"\n{'='*55}")
    print(f"V5b baseline:             KILE AP 41.79% / LIR F1 49.90%")
    print(f"RapidTable LIR:           KILE AP {kile_ap:.2f}% / LIR F1 {lir_f1:.2f}%")
    print(f"LIR delta:                {lir_f1 - 49.90:+.2f}pp")
    print(f"{'='*55}")

    # Per-fieldtype LIR breakdown
    print("\nPer-fieldtype LIR F1:")
    for ft in sorted(_lir_types()):
        try:
            ft_metrics = result.get_metrics("lir", fieldtype=ft)
            f1 = ft_metrics.get("f1", 0) * 100
            tp = ft_metrics.get("TP", 0)
            print(f"  {ft:<40} F1={f1:.1f}% TP={tp}")
        except Exception:
            pass


def _lir_types():
    from beat_docile.extract import _LIR_TYPES
    return _LIR_TYPES


if __name__ == "__main__":
    asyncio.run(run_eval(_parse_args()))
