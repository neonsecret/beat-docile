#!/usr/bin/env python
"""Phase 4b eval: YOLOv12s Table regions → RapidTable crop → LIR F1.

Detects Table regions per page with YOLOv12s-DocLayNet, crops to those regions,
runs RapidTable SLANETPLUS on the crop only (not full page), then emits LIR fields.

Keeps V5b KILE fields unchanged. Falls back to V5b LIR for pages with no Table regions.

Usage:
    DATA_ROOT=data uv run python tools/eval_lir_chained.py [--docs 50]
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
from beat_docile.layout_regions import detect_regions
from beat_docile.tabled_lir import extract_lir_chained
from docile.dataset import Dataset, Field


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--docs", type=int, default=50)
    p.add_argument("--output", default="predictions/lir_chained_50.json")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--conf", type=float, default=0.3, help="YOLO confidence threshold")
    return p.parse_args()


async def run(args: argparse.Namespace) -> None:
    v5b_path = PROJECT_ROOT / "predictions" / "v5b_50.json"
    v5b_data: dict[str, list[dict]] = json.loads(v5b_path.read_text())
    target_docids = list(v5b_data.keys())[: args.docs]
    print(f"Target docs: {len(target_docids)}")

    v5b_kile: dict[str, list[dict]] = {}
    v5b_lir: dict[str, list[dict]] = {}
    for docid, fields in v5b_data.items():
        v5b_kile[docid] = [f for f in fields if f.get("line_item_id") is None]
        v5b_lir[docid] = [f for f in fields if f.get("line_item_id") is not None]

    print("Loading val split...")
    dataset = load_split("val")
    doc_map = {d.docid: d for d in dataset if d.docid in set(target_docids)}
    docs = [doc_map[did] for did in target_docids if did in doc_map]
    print(f"Matched {len(docs)} docs")

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[dict]] = {}
    if out_path.exists():
        all_results = json.loads(out_path.read_text())
        print(f"Resuming — already done: {len(all_results)} docs")

    remaining = [d for d in docs if d.docid not in all_results]
    print(f"Remaining: {len(remaining)} docs\n")

    errors: list[str] = []
    start_t = time.time()
    stats = {"chained": 0, "fallback": 0, "table_regions_total": 0}

    for i, doc in enumerate(remaining):
        docid = doc.docid
        try:
            pages = list(iter_pages(doc))

            # Collect LIR per page using chained pipeline
            doc_lir: list[dict] = []
            chained_any = False

            for page in pages:
                # Detect Table regions with YOLOv12s
                regions = detect_regions(page.image, conf=args.conf)
                table_regions = [r for r in regions if r["label"] == "Table"]
                stats["table_regions_total"] += len(table_regions)

                if table_regions:
                    page_lir = await extract_lir_chained(page, table_regions, args.model)
                    if page_lir:
                        doc_lir.extend(f.to_dict() for f in page_lir)
                        chained_any = True

            if chained_any:
                stats["chained"] += 1
            else:
                # Fallback: use V5b LIR for this doc
                doc_lir = v5b_lir.get(docid, [])
                stats["fallback"] += 1

            all_results[docid] = v5b_kile.get(docid, []) + doc_lir

        except Exception as e:
            errors.append(f"{docid}: {e}")
            all_results[docid] = v5b_data.get(docid, [])

        if (i + 1) % 5 == 0 or (i + 1) == len(remaining):
            elapsed = time.time() - start_t
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(remaining) - i - 1) / rate if rate > 0 else 0
            out_path.write_text(json.dumps(all_results, indent=2))
            print(
                f"[{i+1}/{len(remaining)}] {elapsed:.0f}s rate={rate:.2f}/s eta={eta:.0f}s "
                f"| chained={stats['chained']} fallback={stats['fallback']} "
                f"table_regions={stats['table_regions_total']}"
            )

    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nDone. {len(all_results)} docs → {out_path}")
    print(f"Stats: chained={stats['chained']} fallback={stats['fallback']} "
          f"table_regions_total={stats['table_regions_total']}")
    if errors:
        print(f"Errors ({len(errors)}): {errors[:5]}")

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
        split_name="lir_chained_50",
        dataset_path=DATA_ROOT,
        load_annotations=True,
        load_ocr=False,
        docids=list(all_results.keys()),
    )
    result = do_eval(eval_dataset, kile_preds, lir_preds)
    scores = print_scores(result)

    kile_ap = scores.get("kile_AP", 0) * 100
    lir_f1 = scores.get("lir_f1", 0) * 100
    print(f"\n{'='*58}")
    print(f"V5b baseline:         KILE AP 41.79% / LIR F1 49.90%")
    print(f"Phase 4b chained:     KILE AP {kile_ap:.2f}% / LIR F1 {lir_f1:.2f}%")
    print(f"LIR delta:            {lir_f1 - 49.90:+.2f}pp")
    print(f"{'='*58}")

    print("\nPer-fieldtype LIR F1:")
    from beat_docile.extract import _LIR_TYPES
    for ft in sorted(_LIR_TYPES):
        try:
            ft_metrics = result.get_metrics("lir", fieldtype=ft)
            f1 = ft_metrics.get("f1", 0) * 100
            tp = ft_metrics.get("TP", 0)
            if f1 > 0 or tp > 0:
                print(f"  {ft:<42} F1={f1:.1f}% TP={tp}")
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(run(_parse_args()))
