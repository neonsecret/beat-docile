#!/usr/bin/env python
"""GLM-OCR 500-doc val inference (full v2 docid set).

Run after run_glm_ocr_50.py confirms KILE > 20% on 50 docs.

Usage:
    uv run python tools/run_glm_ocr_500.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("DATA_ROOT", str(PROJECT_ROOT / "data"))

from docile.dataset import Dataset, Field  # noqa: E402

from beat_docile.data import iter_pages, load_split  # noqa: E402
from beat_docile.eval import print_scores, run_eval  # noqa: E402
from beat_docile.glm_ocr_extract import MODEL_ID, extract_page  # noqa: E402

V2_PREDS_PATH = PROJECT_ROOT / "predictions" / "v2_preds.json"
OUT_PATH = PROJECT_ROOT / "predictions" / "glm_ocr_500.json"

V2_KILE = 44.61
V2_LIR = 50.89


def main() -> None:
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    if not V2_PREDS_PATH.exists():
        print(f"ERROR: {V2_PREDS_PATH} not found.")
        sys.exit(1)

    target_docids = list(json.loads(V2_PREDS_PATH.read_text()).keys())
    print(f"Target: {len(target_docids)} docs from v2_preds.json")

    dataset = load_split("val")
    docs = [d for d in dataset if d.docid in set(target_docids)]
    print(f"Matched {len(docs)} docs in val split\n")

    all_results: dict[str, list[dict]] = {}
    if args.out.exists():
        all_results = json.loads(args.out.read_text())
        print(f"Resuming — already done: {len(all_results)}/{len(docs)} docs")

    remaining = [d for d in docs if d.docid not in all_results]
    total = len(docs)
    completed = len(all_results)
    errors: list[str] = []
    start_t = time.time()

    for doc in remaining:
        try:
            doc_fields: list[Field] = []
            for page in iter_pages(doc):
                kile, lir = extract_page(page, args.model_id)
                doc_fields.extend(kile)
                doc_fields.extend(lir)
            all_results[doc.docid] = [f.to_dict() for f in doc_fields]
        except Exception as e:
            errors.append(f"{doc.docid}: {e}")
            all_results[doc.docid] = []

        completed += 1
        elapsed = time.time() - start_t
        n_done = completed - (total - len(remaining))
        rate = n_done / elapsed if elapsed > 0 else 0
        eta = (total - completed) / rate if rate > 0 else float("inf")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(all_results, indent=2))
        print(
            f"[{completed}/{total}] {doc.docid} — "
            f"{len(all_results[doc.docid])} fields, "
            f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s"
        )

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"  {e}")

    # Eval
    from beat_docile.config import DATA_ROOT as _DATA_ROOT

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
        dataset_path=_DATA_ROOT,
        load_annotations=True,
        load_ocr=False,
        docids=list(all_results.keys()),
    )
    result = run_eval(eval_dataset, kile_preds, lir_preds)
    scores = print_scores(result)
    kile_ap = scores.get("kile_AP", 0) * 100
    lir_f1 = scores.get("lir_f1", 0) * 100

    print(f"\n{'='*60}")
    print("RESULTS — 500 docs")
    print(f"{'='*60}")
    print(f"v2 baseline:        KILE {V2_KILE:.2f}%  LIR {V2_LIR:.2f}%")
    print(f"GLM-OCR standalone: KILE {kile_ap:.2f}%  LIR {lir_f1:.2f}%  "
          f"(Δ KILE {kile_ap - V2_KILE:+.2f}pp)")
    print(f"{'='*60}")
    print(f"\nPredictions: {args.out}")


if __name__ == "__main__":
    main()
