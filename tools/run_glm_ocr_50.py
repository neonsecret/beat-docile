#!/usr/bin/env python
"""GLM-OCR spike + 50-doc val inference.

Steps:
  1. Spike (5 docs): verify model loads and outputs non-empty fields
  2. Full 50-doc inference → predictions/glm_ocr_50.json
  3. Print KILE AP / LIR F1 vs v2 baseline (44.61% / 50.89%)

Usage:
    # On neon (CUDA, 3070):
    uv run python tools/run_glm_ocr_50.py

    # Skip spike, go straight to inference:
    uv run python tools/run_glm_ocr_50.py --no-spike

    # Spike only:
    uv run python tools/run_glm_ocr_50.py --spike-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("DATA_ROOT", str(PROJECT_ROOT / "data"))

from docile.dataset import Dataset, Field  # noqa: E402

from beat_docile.data import iter_pages, load_split  # noqa: E402
from beat_docile.eval import print_scores, run_eval  # noqa: E402
from beat_docile.glm_ocr_extract import MODEL_ID, extract_page  # noqa: E402

V2_PREDS_PATH = PROJECT_ROOT / "predictions" / "v2_preds.json"
OUT_PATH = PROJECT_ROOT / "predictions" / "glm_ocr_50.json"

V2_KILE = 44.61
V2_LIR = 50.89
N_DOCS = 50


def _fields_to_dicts(fields: list[Field]) -> list[dict]:
    return [f.to_dict() for f in fields]


def run_spike(docs: list, model_id: str, n: int = 5) -> None:
    print(f"\n{'='*60}")
    print(f"SPIKE: first {n} docs")
    print(f"{'='*60}")
    for doc in docs[:n]:
        for page in iter_pages(doc):
            kile, lir = extract_page(page, model_id)
            print(f"\n  {doc.docid} p{page.page_index}: {len(kile)} KILE + {len(lir)} LIR fields")
            for f in (kile + lir)[:5]:
                bbox_str = f"[{f.bbox.left:.3f},{f.bbox.top:.3f},{f.bbox.right:.3f},{f.bbox.bottom:.3f}]"
                print(f"    {f.fieldtype:<35} {bbox_str}")
            if len(kile) + len(lir) > 5:
                print(f"    ... and {len(kile) + len(lir) - 5} more")
            break  # first page only
    print("\nSpike complete.\n")


def run_inference(
    docs: list,
    model_id: str,
    out_path: Path,
) -> dict[str, list[dict]]:
    all_results: dict[str, list[dict]] = {}
    if out_path.exists():
        all_results = json.loads(out_path.read_text())
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
                kile, lir = extract_page(page, model_id)
                doc_fields.extend(kile)
                doc_fields.extend(lir)
            all_results[doc.docid] = _fields_to_dicts(doc_fields)
        except Exception as e:
            errors.append(f"{doc.docid}: {e}")
            all_results[doc.docid] = []

        completed += 1
        elapsed = time.time() - start_t
        n_done = completed - (total - len(remaining))
        rate = n_done / elapsed if elapsed > 0 else 0
        eta = (total - completed) / rate if rate > 0 else float("inf")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(all_results, indent=2))
        print(
            f"[{completed}/{total}] {doc.docid} — "
            f"{len(all_results[doc.docid])} fields, "
            f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s"
        )

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"  {e}")

    return all_results


def eval_predictions(preds_dict: dict[str, list[dict]], label: str) -> tuple[float, float]:
    from beat_docile.config import DATA_ROOT as _DATA_ROOT

    kile_preds: dict[str, list[Field]] = {}
    lir_preds: dict[str, list[Field]] = {}
    for docid, fields in preds_dict.items():
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
        docids=list(preds_dict.keys()),
    )
    result = run_eval(eval_dataset, kile_preds, lir_preds)
    scores = print_scores(result)
    kile_ap = scores.get("kile_AP", 0) * 100
    lir_f1 = scores.get("lir_f1", 0) * 100
    print(f"  [{label}] KILE AP: {kile_ap:.2f}%  LIR F1: {lir_f1:.2f}%")
    return kile_ap, lir_f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--spike-only", action="store_true")
    parser.add_argument("--no-spike", action="store_true")
    parser.add_argument("--n", type=int, default=N_DOCS, help="Number of docs to process")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    if not V2_PREDS_PATH.exists():
        print(f"ERROR: {V2_PREDS_PATH} not found. Need v2 baseline predictions.")
        sys.exit(1)

    target_docids = list(json.loads(V2_PREDS_PATH.read_text()).keys())[: args.n]
    print(f"Target: {len(target_docids)} docs from v2_preds.json")

    dataset = load_split("val")
    docs = [d for d in dataset if d.docid in set(target_docids)]
    print(f"Matched {len(docs)} docs in val split")

    if not args.no_spike:
        run_spike(docs, args.model_id, n=5)
        if args.spike_only:
            print("--spike-only: exiting.")
            return

    print(f"\n{'='*60}")
    print(f"Full {args.n}-doc GLM-OCR inference")
    print(f"{'='*60}")
    results = run_inference(docs, args.model_id, args.out)

    print("\nEvaluating ...")
    kile_ap, lir_f1 = eval_predictions(results, "GLM-OCR standalone")

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"v2 baseline:     KILE {V2_KILE:.2f}%  LIR {V2_LIR:.2f}%")
    print(f"GLM-OCR standalone: KILE {kile_ap:.2f}%  LIR {lir_f1:.2f}%  "
          f"(Δ KILE {kile_ap - V2_KILE:+.2f}pp)")
    print(f"{'='*60}")
    print(f"\nPredictions: {args.out}")


if __name__ == "__main__":
    main()
