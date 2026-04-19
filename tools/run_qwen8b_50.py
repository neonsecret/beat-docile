#!/usr/bin/env python
"""Phase 7: Qwen3-VL-8B ensemble arm — 50-doc val inference + ensemble eval.

Steps:
  1. Spike (5 docs): verify model loads and outputs grounded JSON
  2. Full 50-doc inference → predictions/qwen8b_50.json
  3. Ensemble with V5b (weighted-max, weights 1.0/1.5) → predictions/qwen8b_ensemble_50.json
  4. Print KILE AP / LIR F1 for standalone + ensemble vs V5b baseline

Usage:
    # On neon (CUDA, 3070):
    uv run python tools/run_qwen8b_50.py

    # Mac (MPS, testing only — slow):
    uv run python tools/run_qwen8b_50.py --spike-only

    # Skip spike, start full run:
    uv run python tools/run_qwen8b_50.py --no-spike
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

from beat_docile.data import load_split, iter_pages
from beat_docile.ensemble import load_predictions, merge_predictions, save_predictions
from beat_docile.eval import run_eval, print_scores
from beat_docile.qwen8b_extract import Qwen8BExtractor, MODEL_ID
from docile.dataset import Dataset, Field

V5B_50_PATH = PROJECT_ROOT / "predictions" / "v5b_50.json"
QWEN8B_50_PATH = PROJECT_ROOT / "predictions" / "qwen8b_50.json"
ENSEMBLE_50_PATH = PROJECT_ROOT / "predictions" / "qwen8b_ensemble_50.json"

V5B_KILE = 41.79
V5B_LIR = 49.90


def _fields_to_dicts(fields: list[Field]) -> list[dict]:
    return [f.to_dict() for f in fields]


def run_spike(extractor: Qwen8BExtractor, docs: list, n: int = 5) -> None:
    """Run on first n docs; print raw outputs for visual verification."""
    print(f"\n{'='*60}")
    print(f"SPIKE: first {n} docs")
    print("='*60")
    for doc in docs[:n]:
        with doc:
            for page in iter_pages(doc):
                raw_fields = extractor.extract_page(page)
                print(f"\n  {doc.docid} p{page.page_index}: {len(raw_fields)} fields")
                for f in raw_fields[:5]:
                    bbox_str = f"[{f.bbox.left:.3f},{f.bbox.top:.3f},{f.bbox.right:.3f},{f.bbox.bottom:.3f}]"
                    print(f"    {f.fieldtype:<35} {bbox_str}  '{f.text[:40]}'")
                if len(raw_fields) > 5:
                    print(f"    ... and {len(raw_fields) - 5} more")
                break  # first page only for spike
    print(f"\nSpike complete — model loads and outputs look sane.\n")


def run_inference(
    extractor: Qwen8BExtractor,
    docs: list,
    out_path: Path,
    all_pages: bool = True,
) -> dict[str, list[dict]]:
    """Run inference on all docs; save + resume from out_path."""
    all_results: dict[str, list[dict]] = {}
    if out_path.exists():
        all_results = json.loads(out_path.read_text())
        print(f"Resuming — already done: {len(all_results)}/{len(docs)} docs")

    remaining = [d for d in docs if d.docid not in all_results]
    total = len(docs)
    completed = len(all_results)
    print(f"Remaining: {len(remaining)} docs\n")

    start_t = time.time()
    errors: list[str] = []

    for doc in remaining:
        try:
            doc_fields: list[Field] = []
            for page in iter_pages(doc):
                if not all_pages and page.page_index > 0:
                    continue
                doc_fields.extend(extractor.extract_page(page))
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


def eval_predictions(
    preds_dict: dict[str, list[dict]],
    label: str,
    docids: list[str],
) -> tuple[float, float]:
    """Eval a prediction dict; return (kile_ap, lir_f1) as percentages."""
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
        docids=docids,
    )
    result = run_eval(eval_dataset, kile_preds, lir_preds)
    scores = print_scores(result)
    kile_ap = scores.get("kile_AP", 0) * 100
    lir_f1 = scores.get("lir_f1", 0) * 100
    print(f"  [{label}] KILE AP: {kile_ap:.2f}%  LIR F1: {lir_f1:.2f}%")
    return kile_ap, lir_f1


def run_ensemble(
    v5b_path: Path,
    qwen8b_path: Path,
    out_path: Path,
    v5b_weight: float = 1.0,
    qwen8b_weight: float = 1.5,
) -> dict[str, list[dict]]:
    """Merge V5b + Qwen3-VL-8B via ensemble.py weighted-max."""
    v5b_preds = load_predictions(v5b_path)
    qwen8b_preds = load_predictions(qwen8b_path)
    merged = merge_predictions(
        sources=[v5b_preds, qwen8b_preds],
        weights=[v5b_weight, qwen8b_weight],
        iou_threshold=0.5,
        score_combine="weighted_max",
    )
    save_predictions(merged, out_path)
    total = sum(len(v) for v in merged.values())
    print(f"\nEnsemble saved: {len(merged)} docs, {total} fields → {out_path}")
    merged_dicts = {docid: [f.to_dict() for f in fields] for docid, fields in merged.items()}
    return merged_dicts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--spike-only", action="store_true", help="Run spike only (5 docs)")
    parser.add_argument("--no-spike", action="store_true", help="Skip spike, go straight to full run")
    parser.add_argument("--no-ensemble", action="store_true", help="Skip ensemble step")
    parser.add_argument(
        "--v5b-weight", type=float, default=1.0,
        help="Ensemble weight for V5b (default 1.0)"
    )
    parser.add_argument(
        "--qwen8b-weight", type=float, default=1.5,
        help="Ensemble weight for Qwen3-VL-8B (default 1.5)"
    )
    args = parser.parse_args()

    if not V5B_50_PATH.exists():
        print(f"ERROR: {V5B_50_PATH} not found. Run V5b baseline first.")
        sys.exit(1)

    target_docids = list(json.loads(V5B_50_PATH.read_text()).keys())
    print(f"Target: {len(target_docids)} docs from v5b_50.json")

    dataset = load_split("val")
    docs = [d for d in dataset if d.docid in set(target_docids)]
    print(f"Matched {len(docs)} docs in val split")

    print(f"\nLoading Qwen3-VL-8B-Instruct ({args.model_id}) ...")
    extractor = Qwen8BExtractor(model_id=args.model_id)

    # ── Spike ──────────────────────────────────────────────────────────────────
    if not args.no_spike:
        run_spike(extractor, docs, n=5)
        if args.spike_only:
            print("--spike-only: exiting after spike.")
            return

    # ── Full 50-doc inference ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Full 50-doc Qwen3-VL-8B inference")
    print(f"{'='*60}")
    qwen8b_results = run_inference(extractor, docs, QWEN8B_50_PATH)

    print("\nEvaluating standalone Qwen3-VL-8B ...")
    kile_standalone, lir_standalone = eval_predictions(
        qwen8b_results, "Qwen3-VL-8B standalone", target_docids
    )

    # ── Ensemble ───────────────────────────────────────────────────────────────
    if not args.no_ensemble:
        print(f"\n{'='*60}")
        print(f"Ensemble: V5b (w={args.v5b_weight}) + Qwen3-VL-8B (w={args.qwen8b_weight})")
        print(f"{'='*60}")
        ensemble_results = run_ensemble(
            V5B_50_PATH, QWEN8B_50_PATH, ENSEMBLE_50_PATH,
            v5b_weight=args.v5b_weight,
            qwen8b_weight=args.qwen8b_weight,
        )
        print("\nEvaluating ensemble ...")
        kile_ensemble, lir_ensemble = eval_predictions(
            ensemble_results, "V5b + Qwen3-VL-8B ensemble", target_docids
        )
    else:
        kile_ensemble = lir_ensemble = None

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 7 RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"V5b baseline:             KILE {V5B_KILE:.2f}%  LIR {V5B_LIR:.2f}%")
    print(f"Qwen3-VL-8B standalone:   KILE {kile_standalone:.2f}%  LIR {lir_standalone:.2f}%  "
          f"(Δ KILE {kile_standalone - V5B_KILE:+.2f}pp)")
    if kile_ensemble is not None:
        print(f"V5b + Qwen3-VL-8B ens:   KILE {kile_ensemble:.2f}%  LIR {lir_ensemble:.2f}%  "
              f"(Δ KILE {kile_ensemble - V5B_KILE:+.2f}pp  "
              f"Δ vs standalone {kile_ensemble - kile_standalone:+.2f}pp)")
    print(f"{'='*60}")
    print(f"\nPredictions saved:")
    print(f"  Standalone: {QWEN8B_50_PATH}")
    if kile_ensemble is not None:
        print(f"  Ensemble:   {ENSEMBLE_50_PATH}")


if __name__ == "__main__":
    main()
