#!/usr/bin/env python
"""Classifier reranker over v2_ensemble_250.json (250-doc gate).

For each predicted field, scores it with the per-fieldtype sklearn MLP classifier
(models/classifiers/), multiplies Claude's score by MLP score, and drops predictions
where MLP score < threshold (default 0.3).

No Claude API calls — pure CPU post-processing on existing predictions.

Usage:
    DATA_ROOT=data uv run python tools/run_classifier_rerank_250.py
    DATA_ROOT=data uv run python tools/run_classifier_rerank_250.py --threshold 0.2
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

from beat_docile.classifiers import (  # noqa: E402
    _parse_ocr_words,
    _words_in_annotation,
    classifier_score,
)
from beat_docile.eval import print_scores, run_eval  # noqa: E402

MODEL_DIR = PROJECT_ROOT / "models" / "classifiers"
DATA_DIR = PROJECT_ROOT / "data"
VAL_250_PATH = PROJECT_ROOT / "tools" / "val_250_docids.json"
ENSEMBLE_500_PATH = PROJECT_ROOT / "predictions" / "v2_ensemble_500.json"
ENSEMBLE_250_PATH = PROJECT_ROOT / "predictions" / "v2_ensemble_250.json"
OUT_PATH = PROJECT_ROOT / "predictions" / "v2_ensemble_classifier_250.json"

ENSEMBLE_KILE = 46.48
ENSEMBLE_LIR = 52.16  # update if known


def _load_ocr_words(docid: str) -> list[list]:
    """Load per-page word lists from OCR JSON. Returns [] on failure."""
    ocr_path = DATA_DIR / "ocr" / f"{docid}.json"
    if not ocr_path.exists():
        return []
    try:
        with ocr_path.open() as f:
            ocr_data = json.load(f)
        return _parse_ocr_words(ocr_data)
    except Exception:
        return []


def rerank_doc_predictions(
    preds: list[dict],
    pages_words: list[list],
    threshold: float,
) -> list[dict]:
    """Rerank one doc's predictions using MLP classifiers.

    For each prediction:
      1. Find OCR words in prediction bbox (page-specific)
      2. Score with MLP for that fieldtype
      3. Multiply Claude score * MLP score
      4. Drop if MLP score < threshold
    Falls back to keeping the prediction unchanged if:
      - No model exists for that fieldtype (MLP returns 0.5)
      - No words found in bbox (MLP can't score — keep prediction)
    """
    reranked = []
    for pred in preds:
        ft = pred.get("fieldtype", "")
        page = pred.get("page", 0)
        bbox = pred.get("bbox", [])
        score = float(pred.get("score", 1.0))

        if not bbox or len(bbox) != 4 or page >= len(pages_words):
            reranked.append(pred)
            continue

        words = pages_words[page]
        bbox_tuple = tuple(bbox)

        span_ids = _words_in_annotation(words, bbox_tuple)
        if not span_ids:
            # No words in bbox — skip MLP scoring, keep pred as-is
            reranked.append(pred)
            continue

        mlp_score = classifier_score(ft, span_ids, words, 1.0, 1.0, MODEL_DIR)

        if mlp_score < threshold:
            continue  # drop this prediction

        combined_score = score * mlp_score
        new_pred = dict(pred)
        new_pred["score"] = combined_score
        reranked.append(new_pred)

    return reranked


def eval_predictions(preds_dict: dict, label: str, docids: list[str]) -> tuple[float, float]:
    from docile.dataset import Dataset, Field

    from beat_docile.config import DATA_ROOT as _DATA_ROOT

    kile_preds: dict[str, list[Field]] = {}
    lir_preds: dict[str, list[Field]] = {}
    for docid in docids:
        kile_preds[docid] = []
        lir_preds[docid] = []
        for fd in preds_dict.get(docid, []):
            f = Field.from_dict(fd)
            if f.line_item_id is not None:
                lir_preds[docid].append(f)
            else:
                kile_preds[docid].append(f)

    eval_dataset = Dataset(
        split_name=f"val_subset_{len(docids)}",  # custom name → no index file → accepts any docids
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Drop predictions where MLP score < threshold (default 0.3)")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    # Load 250 docids
    if not VAL_250_PATH.exists():
        # Create from v2_preds.json first 250 docids
        v2 = json.loads((PROJECT_ROOT / "predictions" / "v2_preds.json").read_text())
        docids_250 = list(v2.keys())[:250]
        VAL_250_PATH.write_text(json.dumps(docids_250, indent=2))
        print(f"Created {VAL_250_PATH}")
    else:
        docids_250 = json.loads(VAL_250_PATH.read_text())

    print(f"250-doc gate: {len(docids_250)} docids")
    print(f"Classifier model dir: {MODEL_DIR}")
    print(f"Available models: {len(list(MODEL_DIR.glob('*.joblib')))}")
    print(f"MLP drop threshold: {args.threshold}\n")

    # Load ensemble predictions — prefer 250 file, fall back to filtering 500
    if ENSEMBLE_250_PATH.exists():
        ensemble_all = json.loads(ENSEMBLE_250_PATH.read_text())
        print(f"Loaded ensemble 250-doc predictions from {ENSEMBLE_250_PATH}")
    else:
        ensemble_all = json.loads(ENSEMBLE_500_PATH.read_text())
        print("Filtering ensemble 500-doc predictions to 250 docids")


    reranked_all: dict[str, list[dict]] = {}
    n_original = 0
    n_kept = 0
    n_dropped = 0
    start_t = time.time()

    for i, docid in enumerate(docids_250):
        preds = ensemble_all.get(docid, [])
        n_original += len(preds)

        pages_words = _load_ocr_words(docid)
        reranked = rerank_doc_predictions(preds, pages_words, args.threshold)

        n_kept += len(reranked)
        n_dropped += len(preds) - len(reranked)
        reranked_all[docid] = reranked

        if (i + 1) % 50 == 0 or (i + 1) == len(docids_250):
            elapsed = time.time() - start_t
            print(f"[{i+1}/250] kept {n_kept}/{n_original} preds, "
                  f"dropped {n_dropped}  ({elapsed:.1f}s)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reranked_all, indent=2))
    print(f"\nSaved to {args.out}")
    print(f"Total: {n_original} preds → {n_kept} kept, {n_dropped} dropped "
          f"({100*n_dropped/max(n_original,1):.1f}% drop rate)\n")

    # Eval reranked
    print("Evaluating reranked predictions...")
    kile_re, lir_re = eval_predictions(reranked_all, "Reranked", docids_250)

    # Eval original ensemble on same 250 docs
    print("Evaluating original ensemble (same 250 docs)...")
    ensemble_250 = {d: ensemble_all.get(d, []) for d in docids_250}
    kile_base, lir_base = eval_predictions(ensemble_250, "Ensemble baseline", docids_250)

    print(f"\n{'='*60}")
    print(f"CLASSIFIER RERANKER — 250-doc gate (threshold={args.threshold})")
    print(f"{'='*60}")
    print(f"Ensemble baseline:   KILE {kile_base:.2f}%  LIR {lir_base:.2f}%")
    print(f"Reranked:            KILE {kile_re:.2f}%  LIR {lir_re:.2f}%  "
          f"(Δ KILE {kile_re - kile_base:+.2f}pp)")
    print(f"{'='*60}")
    gate = "PASS (≥1pp) → confirm with 500" if kile_re - kile_base >= 1.0 else "FAIL (<1pp) → bury"
    print(f"Decision gate: {gate}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
