#!/usr/bin/env python
"""Item #7: Deduplication of same-page same-fieldtype predictions.

Both main pass and targeted pass can extract the same field (e.g., vendor_tax_id).
Combined, this creates duplicate predictions: 1 GT → 1 TP + 1 FP → hurts AP.

Fix: for single-occurrence KILE fields, keep only highest-scoring prediction per
(fieldtype, page). Multi-occurrence fields (tax_detail_*, line items) are exempt.

Post-processes existing refiner_guard_50.json — NO API CALLS needed.

Compare to refiner_guard_50 baseline (44.87% KILE / 52.08% LIR).

Usage:
    DATA_ROOT=data uv run python tools/dedup_50.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
os.environ.setdefault("DATA_ROOT", str(DATA_DIR))

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from beat_docile.config import DATA_ROOT
from docile.dataset import Field

IN_PATH = PROJECT_ROOT / "predictions" / "refiner_guard_50.json"
OUT_PATH = PROJECT_ROOT / "predictions" / "dedup_50.json"

GUARD_KILE = 44.87
GUARD_LIR = 52.08
V5B_KILE = 41.86
V5B_LIR = 52.36

# Fields that can appear multiple times per document (one per tax rate row)
MULTI_OK = {"tax_detail_gross", "tax_detail_net", "tax_detail_rate", "tax_detail_tax"}


def dedup_predictions(raw_fields: list[dict]) -> list[dict]:
    """Dedup same-page same-fieldtype predictions, keeping highest-scoring."""
    parsed = [Field.from_dict(f) for f in raw_fields]

    single_seen: dict[tuple[str, int], Field] = {}
    multi_fields: list[Field] = []

    for f in parsed:
        if f.fieldtype in MULTI_OK or f.line_item_id is not None:
            multi_fields.append(f)
        else:
            key = (f.fieldtype, f.page)
            if key not in single_seen or f.score > single_seen[key].score:
                single_seen[key] = f

    deduped = list(single_seen.values()) + multi_fields
    return [f.to_dict() for f in deduped]


def main() -> None:
    print(f"Input:  {IN_PATH}")
    print(f"Output: {OUT_PATH}")
    print(f"Baseline: refiner_guard_50 {GUARD_KILE}% KILE / {GUARD_LIR}% LIR")

    if not IN_PATH.exists():
        print(f"\n❌ {IN_PATH} not found — run refiner_guard_50.py first")
        return

    all_results = json.loads(IN_PATH.read_text())
    print(f"\nLoaded {len(all_results)} docs from refiner_guard_50.json")

    total_before = sum(len(v) for v in all_results.values())
    deduped = {docid: dedup_predictions(fields) for docid, fields in all_results.items()}
    total_after = sum(len(v) for v in deduped.values())

    print(f"Predictions before dedup: {total_before}")
    print(f"Predictions after dedup:  {total_after}")
    print(f"Removed: {total_before - total_after} duplicates")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(deduped, indent=2))
    print(f"Saved to {OUT_PATH}")

    print("\nRunning evaluation...")
    from beat_docile.eval import run_eval, print_scores
    from docile.dataset import Dataset

    kile_preds: dict[str, list[Field]] = {}
    lir_preds: dict[str, list[Field]] = {}
    for docid, fields in deduped.items():
        kile_preds[docid] = []
        lir_preds[docid] = []
        for fd in fields:
            f = Field.from_dict(fd)
            if f.line_item_id is not None:
                lir_preds[docid].append(f)
            else:
                kile_preds[docid].append(f)

    eval_dataset = Dataset(
        split_name="dedup_50",
        dataset_path=DATA_ROOT,
        load_annotations=True,
        load_ocr=False,
        docids=list(deduped.keys()),
    )
    result = run_eval(eval_dataset, kile_preds, lir_preds)
    scores = print_scores(result)

    kile_ap = scores.get("kile_AP", 0) * 100
    lir_f1 = scores.get("lir_f1", 0) * 100

    print(f"\n{'='*60}")
    print(f"V5b-50 baseline:    KILE {V5B_KILE:.2f}% / LIR {V5B_LIR:.2f}%")
    print(f"refiner_guard_50:   KILE {GUARD_KILE:.2f}% / LIR {GUARD_LIR:.2f}%")
    print(f"dedup_50 (this):    KILE {kile_ap:.2f}% / LIR {lir_f1:.2f}%")
    print(f"Delta vs guard:     KILE {kile_ap - GUARD_KILE:+.2f}pp / LIR {lir_f1 - GUARD_LIR:+.2f}pp")

    if kile_ap > GUARD_KILE and lir_f1 > GUARD_LIR:
        print(f"\n✅ BEATS guard on BOTH — apply dedup to refiner_guard_500 predictions")
    elif kile_ap > GUARD_KILE:
        print(f"\n⚠️  KILE improved (+{kile_ap - GUARD_KILE:.2f}pp), LIR behind by {lir_f1 - GUARD_LIR:.2f}pp")
    elif lir_f1 > GUARD_LIR:
        print(f"\n⚠️  LIR improved but KILE regressed — dedup is counterproductive for KILE")
    else:
        print(f"\n❌ No improvement — duplicate predictions are not hurting AP in practice")


if __name__ == "__main__":
    main()
