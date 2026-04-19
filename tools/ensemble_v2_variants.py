#!/usr/bin/env python
"""Ensemble v2 variants: v2_t00_500 + v2_t03_500 + v2_alt_500.

Loads all three prediction files, ensembles via merge_predictions(),
scores, and compares to v2 baseline.

Requires all three input files to exist (run v2_t03_500.py and v2_alt_500.py first).

Outputs:
  predictions/v2_ensemble_500.json

Usage:
    DATA_ROOT=data uv run python tools/ensemble_v2_variants.py
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
from beat_docile.ensemble import load_predictions, merge_predictions, save_predictions
from docile.dataset import Field

V2_500_KILE = 44.61
V2_500_LIR = 50.89

T00_PATH = PROJECT_ROOT / "predictions" / "v2_t00_500.json"
T03_PATH = PROJECT_ROOT / "predictions" / "v2_t03_500.json"
ALT_PATH = PROJECT_ROOT / "predictions" / "v2_alt_500.json"
OUT_PATH = PROJECT_ROOT / "predictions" / "v2_ensemble_500.json"


def _score(preds: dict[str, list[Field]], label: str) -> tuple[float, float]:
    from beat_docile.eval import run_eval, print_scores
    from docile.dataset import Dataset

    kile_preds: dict[str, list[Field]] = {}
    lir_preds: dict[str, list[Field]] = {}
    for docid, fields in preds.items():
        kile_preds[docid] = [f for f in fields if f.line_item_id is None]
        lir_preds[docid] = [f for f in fields if f.line_item_id is not None]

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
    print(f"  {label}: KILE {kile_ap:.2f}% / LIR {lir_f1:.2f}%")
    return kile_ap, lir_f1


def main() -> None:
    for p, name in [(T00_PATH, "v2_t00_500"), (T03_PATH, "v2_t03_500"), (ALT_PATH, "v2_alt_500")]:
        if not p.exists():
            print(f"❌ Missing: {p} — run the corresponding script first")
            return

    print("Loading predictions...")
    t00 = load_predictions(T00_PATH)
    t03 = load_predictions(T03_PATH)
    alt = load_predictions(ALT_PATH)

    print(f"  v2_t00_500: {len(t00)} docs")
    print(f"  v2_t03_500: {len(t03)} docs")
    print(f"  v2_alt_500: {len(alt)} docs")

    print("\nEnsembling (equal weights, iou_threshold=0.5, weighted_max)...")
    ensemble = merge_predictions(
        sources=[t00, t03, alt],
        weights=None,
        iou_threshold=0.5,
        score_combine="weighted_max",
    )
    print(f"  Ensemble: {sum(len(v) for v in ensemble.values())} total fields across {len(ensemble)} docs")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_predictions(ensemble, OUT_PATH)
    print(f"  Saved → {OUT_PATH}")

    print("\nScoring individual variants...")
    t00_kile, t00_lir = _score(t00, "v2_t00 (T=1.0 baseline)")
    t03_kile, t03_lir = _score(t03, "v2_t03 (T=0.3)")
    alt_kile, alt_lir = _score(alt, "v2_alt (alt prompt, T=1.0)")

    print("\nScoring ensemble...")
    ens_kile, ens_lir = _score(ensemble, "v2_ensemble")

    print(f"\n{'='*65}")
    print(f"{'Variant':<35} {'KILE':>8} {'LIR':>8} {'ΔKILE':>8} {'ΔLIR':>8}")
    print(f"{'-'*65}")
    print(f"{'v2-500 baseline':35} {V2_500_KILE:8.2f} {V2_500_LIR:8.2f} {'—':>8} {'—':>8}")
    print(f"{'v2_t00 (T=1.0)':35} {t00_kile:8.2f} {t00_lir:8.2f} {t00_kile-V2_500_KILE:+8.2f} {t00_lir-V2_500_LIR:+8.2f}")
    print(f"{'v2_t03 (T=0.3)':35} {t03_kile:8.2f} {t03_lir:8.2f} {t03_kile-V2_500_KILE:+8.2f} {t03_lir-V2_500_LIR:+8.2f}")
    print(f"{'v2_alt (alt prompt, T=1.0)':35} {alt_kile:8.2f} {alt_lir:8.2f} {alt_kile-V2_500_KILE:+8.2f} {alt_lir-V2_500_LIR:+8.2f}")
    print(f"{'v2_ensemble':35} {ens_kile:8.2f} {ens_lir:8.2f} {ens_kile-V2_500_KILE:+8.2f} {ens_lir-V2_500_LIR:+8.2f}")
    print(f"{'='*65}")

    if ens_kile > V2_500_KILE + 1.0 and ens_lir >= V2_500_LIR:
        print(f"\n✅ ENSEMBLE BEATS v2 by ≥1pp KILE — new baseline candidate: v2_ensemble_500.json")
    elif ens_kile > V2_500_KILE + 0.5:
        print(f"\n⚠️  Modest KILE gain (+{ens_kile-V2_500_KILE:.2f}pp) — borderline, may be noise")
    elif ens_kile > V2_500_KILE:
        print(f"\n➖ Marginal gain (+{ens_kile-V2_500_KILE:.2f}pp) — within noise, not a new baseline")
    else:
        print(f"\n❌ Ensemble does not beat v2 — variants too correlated or diversity unhelpful")
        print("    Consider majority-vote ensemble instead, or increase IoU threshold.")


if __name__ == "__main__":
    main()
