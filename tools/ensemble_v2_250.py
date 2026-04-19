#!/usr/bin/env python
"""Ensemble v2 variants on 250-doc gate: v2_t00_250 + v2_t03_250 + v2_alt_250.

Loads all three prediction files, ensembles via merge_predictions(),
scores all variants + ensemble, compares to v2 baseline on same 250 docs.

Decision gate:
  ≥1pp above v2-250 → confirm with full 500-doc run
  <1pp gain         → bury here

Requires: v2_t00_250.json, v2_t03_250.json, v2_alt_250.json
Outputs:  predictions/v2_ensemble_250.json

Usage:
    DATA_ROOT=data uv run python tools/ensemble_v2_250.py
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
from docile.dataset import Field, Dataset

DOCIDS_PATH = PROJECT_ROOT / "tools" / "val_250_docids.json"
T00_PATH = PROJECT_ROOT / "predictions" / "v2_t00_250.json"
T03_PATH = PROJECT_ROOT / "predictions" / "v2_t03_250.json"
ALT_PATH = PROJECT_ROOT / "predictions" / "v2_alt_250.json"
OUT_PATH = PROJECT_ROOT / "predictions" / "v2_ensemble_250.json"


def _score(preds: dict[str, list[Field]], label: str, target_docids: list[str]) -> tuple[float, float]:
    from beat_docile.eval import run_eval, print_scores

    kile_preds = {d: [f for f in preds[d] if f.line_item_id is None] for d in preds}
    lir_preds = {d: [f for f in preds[d] if f.line_item_id is not None] for d in preds}

    eval_dataset = Dataset(
        split_name="v2_250_gate",
        dataset_path=DATA_ROOT,
        load_annotations=True,
        load_ocr=False,
        docids=target_docids,
    )
    result = run_eval(eval_dataset, kile_preds, lir_preds)
    scores = print_scores(result)
    kile_ap = scores.get("kile_AP", 0) * 100
    lir_f1 = scores.get("lir_f1", 0) * 100
    return kile_ap, lir_f1


def main() -> None:
    target_docids: list[str] = json.loads(DOCIDS_PATH.read_text())
    print(f"250-doc gate evaluation")
    print(f"Target docids: {len(target_docids)}")

    for p, name in [(T00_PATH, "v2_t00_250"), (T03_PATH, "v2_t03_250"), (ALT_PATH, "v2_alt_250")]:
        if not p.exists():
            print(f"❌ Missing: {p} — run the corresponding script first")
            return

    print("\nLoading predictions...")
    t00 = load_predictions(T00_PATH)
    t03 = load_predictions(T03_PATH)
    alt = load_predictions(ALT_PATH)
    print(f"  v2_t00_250: {len(t00)} docs")
    print(f"  v2_t03_250: {len(t03)} docs")
    print(f"  v2_alt_250: {len(alt)} docs")

    print("\nEnsembling (equal weights, iou_threshold=0.5, weighted_max)...")
    ensemble = merge_predictions(
        sources=[t00, t03, alt],
        weights=None,
        iou_threshold=0.5,
        score_combine="weighted_max",
    )
    print(f"  Ensemble: {sum(len(v) for v in ensemble.values())} fields, {len(ensemble)} docs")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_predictions(ensemble, OUT_PATH)
    print(f"  Saved → {OUT_PATH}")

    print("\nScoring all variants...")
    t00_kile, t00_lir = _score(t00, "v2_t00 (T=1.0 baseline)", target_docids)
    t03_kile, t03_lir = _score(t03, "v2_t03 (T=0.3)", target_docids)
    alt_kile, alt_lir = _score(alt, "v2_alt (alt prompt, T=1.0)", target_docids)
    ens_kile, ens_lir = _score(ensemble, "v2_ensemble", target_docids)

    print(f"\n{'='*68}")
    print(f"{'Variant':<35} {'KILE':>8} {'LIR':>8} {'ΔKILE':>8} {'ΔLIR':>8}")
    print(f"{'-'*68}")
    print(f"{'v2_t00 (T=1.0, baseline)':35} {t00_kile:8.2f} {t00_lir:8.2f} {'—':>8} {'—':>8}")
    print(f"{'v2_t03 (T=0.3)':35} {t03_kile:8.2f} {t03_lir:8.2f} {t03_kile-t00_kile:+8.2f} {t03_lir-t00_lir:+8.2f}")
    print(f"{'v2_alt (alt prompt, T=1.0)':35} {alt_kile:8.2f} {alt_lir:8.2f} {alt_kile-t00_kile:+8.2f} {alt_lir-t00_lir:+8.2f}")
    print(f"{'v2_ensemble':35} {ens_kile:8.2f} {ens_lir:8.2f} {ens_kile-t00_kile:+8.2f} {ens_lir-t00_lir:+8.2f}")
    print(f"{'='*68}")

    if ens_kile >= t00_kile + 1.0:
        print(f"\n✅ ENSEMBLE +{ens_kile-t00_kile:.2f}pp KILE → run full 500-doc confirmation")
    elif ens_kile >= t00_kile + 0.5:
        print(f"\n⚠️  Borderline gain (+{ens_kile-t00_kile:.2f}pp) — likely noise, skip 500-doc")
    elif ens_kile > t00_kile:
        print(f"\n➖ Marginal (+{ens_kile-t00_kile:.2f}pp) — noise, bury ensemble approach")
    else:
        print(f"\n❌ Ensemble does not beat v2_t00 — variants too correlated. BURY.")


if __name__ == "__main__":
    main()
