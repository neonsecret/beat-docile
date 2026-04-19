#!/usr/bin/env python
"""Conservative AOL verifier applied to v2_ensemble on 250-doc gate.

Input: predictions/v2_ensemble_500.json (filtered to 250 docids)
Passes:
  1. Calc verifier: amount_total_gross ≈ amount_total_net + amount_total_tax
     - Math fail → demote × 0.5 + one targeted re-prompt per failing page
  2. Overlap verifier: KILE fields with >50% bbox overlap → demote both × 0.5
Output: predictions/v2_aol_250.json

Compare to v2_ensemble on same 250 docs:
  Decision: ≥1pp KILE lift → 500-doc confirm; <1pp → bury.

Usage:
    DATA_ROOT=data uv run python tools/v2_aol_250.py
    DATA_ROOT=data uv run python tools/v2_aol_250.py --no-reprompt  # pure Python, no API
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
os.environ.setdefault("DATA_ROOT", str(DATA_DIR))
# AOL itself is post-processing — base extraction flags don't matter here
os.environ["BD_USE_REFINER"] = "0"
os.environ["BD_USE_VALIDATOR"] = "0"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from beat_docile.config import DEFAULT_MODEL, DATA_ROOT
from beat_docile.ensemble import load_predictions, save_predictions
from beat_docile.aol_extract import apply_aol_verifiers
from docile.dataset import Field, Dataset

DOCIDS_PATH = PROJECT_ROOT / "tools" / "val_250_docids.json"
ENSEMBLE_PATH = PROJECT_ROOT / "predictions" / "v2_ensemble_500.json"
OUT_PATH = PROJECT_ROOT / "predictions" / "v2_aol_250.json"

MODEL = DEFAULT_MODEL

NO_REPROMPT = "--no-reprompt" in sys.argv


def _score(preds: dict[str, list[Field]], label: str, target_docids: list[str]) -> tuple[float, float]:
    from beat_docile.eval import run_eval, print_scores
    kile_preds = {d: [f for f in preds[d] if f.line_item_id is None] for d in preds}
    lir_preds = {d: [f for f in preds[d] if f.line_item_id is not None] for d in preds}
    ds = Dataset(
        split_name="v2_250_gate",
        dataset_path=DATA_ROOT,
        load_annotations=True,
        load_ocr=False,
        docids=target_docids,
    )
    result = run_eval(ds, kile_preds, lir_preds)
    scores = print_scores(result)
    kile_ap = scores.get("kile_AP", 0) * 100
    lir_f1 = scores.get("lir_f1", 0) * 100
    print(f"  {label}: KILE {kile_ap:.2f}% / LIR {lir_f1:.2f}%")
    return kile_ap, lir_f1


async def main() -> None:
    target_docids: list[str] = json.loads(DOCIDS_PATH.read_text())
    print(f"250-doc AOL gate")
    print(f"Target: {len(target_docids)} docs")
    print(f"Model: {MODEL}")
    print(f"Re-prompt: {'disabled (pure Python)' if NO_REPROMPT else 'enabled'}")

    if not ENSEMBLE_PATH.exists():
        print(f"❌ {ENSEMBLE_PATH} not found — run ensemble_v2_variants.py first")
        return

    # Load ensemble predictions, filter to 250 target docids
    print("\nLoading v2_ensemble predictions...")
    all_ensemble = load_predictions(ENSEMBLE_PATH)
    target_set = set(target_docids)
    ensemble_250 = {d: fields for d, fields in all_ensemble.items() if d in target_set}
    print(f"  Loaded {len(ensemble_250)} / {len(target_docids)} target docs")

    if len(ensemble_250) < len(target_docids):
        missing = [d for d in target_docids if d not in ensemble_250]
        print(f"  ⚠️  Missing {len(missing)} docs from ensemble: {missing[:3]}...")

    # Score baseline (v2_ensemble on 250 docs)
    print("\nBaseline: v2_ensemble on 250 docs")
    baseline_kile, baseline_lir = _score(ensemble_250, "v2_ensemble_250 (baseline)", target_docids)

    # Load dataset for re-prompting (needs OCR for images)
    dataset = None
    if not NO_REPROMPT:
        print("\nLoading dataset with OCR for re-prompting...")
        try:
            dataset = Dataset(
                split_name="v2_250_gate",
                dataset_path=DATA_ROOT,
                load_annotations=True,
                load_ocr=True,
                docids=target_docids,
            )
            print(f"  Loaded {len(list(dataset))} docs with OCR")
        except Exception as e:
            print(f"  ⚠️  Could not load OCR: {e}")
            print("  Falling back to pure-Python mode (no re-prompting)")
            dataset = None

    # Apply AOL verifiers
    print("\nApplying AOL verifiers...")
    aol_preds = await apply_aol_verifiers(
        preds=dict(ensemble_250),
        model=MODEL,
        dataset=dataset,
        do_reprompt=(not NO_REPROMPT) and (dataset is not None),
    )

    # Save result
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_predictions(aol_preds, OUT_PATH)
    print(f"\nSaved → {OUT_PATH}")

    # Score AOL result
    print("\nScoring AOL result...")
    aol_kile, aol_lir = _score(aol_preds, "v2_aol_250 (AOL verifiers)", target_docids)

    delta_kile = aol_kile - baseline_kile
    delta_lir = aol_lir - baseline_lir

    print(f"\n{'='*65}")
    print(f"{'Variant':<40} {'KILE':>8} {'LIR':>8}")
    print(f"{'-'*65}")
    print(f"{'v2_ensemble_250 (baseline)':40} {baseline_kile:8.2f} {baseline_lir:8.2f}")
    print(f"{'v2_aol_250 (AOL verifiers)':40} {aol_kile:8.2f} {aol_lir:8.2f}")
    print(f"{'Delta':40} {delta_kile:+8.2f} {delta_lir:+8.2f}")
    print(f"{'='*65}")

    if delta_kile >= 1.0:
        print(f"\n✅ AOL +{delta_kile:.2f}pp KILE → run full 500-doc confirmation")
    elif delta_kile >= 0.3:
        print(f"\n⚠️  Marginal AOL gain (+{delta_kile:.2f}pp) — borderline, try 500-doc")
    elif delta_kile > -0.3:
        print(f"\n➖ Neutral ({delta_kile:+.2f}pp) — AOL adds no lift, overhead not worth it")
    else:
        print(f"\n❌ AOL hurts KILE ({delta_kile:.2f}pp) — verifiers are over-penalizing. BURY.")
        print("    Investigate: which fields are most demoted? Run with --no-reprompt to isolate.")


if __name__ == "__main__":
    asyncio.run(main())
