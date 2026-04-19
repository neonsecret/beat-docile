#!/usr/bin/env python
"""4-way ensemble: v2_t00 + v2_t03 + v2_alt + v2_gemini on 250-doc gate.

Scores all 4 variants individually + ensemble, compares to v2_t00 baseline.

Decision gate:
  ≥1pp KILE above v2_t00 → confirm with full 500-doc run
  <1pp                   → Gemini adds no useful diversity; bury

Requires: v2_t00_250.json, v2_t03_250.json, v2_alt_250.json, v2_gemini_250.json
Outputs:  predictions/v2_ensemble_4way_250.json

Usage:
    DATA_ROOT=data uv run python tools/ensemble_v2_4way_250.py
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
GEMINI_PATH = PROJECT_ROOT / "predictions" / "v2_gemini_250.json"
OUT_PATH = PROJECT_ROOT / "predictions" / "v2_ensemble_4way_250.json"


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
    return scores.get("kile_AP", 0) * 100, scores.get("lir_f1", 0) * 100


def main() -> None:
    target_docids: list[str] = json.loads(DOCIDS_PATH.read_text())
    print("4-way ensemble evaluation (250-doc gate)")
    print(f"Target docids: {len(target_docids)}")

    for p, name in [
        (T00_PATH, "v2_t00_250"), (T03_PATH, "v2_t03_250"),
        (ALT_PATH, "v2_alt_250"), (GEMINI_PATH, "v2_gemini_250"),
    ]:
        if not p.exists():
            print(f"Missing: {p} — run the extraction script first")
            return

    print("\nLoading predictions...")
    t00 = load_predictions(T00_PATH)
    t03 = load_predictions(T03_PATH)
    alt = load_predictions(ALT_PATH)
    gemini = load_predictions(GEMINI_PATH)
    print(f"  v2_t00:   {len(t00)} docs")
    print(f"  v2_t03:   {len(t03)} docs")
    print(f"  v2_alt:   {len(alt)} docs")
    print(f"  v2_gemini:{len(gemini)} docs")

    print("\n3-way ensemble (baseline)...")
    ens3 = merge_predictions(
        sources=[t00, t03, alt],
        weights=None,
        iou_threshold=0.5,
        score_combine="weighted_max",
    )

    print("4-way ensemble (+ Gemini)...")
    ens4 = merge_predictions(
        sources=[t00, t03, alt, gemini],
        weights=None,
        iou_threshold=0.5,
        score_combine="weighted_max",
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_predictions(ens4, OUT_PATH)
    print(f"  Saved → {OUT_PATH}")

    print("\nScoring all variants...")
    t00_kile, t00_lir = _score(t00, "v2_t00", target_docids)
    t03_kile, t03_lir = _score(t03, "v2_t03", target_docids)
    alt_kile, alt_lir = _score(alt, "v2_alt", target_docids)
    gmn_kile, gmn_lir = _score(gemini, "v2_gemini", target_docids)
    e3_kile, e3_lir = _score(ens3, "3-way ensemble", target_docids)
    e4_kile, e4_lir = _score(ens4, "4-way ensemble", target_docids)

    W = 38
    print(f"\n{'='*70}")
    print(f"{'Variant':<{W}} {'KILE':>8} {'LIR':>8} {'ΔKILE(vs t00)':>14} {'ΔKILE(vs 3way)':>14}")
    print(f"{'-'*70}")
    print(f"{'v2_t00 (T=1.0, baseline)':<{W}} {t00_kile:8.2f} {t00_lir:8.2f} {'—':>14} {'—':>14}")
    print(f"{'v2_t03 (T=0.3)':<{W}} {t03_kile:8.2f} {t03_lir:8.2f} {t03_kile-t00_kile:+14.2f} {'—':>14}")
    print(f"{'v2_alt (alt prompt)':<{W}} {alt_kile:8.2f} {alt_lir:8.2f} {alt_kile-t00_kile:+14.2f} {'—':>14}")
    print(f"{'v2_gemini (Gemini 3 Flash)':<{W}} {gmn_kile:8.2f} {gmn_lir:8.2f} {gmn_kile-t00_kile:+14.2f} {'—':>14}")
    print(f"{'-'*70}")
    print(f"{'3-way ensemble (Sonnet only)':<{W}} {e3_kile:8.2f} {e3_lir:8.2f} {e3_kile-t00_kile:+14.2f} {'—':>14}")
    print(f"{'4-way ensemble (+ Gemini)':<{W}} {e4_kile:8.2f} {e4_lir:8.2f} {e4_kile-t00_kile:+14.2f} {e4_kile-e3_kile:+14.2f}")
    print(f"{'='*70}")

    delta_vs_3way = e4_kile - e3_kile
    if delta_vs_3way >= 1.0:
        print(f"\n✅ 4-way +{delta_vs_3way:.2f}pp vs 3-way → Gemini adds diversity. Run full 500-doc.")
    elif delta_vs_3way >= 0.5:
        print(f"\n⚠️  Borderline ({delta_vs_3way:+.2f}pp vs 3-way) — borderline, user decides.")
    elif delta_vs_3way > 0.0:
        print(f"\n➖ Marginal ({delta_vs_3way:+.2f}pp vs 3-way) — within noise, bury Gemini variant.")
    else:
        print(f"\n❌ 4-way worse than 3-way ({delta_vs_3way:+.2f}pp) — Gemini adds noise. BURY.")


if __name__ == "__main__":
    main()
