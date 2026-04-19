#!/usr/bin/env python
"""Recall-augmentation AOL on 250-doc gate.

Loads v2_ensemble predictions, builds per-cluster KILE field prior from train
annotations, re-prompts Claude for any expected fields that v2_ensemble missed,
adds them (never modifying existing predictions).

250-doc gate decision:
  ≥1pp KILE lift → run 500-doc confirmation
  0–1pp          → noise band, don't run 500-doc
  Negative       → FPs from re-prompting; investigate per-field

Usage:
    DATA_ROOT=data uv run python tools/run_recall_aol_250.py
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
# Recall-AOL is post-processing — base extraction flags don't apply here
os.environ["BD_USE_REFINER"] = "0"
os.environ["BD_USE_VALIDATOR"] = "0"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from beat_docile.config import DEFAULT_MODEL, DATA_ROOT
from beat_docile.ensemble import load_predictions, save_predictions
from beat_docile.recall_aol import build_cluster_field_prior, apply_recall_aol
from docile.dataset import Field, Dataset

DOCIDS_PATH = PROJECT_ROOT / "tools" / "val_250_docids.json"
ENSEMBLE_PATH = PROJECT_ROOT / "predictions" / "v2_ensemble_500.json"
OUT_PATH = PROJECT_ROOT / "predictions" / "v2_ensemble_recall_aol_250.json"

MODEL = DEFAULT_MODEL


def _score(
        preds: dict[str, list[Field]],
        label: str,
        target_docids: list[str],
) -> tuple[float, float]:
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
    print("Recall-augmentation AOL — 250-doc gate")
    print(f"Model:  {MODEL}")
    print(f"Target: {len(target_docids)} docs")

    if not ENSEMBLE_PATH.exists():
        print(f"❌ {ENSEMBLE_PATH} not found — run ensemble_v2_variants.py first")
        return

    # Load ensemble, filter to 250 docs
    print("\nLoading v2_ensemble predictions...")
    all_ensemble = load_predictions(ENSEMBLE_PATH)
    target_set = set(target_docids)
    ensemble_250 = {d: fields for d, fields in all_ensemble.items() if d in target_set}
    print(f"  Loaded {len(ensemble_250)} / {len(target_docids)} target docs")

    if len(ensemble_250) < len(target_docids):
        missing = [d for d in target_docids if d not in ensemble_250]
        print(f"  ⚠️  Missing {len(missing)} docs: {missing[:3]}...")

    # Baseline score
    print("\nBaseline: v2_ensemble on 250 docs")
    baseline_kile, baseline_lir = _score(ensemble_250, "v2_ensemble_250 (baseline)", target_docids)

    # Build cluster field prior from train (no OCR needed — fast)
    print("\nBuilding cluster field prior from train annotations...")
    train_ds = Dataset(
        split_name="train",
        dataset_path=DATA_ROOT,
        load_annotations=True,
        load_ocr=False,
    )
    field_prior = build_cluster_field_prior(train_ds)
    n_clusters = len(field_prior)
    all_sizes = [len(v) for v in field_prior.values()]
    avg_size = sum(all_sizes) / len(all_sizes) if all_sizes else 0
    print(f"  Built prior for {n_clusters} clusters")
    print(f"  Avg expected fields/cluster: {avg_size:.1f}  "
          f"(min={min(all_sizes) if all_sizes else 0}, max={max(all_sizes) if all_sizes else 0})")

    # Load val docs with OCR for page images
    print("\nLoading 250 val docs with OCR (for re-prompting)...")
    val_ds = Dataset(
        split_name="v2_250_gate",
        dataset_path=DATA_ROOT,
        load_annotations=True,
        load_ocr=True,
        docids=target_docids,
    )
    n_val_docs = sum(1 for _ in val_ds)
    print(f"  Loaded {n_val_docs} docs")

    # Re-load iterator (consumed by count above)
    val_ds = Dataset(
        split_name="v2_250_gate",
        dataset_path=DATA_ROOT,
        load_annotations=True,
        load_ocr=True,
        docids=target_docids,
    )

    # Apply recall augmentation
    print("\nApplying recall-augmentation AOL (MAX_WORKERS=4)...")
    augmented, stats = await apply_recall_aol(
        preds=dict(ensemble_250),
        model=MODEL,
        val_dataset=val_ds,
        field_prior=field_prior,
        max_workers=4,
    )

    print(f"\nStats:")
    print(f"  Docs with cluster prior:       {stats['docs_with_cluster']}")
    print(f"  Docs skipped (no cluster):     {stats['docs_skipped_no_cluster']}")
    print(f"  Docs with missing fields:      {stats['docs_with_missing_fields']}")
    print(f"  Missing field-type slots:      {stats['total_missing_field_type_slots']}")
    print(f"  Total re-prompts (pages):      {stats['total_reprompts']}")
    print(f"  Total fields added:            {stats['total_added']}")
    if stats['total_reprompts'] > 0:
        hit_rate = stats['total_added'] / stats['total_reprompts']
        print(f"  Added per reprompt:            {hit_rate:.2f}")

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_predictions(augmented, OUT_PATH)
    print(f"\nSaved → {OUT_PATH}")

    # Score
    print("\nScoring recall-augmented result...")
    aol_kile, aol_lir = _score(augmented, "v2_ensemble_recall_aol_250", target_docids)

    delta_kile = aol_kile - baseline_kile
    delta_lir = aol_lir - baseline_lir

    print(f"\n{'='*65}")
    print(f"{'Variant':<40} {'KILE':>8} {'LIR':>8}")
    print(f"{'-'*65}")
    print(f"{'v2_ensemble_250 (baseline)':40} {baseline_kile:8.2f} {baseline_lir:8.2f}")
    print(f"{'v2_ensemble_recall_aol_250':40} {aol_kile:8.2f} {aol_lir:8.2f}")
    print(f"{'Delta':40} {delta_kile:+8.2f} {delta_lir:+8.2f}")
    print(f"{'='*65}")

    if delta_kile >= 1.0:
        print(f"\n✅ Recall-AOL +{delta_kile:.2f}pp KILE → run full 500-doc confirmation")
    elif delta_kile >= 0.0:
        print(f"\n➖ Marginal/neutral ({delta_kile:+.2f}pp) — noise band, don't run 500-doc")
    else:
        print(f"\n❌ Recall-AOL hurts ({delta_kile:.2f}pp) — added predictions are FPs")
        print("    Investigate: check stats above — which fieldtypes were added most?")
        print("    Consider raising presence_threshold or tightening the prompt.")


if __name__ == "__main__":
    asyncio.run(main())
