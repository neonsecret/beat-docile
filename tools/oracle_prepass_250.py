#!/usr/bin/env python
"""M5 — Oracle pre-pass on 250-doc gate.

Runs main extraction with BD_USE_ORACLE_PREPASS=1 (oracle regex/checksum
candidates injected into Claude prompt as hints before OCR words).

Then ensembles this oracle-prepass run with the existing 3 v2 variants
(v2_t00_250, v2_t03_250, v2_alt_250) for a 4-way ensemble, and compares
to the existing 3-way v2_ensemble_250.

Decision gate:
  >=1pp KILE lift over v2_ensemble_250 → run full 500-doc confirm
  <1pp                                 → bury (pre-pass + post-pass both tested)

Usage:
    DATA_ROOT=data uv run python tools/oracle_prepass_250.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
os.environ.setdefault("DATA_ROOT", str(DATA_DIR))

# Oracle prepass enabled; keep other flags at v2 defaults (T=1.0, standard prompt)
os.environ["BD_USE_ORACLE_PREPASS"] = "1"
os.environ["BD_USE_REFINER"] = "0"
os.environ["BD_USE_VALIDATOR"] = "0"
os.environ["BD_USE_BBOX_VERIFY"] = "0"
os.environ["BD_USE_REFINER_GUARD"] = "0"
os.environ["BD_TEMPERATURE"] = "1.0"
os.environ["BD_ALT_PROMPT"] = "0"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from beat_docile.config import DEFAULT_MODEL, DATA_ROOT
from beat_docile.data import load_split, iter_pages
from beat_docile.extract import extract_page
from beat_docile.fewshot import _build_cluster_index, load_few_shot_examples, build_few_shot_messages
from beat_docile.ensemble import load_predictions, merge_predictions, save_predictions
from docile.dataset import Field, Dataset

DOCIDS_PATH = PROJECT_ROOT / "tools" / "val_250_docids.json"
MAX_WORKERS = 8
PROGRESS_INTERVAL = 25
MODEL = DEFAULT_MODEL

# New oracle-prepass raw predictions (single T=1.0 run)
RAW_PATH = PROJECT_ROOT / "predictions" / "oracle_prepass_250_raw.json"
# 4-way ensemble output
OUT_PATH = PROJECT_ROOT / "predictions" / "v2_ensemble_oracle_prepass_250.json"
# Existing 3-way ensemble baseline
V2_ENS_PATH = PROJECT_ROOT / "predictions" / "v2_ensemble_250.json"
# Existing 3 variant files for ensembling
T00_PATH = PROJECT_ROOT / "predictions" / "v2_t00_250.json"
T03_PATH = PROJECT_ROOT / "predictions" / "v2_t03_250.json"
ALT_PATH = PROJECT_ROOT / "predictions" / "v2_alt_250.json"


def _fields_to_dicts(fields: list[Field]) -> list[dict]:
    return [f.to_dict() for f in fields]


def _score(preds: dict[str, list[Field]], target_docids: list[str]) -> tuple[float, float]:
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


async def run_batch() -> None:
    target_docids: list[str] = json.loads(DOCIDS_PATH.read_text())
    print(f"M5 — Oracle pre-pass 250-doc gate")
    print(f"Target docids: {len(target_docids)}")
    print(f"Model: {MODEL}")
    print(f"Raw output: {RAW_PATH}")
    print(f"Final ensemble: {OUT_PATH}")
    print(f"Flags: BD_USE_ORACLE_PREPASS=1 BD_TEMPERATURE=1.0 BD_ALT_PROMPT=0\n")

    print("Loading val split...")
    dataset = load_split("val")
    target_set = set(target_docids)
    docs = [d for d in dataset if d.docid in target_set]
    docs.sort(key=lambda d: target_docids.index(d.docid))
    print(f"Loaded {len(docs)} target docs")

    print("Building few-shot cache...")
    train_index = _build_cluster_index("train")
    unique_cluster_ids = list({doc.annotation.cluster_id for doc in docs
                                if doc.annotation.cluster_id is not None})
    examples_by_cluster = load_few_shot_examples(unique_cluster_ids, train_index, max_per_cluster=1)
    few_shot_cache: dict[int, list[dict]] = {
        cid: build_few_shot_messages(examples)
        for cid, examples in examples_by_cluster.items()
    }
    print(f"Few-shot cache: {len(few_shot_cache)} clusters\n")

    all_results: dict[str, list[dict]] = {}
    if RAW_PATH.exists():
        all_results = json.loads(RAW_PATH.read_text())
        print(f"Resuming — already done: {len(all_results)} docs")

    remaining = [d for d in docs if d.docid not in all_results]
    total = len(docs)
    completed = len(all_results)
    print(f"Remaining: {len(remaining)} docs\n")

    if remaining:
        sem = asyncio.Semaphore(MAX_WORKERS)
        errors: list[str] = []
        start_t = time.time()

        async def process_doc(doc) -> None:
            nonlocal completed
            async with sem:
                try:
                    fs_messages = few_shot_cache.get(doc.annotation.cluster_id)
                    pages = list(iter_pages(doc))
                    results = await asyncio.gather(
                        *[extract_page(p, MODEL, few_shot_messages=fs_messages) for p in pages]
                    )
                    fields: list[Field] = []
                    for k, l in results:
                        fields.extend(k)
                        fields.extend(l)
                    all_results[doc.docid] = _fields_to_dicts(fields)
                except Exception as e:
                    errors.append(f"{doc.docid}: {e}")
                    all_results[doc.docid] = []

                completed += 1
                if completed % PROGRESS_INTERVAL == 0 or completed == total:
                    elapsed = time.time() - start_t
                    n_done = completed - (total - len(remaining))
                    rate = n_done / elapsed if elapsed > 0 else 0
                    eta = (total - completed) / rate if rate > 0 else float("inf")
                    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
                    RAW_PATH.write_text(json.dumps(all_results, indent=2))
                    print(f"[{completed}/{total}] {elapsed:.0f}s elapsed, "
                          f"{rate:.2f} docs/s, ETA {eta:.0f}s — saved")

        await asyncio.gather(*[process_doc(doc) for doc in remaining])
        RAW_PATH.write_text(json.dumps(all_results, indent=2))
        if errors:
            print(f"\nErrors ({len(errors)}): {errors[:5]}")

    # ── Build 4-way ensemble: oracle_prepass + 3 existing v2 variants ──────────
    print("\nBuilding 4-way ensemble (oracle_prepass + v2_t00 + v2_t03 + v2_alt)...")
    missing = [p for p in (T00_PATH, T03_PATH, ALT_PATH) if not p.exists()]
    if missing:
        print(f"  Missing variant files: {missing}")
        print("  Cannot build 4-way ensemble. Scoring raw oracle_prepass only.")
        oracle_preds = load_predictions(RAW_PATH)
        ens_preds = oracle_preds
    else:
        oracle_preds = load_predictions(RAW_PATH)
        t00 = load_predictions(T00_PATH)
        t03 = load_predictions(T03_PATH)
        alt = load_predictions(ALT_PATH)
        ens_preds = merge_predictions(
            sources=[oracle_preds, t00, t03, alt],
            weights=None,
            iou_threshold=0.5,
            score_combine="weighted_max",
        )
        print(f"  4-way ensemble: {sum(len(v) for v in ens_preds.values())} fields")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_predictions(ens_preds, OUT_PATH)
    print(f"  Saved → {OUT_PATH}")

    # ── Score ─────────────────────────────────────────────────────────────────
    print("\nScoring...")
    oracle_kile, oracle_lir = _score(oracle_preds, target_docids)
    ens_kile, ens_lir = _score(ens_preds, target_docids)

    # Baseline: v2_ensemble_250 filtered to same docids
    baseline_kile = baseline_lir = 0.0
    if V2_ENS_PATH.exists():
        v2_ens = load_predictions(V2_ENS_PATH)
        v2_ens_filtered = {d: v2_ens.get(d, []) for d in target_docids}
        baseline_kile, baseline_lir = _score(v2_ens_filtered, target_docids)
    else:
        print(f"  ⚠️  Baseline {V2_ENS_PATH} not found — cannot compare")

    print(f"\n{'='*72}")
    print(f"{'Variant':<40} {'KILE':>8} {'LIR':>8} {'ΔKILE':>8} {'ΔLIR':>8}")
    print(f"{'-'*72}")
    print(f"{'v2_ensemble_250 (baseline)':40} {baseline_kile:8.2f} {baseline_lir:8.2f} {'—':>8} {'—':>8}")
    print(f"{'oracle_prepass_250_raw (single run)':40} {oracle_kile:8.2f} {oracle_lir:8.2f} "
          f"{oracle_kile-baseline_kile:+8.2f} {oracle_lir-baseline_lir:+8.2f}")
    print(f"{'v2_ensemble_oracle_prepass (4-way)':40} {ens_kile:8.2f} {ens_lir:8.2f} "
          f"{ens_kile-baseline_kile:+8.2f} {ens_lir-baseline_lir:+8.2f}")
    print(f"{'='*72}")

    delta = ens_kile - baseline_kile
    if delta >= 1.0:
        print(f"\n✅ +{delta:.2f}pp KILE — run full 500-doc confirmation")
    elif delta >= 0.5:
        print(f"\n⚠️  Borderline +{delta:.2f}pp — likely noise, skip 500-doc")
    elif delta > 0:
        print(f"\n➖ Marginal +{delta:.2f}pp — noise, BURY oracle pre-pass")
    else:
        print(f"\n❌ {delta:.2f}pp — oracle pre-pass hurts or neutral. BURY.")
        print("   Both post-pass and pre-pass now tested. Move to next experiment.")


if __name__ == "__main__":
    asyncio.run(run_batch())
