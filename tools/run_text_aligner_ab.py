#!/usr/bin/env python
"""V6 text-aligner A/B driver: text-only Claude extraction + precise alignment.

Runs on the same 50 val docids as v5b_50.json and evaluates KILE AP + LIR F1.
If KILE AP > 45%, automatically runs on full 500 val.

Baseline: V5b-50 → KILE 41.86% / LIR 52.36%
Target:   KILE > 45% (precision lift from better alignment)

Usage:
    DATA_ROOT=data uv run python tools/run_text_aligner_ab.py
    DATA_ROOT=data uv run python tools/run_text_aligner_ab.py --full  # force 500
    DATA_ROOT=data uv run python tools/run_text_aligner_ab.py --no-few-shot
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
os.environ.setdefault("DATA_ROOT", str(DATA_DIR))
os.environ["BD_USE_REFINER"] = "0"   # alignment already precise; refiner may hurt
os.environ["BD_USE_VALIDATOR"] = "1"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from docile.dataset import Dataset, Field  # noqa: E402

from beat_docile.config import DATA_ROOT, DEFAULT_MODEL  # noqa: E402
from beat_docile.data import load_split  # noqa: E402
from beat_docile.fewshot import _build_cluster_index  # noqa: E402
from beat_docile.text_extract import extract_documents_text  # noqa: E402

BASELINE_KILE = 41.86
BASELINE_LIR = 52.36
GATE_KILE = 45.0

MODEL = DEFAULT_MODEL
V5B_50_PATH = PROJECT_ROOT / "predictions" / "v5b_50.json"
OUT_50 = PROJECT_ROOT / "predictions" / "v6_textalign_val_50.json"
OUT_500 = PROJECT_ROOT / "predictions" / "v6_textalign_val_500.json"


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fields_to_dicts(fields: list[Field]) -> list[dict]:
    return [f.to_dict() for f in fields]


def _run_eval(docids: list[str], kile_preds: dict, lir_preds: dict) -> dict[str, float]:
    """Run KILE + LIR eval on a subset. Returns {kile_ap, lir_f1}."""
    from beat_docile.eval import print_scores, run_eval

    subset_ds = Dataset(
        split_name="eval_subset",
        dataset_path=DATA_ROOT,
        load_annotations=True,
        load_ocr=False,
        docids=docids,
    )
    result = run_eval(subset_ds, kile_preds, lir_preds)
    scores = print_scores(result)
    return scores


def _save_predictions(
    docids: list[str],
    kile_preds: dict[str, list[Field]],
    lir_preds: dict[str, list[Field]],
    out_path: Path,
) -> None:
    output = {}
    for docid in docids:
        fields_out = []
        for f in kile_preds.get(docid, []):
            fields_out.append(_fields_to_dicts([f])[0])
        for f in lir_preds.get(docid, []):
            fields_out.append(_fields_to_dicts([f])[0])
        output[docid] = fields_out
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Extraction runner
# ─────────────────────────────────────────────────────────────────────────────

def _build_few_shot_cache(
    docs: list,
    train_index: dict | None,
) -> dict[int, list[dict]] | None:
    """Pre-build the cluster few-shot cache once for all docs."""
    if train_index is None:
        return None
    from beat_docile.fewshot import build_few_shot_messages, load_few_shot_examples

    unique_cids = []
    for doc in docs:
        try:
            cid = doc.annotation.cluster_id
            if cid is not None:
                unique_cids.append(cid)
        except Exception:
            pass
    unique_cids = list(set(unique_cids))
    if not unique_cids:
        return {}

    print(f"  Building few-shot cache for {len(unique_cids)} clusters...")
    examples_by_cluster = load_few_shot_examples(unique_cids, train_index, max_per_cluster=1)
    cache = {
        cid: build_few_shot_messages(examples)
        for cid, examples in examples_by_cluster.items()
    }
    print(f"  Cache built: {len(cache)} clusters with examples")
    return cache


async def run_extraction(
    docs: list,
    train_index: dict | None,
    label: str,
) -> tuple[dict[str, list[Field]], dict[str, list[Field]]]:
    """Run text-aligner pipeline on docs in chunks; return (kile_preds, lir_preds).

    Passes train_index to each chunk so extract_documents_text builds the few-shot
    cache per batch. The first batch computes OCR snapping (slow); subsequent batches
    reuse the cached snapping (fast). This is faster than pre-building all at once.
    """
    kile_all: dict[str, list[Field]] = {}
    lir_all: dict[str, list[Field]] = {}
    done = 0
    total = len(docs)
    t0 = time.time()

    chunk_size = 10
    chunks = [docs[i:i + chunk_size] for i in range(0, len(docs), chunk_size)]

    for chunk in chunks:
        kile_batch, lir_batch = await extract_documents_text(
            chunk,
            model=MODEL,
            train_index=train_index,
        )
        kile_all.update(kile_batch)
        lir_all.update(lir_batch)
        done += len(chunk)
        elapsed = time.time() - t0
        eta = (elapsed / done) * (total - done) if done < total else 0
        print(
            f"  [{label}] {done}/{total} done"
            f" | elapsed {elapsed:.0f}s | ETA {eta:.0f}s"
            f" | {sum(len(v) for v in kile_all.values())} KILE"
            f" | {sum(len(v) for v in lir_all.values())} LIR"
        )

    return kile_all, lir_all


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    print(f"Model: {MODEL}")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print("BD_USE_VALIDATOR=1, BD_USE_REFINER=0")
    print(f"Baseline: KILE {BASELINE_KILE}% / LIR {BASELINE_LIR}%\n")

    # ── Load val dataset ──────────────────────────────────────────────────────
    print("Loading val split...")
    val_ds = load_split("val")
    all_val_docs = list(val_ds)
    all_val_docids = [d.docid for d in all_val_docs]
    doc_by_id = {d.docid: d for d in all_val_docs}

    # ── Get 50-doc subset (same as v5b_50.json) ───────────────────────────────
    if not V5B_50_PATH.exists():
        print(f"ERROR: {V5B_50_PATH} not found — cannot determine 50-doc subset")
        sys.exit(1)

    v5b_50 = json.loads(V5B_50_PATH.read_text())
    docids_50 = list(v5b_50.keys())
    print(f"50-doc subset: {len(docids_50)} docids from {V5B_50_PATH.name}")

    # ── Few-shot setup ────────────────────────────────────────────────────────
    train_index = None
    if not args.no_few_shot:
        print("Building train cluster index for few-shot...")
        train_index = _build_cluster_index("train")
        print(f"  {len(train_index)} clusters loaded")

    # ── Phase 1: 50-doc A/B ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 1: 50-doc A/B run")
    print(f"{'='*60}")

    docs_50 = [doc_by_id[did] for did in docids_50 if did in doc_by_id]
    if len(docs_50) != len(docids_50):
        missing = set(docids_50) - set(doc_by_id)
        print(f"WARNING: {len(missing)} docids not found in val split")

    t1 = time.time()
    kile_50, lir_50 = await run_extraction(docs_50, train_index, label="50-doc")
    elapsed_50 = time.time() - t1

    print(f"\n50-doc extraction done in {elapsed_50:.0f}s")

    # Fill in any missing docids with empty lists (required by eval)
    for did in docids_50:
        kile_50.setdefault(did, [])
        lir_50.setdefault(did, [])

    _save_predictions(docids_50, kile_50, lir_50, OUT_50)

    print("\nEvaluating 50-doc subset...")
    scores_50 = _run_eval(docids_50, kile_50, lir_50)

    kile_ap = scores_50.get("kile_AP", 0.0) * 100
    lir_f1 = scores_50.get("lir_f1", 0.0) * 100

    print(f"\n{'='*60}")
    print("50-doc RESULTS")
    print(f"  text-aligner: KILE {kile_ap:.2f}% / LIR {lir_f1:.2f}%")
    print(f"  V5b baseline: KILE {BASELINE_KILE:.2f}% / LIR {BASELINE_LIR:.2f}%")
    delta_kile = kile_ap - BASELINE_KILE
    delta_lir = lir_f1 - BASELINE_LIR
    print(f"  Delta:        KILE {delta_kile:+.2f}pp / LIR {delta_lir:+.2f}pp")
    print(f"{'='*60}")

    # ── Decision gate ─────────────────────────────────────────────────────────
    run_full = args.full or (kile_ap >= GATE_KILE)

    if kile_ap < 35.0:
        print(f"\nWARNING: KILE {kile_ap:.2f}% < 35% — something is wrong.")
        print("Check alignment confidence distribution and per-field failures.")
        if not args.full:
            print("Aborting full-500 run. Use --full to force.")
            return

    if not run_full:
        print(f"\nKILE {kile_ap:.2f}% < gate {GATE_KILE}% — skipping full-500 run.")
        print("Debug alignment issues before proceeding.")
        return

    # ── Phase 2: Full 500-doc run ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PHASE 2: Full 500-doc run (KILE {kile_ap:.2f}% >= gate {GATE_KILE}%)")
    print(f"{'='*60}")

    t2 = time.time()
    kile_500, lir_500 = await run_extraction(all_val_docs, train_index, label="500-doc")
    elapsed_500 = time.time() - t2

    print(f"\n500-doc extraction done in {elapsed_500:.0f}s ({elapsed_500/60:.1f}min)")

    for did in all_val_docids:
        kile_500.setdefault(did, [])
        lir_500.setdefault(did, [])

    _save_predictions(all_val_docids, kile_500, lir_500, OUT_500)

    print("\nEvaluating full 500-doc val split...")
    scores_500 = _run_eval(all_val_docids, kile_500, lir_500)

    kile_ap_500 = scores_500.get("kile_AP", 0.0) * 100
    lir_f1_500 = scores_500.get("lir_f1", 0.0) * 100

    print(f"\n{'='*60}")
    print("500-doc FINAL RESULTS")
    print(f"  text-aligner: KILE {kile_ap_500:.2f}% / LIR {lir_f1_500:.2f}%")
    print("  V5b (full):   KILE 41.79% / LIR 49.90%")
    delta_kile_500 = kile_ap_500 - 41.79
    delta_lir_500 = lir_f1_500 - 49.90
    print(f"  Delta:        KILE {delta_kile_500:+.2f}pp / LIR {delta_lir_500:+.2f}pp")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V6 text-aligner A/B evaluation")
    parser.add_argument("--full", action="store_true", help="Force full 500-doc run regardless of gate")
    parser.add_argument("--no-few-shot", action="store_true", help="Disable cluster-based few-shot")
    args = parser.parse_args()
    asyncio.run(main(args))
