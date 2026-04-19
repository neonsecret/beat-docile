#!/usr/bin/env python
"""GutenOCR 50-doc eval: Mode A (replace OCR) and Mode B (augment V5b).

Spike: --limit 5 --mode a
Mode A 50-doc: --limit 50 --mode a
Mode B 50-doc: --limit 50 --mode b  (requires v5b_50.json as base)

Saves predictions to:
  predictions/gutenocr_mode_a_50.json
  predictions/gutenocr_mode_b_50.json

Results compared against V5b 41.79% KILE baseline.

Usage:
    uv run python tools/run_gutenocr_50.py --mode a --limit 5   # spike
    uv run python tools/run_gutenocr_50.py --mode a --limit 50
    uv run python tools/run_gutenocr_50.py --mode b --limit 50
    uv run python tools/run_gutenocr_50.py --mode a --limit 50 --fresh
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import os
os.environ.setdefault("DATA_ROOT", str(PROJECT_ROOT / "data"))

from docile.dataset import Dataset, Field

V5B_50_PATH = PROJECT_ROOT / "predictions" / "v5b_50.json"
OUT_MODE_A = PROJECT_ROOT / "predictions" / "gutenocr_mode_a_50.json"
OUT_MODE_B = PROJECT_ROOT / "predictions" / "gutenocr_mode_b_50.json"

V5B_KILE_BASELINE = 41.79
V5B_LIR_BASELINE = 49.90


def load_v5b_fields(docid: str, raw: dict) -> tuple[list[Field], list[Field]]:
    """Split saved V5b predictions into KILE and LIR Field lists."""
    kile, lir = [], []
    for fd in raw.get(docid, []):
        f = Field.from_dict(fd)
        if f.line_item_id is not None:
            lir.append(f)
        else:
            kile.append(f)
    return kile, lir


def fields_to_dicts(fields: list[Field]) -> list[dict]:
    return [f.to_dict() for f in fields]


def run_eval_and_print(
    all_results: dict[str, list[dict]],
    data_root: Path,
    mode_label: str,
) -> dict[str, float]:
    from beat_docile.eval import run_eval, print_scores

    kile_preds: dict[str, list[Field]] = {}
    lir_preds: dict[str, list[Field]] = {}
    for docid, fds in all_results.items():
        kile_preds[docid] = []
        lir_preds[docid] = []
        for fd in fds:
            f = Field.from_dict(fd)
            if f.line_item_id is not None:
                lir_preds[docid].append(f)
            else:
                kile_preds[docid].append(f)

    eval_dataset = Dataset(
        split_name="val",
        dataset_path=data_root,
        load_annotations=True,
        load_ocr=False,
        docids=list(all_results.keys()),
    )
    result = run_eval(eval_dataset, kile_preds, lir_preds)
    scores = print_scores(result)

    kile_ap = scores.get("kile_AP", 0.0) * 100
    lir_f1 = scores.get("lir_f1", 0.0) * 100

    print(f"\n{'='*60}")
    print(f"Mode:         {mode_label}")
    print(f"V5b baseline: KILE AP {V5B_KILE_BASELINE:.2f}%  /  LIR F1 {V5B_LIR_BASELINE:.2f}%")
    print(f"GutenOCR:     KILE AP {kile_ap:.2f}%  /  LIR F1 {lir_f1:.2f}%")
    print(f"Delta:        KILE {kile_ap - V5B_KILE_BASELINE:+.2f}pp  /  LIR {lir_f1 - V5B_LIR_BASELINE:+.2f}pp")
    print(f"{'='*60}\n")
    return scores


def run_mode_a(
    docs: list,
    extractor,
    out_path: Path,
    all_results: dict[str, list[dict]],
    data_root: Path,
) -> None:
    from beat_docile.data import iter_pages
    from beat_docile.gutenocr_extract import extract_document_mode_a

    total = len(docs)
    start_t = time.time()

    for i, doc in enumerate(docs):
        if doc.docid in all_results:
            continue
        try:
            with doc:
                fields = extract_document_mode_a(extractor, doc)
            all_results[doc.docid] = fields_to_dicts(fields)
        except Exception as e:
            print(f"  ERROR {doc.docid}: {e}")
            all_results[doc.docid] = []

        done = i + 1
        if done % 5 == 0 or done == total:
            elapsed = time.time() - start_t
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else float("inf")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(all_results, indent=2))
            print(f"  [{done}/{total}] {elapsed:.0f}s  rate={rate:.2f}doc/s  ETA={eta:.0f}s")

    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nMode A done → {out_path}")
    run_eval_and_print(all_results, data_root, "A (replace)")


def run_mode_b(
    docs: list,
    extractor,
    v5b_raw: dict,
    out_path: Path,
    all_results: dict[str, list[dict]],
    data_root: Path,
) -> None:
    from beat_docile.gutenocr_extract import augment_document_mode_b

    total = len(docs)
    start_t = time.time()

    for i, doc in enumerate(docs):
        if doc.docid in all_results:
            continue
        v5b_kile, v5b_lir = load_v5b_fields(doc.docid, v5b_raw)
        existing_fields = v5b_kile + v5b_lir

        try:
            with doc:
                augmented = augment_document_mode_b(extractor, doc, existing_fields)
            all_results[doc.docid] = fields_to_dicts(augmented)
        except Exception as e:
            print(f"  ERROR {doc.docid}: {e}")
            all_results[doc.docid] = fields_to_dicts(existing_fields)

        done = i + 1
        if done % 5 == 0 or done == total:
            elapsed = time.time() - start_t
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else float("inf")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(all_results, indent=2))
            print(f"  [{done}/{total}] {elapsed:.0f}s  rate={rate:.2f}doc/s  ETA={eta:.0f}s")

    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nMode B done → {out_path}")
    run_eval_and_print(all_results, data_root, "B (augment V5b)")


def main() -> None:
    parser = argparse.ArgumentParser(description="GutenOCR 50-doc eval")
    parser.add_argument("--mode", choices=["a", "b"], required=True, help="a=replace, b=augment")
    parser.add_argument("--limit", type=int, default=50, help="Number of val docs")
    parser.add_argument("--model-id", default="rootsautomation/GutenOCR-3B")
    parser.add_argument("--cache-dir", type=Path, default=None, help="HF model cache dir")
    parser.add_argument("--fresh", action="store_true", help="Ignore saved predictions")
    args = parser.parse_args()

    from beat_docile.config import DATA_ROOT
    from beat_docile.gutenocr_extract import GutenOCRExtractor

    out_path = OUT_MODE_A if args.mode == "a" else OUT_MODE_B
    mode_label = "A (replace)" if args.mode == "a" else "B (augment)"

    print(f"GutenOCR eval — Mode {mode_label}")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"Model: {args.model_id}")
    print(f"Output: {out_path}")
    print(f"Limit: {args.limit} docs\n")

    if args.mode == "b" and not V5B_50_PATH.exists():
        print(f"ERROR: v5b_50.json not found at {V5B_50_PATH}")
        print("Run v5b_full_val.py first.")
        sys.exit(1)

    # Load target docids from V5b 50-doc set for consistency
    if V5B_50_PATH.exists():
        v5b_raw = json.loads(V5B_50_PATH.read_text())
        target_docids = set(list(v5b_raw.keys())[:args.limit])
    else:
        v5b_raw = {}
        target_docids = None

    print("Loading val split...")
    dataset = Dataset(
        split_name="val",
        dataset_path=DATA_ROOT,
        load_annotations=True,
        load_ocr=True,
    )
    if target_docids:
        docs = [d for d in dataset if d.docid in target_docids]
    else:
        docs = list(dataset)[: args.limit]
    print(f"  {len(docs)} docs selected\n")

    # Resumability
    all_results: dict[str, list[dict]] = {}
    if not args.fresh and out_path.exists():
        all_results = json.loads(out_path.read_text())
        print(f"Resuming — already done: {len(all_results)} docs")

    remaining = [d for d in docs if d.docid not in all_results]
    print(f"Remaining: {len(remaining)} docs\n")

    # ── Spike check ──────────────────────────────────────────────────────────
    if args.limit <= 5:
        print("=== SPIKE MODE (≤5 docs) ===")
        print("Checking model load + output sanity only.\n")

    # ── Load model ───────────────────────────────────────────────────────────
    extractor = GutenOCRExtractor(model_id=args.model_id, cache_dir=args.cache_dir)

    # ── Run eval ─────────────────────────────────────────────────────────────
    if args.mode == "a":
        run_mode_a(remaining, extractor, out_path, all_results, DATA_ROOT)
    else:
        run_mode_b(remaining, extractor, v5b_raw, out_path, all_results, DATA_ROOT)

    if args.limit <= 5:
        print("\n=== SPIKE COMPLETE ===")
        print("Check output above for sensible JSON + non-empty field predictions.")
        print("If good, run with --limit 50 for full eval.")


if __name__ == "__main__":
    main()
