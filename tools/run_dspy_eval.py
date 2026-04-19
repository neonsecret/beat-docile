#!/usr/bin/env python3
"""Evaluate the DSPy-optimized extraction module on val docs.

Loads the compiled DSPy program and runs it on N held-out val docs.
Computes full DocILE KILE AP and LIR F1 for direct comparison with V5b baseline.

Baseline (V5b, 50 docs): KILE AP=41.86%, LIR F1=52.36%
Baseline (V5b, full 500): KILE AP=41.79%, LIR F1=49.90%

Usage:
    DATA_ROOT=data uv run python tools/run_dspy_eval.py
    DATA_ROOT=data uv run python tools/run_dspy_eval.py --n-docs 50 --start-idx 300
    DATA_ROOT=data uv run python tools/run_dspy_eval.py --compiled models/dspy/compiled.json

Note: --start-idx should be >= --train-n from run_dspy_optimize.py (default 300)
to avoid evaluating on docs seen during optimization.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
os.environ.setdefault("DATA_ROOT", str(DATA_DIR))
os.environ.setdefault("BD_USE_REFINER", "0")
os.environ.setdefault("BD_USE_VALIDATOR", "0")

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from beat_docile.config import DATA_ROOT
from beat_docile.data import load_split, iter_pages
from beat_docile.dspy_optimizer import configure_dspy_lm, DocILEExtractionModule
import dspy
from beat_docile.extract import _words_to_prompt, _parse_response
from beat_docile.eval import run_eval, print_scores
from docile.dataset import Dataset, Field


def _image_to_b64(image) -> str:
    import base64, io
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DSPy-optimized extraction")
    parser.add_argument("--n-docs", type=int, default=50,
                        help="Number of val docs to run (default 50 for speed)")
    parser.add_argument("--start-idx", type=int, default=300,
                        help="Val doc start index (skip training docs, default 300)")
    parser.add_argument("--compiled", type=str, default=None,
                        help="Path to compiled DSPy JSON (default: models/dspy/compiled.json)")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--use-refiner", action="store_true",
                        help="Enable V5b refiner on DSPy outputs")
    args = parser.parse_args()

    if args.use_refiner:
        os.environ["BD_USE_REFINER"] = "1"
        os.environ["BD_USE_VALIDATOR"] = "1"

    # Prefer pkl (save_program=True format), fall back to json
    _default = PROJECT_ROOT / "models" / "dspy"
    if args.compiled:
        compiled_path = args.compiled
    elif (_default / "compiled_miprov2.json").exists():
        compiled_path = str(_default / "compiled_miprov2.json")
    else:
        compiled_path = str(_default / "compiled_miprov2.json")

    print("=== DSPy Optimized Module Eval ===")
    print(f"  DATA_ROOT:   {DATA_ROOT}")
    print(f"  Docs:        {args.n_docs} (starting at idx {args.start_idx})")
    print(f"  Compiled:    {compiled_path}")
    print(f"  Use refiner: {args.use_refiner}")

    # Configure DSPy — disable cache so Sonnet eval uses real API calls, not Haiku cache hits
    configure_dspy_lm(model=args.model, max_tokens=8192)
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

    # Load optimized module
    module = DocILEExtractionModule()
    compiled = Path(compiled_path)
    if compiled.exists():
        module.load(str(compiled))
        print("  Loaded optimized program ✓")
    else:
        print(f"  WARNING: {compiled_path} not found — using unoptimized baseline module")

    # Load val docs
    val_ds = load_split("val")
    all_val_docs = list(val_ds)
    eval_docs = all_val_docs[args.start_idx : args.start_idx + args.n_docs]
    eval_docids = [d.docid for d in eval_docs]
    print(f"\nRunning on {len(eval_docs)} docs (idx {args.start_idx}–{args.start_idx+len(eval_docs)-1})...")

    kile_preds: dict[str, list[Field]] = {}
    lir_preds: dict[str, list[Field]] = {}
    errors = 0
    t0 = time.time()

    for i, doc in enumerate(eval_docs):
        kile_preds[doc.docid] = []
        lir_preds[doc.docid] = []
        try:
            with doc:
                for page in iter_pages(doc):
                    image_b64 = _image_to_b64(page.image)
                    words_layout = _words_to_prompt(page.words)

                    pred = module(words_layout=words_layout, image_b64=image_b64)
                    fields_json = getattr(pred, "fields_json", "") or ""

                    kile, lir = _parse_response(fields_json, page.words, page.page_index)
                    kile_preds[doc.docid].extend(kile)
                    lir_preds[doc.docid].extend(lir)
        except Exception as e:
            errors += 1
            print(f"  Error on {doc.docid}: {e}")

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(eval_docs) - i - 1) / rate
            print(f"  {i+1}/{len(eval_docs)} docs ({rate:.1f}/s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"\nExtraction complete: {len(eval_docs)} docs in {elapsed:.0f}s ({errors} errors)")

    # Evaluate using DocILE subset dataset (docids= param avoids full-dataset AP penalty)
    print("\nRunning DocILE evaluation...")
    subset_ds = Dataset(
        split_name="dspy_eval_subset",
        dataset_path=DATA_ROOT,
        load_annotations=True,
        load_ocr=False,
        docids=eval_docids,
    )
    result = run_eval(subset_ds, kile_preds, lir_preds)

    print("\n=== Results ===")
    scores = print_scores(result)

    kile_ap = scores.get("kile_AP", 0.0)
    lir_f1 = scores.get("lir_f1", 0.0)

    print(f"\n{'=' * 50}")
    print(f"DSPy optimized ({args.n_docs} docs):")
    print(f"  KILE AP: {kile_ap*100:.2f}%")
    print(f"  LIR F1:  {lir_f1*100:.2f}%")
    print(f"\nV5b baseline (50 docs):")
    print(f"  KILE AP: 41.86%")
    print(f"  LIR F1:  52.36%")
    print(f"\nDelta:")
    print(f"  KILE AP: {(kile_ap - 0.4186)*100:+.2f}pp")
    print(f"  LIR F1:  {(lir_f1 - 0.5236)*100:+.2f}pp")
    print(f"{'=' * 50}")

    # Also show the optimized instruction if available
    try:
        from beat_docile.optimized_prompt import OPTIMIZED_INSTRUCTION, OPTIMIZATION_SCORE
        if OPTIMIZED_INSTRUCTION:
            print(f"\nMIPROv2 train-time field F1 score: {OPTIMIZATION_SCORE:.4f}")
            print(f"Optimized instruction (first 300 chars):")
            print(OPTIMIZED_INSTRUCTION[:300])
    except ImportError:
        pass


if __name__ == "__main__":
    main()
