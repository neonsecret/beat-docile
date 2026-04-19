#!/usr/bin/env python
"""Eval: Code-Factory vs V5b on the same 50 val docs.

For each val doc:
  - If Code-Factory script exists for its cluster → use CF for KILE
  - Otherwise → use V5b for KILE
  - LIR always uses V5b
  - Targeted financial pass always runs (both paths)

Results saved to predictions/code_factory_50.json.
Eval output: KILE AP and LIR F1 vs V5b 41.79% baseline.

Usage:
    uv run python tools/run_code_factory_50.py
    uv run python tools/run_code_factory_50.py --fresh   # ignore saved predictions
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
sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("DATA_ROOT", str(PROJECT_ROOT / "data"))
os.environ["BD_USE_REFINER"] = "1"
os.environ["BD_USE_VALIDATOR"] = "1"
os.environ["BD_USE_BBOX_VERIFY"] = "0"

from beat_docile.config import DEFAULT_MODEL, DATA_ROOT
from beat_docile.data import WordBox, iter_pages
from beat_docile.extract import extract_page, extract_page_targeted, _KILE_TYPES, _LIR_TYPES
from beat_docile.fewshot import _build_cluster_index, load_few_shot_examples, build_few_shot_messages
from beat_docile.code_factory import has_script, run_script, results_to_fields, load_metadata
from docile.dataset import Dataset, Field

MODEL = DEFAULT_MODEL
MAX_WORKERS = 6
V5B_50_PATH = PROJECT_ROOT / "predictions" / "v5b_50.json"
OUT_PATH = PROJECT_ROOT / "predictions" / "code_factory_50.json"


def _fields_to_dicts(fields: list[Field]) -> list[dict]:
    return [f.to_dict() for f in fields]


async def process_doc(
    doc,
    few_shot_cache: dict[int, list[dict]],
    sem: asyncio.Semaphore,
) -> tuple[str, list[dict]]:
    """Extract fields for one doc. Returns (docid, fields_dicts)."""
    async with sem:
        cluster_id: int | None = None
        try:
            cluster_id = doc.annotation.cluster_id
        except Exception:
            pass

        pages = list(iter_pages(doc))
        fs_messages = few_shot_cache.get(cluster_id) if cluster_id is not None else None

        # ── Code-Factory path ───────────────────────────────────────────────
        cf_kile: list[Field] = []
        used_cf = False

        if cluster_id is not None and has_script(cluster_id):
            for page in pages:
                results = run_script(cluster_id, page.words, timeout=5.0)
                if results:
                    used_cf = True
                    kile, lir = results_to_fields(results, page.words, page.page_index)
                    cf_kile.extend(kile)
                    # Note: CF scripts primarily target KILE — LIR handled by V5b below

        # ── V5b path (always for LIR; for KILE if CF missed or no script) ──
        main_tasks = [
            extract_page(p, MODEL, few_shot_messages=fs_messages) for p in pages
        ]
        targeted_tasks = [extract_page_targeted(p, MODEL) for p in pages]
        all_results = await asyncio.gather(*main_tasks, *targeted_tasks)
        n = len(pages)

        v5b_kile: list[Field] = []
        v5b_lir: list[Field] = []
        for kile, lir in all_results[:n]:
            v5b_kile.extend(kile)
            v5b_lir.extend(lir)
        for fields in all_results[n:]:
            for f in fields:
                if f.line_item_id is not None:
                    v5b_lir.append(f)
                else:
                    v5b_kile.append(f)

        # ── Merge ────────────────────────────────────────────────────────────
        if used_cf and cf_kile:
            # CF provided KILE predictions → use CF for KILE, V5b for LIR
            final_kile = cf_kile
        else:
            # No CF hit → full V5b
            final_kile = v5b_kile

        final_fields = final_kile + v5b_lir
        return doc.docid, _fields_to_dicts(final_fields)


async def run_batch(fresh: bool = False) -> None:
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"Model: {MODEL}")
    print(f"Output: {OUT_PATH}")

    if not V5B_50_PATH.exists():
        print(f"ERROR: v5b_50.json not found at {V5B_50_PATH}")
        print("Run tools/v5b_full_val.py or a 50-doc subset first.")
        sys.exit(1)

    target_docids = set(json.loads(V5B_50_PATH.read_text()).keys())
    print(f"Target docids: {len(target_docids)}")

    # Print Code-Factory coverage stats
    meta = load_metadata()
    print(f"Code-Factory scripts available: {len(meta)}")

    print("Loading val split...")
    dataset = Dataset(
        split_name="val",
        dataset_path=DATA_ROOT,
        load_annotations=True,
        load_ocr=False,
    )
    docs = [d for d in dataset if d.docid in target_docids]
    print(f"Matched {len(docs)} docs")

    print("Building few-shot cache...")
    train_index = _build_cluster_index("train")
    unique_cids = list({
        d.annotation.cluster_id for d in docs
        if d.annotation.cluster_id is not None
    })
    examples_by_cid = load_few_shot_examples(unique_cids, train_index, max_per_cluster=1)
    few_shot_cache: dict[int, list[dict]] = {
        cid: build_few_shot_messages(exs)
        for cid, exs in examples_by_cid.items()
    }
    print(f"  Few-shot cache: {len(few_shot_cache)} clusters")

    # CF coverage preview
    cf_docs = [d for d in docs if has_script(d.annotation.cluster_id or -1)]
    print(f"  Code-Factory coverage: {len(cf_docs)}/{len(docs)} docs ({len(cf_docs)/max(1,len(docs)):.0%})")

    # Resumability
    all_results: dict[str, list[dict]] = {}
    if not fresh and OUT_PATH.exists():
        all_results = json.loads(OUT_PATH.read_text())
        print(f"Resuming — already done: {len(all_results)} docs")

    remaining = [d for d in docs if d.docid not in all_results]
    total = len(docs)
    completed = len(all_results)
    print(f"Remaining: {len(remaining)} docs\n")

    sem = asyncio.Semaphore(MAX_WORKERS)
    start_t = time.time()
    errors: list[str] = []

    async def handle_doc(doc) -> None:
        nonlocal completed
        try:
            docid, fields = await process_doc(doc, few_shot_cache, sem)
            all_results[docid] = fields
        except Exception as e:
            errors.append(f"{doc.docid}: {e}")
            all_results[doc.docid] = []

        completed += 1
        if completed % 5 == 0 or completed == total:
            elapsed = time.time() - start_t
            n_done = completed - (total - len(remaining))
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (total - completed) / rate if rate > 0 else float("inf")
            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            OUT_PATH.write_text(json.dumps(all_results, indent=2))
            print(f"[{completed}/{total}] {elapsed:.0f}s, {rate:.1f} docs/s, ETA {eta:.0f}s")

    await asyncio.gather(*[handle_doc(d) for d in remaining])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(all_results, indent=2))
    print(f"\nExtraction done. {len(all_results)} docs → {OUT_PATH}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"  {e}")

    # ── Eval ────────────────────────────────────────────────────────────────
    print("\nRunning KILE + LIR evaluation...")
    from beat_docile.eval import run_eval, print_scores

    kile_preds: dict[str, list[Field]] = {}
    lir_preds: dict[str, list[Field]] = {}
    for docid, fields in all_results.items():
        kile_preds[docid] = []
        lir_preds[docid] = []
        for fd in fields:
            f = Field.from_dict(fd)
            if f.line_item_id is not None:
                lir_preds[docid].append(f)
            else:
                kile_preds[docid].append(f)

    eval_dataset = Dataset(
        split_name="val",
        dataset_path=DATA_ROOT,
        load_annotations=True,
        load_ocr=False,
        docids=list(all_results.keys()),
    )
    result = run_eval(eval_dataset, kile_preds, lir_preds)
    scores = print_scores(result)

    kile_ap = scores.get("kile_AP", 0.0) * 100
    lir_f1 = scores.get("lir_f1", 0.0) * 100

    # V5b baseline on 50 docs (from existing eval)
    v5b_kile = 41.79
    v5b_lir = 49.90

    print(f"\n{'='*55}")
    print(f"V5b baseline:        KILE AP {v5b_kile:.2f}%  /  LIR F1 {v5b_lir:.2f}%")
    print(f"Code-Factory:        KILE AP {kile_ap:.2f}%  /  LIR F1 {lir_f1:.2f}%")
    print(f"Delta:               KILE {kile_ap - v5b_kile:+.2f}pp  /  LIR {lir_f1 - v5b_lir:+.2f}pp")
    print(f"CF coverage:         {len(cf_docs)}/{len(docs)} docs ({len(cf_docs)/max(1,len(docs)):.0%})")
    print(f"Scripts available:   {len(meta)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="Ignore saved predictions and recompute all")
    args = parser.parse_args()
    asyncio.run(run_batch(fresh=args.fresh))


if __name__ == "__main__":
    main()
