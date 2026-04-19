#!/usr/bin/env python3
"""Evaluate SAIL retrieval on 30 NO_MATCH val docs.

Eval protocol (Phase 5):
1. Identify NO_MATCH val docs (cluster_id absent from train)
2. Sample 30 of them
3. For each: retrieve top-3 SAIL examples, compare vs zero-shot (cluster fallback = empty)
4. Run V5b extraction with SAIL few-shot on the 30 docs
5. Compare KILE AP: SAIL few-shot vs zero-shot on NO_MATCH subset

Usage:
    python tools/eval_sail_no_match.py [--n-docs 30] [--model MODEL]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))
os.environ.setdefault("DATA_ROOT", str(_ROOT / "data"))
os.environ["BD_USE_REFINER"] = "1"
os.environ["BD_USE_VALIDATOR"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_no_match_val_docs(n: int = 30):
    """Return up to n val docs whose cluster_id is absent from any train doc."""
    from docile.dataset import Dataset
    from beat_docile.config import DATA_ROOT

    logger.info("Loading train dataset for cluster_id set")
    train_ds = Dataset("train", DATA_ROOT, load_annotations=True, load_ocr=False)
    train_cluster_ids = {doc.annotation.cluster_id for doc in train_ds}
    logger.info("Train cluster IDs: %d unique", len(train_cluster_ids))

    logger.info("Loading val dataset")
    val_ds = Dataset("val", DATA_ROOT, load_annotations=True, load_ocr=True)
    no_match = [
        doc for doc in val_ds
        if doc.annotation.cluster_id not in train_cluster_ids
    ]
    logger.info("NO_MATCH val docs: %d/500", len(no_match))
    return no_match[:n], val_ds


async def run_extraction_no_fewshot(docs, model: str) -> tuple[dict, dict]:
    """Zero-shot extraction (no few-shot) for the given docs."""
    from beat_docile.extract import extract_documents
    return await extract_documents(docs, model, train_index=None, targeted_pass=True)


async def run_extraction_sail_fewshot(docs, model: str) -> tuple[dict, dict]:
    """SAIL-retrieval few-shot extraction for the given docs."""
    from beat_docile.extract import extract_page, extract_page_targeted, _TARGETED_FIELDS
    from beat_docile.data import iter_pages
    from beat_docile.fewshot import build_few_shot_messages
    from beat_docile.sail_retrieval import get_retriever
    from docile.dataset import Field
    import asyncio

    retriever = get_retriever()
    kile_preds: dict[str, list[Field]] = {}
    lir_preds: dict[str, list[Field]] = {}

    async def process_doc(doc) -> None:
        kile_preds[doc.docid] = []
        lir_preds[doc.docid] = []

        # Get SAIL few-shot examples (opens doc internally)
        examples = retriever.select_few_shot(doc, k=3)
        fs_messages = build_few_shot_messages(examples) if examples else None
        logger.info(
            "  Doc %s: %d SAIL examples selected from clusters %s",
            doc.docid[:8], len(examples),
            [ex.cluster_id for ex in examples],
        )

        # Extract pages
        pages = list(iter_pages(doc))
        main_tasks = [extract_page(page, model, few_shot_messages=fs_messages) for page in pages]
        targeted_tasks = [extract_page_targeted(page, model) for page in pages]
        all_results = await asyncio.gather(*main_tasks, *targeted_tasks)
        n = len(pages)
        for kile, lir in all_results[:n]:
            kile_preds[doc.docid].extend(kile)
            lir_preds[doc.docid].extend(lir)
        for fields in all_results[n:]:
            for f in fields:
                if f.line_item_id is not None:
                    lir_preds[doc.docid].append(f)
                else:
                    kile_preds[doc.docid].append(f)

    await asyncio.gather(*[process_doc(doc) for doc in docs])
    return kile_preds, lir_preds


def evaluate_kile(kile_preds: dict[str, list], docids: list[str]) -> float:
    """Compute KILE AP on the given doc subset using docile evaluation."""
    from docile.dataset import Dataset
    from docile.evaluation import evaluate_dataset
    from beat_docile.config import DATA_ROOT
    from beat_docile.eval import run_eval, print_scores

    subset_ds = Dataset(
        "val", DATA_ROOT, load_annotations=True, load_ocr=False, docids=docids,
    )
    # run_eval fills missing docids with empty lists
    result = run_eval(subset_ds, dict(kile_preds), {})
    scores = print_scores(result)
    return scores.get("kile_AP", 0.0)


def show_sample_retrievals(docs, retriever, n_samples: int = 3) -> list[dict]:
    """Return dicts describing SAIL retrievals for N val docs."""
    from beat_docile.sail_retrieval import load_sail_index, _DEFAULT_INDEX_PATH
    import numpy as np

    index = retriever._ensure_index()

    samples = []
    for doc in docs[:n_samples]:
        from beat_docile.sail_retrieval import entity_vec_from_ocr, _l2_normalize_rows
        from beat_docile.cluster_infer import embed_doc_qwen3vl

        model = retriever._ensure_model()
        with doc:
            visual_emb = embed_doc_qwen3vl(doc, model, retriever._device)
            try:
                page0_words = doc.ocr.get_all_words(page=0, snapped=False)
                ocr_text = " ".join(w.text for w in page0_words)
            except Exception:
                ocr_text = ""

        entity_vec = entity_vec_from_ocr(ocr_text)
        ev_norm = np.linalg.norm(entity_vec)
        entity_vec_n = entity_vec / (ev_norm + 1e-8) if ev_norm > 1e-8 else entity_vec

        visual_sims = index.visual_embs @ visual_emb
        entity_sims = index.entity_vecs @ entity_vec_n
        combined = retriever._alpha * visual_sims + (1.0 - retriever._alpha) * entity_sims
        top3_idx = np.argsort(combined)[::-1][:3]

        picks = []
        for idx in top3_idx:
            picks.append({
                "train_docid": index.docids[idx],
                "cluster_id": int(index.cluster_ids[idx]),
                "visual_sim": float(visual_sims[idx]),
                "entity_sim": float(entity_sims[idx]),
                "combined_sim": float(combined[idx]),
            })

        samples.append({
            "val_docid": doc.docid,
            "val_cluster_id": doc.annotation.cluster_id,
            "top3_sail_picks": picks,
        })

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-docs", type=int, default=30, help="Number of NO_MATCH val docs to eval")
    parser.add_argument("--model", default=None, help="Claude model (default: config DEFAULT_MODEL)")
    parser.add_argument("--out", type=Path, default=None, help="JSON output path")
    args = parser.parse_args()

    from beat_docile.config import DEFAULT_MODEL
    model = args.model or DEFAULT_MODEL
    out_path = args.out or (_ROOT / "predictions" / "sail_no_match_eval.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=== SAIL NO_MATCH Evaluation ===")
    logger.info("Model: %s", model)
    logger.info("N docs: %d", args.n_docs)

    # Step 1: Find NO_MATCH val docs
    no_match_docs, val_ds = find_no_match_val_docs(args.n_docs)
    logger.info("Using %d NO_MATCH val docs", len(no_match_docs))

    # Step 2: Show sample retrievals (before running inference)
    from beat_docile.sail_retrieval import get_retriever
    retriever = get_retriever()
    logger.info("Computing sample SAIL retrievals...")
    sample_retrievals = show_sample_retrievals(no_match_docs, retriever, n_samples=3)
    for s in sample_retrievals:
        logger.info("\nVal doc %s (cluster %s):", s["val_docid"][:8], s["val_cluster_id"])
        for pick in s["top3_sail_picks"]:
            logger.info(
                "  → train %s (cluster %s) | vis=%.3f ent=%.3f comb=%.3f",
                pick["train_docid"][:8], pick["cluster_id"],
                pick["visual_sim"], pick["entity_sim"], pick["combined_sim"],
            )

    # Step 3: Run zero-shot extraction
    logger.info("\n--- Zero-shot extraction (no few-shot) ---")
    t0 = time.monotonic()
    zs_kile, zs_lir = asyncio.run(run_extraction_no_fewshot(no_match_docs, model))
    zs_elapsed = time.monotonic() - t0
    logger.info("Zero-shot done in %.1fs", zs_elapsed)

    # Step 4: Run SAIL few-shot extraction
    logger.info("\n--- SAIL few-shot extraction ---")
    t1 = time.monotonic()
    sail_kile, sail_lir = asyncio.run(run_extraction_sail_fewshot(no_match_docs, model))
    sail_elapsed = time.monotonic() - t1
    logger.info("SAIL done in %.1fs", sail_elapsed)

    # Step 5: Evaluate both
    logger.info("\n--- KILE AP Evaluation ---")
    zs_ap = evaluate_kile(zs_kile, no_match_docs)
    sail_ap = evaluate_kile(sail_kile, no_match_docs)
    delta = sail_ap - zs_ap
    logger.info("Zero-shot KILE AP: %.2f%%", zs_ap * 100)
    logger.info("SAIL few-shot KILE AP: %.2f%%", sail_ap * 100)
    logger.info("Delta: %+.2f%%", delta * 100)

    # Save results
    results = {
        "n_docs": len(no_match_docs),
        "model": model,
        "zero_shot_kile_ap": zs_ap,
        "sail_kile_ap": sail_ap,
        "delta_pp": delta * 100,
        "sample_retrievals": sample_retrievals,
        "val_docids": [doc.docid for doc in no_match_docs],
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
