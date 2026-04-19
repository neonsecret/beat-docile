#!/usr/bin/env python
"""Phase 6 eval: YOLOv12s-DocLayNet region scoping on same 50 val docs.

Three sub-modes (--mode):
  spike  — visualize regions on 5 docs, save annotated PNGs to tools/_out/doclayout_samples/
  mode_a — region-tagged prompt (REGION:X before each row); eval vs V5b
  mode_b — noise-region word filter (remove Caption/Formula/Picture/List-item); eval vs V5b

V5b baseline: KILE AP 41.79% / LIR F1 49.90% (from predictions/v5b_50.json).

Usage:
  DATA_ROOT=data uv run python tools/run_doclayout_50.py --mode spike
  DATA_ROOT=data uv run python tools/run_doclayout_50.py --mode mode_a
  DATA_ROOT=data uv run python tools/run_doclayout_50.py --mode mode_b
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
os.environ["BD_USE_REFINER"] = "1"
os.environ["BD_USE_VALIDATOR"] = "1"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from beat_docile.config import DEFAULT_MODEL, DATA_ROOT
from beat_docile.data import load_split, iter_pages
from beat_docile.extract import extract_page_targeted
from beat_docile.fewshot import _build_cluster_index, load_few_shot_examples, build_few_shot_messages
from beat_docile.layout_regions import (
    detect_regions,
    assign_word_regions,
    annotate_page_with_regions,
    extract_page_mode_a,
    extract_page_mode_b,
)
from docile.dataset import Field

MAX_WORKERS = 8
MODEL = DEFAULT_MODEL

V5B_50_PATH = PROJECT_ROOT / "predictions" / "v5b_50.json"
OUT_DIR = PROJECT_ROOT / "predictions"
SPIKE_OUT_DIR = PROJECT_ROOT / "tools" / "_out" / "doclayout_samples"

V5B_KILE = 41.79
V5B_LIR = 49.90


def _fields_to_dicts(fields: list[Field]) -> list[dict]:
    return [f.to_dict() for f in fields]


# ── Spike: visualize regions on 5 docs ───────────────────────────────────────

def run_spike(docs: list, n_docs: int = 5) -> None:
    from PIL import ImageDraw, ImageFont
    import colorsys

    SPIKE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Assign a distinct color per region label
    region_colors: dict[str, tuple[int, int, int]] = {}
    _palette = [
        (220, 50, 50),   # Caption — red
        (50, 180, 50),   # Footnote — green
        (180, 50, 220),  # Formula — purple
        (50, 50, 220),   # List-item — blue
        (220, 140, 50),  # Page-footer — orange
        (50, 180, 220),  # Page-header — cyan
        (220, 220, 50),  # Picture — yellow
        (140, 80, 50),   # Section-header — brown
        (50, 220, 140),  # Table — teal
        (180, 180, 180), # Text — gray
        (220, 80, 180),  # Title — pink
    ]
    known_labels = [
        "Caption", "Footnote", "Formula", "List-item",
        "Page-footer", "Page-header", "Picture", "Section-header",
        "Table", "Text", "Title",
    ]
    for i, lbl in enumerate(known_labels):
        region_colors[lbl] = _palette[i % len(_palette)]

    print(f"\n=== SPIKE: visualizing {n_docs} docs ===")
    for doc_idx, doc in enumerate(docs[:n_docs]):
        print(f"  [{doc_idx+1}/{n_docs}] {doc.docid}")
        pages = list(iter_pages(doc))
        for page in pages:
            img = page.image.copy().convert("RGB")
            w, h = img.size
            regions = detect_regions(page.image)
            word_regions = assign_word_regions(page.words, regions, w, h)

            draw = ImageDraw.Draw(img, "RGBA")

            # Draw region boxes
            for r in regions:
                x1, y1, x2, y2 = r["bbox"]
                label = r["label"]
                color = region_colors.get(label, (200, 200, 200))
                # Semi-transparent fill
                draw.rectangle([x1, y1, x2, y2], fill=(*color, 40), outline=(*color, 200), width=2)
                draw.text((x1 + 3, y1 + 1), f"{label} {r['conf']:.2f}", fill=(*color, 255))

            # Draw word centers colored by assigned region
            for w_box in page.words:
                cx = int((w_box.bbox[0] + w_box.bbox[2]) / 2 * w)
                cy = int((w_box.bbox[1] + w_box.bbox[3]) / 2 * h)
                region = word_regions.get(w_box.id, "Text")
                color = region_colors.get(region, (200, 200, 200))
                draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(*color, 180))

            out_path = SPIKE_OUT_DIR / f"{doc.docid}_p{page.page_index}.png"
            img.save(out_path)
            print(f"    Saved: {out_path.name}  ({len(regions)} regions, {len(page.words)} words)")

    print(f"\nSpike done. PNGs saved to {SPIKE_OUT_DIR}")

    # Print region distribution summary
    all_labels: list[str] = []
    for doc in docs[:n_docs]:
        for page in iter_pages(doc):
            regions = detect_regions(page.image)
            all_labels.extend(r["label"] for r in regions)

    from collections import Counter
    counts = Counter(all_labels)
    print("\nRegion label distribution across spike docs:")
    for label, cnt in counts.most_common():
        print(f"  {label:20s}: {cnt}")


# ── Mode A/B extraction ───────────────────────────────────────────────────────

async def run_extraction(
    docs: list,
    mode: str,
    out_path: Path,
    few_shot_cache: dict[int, list[dict]],
) -> dict[str, list[dict]]:
    all_results: dict[str, list[dict]] = {}
    if out_path.exists():
        all_results = json.loads(out_path.read_text())
        print(f"Resuming — already done: {len(all_results)} docs")

    remaining = [d for d in docs if d.docid not in all_results]
    total = len(docs)
    completed = len(all_results)
    errors: list[str] = []
    start_t = time.time()
    sem = asyncio.Semaphore(MAX_WORKERS)

    async def process_doc(doc) -> None:
        nonlocal completed
        async with sem:
            try:
                fs_messages = few_shot_cache.get(doc.annotation.cluster_id)
                pages = list(iter_pages(doc))

                # Detect regions for all pages
                page_word_regions = []
                for page in pages:
                    word_regions = annotate_page_with_regions(page)
                    page_word_regions.append(word_regions)

                # Main extraction (mode A or B)
                if mode == "mode_a":
                    main_tasks = [
                        extract_page_mode_a(p, MODEL, wr, few_shot_messages=fs_messages)
                        for p, wr in zip(pages, page_word_regions)
                    ]
                else:  # mode_b
                    main_tasks = [
                        extract_page_mode_b(p, MODEL, wr, few_shot_messages=fs_messages)
                        for p, wr in zip(pages, page_word_regions)
                    ]

                # Targeted pass (standard — not region-filtered)
                targeted_tasks = [extract_page_targeted(p, MODEL) for p in pages]

                all_page_results = await asyncio.gather(*main_tasks, *targeted_tasks)
                n = len(pages)
                main_results = all_page_results[:n]
                targeted_results = all_page_results[n:]

                kile: list[Field] = []
                lir: list[Field] = []
                for k, l in main_results:
                    kile.extend(k)
                    lir.extend(l)
                for fields in targeted_results:
                    for f in fields:
                        if f.line_item_id is not None:
                            lir.append(f)
                        else:
                            kile.append(f)

                all_results[doc.docid] = _fields_to_dicts(kile + lir)

            except Exception as e:
                import traceback
                errors.append(f"{doc.docid}: {e}\n{traceback.format_exc()}")
                all_results[doc.docid] = []

            completed += 1
            if completed % 5 == 0 or completed == total:
                elapsed = time.time() - start_t
                n_done = completed - (total - len(remaining))
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (total - completed) / rate if rate > 0 else float("inf")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(all_results, indent=2))
                print(f"[{completed}/{total}] done — {elapsed:.0f}s elapsed, "
                      f"{rate:.1f} docs/s, ETA {eta:.0f}s")

    await asyncio.gather(*[process_doc(doc) for doc in remaining])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors[:5]:
            print(f"  {e}")
    return all_results


def _eval_and_print(all_results: dict, label: str, mode: str = "mode_a") -> dict[str, float]:
    from beat_docile.eval import run_eval, print_scores
    from docile.dataset import Dataset

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
        split_name=f"doclayout_{mode.replace('_', '')}_50",
        dataset_path=DATA_ROOT,
        load_annotations=True,
        load_ocr=False,
        docids=list(all_results.keys()),
    )
    result = run_eval(eval_dataset, kile_preds, lir_preds)
    print(f"\n--- {label} ---")
    scores = print_scores(result)
    kile_ap = scores.get("kile_AP", 0) * 100
    lir_f1 = scores.get("lir_f1", 0) * 100
    print(f"\n{'='*60}")
    print(f"V5b baseline:  KILE AP {V5B_KILE:.2f}% / LIR F1 {V5B_LIR:.2f}%")
    print(f"{label:14s}:  KILE AP {kile_ap:.2f}% / LIR F1 {lir_f1:.2f}%")
    print(f"Delta:         KILE {kile_ap - V5B_KILE:+.2f}pp / LIR {lir_f1 - V5B_LIR:+.2f}pp")
    return scores


async def main(mode: str) -> None:
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"Model: {MODEL}")
    print(f"Mode: {mode}")
    print(f"YOLOv12s-DocLayNet: models/yolov12s-doclaynet.pt (fallback from PP-DocLayout-Plus-S; "
          f"paddlepaddle has no Python 3.14 wheel)")

    target_docids = set(json.loads(V5B_50_PATH.read_text()).keys())
    print(f"\nTarget docids: {len(target_docids)} (from v5b_50.json)")

    print("Loading val split...")
    dataset = load_split("val")
    docs = [d for d in dataset if d.docid in target_docids]
    print(f"Matched {len(docs)} docs")

    if mode == "spike":
        run_spike(docs, n_docs=5)
        return

    print("Building train cluster index for few-shot...")
    train_index = _build_cluster_index("train")
    unique_cluster_ids = list({doc.annotation.cluster_id for doc in docs
                                if doc.annotation.cluster_id is not None})
    examples_by_cluster = load_few_shot_examples(unique_cluster_ids, train_index, max_per_cluster=1)
    few_shot_cache: dict[int, list[dict]] = {
        cid: build_few_shot_messages(examples)
        for cid, examples in examples_by_cluster.items()
    }
    print(f"Few-shot cache: {len(few_shot_cache)} clusters")

    out_path = OUT_DIR / f"v5b_50_{mode}.json"
    print(f"Output: {out_path}\n")

    all_results = await run_extraction(docs, mode, out_path, few_shot_cache)
    print(f"\nExtraction complete. {len(all_results)} docs saved.")

    label = "Mode A (regions)" if mode == "mode_a" else "Mode B (filtered)"
    _eval_and_print(all_results, label, mode=mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["spike", "mode_a", "mode_b"], required=True)
    args = parser.parse_args()
    asyncio.run(main(args.mode))
