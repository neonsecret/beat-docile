#!/usr/bin/env python
"""Rebuild snapped_geometry cache at 200 DPI with reduced min_char_width_margin.

Strips existing snapped_geometry from OCR JSON files, monkey-patches
_foreground_text_bbox with reduced min_char_width_margin (default 6 → arg),
then re-runs snapping at 200 DPI for target docs.

After this runs, V5b eval can be executed unchanged — both data.py (words) and
evaluate.py (PCCSet) will read the rebuilt 200-DPI cache, so results are
fully self-consistent.

Usage:
    # Rebuild 50 val docs with margin=3
    DATA_ROOT=data uv run python tools/rebuild_snap_cache_200dpi.py --min-char-width 3

    # Rebuild 50 val docs with margin=4
    DATA_ROOT=data uv run python tools/rebuild_snap_cache_200dpi.py --min-char-width 4

    # Rebuild all 500 val docs
    DATA_ROOT=data uv run python tools/rebuild_snap_cache_200dpi.py --min-char-width 3 --all-val
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
os.environ.setdefault("DATA_ROOT", str(DATA_DIR))

sys.path.insert(0, str(PROJECT_ROOT / "src"))

import docile.dataset.document_ocr as _docr
from beat_docile.config import DATA_ROOT
from beat_docile.data import load_split


def _build_patched_snap(min_char_width_margin: int):
    """Return a replacement for _foreground_text_bbox with reduced min_char_width_margin."""
    _orig = _docr._foreground_text_bbox

    def _patched(
        foreground_mask,
        margin_size=5,
        min_char_width_margin=min_char_width_margin,
        min_line_height_margin=10,
        min_char_width_inside=2,
        min_line_height_inside=5,
    ):
        return _orig(
            foreground_mask,
            margin_size=margin_size,
            min_char_width_margin=min_char_width_margin,
            min_line_height_margin=min_line_height_margin,
            min_char_width_inside=min_char_width_inside,
            min_line_height_inside=min_line_height_inside,
        )

    return _patched


def strip_snapped_geometry(ocr_path: Path) -> bool:
    """Remove all snapped_geometry keys from an OCR JSON. Returns True if any were removed."""
    data = json.loads(ocr_path.read_bytes())
    removed = 0
    for page in data.get("pages", []):
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                for word in line.get("words", []):
                    if "snapped_geometry" in word:
                        del word["snapped_geometry"]
                        removed += 1
    if removed:
        ocr_path.write_text(json.dumps(data))
    return removed > 0


def rebuild_snap_for_doc(doc, dpi: int = 200) -> int:
    """Rebuild snapped_geometry for all pages of doc at given DPI. Returns word count."""
    total_words = 0
    with doc:
        for page_idx in range(doc.page_count):
            w, h = doc.page_image_size(page_idx, dpi=dpi)
            page_image = doc.page_image(page_idx, image_size=(w, h))
            words = doc.ocr.get_all_words(
                page=page_idx,
                snapped=True,
                use_cached_snapping=True,
                get_page_image=lambda _img=page_image: _img,
            )
            total_words += len(words)
    return total_words


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-char-width", type=int, default=3,
                        help="min_char_width_margin (default 3, original 6)")
    parser.add_argument("--dpi", type=int, default=200,
                        help="DPI for snap image (default 200, matches evaluator)")
    parser.add_argument("--all-val", action="store_true",
                        help="Rebuild all 500 val docs instead of just the 50 target docs")
    args = parser.parse_args()

    v5b_50_path = PROJECT_ROOT / "predictions" / "v5b_50.json"
    if not args.all_val:
        target_docids = set(json.loads(v5b_50_path.read_text()).keys())
        print(f"Target: {len(target_docids)} docs (from v5b_50.json)")
    else:
        target_docids = None
        print("Target: all 500 val docs")

    print(f"Snap DPI: {args.dpi}")
    print(f"min_char_width_margin: 6 → {args.min_char_width}")

    # Monkey-patch before loading any documents
    _orig = _docr._foreground_text_bbox
    _docr._foreground_text_bbox = _build_patched_snap(args.min_char_width)
    print(f"Monkey-patched _foreground_text_bbox with min_char_width_margin={args.min_char_width}")

    print("Loading val split...")
    dataset = load_split("val")
    docs = [d for d in dataset if (target_docids is None or d.docid in target_docids)]
    print(f"Processing {len(docs)} docs\n")

    ocr_dir = DATA_DIR / "ocr"
    stripped = 0
    rebuilt = 0

    for i, doc in enumerate(docs):
        ocr_path = ocr_dir / f"{doc.docid}.json"
        if not ocr_path.exists():
            print(f"  [{i+1}/{len(docs)}] {doc.docid}: OCR file not found, skip")
            continue

        # Step 1: strip existing snapped_geometry from disk
        had_snaps = strip_snapped_geometry(ocr_path)
        if had_snaps:
            stripped += 1

        # Step 2: rebuild snapped_geometry via docile's caching mechanism
        n_words = rebuild_snap_for_doc(doc, dpi=args.dpi)
        rebuilt += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(docs):
            print(f"  [{i+1}/{len(docs)}] done — {rebuilt} rebuilt, {stripped} had existing snaps")

    # Restore original function
    _docr._foreground_text_bbox = _orig

    print(f"\nDone. Rebuilt {rebuilt} docs, {stripped} had existing snapped_geometry stripped.")
    print(f"Cache now contains {args.dpi}-DPI snaps with min_char_width_margin={args.min_char_width}.")
    print("Run V5b eval to measure impact:")
    print(f"  DATA_ROOT=data uv run python tools/v5b_resnapped_50.py --snap-label margin{args.min_char_width}")


if __name__ == "__main__":
    main()
