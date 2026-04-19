#!/usr/bin/env python3
"""Build SAIL retrieval index for DocILE.

Combines existing Qwen3-VL-Embedding-2B visual embeddings (from
models/qwen3vl_train_embeddings.npz) with entity-level features (field-type
presence vectors from gold annotations) and pre-computed words_layout /
gold_json strings for fast FewShotExample construction at inference time.

Usage
-----
    # Validate on 100 docs first
    python tools/build_sail_index.py --sample 100

    # Build full index
    python tools/build_sail_index.py

Output
------
    models/sail_index/sail_index.npz
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import os
os.environ.setdefault("DATA_ROOT", str(_ROOT / "data"))

from beat_docile.config import DATA_ROOT
from beat_docile.extract import _words_to_prompt
from beat_docile.fewshot import _gold_to_compact_json
from beat_docile.sail_retrieval import ALL_FIELD_TYPES, entity_vec_from_gold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_sail_index(
    visual_embs_path: Path,
    output_path: Path,
    sample: int | None = None,
) -> dict:
    """Build and save SAIL index NPZ.

    Loads visual embeddings from an existing Qwen3-VL NPZ, then for each
    train doc computes: entity presence vector (from gold), words_layout
    (from OCR page 0), and gold_json (from annotations).

    Returns stats dict.
    """
    from docile.dataset import Dataset
    from beat_docile.data import WordBox

    logger.info("Loading visual embeddings from %s", visual_embs_path)
    vis_data = np.load(visual_embs_path, allow_pickle=True)
    vis_embs: np.ndarray = vis_data["embeddings"].astype(np.float32)
    vis_docids: list[str] = [str(d) for d in vis_data["docids"]]
    vis_cluster_ids: np.ndarray = vis_data["cluster_ids"].astype(np.int32)
    logger.info(
        "Visual embeddings: %d docs, embed_dim=%d",
        len(vis_docids), vis_embs.shape[1],
    )

    vis_emb_by_docid = {d: vis_embs[i] for i, d in enumerate(vis_docids)}
    vis_cid_by_docid = {d: int(vis_cluster_ids[i]) for i, d in enumerate(vis_docids)}

    # Build set of docids with cached OCR JSON — only process these to avoid
    # triggering DocTR re-inference from PDFs that are not locally available.
    ocr_dir = Path(DATA_ROOT) / "ocr"
    ocr_cached_docids = {f.stem for f in ocr_dir.glob("*.json")}
    logger.info("Cached OCR available for %d train docs", len(ocr_cached_docids & set(vis_docids)))

    # Use load_ocr=False to allow lazy OCR access only for cached docs
    logger.info("Loading train dataset (annotations, lazy OCR) from %s", DATA_ROOT)
    train_ds = Dataset("train", DATA_ROOT, load_annotations=True, load_ocr=False)
    train_doc_map = {doc.docid: doc for doc in train_ds}
    logger.info("Train dataset: %d docs", len(train_doc_map))

    # Only process docs that have both visual embeddings AND cached OCR
    eligible = [d for d in vis_docids if d in ocr_cached_docids]
    docs_to_process = eligible[:sample] if sample else eligible
    logger.info(
        "Processing %d docs with cached OCR (sample=%s, total eligible=%d)",
        len(docs_to_process), sample, len(eligible),
    )

    out_visual_embs: list[np.ndarray] = []
    out_entity_vecs: list[np.ndarray] = []
    out_docids: list[str] = []
    out_cluster_ids: list[int] = []
    out_gold_jsons: list[str] = []
    out_words_layouts: list[str] = []
    n_skipped = 0
    t0 = time.monotonic()

    for i, docid in enumerate(docs_to_process):
        doc = train_doc_map.get(docid)
        if doc is None:
            logger.debug("No train doc found for docid=%s", docid)
            n_skipped += 1
            continue

        try:
            with doc:
                # Gold-based entity vector
                fields = doc.annotation.fields
                li_fields = doc.annotation.li_fields
                entity_vec = entity_vec_from_gold(fields, li_fields)
                gold_json = _gold_to_compact_json(fields, li_fields)

                # words_layout from page 0 raw OCR (no snapping needed for index)
                try:
                    raw_words = doc.ocr.get_all_words(page=0, snapped=False)
                    word_boxes = [
                        WordBox(id=j, text=f.text, bbox=f.bbox.to_tuple(), page=0)
                        for j, f in enumerate(raw_words)
                    ]
                    words_layout = _words_to_prompt(word_boxes)
                except Exception:
                    words_layout = ""

        except Exception as e:
            logger.warning("Skipping docid=%s: %s", docid, e)
            n_skipped += 1
            continue

        out_visual_embs.append(vis_emb_by_docid[docid])
        out_entity_vecs.append(entity_vec)
        out_docids.append(docid)
        out_cluster_ids.append(vis_cid_by_docid[docid])
        out_gold_jsons.append(gold_json)
        out_words_layouts.append(words_layout)

        if (i + 1) % 200 == 0:
            elapsed = time.monotonic() - t0
            rate = (i + 1) / elapsed
            eta_min = (len(docs_to_process) - i - 1) / max(rate, 1e-6) / 60
            logger.info(
                "  %d/%d (%.1f/s, ETA %.0f min)",
                i + 1, len(docs_to_process), rate, eta_min,
            )

    if not out_visual_embs:
        raise RuntimeError("No docs processed — check visual_embs_path and DATA_ROOT")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        visual_embs=np.stack(out_visual_embs).astype(np.float32),
        entity_vecs=np.stack(out_entity_vecs).astype(np.float32),
        docids=np.array(out_docids, dtype=object),
        cluster_ids=np.array(out_cluster_ids, dtype=np.int32),
        gold_jsons=np.array(out_gold_jsons, dtype=object),
        words_layouts=np.array(out_words_layouts, dtype=object),
    )

    elapsed = time.monotonic() - t0
    stats = {
        "n_processed": len(out_docids),
        "n_skipped": n_skipped,
        "entity_dim": len(ALL_FIELD_TYPES),
        "visual_dim": vis_embs.shape[1],
        "output_path": str(output_path),
        "elapsed_sec": round(elapsed, 1),
    }
    logger.info("SAIL index saved: %s", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--visual-embs",
        type=Path,
        default=Path("models/qwen3vl_train_embeddings.npz"),
        help="Existing Qwen3-VL-Embedding-2B NPZ (default: models/qwen3vl_train_embeddings.npz)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/sail_index/sail_index.npz"),
        help="Output path for SAIL index NPZ (default: models/sail_index/sail_index.npz)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N docs (useful for validation)",
    )
    args = parser.parse_args()

    # Resolve relative paths from project root
    visual_embs_path = (
        args.visual_embs if args.visual_embs.is_absolute()
        else _ROOT / args.visual_embs
    )
    output_path = (
        args.output if args.output.is_absolute()
        else _ROOT / args.output
    )

    stats = build_sail_index(
        visual_embs_path=visual_embs_path,
        output_path=output_path,
        sample=args.sample,
    )

    print("\nSAIL index build complete:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
