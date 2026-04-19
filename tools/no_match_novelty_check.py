"""Are NO_MATCH val docs (different cluster_id from any train doc) actually novel,
or are they visually similar to some train doc that just got a different cluster label?

Quick proxy: text-vocabulary Jaccard between val NO_MATCH docs and ALL train docs
that have OCR JSON on disk. We bypass docile.Dataset's PDF requirement by reading
OCR JSON files directly.

For invoices, text vocab strongly correlates with visual layout (boilerplate words
like vendor name, address, table headers all repeat).
"""
from __future__ import annotations

import json
import os
import re
import statistics

from docile.dataset import Dataset

DATA_ROOT = "data"
OCR_DIR = "data/ocr"


def ocr_vocab_from_json(docid: str) -> set[str]:
    """Extract normalized token set from OCR JSON page 0 only."""
    path = os.path.join(OCR_DIR, f"{docid}.json")
    if not os.path.exists(path):
        return set()
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return set()
    if not d.get("pages"):
        return set()
    page = d["pages"][0]
    tokens: set[str] = set()
    for block in page.get("blocks", []):
        for line in block.get("lines", []):
            for w in line.get("words", []):
                t = w.get("value", "")
                norm = re.sub(r"[^a-zA-Z0-9]", "", t.lower())
                if len(norm) >= 2:
                    tokens.add(norm)
    return tokens


def main() -> None:
    print("Loading splits (annotations only, no OCR/PDF load)...")
    val_ds = Dataset("val", DATA_ROOT, load_annotations=True, load_ocr=False)
    train_ds = Dataset("train", DATA_ROOT, load_annotations=True, load_ocr=False)

    train_clusters = {doc.annotation.cluster_id for doc in train_ds}
    no_match_val = [doc for doc in val_ds if doc.annotation.cluster_id not in train_clusters]
    print(f"NO_MATCH val docs: {len(no_match_val)}")

    sample = no_match_val[:30]

    print("Building train vocabs from OCR JSON files (skip docs without OCR on disk)...")
    train_vocabs: list[set[str]] = []
    train_docids: list[str] = []
    skipped = 0
    for i, td in enumerate(train_ds):
        if i % 1000 == 0 and i:
            print(f"  {i}/5180 ({skipped} skipped)")
        v = ocr_vocab_from_json(td.docid)
        if v:
            train_vocabs.append(v)
            train_docids.append(td.docid)
        else:
            skipped += 1
    print(f"Built {len(train_vocabs)} train vocabs, {skipped} skipped (no OCR JSON)")

    print(f"\nMatching {len(sample)} NO_MATCH val docs to nearest train doc by Jaccard...")
    results = []
    for vd in sample:
        vv = ocr_vocab_from_json(vd.docid)
        if not vv:
            continue
        sims = []
        for tv, tdid in zip(train_vocabs, train_docids):
            inter = len(vv & tv)
            union = len(vv | tv)
            jacc = inter / union if union else 0.0
            sims.append((jacc, tdid))
        sims.sort(reverse=True)
        top3 = sims[:3]
        results.append({
            "val_docid": vd.docid,
            "val_cluster_id": vd.annotation.cluster_id,
            "top1_jaccard": top3[0][0],
            "top1_train_docid": top3[0][1],
            "top3_jaccards": [round(j, 3) for j, _ in top3],
        })

    print(f"\n=== NO_MATCH val novelty (n={len(results)}) ===")
    top1s = [r["top1_jaccard"] for r in results]
    print(f"Top-1 Jaccard distribution:")
    print(f"  median: {statistics.median(top1s):.3f}")
    print(f"  mean  : {statistics.mean(top1s):.3f}")
    print(f"  min   : {min(top1s):.3f}")
    print(f"  max   : {max(top1s):.3f}")

    very_similar = sum(1 for j in top1s if j >= 0.7)
    similar = sum(1 for j in top1s if 0.5 <= j < 0.7)
    different = sum(1 for j in top1s if 0.3 <= j < 0.5)
    novel = sum(1 for j in top1s if j < 0.3)
    print(f"\nBuckets (text-Jaccard proxy):")
    print(f"  VERY SIMILAR (j>=0.7): {very_similar} ({100*very_similar/len(top1s):.1f}%)")
    print(f"  SIMILAR      (0.5-0.7): {similar} ({100*similar/len(top1s):.1f}%)")
    print(f"  DIFFERENT    (0.3-0.5): {different} ({100*different/len(top1s):.1f}%)")
    print(f"  NOVEL        (<0.3):   {novel} ({100*novel/len(top1s):.1f}%)")

    print(f"\nTop 5 most-similar matches:")
    for r in sorted(results, key=lambda x: -x["top1_jaccard"])[:5]:
        print(f"  val={r['val_docid'][:12]} cid={r['val_cluster_id']:5d}  "
              f"top1={r['top1_jaccard']:.3f}  train={r['top1_train_docid'][:12]}  top3={r['top3_jaccards']}")
    print(f"\nBottom 5 (most novel):")
    for r in sorted(results, key=lambda x: x["top1_jaccard"])[:5]:
        print(f"  val={r['val_docid'][:12]} cid={r['val_cluster_id']:5d}  "
              f"top1={r['top1_jaccard']:.3f}  train={r['top1_train_docid'][:12]}")

    with open("no_match_novelty_report.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nFull report -> no_match_novelty_report.json")


if __name__ == "__main__":
    main()
