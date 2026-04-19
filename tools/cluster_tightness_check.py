"""Cluster tightness check for Code-Factory feasibility.

For each val cluster that matches a train cluster:
  - Look at the train docs in that cluster (>=3 docs ideally).
  - For each fieldtype, measure: (a) presence rate (% of docs that have it),
    (b) bbox position variance (std of normalized x_center, y_center),
    (c) text-pattern regularity (do the values share a regex?).

Output: per-cluster tightness summary + aggregate verdict.

A "tight" cluster (regex-extractable by a single Python script) needs:
  - Most fieldtypes present in >=80% of docs
  - Position std < 0.05 (i.e., field appears in same screen region across docs)

If <30% of clusters are tight => Code-Factory needs build-time-ReAct + escape-hatch.
If >70% are tight => single-script-per-cluster is fine.
"""
from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict

from docile.dataset import Dataset

DATA_ROOT = "data"


def main() -> None:
    print("Loading val + train annotations...")
    val_ds = Dataset("val", DATA_ROOT, load_annotations=True, load_ocr=False)
    train_ds = Dataset("train", DATA_ROOT, load_annotations=True, load_ocr=False)

    # Map cluster_id -> list of train docs
    train_by_cluster: dict[int, list] = defaultdict(list)
    for doc in train_ds:
        train_by_cluster[doc.annotation.cluster_id].append(doc)

    val_clusters = set()
    for doc in val_ds:
        val_clusters.add(doc.annotation.cluster_id)

    matched = [c for c in val_clusters if c in train_by_cluster]
    matched_with_3plus = [c for c in matched if len(train_by_cluster[c]) >= 3]

    print(f"Val unique clusters: {len(val_clusters)}")
    print(f"Val clusters with train match: {len(matched)} ({100*len(matched)/len(val_clusters):.1f}%)")
    print(f"Val clusters with >=3 train docs (analyzable): {len(matched_with_3plus)}")
    print()

    tight_count = 0
    loose_count = 0
    medium_count = 0
    per_cluster_summary = []

    for cid in matched_with_3plus:
        train_docs = train_by_cluster[cid][:10]  # max 10 per cluster
        # Collect fieldtype -> [(x_center, y_center, text), ...]
        ft_data: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
        n_docs = len(train_docs)
        for doc in train_docs:
            for f in doc.annotation.fields:
                if f.bbox is None:
                    continue
                bb = f.bbox.to_tuple()  # (l, t, r, b) normalized [0,1]
                xc = (bb[0] + bb[2]) / 2
                yc = (bb[1] + bb[3]) / 2
                ft_data[f.fieldtype].append((xc, yc, f.text or ""))

        # Per-fieldtype: presence rate + position std
        per_ft = []
        for ft, items in ft_data.items():
            presence = len(items) / n_docs
            if len(items) >= 2:
                xs = [x for x, _, _ in items]
                ys = [y for _, y, _ in items]
                x_std = statistics.pstdev(xs)
                y_std = statistics.pstdev(ys)
                pos_std = max(x_std, y_std)
            else:
                pos_std = 0.0
            # Pattern regularity: do all texts share regex shape?
            shapes = ["".join("9" if c.isdigit() else "A" if c.isalpha() else c for c in t)[:20] for _, _, t in items]
            top_shape = max(set(shapes), key=shapes.count) if shapes else ""
            shape_match = sum(1 for s in shapes if s == top_shape) / len(shapes) if shapes else 0
            per_ft.append({
                "ft": ft,
                "presence": presence,
                "pos_std": pos_std,
                "shape_match": shape_match,
                "n": len(items),
            })

        if not per_ft:
            continue

        # Cluster tightness verdict
        avg_presence = statistics.mean(x["presence"] for x in per_ft)
        avg_pos_std = statistics.mean(x["pos_std"] for x in per_ft)
        avg_shape_match = statistics.mean(x["shape_match"] for x in per_ft)
        # Tight: high presence, low pos_std, high shape regularity
        if avg_presence >= 0.8 and avg_pos_std <= 0.05 and avg_shape_match >= 0.7:
            verdict = "TIGHT"
            tight_count += 1
        elif avg_presence >= 0.5 and avg_pos_std <= 0.10:
            verdict = "MEDIUM"
            medium_count += 1
        else:
            verdict = "LOOSE"
            loose_count += 1

        per_cluster_summary.append({
            "cluster_id": cid,
            "n_train_docs": n_docs,
            "n_fieldtypes": len(per_ft),
            "avg_presence": round(avg_presence, 3),
            "avg_pos_std": round(avg_pos_std, 4),
            "avg_shape_match": round(avg_shape_match, 3),
            "verdict": verdict,
        })

    total = tight_count + medium_count + loose_count
    print(f"=== Cluster tightness verdict (n={total}) ===")
    print(f"  TIGHT  : {tight_count} ({100*tight_count/total:.1f}%) — single regex script will work")
    print(f"  MEDIUM : {medium_count} ({100*medium_count/total:.1f}%) — needs flexible script or build-time ReAct")
    print(f"  LOOSE  : {loose_count} ({100*loose_count/total:.1f}%) — Code-Factory likely fails, fall back to V5b")
    print()

    # Show 5 examples of each
    by_verdict = defaultdict(list)
    for s in per_cluster_summary:
        by_verdict[s["verdict"]].append(s)
    for v in ["TIGHT", "MEDIUM", "LOOSE"]:
        print(f"--- 3 example {v} clusters ---")
        for s in by_verdict[v][:3]:
            print(f"  cid={s['cluster_id']:4d} n_docs={s['n_train_docs']:2d} n_fts={s['n_fieldtypes']:2d}  "
                  f"presence={s['avg_presence']:.2f} pos_std={s['avg_pos_std']:.3f} shape={s['avg_shape_match']:.2f}")
        print()

    # Coverage: how many val docs are in tight clusters?
    val_doc_counts = {"TIGHT": 0, "MEDIUM": 0, "LOOSE": 0, "UNANALYZABLE": 0, "NO_MATCH": 0}
    cid_to_verdict = {s["cluster_id"]: s["verdict"] for s in per_cluster_summary}
    for doc in val_ds:
        cid = doc.annotation.cluster_id
        if cid not in train_by_cluster:
            val_doc_counts["NO_MATCH"] += 1
        elif cid not in cid_to_verdict:
            val_doc_counts["UNANALYZABLE"] += 1
        else:
            val_doc_counts[cid_to_verdict[cid]] += 1
    total_val = sum(val_doc_counts.values())
    print(f"=== Val doc coverage by cluster verdict (n={total_val}) ===")
    for k, v in val_doc_counts.items():
        print(f"  {k:14s}: {v:4d} ({100*v/total_val:.1f}%)")

    # Save full report
    with open("cluster_tightness_report.json", "w") as fh:
        json.dump({
            "summary": {
                "tight": tight_count,
                "medium": medium_count,
                "loose": loose_count,
            },
            "val_doc_coverage": val_doc_counts,
            "per_cluster": per_cluster_summary,
        }, fh, indent=2)
    print("\nFull report -> cluster_tightness_report.json")


if __name__ == "__main__":
    main()
