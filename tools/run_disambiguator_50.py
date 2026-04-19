#!/usr/bin/env python
"""Phase 8 eval: GLiNER disambiguator on V5b 50-doc val predictions.

Steps:
  1. Load v5b_50.json (50-doc KILE+LIR predictions)
  2. For each doc, load OCR words via iter_pages (no re-extraction)
  3. Run GLiNER conflict resolver
  4. Eval KILE AP before/after on same 50 docs
  5. Report per-pair contribution table + conflict statistics
  6. Save resolved predictions → predictions/v5b_50_disambig.json
  7. Write .planning/phases/PHASE-8-gliner-disambiguator/RESULT.md

Usage:
    DATA_ROOT=data uv run python tools/run_disambiguator_50.py

Latency note: first 5 docs include GLiNER model load (~430M, CPU).
Subsequent docs are significantly faster.
"""

from __future__ import annotations

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

from beat_docile.config import DATA_ROOT
from beat_docile.data import WordBox, load_split, iter_pages
from beat_docile.disambiguator import DisambigStats, resolve_conflicts
from docile.dataset import Dataset, Field

V5B_50_PATH = PROJECT_ROOT / "predictions" / "v5b_50.json"
OUT_PATH = PROJECT_ROOT / "predictions" / "v5b_50_disambig.json"
RESULT_DIR = PROJECT_ROOT / ".planning" / "phases" / "PHASE-8-gliner-disambiguator"
RESULT_PATH = RESULT_DIR / "RESULT.md"

V5B_KILE = 41.79
V5B_LIR = 49.90


def _load_fields(field_dicts: list[dict]) -> list[Field]:
    return [Field.from_dict(fd) for fd in field_dicts]


def _fields_to_dicts(fields: list[Field]) -> list[dict]:
    return [f.to_dict() for f in fields]


def _build_words_by_page(doc) -> dict[int, list[WordBox]]:
    result: dict[int, list[WordBox]] = {}
    with doc:
        for page in iter_pages(doc):
            result[page.page_index] = page.words
    return result


def _merge_stats(total: DisambigStats, s: DisambigStats) -> None:
    total.n_conflicts += s.n_conflicts
    total.n_resolved += s.n_resolved
    total.n_abstained += s.n_abstained
    for k, v in s.pair_counts.items():
        total.pair_counts[k] = total.pair_counts.get(k, 0) + v
    for k, v in s.pair_resolved.items():
        total.pair_resolved[k] = total.pair_resolved.get(k, 0) + v


def run_eval(
    all_results: dict[str, list[dict]],
) -> dict[str, float]:
    from beat_docile.eval import run_eval as _run_eval, print_scores

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

    # Use custom split name (no index file) so passed docids are accepted as-is
    dataset = Dataset(
        split_name="disambig_50",
        dataset_path=DATA_ROOT,
        load_annotations=True,
        load_ocr=False,
        docids=list(all_results.keys()),
    )
    result = _run_eval(dataset, kile_preds, lir_preds)
    return print_scores(result)


def _fmt_pair_table(pair_counts: dict[str, int], pair_resolved: dict[str, int]) -> str:
    if not pair_counts:
        return "_No conflicts found._\n"
    all_keys = sorted(set(pair_counts) | set(pair_resolved))
    rows = ["| Pair | Triggered | Resolved | Resolution rate |", "|---|---|---|---|"]
    for k in all_keys:
        triggered = pair_counts.get(k, 0)
        resolved = pair_resolved.get(k, 0)
        rate = f"{resolved / triggered:.0%}" if triggered else "—"
        rows.append(f"| `{k}` | {triggered} | {resolved} | {rate} |")
    return "\n".join(rows)


def write_result_md(
    stats: DisambigStats,
    n_docs: int,
    latency_5_docs: float,
    scores_before: dict[str, float],
    scores_after: dict[str, float],
    latency_total: float,
) -> None:
    kile_before = scores_before.get("kile_AP", 0) * 100
    kile_after = scores_after.get("kile_AP", 0) * 100
    lir_before = scores_before.get("lir_f1", 0) * 100
    lir_after = scores_after.get("lir_f1", 0) * 100

    delta_kile = kile_after - kile_before
    delta_lir = lir_after - lir_before

    abstain_rate = stats.n_abstained / stats.n_conflicts if stats.n_conflicts else 0.0
    resolve_rate = stats.n_resolved / stats.n_conflicts if stats.n_conflicts else 0.0

    pair_table = _fmt_pair_table(stats.pair_counts, stats.pair_resolved)

    avg_latency_per_doc = latency_total / n_docs if n_docs else 0.0

    recommend = "INTEGRATE" if delta_kile >= 0.5 else ("MARGINAL — investigate" if delta_kile >= 0 else "NEGATIVE — do not integrate")

    md = f"""# Phase 8 — GLiNER Disambiguator: Results

## Summary

| Metric | V5b baseline | + GLiNER disambig | Delta |
|---|---|---|---|
| KILE AP (50 docs) | {kile_before:.2f}% | {kile_after:.2f}% | **{delta_kile:+.2f}pp** |
| LIR F1 (50 docs) | {lir_before:.2f}% | {lir_after:.2f}% | {delta_lir:+.2f}pp |

V5b full-val baseline: **{V5B_KILE:.2f}% KILE / {V5B_LIR:.2f}% LIR** (500 docs)

## Conflict-group statistics ({n_docs} docs)

| Stat | Value |
|---|---|
| Total conflicts triggered | {stats.n_conflicts} |
| Resolved (winner chosen) | {stats.n_resolved} ({resolve_rate:.0%}) |
| Abstained (GLiNER gap < 0.15) | {stats.n_abstained} ({abstain_rate:.0%}) |
| Avg conflicts / doc | {stats.n_conflicts / n_docs:.1f} |

## Per-pair contribution table

{pair_table}

## Latency

- First 5 docs (incl. model load): {latency_5_docs:.1f}s
- Total 50 docs: {latency_total:.1f}s
- Avg per doc: {avg_latency_per_doc:.2f}s

## Model

- `knowledgator/gliner-multitask-large-v0.5` (Apache 2.0, ~430M DeBERTa-large, CPU)
- Description-based labels per ZeroNER ACL 2025

## Thresholds

| Parameter | Value | Rationale |
|---|---|---|
| `CONFLICT_GAP_THRESHOLD` | 0.20 | V5b score gap below this → "too close to call" |
| `RESOLVE_GAP_THRESHOLD` | 0.15 | GLiNER must be this confident to override V5b |
| `BBOX_OVERLAP_THRESHOLD` | 0.30 | Min overlap fraction of smaller bbox = same span |

## Recommendation

**{recommend}**

Delta KILE: {delta_kile:+.2f}pp vs V5b 50-doc baseline ({kile_before:.2f}%).
"""

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(md)
    print(f"\nWrote RESULT.md → {RESULT_PATH}")


def main() -> None:
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"Input: {V5B_50_PATH}")
    print(f"Output: {OUT_PATH}")

    raw_v5b: dict[str, list[dict]] = json.loads(V5B_50_PATH.read_text())
    docids = list(raw_v5b.keys())
    print(f"\nLoaded {len(docids)} docs from v5b_50.json")

    print("Loading val split for OCR words...")
    dataset = load_split("val")
    docs = {d.docid: d for d in dataset if d.docid in set(docids)}
    print(f"Matched {len(docs)} docs in val split")

    total_stats = DisambigStats()
    disambig_results: dict[str, list[dict]] = {}

    print("\nRunning GLiNER disambiguator...")
    print("(First doc triggers model download/load — may take a minute)\n")

    t_start = time.time()
    t_5docs: float | None = None

    for idx, docid in enumerate(docids):
        if docid not in docs:
            disambig_results[docid] = raw_v5b[docid]
            continue

        doc = docs[docid]
        fields = _load_fields(raw_v5b[docid])

        words_by_page = _build_words_by_page(doc)

        resolved, stats = resolve_conflicts(fields, words_by_page)
        _merge_stats(total_stats, stats)

        disambig_results[docid] = _fields_to_dicts(resolved)

        elapsed = time.time() - t_start
        n_done = idx + 1
        if n_done == 5:
            t_5docs = elapsed

        if n_done % 10 == 0 or n_done == len(docids):
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (len(docids) - n_done) / rate if rate > 0 else float("inf")
            print(
                f"[{n_done}/{len(docids)}] {elapsed:.0f}s elapsed | "
                f"conflicts so far: {total_stats.n_conflicts} | "
                f"resolved: {total_stats.n_resolved} | "
                f"ETA {eta:.0f}s"
            )

    t_total = time.time() - t_start
    if t_5docs is None:
        t_5docs = t_total

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(disambig_results, indent=2))
    print(f"\nSaved {len(disambig_results)} docs → {OUT_PATH}")

    print("\n--- Conflict Statistics ---")
    print(f"  Total conflicts : {total_stats.n_conflicts}")
    print(f"  Resolved        : {total_stats.n_resolved}")
    print(f"  Abstained       : {total_stats.n_abstained}")
    print(f"  Per-pair triggers:")
    for k, v in sorted(total_stats.pair_counts.items(), key=lambda x: -x[1]):
        resolved = total_stats.pair_resolved.get(k, 0)
        print(f"    {k}: {v} conflicts, {resolved} resolved")

    print(f"\n--- Latency ---")
    print(f"  First 5 docs (incl. model load): {t_5docs:.1f}s")
    print(f"  Total {len(docids)} docs: {t_total:.1f}s")
    print(f"  Avg per doc: {t_total / len(docids):.2f}s")

    print("\n--- KILE AP BEFORE (V5b) ---")
    scores_before = run_eval(raw_v5b)

    print("\n--- KILE AP AFTER (+ GLiNER disambig) ---")
    scores_after = run_eval(disambig_results)

    kile_before = scores_before.get("kile_AP", 0) * 100
    kile_after = scores_after.get("kile_AP", 0) * 100
    lir_before = scores_before.get("lir_f1", 0) * 100
    lir_after = scores_after.get("lir_f1", 0) * 100

    print(f"\n{'='*60}")
    print(f"V5b (50 docs):             KILE AP {kile_before:.2f}% / LIR F1 {lir_before:.2f}%")
    print(f"+ GLiNER disambiguator:    KILE AP {kile_after:.2f}% / LIR F1 {lir_after:.2f}%")
    print(f"Delta:                     KILE {kile_after - kile_before:+.2f}pp / LIR {lir_after - lir_before:+.2f}pp")
    print(f"{'='*60}")

    write_result_md(
        stats=total_stats,
        n_docs=len(docids),
        latency_5_docs=t_5docs,
        scores_before=scores_before,
        scores_after=scores_after,
        latency_total=t_total,
    )


if __name__ == "__main__":
    main()
