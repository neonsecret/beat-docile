#!/usr/bin/env python
"""A/B test: classifier candidate augmentation (Option C = B + A) on V5b 50-doc subset.

Option B — Rerank: multiply each Claude prediction's score by the classifier's score
            for the same bbox span.  Lower-scored hallucinations sink; good matches stay.
Option A — Augment: for each KILE fieldtype absent from Claude's predictions, add the
            classifier's top candidate if its score > 0.85.

Both options together = Option C (the recommended setting run here).

Usage:
    uv run python tools/run_cls_aug_ab.py [--skip-eval]

Output:
    predictions/v5b_clsaug_50.json

Baseline: V5b 50-doc  KILE AP = 41.86%   LIR F1 = 52.36%
Gate:      KILE AP > 43%  →  run on full 500-val
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
os.environ.setdefault("DATA_ROOT", str(PROJECT_ROOT / "data"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


from beat_docile.classifiers import _ALL_FIELDTYPES, _parse_ocr_words  # noqa: E402
from beat_docile.cls_candidates import (  # noqa: E402
    _SKIP_FOR_RECALL,
    CandidateSpan,
    _default_max_span,
    generate_candidates,
    score_bbox_span,
)
from beat_docile.data import WordBox  # noqa: E402

_LOG = logging.getLogger(__name__)

MODEL_DIR = PROJECT_ROOT / "models" / "classifiers"
V5B_PATH = PROJECT_ROOT / "predictions" / "v5b_50.json"
OUT_PATH = PROJECT_ROOT / "predictions" / "v5b_clsaug_50.json"
OCR_DIR = PROJECT_ROOT / "data" / "ocr"

V5B_BASELINE = {"kile_ap": 41.86, "lir_f1": 52.36}

# Option B: floor for classifier factor so a neutral 0.5 fallback doesn't halve Claude scores.
_MIN_CLS_FACTOR = 0.7
# Option A: add recalled fields only when classifier confidence exceeds this threshold.
_RECALL_THRESHOLD = 0.85
# Sliding-window threshold for generating Option A candidates.
_GEN_THRESHOLD = 0.7


# ---------------------------------------------------------------------------
# OCR loading
# ---------------------------------------------------------------------------


def load_ocr_words(docid: str) -> list[list[WordBox]]:
    """Load OCR word lists per page from the local OCR JSON.

    Returns pages_words[page_idx] = list[WordBox] with global word IDs.
    """
    ocr_path = OCR_DIR / f"{docid}.json"
    if not ocr_path.exists():
        return []
    with ocr_path.open() as f:
        ocr_data = json.load(f)
    return _parse_ocr_words(ocr_data)


# ---------------------------------------------------------------------------
# Option B: rerank existing predictions by classifier score
# ---------------------------------------------------------------------------


def apply_option_b(
    fields: list[dict],
    pages_words: list[list[WordBox]],
) -> list[dict]:
    """Multiply each prediction's score by its classifier score (Option B).

    Falls back to the original score when the classifier model is missing or
    no OCR words are covered by the prediction bbox.
    """
    updated: list[dict] = []
    for fd in fields:
        ft = fd["fieldtype"]
        page = int(fd["page"])
        bbox: tuple[float, float, float, float] = tuple(fd["bbox"])  # type: ignore[assignment]
        orig_score = float(fd.get("score", 0.8))

        if page < len(pages_words):
            page_words = pages_words[page]
            cls = score_bbox_span(ft, bbox, page_words, MODEL_DIR)
            factor = max(cls, _MIN_CLS_FACTOR)
            new_score = orig_score * factor
        else:
            new_score = orig_score

        updated.append({**fd, "score": new_score})
    return updated


# ---------------------------------------------------------------------------
# Option A: add missed KILE fieldtypes
# ---------------------------------------------------------------------------


def apply_option_a(
    fields: list[dict],
    pages_words: list[list[WordBox]],
) -> list[dict]:
    """For every KILE fieldtype absent from fields, attempt classifier recall (Option A).

    Adds at most one prediction per absent fieldtype (the highest-scoring candidate
    across all pages), provided the score exceeds _RECALL_THRESHOLD.
    LIR fieldtypes and weak classifiers are skipped (see _SKIP_FOR_RECALL).
    """
    present_kile: set[str] = {
        fd["fieldtype"] for fd in fields if fd.get("line_item_id") is None
    }

    additions: list[dict] = []
    for ft in _ALL_FIELDTYPES:
        if ft in present_kile or ft in _SKIP_FOR_RECALL:
            continue

        best: CandidateSpan | None = None
        for page_words in pages_words:
            if not page_words:
                continue
            candidates = generate_candidates(
                fieldtype=ft,
                words=page_words,
                page_w=1.0,
                page_h=1.0,
                model_dir=MODEL_DIR,
                max_span_words=_default_max_span(ft),
                score_threshold=_GEN_THRESHOLD,
            )
            if candidates:
                top = candidates[0]
                if best is None or top.score > best.score:
                    best = top

        if best is not None and best.score >= _RECALL_THRESHOLD:
            additions.append({
                "bbox": list(best.bbox),
                "page": best.page,
                "score": best.score,
                "text": best.text or None,
                "fieldtype": ft,
                "line_item_id": None,
                "use_only_for_ap": False,
            })
            _LOG.debug("Option A added %s  score=%.3f  text=%r", ft, best.score, best.text)

    return fields + additions


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _load_dataset_subset(docids: list[str]):
    from docile.dataset import Dataset

    from beat_docile.config import DATA_ROOT

    return Dataset(
        split_name="smoke_subset",
        dataset_path=DATA_ROOT,
        load_annotations=True,
        load_ocr=False,
        docids=docids,
    )


def evaluate(preds_path: Path, docids: list[str]) -> dict[str, float]:
    from docile.dataset import Field

    from beat_docile.eval import print_scores, run_eval

    raw = json.loads(preds_path.read_text())
    kile_preds: dict[str, list[Field]] = {}
    lir_preds: dict[str, list[Field]] = {}
    for docid, fields in raw.items():
        kile_preds[docid] = []
        lir_preds[docid] = []
        for fd in fields:
            f = Field.from_dict(fd)
            if f.line_item_id is not None:
                lir_preds[docid].append(f)
            else:
                kile_preds[docid].append(f)

    dataset = _load_dataset_subset(docids)
    result = run_eval(dataset, kile_preds, lir_preds)
    return print_scores(result)


def per_field_analysis(preds_path: Path, docids: list[str]) -> None:
    """Print per-fieldtype AP / F1 for quick diagnosis."""
    from docile.dataset import Field

    from beat_docile.eval import run_eval

    raw = json.loads(preds_path.read_text())
    kile_preds: dict[str, list[Field]] = {}
    lir_preds: dict[str, list[Field]] = {}
    for docid, fields in raw.items():
        kile_preds[docid] = []
        lir_preds[docid] = []
        for fd in fields:
            f = Field.from_dict(fd)
            if f.line_item_id is not None:
                lir_preds[docid].append(f)
            else:
                kile_preds[docid].append(f)

    dataset = _load_dataset_subset(docids)
    result = run_eval(dataset, kile_preds, lir_preds)

    print("\n=== Per-fieldtype breakdown ===")
    for task in ("kile", "lir"):
        if task not in result.task_to_docid_to_matching:
            continue
        try:
            per_field = result.get_metrics_per_field(task)
            print(f"\n{task.upper()}:")
            for ft, m in sorted(
                per_field.items(),
                key=lambda x: x[1].get("AP", x[1].get("f1", 0)),
            ):
                metric = m.get("AP", m.get("f1", 0))
                print(f"  {ft:42s} {metric:.4f}")
        except Exception as exc:
            print(f"  (per-field breakdown unavailable: {exc})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    skip_eval = "--skip-eval" in sys.argv

    raw = json.loads(V5B_PATH.read_text())
    docids = list(raw.keys())
    print(f"Loaded {len(docids)} docs from {V5B_PATH.name}")

    t0 = time.time()
    augmented: dict[str, list[dict]] = {}

    for i, (docid, fields) in enumerate(raw.items(), 1):
        pages_words = load_ocr_words(docid)
        if not pages_words:
            _LOG.warning("No OCR for %s — keeping original predictions", docid)
            augmented[docid] = fields
            continue

        fields_b = apply_option_b(fields, pages_words)
        fields_c = apply_option_a(fields_b, pages_words)
        augmented[docid] = fields_c

        if i % 10 == 0 or i == len(docids):
            elapsed = time.time() - t0
            print(f"  [{i}/{len(docids)}]  elapsed={elapsed:.1f}s")

    OUT_PATH.write_text(json.dumps(augmented, indent=2))
    print(f"\nAugmented predictions written to {OUT_PATH}")

    orig_total = sum(len(v) for v in raw.values())
    aug_total = sum(len(v) for v in augmented.values())
    print(f"Original fields: {orig_total}  →  Augmented: {aug_total}  (+{aug_total - orig_total})")

    if skip_eval:
        print("Skipping evaluation (--skip-eval).")
        return

    print("\n=== Evaluation ===")
    scores = evaluate(OUT_PATH, docids)

    print("\n=== Delta vs V5b baseline ===")
    kile_val = scores.get("kile_AP", 0) * 100
    lir_val = scores.get("lir_f1", 0) * 100
    kile_delta = kile_val - V5B_BASELINE["kile_ap"]
    lir_delta = lir_val - V5B_BASELINE["lir_f1"]
    print(f"KILE AP: {kile_val:.2f}%  (baseline {V5B_BASELINE['kile_ap']:.2f}%)  delta={kile_delta:+.2f}pp")
    print(f"LIR F1:  {lir_val:.2f}%  (baseline {V5B_BASELINE['lir_f1']:.2f}%)  delta={lir_delta:+.2f}pp")

    if kile_val > 43.0:
        print("\n*** Gate PASSED (KILE AP > 43%). Run on full 500-val next. ***")
    else:
        print("\nGate not met (KILE AP ≤ 43%). Investigate per-field breakdown.")

    per_field_analysis(OUT_PATH, docids)


if __name__ == "__main__":
    main()
