"""[ACTIVE] DocILE evaluator wrapper — KILE AP and LIR F1 via evaluate_dataset.

Status: ACTIVE — used in current best (v2_ensemble).
See KNOWLEDGE_BASE.md §3 for the architecture map.

Uses docile.evaluation.evaluate_dataset Python API directly.
Requires all split docids to be present (even [] for empty). Ref: EVAL_SPEC §3, §4.
"""

from __future__ import annotations

from docile.dataset import Dataset, Field
from docile.evaluation import EvaluationResult, evaluate_dataset


def run_eval(
    dataset: Dataset,
    kile_preds: dict[str, list[Field]],
    lir_preds: dict[str, list[Field]],
) -> EvaluationResult:
    """Run KILE + LIR evaluation. All docids in dataset must appear in both dicts."""
    # Ensure every docid present (evaluator requires it)
    for doc in dataset:
        kile_preds.setdefault(doc.docid, [])
        lir_preds.setdefault(doc.docid, [])

    return evaluate_dataset(
        dataset=dataset,
        docid_to_kile_predictions=kile_preds,
        docid_to_lir_predictions=lir_preds,
    )


def print_scores(result: EvaluationResult) -> dict[str, float]:
    """Print and return primary metrics."""
    scores: dict[str, float] = {}

    for task in ("kile", "lir"):
        if task not in result.task_to_docid_to_matching:
            print(f"{task.upper()}: no predictions")
            continue
        metrics = result.get_metrics(task)
        primary = "AP" if task == "kile" else "f1"
        val = metrics.get(primary, 0.0)
        scores[f"{task}_{primary}"] = val
        print(
            f"{task.upper()} {primary}: {val:.4f}  (precision={metrics.get('precision', 0):.4f}, recall={metrics.get('recall', 0):.4f})"
        )

    return scores
