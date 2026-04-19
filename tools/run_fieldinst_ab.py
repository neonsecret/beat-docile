"""A/B test: field_instructions guidance — extract 50 val docs, eval, compare vs V5b baseline."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Force env flags before any imports
os.environ["BD_USE_REFINER"] = "1"
os.environ["BD_USE_VALIDATOR"] = "1"
os.environ["BD_USE_BBOX_VERIFY"] = "0"

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from beat_docile.config import DATA_ROOT, DEFAULT_MODEL
from beat_docile.data import load_split
from beat_docile.extract import extract_documents
from beat_docile.eval import run_eval, print_scores
from docile.dataset import Dataset, Field

V5B_BASELINE = {"kile_ap": 41.86, "lir_f1": 52.36}
PREDICTIONS_PATH = Path(__file__).parent.parent / "predictions" / "v5b_fieldinst_50.json"
DOCIDS_PATH = Path(__file__).parent.parent / "predictions" / "v5b_50.json"


def get_50_docids() -> list[str]:
    raw = json.loads(DOCIDS_PATH.read_text())
    return list(raw.keys())


async def run_extraction(docids: list[str]) -> dict[str, list]:
    dataset = load_split("val")
    docs = [d for d in dataset if d.docid in set(docids)]
    print(f"Loaded {len(docs)} docs from val split")

    from beat_docile.fewshot import _build_cluster_index
    print("Building train cluster index...")
    train_index = _build_cluster_index("train")
    print(f"  {len(train_index)} clusters loaded")

    # ThreadPoolExecutor for parallel I/O; asyncio handles the concurrency
    loop = asyncio.get_event_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=8))

    print("Running extraction (REFINER=1, VALIDATOR=1, BBOX_VERIFY=0)...")
    kile_preds, lir_preds = await extract_documents(
        docs, DEFAULT_MODEL, train_index=train_index, targeted_pass=True, self_consistency=False
    )

    output: dict[str, list] = {}
    for doc in docs:
        fields_out = []
        for f in kile_preds.get(doc.docid, []):
            fields_out.append(f.to_dict())
        for f in lir_preds.get(doc.docid, []):
            fields_out.append(f.to_dict())
        output[doc.docid] = fields_out

    PREDICTIONS_PATH.write_text(json.dumps(output, indent=2))
    print(f"Predictions written to {PREDICTIONS_PATH}")
    return output


def evaluate(docids: list[str]) -> dict[str, float]:
    raw = json.loads(PREDICTIONS_PATH.read_text())
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

    subset_dataset = Dataset(
        split_name="smoke_subset",
        dataset_path=DATA_ROOT,
        load_annotations=True,
        load_ocr=False,
        docids=docids,
    )

    result = run_eval(subset_dataset, kile_preds, lir_preds)
    scores = print_scores(result)
    return scores


def per_field_analysis(docids: list[str]) -> None:
    """Print per-fieldtype AP or F1 to spot regressions."""
    raw = json.loads(PREDICTIONS_PATH.read_text())
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

    subset_dataset = Dataset(
        split_name="smoke_subset",
        dataset_path=DATA_ROOT,
        load_annotations=True,
        load_ocr=False,
        docids=docids,
    )

    from beat_docile.eval import run_eval
    result = run_eval(subset_dataset, kile_preds, lir_preds)

    print("\n=== Per-fieldtype breakdown ===")
    for task in ("kile", "lir"):
        if task not in result.task_to_docid_to_matching:
            continue
        try:
            per_field = result.get_metrics_per_field(task)
            print(f"\n{task.upper()}:")
            for ft, m in sorted(per_field.items(), key=lambda x: x[1].get("AP", x[1].get("f1", 0))):
                metric = m.get("AP", m.get("f1", 0))
                print(f"  {ft:40s} {metric:.4f}")
        except Exception as e:
            print(f"  (per-field breakdown unavailable: {e})")


def main() -> None:
    docids = get_50_docids()
    print(f"Running A/B test on {len(docids)} docids")

    if PREDICTIONS_PATH.exists():
        print(f"Predictions already exist at {PREDICTIONS_PATH} — skipping extraction")
    else:
        asyncio.run(run_extraction(docids))

    print("\n=== Evaluation ===")
    scores = evaluate(docids)

    print("\n=== Delta vs V5b baseline ===")
    kile_delta = scores.get("kile_AP", 0) * 100 - V5B_BASELINE["kile_ap"]
    lir_delta = scores.get("lir_f1", 0) * 100 - V5B_BASELINE["lir_f1"]
    print(f"KILE AP: {scores.get('kile_AP', 0)*100:.2f}%  (baseline {V5B_BASELINE['kile_ap']:.2f}%)  delta={kile_delta:+.2f}pp")
    print(f"LIR F1:  {scores.get('lir_f1', 0)*100:.2f}%  (baseline {V5B_BASELINE['lir_f1']:.2f}%)  delta={lir_delta:+.2f}pp")

    if kile_delta > 0 and lir_delta > 0:
        print("\nVerdict: BOTH metrics improved — KEEP field guidance.")
    elif kile_delta < 0 or lir_delta < 0:
        print("\nVerdict: At least one metric regressed — REVERT or investigate.")
    else:
        print("\nVerdict: No clear signal.")

    per_field_analysis(docids)


if __name__ == "__main__":
    main()
