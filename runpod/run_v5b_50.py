"""V5b validation runner: extract 50 specific val docs, eval, print KILE AP / LIR F1.

Usage on Mac:
  cd /path/to/beat_docile
  DATA_ROOT=$(pwd)/data BD_USE_REFINER=1 BD_USE_VALIDATOR=1 \
    uv run python runpod/run_v5b_50.py predictions/v5b_50.json
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from docile.dataset import Dataset

from beat_docile.config import DATA_ROOT, DEFAULT_MODEL
from beat_docile.extract import extract_documents
from beat_docile.eval import run_eval, print_scores
from beat_docile.fewshot import _build_cluster_index


def main(out_path: str) -> None:
    pred_baseline = Path(__file__).parent.parent / "predictions" / "v5_baseline_50.json"
    docids = list(json.loads(pred_baseline.read_text()).keys())
    assert len(docids) == 50, f"Expected 50 docids, got {len(docids)}"
    print(f"Running V5b on {len(docids)} val docs (first: {docids[0]})")

    dataset = Dataset(
        split_name="v5b_subset", dataset_path=DATA_ROOT,
        load_annotations=True, load_ocr=True, docids=docids,
    )
    docs = list(dataset)
    print(f"Loaded {len(docs)} docs from {DATA_ROOT}")

    print("Building train cluster index for few-shot...")
    train_index = _build_cluster_index("train")
    print(f"Loaded {len(train_index)} clusters")

    print(f"Extracting with model={DEFAULT_MODEL}, refiner=ON, validator=ON, targeted=ON...")
    kile_preds, lir_preds = asyncio.run(
        extract_documents(docs, DEFAULT_MODEL, train_index=train_index,
                          targeted_pass=True, self_consistency=False)
    )
    total_k = sum(len(v) for v in kile_preds.values())
    total_l = sum(len(v) for v in lir_preds.values())
    print(f"Extracted {total_k} KILE / {total_l} LIR fields")

    out = {}
    for did in docids:
        fields = []
        for f in kile_preds.get(did, []):
            fields.append(f.to_dict())
        for f in lir_preds.get(did, []):
            fields.append(f.to_dict())
        out[did] = fields
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"Wrote predictions to {out_path}")

    eval_dataset = Dataset(
        split_name="v5b_eval", dataset_path=DATA_ROOT,
        load_annotations=True, load_ocr=False, docids=docids,
    )
    print("Evaluating...")
    result = run_eval(eval_dataset, kile_preds, lir_preds)
    scores = print_scores(result)
    print("\n=== V5b SCORES (50 docs) ===")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "predictions/v5b_50.json"
    main(out)
