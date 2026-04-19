#!/usr/bin/env python
"""Per-field-type AP breakdown: v2 vs refiner_guard_500.

Uses existing predictions — no API calls. Runs DocILE eval once for each KILE
field type by filtering predictions and GT to that type.

Outputs:
  predictions/refiner_per_field_breakdown.json
  predictions/refiner_per_field_report.md

Usage:
    DATA_ROOT=data uv run python tools/per_field_breakdown.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
os.environ.setdefault("DATA_ROOT", str(PROJECT_ROOT / "data"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from beat_docile.config import DATA_ROOT
from docile.dataset import Dataset, Field

V2_PATH = PROJECT_ROOT / "predictions" / "v2_preds.json"
GUARD_PATH = PROJECT_ROOT / "predictions" / "refiner_guard_500.json"
OUT_JSON = PROJECT_ROOT / "predictions" / "refiner_per_field_breakdown.json"
OUT_MD = PROJECT_ROOT / "predictions" / "refiner_per_field_report.md"

_KILE_TYPES = [
    "account_num", "amount_due", "amount_paid", "amount_total_gross", "amount_total_net",
    "amount_total_tax", "bank_num", "bic", "currency_code_amount_due",
    "customer_billing_address", "customer_billing_name", "customer_delivery_address",
    "customer_delivery_name", "customer_id", "customer_order_id", "customer_other_address",
    "customer_other_name", "customer_registration_id", "customer_tax_id", "date_due",
    "date_issue", "document_id", "iban", "order_id", "payment_reference", "payment_terms",
    "tax_detail_gross", "tax_detail_net", "tax_detail_rate", "tax_detail_tax",
    "vendor_address", "vendor_email", "vendor_name", "vendor_order_id",
    "vendor_registration_id", "vendor_tax_id",
]


def compute_ap_from_matching(matching_list, n_gt: int) -> float:
    """Compute AP from a list of (pred, is_tp) sorted by score desc."""
    if n_gt == 0:
        return float("nan")
    tp_count = 0
    fp_count = 0
    ap = 0.0
    prev_recall = 0.0
    for _, is_tp in matching_list:
        if is_tp:
            tp_count += 1
            recall = tp_count / n_gt
            precision = tp_count / (tp_count + fp_count)
            ap += precision * (recall - prev_recall)
            prev_recall = recall
        else:
            fp_count += 1
    return ap


def load_kile_preds(path: Path) -> dict[str, list[Field]]:
    raw = json.loads(path.read_text())
    result: dict[str, list[Field]] = {}
    for docid, fields in raw.items():
        result[docid] = [Field.from_dict(f) for f in fields if f.get("line_item_id") is None]
    return result


def per_field_ap(
    all_preds: dict[str, list[Field]],
    dataset: Dataset,
    fieldtype: str,
) -> float:
    from beat_docile.eval import run_eval

    filtered_preds: dict[str, list[Field]] = {
        docid: [f for f in fields if f.fieldtype == fieldtype]
        for docid, fields in all_preds.items()
    }
    empty_lir: dict[str, list[Field]] = {docid: [] for docid in all_preds}

    try:
        result = run_eval(dataset, filtered_preds, empty_lir)
    except Exception:
        return float("nan")
    kile_matching = result.task_to_docid_to_matching.get("kile", {})

    # Collect all predictions with match status across docs, sorted by score
    all_pred_match: list[tuple[float, bool]] = []
    n_gt = 0
    for docid, matching in kile_matching.items():
        for pred, match in matching.ordered_predictions_with_match:
            if pred.fieldtype != fieldtype:
                continue
            all_pred_match.append((pred.score, match is not None))
        for fn in matching.false_negatives:
            if fn.fieldtype == fieldtype:
                n_gt += 1
        for pred, match in matching.ordered_predictions_with_match:
            if pred.fieldtype == fieldtype and match is not None:
                n_gt += 1

    all_pred_match.sort(key=lambda x: -x[0])
    return compute_ap_from_matching([(None, is_tp) for _, is_tp in all_pred_match], n_gt)


def main() -> None:
    print("Loading predictions...")
    v2_preds = load_kile_preds(V2_PATH)
    guard_preds = load_kile_preds(GUARD_PATH)
    docids = sorted(v2_preds.keys())
    print(f"  v2: {len(v2_preds)} docs | guard: {len(guard_preds)} docs")

    print("Loading val dataset...")
    dataset = Dataset(
        split_name="val",
        dataset_path=DATA_ROOT,
        load_annotations=True,
        load_ocr=False,
    )

    results = []
    for i, ft in enumerate(_KILE_TYPES):
        print(f"  [{i+1:2d}/36] {ft} ...", end=" ", flush=True)
        v2_ap = per_field_ap(v2_preds, dataset, ft)
        guard_ap = per_field_ap(guard_preds, dataset, ft)
        delta = guard_ap - v2_ap
        print(f"v2={v2_ap:.4f} guard={guard_ap:.4f} Δ={delta:+.4f}")
        results.append({"fieldtype": ft, "v2_ap": v2_ap, "guard_ap": guard_ap, "delta": delta})

    # Sort by delta (most benefited → most harmed)
    results.sort(key=lambda x: -x["delta"])

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nSaved JSON → {OUT_JSON}")

    # Markdown report
    lines = [
        "# Refiner Guard vs v2: Per-Field KILE AP Breakdown (500 docs)",
        "",
        "Sorted by `guard_ap - v2_ap` (most benefited at top, most harmed at bottom).",
        "",
        "| Field Type | v2 AP | guard AP | Δ | Verdict |",
        "|---|---|---|---|---|",
    ]
    positive = []
    negative = []
    for r in results:
        ft = r["fieldtype"]
        v2 = r["v2_ap"]
        gd = r["guard_ap"]
        delta = r["delta"]
        verdict = "✅ refiner helps" if delta > 0.005 else ("❌ refiner hurts" if delta < -0.005 else "➖ neutral")
        lines.append(f"| {ft} | {v2:.4f} | {gd:.4f} | {delta:+.4f} | {verdict} |")
        if delta > 0.005:
            positive.append(ft)
        elif delta < -0.005:
            negative.append(ft)

    lines += [
        "",
        "## Summary",
        f"- **Fields where refiner helps** ({len(positive)}): {', '.join(positive) if positive else 'none'}",
        f"- **Fields where refiner hurts** ({len(negative)}): {', '.join(negative) if negative else 'none'}",
        "",
    ]

    if positive and len(positive) <= 10 and len(negative) >= 10:
        lines.append("## Selective refiner candidate")
        lines.append(f"A selective refiner active ONLY for: `{', '.join(positive)}` might recover some lost AP.")
        lines.append("Estimated impact: rerun with refiner ON for only these field types.")

    OUT_MD.write_text("\n".join(lines))
    print(f"Saved Markdown → {OUT_MD}")

    print("\n=== TOP WINS (refiner helps most) ===")
    for r in results[:5]:
        print(f"  {r['fieldtype']:40s} Δ={r['delta']:+.4f}")
    print("\n=== TOP LOSSES (refiner hurts most) ===")
    for r in results[-5:]:
        print(f"  {r['fieldtype']:40s} Δ={r['delta']:+.4f}")


if __name__ == "__main__":
    main()
