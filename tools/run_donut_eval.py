#!/usr/bin/env python3
"""Evaluate Qwen2-VL-2B-Instruct on 50 DocILE val docs.

Setup on neon (run once):
  mkdir -p ~/donut_work && cd ~/donut_work
  # Sync source from Mac:
  #   rsync -av <mac>:~/projects/beat_docile/ ~/donut_work/beat_docile/
  pip install transformers accelerate pillow tqdm
  # qwen_vl_utils is optional; fallback is built into donut_extract.py
  pip install qwen-vl-utils 2>/dev/null || true

Run:
  DATA_ROOT=~/docile_data \\
  PREDICTIONS_DIR=~/donut_work/predictions \\
  python tools/run_donut_eval.py \\
    --v5b-path predictions/v5b_50.json \\
    --out predictions/donut_val_50.json \\
    --max-docs 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running directly from project root without pip install
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # DocILE has some truncated images


def _field_to_dict(field) -> dict:
    """Serialize a docile Field to JSON-serializable dict."""
    bbox = field.bbox
    return {
        "bbox": [bbox.left, bbox.top, bbox.right, bbox.bottom],
        "page": field.page,
        "score": float(field.score) if field.score is not None else 0.8,
        "text": None,
        "fieldtype": field.fieldtype,
        "line_item_id": field.line_item_id,
        "use_only_for_ap": False,
    }


def _extraction_to_fields(
    extraction: dict,
    doc,  # DocILE Document
) -> tuple[list, list]:
    """Convert donut extraction dict to (kile_fields, lir_fields).

    Uses align.py's find_span to locate each text value in the snapped OCR words.
    Tries every page; keeps the first match.
    """
    from beat_docile.align import find_span
    from beat_docile.data import iter_pages
    from docile.dataset import BBox, Field

    KILE_TYPES = {
        "account_num", "amount_due", "amount_paid", "amount_total_gross",
        "amount_total_net", "amount_total_tax", "bank_num", "bic",
        "currency_code_amount_due", "customer_billing_address",
        "customer_billing_name", "customer_delivery_address",
        "customer_delivery_name", "customer_id", "customer_order_id",
        "customer_other_address", "customer_other_name",
        "customer_registration_id", "customer_tax_id", "date_due",
        "date_issue", "document_id", "iban", "order_id",
        "payment_reference", "payment_terms", "tax_detail_gross",
        "tax_detail_net", "tax_detail_rate", "tax_detail_tax",
        "vendor_address", "vendor_email", "vendor_name", "vendor_order_id",
        "vendor_registration_id", "vendor_tax_id",
    }
    LIR_TYPES = {
        "line_item_amount_gross", "line_item_amount_net", "line_item_code",
        "line_item_currency", "line_item_date", "line_item_description",
        "line_item_discount_amount", "line_item_discount_rate",
        "line_item_hts_number", "line_item_order_id",
        "line_item_person_name", "line_item_position",
        "line_item_quantity", "line_item_tax", "line_item_tax_rate",
        "line_item_unit_price_gross", "line_item_unit_price_net",
        "line_item_units_of_measure", "line_item_weight",
    }

    # Collect all pages once to avoid re-opening document
    pages = list(iter_pages(doc))

    def _span_to_field(text: str, fieldtype: str, line_item_id=None) -> "Field | None":
        """Try each page; return Field on first successful alignment."""
        for page_ctx in pages:
            span = find_span(text, page_ctx.words, min_ratio=0.70)
            if span is None:
                continue
            start, end = span
            ws = page_ctx.words[start: end + 1]
            bbox = BBox(
                min(w.bbox[0] for w in ws),
                min(w.bbox[1] for w in ws),
                max(w.bbox[2] for w in ws),
                max(w.bbox[3] for w in ws),
            )
            return Field(
                bbox=bbox,
                page=page_ctx.page_index,
                fieldtype=fieldtype,
                score=0.8,
                line_item_id=line_item_id,
            )
        return None

    kile_fields: list = []
    lir_fields: list = []

    # ── KILE ─────────────────────────────────────────────────────────────────
    for ft, val in extraction.get("kile", {}).items():
        if ft not in KILE_TYPES:
            continue
        values: list[str] = val if isinstance(val, list) else [val]
        for text in values:
            text = str(text).strip()
            if not text:
                continue
            f = _span_to_field(text, ft)
            if f is not None:
                kile_fields.append(f)

    # ── LIR ──────────────────────────────────────────────────────────────────
    for li_idx, item in enumerate(extraction.get("line_items", []), start=1):
        for ft, text in item.items():
            if ft not in LIR_TYPES:
                continue
            text = str(text).strip()
            if not text:
                continue
            f = _span_to_field(text, ft, line_item_id=li_idx)
            if f is not None:
                lir_fields.append(f)

    return kile_fields, lir_fields


def main() -> None:
    parser = argparse.ArgumentParser(description="Donut/Qwen2-VL DocILE eval")
    parser.add_argument(
        "--v5b-path",
        default="predictions/v5b_50.json",
        help="Path to v5b_50.json (source of the 50 docids)",
    )
    parser.add_argument(
        "--out",
        default="predictions/donut_val_50.json",
        help="Output predictions JSON path",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=50,
        help="Stop after N docs (use 5-10 for a quick probe)",
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="HuggingFace model ID",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip DocILE evaluation (useful if docile not installed)",
    )
    args = parser.parse_args()

    # ── Load docids from v5b reference ───────────────────────────────────────
    v5b_path = Path(args.v5b_path)
    if not v5b_path.exists():
        print(f"ERROR: v5b path not found: {v5b_path}", file=sys.stderr)
        sys.exit(1)
    with open(v5b_path) as f:
        v5b = json.load(f)
    docids = list(v5b.keys())[: args.max_docs]
    print(f"Running on {len(docids)} docs (model: {args.model_id})")

    # ── GPU memory check ─────────────────────────────────────────────────────
    try:
        import torch
        if torch.cuda.is_available():
            free_gb = torch.cuda.mem_get_info()[0] / 1e9
            print(f"GPU free: {free_gb:.1f} GB")
            if free_gb < 3.5:
                print("WARNING: < 3.5 GB free — model may OOM", file=sys.stderr)
    except Exception:
        pass

    # ── Load model ────────────────────────────────────────────────────────────
    from beat_docile.donut_extract import extract_donut_images, load_model

    print("Loading model...")
    model, processor = load_model(args.model_id, args.device)

    try:
        import torch
        if torch.cuda.is_available():
            used_gb = torch.cuda.memory_allocated() / 1e9
            print(f"GPU after model load: {used_gb:.1f} GB allocated")
    except Exception:
        pass

    # ── Load DocILE dataset subset ────────────────────────────────────────────
    from beat_docile.config import DATA_ROOT
    from beat_docile.data import iter_pages
    from docile.dataset import Dataset

    print(f"Loading DocILE val subset ({len(docids)} docs) from {DATA_ROOT}...")
    dataset = Dataset(
        split_name="val",
        dataset_path=DATA_ROOT,
        load_annotations=True,
        load_ocr=True,
    )
    # Build lookup by docid
    docid_to_doc = {doc.docid: doc for doc in dataset if doc.docid in set(docids)}
    print(f"Matched {len(docid_to_doc)}/{len(docids)} docids in dataset")

    # ── Inference loop ────────────────────────────────────────────────────────
    try:
        from tqdm import tqdm
        bar = tqdm(docids, desc="extracting")
    except ImportError:
        bar = docids  # type: ignore[assignment]

    kile_preds: dict[str, list] = {}
    lir_preds: dict[str, list] = {}
    raw_log: dict[str, list[str]] = {}
    n_aligned_kile = 0
    n_aligned_lir = 0

    for docid in bar:
        doc = docid_to_doc.get(docid)
        if doc is None:
            kile_preds[docid] = []
            lir_preds[docid] = []
            continue

        # Get page images from DocILE (uses snapped OCR coords)
        pages = list(iter_pages(doc))
        images = [p.image for p in pages]

        extraction = extract_donut_images(images, model, processor, args.device)
        raw_log[docid] = extraction.get("raw_outputs", [])

        kile_fields, lir_fields = _extraction_to_fields(extraction, doc)
        kile_preds[docid] = kile_fields
        lir_preds[docid] = lir_fields
        n_aligned_kile += len(kile_fields)
        n_aligned_lir += len(lir_fields)

    # Ensure all docids present (evaluator requires it)
    for docid in docids:
        kile_preds.setdefault(docid, [])
        lir_preds.setdefault(docid, [])

    print(
        f"Aligned: {n_aligned_kile} KILE fields, {n_aligned_lir} LIR fields "
        f"across {len(docids)} docs"
    )

    # ── Serialize predictions ─────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    combined: dict[str, list] = {}
    for docid in docids:
        combined[docid] = (
            [_field_to_dict(f) for f in kile_preds.get(docid, [])]
            + [_field_to_dict(f) for f in lir_preds.get(docid, [])]
        )

    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Saved predictions → {out_path}")

    # Save raw model outputs for debugging
    raw_path = out_path.with_suffix(".raw.json")
    with open(raw_path, "w") as f:
        json.dump(raw_log, f, indent=2)
    print(f"Saved raw outputs → {raw_path}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    if args.no_eval:
        print("Skipping eval (--no-eval)")
        return

    from beat_docile.eval import print_scores, run_eval

    # Build eval-compatible Dataset (subset)
    eval_dataset = Dataset(
        split_name="val",
        dataset_path=DATA_ROOT,
        load_annotations=True,
        load_ocr=False,
    )
    # Filter to just our docids
    eval_kile = {d: kile_preds.get(d, []) for d in docids}
    eval_lir = {d: lir_preds.get(d, []) for d in docids}

    # DocILE evaluator needs ALL split docids — pad with empty for the rest
    for doc in eval_dataset:
        eval_kile.setdefault(doc.docid, [])
        eval_lir.setdefault(doc.docid, [])

    print("\nRunning DocILE evaluation...")
    result = run_eval(eval_dataset, eval_kile, eval_lir)
    scores = print_scores(result)
    print(f"\nKILE AP : {scores.get('kile_AP', 0):.4f}")
    print(f"LIR F1  : {scores.get('lir_f1', 0):.4f}")
    print("\nBaseline (V5b 50-doc): KILE 41.86% / LIR 52.36%")
    kile_delta = scores.get("kile_AP", 0) - 0.4186
    lir_delta = scores.get("lir_f1", 0) - 0.5236
    print(f"Delta vs V5b:  KILE {kile_delta:+.4f}  LIR {lir_delta:+.4f}")


if __name__ == "__main__":
    main()
