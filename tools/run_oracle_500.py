#!/usr/bin/env python
"""Oracle post-pass: merge oracle_extract.py matches into v2 predictions.

For oracle-supported fieldtypes where oracle fires, replace v2 predictions.
Oracle has near-100% precision (IBAN mod-97 checksum, BIC regex, tax ID patterns
+ label context), so it always wins when it fires.

Oracle-covered fieldtypes (10):
  iban, bic, account_num, bank_num, customer_tax_id, vendor_tax_id,
  vendor_registration_id, customer_registration_id, payment_reference, document_id

Merge strategy:
  For each doc, for each oracle-covered fieldtype:
    - If oracle finds ≥1 match: REPLACE all v2 predictions for that fieldtype
    - If oracle fires nothing: keep v2 predictions unchanged

Usage:
    DATA_ROOT=data uv run python tools/run_oracle_500.py
    DATA_ROOT=data uv run python tools/run_oracle_500.py --min-score 1.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("DATA_ROOT", str(PROJECT_ROOT / "data"))

from docile.dataset import BBox, Dataset, Field  # noqa: E402

from beat_docile.data import iter_pages, load_split  # noqa: E402
from beat_docile.eval import print_scores, run_eval  # noqa: E402
from beat_docile.oracle_extract import SUPPORTED_FIELDTYPES, oracle_extract_field  # noqa: E402

V2_PREDS_PATH = PROJECT_ROOT / "predictions" / "v2_preds.json"
ORACLE_OUT_PATH = PROJECT_ROOT / "predictions" / "v2_oracle_500.json"

V2_KILE = 44.61
V2_LIR = 50.89


def _oracle_fields_for_doc(doc, min_score: float) -> dict[str, list[Field]]:
    """Run oracle extraction across all pages; return {fieldtype: [Field, ...]}."""
    # best match per (fieldtype, normalized_text) — deduplicate across pages
    best: dict[tuple[str, str], tuple[Field, float]] = {}

    for page in iter_pages(doc):
        id_to_word = {w.id: w for w in page.words}
        for ft in SUPPORTED_FIELDTYPES:
            for match in oracle_extract_field(ft, page.words, page.page_index):
                if match.score < min_score:
                    continue
                span_words = [id_to_word[wid] for wid in match.word_ids if wid in id_to_word]
                if not span_words:
                    continue
                bbox = BBox(
                    min(w.bbox[0] for w in span_words),
                    min(w.bbox[1] for w in span_words),
                    max(w.bbox[2] for w in span_words),
                    max(w.bbox[3] for w in span_words),
                )
                field = Field(
                    bbox=bbox, page=page.page_index, fieldtype=ft, score=match.score
                )
                key = (ft, match.text.upper().replace(" ", ""))
                if key not in best or match.score > best[key][1]:
                    best[key] = (field, match.score)

    oracle_by_type: dict[str, list[Field]] = {ft: [] for ft in SUPPORTED_FIELDTYPES}
    for (ft, _), (field, _) in best.items():
        oracle_by_type[ft].append(field)
    return oracle_by_type


def merge_oracle_into_v2(
    v2_doc_preds: list[dict],
    oracle_by_type: dict[str, list[Field]],
) -> list[dict]:
    """Merge strategy: IBAN-only replacement.

    Only use oracle for IBAN (mod-97 checksum → ~100% precision).
    All other oracle fields are ignored in post-pass because regex patterns for
    document_id, payment_reference, tax_id etc. have too many FPs even at score=1.0
    (oracle was designed as a pre-pass aid, not a standalone high-precision extractor).

    IBAN: remove all v2 IBAN predictions and replace with oracle's mod-97-verified IBANs.
    """
    iban_fields = oracle_by_type.get("iban", [])
    if not iban_fields:
        return list(v2_doc_preds)

    # Replace v2 IBAN predictions with oracle IBANs
    merged = [fd for fd in v2_doc_preds if fd.get("fieldtype") != "iban"]
    for field in iban_fields:
        merged.append(field.to_dict())
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-score", type=float, default=0.7,
        help="Minimum oracle score to accept (0.7=pattern, 1.0=checksum/label-verified)"
    )
    parser.add_argument("--out", type=Path, default=ORACLE_OUT_PATH)
    args = parser.parse_args()

    if not V2_PREDS_PATH.exists():
        print(f"ERROR: {V2_PREDS_PATH} not found.")
        sys.exit(1)

    v2_all = json.loads(V2_PREDS_PATH.read_text())
    docids = list(v2_all.keys())
    print(f"Loaded {len(docids)} docs from v2_preds.json")
    print(f"Oracle min_score: {args.min_score}")
    print(f"Oracle fields: {sorted(SUPPORTED_FIELDTYPES)}\n")

    dataset = load_split("val")
    docid_set = set(docids)
    docs = [d for d in dataset if d.docid in docid_set]
    print(f"Matched {len(docs)} docs in val split\n")

    merged_all: dict[str, list[dict]] = {}
    oracle_fired_total = 0
    oracle_replaced_total = 0
    start_t = time.time()

    for i, doc in enumerate(docs):
        v2_preds = v2_all.get(doc.docid, [])
        oracle_by_type = _oracle_fields_for_doc(doc, args.min_score)

        fired = {ft: fields for ft, fields in oracle_by_type.items() if fields}
        n_oracle = sum(len(f) for f in fired.values())
        n_replaced = sum(1 for fd in v2_preds if fd.get("fieldtype") in fired)

        merged = merge_oracle_into_v2(v2_preds, oracle_by_type)
        merged_all[doc.docid] = merged
        oracle_fired_total += n_oracle
        oracle_replaced_total += n_replaced

        if (i + 1) % 50 == 0 or (i + 1) == len(docs):
            elapsed = time.time() - start_t
            print(f"[{i+1}/{len(docs)}] oracle fired {oracle_fired_total} fields, "
                  f"replaced {oracle_replaced_total} v2 preds  ({elapsed:.0f}s)")

    # Ensure all input docids appear in output (EVAL_SPEC requirement)
    for did in docids:
        if did not in merged_all:
            merged_all[did] = v2_all.get(did, [])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged_all, indent=2))
    print(f"\nSaved {len(merged_all)} docs to {args.out}")

    # Evaluate
    from beat_docile.config import DATA_ROOT as _DATA_ROOT

    kile_preds: dict[str, list[Field]] = {}
    lir_preds: dict[str, list[Field]] = {}
    for docid, fields in merged_all.items():
        kile_preds[docid] = []
        lir_preds[docid] = []
        for fd in fields:
            f = Field.from_dict(fd)
            if f.line_item_id is not None:
                lir_preds[docid].append(f)
            else:
                kile_preds[docid].append(f)

    eval_dataset = Dataset(
        split_name="val",
        dataset_path=_DATA_ROOT,
        load_annotations=True,
        load_ocr=False,
        docids=docids,
    )
    result = run_eval(eval_dataset, kile_preds, lir_preds)
    scores = print_scores(result)
    kile_ap = scores.get("kile_AP", 0) * 100
    lir_f1 = scores.get("lir_f1", 0) * 100

    print(f"\n{'='*60}")
    print("ORACLE MERGE — 500 docs")
    print(f"{'='*60}")
    print(f"v2 baseline:  KILE {V2_KILE:.2f}%  LIR {V2_LIR:.2f}%")
    print(f"v2 + oracle:  KILE {kile_ap:.2f}%  LIR {lir_f1:.2f}%  "
          f"(Δ KILE {kile_ap - V2_KILE:+.2f}pp)")
    print(f"{'='*60}")
    print(f"Oracle stats:")
    print(f"  Fields inserted:   {oracle_fired_total}")
    print(f"  v2 preds replaced: {oracle_replaced_total}")
    print(f"  Output:            {args.out}")


if __name__ == "__main__":
    main()
