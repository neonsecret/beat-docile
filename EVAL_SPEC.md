# DocILE Evaluation Specification

Authoritative reference for the DocILE benchmark. Citations into cloned repo at `references/docile/` or PDFs in `references/papers/`.

---

## 1. Field Catalog

### KILE Field Types (36)
`docile/dataset/__init__.py:8-45`

| Field type | Description |
|---|---|
| `account_num` | Bank account number |
| `amount_due` | Amount due for payment |
| `amount_paid` | Amount already paid |
| `amount_total_gross` | Total gross (with tax) |
| `amount_total_net` | Total net (before tax) |
| `amount_total_tax` | Total tax amount |
| `bank_num` | Bank routing/sort code |
| `bic` | BIC/SWIFT code |
| `currency_code_amount_due` | Currency symbol/code for amount due |
| `customer_billing_address` | Customer billing address block |
| `customer_billing_name` | Customer billing name |
| `customer_delivery_address` | Customer delivery address block |
| `customer_delivery_name` | Customer delivery name |
| `customer_id` | Customer identifier |
| `customer_order_id` | Order ID issued by the customer |
| `customer_other_address` | Other customer address |
| `customer_other_name` | Other customer name |
| `customer_registration_id` | Customer company registration ID |
| `customer_tax_id` | Customer VAT/tax ID |
| `date_due` | Payment due date |
| `date_issue` | Invoice issue date |
| `document_id` | Document/invoice number |
| `iban` | IBAN |
| `order_id` | Order identifier |
| `payment_reference` | Payment reference string |
| `payment_terms` | Payment terms text |
| `tax_detail_gross` | Per-tax-rate gross |
| `tax_detail_net` | Per-tax-rate net |
| `tax_detail_rate` | Tax rate percentage |
| `tax_detail_tax` | Per-tax-rate tax amount |
| `vendor_address` | Vendor address block |
| `vendor_email` | Vendor email |
| `vendor_name` | Vendor name |
| `vendor_order_id` | Order ID issued by the vendor |
| `vendor_registration_id` | Vendor company registration ID |
| `vendor_tax_id` | Vendor VAT/tax ID |

### LIR Field Types (19)
`docile/dataset/__init__.py:47-67`

| Field type | Description |
|---|---|
| `line_item_amount_gross` | Line item gross amount |
| `line_item_amount_net` | Line item net amount |
| `line_item_code` | Product/SKU code |
| `line_item_currency` | Currency for this line item |
| `line_item_date` | Date on line item |
| `line_item_description` | Product/service description |
| `line_item_discount_amount` | Discount amount |
| `line_item_discount_rate` | Discount rate |
| `line_item_hts_number` | Harmonized tariff schedule number |
| `line_item_order_id` | Order ID for this line item |
| `line_item_person_name` | Person name on line item |
| `line_item_position` | Row/position number |
| `line_item_quantity` | Quantity |
| `line_item_tax` | Tax amount |
| `line_item_tax_rate` | Tax rate |
| `line_item_unit_price_gross` | Unit price gross |
| `line_item_unit_price_net` | Unit price net |
| `line_item_units_of_measure` | Units of measure |
| `line_item_weight` | Weight |

---

## 2. Prediction JSON Format

`docile/dataset/field.py:10-47`, `README.md:104-131`

Top-level: `{docid: [field, ...], ...}`. Minimal example:

```json
{"516f2d61ea404b30a9192a72": [
  {"bbox":[0.133,0.185,0.344,0.226],"page":0,"fieldtype":"customer_billing_address","score":0.95},
  {"bbox":[0.177,0.388,0.224,0.401],"page":0,"fieldtype":"line_item_code","line_item_id":1,"score":0.87}
]}
```

| Key | Required | Notes |
|---|---|---|
| `bbox` | Yes | `[left,top,right,bottom]` relative [0,1] |
| `page` | Yes | 0-based int |
| `fieldtype` | Yes | Exact string from §1 |
| `score` | No | All-or-none per task |
| `text` | No | Secondary metric only |
| `line_item_id` | LIR only | Required LIR, forbidden KILE |
| `use_only_for_ap` | No | Default false; excludes from F1/TP/FP/FN |

`bbox` order: `BBox(*(dct_copy.pop("bbox")))` — `field.py:38`.

---

## 3. The AP Metric — How Matching Actually Works

### Matching rule: Pseudo-Character Centers (PCC), not IoU

A prediction matches a gold annotation iff: same `fieldtype`, same `page`, PCC-IoU >= 1.0 (default).

`docile/evaluation/pcc_field_matching.py:147-158`:
```python
def pccs_iou(pcc_set, gold_bbox, pred_bbox, page):
    golds = pcc_set.get_covered_pccs(gold_bbox, page)
    preds = pcc_set.get_covered_pccs(pred_bbox, page)
    if len(golds) == len(preds) == 0:
        return 1
    return len(golds.intersection(preds)) / len(golds.union(preds))
```

PCCs: centers of equal-width character slots within each snapped OCR word box (`docile/evaluation/pcc.py:88-97`). With threshold 1.0, pred and gold bboxes must cover **exactly the same PCC set** — no extra or missing character centers.

### True positives, false positives, duplicates

Greedy by descending `score`; ties: original list order, then hashed docid+index (`docile/evaluation/pcc_field_matching.py:193-204`). Each gold matches at most once; duplicate hits are false positives. `use_only_for_ap=True` predictions sort after all `use_only_for_ap=False` ones regardless of numeric score (`docile/dataset/field.py:55-64`).

### Confidence scores

All-or-none per task (`docile/evaluation/evaluate.py:436-445`); mixing raises `PredictionsValidationError`. Without scores, list order determines the PR curve and therefore AP.

### Per-field-type vs overall

Primary: micro-average AP (KILE) or F1 (LIR). `evaluate.py:24`: `TASK_TO_PRIMARY_METRIC_NAME = {"kile": "AP", "lir": "f1"}`. Per-fieldtype via `--evaluate-fieldtypes` or `get_metrics(task, fieldtype=ft)` (`evaluate.py:200-212`).

---

## 4. How to Run the Official Evaluator

Entry point `docile_evaluate` (`pyproject.toml:76`):

```bash
docile_evaluate \
  --task KILE \
  --dataset-path /data/docile/ \
  --split val \
  --predictions /tmp/kile_predictions.json \
  --evaluate-fieldtypes \
  --store-evaluation-result /tmp/kile_eval.json
```

Replace `--task KILE` with `--task LIR` for LIR. Source: `docile/cli/evaluate.py:60-178`.

Required: `--task`, `--dataset-path`, `--split`, `--predictions`. Output is a GitHub-markdown table with AP, F1, precision, recall, TP, FP, FN per fieldtype and x-shot subsets. `--primary-metric-only` prints a single float.

---

## 5. Gotchas That Silently Score 0

- **Pixel coords instead of relative**: bbox values outside [0,1] raise `PredictionsValidationError` (`evaluate.py:404-411`). Pixel values that happen to fall in [0,1] (small images) silently mismatch PCCs → AP=0.
- **Misspelled fieldtype**: no error; unknown types match zero gold fields — all predictions for that type are false positives (`pcc_field_matching.py:186-204`).
- **Wrong LIR grouping**: correct fields with wrong `line_item_id` grouping silently reduce F1. Missing `line_item_id` in LIR raises error (`evaluate.py:425-433`).
- **`line_item_id` present for KILE**: raises `PredictionsValidationError` (`evaluate.py:415-423`).
- **Mixed score presence**: some scored, some not → `PredictionsValidationError` (`evaluate.py:436-445`).
- **Missing or extra docids**: all split docids must appear (even `[]`); extra docids rejected (`evaluate.py:447-459`).
- **Over 1000 predictions per page**: `PredictionsValidationError` (`evaluate.py:385-394`).
- **Bbox covers no PCCs**: too small or misregistered → PCC-IoU=0, always a false positive. Failure mode when using PDF text coords instead of DocTR OCR coords.
- **Unsnapped OCR bboxes for predictions**: PCCs come from *snapped* words only (`pcc.py:56-69`). Raw bboxes cause borderline PCC mismatches.

---

## 6. LIR-Specific Rules

`docile/evaluation/line_item_matching.py:90-184`

**Stage 1 — Line item alignment**: bipartite graph between predicted and gold `line_item_id` groups. Edge weight = PCC-IoU field matches for that pair (excluding `use_only_for_ap=True`). NetworkX maximum-weight matching picks pred→gold correspondence (`line_item_matching.py:133-152`).

**Stage 2 — Field matching**: standard PCC-IoU matching within each matched (pred LI, gold LI) pair (`line_item_matching.py:141-146`).

- Fields in an unmatched predicted LI are all false positives.
- One predicted LI maps to at most one gold LI.
- `line_item_id` integer values are arbitrary; only grouping matters.

AMBIGUOUS: Paper (docile-benchmark.pdf p.10) does not clarify whether LIR AP is a primary or secondary metric. Code confirms F1 is primary (`evaluate.py:24`); AP is reported but not used for leaderboard ranking (`docile-overview-icdar2023.pdf §3`).

---

## 7. OCR Word Grid Access

DocTR OCR stored at `ocr/<docid>.json`. `docile/dataset/document_ocr.py:46-105`.

```python
with document:  # memory caching
    words = document.ocr.get_all_words(page=0, snapped=True)
    # List[Field]: .text, .bbox (relative [0,1]), .page
```

Raw JSON path: `ocr["pages"][p]["blocks"][b]["lines"][l]["words"][w]`
Each word: `{"value": str, "confidence": float, "geometry": [[l,t],[r,b]], "snapped_geometry": [[l,t],[r,b]]}` — `snapped_geometry` written on first snapped access (`document_ocr.py:123-145`).

**Snapping**: Otsu binarization + two-phase margin shrinking on the page image crop. Source: `_snap_bbox_to_text` at `document_ocr.py:161`.

**Critical**: PCCs derive from snapped bboxes only (`pcc.py:56-69`). Predictions must align to the snapped OCR grid — not raw DocTR boxes, not PDF text layer coords.
