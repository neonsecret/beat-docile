"""[EXPERIMENTAL] DSPy 3.1.3 MIPROv2 prompt optimizer for DocILE extraction.

Status: EXPERIMENTAL — previous run used Haiku as evaluator; Haiku-optimized
instructions hurt Sonnet's stronger baseline (-0.77pp KILE on 50-doc).
See KNOWLEDGE_BASE.md §6.9. With Sonnet as evaluator (§8.6), estimated +1-4pp.
Cost ~$30 per optimization run.

Critical: litellm.drop_params = True must be set before DSPy configures LMs —
Vertex AI rejects DSPy's extra params (notably 'n') without it.
"""

from __future__ import annotations

import base64
import io
import json
import re
from typing import Any

import dspy
import litellm

litellm.drop_params = True

from .config import DEFAULT_MODEL, VERTEX_LOCATION, VERTEX_PROJECT_ID  # noqa: E402

# ── Field catalog (static reference — not optimized by MIPROv2) ──────────────
FIELD_CATALOG = """KILE fields (36 types, at most 1 per page unless noted):
account_num | amount_due | amount_paid | amount_total_gross | amount_total_net | amount_total_tax
bank_num | bic | currency_code_amount_due | document_id | iban | order_id | payment_reference | payment_terms
customer_billing_address | customer_billing_name | customer_delivery_address | customer_delivery_name
customer_id | customer_order_id | customer_other_address | customer_other_name
customer_registration_id | customer_tax_id | date_due | date_issue
tax_detail_gross | tax_detail_net | tax_detail_rate | tax_detail_tax (multiple allowed — one per tax rate row)
vendor_address | vendor_email | vendor_name | vendor_order_id | vendor_registration_id | vendor_tax_id

LIR fields (19 types, one set per line item row; same line_item_id = same row):
line_item_amount_gross | line_item_amount_net | line_item_code | line_item_currency | line_item_date
line_item_description | line_item_discount_amount | line_item_discount_rate | line_item_hts_number
line_item_order_id | line_item_person_name | line_item_position | line_item_quantity | line_item_tax
line_item_tax_rate | line_item_unit_price_gross | line_item_unit_price_net | line_item_units_of_measure | line_item_weight"""

_OUTPUT_FORMAT = (
    '{"fields":[{"fieldtype":"...","word_ids":[id,...],"text":"...","score":0.9}],'
    '"line_items":[{"line_item_id":1,"fields":[...]}]}'
)


# ── DSPy Signature ────────────────────────────────────────────────────────────
class DocILEExtractionSig(dspy.Signature):
    """Extract all KILE and LIR fields from an invoice document page.

    Use the OCR word list to identify field values. For each field:
    - Select word_ids that are exactly the field value, not surrounding labels or colons
    - For address blocks: include ALL words across all address lines
    - For tax_detail_* fields: output one set per tax rate row
    - Assign score 0.0-1.0 for your extraction confidence
    Group line item fields by visual row using line_item_id (same id = same row).
    Mutual exclusion: address fields and amount fields must have disjoint word_id sets.
    Return valid JSON only — no markdown fences, no explanation."""

    field_catalog: str = dspy.InputField(desc="Complete catalog of valid KILE and LIR field types")
    words_layout: str = dspy.InputField(
        desc="OCR words grouped by visual row: 'R{i}(y≈{top}): {id}:{text} ...'"
    )
    page_image: dspy.Image = dspy.InputField(desc="Invoice page image for visual layout context")
    fields_json: str = dspy.OutputField(desc=f"Extraction result JSON. Format: {_OUTPUT_FORMAT}")


# ── DSPy Module ───────────────────────────────────────────────────────────────
class DocILEExtractionModule(dspy.Module):
    """DSPy module for DocILE KILE+LIR extraction. Wraps a single Predict call."""

    def __init__(self):
        self.extract = dspy.Predict(DocILEExtractionSig)

    def forward(self, words_layout: str, image_b64: str) -> dspy.Prediction:
        image = dspy.Image(url=f"data:image/png;base64,{image_b64}")
        return self.extract(
            field_catalog=FIELD_CATALOG,
            words_layout=words_layout,
            page_image=image,
        )


# ── Metric function ───────────────────────────────────────────────────────────
def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def kile_metric(example: dspy.Example, pred: dspy.Prediction, trace: Any = None) -> bool | float:
    """Fast KILE F1 proxy: fieldtype + normalized text exact match.

    Returns bool when trace is not None (bootstrapping), float otherwise.
    Threshold for bootstrapping accept: F1 > 0.4.
    """
    try:
        raw = getattr(pred, "fields_json", "") or ""
        # Strip DSPy adapter markup if present
        raw = re.sub(r"\[\[.*?\]\]", "", raw).strip()
        data = json.loads(raw)
    except (json.JSONDecodeError, AttributeError, TypeError):
        return False if trace is not None else 0.0

    pred_pairs: set[tuple[str, str]] = set()
    for field in data.get("fields", []):
        ft = str(field.get("fieldtype", "")).strip()
        text = _normalize(field.get("text", ""))
        if ft and text:
            pred_pairs.add((ft, text))

    gold_pairs: set[tuple[str, str]] = set()
    for gf in example.gold_kile or []:
        if gf.text:
            gold_pairs.add((gf.fieldtype, _normalize(gf.text)))

    if not gold_pairs and not pred_pairs:
        return True if trace is not None else 1.0
    if not gold_pairs or not pred_pairs:
        return False if trace is not None else 0.0

    tp = len(pred_pairs & gold_pairs)
    precision = tp / len(pred_pairs)
    recall = tp / len(gold_pairs)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    if trace is not None:
        return f1 > 0.4
    return f1


# ── Data loading ──────────────────────────────────────────────────────────────
def _image_to_b64(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def build_dspy_examples(docs: list, max_docs: int = 300) -> list[dspy.Example]:
    """Build dspy.Examples from DocILE Document objects with gold annotations.

    Uses first page only (page 0) — most invoice KILE fields appear there.
    Each example stores: words_layout, image_b64 (inputs) + gold_kile (for metric).
    """
    from .data import iter_pages
    from .extract import _words_to_prompt

    examples: list[dspy.Example] = []
    docs_subset = list(docs)[:max_docs]
    print(f"Building {len(docs_subset)} DSPy examples...")

    for i, doc in enumerate(docs_subset):
        try:
            with doc:
                pages = list(iter_pages(doc))
                if not pages:
                    continue
                page = pages[0]

                gold_kile = list(doc.annotation.fields)
                gold_lir = list(doc.annotation.li_fields)

                ex = dspy.Example(
                    words_layout=_words_to_prompt(page.words),
                    image_b64=_image_to_b64(page.image),
                    gold_kile=gold_kile,
                    gold_lir=gold_lir,
                    docid=doc.docid,
                ).with_inputs("words_layout", "image_b64")

                examples.append(ex)
        except Exception as e:
            print(f"  Skipping {getattr(doc, 'docid', '?')}: {e}")

        if (i + 1) % 50 == 0:
            print(f"  Loaded {i + 1}/{len(docs_subset)}")

    print(f"Built {len(examples)} examples")
    return examples


# ── LM configuration ──────────────────────────────────────────────────────────
def configure_dspy_lm(model: str | None = None, max_tokens: int = 4096) -> dspy.LM:
    """Configure DSPy with Vertex AI LM via LiteLLM. Sets as global default.

    Uses VERTEXAI_PROJECT / VERTEXAI_LOCATION env vars (not project/location —
    LiteLLM silently ignores those unprefixed variants for Vertex AI).
    """
    import os

    os.environ.setdefault("VERTEXAI_PROJECT", VERTEX_PROJECT_ID)
    os.environ.setdefault("VERTEXAI_LOCATION", VERTEX_LOCATION)

    model_name = model or DEFAULT_MODEL
    litellm_model = f"vertex_ai/{model_name}"

    lm = dspy.LM(litellm_model, max_tokens=max_tokens, temperature=1.0)
    dspy.configure(lm=lm)
    print(f"DSPy LM: {litellm_model} (project={VERTEX_PROJECT_ID}, location={VERTEX_LOCATION})")
    return lm
