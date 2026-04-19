"""[RESEARCH-BURIED] GLM-OCR + PP-DocLayoutV3 two-stage extractor for DocILE KILE + LIR.

Status: RESEARCH-BURIED — 4.23% KILE on 21-doc partial run (ceiling ~4-6%). Root cause:
GLM-OCR's OmniDocBench SOTA is text-recognition accuracy, not schema-conformant KIE;
"Information Extraction" is an emergent capability, not a primary training objective.
See KNOWLEDGE_BASE.md §6.1 for full root-cause analysis.

Model:   zai-org/GLM-OCR (MIT license, code Apache 2.0)
VRAM:    ~4-6 GB FP16 (fits 8 GB RTX 3070; 2.65 GB on disk)
HF URL:  https://huggingface.co/zai-org/GLM-OCR

Install:
    uv add 'glmocr[selfhosted]'
    PP-DocLayoutV3 inside glmocr is PaddlePaddle-based. On neon (WSL2 + CUDA):
        pip install paddlepaddle-gpu
    If PaddlePaddle is unavailable, try the ONNX variant:
        uv add 'glmocr[onnx]'
    WARNING: glmocr[selfhosted] also loads GLM-OCR VLM internally for OCR.
    Running extract_page() additionally loads GLM-OCR for the KIE pass, which
    may push VRAM usage to ~8-10 GB. Monitor with: nvidia-smi -lms 1000

Flow per page:
  Stage 1 — Layout: glmocr.parse(tmp_png) → PP-DocLayoutV3 regions.
    Each region: {label, content, bbox_2d: [x1,y1,x2,y2] in 0-1000}.
    Table regions are used to scope LIR extraction.
  Stage 2 — KIE: GLM-OCR VLM with {fieldtype: ""} schema → {fieldtype: text}.
  Stage 3 — Bbox: match each KIE value to best region via rapidfuzz.partial_ratio,
    then assign bbox per BD_GLM_BBOX_MODE (see below).

Bbox modes (BD_GLM_BBOX_MODE env var):
  words  (default): scope DocTR OCR words to matched region bbox, align value
                    text within those words via find_span → union of snapped word
                    bboxes. PCC-IoU=1.0 safe.
  region:           use matched region bbox_2d / 1000 directly. Faster,
                    loosely bounded. May reduce KILE AP for short single-word
                    fields (e.g., document_id, amount_due).
  Fallback: if no region matches (fuzz score < threshold) or region bbox is None,
  falls back to global find_span across all page words.

LIR:
  PP-DocLayoutV3 "table" label regions are extracted; a second KIE pass with the
  {lir_field: []} schema runs on each cropped table region image. Fields are
  aligned to OCR words within the table region bbox.

Score: uniform 1.0 per field (no per-field confidence from GLM-OCR or glmocr.parse).

Run:
    uv run python tools/run_glm_ocr_50.py [--spike-only] [--no-spike]
    uv run python tools/run_glm_ocr_500.py
"""

from __future__ import annotations

import json
import os
import re
import tempfile

from docile.dataset import BBox, Field

from .align import find_span
from .data import PageContext, WordBox, iter_pages

MODEL_ID = "zai-org/GLM-OCR"

_SNAP_MIN_RATIO = 0.65
_FUZZ_THRESHOLD = 60  # rapidfuzz partial_ratio threshold for value→region match

_KILE_FIELDS: list[str] = [
    "account_num",
    "amount_due",
    "amount_paid",
    "amount_total_gross",
    "amount_total_net",
    "amount_total_tax",
    "bank_num",
    "bic",
    "currency_code_amount_due",
    "customer_billing_address",
    "customer_billing_name",
    "customer_delivery_address",
    "customer_delivery_name",
    "customer_id",
    "customer_order_id",
    "customer_other_address",
    "customer_other_name",
    "customer_registration_id",
    "customer_tax_id",
    "date_due",
    "date_issue",
    "document_id",
    "iban",
    "order_id",
    "payment_reference",
    "payment_terms",
    "tax_detail_gross",
    "tax_detail_net",
    "tax_detail_rate",
    "tax_detail_tax",
    "vendor_address",
    "vendor_email",
    "vendor_name",
    "vendor_order_id",
    "vendor_registration_id",
    "vendor_tax_id",
]

_LIR_FIELDS: list[str] = [
    "line_item_amount_gross",
    "line_item_amount_net",
    "line_item_code",
    "line_item_currency",
    "line_item_date",
    "line_item_description",
    "line_item_discount_amount",
    "line_item_discount_rate",
    "line_item_hts_number",
    "line_item_order_id",
    "line_item_person_name",
    "line_item_position",
    "line_item_quantity",
    "line_item_tax",
    "line_item_tax_rate",
    "line_item_unit_price_gross",
    "line_item_unit_price_net",
    "line_item_units_of_measure",
    "line_item_weight",
]

# Multi-occurrence fields (one per tax-rate row)
_MULTI_KILE: frozenset[str] = frozenset(
    {
        "tax_detail_gross",
        "tax_detail_net",
        "tax_detail_rate",
        "tax_detail_tax",
    }
)

_KILE_SCHEMA: dict = {ft: [] if ft in _MULTI_KILE else "" for ft in _KILE_FIELDS}
_LIR_SCHEMA: dict = {"line_items": [{ft: "" for ft in _LIR_FIELDS}]}

# Batch size for KIE passes — model skips middle fields with large schemas
_KIE_BATCH_SIZE = 9


# ── vLLM client (shared by layout + KIE) ──────────────────────────────────────

_VLLM_HOST = os.environ.get("GLMOCR_VLLM_HOST", "localhost")
_VLLM_PORT = int(os.environ.get("GLMOCR_VLLM_PORT", "8000"))


def _image_to_data_url(image) -> str:
    """Convert PIL image to base64 data URL for vLLM API."""
    import base64
    import io

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


# ── Stage 1: Layout detection ─────────────────────────────────────────────────


def _parse_layout(image, model_id: str = MODEL_ID) -> list[dict]:
    """Run glmocr layout detection via local vLLM server.

    Returns list of region dicts: {label, content, bbox_2d: [x1,y1,x2,y2] (0-1000)}.
    Falls back to [] if glmocr or vLLM is unavailable.
    """
    try:
        import glmocr  # type: ignore[import]
    except ImportError:
        return []

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        image.save(f, format="PNG")
        tmp_path = f.name
    try:
        with glmocr.GlmOcr(
            mode="selfhosted",
            ocr_api_host=_VLLM_HOST,
            ocr_api_port=_VLLM_PORT,
            model=model_id,
        ) as parser:
            result = parser.parse(tmp_path)
            return result.json_result[0] if result.json_result else []
    except Exception:
        return []
    finally:
        os.unlink(tmp_path)


# ── Stage 2: KIE via vLLM OpenAI API ─────────────────────────────────────────


def _load_model(model_id: str = MODEL_ID) -> tuple:
    """Return (model, processor, device) for the given model_id.

    In vLLM mode (default) inference runs via HTTP, so local weights are not loaded.
    Returns (None, None, "cpu") as a no-op; callers pass these through to _run_kie.
    """
    return None, None, "cpu"


def _run_kie(
    image,
    schema: dict,
    model,
    processor,
    device: str,
    max_new_tokens: int = 512,
) -> dict:
    """Run one KIE pass by calling the vLLM OpenAI API.

    model/processor/device are accepted for API compatibility but ignored in vLLM mode.
    Returns parsed JSON dict, or {} on failure.
    """
    from openai import OpenAI

    client = OpenAI(
        base_url=f"http://{_VLLM_HOST}:{_VLLM_PORT}/v1",
        api_key="dummy",
    )
    data_url = _image_to_data_url(image)
    prompt_text = f"Information Extraction:\n{json.dumps(schema)}"

    try:
        resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ],
            max_tokens=max_new_tokens,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content or ""
    except Exception:
        return {}

    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pairs = re.findall(r'"([^"]+)":\s*"([^"]*)"', raw)
        if pairs:
            return {k: v for k, v in pairs if v}
        return {}


def _run_kie_batched(
    image,
    fields: list[str],
    model,
    processor,
    device: str,
    batch_size: int = _KIE_BATCH_SIZE,
) -> dict:
    """Run KIE in batches to avoid model skipping middle fields in large schemas."""
    result: dict = {}
    for i in range(0, len(fields), batch_size):
        batch = fields[i : i + batch_size]
        schema = {ft: [] if ft in _MULTI_KILE else "" for ft in batch}
        batch_result = _run_kie(image, schema, model, processor, device)
        result.update(batch_result)
    return result


# ── Stage 3: Bbox resolution ──────────────────────────────────────────────────


def _word_in_bbox(word: WordBox, bbox: BBox) -> bool:
    """True if word center falls inside bbox."""
    cx = (word.bbox[0] + word.bbox[2]) / 2
    cy = (word.bbox[1] + word.bbox[3]) / 2
    return bbox.left <= cx <= bbox.right and bbox.top <= cy <= bbox.bottom


def _region_bbox(region: dict) -> BBox | None:
    """Convert region bbox_2d [0-1000] to relative [0-1] BBox."""
    bbox_2d = region.get("bbox_2d")
    if not bbox_2d or len(bbox_2d) != 4:
        return None
    x1, y1, x2, y2 = bbox_2d
    return BBox(x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0)


def _span_to_bbox(span: tuple[int, int], words: list[WordBox]) -> BBox:
    start, end = span
    sw = words[start : end + 1]
    return BBox(
        min(w.bbox[0] for w in sw),
        min(w.bbox[1] for w in sw),
        max(w.bbox[2] for w in sw),
        max(w.bbox[3] for w in sw),
    )


def _fuzz_match_region(
    value: str, regions: list[dict], threshold: int = _FUZZ_THRESHOLD
) -> dict | None:
    """Return region whose content best fuzzy-matches value, or None."""
    try:
        from rapidfuzz import fuzz  # type: ignore[import]
    except ImportError:
        return None

    best_score = 0
    best: dict | None = None
    value_lower = value.lower()
    for region in regions:
        content = str(region.get("content") or "")
        score = fuzz.partial_ratio(value_lower, content.lower())
        if score > best_score:
            best_score = score
            best = region
    return best if best_score >= threshold else None


def _resolve_field_bbox(
    value: str,
    matched_region: dict | None,
    words: list[WordBox],
    mode: str,
) -> BBox | None:
    """Map a KIE text value to a DocILE-compatible snapped bbox.

    mode='region': use matched region bbox_2d / 1000 directly (fast, loose).
    mode='words':  scope OCR words to region, align value text via find_span,
                   return union of matched snapped word bboxes (PCC-safe).
    Fallback when no region matched: global find_span over all page words.
    """
    rbbox = _region_bbox(matched_region) if matched_region is not None else None

    if mode == "region":
        if rbbox is not None:
            return rbbox
        # No region → global text alignment fallback
        span = find_span(value, words, min_ratio=_SNAP_MIN_RATIO)
        return _span_to_bbox(span, words) if span else None

    # mode == "words"
    if rbbox is not None:
        scoped = [w for w in words if _word_in_bbox(w, rbbox)]
        if scoped:
            span = find_span(value, scoped, min_ratio=_SNAP_MIN_RATIO)
            if span is not None:
                return _span_to_bbox(span, scoped)
        # No word match inside region → fall back to region bbox itself
        return rbbox

    # No region at all → global text alignment
    span = find_span(value, words, min_ratio=_SNAP_MIN_RATIO)
    return _span_to_bbox(span, words) if span else None


# ── KILE + LIR parsing ────────────────────────────────────────────────────────


def _parse_kile(
    kile_dict: dict,
    regions: list[dict],
    words: list[WordBox],
    page_idx: int,
    mode: str,
) -> list[Field]:
    """Build KILE Field objects from KIE output, using regions for bbox."""
    fields: list[Field] = []
    for ft in _KILE_FIELDS:
        val = kile_dict.get(ft)
        if not val:
            continue
        texts = val if isinstance(val, list) else [val]
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                continue
            matched_region = _fuzz_match_region(text, regions)
            bbox = _resolve_field_bbox(text, matched_region, words, mode)
            if bbox is None:
                continue
            fields.append(Field(bbox=bbox, page=page_idx, fieldtype=ft, score=1.0))
    return fields


def _parse_lir_from_tables(
    image,
    table_regions: list[dict],
    words: list[WordBox],
    page_idx: int,
    model=None,
    processor=None,
    device: str = "cpu",
) -> list[Field]:
    """Run LIR KIE on each PP-DocLayoutV3 table region; align to scoped OCR words."""
    fields: list[Field] = []
    li_id_offset = 1

    for region in table_regions:
        rbbox = _region_bbox(region)
        if rbbox is None:
            continue

        w_px, h_px = image.size
        crop_box = (
            int(rbbox.left * w_px),
            int(rbbox.top * h_px),
            int(rbbox.right * w_px),
            int(rbbox.bottom * h_px),
        )
        table_img = image.crop(crop_box)

        lir_raw = _run_kie(table_img, _LIR_SCHEMA, model, processor, device)
        line_items = lir_raw.get("line_items", [])
        if not isinstance(line_items, list):
            continue

        scoped_words = [w for w in words if _word_in_bbox(w, rbbox)]

        for li_idx, item in enumerate(line_items):
            if not isinstance(item, dict):
                continue
            li_id = li_id_offset + li_idx
            for ft in _LIR_FIELDS:
                text = item.get(ft, "")
                if not isinstance(text, str) or not text.strip():
                    continue
                span = find_span(text, scoped_words, min_ratio=_SNAP_MIN_RATIO)
                if span is None:
                    continue
                fields.append(
                    Field(
                        bbox=_span_to_bbox(span, scoped_words),
                        page=page_idx,
                        fieldtype=ft,
                        score=1.0,
                        line_item_id=li_id,
                    )
                )

        li_id_offset += len(line_items)

    return fields


# ── Page + document-level API ─────────────────────────────────────────────────


def extract_page(
    page: PageContext,
    model_id: str = MODEL_ID,
) -> tuple[list[Field], list[Field]]:
    """Extract KILE + LIR fields from one page.

    Stage 1: glmocr.parse → PP-DocLayoutV3 layout regions with bboxes.
    Stage 2: GLM-OCR VLM KIE → {fieldtype: value} for KILE schema.
    Stage 3: Match values to regions (rapidfuzz) → assign snapped bbox.
    LIR: second KIE pass per table region, word-aligned within region scope.

    Returns (kile_fields, lir_fields).
    """
    if not page.words:
        return [], []

    model, processor, device = _load_model(model_id)
    mode = os.environ.get("BD_GLM_BBOX_MODE", "words")

    regions = _parse_layout(page.image)
    if not regions:
        import sys

        print(
            "[glm_ocr] WARN: no layout regions — check vLLM server "
            f"(GLMOCR_VLLM_HOST={_VLLM_HOST} PORT={_VLLM_PORT}). "
            "KIE will still run; bbox falls back to global text-align.",
            file=sys.stderr,
        )
    table_regions = [r for r in regions if r.get("label") == "table"]

    kile_raw = _run_kie_batched(page.image, _KILE_FIELDS, model, processor, device)
    kile_fields = _parse_kile(kile_raw, regions, page.words, page.page_index, mode)

    lir_fields = _parse_lir_from_tables(
        page.image,
        table_regions,
        page.words,
        page.page_index,
        model,
        processor,
        device,
    )

    return kile_fields, lir_fields


def extract_documents(
    docids: list[str],
    dataset,
    dpi: int = 200,
    model_id: str = MODEL_ID,
) -> dict[str, list[dict]]:
    """Extract KILE + LIR for the given docids.

    Returns {docid: [field_dict, ...]} for every docid in input (empty list if
    no fields found). field_dict matches the DocILE prediction JSON format.

    Args:
        docids:   docids to process (must all exist in dataset).
        dataset:  DocILE Dataset object (already loaded).
        dpi:      ignored — iter_pages uses 150 DPI from data.py; kept for API parity.
        model_id: HF model identifier (default: zai-org/GLM-OCR).
    """
    predictions: dict[str, list[dict]] = {did: [] for did in docids}
    docid_set = set(docids)

    for doc in dataset:
        if doc.docid not in docid_set:
            continue
        doc_fields: list[Field] = []
        for page in iter_pages(doc):
            kile, lir = extract_page(page, model_id)
            doc_fields.extend(kile)
            doc_fields.extend(lir)
        predictions[doc.docid] = [f.to_dict() for f in doc_fields]

    return predictions
