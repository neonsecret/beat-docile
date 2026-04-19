"""[RESEARCH-BURIED] PP-DocLayoutV3 / YOLOv12-DocLayNet region scoping for Sonnet.

Status: RESEARCH-BURIED — both modes bury at 50-doc scale (-0 to -9pp KILE).
See KNOWLEDGE_BASE.md §6.17 for details. Mode A (region tags in prompt) and
Mode B (noise-word filtering) both tested and negative. A third untested mode
(semantic-region crop → re-prompt on crop) is the remaining opportunity (§6.17).

Uses YOLOv12s-DocLayNet (Apache 2.0, 11 DocLayNet classes including
Page-header/Page-footer/Table). Requires models/yolov12s-doclaynet.pt.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

    from .data import PageContext, WordBox

_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "yolov12s-doclaynet.pt"

# DocLayNet regions that are noise on invoice docs (no invoice field lives here)
_NOISE_REGIONS = {"Caption", "Formula", "Picture", "List-item"}

# DocLayNet regions that are relevant for invoices
_RELEVANT_REGIONS = {
    "Text",
    "Page-header",
    "Page-footer",
    "Table",
    "Title",
    "Section-header",
    "Footnote",
}

# Per-fieldtype region priors (Mode B filtering guide — unused in simple Mode B,
# kept here for future per-fieldtype extraction experiments).
FIELD_REGIONS: dict[str, set[str]] = {
    "document_id": {"Page-header", "Text"},
    "date_issue": {"Page-header", "Text"},
    "date_due": {"Page-header", "Text"},
    "order_id": {"Page-header", "Text"},
    "customer_order_id": {"Page-header", "Text"},
    "vendor_order_id": {"Page-header", "Text"},
    "account_num": {"Page-footer", "Text", "Footnote"},
    "bank_num": {"Page-footer", "Text", "Footnote"},
    "bic": {"Page-footer", "Text", "Footnote"},
    "iban": {"Page-footer", "Text", "Footnote"},
    "payment_reference": {"Page-footer", "Text", "Footnote"},
    "payment_terms": {"Page-footer", "Text", "Footnote"},
    "vendor_tax_id": {"Page-footer", "Text", "Footnote"},
    "vendor_registration_id": {"Page-footer", "Text", "Footnote"},
    "customer_tax_id": {"Page-footer", "Text", "Footnote"},
    "customer_registration_id": {"Page-footer", "Text", "Footnote"},
    "vendor_address": {"Text", "Page-header"},
    "customer_billing_address": {"Text"},
    "customer_delivery_address": {"Text"},
    "customer_other_address": {"Text"},
    "vendor_name": {"Title", "Section-header", "Page-header", "Text"},
    "customer_billing_name": {"Title", "Text"},
    "customer_delivery_name": {"Title", "Text"},
    "customer_other_name": {"Title", "Text"},
    "amount_total_gross": {"Page-footer", "Table", "Text"},
    "amount_total_net": {"Page-footer", "Table", "Text"},
    "amount_total_tax": {"Page-footer", "Table", "Text"},
    "amount_due": {"Page-footer", "Table", "Text"},
    "amount_paid": {"Page-footer", "Table", "Text"},
    "tax_detail_gross": {"Table", "Text"},
    "tax_detail_net": {"Table", "Text"},
    "tax_detail_rate": {"Table", "Text"},
    "tax_detail_tax": {"Table", "Text"},
    "vendor_email": {"Page-footer", "Text"},
    "currency_code_amount_due": {"Page-footer", "Table", "Text"},
    # Line items: strict table scope
    "line_item_amount_gross": {"Table"},
    "line_item_amount_net": {"Table"},
    "line_item_code": {"Table"},
    "line_item_currency": {"Table"},
    "line_item_date": {"Table"},
    "line_item_description": {"Table", "Text"},
    "line_item_discount_amount": {"Table"},
    "line_item_discount_rate": {"Table"},
    "line_item_hts_number": {"Table"},
    "line_item_order_id": {"Table"},
    "line_item_person_name": {"Table"},
    "line_item_position": {"Table"},
    "line_item_quantity": {"Table"},
    "line_item_tax": {"Table"},
    "line_item_tax_rate": {"Table"},
    "line_item_unit_price_gross": {"Table"},
    "line_item_unit_price_net": {"Table"},
    "line_item_units_of_measure": {"Table"},
    "line_item_weight": {"Table"},
}


# ── Model (lazy singleton) ─────────────────────────────────────────────────────

_model = None


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO

        if not _MODEL_PATH.exists():
            raise FileNotFoundError(
                f"YOLOv12s-DocLayNet not found at {_MODEL_PATH}. "
                'Run: uv run python -c "from huggingface_hub import hf_hub_download; '
                "hf_hub_download('hantian/yolo-doclaynet', 'yolov12s-doclaynet.pt', local_dir='models')\""
            )
        _model = YOLO(str(_MODEL_PATH))
    return _model


# ── Core detection & assignment ────────────────────────────────────────────────


def detect_regions(page_image: PILImage, conf: float = 0.3) -> list[dict]:
    """Run YOLOv12s-DocLayNet on page image. Returns [{label, bbox:[x1,y1,x2,y2], conf}]."""
    model = _get_model()
    results = model.predict(page_image, verbose=False, conf=conf)
    regions = []
    for r in results:
        if r.boxes is None:
            continue
        for i in range(len(r.boxes)):
            cls_id = int(r.boxes.cls[i])
            label = r.names[cls_id]
            confidence = float(r.boxes.conf[i])
            xyxy = r.boxes.xyxy[i].tolist()  # pixel coords
            regions.append({"label": label, "bbox": xyxy, "conf": confidence})
    return regions


def assign_word_regions(
    words: list[WordBox],
    regions: list[dict],
    img_w: int,
    img_h: int,
) -> dict[int, str]:
    """Map word_id → region label by center-point containment.

    When a word center falls in multiple regions, pick the smallest (most specific).
    Fallback: "Text".
    """
    id_to_region: dict[int, str] = {}
    for w in words:
        cx = (w.bbox[0] + w.bbox[2]) / 2 * img_w
        cy = (w.bbox[1] + w.bbox[3]) / 2 * img_h
        matched = "Text"
        best_area = float("inf")
        for r in regions:
            x1, y1, x2, y2 = r["bbox"]
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                area = (x2 - x1) * (y2 - y1)
                if area < best_area:
                    best_area = area
                    matched = r["label"]
        id_to_region[w.id] = matched
    return id_to_region


def annotate_page_with_regions(page: PageContext) -> dict[int, str]:
    """Convenience: detect regions on page image and assign word regions in one call."""
    img_w, img_h = page.image.size
    regions = detect_regions(page.image)
    return assign_word_regions(page.words, regions, img_w, img_h)


# ── Mode A: region-annotated prompt ───────────────────────────────────────────


def words_to_prompt_mode_a(words: list[WordBox], word_regions: dict[int, str]) -> str:
    """Row-grouped word prompt with dominant REGION tag prefixed per row.

    Format: "[REGION:Page-header] R0(y≈0.050): 1:Invoice  2:No.  3:12345"
    """
    if not words:
        return ""

    row_gap = 0.012
    rows: list[list[WordBox]] = []
    current_row: list[WordBox] = []
    sorted_words = sorted(words, key=lambda w: (w.bbox[1], w.bbox[0]))

    for w in sorted_words:
        if not current_row or abs(w.bbox[1] - current_row[0].bbox[1]) <= row_gap:
            current_row.append(w)
        else:
            rows.append(sorted(current_row, key=lambda x: x.bbox[0]))
            current_row = [w]
    if current_row:
        rows.append(sorted(current_row, key=lambda x: x.bbox[0]))

    lines = []
    for i, row in enumerate(rows):
        row_y = row[0].bbox[1]
        region_counts = Counter(word_regions.get(w.id, "Text") for w in row)
        dominant = region_counts.most_common(1)[0][0]
        tokens = "  ".join(f"{w.id}:{w.text}" for w in row)
        lines.append(f"[REGION:{dominant}] R{i}(y≈{row_y:.3f}): {tokens}")
    return "\n".join(lines)


# ── Mode B: noise-filtered word list ──────────────────────────────────────────


def filter_words_mode_b(words: list[WordBox], word_regions: dict[int, str]) -> list[WordBox]:
    """Remove words from noise regions (Caption, Formula, Picture, List-item).

    Keeps Text, Page-header, Page-footer, Table, Title, Section-header, Footnote.
    Words with no detected region (default "Text") are kept.
    """
    return [w for w in words if word_regions.get(w.id, "Text") not in _NOISE_REGIONS]


# ── Async extract wrappers (do NOT modify extract.py) ─────────────────────────


async def extract_page_mode_a(
    page: PageContext,
    model: str,
    word_regions: dict[int, str],
    few_shot_messages: list[dict] | None = None,
) -> tuple[list, list]:
    """extract_page variant with REGION-tagged prompt (Mode A). Read-only wrapper."""
    from .extract import _SYSTEM, _image_to_b64, _parse_response
    from .vertex import complete

    if not page.words:
        return [], []

    img_b64 = _image_to_b64(page.image)
    words_layout = words_to_prompt_mode_a(page.words, word_regions)
    user_content = [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": img_b64},
        },
        {
            "type": "text",
            "text": (
                f"[Document words]\n{words_layout}\n\n"
                "[Task]\n"
                "Extract all fields from this invoice page. "
                "Use the word ids shown above. The [REGION:X] tag before each row "
                "indicates the layout region (Page-header, Page-footer, Table, Text, etc.). "
                "Return JSON only."
            ),
        },
    ]

    messages = []
    if few_shot_messages:
        messages.extend(few_shot_messages)
    messages.append({"role": "user", "content": user_content})

    msg = await complete(
        model=model,
        system=_SYSTEM,
        messages=messages,
        max_tokens=4096,
        cache_system=True,
    )
    raw = msg.content[0].text if msg.content else ""
    return _parse_response(raw, page.words, page.page_index)


async def extract_page_mode_b(
    page: PageContext,
    model: str,
    word_regions: dict[int, str],
    few_shot_messages: list[dict] | None = None,
) -> tuple[list, list]:
    """extract_page variant with noise words filtered out (Mode B). Read-only wrapper."""
    from .data import PageContext as PC  # noqa: N817
    from .extract import extract_page

    filtered_words = filter_words_mode_b(page.words, word_regions)
    filtered_page = PC(
        docid=page.docid,
        page_index=page.page_index,
        image=page.image,
        words=filtered_words,
    )
    return await extract_page(filtered_page, model, few_shot_messages=few_shot_messages)
