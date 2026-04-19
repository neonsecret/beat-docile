"""[RESEARCH-BURIED] RapidTable-based LIR extraction for DocILE.

Status: RESEARCH-BURIED — -38pp LIR (full-page), -21pp LIR (cropped region).
See KNOWLEDGE_BASE.md §6.16 for details. RapidTable's TSR doesn't generalize
to DocILE invoice diversity (merged cells, multi-row items, non-standard headers).
Newer 2026-era TSR models on cropped regions could improve this (see §8.10).

Pipeline: RapidTable → HTML parse → Sonnet column classification → spatial join
→ Field objects with line_item_id.
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from typing import Any

import numpy as np
from docile.dataset import BBox, Field

from .data import PageContext, WordBox
from .extract import _LIR_TYPES
from .vertex import complete

_ENGINE = None  # lazy singleton


def _get_engine():
    global _ENGINE
    if _ENGINE is None:
        from rapid_table import ModelType, RapidTable, RapidTableInput

        # PPSTRUCTURE_EN has better detection on full-page invoice images
        _ENGINE = RapidTable(RapidTableInput(model_type=ModelType.PPSTRUCTURE_EN))
    return _ENGINE


def _words_to_ocr_input(
    words: list[WordBox], img_w: int, img_h: int
) -> tuple[np.ndarray, tuple[str, ...], tuple[float, ...]]:
    """Convert DocTR words to RapidTable OCR result format.

    Returns (boxes, texts, scores) where boxes is (N, 4, 2) polygon in pixel coords.
    """
    if not words:
        # Return empty arrays with correct shape
        return np.zeros((0, 4, 2), dtype=np.float32), (), ()

    boxes = []
    for w in words:
        left, t, r, b = w.bbox
        x1, y1, x2, y2 = left * img_w, t * img_h, r * img_w, b * img_h
        # 4-point polygon: TL, TR, BR, BL
        boxes.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

    return (
        np.array(boxes, dtype=np.float32),
        tuple(w.text for w in words),
        tuple(1.0 for _ in words),
    )


# ── Column classification ────────────────────────────────────────────────────

_CLASSIFY_SYSTEM = """You analyze rows from an invoice table and identify column types.

LIR field types (19): line_item_amount_gross, line_item_amount_net, line_item_code,
line_item_currency, line_item_date, line_item_description,
line_item_discount_amount, line_item_discount_rate, line_item_hts_number,
line_item_order_id, line_item_person_name, line_item_position,
line_item_quantity, line_item_tax, line_item_tax_rate,
line_item_unit_price_gross, line_item_unit_price_net,
line_item_units_of_measure, line_item_weight

Return JSON only, no fences:
{
  "header_row": 0,
  "columns": {"0": "line_item_description", "1": null, "2": "line_item_quantity"}
}

Rules:
1. header_row: which row index (0-based) contains the column labels/headers
2. columns: for each column index, the LIR fieldtype or null if not applicable
3. null for: row numbers, checkboxes, blank columns, non-LIR content
4. "Amount"/"Total" without qualifier → line_item_amount_gross
5. "Unit Price" without qualifier → line_item_unit_price_net
6. "Description"/"Item"/"Product"/"Service" → line_item_description
7. "Qty"/"Quantity"/"Count" → line_item_quantity
8. "Tax Rate"/"VAT%" → line_item_tax_rate; "Tax"/"VAT" (amount) → line_item_tax
9. If no header row identifiable or no LIR columns exist, return {"header_row": 0, "columns": {}}
"""


async def _classify_table_structure(
    table_rows: list[dict[int, str]],
    model: str,
) -> tuple[int, dict[int, str | None]]:
    """One Sonnet call to identify header row and column fieldtypes.

    table_rows: first N rows as list of {col_idx: cell_text} dicts.
    Returns: (header_row_idx, {col_idx: lir_fieldtype_or_None})
    """
    lines = []
    for row_idx, row in enumerate(table_rows):
        cells_str = " | ".join(
            f"col{col}: {text[:40]!r}" for col, text in sorted(row.items()) if text
        )
        lines.append(f"row {row_idx}: {cells_str}")

    user_text = "Invoice table rows (first rows):\n" + "\n".join(lines)
    msg = await complete(
        model=model,
        system=_CLASSIFY_SYSTEM,
        messages=[{"role": "user", "content": user_text}],
        max_tokens=768,
        cache_system=True,
        temperature=0.0,
    )
    raw = (msg.content[0].text if msg.content else "{}").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return 0, {}

    header_row = int(parsed.get("header_row", 0))
    col_map_raw = parsed.get("columns", {})

    col_map: dict[int, str | None] = {}
    for k, v in col_map_raw.items():
        try:
            col_idx = int(k)
        except ValueError:
            continue
        if v is None or v == "null":
            col_map[col_idx] = None
        elif v in _LIR_TYPES:
            col_map[col_idx] = v
    return header_row, col_map


# ── HTML table parser ────────────────────────────────────────────────────────


class _TableHTMLParser(HTMLParser):
    """Extract (row_idx, col_idx, text) from HTML table."""

    def __init__(self):
        super().__init__()
        self.cells: list[tuple[int, int, str]] = []
        self._row = 0
        self._col = 0
        self._in_cell = False
        self._cell_text = ""

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._col = 0
        elif tag in ("td", "th"):
            self._in_cell = True
            self._cell_text = ""

    def handle_endtag(self, tag):
        if tag == "tr":
            self._row += 1
        elif tag in ("td", "th"):
            self.cells.append((self._row, self._col, self._cell_text.strip()))
            self._col += 1
            self._in_cell = False

    def handle_data(self, data):
        if self._in_cell:
            self._cell_text += data


def _parse_html_table(html: str) -> list[tuple[int, int, str]]:
    parser = _TableHTMLParser()
    parser.feed(html)
    return parser.cells


# ── Spatial join helpers ─────────────────────────────────────────────────────


def _normalize_cell(
    pixel_bbox: np.ndarray, img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """Convert cell bbox (4-point or 8-point polygon) to normalized AABB [0,1]."""
    coords = np.array(pixel_bbox).flatten()
    xs = coords[0::2]  # x coords at even indices
    ys = coords[1::2]  # y coords at odd indices
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    return (
        float(x1) / img_w,
        float(y1) / img_h,
        float(x2) / img_w,
        float(y2) / img_h,
    )


def _words_in_cell(
    words: list[WordBox], cell_norm: tuple[float, float, float, float]
) -> list[WordBox]:
    left, t, r, b = cell_norm
    matched = []
    for w in words:
        cx = (w.bbox[0] + w.bbox[2]) / 2
        cy = (w.bbox[1] + w.bbox[3]) / 2
        if left <= cx <= r and t <= cy <= b:
            matched.append(w)
    return matched


def _merge_bboxes(words: list[WordBox]) -> BBox:
    return BBox(
        min(w.bbox[0] for w in words),
        min(w.bbox[1] for w in words),
        max(w.bbox[2] for w in words),
        max(w.bbox[3] for w in words),
    )


# ── SLANET engine for pre-cropped tables ────────────────────────────────────

_ENGINE_SLANET = None


def _get_slanet_engine():
    global _ENGINE_SLANET
    if _ENGINE_SLANET is None:
        from rapid_table import ModelType, RapidTable, RapidTableInput

        _ENGINE_SLANET = RapidTable(RapidTableInput(model_type=ModelType.SLANETPLUS))
    return _ENGINE_SLANET


# ── Phase 4b: chained layout-detection + RapidTable on crop ─────────────────


async def extract_lir_chained(
    page: PageContext,
    table_regions: list[dict],
    model: str,
) -> list[Field]:
    """Phase 4b pipeline: YOLOv12s Table bboxes → RapidTable on crop → LIR Fields.

    table_regions: list of {"label":"Table","bbox":[tx1,ty1,tx2,ty2],"conf":float}
      where bbox is in pixel coords of page.image.

    Steps:
      1. Crop page.image to each Table region
      2. Filter words to those inside the region; convert to crop pixel coords for OCR
      3. Run SLANETPLUS on the crop → cell_bboxes (crop pixels) + logic_points
      4. Parse HTML (uses OCR words) → collect first 5 rows per region for Claude
      5. One Claude call per page: classify columns → fieldtype map
      6. Emit Fields: cell bbox → page pixel → page [0,1] → spatial join to words
    """
    if not table_regions:
        return []

    engine = _get_slanet_engine()
    img_w, img_h = page.image.size  # PIL (width, height)

    # Per-region: run RapidTable on crop, collect (crop_offset, output)
    region_outputs: list[tuple[float, float, float, float, Any]] = []
    region_htmls: list[str] = []

    for reg in table_regions:
        tx1, ty1, tx2, ty2 = reg["bbox"]
        # Clamp to image bounds
        tx1 = max(0.0, float(tx1))
        ty1 = max(0.0, float(ty1))
        tx2 = min(float(img_w), float(tx2))
        ty2 = min(float(img_h), float(ty2))
        cw = tx2 - tx1
        ch = ty2 - ty1
        if cw < 10 or ch < 10:
            continue

        # Crop page image to table region
        crop = page.image.crop((tx1, ty1, tx2, ty2))
        crop_np = np.array(crop)

        # Filter words inside table region; convert to crop pixel coords
        region_words = []
        for w in page.words:
            cx = (w.bbox[0] + w.bbox[2]) / 2 * img_w
            cy = (w.bbox[1] + w.bbox[3]) / 2 * img_h
            if tx1 <= cx <= tx2 and ty1 <= cy <= ty2:
                region_words.append(w)

        if not region_words:
            # Still run RapidTable — it will parse structure without OCR text
            ocr_input = (np.zeros((0, 4, 2), dtype=np.float32), (), ())
        else:
            # Convert word bboxes to crop pixel coords (4-point polygon)
            boxes = []
            for w in region_words:
                wl, wt, wr, wb = (
                    w.bbox[0] * img_w - tx1,
                    w.bbox[1] * img_h - ty1,
                    w.bbox[2] * img_w - tx1,
                    w.bbox[3] * img_h - ty1,
                )
                boxes.append([[wl, wt], [wr, wt], [wr, wb], [wl, wb]])
            ocr_input = (
                np.array(boxes, dtype=np.float32),
                tuple(w.text for w in region_words),
                tuple(1.0 for _ in region_words),
            )

        output = engine(crop_np, ocr_results=[ocr_input])
        region_outputs.append((tx1, ty1, tx2, ty2, output))

        if output.pred_htmls:
            region_htmls.extend(output.pred_htmls)

    if not region_outputs:
        return []

    # Collect first 5 rows from all detected tables for Claude
    all_sample_rows: list[dict[int, str]] = []
    for html in region_htmls:
        cells = _parse_html_table(html)
        rows: dict[int, dict[int, str]] = {}
        for row_idx, col_idx, text in cells:
            if row_idx > 6:
                continue
            rows.setdefault(row_idx, {})[col_idx] = text
        all_sample_rows.extend(rows[r] for r in sorted(rows)[:5])
        if len(all_sample_rows) >= 10:
            break

    if not all_sample_rows:
        return []

    header_row, col_to_fieldtype = await _classify_table_structure(all_sample_rows[:10], model)
    if not col_to_fieldtype:
        return []

    # Emit Fields: cell bbox (crop pixels) → page pixels → normalized → spatial join
    all_fields: list[Field] = []
    for table_idx, (tx1, ty1, _tx2, _ty2, output) in enumerate(region_outputs):
        if not output.cell_bboxes:
            continue

        for t_idx in range(len(output.cell_bboxes)):
            cell_bboxes = output.cell_bboxes[t_idx]
            logic_pts = output.logic_points[t_idx]
            if cell_bboxes is None or len(cell_bboxes) == 0:
                continue
            if logic_pts is None or len(logic_pts) == 0:
                continue

            for i in range(len(cell_bboxes)):
                row_start = int(logic_pts[i][0])
                col_start = int(logic_pts[i][2])
                if row_start <= header_row:
                    continue
                fieldtype = col_to_fieldtype.get(col_start)
                if not fieldtype:
                    continue

                # Map crop-relative cell bbox → page-normalized [0,1]
                # _normalize_cell handles both 4-pt and 8-pt polygon formats
                coords = np.array(cell_bboxes[i]).flatten()
                xs = coords[0::2]
                ys = coords[1::2]
                # Offset by crop origin, normalize by page dimensions
                page_norm = (
                    float(xs.min() + tx1) / img_w,
                    float(ys.min() + ty1) / img_h,
                    float(xs.max() + tx1) / img_w,
                    float(ys.max() + ty1) / img_h,
                )

                matched = _words_in_cell(page.words, page_norm)
                if not matched:
                    continue

                unique_row_id = page.page_index * 10000 + table_idx * 100 + row_start
                all_fields.append(
                    Field(
                        bbox=_merge_bboxes(matched),
                        page=page.page_index,
                        fieldtype=fieldtype,
                        line_item_id=unique_row_id,
                        score=0.9,
                    )
                )

    return all_fields


# ── Main per-doc extractor ───────────────────────────────────────────────────


async def extract_lir_for_doc(
    pages: list[PageContext],
    model: str,
) -> list[Field]:
    """Full RapidTable LIR pipeline for one document. Returns LIR Field list.

    Returns empty list if no tables found (caller should fall back to V5b LIR).
    """
    engine = _get_engine()

    # Run RapidTable on each page (CPU, synchronous)
    # Pass DocTR words as OCR input so we get header text in pred_html
    page_outputs: list[tuple[PageContext, Any]] = []
    for page in pages:
        img_np = np.array(page.image)
        img_w_px, img_h_px = page.image.size  # PIL (width, height)
        ocr_input = _words_to_ocr_input(page.words, img_w_px, img_h_px)
        output = engine(img_np, ocr_results=[ocr_input])
        page_outputs.append((page, output))

    # Build per-table structure: collect first 15 rows per table for Claude
    # Key: (page_idx, table_idx) → list of {col_idx: text} dicts (one per row)
    table_first_rows: dict[tuple[int, int], list[dict[int, str]]] = {}

    for page, output in page_outputs:
        if not output.pred_htmls:
            continue
        for t_idx, html in enumerate(output.pred_htmls):
            cells = _parse_html_table(html)
            rows: dict[int, dict[int, str]] = {}
            for row_idx, col_idx, text in cells:
                if row_idx > 14:  # first 15 rows for header detection
                    continue
                rows.setdefault(row_idx, {})[col_idx] = text
            if rows:
                table_first_rows[(page.page_index, t_idx)] = [rows[r] for r in sorted(rows)]

    if not table_first_rows:
        return []

    # One Claude call: collect first 10 rows from each table across all pages
    # Cap at 20 total rows to keep prompt reasonable
    combined_rows: list[dict[int, str]] = []
    for rows in table_first_rows.values():
        combined_rows.extend(rows[:10])
        if len(combined_rows) >= 20:
            break

    header_row, col_to_fieldtype = await _classify_table_structure(combined_rows[:20], model)
    if not col_to_fieldtype:
        return []

    # Emit LIR Fields: one per non-empty data cell with a known fieldtype
    all_fields: list[Field] = []

    for page, output in page_outputs:
        if not output.cell_bboxes:
            continue
        img_w, img_h = page.image.size  # PIL .size = (width, height)

        for t_idx in range(len(output.cell_bboxes)):
            cell_bboxes = output.cell_bboxes[t_idx]
            logic_pts = output.logic_points[t_idx]

            if cell_bboxes is None or len(cell_bboxes) == 0:
                continue
            if logic_pts is None or len(logic_pts) == 0:
                continue

            for i in range(len(cell_bboxes)):
                row_start = int(logic_pts[i][0])
                col_start = int(logic_pts[i][2])

                # Skip header row and any rows above it
                if row_start <= header_row:
                    continue

                fieldtype = col_to_fieldtype.get(col_start)
                if not fieldtype:
                    continue

                norm_bbox = _normalize_cell(cell_bboxes[i], img_w, img_h)
                matched = _words_in_cell(page.words, norm_bbox)
                if not matched:
                    continue

                # Make line_item_id unique across pages and tables
                unique_row_id = page.page_index * 10000 + t_idx * 100 + row_start

                all_fields.append(
                    Field(
                        bbox=_merge_bboxes(matched),
                        page=page.page_index,
                        fieldtype=fieldtype,
                        line_item_id=unique_row_id,
                        score=0.9,
                    )
                )

    return all_fields
