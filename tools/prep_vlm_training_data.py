"""Convert DocILE train split to Qwen3-VL fine-tuning JSONL.

Output format: LLaMA-Factory sharegpt with Qwen3-VL grounding JSON.
Bbox: DocILE normalized [0,1] → Qwen3-VL [0,1000] (multiply × 1000, round).
Image: cached_images/{docid}/0.png or rendered from pdfs/{docid}.pdf.

PREREQUISITE — download full DocILE train split first (we only have 144/5180 locally):
    DOCILE_TOKEN=<your-token> bash runpod/download_data.sh

Usage:
    uv run python tools/prep_vlm_training_data.py [--data-root PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DATA_ROOT_DEFAULT = Path(__file__).parent.parent / "data"

KILE_FIELD_TYPES = [
    "account_num", "amount_due", "amount_paid", "amount_total_gross",
    "amount_total_net", "amount_total_tax", "bank_num", "bic",
    "currency_code_amount_due", "customer_billing_address", "customer_billing_name",
    "customer_delivery_address", "customer_delivery_name", "customer_id",
    "customer_order_id", "customer_other_address", "customer_other_name",
    "customer_registration_id", "customer_tax_id", "date_due", "date_issue",
    "document_id", "iban", "order_id", "payment_reference", "payment_terms",
    "seller_address", "seller_ico", "seller_name", "seller_registration_id",
    "seller_tax_id", "seller_vat_id", "tax_detail_gross", "tax_detail_net",
    "tax_detail_rate", "tax_detail_tax",
]

SYSTEM_PROMPT = (
    "You are a document field extraction assistant. "
    "Extract KILE invoice fields from the document image. "
    "For each detected field output its text value and bounding box "
    "as bbox_2d: [x1, y1, x2, y2] with coordinates in 0-1000 scale "
    "(0=left/top edge, 1000=right/bottom edge). "
    "Output ONLY valid JSON, no other text."
)

USER_PROMPT = (
    "<image>\n"
    "Extract all KILE invoice fields visible on this page. "
    "Return a JSON object where each key is a field type name and each value is an object "
    "with 'text' (the extracted string) and 'bbox_2d' ([x1, y1, x2, y2] in 0-1000 scale). "
    "If a field is not present, omit it. "
    f"Possible field types: {', '.join(KILE_FIELD_TYPES)}."
)

# Render scale for PDF → PNG (150 DPI equivalent; pypdfium2 scale=1.0 ≈ 72 DPI)
PDF_RENDER_SCALE = 150 / 72


def docile_bbox_to_qwen(bbox: list[float]) -> list[int]:
    """Convert DocILE [0,1] bbox to Qwen3-VL [0,1000] int coords."""
    x1, y1, x2, y2 = bbox
    return [round(x1 * 1000), round(y1 * 1000), round(x2 * 1000), round(y2 * 1000)]


def get_page0_image_path(docid: str, data_root: Path, render_dir: Path) -> Path | None:
    """Return path to page-0 PNG; render from PDF if needed. None if unavailable."""
    img_dir = data_root / "cached_images"
    # Try plain dir and resolution-suffixed dir (e.g. {docid}__1275x1650)
    for candidate_dir in [img_dir / docid, img_dir / f"{docid}__1275x1650"]:
        candidate = candidate_dir / "0.png"
        if candidate.exists():
            return candidate

    pdf_path = data_root / "pdfs" / f"{docid}.pdf"
    if not pdf_path.exists():
        return None

    render_dir.mkdir(parents=True, exist_ok=True)
    out_png = render_dir / f"{docid}_page0.png"
    if out_png.exists():
        return out_png

    try:
        import pypdfium2
        pdf = pypdfium2.PdfDocument(str(pdf_path))
        page = pdf[0]
        img = page.render(scale=PDF_RENDER_SCALE).to_pil()
        img.save(str(out_png))
        return out_png
    except Exception as e:
        print(f"  [warn] Failed to render {docid}: {e}")
        return None


def build_gpt_response(fields_page0: list[dict]) -> str:
    """Build assistant JSON response from page-0 fields."""
    output: dict[str, dict] = {}
    for f in fields_page0:
        ft = f["fieldtype"]
        output[ft] = {
            "text": f["text"],
            "bbox_2d": docile_bbox_to_qwen(f["bbox"]),
        }
    return json.dumps(output, ensure_ascii=False)


def process_split(data_root: Path, out_path: Path) -> None:
    train_docids: list[str] = json.loads((data_root / "train.json").read_text())
    ann_dir = data_root / "annotations"
    render_dir = data_root / "vlm_rendered_pages"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    skipped_no_image = 0
    skipped_no_fields = 0
    written = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for i, docid in enumerate(train_docids):
            if i % 500 == 0:
                print(f"  [{i}/{len(train_docids)}] written={written} skipped_img={skipped_no_image}")

            img_path = get_page0_image_path(docid, data_root, render_dir)
            if img_path is None:
                skipped_no_image += 1
                continue

            ann_path = ann_dir / f"{docid}.json"
            if not ann_path.exists():
                skipped_no_fields += 1
                continue

            ann = json.loads(ann_path.read_text())
            fields_page0 = [
                f for f in ann.get("field_extractions", [])
                if f.get("page", 0) == 0
            ]
            if not fields_page0:
                skipped_no_fields += 1
                continue

            record = {
                "conversations": [
                    {"from": "human", "value": USER_PROMPT},
                    {"from": "gpt", "value": build_gpt_response(fields_page0)},
                ],
                "system": SYSTEM_PROMPT,
                "image": str(img_path.resolve()),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"\nDone.")
    print(f"  Written:               {written}")
    print(f"  Skipped (no image):    {skipped_no_image}  ← need full dataset download")
    print(f"  Skipped (no fields):   {skipped_no_fields}")
    print(f"  Output:                {out_path}")

    if written == 0:
        return

    with out_path.open() as f:
        sample = json.loads(f.readline())
    print("\n--- Sample record ---")
    print("Image:", sample["image"])
    gpt_val = json.loads(sample["conversations"][1]["value"])
    print(f"GPT fields ({len(gpt_val)} total):", list(gpt_val.keys()))
    first_key, first_val = next(iter(gpt_val.items()))
    print(f"  '{first_key}': {first_val}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prep DocILE → Qwen3-VL JSONL")
    parser.add_argument(
        "--data-root", type=Path, default=DATA_ROOT_DEFAULT,
        help="Path to DocILE data root (default: ./data)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/vlm_training/train.jsonl"),
        help="Output JSONL path (default: data/vlm_training/train.jsonl)",
    )
    args = parser.parse_args()
    process_split(args.data_root, args.out)


if __name__ == "__main__":
    main()
