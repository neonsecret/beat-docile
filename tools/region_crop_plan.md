# Item #1: PP-DocLayoutV3 Semantic Region Crop — Test Plan

## Hypothesis

PP-DocLayoutV3 (PaddlePaddle/OCR-Layout, 1.21M params, Apache 2.0) segments invoice pages
into semantic regions (header, body, footer, table, figure, title). Cropping the page to the
relevant semantic region before passing to Claude could help in two ways:

1. **Reduce distracting context**: Address blocks and financial codes are in specific regions.
   Feeding only the header region for vendor_name/vendor_address would reduce hallucination.
2. **Isolate table rows**: For LIR, feeding only the table region could improve line-item
   precision by eliminating non-table words from the word list.

## Why it was previously negative

From journey log: "DocLayout tags (Phase 6a) -0.1pp KILE / -3pp LIR" and "DocLayout filter
(Phase 6b) -9.2pp KILE / -3.5pp LIR". Two failure modes:
- Phase 6a: Adding region labels to the prompt was noise — Claude already sees the image.
- Phase 6b: Hard-filtering words outside the predicted region discarded valid words (PP-DocLayoutV3
  mislabels 15-20% of fields, especially for non-standard invoices).

## Why to retry (conservative version)

The conservative version doesn't filter hard. Instead:
1. Run PP-DocLayoutV3 on each page to get region bboxes.
2. For KILE non-address fields: identify the region containing the most words.
   Pass only words in that region as context (keeps 80-90% of pages intact).
3. For LIR: identify table regions specifically. Add a note to the prompt: "Tables are
   at regions: [bbox list]" without filtering words.
4. For address fields: no change (addresses need the full page context).

Gate: `BD_USE_REGION_CROP=1`

## Implementation plan

### Step 1: Verify PP-DocLayoutV3 availability on neon
```bash
ssh neon@100.98.171.97 "python3 -c 'from paddleocr import PPStructure; print(\"OK\")'"
```

If unavailable:
```bash
pip install paddlepaddle paddleocr
python3 -c "from paddleocr import PPStructure; s = PPStructure(layout=True, ocr=False); print('loaded')"
```

### Step 2: Build region extractor (on neon, CPU)
File: `src/beat_docile/region_crop.py`
```python
def get_regions(image) -> list[dict]:
    """Run PP-DocLayoutV3 on page image, return list of {type, bbox}."""
    ...

def get_table_regions(image) -> list[BBox]:
    """Return only table region bboxes (normalized 0-1)."""
    ...
```

### Step 3: 50-doc A/B on Mac
File: `tools/region_crop_50.py`
- `BD_USE_REGION_CROP=1`
- For LIR: add table region hint to prompt (no word filtering)
- Compare vs refiner_guard_50 baseline (44.87% KILE / 52.08% LIR)

### Step 4: Interpret results
- If LIR improves ≥+0.5pp and KILE neutral: APPLY (table-region hint is safe)
- If LIR improves ≥+0.5pp but KILE regresses: more investigation
- If no improvement: BURY permanently

## Notes
- PP-DocLayoutV3 runs at ~100ms/page on CPU — acceptable for 50-doc test
- Neon 3070 GPU would accelerate it 10x but CPU is sufficient for this experiment
- Do NOT use hard word-filtering (Phase 6b lesson). Hints only.
- Most invoices are 1-2 pages, so region detection is a minor overhead
