"""Unit tests for data.py. Ref: EVAL_SPEC §7."""

import pytest
from PIL import Image

from beat_docile.config import DATA_ROOT
from beat_docile.data import iter_pages, load_split


@pytest.mark.skipif(not (DATA_ROOT / "annotations").exists(), reason="DocILE dataset not available")
def test_val_first_doc_first_page():
    dataset = load_split("val")
    doc = next(iter(dataset))
    pages = list(iter_pages(doc))
    assert len(pages) >= 1
    page = pages[0]
    assert isinstance(page.image, Image.Image)
    assert len(page.words) > 0
    assert page.docid == doc.docid
    assert page.page_index == 0
    # All words should have valid relative bboxes
    for w in page.words:
        left, t, r, b = w.bbox
        assert 0 <= left <= r <= 1, f"invalid bbox left/right: {w.bbox}"
        assert 0 <= t <= b <= 1, f"invalid bbox top/bottom: {w.bbox}"
