"""[ACTIVE] DocILE dataset loader — page iterator with snapped OCR words.

Status: ACTIVE — used in current best (v2_ensemble).
See KNOWLEDGE_BASE.md §3 for the architecture map.

Always uses snapped=True for word bboxes — PCCs are derived from snapped coords only.
Ref: EVAL_SPEC §7.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

from docile.dataset import Dataset, Document, Field
from PIL import Image

from .config import DATA_ROOT


@dataclasses.dataclass
class WordBox:
    id: int
    text: str
    bbox: tuple[float, float, float, float]  # (left, top, right, bottom) relative [0,1]
    page: int


@dataclasses.dataclass
class PageContext:
    docid: str
    page_index: int
    image: Image.Image
    words: list[WordBox]


def load_split(split: str) -> Dataset:
    """Load a DocILE split (train/val/test/trainval) from DATA_ROOT."""
    return Dataset(split_name=split, dataset_path=DATA_ROOT, load_annotations=True, load_ocr=True)


def iter_pages(doc: Document) -> Iterator[PageContext]:
    """Yield PageContext for every page of a document with snapped word boxes."""
    with doc:
        for page_idx in range(doc.page_count):
            # Render at 150 DPI: default is 200 DPI, use image_size to scale
            w150, h150 = doc.page_image_size(page_idx, dpi=150)
            image = doc.page_image(page_idx, image_size=(w150, h150))

            words_fields: list[Field] = doc.ocr.get_all_words(
                page=page_idx,
                snapped=True,
                use_cached_snapping=True,
                get_page_image=lambda _img=image: _img,
            )

            words = [
                WordBox(
                    id=i,
                    text=f.text,
                    bbox=f.bbox.to_tuple(),
                    page=page_idx,
                )
                for i, f in enumerate(words_fields)
            ]

            yield PageContext(
                docid=doc.docid,
                page_index=page_idx,
                image=image,
                words=words,
            )


def load_gold(split: str) -> dict[str, list[Field]]:
    """Return {docid: [Field, ...]} gold annotations for a split."""
    dataset = load_split(split)
    result: dict[str, list[Field]] = {}
    for doc in dataset:
        result[doc.docid] = doc.annotation.fields
    return result
