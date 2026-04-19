"""[ACTIVE] Cluster-based few-shot retrieval for Claude extraction.

Status: ACTIVE — used in current best (v2_ensemble).
See KNOWLEDGE_BASE.md §3 for the architecture map.

75% of val/test docs have exact template matches in the train split.
We show 1-2 annotated train examples from the same cluster to Claude
before the query document, calibrating it to the specific invoice layout.
The few-shot examples include page image, row-grouped word list, and gold JSON.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass

from docile.dataset import Dataset, Field

from .config import DATA_ROOT
from .data import iter_pages


@dataclass
class FewShotExample:
    docid: str
    cluster_id: int
    image_b64: str
    words_layout: str  # row-grouped word list
    gold_json: str  # compact JSON for the expected extraction


def _build_cluster_index(split: str = "train") -> dict[int, list[str]]:
    """Return {cluster_id: [docid, ...]} for the given split."""
    ds = Dataset(split, DATA_ROOT, load_annotations=True, load_ocr=False)
    index: dict[int, list[str]] = {}
    for doc in ds:
        cid = doc.annotation.cluster_id
        index.setdefault(cid, []).append(doc.docid)
    return index


def _gold_to_compact_json(fields: list[Field], li_fields: list[Field]) -> str:
    """Convert gold fields to a compact JSON showing fieldtype→text (no word_ids).

    This format shows Claude WHAT fields exist and their values —
    it doesn't need to reference word_ids since the few-shot is illustrative.
    """
    kile_out = [{"fieldtype": f.fieldtype, "text": f.text or ""} for f in fields]
    # Group LIR fields by line_item_id
    li_groups: dict[int, list[dict]] = {}
    for f in li_fields:
        li_id = f.line_item_id or 0
        li_groups.setdefault(li_id, []).append({"fieldtype": f.fieldtype, "text": f.text or ""})
    lir_out = [{"line_item_id": k, "fields": v} for k, v in sorted(li_groups.items())]
    return json.dumps({"fields": kile_out, "line_items": lir_out}, separators=(",", ":"))


def _image_to_b64(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def load_few_shot_examples(
    cluster_ids: list[int],
    train_index: dict[int, list[str]],
    max_per_cluster: int = 1,
) -> dict[int, list[FewShotExample]]:
    """Load rendered few-shot examples for the requested cluster IDs.

    Returns {cluster_id: [FewShotExample, ...]} — empty list for unknown clusters.
    Loads from the train split using the cluster index.

    Uses the full train Dataset (load_ocr=False) and opens each needed doc
    individually via context manager to avoid the docids-subset validation error.
    """
    from .extract import _words_to_prompt  # avoid circular at module level

    # Collect all needed docids
    needed: dict[str, int] = {}  # docid → cluster_id
    for cid in set(cluster_ids):
        for docid in train_index.get(cid, [])[:max_per_cluster]:
            needed[docid] = cid

    if not needed:
        return {}

    # Load the full train split with annotations only (OCR loaded per-doc below)
    train_ds = Dataset(
        "train",
        DATA_ROOT,
        load_annotations=True,
        load_ocr=False,
    )
    # Build a lookup from docid to Document object
    doc_lookup = {doc.docid: doc for doc in train_ds if doc.docid in needed}

    result: dict[int, list[FewShotExample]] = {}
    for docid, cid in needed.items():
        doc = doc_lookup.get(docid)
        if doc is None:
            continue
        # Open doc with context manager to load OCR into memory temporarily
        with doc:
            pages = list(iter_pages(doc))
            if not pages:
                continue
            page = pages[0]  # first page only for few-shot

            gold_fields = doc.annotation.fields
            gold_li = doc.annotation.li_fields

            example = FewShotExample(
                docid=doc.docid,
                cluster_id=cid,
                image_b64=_image_to_b64(page.image),
                words_layout=_words_to_prompt(page.words),
                gold_json=_gold_to_compact_json(gold_fields, gold_li),
            )
        result.setdefault(cid, []).append(example)

    return result


def build_few_shot_messages(examples: list[FewShotExample]) -> list[dict]:
    """Build multi-turn messages representing few-shot demonstrations.

    Returns a list of {role, content} dicts to prepend before the query user message.
    Claude sees: user (example image+words) → assistant (gold extraction) x N examples.
    """
    messages = []
    for ex in examples:
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": ex.image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Words grouped by visual row:\n{ex.words_layout}\n\n"
                            "Extract all fields from this invoice page. Return JSON only."
                        ),
                    },
                ],
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": ex.gold_json,
            }
        )
    return messages
