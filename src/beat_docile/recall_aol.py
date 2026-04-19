"""[EXPERIMENTAL] Recall-augmentation AOL: ADD missing KILE fields via cluster priors.

Status: EXPERIMENTAL — built and tested at 250-doc gate; net +0.02pp KILE (neutral).
See KNOWLEDGE_BASE.md §5.3 for design rationale and §8.15 for next steps.

Architectural rule: NEVER modify or demote existing predictions.
Only ADD fields the ensemble missed, guided by per-cluster priors from train data.
The 4% hit rate ceiling can be raised with a tighter per-doc field-presence prior
(see §8.15) or focused OCR-neighborhood crops for missing fields.

Design:
  1. Build per-cluster field prior from train annotations (≥50% presence threshold)
  2. For each val doc, find KILE fieldtypes present in cluster prior but absent in ensemble
  3. Re-prompt Claude for ONLY those missing fields, per page
  4. ADD new fields with score *0.7 (second-pass discount) — no overlap with existing
"""

from __future__ import annotations

import asyncio
import json
import re

from docile.dataset import BBox, Field

from .data import WordBox, iter_pages
from .ensemble import _iou
from .extract import _KILE_TYPES, _image_to_b64, _words_to_prompt
from .vertex import complete

# Multi-occurrence KILE fieldtypes (one per tax-rate row — can have >1 per doc)
_MULTI_OCCURRENCE = {"tax_detail_gross", "tax_detail_net", "tax_detail_rate", "tax_detail_tax"}

# Score discount for second-pass additions so they rank below confident first-pass TPs
_RECALL_SCORE_DISCOUNT = 0.7

# Fraction of train docs in cluster that must have a fieldtype for it to be "expected"
_PRESENCE_THRESHOLD = 0.50

_SYSTEM_RECALL = """You are a document information extraction assistant.

A previous extraction pass was run on this document. The fields listed in [Target Fields] were NOT found by that pass. Either they are genuinely absent from this document, or the previous pass missed them.

Your job: carefully look ONLY for the specific fields in [Target Fields].

Output format (JSON only, no markdown fences):
{"fields": [{"fieldtype": "...", "word_ids": [...], "text": "...", "score": 0.0-1.0}], "line_items": []}

Rules:
- word_ids must reference ids from the provided word list (format "id:text")
- Include ONLY the exact value words — not labels, colons, or surrounding text
- score: your confidence 0.0-1.0 for this specific extraction
- If a field is genuinely absent, do NOT include it in the output — omitting is always safe
- Return valid JSON only, no explanation, no markdown
- A false positive (claiming a field is present when it is not) hurts more than a miss — be conservative
- Better to return empty {"fields": [], "line_items": []} than to guess
"""


def build_cluster_field_prior(
    train_dataset,
    presence_threshold: float = _PRESENCE_THRESHOLD,
) -> dict[int, set[str]]:
    """Build {cluster_id: set[fieldtype]} from train annotations.

    A fieldtype is included for a cluster if it appears in ≥presence_threshold
    fraction of train docs in that cluster (KILE fields only, not LIR).
    Does NOT require OCR — only annotation labels are read.
    """
    cluster_field_counts: dict[int, dict[str, int]] = {}
    cluster_doc_counts: dict[int, int] = {}

    for doc in train_dataset:
        try:
            cid = doc.annotation.cluster_id
        except Exception:
            cid = None
        if cid is None or cid < 0:
            continue

        cluster_doc_counts[cid] = cluster_doc_counts.get(cid, 0) + 1
        seen_types = {
            f.fieldtype
            for f in doc.annotation.fields
            if f.line_item_id is None and f.fieldtype in _KILE_TYPES
        }
        for ft in seen_types:
            cluster_field_counts.setdefault(cid, {}).setdefault(ft, 0)
            cluster_field_counts[cid][ft] += 1

    field_prior: dict[int, set[str]] = {}
    for cid, counts in cluster_field_counts.items():
        n_docs = cluster_doc_counts[cid]
        field_prior[cid] = {ft for ft, cnt in counts.items() if cnt / n_docs >= presence_threshold}
    return field_prior


def _find_missing_kile_types(
    doc_preds: list[Field],
    expected_fields: set[str],
) -> set[str]:
    """Return fieldtypes in expected_fields with NO prediction in doc_preds (KILE only)."""
    found_types = {f.fieldtype for f in doc_preds if f.line_item_id is None}
    return expected_fields - found_types


def _has_overlap(new_field: Field, existing: list[Field], iou_threshold: float = 0.01) -> bool:
    """True if new_field overlaps any existing prediction of the same fieldtype on the same page."""
    for f in existing:
        if f.fieldtype == new_field.fieldtype and f.page == new_field.page and _iou(f.bbox, new_field.bbox) > iou_threshold:
            return True
    return False


def _parse_recall_response(
    raw: str,
    words: list[WordBox],
    page_idx: int,
    target_types: set[str],
) -> list[Field]:
    """Parse recall re-prompt JSON → list of Field. Only keep fields in target_types."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []

    id_to_word = {w.id: w for w in words}
    results: list[Field] = []

    for item in data.get("fields", []):
        ft = item.get("fieldtype", "")
        if ft not in target_types:
            continue
        word_ids = item.get("word_ids", [])
        bboxes = [id_to_word[wid].bbox for wid in word_ids if wid in id_to_word]
        if not bboxes:
            continue
        bbox = BBox(
            min(b[0] for b in bboxes),
            min(b[1] for b in bboxes),
            max(b[2] for b in bboxes),
            max(b[3] for b in bboxes),
        )
        raw_score = float(item.get("score", 0.7))
        score = raw_score * _RECALL_SCORE_DISCOUNT
        results.append(Field(bbox=bbox, page=page_idx, fieldtype=ft, score=score))

    return results


async def _recall_page(
    page,
    model: str,
    missing_types: set[str],
) -> list[Field]:
    """Re-prompt Claude for one page, asking only for missing fieldtypes."""
    if not page.words or not missing_types:
        return []

    img_b64 = _image_to_b64(page.image)
    words_layout = _words_to_prompt(page.words)
    target_list = ", ".join(sorted(missing_types))

    msg = await complete(
        model=model,
        system=_SYSTEM_RECALL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": img_b64},
                    },
                    {
                        "type": "text",
                        "text": (
                            f"[Target Fields]\n{target_list}\n\n"
                            f"[Document words]\n{words_layout}\n\n"
                            f"[Task]\nLook for these specific fields: {target_list}. "
                            "If any field is genuinely not present in this document, do NOT include it. "
                            "Omitting is always safe. Return JSON only."
                        ),
                    },
                ],
            }
        ],
        max_tokens=1024,
        cache_system=True,
    )

    raw = msg.content[0].text if msg.content else ""
    return _parse_recall_response(raw, page.words, page.page_index, missing_types)


async def apply_recall_aol(
    preds: dict[str, list[Field]],
    model: str,
    val_dataset,
    field_prior: dict[int, set[str]],
    max_workers: int = 4,
) -> tuple[dict[str, list[Field]], dict]:
    """Apply recall-augmentation to val docs by adding cluster-prior-guided missing fields.

    For each val doc:
      1. Get cluster_id → look up expected KILE fields from prior
      2. Find which expected fields are completely absent from ensemble predictions
      3. Re-prompt Claude per page for ONLY those missing fields
      4. ADD new non-overlapping finds with score *0.7; never touch existing predictions

    Single-occurrence fields: keep only the highest-scoring candidate across all pages.
    Multi-occurrence fields (tax_detail_*): keep all non-overlapping candidates.

    Returns (augmented_preds, stats_dict).
    """
    sem = asyncio.Semaphore(max_workers)
    augmented: dict[str, list[Field]] = {docid: list(fields) for docid, fields in preds.items()}
    stats: dict[str, int] = {
        "docs_with_cluster": 0,
        "docs_skipped_no_cluster": 0,
        "docs_with_missing_fields": 0,
        "total_missing_field_type_slots": 0,
        "total_reprompts": 0,
        "total_added": 0,
    }

    async def process_doc(doc) -> None:
        docid = doc.docid
        if docid not in preds:
            return

        try:
            cid = doc.annotation.cluster_id
        except Exception:
            cid = None

        if cid is None or cid < 0 or cid not in field_prior:
            stats["docs_skipped_no_cluster"] += 1
            return

        stats["docs_with_cluster"] += 1
        expected = field_prior[cid]
        missing = _find_missing_kile_types(preds[docid], expected)

        if not missing:
            return

        stats["docs_with_missing_fields"] += 1
        stats["total_missing_field_type_slots"] += len(missing)

        async def prompt_page_bounded(page):
            async with sem:
                stats["total_reprompts"] += 1
                return await _recall_page(page, model, missing)

        pages = list(iter_pages(doc))
        page_results = await asyncio.gather(*[prompt_page_bounded(p) for p in pages])

        # Collect new candidates per fieldtype
        existing = augmented[docid]
        new_by_type: dict[str, list[Field]] = {}
        for new_fields in page_results:
            for f in new_fields:
                if not _has_overlap(f, existing):
                    new_by_type.setdefault(f.fieldtype, []).append(f)

        # Add: multi-occurrence fields → keep all; single-occurrence → keep best only
        for ft, candidates in new_by_type.items():
            if ft in _MULTI_OCCURRENCE:
                for f in candidates:
                    existing.append(f)
                    stats["total_added"] += 1
            else:
                best = max(candidates, key=lambda f: f.score)
                existing.append(best)
                stats["total_added"] += 1

    tasks = [process_doc(doc) for doc in val_dataset if doc.docid in preds]
    await asyncio.gather(*tasks)

    return augmented, stats
