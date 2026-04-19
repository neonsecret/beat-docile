"""[RESEARCH-BURIED] 3-pass bbox verifier for re-grounding Claude's word_id selections.

Status: RESEARCH-BURIED — +1.36pp on 50-doc subset, -1.86pp on full 500-doc.
See KNOWLEDGE_BASE.md §3.3 for build details; §6.2 for the 500-doc regression.
Default OFF (gate with BD_USE_BBOX_VERIFY=1 env flag).

Pass 1: Keyword hit ratio (lexical signal — fast, always runs).
Pass 2: SequenceMatcher fuzzy ratio (catches OCR/spacing differences).
Pass 3: Haiku LLM tie-breaker (only when max(p1+p2) < 0.4).
Composite: 0.60 * p1 + 0.40 * p2.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass

from docile.dataset import BBox

from .data import WordBox

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "cannot",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "from",
        "by",
        "and",
        "or",
        "but",
        "not",
        "if",
        "as",
        "so",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
    ]
)

_LLM_THRESHOLD = 0.4


@dataclass
class BboxVerification:
    word_ids: list[int]
    bbox: BBox
    confidence: float
    corrected: bool


def _cluster_bbox(word_ids: list[int], words: list[WordBox]) -> BBox | None:
    """Merge word bboxes into enclosing BBox (min/max). Returns None if no valid ids."""
    id_to_word = {w.id: w for w in words}
    bboxes = [
        id_to_word[wid].bbox
        for wid in word_ids
        if wid in id_to_word and id_to_word[wid].bbox is not None
    ]
    if not bboxes:
        return None
    return BBox(
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    )


def _cluster_text(word_ids: list[int], id_to_word: dict[int, WordBox]) -> str:
    """Join word texts for a cluster in id-sorted order (OCR reading order)."""
    return " ".join(id_to_word[wid].text for wid in sorted(word_ids) if wid in id_to_word)


def build_candidate_clusters(
    proposed_word_ids: list[int],
    words: list[WordBox],
    expansion_radius: int = 3,
) -> list[list[int]]:
    """Generate candidate word_id clusters to test against extracted_text.

    Produces:
    - The proposed cluster itself (always first)
    - Nearby words (±expansion_radius from the min/max proposed id)
    - Same-row words (words within row_gap of proposed bbox top)
    - Contiguous runs of similar span size, offset by ±expansion_radius

    Returns deduplicated list of candidate clusters (each sorted list of word_ids).
    """
    if not words:
        return [list(proposed_word_ids)] if proposed_word_ids else []

    id_to_word = {w.id: w for w in words}
    all_ids = sorted(w.id for w in words)
    id_set = set(all_ids)

    if not proposed_word_ids:
        return []

    valid_proposed = sorted(set(proposed_word_ids) & id_set)
    if not valid_proposed:
        return []

    candidates: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()

    def add_candidate(ids: list[int]) -> None:
        ids_valid = sorted(set(ids) & id_set)
        if not ids_valid:
            return
        key = tuple(ids_valid)
        if key not in seen:
            seen.add(key)
            candidates.append(ids_valid)

    min_id = min(valid_proposed)
    max_id = max(valid_proposed)

    # 1. Proposed cluster itself (always first)
    add_candidate(valid_proposed)

    # 2. Expanded window: ±expansion_radius from min/max proposed id
    add_candidate(
        [i for i in all_ids if min_id - expansion_radius <= i <= max_id + expansion_radius]
    )

    # Left-only and right-only expansions
    add_candidate([i for i in all_ids if min_id - expansion_radius <= i <= max_id])
    add_candidate([i for i in all_ids if min_id <= i <= max_id + expansion_radius])

    # 3. Same-row words: within row_gap of proposed bbox centroid
    row_gap = 0.012
    proposed_tops = [id_to_word[wid].bbox[1] for wid in valid_proposed if wid in id_to_word]
    if proposed_tops:
        avg_top = sum(proposed_tops) / len(proposed_tops)
        add_candidate([w.id for w in words if abs(w.bbox[1] - avg_top) <= row_gap])

    # 4. Contiguous runs that partially overlap proposed — sliding windows of similar span
    span_size = max(1, max_id - min_id + 1)
    for offset in range(-expansion_radius, expansion_radius + 1):
        start = min_id + offset
        add_candidate([i for i in all_ids if start <= i < start + span_size])
        # Slightly wider window to capture boundary words
        add_candidate([i for i in all_ids if start - 1 <= i <= start + span_size])

    return candidates


def verify_bbox(
    fieldtype: str,
    proposed_word_ids: list[int],
    extracted_text: str,
    words: list[WordBox],
    vertex_client,
    use_llm_fallback: bool = True,
    haiku_model: str = "claude-haiku-4-5",
) -> BboxVerification:
    """3-pass verification that proposed_word_ids correctly ground extracted_text.

    Pass 1 — Keyword hit ratio:
        Non-stopword tokens in extracted_text; fraction found in each candidate cluster.

    Pass 2 — SequenceMatcher fuzzy ratio:
        difflib ratio between extracted_text and cluster text (case-insensitive).

    Pass 3 — LLM fallback (Haiku):
        Only when max(p1 + p2) across all candidates < 0.4 AND use_llm_fallback=True.
        Top-5 clusters sent to Haiku; winner's composite is boosted to 1.0.

    Composite: 0.60 * p1 + 0.40 * p2.
    Defensive: any exception returns proposed_word_ids unchanged.
    """
    fallback = BboxVerification(
        word_ids=list(proposed_word_ids),
        bbox=_cluster_bbox(proposed_word_ids, words) or BBox(0.0, 0.0, 0.0, 0.0),
        confidence=0.0,
        corrected=False,
    )

    try:
        if not proposed_word_ids or not words or not extracted_text.strip():
            return fallback

        id_to_word = {w.id: w for w in words}
        candidates = build_candidate_clusters(proposed_word_ids, words)
        if not candidates:
            return fallback

        # Extract non-stopword keywords from extracted_text
        keywords = [
            t.lower()
            for t in re.findall(r"[a-zA-Z0-9]+", extracted_text)
            if len(t) >= 2 and t.lower() not in _STOPWORDS
        ]

        # Score each candidate cluster
        # scored entries: (composite, p1+p2, word_ids)
        scored: list[tuple[float, float, list[int]]] = []
        for cluster_ids in candidates:
            cluster_lower = _cluster_text(cluster_ids, id_to_word).lower()

            # Pass 1: keyword hit ratio
            p1 = (
                sum(1 for kw in keywords if kw in cluster_lower) / len(keywords)
                if keywords
                else 0.0
            )

            # Pass 2: SequenceMatcher fuzzy ratio
            p2 = difflib.SequenceMatcher(None, extracted_text.lower(), cluster_lower).ratio()

            composite = 0.60 * p1 + 0.40 * p2
            scored.append((composite, p1 + p2, cluster_ids))

        # Pass 3: LLM fallback when all scores are low
        max_p1_p2 = max(s[1] for s in scored)
        if use_llm_fallback and max_p1_p2 < _LLM_THRESHOLD and vertex_client is not None:
            top5 = sorted(scored, key=lambda x: x[1], reverse=True)[:5]
            cluster_reprs = [
                f"{i}: [{_cluster_text(ids, id_to_word)}]" for i, (_, _, ids) in enumerate(top5)
            ]
            prompt = (
                f'Which of these word clusters best contains the value "{extracted_text}" '
                f"for field {fieldtype}? "
                f"Reply with the cluster index only (0-{len(top5) - 1}).\n\n"
                + "\n".join(cluster_reprs)
            )
            try:
                response = vertex_client.messages.create(
                    model=haiku_model,
                    max_tokens=16,
                    system=(
                        "You are a document field extraction verifier. "
                        "Reply with a single integer index only."
                    ),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                raw = response.content[0].text.strip()
                m = re.search(r"\d+", raw)
                if m:
                    idx = int(m.group())
                    if 0 <= idx < len(top5):
                        llm_ids = top5[idx][2]
                        # Boost the LLM-selected candidate; update in-place if present
                        for j, (_, p12, ids) in enumerate(scored):
                            if ids == llm_ids:
                                scored[j] = (1.0, p12, ids)
                                break
                        else:
                            scored.append((1.0, 1.0, llm_ids))
            except Exception as exc:
                logger.warning("verify_bbox: LLM fallback failed for %s: %s", fieldtype, exc)

        best_composite, _, best_ids = max(scored, key=lambda x: x[0])
        proposed_composite = scored[0][0]  # proposed is always first in candidates

        # Defensive: when both proposed and best are very low, return proposed unchanged
        if proposed_composite < 0.1 and best_composite < 0.1:
            return BboxVerification(
                word_ids=list(proposed_word_ids),
                bbox=_cluster_bbox(proposed_word_ids, words) or BBox(0.0, 0.0, 0.0, 0.0),
                confidence=proposed_composite,
                corrected=False,
            )

        corrected = set(best_ids) != set(proposed_word_ids)
        final_ids = best_ids if corrected else list(proposed_word_ids)
        final_bbox = _cluster_bbox(final_ids, words) or BBox(0.0, 0.0, 0.0, 0.0)

        return BboxVerification(
            word_ids=final_ids,
            bbox=final_bbox,
            confidence=best_composite,
            corrected=corrected,
        )

    except Exception as exc:
        logger.exception("verify_bbox: unexpected error for %s: %s", fieldtype, exc)
        return fallback
