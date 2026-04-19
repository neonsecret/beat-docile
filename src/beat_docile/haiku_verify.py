"""[RESEARCH-BURIED] Haiku-backed span-precision verifier for DocILE field extraction.

Status: RESEARCH-BURIED — built and OFF in current best. Was part of the V6
ReAct pipeline (see KNOWLEDGE_BASE.md §6.7); also integrated into bbox_verify
(§6.2 / §3.3) as Pass 3. Net effect negative when combined with the refiner.
The verdict logic (accept/correct/reject) is sound but fires in a pipeline that
was itself buried. Preserved for potential future use in auditable-confidence
ensembles (§5.5).
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Literal

from .data import WordBox

logger = logging.getLogger(__name__)

_DEFAULT_HAIKU_MODEL = "claude-haiku-4-5"

# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM = """You are a span-precision verifier for document field extraction in the DocILE benchmark.

## Your task
Given an invoice document (as word positions) and a candidate field extraction, decide:
- **accept**: The candidate word_ids point to EXACTLY the right words for this field value. No changes needed.
- **correct**: The candidate is semantically right but the word_ids are wrong. Return the corrected word_ids.
- **reject**: The candidate is fundamentally wrong — wrong value, wrong field entirely, or field is absent.

## The metric you are optimizing: PCC-IoU = 1.0
DocILE uses Pseudo-Character Centers (PCCs). Each predicted bbox must cover EXACTLY the same characters as the gold annotation — not one character more or less. This means:
- **Label overrun**: If candidate includes a label word before the value (e.g. "Invoice No:" before "INV-001"), it FAILS. You must correct to value-only words.
- **Missing continuation**: If an address spans multiple lines and candidate only has line 1, it FAILS. Correct to include all address lines.
- **Trailing garbage**: If candidate includes a trailing colon, label, or unrelated word, it FAILS. Correct by removing it.
- **Wrong field**: If the words describe a different field type (e.g. customer name selected for customer_id), REJECT.

## Verdict rules
- **accept**: word_ids already point to exactly the value words. Nothing is missing; nothing is extra.
- **correct**: word_ids are for the right value but span is imprecise. Return corrected word_ids.
- **reject**: Candidate text/position is for the wrong field, or field is simply not there. Return empty [].

## Few-shot examples

### Example 1 — Label overrun → correct
Document words:
R0(y≈0.150): 42:Invoice  43:No:  44:INV-2024-001
R1(y≈0.200): 45:Date  46:2024-01-15

Candidate: fieldtype=document_id, word_ids=[42, 43, 44], text="Invoice No: INV-2024-001"
Analysis: Words 42 ("Invoice") and 43 ("No:") are labels. Only word 44 is the value.
{"verdict": "correct", "word_ids": [44], "confidence": 0.95, "reasoning": "Stripped label prefix 'Invoice No:'"}

### Example 2 — Missed continuation → correct
Document words:
R0(y≈0.300): 10:Acme  11:Corp
R1(y≈0.320): 12:123  13:Main  14:St
R2(y≈0.340): 15:London  16:EC1A  17:1BB

Candidate: fieldtype=customer_billing_address, word_ids=[10, 11], text="Acme Corp"
Analysis: Address continues across R1 and R2 — street and postcode are missing.
{"verdict": "correct", "word_ids": [10, 11, 12, 13, 14, 15, 16, 17], "confidence": 0.90, "reasoning": "Address missing continuation lines with street and city"}

### Example 3 — Wrong field → reject
Document words:
R0(y≈0.100): 5:John  6:Smith  7:Ltd
R1(y≈0.500): 8:CUST-001

Candidate: fieldtype=customer_id, word_ids=[5, 6, 7], text="John Smith Ltd"
Analysis: Words 5-7 are the customer NAME. Word 8 is the actual customer ID.
{"verdict": "reject", "word_ids": [], "confidence": 0.92, "reasoning": "Words are customer name, not customer_id; word 8 is the actual ID"}

### Example 4 — Already correct → accept
Document words:
R0(y≈0.600): 100:€  101:1,234.56

Candidate: fieldtype=amount_total_gross, word_ids=[100, 101], text="€ 1,234.56"
Analysis: Currency symbol and amount are both part of the value. Span is exact.
{"verdict": "accept", "word_ids": [100, 101], "confidence": 0.97, "reasoning": "Span is correct — currency symbol and amount included"}

### Example 5 — Trailing label → correct
Document words:
R0(y≈0.700): 200:GB29NWBK60161331926819  201:IBAN

Candidate: fieldtype=iban, word_ids=[200, 201], text="GB29NWBK60161331926819 IBAN"
Analysis: Word 201 ("IBAN") is a label that follows the value, not part of it.
{"verdict": "correct", "word_ids": [200], "confidence": 0.93, "reasoning": "Trailing label 'IBAN' stripped from value span"}

## Output format
Return valid JSON and NOTHING ELSE — no markdown fences, no explanation outside the JSON:
{"verdict": "accept"|"correct"|"reject", "word_ids": [...], "confidence": 0.0-1.0, "reasoning": "..."}

- On accept: word_ids = the original candidate word_ids unchanged.
- On correct: word_ids = the corrected word_ids (must be valid ids from the document).
- On reject: word_ids = [].
- confidence: your certainty about this verdict (0.0-1.0).
- reasoning: one short sentence.
"""


# ── Local word renderer ────────────────────────────────────────────────────────


def _words_to_prompt_local(words: list[WordBox]) -> str:
    """Render words as row-grouped layout for the verification prompt.

    Groups words into visual rows by top-y proximity (same style as extract.py).
    Format: 'R{i}(y≈{top:.3f}): {id}:{text}  {id}:{text} ...'
    """
    if not words:
        return "(no words)"

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
        tokens = "  ".join(f"{w.id}:{w.text}" for w in row)
        lines.append(f"R{i}(y≈{row_y:.3f}): {tokens}")
    return "\n".join(lines)


# ── Result dataclass ───────────────────────────────────────────────────────────


@dataclass
class VerificationResult:
    """Outcome of a Haiku verification pass on one candidate span."""

    verdict: Literal["accept", "correct", "reject"]
    word_ids: list[int]
    confidence: float
    reasoning: str = dc_field(default="")


# ── Response parser ────────────────────────────────────────────────────────────


def _parse_verdict(
    raw: str,
    candidate_word_ids: list[int],
    valid_word_ids: set[int],
) -> VerificationResult:
    """Parse Haiku JSON. Returns defensive accept on any failure."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Haiku verdict parse failed: %r", raw[:300])
        return VerificationResult(
            verdict="accept",
            word_ids=list(candidate_word_ids),
            confidence=0.5,
            reasoning="parse_failed",
        )

    verdict = data.get("verdict", "")
    if verdict not in ("accept", "correct", "reject"):
        logger.warning("Haiku returned unknown verdict %r", verdict)
        return VerificationResult(
            verdict="accept",
            word_ids=list(candidate_word_ids),
            confidence=0.5,
            reasoning="parse_failed",
        )

    confidence = float(data.get("confidence", 0.5))
    reasoning = str(data.get("reasoning", ""))

    if verdict == "accept":
        return VerificationResult(
            verdict="accept",
            word_ids=list(candidate_word_ids),
            confidence=confidence,
            reasoning=reasoning,
        )

    if verdict == "reject":
        return VerificationResult(
            verdict="reject",
            word_ids=[],
            confidence=confidence,
            reasoning=reasoning,
        )

    # verdict == "correct"
    raw_ids: list[int] = data.get("word_ids", [])
    valid_new_ids = [wid for wid in raw_ids if wid in valid_word_ids]
    if not valid_new_ids:
        logger.warning("Haiku 'correct' verdict returned no valid word_ids; falling back to accept")
        return VerificationResult(
            verdict="accept",
            word_ids=list(candidate_word_ids),
            confidence=0.5,
            reasoning="parse_failed",
        )

    return VerificationResult(
        verdict="correct",
        word_ids=valid_new_ids,
        confidence=confidence,
        reasoning=reasoning,
    )


# ── Core verification function ─────────────────────────────────────────────────


def verify_span(
    fieldtype: str,
    candidate_word_ids: list[int],
    candidate_text: str,
    words: list[WordBox],
    vertex_client,
    model: str = _DEFAULT_HAIKU_MODEL,
) -> VerificationResult:
    """Verify one candidate span via a single Haiku call.

    Prompt: system explains PCC-IoU=1.0 metric + few-shot examples; user provides
    row-grouped doc layout and the candidate {fieldtype, word_ids, text}.
    Haiku returns JSON {verdict, word_ids, confidence, reasoning}.

    On API / parse failure: returns defensive accept so no field is silently dropped.
    On correct verdict with all-invalid word_ids: returns defensive accept.

    Args:
        fieldtype: DocILE field type string.
        candidate_word_ids: Word IDs from the extraction to verify.
        candidate_text: Extracted text string (for context).
        words: Full page word list.
        vertex_client: AnthropicVertex client instance (from vertex.get_client()).
        model: Haiku model identifier.

    Returns:
        VerificationResult with verdict, (possibly corrected) word_ids, confidence.
    """
    valid_word_ids: set[int] = {w.id for w in words}
    doc_layout = _words_to_prompt_local(words)
    candidate_ids_str = ", ".join(str(wid) for wid in candidate_word_ids)

    user_text = (
        f"Document words (row-grouped, format 'id:text'):\n{doc_layout}\n\n"
        f"Candidate extraction to verify:\n"
        f"  fieldtype: {fieldtype}\n"
        f"  word_ids: [{candidate_ids_str}]\n"
        f"  text: {candidate_text!r}\n\n"
        "Verify this candidate. Return JSON only."
    )

    try:
        msg = vertex_client.messages.create(
            model=model,
            max_tokens=256,
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_text}],
        )
        raw = msg.content[0].text if msg.content else ""
    except Exception as exc:
        logger.error("Haiku API call failed for %s: %s", fieldtype, exc)
        return VerificationResult(
            verdict="accept",
            word_ids=list(candidate_word_ids),
            confidence=0.5,
            reasoning="api_error",
        )

    return _parse_verdict(raw, candidate_word_ids, valid_word_ids)


# ── Batch verification ─────────────────────────────────────────────────────────


def verify_extractions_batch(
    extractions: dict[str, list[tuple[list[int], str]]],
    words: list[WordBox],
    vertex_client,
    model: str = _DEFAULT_HAIKU_MODEL,
    parallel: int = 8,
) -> dict[str, list[VerificationResult]]:
    """Run verify_span across all candidates in extractions using a thread pool.

    Each (fieldtype, candidate) pair is an independent Haiku call. Results are
    returned in the same structure as extractions.

    Args:
        extractions: fieldtype -> [(word_ids, text), ...] from extraction step.
        words: Full page word list (same page as extractions).
        vertex_client: AnthropicVertex client instance.
        model: Haiku model identifier.
        parallel: Max concurrent Haiku calls.

    Returns:
        Dict with same keys as extractions; each value is a list of VerificationResults
        in the same order as the input candidates.
    """
    tasks: list[tuple[str, int, list[int], str]] = []
    for fieldtype, candidates in extractions.items():
        for idx, (word_ids, text) in enumerate(candidates):
            tasks.append((fieldtype, idx, word_ids, text))

    results: dict[str, list[VerificationResult | None]] = {
        ft: [None] * len(candidates) for ft, candidates in extractions.items()
    }

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        future_to_pos = {
            executor.submit(verify_span, ft, word_ids, text, words, vertex_client, model): (ft, idx)
            for ft, idx, word_ids, text in tasks
        }
        for future in as_completed(future_to_pos):
            ft, idx = future_to_pos[future]
            results[ft][idx] = future.result()

    final: dict[str, list[VerificationResult]] = {}
    for ft, res_list in results.items():
        final[ft] = [
            r
            if r is not None
            else VerificationResult(
                verdict="accept",
                word_ids=list(extractions[ft][i][0]),
                confidence=0.5,
                reasoning="missing_result",
            )
            for i, r in enumerate(res_list)
        ]
    return final


# ── Apply verdicts ─────────────────────────────────────────────────────────────


def apply_verification(
    extractions: dict[str, list[tuple[list[int], str]]],
    verifications: dict[str, list[VerificationResult]],
    min_confidence: float = 0.6,
) -> dict[str, list[tuple[list[int], str]]]:
    """Apply verification verdicts to produce updated extractions.

    Verdict application rules:
    - accept (any confidence): keep original word_ids unchanged.
    - correct (confidence >= min_confidence): replace word_ids with corrected ones.
    - correct (confidence < min_confidence): keep original (Haiku unsure of its correction).
    - reject (confidence >= min_confidence): drop the candidate entirely.
    - reject (confidence < min_confidence): keep original (Haiku unsure of rejection).

    Args:
        extractions: Original extractions dict (fieldtype -> [(word_ids, text), ...]).
        verifications: Parallel-structured dict from verify_extractions_batch.
        min_confidence: Threshold for acting on correct/reject verdicts.

    Returns:
        Updated extractions dict with verdicts applied.
    """
    updated: dict[str, list[tuple[list[int], str]]] = {}

    for fieldtype, candidates in extractions.items():
        field_results = verifications.get(fieldtype, [])
        updated_candidates: list[tuple[list[int], str]] = []

        for i, (word_ids, text) in enumerate(candidates):
            if i >= len(field_results):
                updated_candidates.append((word_ids, text))
                continue

            vr = field_results[i]

            if vr.verdict == "accept":
                updated_candidates.append((word_ids, text))

            elif vr.verdict == "correct":
                if vr.confidence >= min_confidence:
                    updated_candidates.append((vr.word_ids, text))
                else:
                    updated_candidates.append((word_ids, text))

            else:  # reject
                if vr.confidence >= min_confidence:
                    logger.debug(
                        "Dropped %s candidate (reject, conf=%.2f): %s",
                        fieldtype,
                        vr.confidence,
                        vr.reasoning,
                    )
                    # candidate dropped — do not append
                else:
                    updated_candidates.append((word_ids, text))

        updated[fieldtype] = updated_candidates

    return updated
