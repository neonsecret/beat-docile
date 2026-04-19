"""[RESEARCH-BURIED] Conservative AOL: calc + overlap verifiers on existing predictions.

Status: RESEARCH-BURIED — all four configurations regress AP (-1.45pp to -4.20pp).
See KNOWLEDGE_BASE.md §6.6 for details. Root cause: rank disturbance from demoting
valid TPs on a high-AP baseline (see §7.4). Score-modifying verifiers are
structurally wrong on a strong baseline. Best remaining use: HITL escalation
signal rather than score-modifier (see §8.11).

Post-processes existing predictions with two KEEP-skewed verifier passes:
  1. Calc verifier: amount_total_gross ≈ amount_total_net + amount_total_tax
     - Math fail → demote * 0.5 + optional one-shot re-prompt per page
  2. Overlap verifier: two KILE fields with >50% bbox overlap (by smaller)
     - Overlap → demote BOTH * 0.5 (never delete)
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from docile.dataset import BBox, Field

_AMOUNT_FIELDS = {"amount_total_net", "amount_total_gross", "amount_total_tax"}
_CALC_TOLERANCE = 0.03  # 3% relative tolerance for gross ≈ net + tax


# ── Amount text parsing ───────────────────────────────────────────────────────


def parse_amount(text: str | None) -> float | None:
    """Best-effort parse of a currency amount string to float.

    Handles: "1,234.56", "1.234,56", "1234.56", "EUR 1234", "€ 1.234,56 EUR".
    Returns None if unparseable.
    """
    if not text:
        return None
    # Strip currency symbols, letters (except E for exponents), spaces
    cleaned = re.sub(r"[€$£¥₹\u20a3\u20b9a-df-zA-DF-Z\s]", "", text)
    # Detect European format: e.g. "1.234,56" (period = thousands, comma = decimal)
    if re.search(r"\.\d{3},\d{2}$", cleaned) or (
        "," in cleaned and "." in cleaned and cleaned.rindex(",") > cleaned.rindex(".")
    ):
        cleaned = cleaned.replace(".", "").replace(",", ".")
    elif "," in cleaned and "." not in cleaned:
        # Could be "1,234" (thousands) or "1,23" (decimal) — check digit count after comma
        after_comma = cleaned.rsplit(",", 1)[-1]
        cleaned = cleaned.replace(",", "") if len(after_comma) == 3 else cleaned.replace(",", ".")
    elif "." in cleaned and "," not in cleaned:
        after_dot = cleaned.rsplit(".", 1)[-1]
        if len(after_dot) == 3 and cleaned.count(".") == 1:
            # Could be "1.234" as thousands separator — ambiguous; keep as-is
            pass
    try:
        return float(cleaned)
    except ValueError:
        return None


# ── Bbox overlap ──────────────────────────────────────────────────────────────


def _overlap_fraction(a: BBox, b: BBox) -> float:
    """Intersection area / min(area_a, area_b). Returns 0 if no overlap or degenerate bbox."""
    ix1 = max(a.left, b.left)
    iy1 = max(a.top, b.top)
    ix2 = min(a.right, b.right)
    iy2 = min(a.bottom, b.bottom)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, a.right - a.left) * max(0.0, a.bottom - a.top)
    area_b = max(0.0, b.right - b.left) * max(0.0, b.bottom - b.top)
    min_area = min(area_a, area_b)
    return inter / min_area if min_area > 1e-9 else 0.0


# ── Calc verifier ─────────────────────────────────────────────────────────────


@dataclass
class CalcFailure:
    docid: str
    page: int
    net_text: str | None
    tax_text: str | None
    gross_text: str | None
    net_val: float | None
    tax_val: float | None
    gross_val: float | None


def apply_calc_verifier(
    preds: dict[str, list[Field]],
    demote_factor: float = 0.5,
    tolerance: float = _CALC_TOLERANCE,
) -> tuple[dict[str, list[Field]], list[CalcFailure]]:
    """Check amount_total_gross ≈ amount_total_net + amount_total_tax.

    On failure: demote all three fields on that page by demote_factor.
    Returns updated preds and list of CalcFailure records for optional re-prompting.
    Never deletes any field.
    """
    failures: list[CalcFailure] = []
    updated: dict[str, list[Field]] = {}

    for docid, fields in preds.items():
        # Group amount fields by page
        pages_with_amounts: dict[int, dict[str, Field]] = {}
        for f in fields:
            if f.fieldtype in _AMOUNT_FIELDS:
                if f.page not in pages_with_amounts:
                    pages_with_amounts[f.page] = {}
                # Keep highest-score if multiple (shouldn't happen but be safe)
                prev = pages_with_amounts[f.page].get(f.fieldtype)
                if prev is None or f.score > prev.score:
                    pages_with_amounts[f.page][f.fieldtype] = f

        fail_pages: set[int] = set()
        for page, amount_map in pages_with_amounts.items():
            # Need at least two of the three fields to do a meaningful check
            if len(amount_map) < 2:
                continue
            net_f = amount_map.get("amount_total_net")
            tax_f = amount_map.get("amount_total_tax")
            gross_f = amount_map.get("amount_total_gross")

            net_val = parse_amount(net_f.text if net_f else None)
            tax_val = parse_amount(tax_f.text if tax_f else None)
            gross_val = parse_amount(gross_f.text if gross_f else None)

            # Only check when all three are present and parseable
            if net_val is None or tax_val is None or gross_val is None:
                continue

            expected = net_val + tax_val
            denom = max(abs(gross_val), 0.01)
            if abs(gross_val - expected) / denom > tolerance:
                fail_pages.add(page)
                failures.append(
                    CalcFailure(
                        docid=docid,
                        page=page,
                        net_text=net_f.text if net_f else None,
                        tax_text=tax_f.text if tax_f else None,
                        gross_text=gross_f.text if gross_f else None,
                        net_val=net_val,
                        tax_val=tax_val,
                        gross_val=gross_val,
                    )
                )

        if not fail_pages:
            updated[docid] = list(fields)
            continue

        # Demote failing amount fields by demote_factor
        demoted = []
        for f in fields:
            if f.page in fail_pages and f.fieldtype in _AMOUNT_FIELDS:
                demoted.append(
                    Field(
                        bbox=f.bbox,
                        page=f.page,
                        fieldtype=f.fieldtype,
                        score=f.score * demote_factor,
                        line_item_id=f.line_item_id,
                        text=f.text,
                    )
                )
            else:
                demoted.append(f)
        updated[docid] = demoted

    return updated, failures


# ── Overlap verifier ──────────────────────────────────────────────────────────


_OVERLAP_THRESHOLD = 0.5
_MIN_BBOX_AREA = 1e-6  # skip degenerate zero-area bboxes


def apply_overlap_verifier(
    preds: dict[str, list[Field]],
    demote_factor: float = 0.5,
) -> dict[str, list[Field]]:
    """Demote pairs of KILE fields with >50% bbox overlap (by smaller area) * 0.5.

    Only checks KILE fields (line_item_id is None) of DIFFERENT fieldtypes.
    Never deletes. Tie-break when demoting: lower-score field gets the heavier
    demotion (but both get *0.5 — symmetrical treatment).
    """
    updated: dict[str, list[Field]] = {}

    for docid, fields in preds.items():
        kile = [f for f in fields if f.line_item_id is None]
        lir = [f for f in fields if f.line_item_id is not None]

        # Find pairs to demote — track by index
        demote_idx: set[int] = set()
        pages: dict[int, list[tuple[int, Field]]] = {}
        for i, f in enumerate(kile):
            if f.page not in pages:
                pages[f.page] = []
            pages[f.page].append((i, f))

        for page_fields in pages.values():
            n = len(page_fields)
            for a in range(n):
                i, fi = page_fields[a]
                bbox_fi = fi.bbox
                area_fi = max(0.0, bbox_fi.right - bbox_fi.left) * max(
                    0.0, bbox_fi.bottom - bbox_fi.top
                )
                if area_fi < _MIN_BBOX_AREA:
                    continue
                for b in range(a + 1, n):
                    j, fj = page_fields[b]
                    if fi.fieldtype == fj.fieldtype:
                        continue  # same type OK (multi-occurrence like tax_detail_*)
                    bbox_fj = fj.bbox
                    area_fj = max(0.0, bbox_fj.right - bbox_fj.left) * max(
                        0.0, bbox_fj.bottom - bbox_fj.top
                    )
                    if area_fj < _MIN_BBOX_AREA:
                        continue
                    overlap = _overlap_fraction(bbox_fi, bbox_fj)
                    if overlap >= _OVERLAP_THRESHOLD:
                        demote_idx.add(i)
                        demote_idx.add(j)

        new_kile = []
        for i, f in enumerate(kile):
            if i in demote_idx:
                new_kile.append(
                    Field(
                        bbox=f.bbox,
                        page=f.page,
                        fieldtype=f.fieldtype,
                        score=f.score * demote_factor,
                        line_item_id=f.line_item_id,
                        text=f.text,
                    )
                )
            else:
                new_kile.append(f)

        updated[docid] = new_kile + lir

    return updated


# ── Selective re-prompt ───────────────────────────────────────────────────────


_SYSTEM_REPROMPT = """You are a document information extraction assistant for the DocILE benchmark.

Re-extract ONLY these three financial amount fields from the invoice:
- amount_total_net: Total net amount before tax
- amount_total_gross: Total gross amount including tax (must equal net + tax)
- amount_total_tax: Total tax/VAT amount

Use the word ids from the word list. Return ONLY valid JSON, no markdown fences:
{"fields": [
  {"fieldtype": "amount_total_net", "word_ids": [1, 2], "text": "100.00", "score": 0.9},
  {"fieldtype": "amount_total_gross", "word_ids": [3, 4], "text": "121.00", "score": 0.9},
  {"fieldtype": "amount_total_tax", "word_ids": [5], "text": "21.00", "score": 0.9}
]}

Omit any field not present on this page. Verify: net + tax should equal gross.
"""


async def reprompt_amount_fields(
    page,  # PageContext
    model: str,
    failure: CalcFailure,
) -> list[Field]:
    """One targeted API call to re-extract amount_total_* for a calc-failing page.

    Returns new Field objects (potentially empty list on parse failure).
    """
    from .extract import _image_to_b64, _parse_response, _words_to_prompt
    from .vertex import complete

    words_layout = _words_to_prompt(page.words)
    img_b64 = _image_to_b64(page.image)

    context = (
        f"[Previous extraction — math check failed]\n"
        f"  amount_total_net:   {failure.net_text!r} → {failure.net_val}\n"
        f"  amount_total_tax:   {failure.tax_text!r} → {failure.tax_val}\n"
        f"  amount_total_gross: {failure.gross_text!r} → {failure.gross_val}\n"
        f"  Check: {failure.net_val} + {failure.tax_val} = "
        f"{(failure.net_val or 0) + (failure.tax_val or 0):.2f} "
        f"≠ {failure.gross_val} (diff = "
        f"{abs(failure.gross_val - (failure.net_val or 0) - (failure.tax_val or 0)):.2f})\n"
    )

    user_content = [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": img_b64},
        },
        {
            "type": "text",
            "text": (
                f"{context}\n"
                f"[Document words]\n{words_layout}\n\n"
                "[Task]\n"
                "Re-extract the three amount fields. "
                "Select word_ids that exactly cover each numeric value. "
                "Return JSON only."
            ),
        },
    ]

    msg = await complete(
        model=model,
        system=_SYSTEM_REPROMPT,
        messages=[{"role": "user", "content": user_content}],
        max_tokens=1024,
        cache_system=False,
        temperature=1.0,
    )

    raw = msg.content[0].text if msg.content else ""
    kile_fields, _ = _parse_response(raw, page.words, page.page_index)
    return [f for f in kile_fields if f.fieldtype in _AMOUNT_FIELDS]


async def apply_selective_reprompt(
    preds: dict[str, list[Field]],
    failures: list[CalcFailure],
    dataset,  # docile Dataset
    model: str,
) -> dict[str, list[Field]]:
    """For each CalcFailure page, re-prompt and replace if new fields pass calc.

    Loads the page context from dataset. On re-prompt failure or continued
    math fail, keeps the already-demoted (*0.5) original fields.
    """
    if not failures:
        return preds

    from .data import iter_pages

    # Build docid → doc map
    doc_map = {doc.docid: doc for doc in dataset}

    updated = {docid: list(fields) for docid, fields in preds.items()}
    sem = asyncio.Semaphore(4)  # limit concurrent API calls

    async def process_failure(failure: CalcFailure) -> None:
        async with sem:
            doc = doc_map.get(failure.docid)
            if doc is None:
                return

            # Find the page context
            page_ctx = None
            for p in iter_pages(doc):
                if p.page_index == failure.page:
                    page_ctx = p
                    break
            if page_ctx is None:
                return

            try:
                new_fields = await reprompt_amount_fields(page_ctx, model, failure)
            except Exception:
                return  # keep demoted originals on error

            if not new_fields:
                return

            # Check if new fields pass calc
            new_map = {f.fieldtype: f for f in new_fields}
            net_v = parse_amount(
                new_map.get("amount_total_net", Field.__new__(Field)).text
                if "amount_total_net" in new_map
                else None
            )
            tax_v = parse_amount(
                new_map.get("amount_total_tax", Field.__new__(Field)).text
                if "amount_total_tax" in new_map
                else None
            )
            gross_v = parse_amount(
                new_map.get("amount_total_gross", Field.__new__(Field)).text
                if "amount_total_gross" in new_map
                else None
            )

            # Parse cleanly
            net_v = (
                parse_amount(new_map["amount_total_net"].text)
                if "amount_total_net" in new_map
                else None
            )
            tax_v = (
                parse_amount(new_map["amount_total_tax"].text)
                if "amount_total_tax" in new_map
                else None
            )
            gross_v = (
                parse_amount(new_map["amount_total_gross"].text)
                if "amount_total_gross" in new_map
                else None
            )

            passes_calc = False
            if net_v is not None and tax_v is not None and gross_v is not None:
                expected = net_v + tax_v
                denom = max(abs(gross_v), 0.01)
                passes_calc = abs(gross_v - expected) / denom <= _CALC_TOLERANCE

            if not passes_calc:
                return  # new fields also fail — keep demoted originals

            # Replace demoted originals with new (full-score) fields on this page
            old_fields = updated[failure.docid]
            kept = [
                f
                for f in old_fields
                if not (f.page == failure.page and f.fieldtype in _AMOUNT_FIELDS)
            ]
            updated[failure.docid] = kept + new_fields

    await asyncio.gather(*[process_failure(f) for f in failures])
    return updated


# ── Top-level orchestrator ────────────────────────────────────────────────────


async def apply_aol_verifiers(
    preds: dict[str, list[Field]],
    model: str,
    dataset=None,
    do_reprompt: bool = True,
    demote_factor: float = 0.5,
    calc_tolerance: float = _CALC_TOLERANCE,
) -> dict[str, list[Field]]:
    """Apply both AOL verifier passes to existing predictions.

    1. Calc verifier (pure Python, no API)
    2. Overlap verifier (pure Python, no API)
    3. Selective re-prompt (API, only for calc-failing pages, if dataset provided)

    Args:
        preds: Existing {docid: [Field]} predictions (e.g. v2_ensemble).
        model: Model name for re-prompting.
        dataset: docile Dataset with OCR loaded (needed for re-prompting images/words).
        do_reprompt: If False, skip API re-prompting (just demote).
    """
    print(f"  [AOL] Running calc verifier (demote={demote_factor}, tol={calc_tolerance:.0%})...")
    preds, failures = apply_calc_verifier(
        preds, demote_factor=demote_factor, tolerance=calc_tolerance
    )
    print(
        f"  [AOL] Calc failures: {len(failures)} pages across "
        f"{len({f.docid for f in failures})} docs — demoted *{demote_factor}"
    )

    print("  [AOL] Running overlap verifier...")
    preds = apply_overlap_verifier(preds, demote_factor=demote_factor)
    print("  [AOL] Overlap verifier complete.")

    if do_reprompt and failures and dataset is not None:
        print(f"  [AOL] Re-prompting {len(failures)} failing pages...")
        preds = await apply_selective_reprompt(preds, failures, dataset, model)
        print("  [AOL] Re-prompting complete.")
    elif failures and not do_reprompt:
        print("  [AOL] Re-prompting skipped (do_reprompt=False).")

    return preds
