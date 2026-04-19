"""Field-level debug runner for V6 ReAct pipeline on 5 worst-performing docs.

Re-runs V6 on the 5 target docs sequentially with monkey-patched instrumentation
to capture: triage output, per-field ReAct candidates, Haiku verify verdicts,
and cross-field verifier removals. Produces per-doc JSON traces in
failure_analysis_traces/{docid}.json and classifies each V5b-vs-V6 miss.

Usage: uv run python tools/v6_debug_run.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project root on sys.path so src/ imports work
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Target documents and paths
# ---------------------------------------------------------------------------
TARGET_DOCS: list[str] = [
    "00c87916e4a44197b45b0f8b",
    "01628ff7c56f4b1995c3048e",
    "07f1cdc6b0384ebaa9c73e1d",
    "15dd967792504e9e9aa4ba99",
    "0cfdc8b6d1d04849b34026ba",
]

V5B_PATH = _REPO_ROOT / "predictions" / "v5b_50.json"
OUTPUT_DIR = _REPO_ROOT / "failure_analysis_traces"
DATA_PATH = _REPO_ROOT / "data"

# ---------------------------------------------------------------------------
# Per-run capture state (reset per doc)
# ---------------------------------------------------------------------------
_capture: dict[str, Any] = {}


def _reset_capture(docid: str) -> None:
    _capture.clear()
    _capture["docid"] = docid
    _capture["pages"] = []


def _current_page() -> dict:
    """Return or create the capture dict for the current page."""
    if not _capture["pages"]:
        _capture["pages"].append({"page_idx": 0, "triage": [], "fields": {}, "verifier_removals": []})
    return _capture["pages"][-1]


def _new_page() -> None:
    idx = len(_capture["pages"])
    _capture["pages"].append({"page_idx": idx, "triage": [], "fields": {}, "verifier_removals": []})


# ---------------------------------------------------------------------------
# Monkey-patch wrappers
# ---------------------------------------------------------------------------

def _make_wrapped_triage(original_fn: Any) -> Any:
    def wrapped(words: Any, image: Any, vertex_client: Any) -> list[str]:
        result = original_fn(words, image, vertex_client)
        page = _current_page()
        page["triage"] = list(result)
        logger.info(
            "[DEBUG] triage for page %d: %s",
            page["page_idx"],
            ", ".join(result[:15]) + ("..." if len(result) > 15 else ""),
        )
        return result
    return wrapped


def _make_wrapped_extract_field_react(original_fn: Any) -> Any:
    def wrapped(
        fieldtype: str,
        words: Any,
        image: Any,
        cluster_id: Any,
        train_docs: Any,
        vertex_client: Any,
        max_steps: int = 8,
        model_dir: Any = None,
    ) -> list[Any]:
        candidates = original_fn(
            fieldtype=fieldtype,
            words=words,
            image=image,
            cluster_id=cluster_id,
            train_docs=train_docs,
            vertex_client=vertex_client,
            max_steps=max_steps,
            model_dir=model_dir,
        )
        page = _current_page()
        page["fields"].setdefault(fieldtype, {})["react_candidates"] = [
            {"word_ids": c.word_ids, "text": c.text, "score": c.score, "reason": c.reason}
            for c in candidates
        ]
        logger.info(
            "[DEBUG] react %s → %d candidates: %s",
            fieldtype,
            len(candidates),
            [c.text[:30] for c in candidates[:3]],
        )
        return candidates
    return wrapped


def _make_wrapped_verify_extractions(original_fn: Any) -> Any:
    def wrapped(
        extractions: dict[str, list[Any]],
        words: Any,
        image: Any,
        vertex_client: Any,
    ) -> dict[str, list[Any]]:
        before_keys = {ft for ft, cs in extractions.items() if cs}
        result = original_fn(extractions, words, image, vertex_client)
        after_keys = {ft for ft, cs in result.items() if cs}
        removed = before_keys - after_keys
        page = _current_page()
        page["verifier_removals"] = list(removed)
        if removed:
            logger.info("[DEBUG] cross-field verifier removed: %s", sorted(removed))
        return result
    return wrapped


def _make_wrapped_verify_span(original_fn: Any) -> Any:
    def wrapped(
        fieldtype: str,
        candidate_word_ids: list[int],
        candidate_text: str,
        words: Any,
        vertex_client: Any,
        model: str = "claude-haiku-4-5",
    ) -> Any:
        result = original_fn(fieldtype, candidate_word_ids, candidate_text, words, vertex_client, model)
        page = _current_page()
        field_data = page["fields"].setdefault(fieldtype, {})
        verdicts = field_data.setdefault("haiku_verdicts", [])
        verdicts.append({
            "candidate_word_ids": candidate_word_ids,
            "candidate_text": candidate_text[:60],
            "verdict": result.verdict,
            "confidence": result.confidence,
            "corrected_word_ids": result.word_ids if result.verdict != "accept" else None,
            "reasoning": result.reasoning,
        })
        logger.info(
            "[DEBUG] haiku_verify %s → %s (conf=%.2f): %s",
            fieldtype,
            result.verdict,
            result.confidence,
            result.reasoning[:80],
        )
        return result
    return wrapped


# ---------------------------------------------------------------------------
# Failure-mode classification
# ---------------------------------------------------------------------------

_FAILURE_MODES = (
    "triage_skipped",
    "react_returned_empty",
    "react_returned_wrong",
    "haiku_rejected",
    "cross_field_verifier_removed",
    "correct_v5b_was_wrong",
    "match",
)


def _classify_miss(
    fieldtype: str,
    v5b_instances: list[dict],
    page_traces: list[dict],
) -> tuple[str, str]:
    """Return (failure_mode, evidence_str) for a fieldtype V5b had but V6 missed."""
    # Flatten all pages
    triage_fields_all: set[str] = set()
    react_candidates: list[dict] = []
    haiku_verdicts: list[dict] = []
    verifier_removed_all: set[str] = set()

    for page in page_traces:
        triage_fields_all.update(page.get("triage", []))
        fd = page.get("fields", {}).get(fieldtype, {})
        react_candidates.extend(fd.get("react_candidates", []))
        haiku_verdicts.extend(fd.get("haiku_verdicts", []))
        if fieldtype in page.get("verifier_removals", []):
            verifier_removed_all.add(fieldtype)

    # 1. Triage never listed it
    if fieldtype not in triage_fields_all:
        return (
            "triage_skipped",
            f"triage output across {len(page_traces)} pages never included '{fieldtype}'. "
            f"Sample triage: {sorted(list(triage_fields_all))[:8]}",
        )

    # 2. ReAct returned no candidates
    if not react_candidates:
        return (
            "react_returned_empty",
            f"triage included '{fieldtype}' but ReAct emitted 0 candidates",
        )

    # 3. Cross-field verifier removed the whole field
    if fieldtype in verifier_removed_all:
        return (
            "cross_field_verifier_removed",
            f"cross-field verifier removed '{fieldtype}' after ReAct returned {len(react_candidates)} candidates",
        )

    # 4. Haiku rejected all candidates
    if haiku_verdicts:
        all_rejected = all(
            v["verdict"] == "reject" and v["confidence"] >= 0.6
            for v in haiku_verdicts
        )
        if all_rejected:
            sample = haiku_verdicts[0]
            return (
                "haiku_rejected",
                f"Haiku rejected all {len(haiku_verdicts)} candidates (conf >= 0.6). "
                f"Sample: verdict={sample['verdict']}, text='{sample['candidate_text'][:30]}', "
                f"reason='{sample['reasoning'][:60]}'",
            )

    # 5. ReAct returned something but bbox doesn't match V5b
    if react_candidates:
        return (
            "react_returned_wrong",
            f"ReAct returned {len(react_candidates)} candidate(s) but none matched V5b gold. "
            f"Sample candidate text: '{react_candidates[0]['text'][:40]}'",
        )

    return ("react_returned_empty", "no candidates at any pipeline stage")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_debug_on_doc(
    docid: str,
    v5b_data: dict[str, list[dict]],
    vertex_client: Any,
    train_docs: dict,
) -> dict:
    """Instrument and run V6 on one doc; return structured trace + classification."""
    from docile.dataset import Dataset
    from beat_docile.data import iter_pages
    from beat_docile.v6_pipeline import _build_haiku_verifier
    from beat_docile.extract import _KILE_TYPES, _LIR_TYPES
    import beat_docile.react_extract as react_mod

    _reset_capture(docid)

    val_ds = Dataset(
        "debug_subset",
        str(DATA_PATH),
        load_annotations=True,
        load_ocr=True,
        docids=[docid],
    )
    docs = list(val_ds)
    if not docs:
        logger.error("Doc %s not found in dataset", docid)
        return {"docid": docid, "error": "not_found"}

    doc = docs[0]
    cluster_id = doc.annotation.cluster_id
    all_fields: list[dict] = []

    for page_idx, page_ctx in enumerate(iter_pages(doc)):
        if page_idx > 0:
            _new_page()

        candidate_verifier = _build_haiku_verifier(page_ctx.words, vertex_client)

        # Use extract_page_react (already patched via module-level monkeypatching)
        kile_fields, lir_fields = react_mod.extract_page_react(
            words=page_ctx.words,
            image=page_ctx.image,
            cluster_id=cluster_id,
            train_docs=train_docs,
            vertex_client=vertex_client,
            candidate_verifier=candidate_verifier,
            model_dir=None,
            max_steps_per_field=8,
        )
        for f in kile_fields + lir_fields:
            all_fields.append({
                "fieldtype": f.fieldtype,
                "bbox": [f.bbox.left, f.bbox.top, f.bbox.right, f.bbox.bottom],
                "page": f.page,
                "text": f.text,
                "score": f.score,
            })

    logger.info("[DEBUG] doc %s: %d total fields extracted", docid, len(all_fields))

    # --- Classify misses vs V5b ---
    v5b_fields = v5b_data.get(docid, [])
    v5b_by_type: dict[str, list[dict]] = {}
    for f in v5b_fields:
        v5b_by_type.setdefault(f["fieldtype"], []).append(f)

    v6_types = {f["fieldtype"] for f in all_fields}
    page_traces = _capture["pages"]

    miss_rows: list[dict] = []
    for fieldtype, instances in sorted(v5b_by_type.items()):
        if fieldtype in v6_types:
            miss_rows.append({
                "fieldtype": fieldtype,
                "v5b_emitted": True,
                "v6_emitted": True,
                "failure_mode": "match",
                "evidence": f"both emitted ({len(instances)} V5b instances)",
            })
        else:
            mode, evidence = _classify_miss(fieldtype, instances, page_traces)
            miss_rows.append({
                "fieldtype": fieldtype,
                "v5b_emitted": True,
                "v6_emitted": False,
                "failure_mode": mode,
                "evidence": evidence,
            })

    trace = {
        "docid": docid,
        "v5b_field_count": len(v5b_fields),
        "v6_field_count": len(all_fields),
        "pages": page_traces,
        "miss_classification": miss_rows,
        "v6_fields": all_fields,
    }
    return trace


def main() -> None:
    logger.info("Loading V5b predictions from %s", V5B_PATH)
    v5b_data: dict[str, list[dict]] = json.loads(V5B_PATH.read_text())

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Apply monkey patches BEFORE importing anything that triggers pipeline code
    import beat_docile.react_extract as react_mod
    import beat_docile.haiku_verify as haiku_mod

    react_mod.triage_fields = _make_wrapped_triage(react_mod.triage_fields)
    react_mod.extract_field_react = _make_wrapped_extract_field_react(react_mod.extract_field_react)
    react_mod.verify_extractions = _make_wrapped_verify_extractions(react_mod.verify_extractions)
    haiku_mod.verify_span = _make_wrapped_verify_span(haiku_mod.verify_span)

    logger.info("Monkey-patches applied: triage, extract_field_react, verify_extractions, verify_span")

    from beat_docile.vertex import get_client
    from beat_docile.v6_pipeline import _load_train_docs

    vertex_client = get_client()
    train_docs = _load_train_docs(DATA_PATH)

    for docid in TARGET_DOCS:
        logger.info("=" * 60)
        logger.info("Running doc: %s", docid)
        logger.info("=" * 60)
        try:
            trace = run_debug_on_doc(docid, v5b_data, vertex_client, train_docs)
        except Exception as exc:
            logger.error("Doc %s failed: %s", docid, exc, exc_info=True)
            trace = {"docid": docid, "error": str(exc)}

        out_path = OUTPUT_DIR / f"{docid}.json"
        out_path.write_text(json.dumps(trace, indent=2, default=str))
        logger.info("Trace written: %s", out_path)

    logger.info("All 5 docs complete. Traces in %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
