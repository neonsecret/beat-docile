"""[ARCHIVED] End-to-end V6 extraction pipeline: ReAct + Haiku verify + classifiers.

Status: ARCHIVED — 22.7% KILE (catastrophic regression). See KNOWLEDGE_BASE.md §6.7.
Root cause: triage gate dropped 48% of fields; cross-field verifier deleted whole
field arrays. Both gates were delete-skewed. Kept for code-archaeology only.

Original design: integration layer wiring Phase A (tools/react_extract), Phase B
(classifiers), and Phase F (haiku_verify) with batch runner and evaluator.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from docile.dataset import Dataset, Field
from docile.evaluation import evaluate_dataset

from .data import WordBox, iter_pages
from .haiku_verify import apply_verification, verify_extractions_batch
from .react_extract import extract_page_react
from .tools import Candidate
from .vertex import get_client

logger = logging.getLogger(__name__)

_DATA_PATH = Path("data")
_MODEL_DIR = Path("models/classifiers")


# ── Haiku verifier wrapper ────────────────────────────────────────────────────


def _build_haiku_verifier(
    words: list[WordBox],
    vertex_client: Any,
) -> Callable[[str, list[Candidate]], list[Candidate]]:
    """Build the candidate_verifier callback for extract_page_react.

    The returned closure converts list[Candidate] → haiku_verify input, calls
    verify_extractions_batch + apply_verification, then reconstructs Candidate
    objects preserving all metadata (score, source, reason, line_item_id).

    Args:
        words: Full page word list (closure context for verify_span calls).
        vertex_client: AnthropicVertex client instance.

    Returns:
        Callable with signature (fieldtype, candidates) -> candidates.
    """

    def verifier(fieldtype: str, candidates: list[Candidate]) -> list[Candidate]:
        if not candidates:
            return candidates

        extractions: dict[str, list[tuple[list[int], str]]] = {
            fieldtype: [(c.word_ids, c.text) for c in candidates]
        }
        verifications = verify_extractions_batch(extractions, words, vertex_client, parallel=8)
        # Call apply_verification per spec (authoritative side-effect check)
        apply_verification(extractions, verifications, min_confidence=0.6)

        # Rebuild Candidates from VerificationResult, preserving metadata
        results_list = verifications.get(fieldtype, [])
        output: list[Candidate] = []
        for i, cand in enumerate(candidates):
            if i >= len(results_list):
                output.append(cand)
                continue
            vr = results_list[i]
            if vr.verdict == "accept":
                output.append(cand)
            elif vr.verdict == "correct":
                if vr.confidence >= 0.6:
                    output.append(dataclasses.replace(cand, word_ids=vr.word_ids))
                else:
                    output.append(cand)
            else:  # reject
                if vr.confidence < 0.6:
                    output.append(cand)
                # confidence >= 0.6 → candidate dropped

        return output

    return verifier


# ── Train doc loader ──────────────────────────────────────────────────────────


def _load_train_docs(data_path: Path = _DATA_PATH) -> dict:
    """Load train annotation data for cluster-based few-shot lookup.

    Reads data_path/train.json for docids, then parses data_path/annotations/*.json
    to build {docid: {cluster_id: int, fields: [{fieldtype, text}]}} — the exact
    shape expected by tools.cluster_fewshot.

    Args:
        data_path: Root data directory containing train.json + annotations/.

    Returns:
        Dict {docid: {cluster_id, fields}} for all train docs with annotation files.
    """
    train_docids_path = data_path / "train.json"
    if not train_docids_path.exists():
        logger.warning("train.json not found at %s — few-shot disabled", data_path)
        return {}

    train_docids: list[str] = json.loads(train_docids_path.read_text())
    ann_dir = data_path / "annotations"
    result: dict = {}

    for docid in train_docids:
        ann_path = ann_dir / f"{docid}.json"
        if not ann_path.exists():
            continue
        try:
            ann_data = json.loads(ann_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Skip annotation %s: %s", docid, exc)
            continue

        cluster_id = ann_data.get("metadata", {}).get("cluster_id")
        fields: list[dict] = [
            {"fieldtype": item["fieldtype"], "text": item.get("text") or ""}
            for item in ann_data.get("field_extractions", [])
            if "fieldtype" in item
        ]
        fields.extend(
            {"fieldtype": item["fieldtype"], "text": item.get("text") or ""}
            for item in ann_data.get("line_item_extractions", [])
            if "fieldtype" in item
        )
        result[docid] = {"cluster_id": cluster_id, "fields": fields}

    logger.info("Loaded %d train doc annotations for few-shot", len(result))
    return result


# ── Serialization ─────────────────────────────────────────────────────────────


def _field_to_dict(f: Field) -> dict:
    """Serialize a docile Field to the prediction JSON schema.

    Output matches the v5b_50.json field structure:
      {bbox, page, score, text, fieldtype, line_item_id, use_only_for_ap}.
    """
    return {
        "bbox": [f.bbox.left, f.bbox.top, f.bbox.right, f.bbox.bottom],
        "page": f.page,
        "score": f.score,
        "text": f.text,
        "fieldtype": f.fieldtype,
        "line_item_id": f.line_item_id,
        "use_only_for_ap": False,
    }


# ── Document-level extraction ─────────────────────────────────────────────────


def extract_document_v6(
    docid: str,
    doc: Any,
    train_docs: dict,
    vertex_client: Any,
    use_haiku_verify: bool = True,
    use_classifier_tool: bool = True,
    max_steps_per_field: int = 8,
) -> dict:
    """End-to-end V6 extraction for one document.

    Pipeline:
      1. For each page: load words via iter_pages (snapped OCR, correct for PCC-IoU).
      2. For each page: extract_page_react with optional haiku_verify wrapper.
      3. Concatenate per-page results into (kile_fields, lir_fields).
      4. Return dict {"docid": docid, "fields": [...]} matching v5b_50.json schema.

    Output schema (per field):
      {"fieldtype": str, "bbox": [l,t,r,b], "page": int, "score": float,
       "line_item_id": int|None, "text": str|None, "use_only_for_ap": False}

    Args:
        docid: Document identifier string.
        doc: docile.dataset.Document instance (loaded with annotations + OCR).
        train_docs: Pre-loaded dict {docid: {cluster_id, fields}} for few-shot.
        vertex_client: AnthropicVertex client instance.
        use_haiku_verify: If True, run Haiku bbox-precision verifier per field.
        use_classifier_tool: If True, expose classifier_score tool to Claude.
        max_steps_per_field: Hard cap on ReAct steps per field extraction.

    Returns:
        Dict {"docid": docid, "fields": [field_dict, ...]}.
    """
    model_dir: Path | None = _MODEL_DIR if use_classifier_tool else None
    cluster_id: int | None = doc.annotation.cluster_id

    kile_fields_all: list[Field] = []
    lir_fields_all: list[Field] = []

    for page_ctx in iter_pages(doc):
        candidate_verifier = (
            _build_haiku_verifier(page_ctx.words, vertex_client) if use_haiku_verify else None
        )
        kile_fields, lir_fields = extract_page_react(
            words=page_ctx.words,
            image=page_ctx.image,
            cluster_id=cluster_id,
            train_docs=train_docs,
            vertex_client=vertex_client,
            candidate_verifier=candidate_verifier,
            model_dir=model_dir,
            max_steps_per_field=max_steps_per_field,
        )
        kile_fields_all.extend(kile_fields)
        lir_fields_all.extend(lir_fields)

    fields_out = [_field_to_dict(f) for f in kile_fields_all + lir_fields_all]
    logger.info("extract_document_v6 %s: %d fields total", docid, len(fields_out))
    return {"docid": docid, "fields": fields_out}


# ── Batch runner ──────────────────────────────────────────────────────────────


def run_v6_on_docids(
    docids: list[str],
    output_path: Path,
    use_haiku_verify: bool = True,
    use_classifier_tool: bool = True,
    progress: bool = True,
    data_path: Path = _DATA_PATH,
    max_workers: int = 8,
) -> dict[str, list[dict]]:
    """Batch driver. Loads dataset, processes docs in parallel via ThreadPoolExecutor,
    writes consolidated JSON to output_path. Same schema as predictions/v5b_50.json.

    Args:
        docids: List of DocILE document IDs to process.
        output_path: Destination JSON path.
        use_haiku_verify: Enable Haiku bbox-precision verifier.
        use_classifier_tool: Expose classifier_score tool to Claude.
        progress: Display rich progress bar.
        data_path: Root data directory (must contain annotations/ and OCR/).
        max_workers: Number of parallel doc-processing threads (default 8).

    Returns:
        Dict {docid: [field_dict, ...]} (same object written to output_path).
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    vertex_client = get_client()
    train_docs = _load_train_docs(data_path)

    val_ds = Dataset(
        "smoke_subset",
        str(data_path),
        load_annotations=True,
        load_ocr=True,
        docids=docids,
    )

    results: dict[str, list[dict]] = {docid: [] for docid in docids}
    results_lock = threading.Lock()
    docs_list: list[Any] = list(val_ds)
    completed = 0

    def _process_doc(doc: Any) -> tuple[str, list[dict]]:
        try:
            result = extract_document_v6(
                docid=doc.docid,
                doc=doc,
                train_docs=train_docs,
                vertex_client=vertex_client,
                use_haiku_verify=use_haiku_verify,
                use_classifier_tool=use_classifier_tool,
            )
            return doc.docid, result["fields"]
        except Exception as exc:
            logger.error("Failed to extract doc %s: %s", doc.docid, exc)
            return doc.docid, []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_doc = {executor.submit(_process_doc, doc): doc for doc in docs_list}
        iterable = as_completed(future_to_doc)
        if progress:
            try:
                from rich.progress import track

                iterable = track(
                    as_completed(future_to_doc), total=len(docs_list), description="V6 extraction…"
                )
            except ImportError:
                pass
        for future in iterable:
            docid, fields = future.result()
            with results_lock:
                results[docid] = fields
                completed += 1
                if completed % 5 == 0 or completed == len(docs_list):
                    logger.info("Progress: %d/%d docs done", completed, len(docs_list))

    output_path.write_text(json.dumps(results, indent=2))
    logger.info("Wrote %d doc predictions to %s", len(results), output_path)
    return results


# ── Evaluator ─────────────────────────────────────────────────────────────────


def evaluate_v6(
    predictions_path: Path,
    data_path: Path = _DATA_PATH,
) -> dict:
    """Wrap docile.evaluation.evaluate_dataset; return dict with primary metrics.

    Reuses the pattern from /tmp/eval_ablations.py. Loads predictions JSON,
    builds KILE/LIR split, runs official evaluator, returns scalar metrics.

    Args:
        predictions_path: Path to predictions JSON (format: {docid: [field_dict]}).
        data_path: Root data directory for Dataset construction.

    Returns:
        Dict with keys: kile_ap, kile_p, kile_r, lir_f1, lir_p, lir_r.
    """
    raw = json.loads(predictions_path.read_text())
    docids = list(raw.keys())

    ds = Dataset(
        "smoke_subset",
        str(data_path),
        load_annotations=True,
        load_ocr=False,
        docids=docids,
    )

    kile_preds: dict[str, list[Field]] = {}
    lir_preds: dict[str, list[Field]] = {}

    for docid, fields in raw.items():
        kile_preds[docid] = []
        lir_preds[docid] = []
        for fd in fields:
            f = Field.from_dict(fd)
            if f.line_item_id is not None:
                lir_preds[docid].append(f)
            else:
                kile_preds[docid].append(f)

    for doc in ds:
        kile_preds.setdefault(doc.docid, [])
        lir_preds.setdefault(doc.docid, [])

    result = evaluate_dataset(ds, kile_preds, lir_preds)
    mk = result.get_metrics("kile")
    ml = result.get_metrics("lir")

    metrics = {
        "kile_ap": mk.get("AP", 0.0),
        "kile_p": mk.get("precision", 0.0),
        "kile_r": mk.get("recall", 0.0),
        "lir_f1": ml.get("f1", 0.0),
        "lir_p": ml.get("precision", 0.0),
        "lir_r": ml.get("recall", 0.0),
    }
    logger.info(
        "evaluate_v6: KILE AP=%.4f  LIR F1=%.4f",
        metrics["kile_ap"],
        metrics["lir_f1"],
    )
    return metrics
