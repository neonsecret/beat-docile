#!/usr/bin/env python
"""Build-time Code-Factory loop: generate extraction scripts per cluster.

For each cluster:
  1. Load up to N_TRAIN_DOCS training docs from the cluster
  2. Render their page-0 word layouts + gold annotations
  3. Call Sonnet to write a Python extraction script
  4. Run script on train docs, measure recall
  5. If recall < TARGET_RECALL: iterate (max MAX_ITERS times)
  6. Cache best script + metadata

Usage:
    uv run python tools/build_cluster_scripts.py --cluster 42
    uv run python tools/build_cluster_scripts.py --clusters 1,5,42,99
    uv run python tools/build_cluster_scripts.py --all
    uv run python tools/build_cluster_scripts.py --all --skip-built
    uv run python tools/build_cluster_scripts.py --tight-only  # only TIGHT clusters

Environment: VERTEX_PROJECT_ID, VERTEX_LOCATION, DATA_ROOT (or .env.local)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import os
os.environ.setdefault("DATA_ROOT", str(PROJECT_ROOT / "data"))

from anthropic import AnthropicVertex
from docile.dataset import Dataset

from beat_docile.config import DATA_ROOT, VERTEX_PROJECT_ID, VERTEX_LOCATION, DEFAULT_MODEL
from beat_docile.data import WordBox, iter_pages
from beat_docile.extract import _KILE_TYPES, _LIR_TYPES
from beat_docile.fewshot import _build_cluster_index
from beat_docile.code_factory import (
    SCRIPTS_DIR, METADATA_PATH, script_path, has_script,
    load_metadata, save_metadata,
)

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL = "claude-sonnet-4-6"  # NEVER Haiku for script generation
N_TRAIN_DOCS = 5             # max train docs per cluster for few-shot
MAX_ITERS = 5                # max revision iterations per cluster
TARGET_RECALL = 0.80         # stop iterating when recall >= this
MAX_SCRIPT_TOKENS = 6000     # generous budget — complex invoices need 200-400 lines
KILE_ONLY = True             # focus scripts on KILE fields only; LIR stays with V5b

VALID_KILE = sorted(_KILE_TYPES)
VALID_LIR = sorted(_LIR_TYPES)
VALID_FIELDS_STR = "KILE field types (ONLY these — no LIR): " + ", ".join(VALID_KILE)

_SYSTEM = f"""You are a Python code generator for invoice document field extraction.

Write `def extract(words: list) -> list:` that extracts KILE fields from invoice OCR words.

WordBox attributes:
- .id: int — word identifier
- .text: str — word text
- .bbox: tuple — (left, top, right, bottom) normalized [0,1]

Return: list of dicts, each with:
- "fieldtype": str — from the allowed list ONLY
- "word_ids": list[int] — IDs of words forming this field's value

{VALID_FIELDS_STR}

`re` is available. No other imports needed.

Rules:
1. Function must be complete and syntactically valid Python
2. Use text patterns (re.search) and bbox positions to identify fields
3. Singleton fields: return at most one entry per fieldtype
4. Return only confident extractions
5. word_ids must reference actual IDs from the words list
6. Keep the function concise — use loops, not one block per field
"""


# ── Word layout and gold annotation formatting ─────────────────────────────────

def _words_layout_for_prompt(words: list[WordBox]) -> str:
    """Compact word table for the build prompt."""
    lines = ["id  text                   left   top    right  bot"]
    for w in words:
        l, t, r, b = w.bbox
        lines.append(f"{w.id:<4} {w.text:<22} {l:.3f}  {t:.3f}  {r:.3f}  {b:.3f}")
    return "\n".join(lines)


def _gold_to_word_ids(field, words: list[WordBox]) -> set[int]:
    """Find word IDs whose centers fall within the gold field bbox on the same page."""
    fb = field.bbox
    result: set[int] = set()
    for w in words:
        if w.page != field.page:
            continue
        cx = (w.bbox[0] + w.bbox[2]) / 2
        cy = (w.bbox[1] + w.bbox[3]) / 2
        if fb.left <= cx <= fb.right and fb.top <= cy <= fb.bottom:
            result.add(w.id)
    return result


def _format_gold(gold_fields: list, gold_li_fields: list, words: list[WordBox]) -> str:
    """Format gold KILE annotations only (LIR omitted — keeps examples concise)."""
    lines: list[str] = []
    for f in gold_fields:
        wids = _gold_to_word_ids(f, words)
        words_text = " ".join(w.text for w in words if w.id in wids)
        lines.append(f"  {f.fieldtype}: word_ids={sorted(wids)} | text={words_text!r}")
    return "\n".join(lines) if lines else "  (no KILE fields annotated)"


# ── Train example loading ──────────────────────────────────────────────────────

def load_train_examples(
    cluster_id: int,
    train_index: dict[int, list[str]],
    n: int = N_TRAIN_DOCS,
) -> list[dict[str, Any]]:
    """Load up to n training documents from the cluster with words + gold."""
    docids = train_index.get(cluster_id, [])[:n]
    if not docids:
        return []

    train_ds = Dataset("train", DATA_ROOT, load_annotations=True, load_ocr=False)
    doc_lookup = {doc.docid: doc for doc in train_ds if doc.docid in docids}

    examples: list[dict[str, Any]] = []
    for docid in docids:
        doc = doc_lookup.get(docid)
        if doc is None:
            continue
        try:
            with doc:
                pages = list(iter_pages(doc))
                if not pages:
                    continue

                # Use page 0 words only (where most KILE fields appear)
                page = pages[0]
                words = page.words

                gold_fields = list(doc.annotation.fields)
                gold_li = list(doc.annotation.li_fields)

                examples.append({
                    "docid": docid,
                    "words": words,
                    "gold_fields": gold_fields,
                    "gold_li": gold_li,
                    "gold": _compute_gold_wids(gold_fields, gold_li, words),
                })
        except Exception as e:
            print(f"  Warning: couldn't load {docid}: {e}")

    return examples


def _compute_gold_wids(
    gold_fields: list,
    gold_li: list,
    words: list[WordBox],
) -> list[dict[str, Any]]:
    """Compute gold word IDs for KILE fields only (LIR excluded — scripts focus on KILE)."""
    result: list[dict[str, Any]] = []
    for f in gold_fields:
        wids = _gold_to_word_ids(f, words)
        result.append({"fieldtype": f.fieldtype, "word_ids": wids, "line_item_id": None})
    return result


# ── Recall measurement ─────────────────────────────────────────────────────────

def measure_recall(
    script_code: str,
    examples: list[dict[str, Any]],
) -> tuple[float, list[str]]:
    """Execute script on train examples, return (recall, failure_messages).

    A gold field is "hit" if a predicted field has the same fieldtype AND
    at least one word_id overlapping with the gold word_ids set.
    """
    sandbox: dict[str, Any] = {"re": re, "WordBox": WordBox}
    try:
        exec(compile(script_code, "<script>", "exec"), sandbox)  # noqa: S102
    except SyntaxError as e:
        return 0.0, [f"SyntaxError: {e}"]
    except Exception as e:
        return 0.0, [f"Compile error: {e}"]

    extract_fn = sandbox.get("extract")
    if not callable(extract_fn):
        return 0.0, ["No callable 'extract' function defined"]

    total_gold = 0
    total_hit = 0
    failures: list[str] = []

    for ex in examples:
        words = ex["words"]
        gold = ex["gold"]
        docid = ex["docid"]

        try:
            predicted = extract_fn(words)
        except Exception as e:
            failures.append(f"{docid}: runtime error: {e}")
            total_gold += len([g for g in gold if g["word_ids"]])
            continue

        if not isinstance(predicted, list):
            failures.append(f"{docid}: extract() returned {type(predicted).__name__}, expected list")
            total_gold += len([g for g in gold if g["word_ids"]])
            continue

        # Build predicted lookup: fieldtype → list of word_id sets
        pred_by_type: dict[str, list[set[int]]] = {}
        for item in predicted:
            if not isinstance(item, dict):
                continue
            ft = item.get("fieldtype", "")
            wids = set(item.get("word_ids") or [])
            pred_by_type.setdefault(ft, []).append(wids)

        for gold_item in gold:
            gold_wids = gold_item["word_ids"]
            if not gold_wids:
                # Gold field has no words (bbox misalignment) — skip
                continue
            ft = gold_item["fieldtype"]
            total_gold += 1

            pred_sets = pred_by_type.get(ft, [])
            hit = any(len(p & gold_wids) > 0 for p in pred_sets)
            if hit:
                total_hit += 1
            else:
                sample_words = [w.text for w in words if w.id in gold_wids][:5]
                failures.append(
                    f"{docid}: missed {ft} | expected words: {sample_words}"
                )

    recall = total_hit / total_gold if total_gold > 0 else 0.0
    return recall, failures


# ── Prompt building ────────────────────────────────────────────────────────────

def _build_initial_user_message(
    cluster_id: int,
    examples: list[dict[str, Any]],
) -> list[dict]:
    """Build the user message content for the first iteration.

    The examples block gets cache_control so it is cached across iterations.
    """
    examples_text_parts = [f"Cluster {cluster_id} — {len(examples)} training examples:\n"]

    for i, ex in enumerate(examples, 1):
        words = ex["words"]
        gold_fields = ex["gold_fields"]
        gold_li = ex["gold_li"]
        layout = _words_layout_for_prompt(words)
        gold_str = _format_gold(gold_fields, gold_li, words)
        examples_text_parts.append(
            f"\n[EXAMPLE {i}/{len(examples)} — doc {ex['docid']}]\n"
            f"Words (page 0):\n{layout}\n\n"
            f"Gold extractions:\n{gold_str}\n"
        )

    examples_text_parts.append(
        "\nWrite `def extract(words: list) -> list:` that reproduces these extractions.\n"
        "Study the text patterns and bbox positions carefully.\n"
        "Return ONLY the Python function. No explanation. No markdown fences."
    )

    return [
        {
            "type": "text",
            "text": "".join(examples_text_parts),
            "cache_control": {"type": "ephemeral"},
        }
    ]


def _build_revision_message(
    recall: float,
    total_gold: int,
    failures: list[str],
) -> str:
    hit = round(recall * total_gold)
    failure_sample = failures[:20]  # cap to avoid huge prompts
    failure_str = "\n".join(f"  - {f}" for f in failure_sample)
    if len(failures) > 20:
        failure_str += f"\n  ... ({len(failures) - 20} more failures)"

    return (
        f"Your function achieved recall {recall:.1%} ({hit}/{total_gold} gold fields matched).\n\n"
        f"Failures:\n{failure_str}\n\n"
        "Revise `def extract(words: list) -> list:` to fix these failures.\n"
        "Return ONLY the Python function code. No markdown fences."
    )


# ── Script extraction from Claude response ────────────────────────────────────

def _extract_code(text: str) -> str:
    """Extract Python function code from Claude's response."""
    text = text.strip()

    # Strip markdown code fences
    if "```python" in text:
        text = text.split("```python", 1)[1]
        text = text.rsplit("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1]
        text = text.rsplit("```", 1)[0]

    return text.strip()


def _is_truncated(code: str) -> bool:
    """Detect if the script was likely truncated mid-generation."""
    if not code:
        return True
    lines = code.rstrip().splitlines()
    last = lines[-1].rstrip() if lines else ""
    # Truncated if last line is incomplete (no closing paren/bracket, ends with operator, etc.)
    return bool(re.search(r"[,(\[\\+\-*=]$", last)) or (
        not last.startswith("    ") and not last.startswith("def ") and not last == "return results"
    )


# ── Core build loop ────────────────────────────────────────────────────────────

def build_cluster(
    cluster_id: int,
    examples: list[dict[str, Any]],
    client: AnthropicVertex,
) -> tuple[str | None, float, int]:
    """Run build-time loop for one cluster. Returns (script, best_recall, n_iters).

    Iterates up to MAX_ITERS times. Accepts the best recall achieved even if < TARGET.
    Returns (None, 0.0, 0) if no valid script could be generated.
    """
    if not examples:
        print(f"  cluster {cluster_id}: no train examples — skip")
        return None, 0.0, 0

    # Pre-compute total gold count once (used for revision message)
    total_gold = sum(
        1 for ex in examples for g in ex["gold"] if g["word_ids"]
    )
    if total_gold == 0:
        print(f"  cluster {cluster_id}: no gold fields with words — skip")
        return None, 0.0, 0

    messages: list[dict] = []
    best_script: str | None = None
    best_recall: float = 0.0
    n_iters = 0

    for iteration in range(1, MAX_ITERS + 1):
        n_iters = iteration

        if iteration == 1:
            user_content = _build_initial_user_message(cluster_id, examples)
        else:
            user_content = _build_revision_message(best_recall, total_gold, failures)  # type: ignore[possibly-undefined]

        messages.append({"role": "user", "content": user_content})

        # Call Sonnet (sync)
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_SCRIPT_TOKENS,
                system=[{
                    "type": "text",
                    "text": _SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=messages,
                temperature=1.0,
            )
        except Exception as e:
            print(f"  [iter {iteration}] API error: {e}")
            break

        raw_text = response.content[0].text if response.content else ""
        messages.append({"role": "assistant", "content": raw_text})

        script_code = _extract_code(raw_text)
        if not script_code.strip().startswith("def extract"):
            print(f"  [iter {iteration}] No 'def extract' found — retrying")
            failures = ["Script must start with 'def extract(words: list) -> list:'. "
                        "Return ONLY the function, no preamble."]
            continue

        if _is_truncated(script_code):
            print(f"  [iter {iteration}] Script appears truncated — requesting shorter version")
            failures = ["Script appears to have been truncated (incomplete). "
                        "Write a SHORTER, more concise function using loops instead of "
                        "per-field blocks. It must be syntactically complete."]
            continue

        recall, failures = measure_recall(script_code, examples)
        hit = round(recall * total_gold)
        print(
            f"  [iter {iteration}/{MAX_ITERS}] recall={recall:.1%} ({hit}/{total_gold}) "
            f"— {len(failures)} failures"
        )

        if recall >= best_recall:  # >= so we always save the latest valid script
            best_recall = recall
            best_script = script_code

        if best_recall >= TARGET_RECALL:
            print(f"  Target reached ({TARGET_RECALL:.0%})")
            break

    return best_script, best_recall, n_iters


# ── Script caching ────────────────────────────────────────────────────────────

def cache_script(
    cluster_id: int,
    script_code: str,
    train_recall: float,
    n_iters: int,
    n_train_docs: int,
    metadata: dict[str, Any],
) -> None:
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    path = script_path(cluster_id)
    path.write_text(script_code)

    metadata[str(cluster_id)] = {
        "cluster_id": cluster_id,
        "train_recall": round(train_recall, 4),
        "n_iters": n_iters,
        "n_train_docs": n_train_docs,
        "last_built": datetime.now(timezone.utc).isoformat(),
    }
    save_metadata(metadata)


# ── Cluster selection helpers ─────────────────────────────────────────────────

def _tight_clusters(train_index: dict[int, list[str]], min_train_docs: int = 3) -> list[int]:
    """Return clusters with >= min_train_docs training docs (TIGHT+MEDIUM candidates)."""
    return sorted(
        cid for cid, docids in train_index.items() if len(docids) >= min_train_docs
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build Code-Factory extraction scripts per cluster")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cluster", type=int, help="Build script for a single cluster ID")
    group.add_argument("--clusters", type=str, help="Comma-separated cluster IDs, e.g. '1,5,42'")
    group.add_argument("--all", action="store_true", help="Build scripts for all clusters with >= 3 train docs")
    group.add_argument("--tight-only", action="store_true", help="Same as --all but label (alias)")
    parser.add_argument("--skip-built", action="store_true", default=True, help="Skip clusters with existing scripts (default: True)")
    parser.add_argument("--force", action="store_true", help="Rebuild even if script exists")
    parser.add_argument("--min-train", type=int, default=3, help="Minimum train docs per cluster (default: 3)")
    args = parser.parse_args()

    skip_built = args.skip_built and not args.force

    print("Loading train cluster index...")
    train_index = _build_cluster_index("train")
    print(f"  {len(train_index)} clusters in train split")

    # Select which clusters to build
    if args.cluster is not None:
        cluster_ids = [args.cluster]
    elif args.clusters is not None:
        cluster_ids = [int(c.strip()) for c in args.clusters.split(",")]
    else:
        cluster_ids = _tight_clusters(train_index, min_train_docs=args.min_train)
        print(f"  {len(cluster_ids)} clusters with >= {args.min_train} train docs")

    if skip_built:
        before = len(cluster_ids)
        cluster_ids = [cid for cid in cluster_ids if not has_script(cid)]
        print(f"  Skipping {before - len(cluster_ids)} already-built clusters → {len(cluster_ids)} to build")

    if not cluster_ids:
        print("Nothing to build. Use --force to rebuild existing scripts.")
        return

    client = AnthropicVertex(project_id=VERTEX_PROJECT_ID, region=VERTEX_LOCATION)
    metadata = load_metadata()

    results: list[dict] = []
    for i, cluster_id in enumerate(cluster_ids, 1):
        print(f"\n[{i}/{len(cluster_ids)}] cluster {cluster_id}")
        train_docs = train_index.get(cluster_id, [])
        print(f"  train docs: {len(train_docs)}, using up to {N_TRAIN_DOCS}")

        try:
            examples = load_train_examples(cluster_id, train_index, n=N_TRAIN_DOCS)
            if not examples:
                print(f"  No examples loaded — skipping")
                results.append({"cluster_id": cluster_id, "status": "skipped_no_examples"})
                continue

            script, recall, n_iters = build_cluster(cluster_id, examples, client)

            if script is None:
                print(f"  No script generated")
                results.append({"cluster_id": cluster_id, "status": "failed", "recall": 0.0})
                continue

            cache_script(cluster_id, script, recall, n_iters, len(examples), metadata)
            status = "ok" if recall >= TARGET_RECALL else "below_target"
            print(f"  Cached. recall={recall:.1%}, iters={n_iters}, status={status}")
            results.append({
                "cluster_id": cluster_id,
                "status": status,
                "recall": recall,
                "n_iters": n_iters,
            })

        except KeyboardInterrupt:
            print("\nInterrupted — partial results saved")
            break
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append({"cluster_id": cluster_id, "status": "error", "error": str(e)})

    # Summary
    print("\n" + "=" * 60)
    print(f"Build complete: {len(results)} clusters processed")
    ok = [r for r in results if r.get("status") in ("ok", "below_target")]
    if ok:
        recalls = [r["recall"] for r in ok]
        above = sum(1 for r in recalls if r >= TARGET_RECALL)
        print(f"  Scripts generated: {len(ok)}")
        print(f"  Recall >= {TARGET_RECALL:.0%}: {above}/{len(ok)}")
        if recalls:
            print(f"  Recall distribution: min={min(recalls):.1%} median={sorted(recalls)[len(recalls)//2]:.1%} max={max(recalls):.1%}")
    failed = [r for r in results if r.get("status") not in ("ok", "below_target")]
    if failed:
        print(f"  Failed/skipped: {len(failed)}")
    print(f"Scripts at: {SCRIPTS_DIR}")
    print(f"Metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
