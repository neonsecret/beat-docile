# Beat DocILE

Few-shot Claude pipeline for the [DocILE benchmark](https://github.com/rossumai/docile) — Key Information Localization and Extraction (KILE) + Line Item Recognition (LIR) on invoice/PO documents. Target: GraphDoc SOTA at 71.25% KILE AP / 75.93% LIR F1.

## Current Best: v2_ensemble — 46.48% KILE / 50.77% LIR (500 val docs)

Three Claude Sonnet runs (T=1.0 / T=0.3 / alt-prompt) merged via `ensemble.py`. Single-model baseline (`v2_preds`) is 44.61% KILE. Gap to GraphDoc: −24.8pp. See `KNOWLEDGE_BASE.md §2` for full architecture description.

---

## Quick Start

```bash
# Install
uv sync

# Smoke test (5 val docs, few-shot on by default)
DATA_ROOT=data uv run bd smoke --limit 5

# Full val extraction → predictions JSON
DATA_ROOT=data uv run bd extract --split val --out predictions/run.json

# Eval a saved predictions file
DATA_ROOT=data uv run bd eval --preds predictions/run.json

# Ensemble three variants (example: v2 ensemble)
DATA_ROOT=data uv run python tools/ensemble_v2_variants.py

# 250-doc gate (half cost, same signal for ≥1pp lifts)
DATA_ROOT=data uv run python tools/<experiment>_250.py
```

**Config** — create `.env.local` at project root:
```
VERTEX_PROJECT_ID=<your-gcp-project>
VERTEX_LOCATION=<your-vertex-region>
DATA_ROOT=/path/to/docile/data
DEFAULT_MODEL=claude-sonnet-4-6
```

**Data layout** — DocILE expects `DATA_ROOT/{train,val,test}/` with `documents/`, `annotation/`, `ocr/` subdirs.

---

## Module Map

### Active (production pipeline)

| Module | Purpose |
|---|---|
| `config.py` | Env-var config: `DATA_ROOT`, `DEFAULT_MODEL`, API credentials |
| `vertex.py` | Anthropic API async client — retry, semaphore, prompt caching |
| `data.py` | DocILE dataset helpers: `iter_pages`, `PageContext`, `WordBox` |
| `extract.py` | **Main extractor** — `extract_documents`, `extract_page`, targeted second pass |
| `fewshot.py` | Cluster-based few-shot builder — loads train examples by cluster id |
| `ensemble.py` | `merge_predictions` — IoU-grouped field merging across multiple runs |
| `eval.py` | `run_eval` / `print_scores` — wraps docile benchmark evaluation |
| `cli.py` | CLI entry point: `bd smoke`, `bd extract`, `bd eval` |
| `refiners.py` | Per-fieldtype word-id refinement (address dedup, amount trimming, etc.) |
| `validators.py` | Format-confidence scoring (IBAN checksum, date patterns, etc.) |
| `align.py` | Text-to-OCR span alignment fallback when word_ids are unavailable |
| `recall_aol.py` | ADD-only recall augmentation via cluster field priors (neutral, +0.02pp) |

### Conditional / flag-gated

| Module | Flag / Use |
|---|---|
| `oracle_extract.py` | `BD_USE_ORACLE_PREPASS=1` — regex/checksum pre-pass for structured fields |
| `field_instructions.py` | `BD_USE_FIELD_INSTRUCTIONS=1` — per-field extraction guidance block |
| `optimized_prompt.py` | DSPy-optimized system prompt (standalone constant, no runtime dep) |
| `cluster_infer.py` | Cluster assignment for test docs (no gold annotation available) |

### Experimental (built, not conclusively evaluated)

| Module | Purpose |
|---|---|
| `sail_retrieval.py` | SAIL semantic retrieval — index built on local GPU host, no-match docs target |
| `gutenocr_extract.py` | GutenOCR-3B (Qwen2.5-VL FT) extractor |
| `qwen3vl_extract.py` | Qwen3-VL-8B-Instruct extractor |
| `tabled_lir.py` | Chained LIR extraction via table-aware prompting |

### Buried (evaluated, proven negative — do not re-run without new hypothesis)

| Module | Result | Why it failed |
|---|---|---|
| `aol_extract.py` | −4.2pp KILE | Score demotion reranks TPs below FPs in AP curve |
| `haiku_verify.py` | Negative | Haiku verification over-prunes correct predictions |
| `bbox_verify.py` | −0.2pp KILE | Verification adds latency with no lift |
| `react_extract.py` | −22pp KILE | Triage gate skipped 48% of fields → recall crater |
| `v6_pipeline.py` / `v6_cli.py` | Buried with ReAct | Same root cause |
| `code_factory.py` | −20pp KILE | Generated code misses field semantics |
| `disambiguator.py` | 0pp | GLiNER operates on wrong abstraction layer |
| `layout_regions.py` | −9pp KILE | DocLayout filter over-excludes valid regions |
| `donut_extract.py` | 2.8% KILE | OCR-free VLM can't align to snapped word grid |
| `dspy_optimizer.py` | Negative | Optimized prompts overfit 50-doc set, regressed on 500 |
| `glm_ocr_extract.py` | Not measured | Wrong specialization — OCR-focused, not KIE |
| `gemini_extract.py` | Not measured | Gemini routing; never fully evaluated |
| `text_extract.py` | Marginal | Text-only variant; no lift over image+text |
| `vlm_extract.py` | Not measured | Generic wrapper; superseded by specific extractors |
| `classifiers.py` / `cls_candidates.py` | Not measured | Infrastructure for classifier re-ranking; buried |
| `embed.py` / `precise_align.py` | Support only | Used by buried approaches |

See `KNOWLEDGE_BASE.md` for full experiment verdicts, failure root causes, and remaining high-EV directions.

---

## How to Add a New Extractor

1. **Mirror the interface** — return `(kile_preds, lir_preds)` as `{docid: [Field, ...]}` dicts; bboxes must be derived from snapped OCR word ids (see `data.py:iter_pages`, `extract.py:_merge_bboxes`).
2. **Gate it on 250 docs first** — use `tools/val_250_docids.json` + `split_name="v2_250_gate"` in `Dataset(...)`. Only promote to 500 if ≥1pp KILE lift confirmed.
3. **Score against v2_ensemble baseline** — baseline on 250 docs is 47.33% KILE / 51.28% LIR (`predictions/v2_ensemble_500.json` filtered to first 250 docids).
4. **Never modify existing prediction scores** — score-modifying post-processors always hurt AP on a high-AP baseline (see `KNOWLEDGE_BASE.md §9` operational lessons).

---

## Ground Rules

- **Modern models only** — mid-2025 or newer. No CLIP, Donut, LayoutLMv3, Florence-2, or anything pre-2025.
- **250-doc gate before 500-doc** — half the API cost, same signal for ≥1pp lifts.
- **Score-modifying verifiers are one-sided risk** — only ADD fields, never demote existing predictions.
- **50-doc results don't generalize** — every approach showing +1–3pp on 50 docs regressed on 500. Use 250-doc gate minimum.
- See `KNOWLEDGE_BASE.md §9` for the full operational lessons list.
