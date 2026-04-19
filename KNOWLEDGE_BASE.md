Engineering Knowledge Base

A consolidated technical record of work on the DocILE benchmark targeting GraphDoc's published SOTA (71.25% KILE AP / 75.93% LIR F1). The objective was to surpass that result on the official 500-document validation split using zero/few-shot architectures rather than full retraining of a graph-based model. This document captures every architecture that was built, every approach that was attempted, the verdicts at full scale, the lessons that came out of those verdicts, and the directions that were identified as promising but not pursued for resource reasons.

---

## 1. The Problem

### 1.1 DocILE benchmark in one paragraph

DocILE provides 5,180 train / 500 validation / 1,000 test invoice and PO documents (with held-out test labels), a held-out OCR layer (DocTR), and a hard-pinned word-snapping geometry. The two scored tasks are KILE (Key Information Localization and Extraction — 36 page-level field types) and LIR (Line Item Recognition — 19 row-grouped field types with `line_item_id` grouping). The dataset also ships ~6,680 annotated docs and a 100K synthetic split.

### 1.2 The metric makes everything harder

The matching rule is **PCC-IoU ≥ 1.0** by default. Each OCR word, after snapping, contributes a set of pseudo-character centers (PCCs). A predicted bbox matches a gold annotation iff (a) same fieldtype, (b) same page, and (c) the predicted bbox covers **exactly the same PCC set** as the gold bbox. One extra word, one missing word — the prediction scores zero.

This is not a soft IoU. It is a binary, set-equal predicate over snapped word units. Any pipeline whose output bboxes do not align precisely to the DocTR snapped word grid scores ~0% regardless of semantic correctness.

### 1.3 The implications

- Generative models that produce *text* and require post-hoc alignment back to OCR words are systematically punished — every alignment ambiguity, every multi-word label boundary error, every financial-code value that doesn't appear verbatim in OCR becomes a zero.
- Models with their own OCR (Donut, OCR-free VLMs) cannot align to DocILE's snapped geometry by definition.
- Even when a model "knows" the right words, picking the right *set boundary* on multi-word fields (addresses, vendor names, multi-token amounts) is the dominant remaining error.
- Fields with noisy OCR transcription where the true value doesn't appear character-for-character in the OCR layer are nearly unrecoverable without spatial grounding.

These implications shape every architectural choice below.

### 1.4 Standing reference architectures

| Architecture | DocILE result | Mechanism |
|---|---|---|
| **GraphDoc** (DocILE SOTA, 2023) | 71.25% KILE / 75.93% LIR | OCR word grid → graph with spatial edges → GNN with per-word per-fieldtype classifier head. PCC-friendly by construction (predicts on existing OCR words, no generation). Trained on DocILE train. |
| **Rossum production backend** (RASG, arXiv:2405.20245) | n/a (commercial) | Hybrid OCR + VLM features + per-field classifier + checksum validators + per-customer schema adaptation + HITL escalation. Built around auditable confidence and graceful degradation. |
| **VDInstruct** (arXiv:2507.09531, July 2025) | 74.2% F1 in-domain (claimed) | Multimodal LLM, OCR-free, dual-encoder (spatial + semantic). Trained with DocILE in-distribution. CC BY 4.0 license declared, weights not yet released as of this writing. |

---

## 2. Standing Best System

### 2.1 Headline numbers

| System | KILE AP | LIR F1 | Eval set |
|---|---|---|---|
| **`v2_ensemble`** (current best) | **46.48%** | **50.77%** | 500 val docs |
| `v2_preds` (single-model baseline) | 44.61% | 50.89% | 500 val docs |
| Gap to GraphDoc | -24.77pp | -25.16pp | — |

### 2.2 Architecture

**Single extractor (`v2_preds`):** Claude Sonnet via the Anthropic API consuming a per-page row-grouped OCR word listing plus a single cluster-matched few-shot example (image + gold-JSON) drawn from the train split. Output is a JSON dict mapping field type → `{word_ids, text, score}`. Bbox is derived as the union of the snapped bboxes of the selected `word_ids`, so bboxes align to the DocTR snapped grid by construction.

**Ensemble extractor (`v2_ensemble`):** Three runs of `v2_preds` with prompt/temperature variants, merged per-field via the existing `ensemble.py` module:

| Variant | Configuration |
|---|---|
| `v2_t00` | T=1.0, default system prompt |
| `v2_t03` | T=0.3, default system prompt |
| `v2_alt` | T=1.0, alternate system prompt with single specific edit |

Per-field merge takes the prediction with the highest score across the three variants (with `weighted_max` score combination). Recall jumps 0.55 → 0.59 because the variants miss different fields; precision drops 0.62 → 0.56 because the union accumulates more FPs. Net is +1.87pp KILE / -0.12pp LIR.

### 2.3 What drove the metric from V1 (zero-shot baseline 27.7% KILE) to V2

Two changes account for essentially all of the +17pp jump:

1. **Cluster-based few-shot retrieval.** Each DocILE training annotation carries a `cluster_id` (template grouping curated by the dataset authors). 75% of val docs have a cluster with at least one annotated train example. Picking that train example as the few-shot demonstration in the Sonnet prompt is the single largest lever. Built-in metadata, no embedding model required. (CLIP/SigLIP retrieval was tried and underperformed cluster_id.)
2. **Row-grouped OCR words in the prompt.** Words are grouped into rows by y-coordinate proximity and emitted as `R{i}(y≈{top:.3f}): {id}:{text}  {id}:{text}` rather than a flat list. Gives the model spatial context without needing the image.

### 2.4 Cost & latency

`v2_ensemble` runs three Anthropic API calls per page. Inference cost is 3× single-model. Each prediction carries an explicit `score` derived from the model's confidence claim plus per-field validation discounts. The output JSON is the standard DocILE prediction format and runs through the official `docile_evaluate` CLI without modification.

The codebase ships with status banners on every module ([ACTIVE] / [EXPERIMENTAL] / [RESEARCH-BURIED] / [ARCHIVED]) and a README module map; this document focuses on findings and architecture, not on file-by-file inventory.

---

## 3. The Original Architectural Hypothesis

A separate research note proposed a Hybrid Spatial-Discriminative Pipeline (HSDP) wrapped in an Agentic Orchestration Layer (AOL):

1. **Phase 1 — Deterministic token graph.** Extract the OCR word grid as ground truth; force all downstream selections to bind to it. Mathematically bounds textual hallucination to zero.
2. **Phase 2 — Diffusion VLM for global spatial planning.** Use a discrete-text diffusion VLM (LLaDA-V class) to output semantic bounding boxes (table block, total block, supplier address) — not text. Diffusion's parallel denoising avoids the topological serialization failure of AR generation on 2D layouts.
3. **Phase 3 — Bidirectional extractive decoding.** Within each semantic region, classify OCR tokens with a JPT-style ("Just Pass Twice") trick: feed `[Input + Input]` to a causal LM and read hidden states from the second pass to recover bidirectional context, then map to the schema with a linear classification head.
4. **Phase 4 — Agentic verification.** An orchestrating agent runs deterministic tool checks (sums, regex, schema constraints) and re-prompts on contradictions.

**Verification status of the citations** (after exhaustive landscape verification):

| Citation | Real? | Verdict for our use |
|---|---|---|
| LLaDA (arXiv:2502.09992, MIT, ICML/NeurIPS 2025) | ✅ | Real; zero document-AI evaluation exists. |
| LLaDA-V (arXiv:2505.16933, CVPR 2026) | ✅ | Real; has GUI-bbox grounding capability; no document-region segmentation training; would require fine-tuning to do Phase 2. |
| DiffusionVL (arXiv:2512.15713, HUST) | ✅ | Real; outputs text not coordinates; does not natively do document semantic segmentation. |
| S³ Self-Adaptive Schema Scaffolding (arXiv:2507.04504, USC) | ✅ | Real; solves structural-adherence on diffusion LMs; Sonnet already at ~99% structural adherence on our schema, so it is solving a problem that doesn't exist for our pipeline. |
| JPT — Just Pass Twice (arXiv:2604.05158, WitnessAI) | ✅ | Real; not zero-shot in the colloquial sense — requires LoRA + classification-head training on the target schema. Echo Embeddings (Springer et al., ICLR 2025, arXiv:2402.15449) is the direct predecessor. |

**What of the original hypothesis is empirically supported:**
- The discriminative-extraction thesis (predict on OCR words rather than generate text) is correct and is the only way to satisfy PCC-IoU=1.0 cleanly. Multiple independent attempts at text-then-align consistently fail.
- The agentic verification layer is supported only in the recall-augmentation direction (see §4.3); score-modifying verifiers are one-sided risk on a high-AP baseline (see §7.4).

**Where the proposal cannot be executed off-the-shelf:**
- Phase 2 (diffusion VLM as document semantic segmenter) has no published baseline. LLaDA-V's GUI-bbox grounding suggests the capability is plausible but adapting to invoices would be the first publication. Unproven.
- Phase 3 (JPT bidirectional classification) is real but requires fine-tuning, putting it in the same cost class as LoRA-tuning a stronger backbone (Qwen3-VL-8B) that already has bbox grounding tokens.
- Phase 4 (agentic verification) is workable only as ADD-only recall augmentation; verification-with-modification regresses on a strong baseline.

---

## 4. Composition Opportunities (How This Work Plugs Into GraphDoc/Rossum-style Systems)

The following compositions take modules from this codebase and re-arrange them as layers on top of, or alongside, GraphDoc-style or Rossum-style architectures. Estimates are first-order and based on either prior small-scale tests or analogous results.

### 5.1 Classifier reranker (GraphDoc-light over an LLM)

**Idea.** Use the trained per-field MLPs (`classifiers.py`) as a soft reranker / FP filter on top of any extractor's predictions. Maps to GraphDoc's per-word per-fieldtype classification head, but as a veto layer rather than the primary predictor.

**Status.** Tested over `v2_ensemble`. Buried at 250-doc scale due to rank disturbance: even at threshold=0.0 (no drops, just reweighting), AP fell -2.5pp because the classifier prior shuffles correct predictions down the rank list. The classifiers were trained on "is this token a typical instance of field X" using random OCR negatives, so they don't track whether a *specific* high-confidence Claude pick is a true positive.

**Where it might still be useful.**
- **LIR-only reranking.** Side finding: LIR F1 improved +1.6pp at threshold=0.3 even when KILE collapsed. Line-item field text patterns are more regular than KILE field patterns; the classifier prior tracks them better.
- **As a feature fed into a different reranker** (e.g., LightGBM with multi-source features) rather than as a direct multiplicative score.
- **As a presence gate** (doc-level "does this fieldtype exist?") rather than a per-prediction score.

### 5.2 Path B — Qwen3-VL-8B LoRA fine-tune on DocILE train

**Idea.** Fine-tune Qwen3-VL-8B (Apache 2.0, native bbox grounding, DocVQA 97% zero-shot) on DocILE train using grounding-style loss: input = page image + question "where is the {field_type}?", output = JSON with `bbox_2d` in 0-1000 normalized coords + extracted text.

**Recipe (validated, in `runpod/qwen_vl_train.py`).**
- LoRA r=32, α=64, dropout=0.05
- Target modules: q/k/v/o + gate/up/down on LM layers; vision tower frozen
- BF16, no quantization (32 GB 5090 has headroom)
- Gradient checkpointing on LM
- Per-device batch=2, grad_accum=8 → effective batch 16
- LR=2e-4 cosine, 3 epochs, ~1251 optimizer steps
- Eval every 125 steps on a held-out 100-doc slice for overfitting detection
- Negative examples (~10-20%): "this field is not on this page" to teach refusal
- Estimated VRAM peak: 22-24 GB (16 GB model + 0.5 GB LoRA/optimizer + 4-6 GB activations)
- Cost: ~$3.50-6.00 on RunPod 5090

**Status.** Training pipeline complete (5 artifacts: setup script, training script, runbook, Mac-side inference module, mocked unit tests passing). Currently executing as of this document's snapshot. Headline projection: 60-70% KILE if the published Qwen3-VL grounding accuracy transfers to DocILE field types.

**Where it composes.**
- As primary extractor (replaces v2_ensemble).
- As an additional voice in the ensemble (cross-architecture diversity, more orthogonal errors than cross-prompt diversity).
- As the discriminative grounding layer underneath an AOL-style schema-reasoning agent (see §5.4).

### 5.3 Recall-augmentation AOL (the one safe agentic shape)

**Idea.** For each document, build a per-cluster field prior from the train split (fields present in ≥50% of cluster training docs). Compare to the extractor's predictions for that doc; identify cluster-expected fields that are completely absent from the predictions; re-prompt the extractor for ONLY those missing fields with an explicit "if genuinely not present, return empty" framing. Append any new finds to the prediction set with a discounted score (×0.7); never modify existing predictions.

**Status.** Built and tested at 250-doc gate. Net effect: +0.02pp KILE — neutral. The design is correct (LIR was flat, confirming no false positives are added — the ADD-only premise holds), but the practical yield was only 4% (8 fields added across 187 re-prompts). Diagnosis: cluster prior is too coarse (50% threshold means cluster docs legitimately vary on optional fields), and Claude correctly refuses to hallucinate when the field isn't there.

**Where it might still be useful.**
- With a tighter prior signal (per-doc field expectation from a stronger model — e.g., a cluster-fine-tuned classifier).
- With re-prompts that include the OCR neighborhood likely to contain the missing field (focused crop), not just the field name.
- For specific missing fields where the prior is high-precision (`document_id` is present in 100% of invoices; if extractor missed it, that's worth retrying).

### 5.4 Rossum-style hybrid: VLM grounding + LLM schema reasoning

**Idea.** Two-stage pipeline:
1. Path B Qwen3-VL FT (or any spatial-grounding VLM) proposes candidate spatial regions per fieldtype with bboxes.
2. Sonnet (or another schema-aware LLM) takes the proposals + the OCR word grid + the schema constraints (mutex rules, conditionals, label semantics) and decides which proposals to accept, reject, or refine.

This composition uses the VLM for what it's good at (spatial localization on document images) and the LLM for what it's good at (schema-level reasoning and exception handling). It is conceptually equivalent to Rossum's hybrid OCR+VLM+classifier+rules stack, executed with 2026-era components.

**Status.** Conditional on path B landing positive. Estimated +10-20pp over v2_ensemble if path B clears 60% KILE.

### 5.5 Auditable confidence ensemble (production-readiness path)

**Idea.** Compose multiple sources of per-field confidence: v2_ensemble's score, bbox_verify's evidence count, classifiers.py's MLP score, oracle_extract's checksum verdict. Where ≥3 sources agree → auto-accept (high confidence). Where sources disagree → flag for HITL review with the specific contradiction surfaced.

**Status.** Designed but not built. Doesn't move KILE AP directly (it's an accept/escalation layer, not an extraction improvement), but it is the production-readiness shape and is the closest match to Rossum's auditable-confidence + HITL operational architecture.

### 5.6 SAIL retrieval + cluster prediction for the no-match population

**Idea.** 25% of val docs have no matching train cluster. Those docs lose ~11.5pp KILE on average compared to cluster-matched docs (because they get zero-shot fallback). Two complementary mechanisms:
- **`cluster_infer.py`** predicts the cluster from the page image (Qwen3-VL embedding, 70% top-1). For no-match docs, pick the predicted cluster's training example as few-shot.
- **SAIL retrieval-ICL** retrieves entity-level similar train docs across clusters (built on 3070, training-free).

Use `cluster_infer` first; fall back to SAIL on low-confidence predictions.

**Status.** Both built, not end-to-end evaluated together. Estimated +2-3pp dataset-wide if it closes most of the no-match drag.

### 5.7 Span-correction model (a 2026 redo of the LayoutLMv3 idea)

**Idea.** Train a small bidirectional model whose only job is to refine v2_ensemble's spans to PCC-aligned exactness. Input: candidate `word_ids` from v2_ensemble + OCR word grid features. Output: corrected `word_ids` (potentially adding/removing one token from each end of the candidate). Much smaller task than full KIE.

**Status.** Not built. The original LayoutLMv3 attempt was the wrong architecture for the wrong task (it tried to do full token classification, not span correction). A modern small bidirectional encoder (e.g., a Qwen3 embedding head with a bilinear span classifier) could plausibly land +5-10pp by fixing the multi-word PCC mismatch directly.

### 5.8 What does NOT compose

- **GraphDoc proper (full GNN training from scratch on DocILE word features).** High implementation cost; #5.1 (classifier reranker) gets a substantial fraction of the benefit cheaper.
- **Per-customer fine-tuning à la Rossum's enterprise deployments.** Requires per-customer training data that doesn't exist in DocILE. Cluster-based few-shot is the cheap proxy that already provided the +17pp from V1 to V2.
- **Drop-in 2026 SOTA OCR models for KIE.** Confirmed empirically: a model SOTA on text-recognition + reading-order benchmarks is not necessarily good at schema-conformant field extraction (see §6.1). The fit-vs-cost question always dominates.

---

## 5. What Was Tried and Did Not Work

Each entry: what was tried, the actual result on the largest evaluation set used, the root cause, and where the underlying idea might still apply.

### 6.1 Drop-in 2026 SOTA: GLM-OCR-0.9B

**What.** GLM-OCR (zai-org, MIT, 0.9B params, ~4 GB VRAM, claims #1 on OmniDocBench V1.5 at 94.62 — beating Qwen3-VL-235B and Gemini-3-Pro). Two-stage SDK: PP-DocLayoutV3 region detection + GLM-OCR VLM KIE. Used `glmocr[selfhosted]` mode with a local vLLM server hosting the GLM-OCR weights.

**Result.** 4.23% KILE (21-doc partial run with full SDK + layout stage active). Below the 10% gate; ceiling estimated 4-6%.

**Root cause.** OmniDocBench scores text recognition + reading order + table structure. DocILE KILE scores schema-conformant field localization. Different tasks. GLM-OCR's training had three task prompts — "Text Recognition", "Table Recognition", "Formula Recognition." The "Information Extraction" prompt with a custom JSON schema is an emergent capability bolted onto the chat template, not a primary training objective. Empirical signature: precision 21% (when it fires it's right), recall 17% (refuses to fire on most fields because schema-conformance was never trained).

**Where it might still apply.** GLM-OCR + LoRA fine-tune on DocILE train would address the missing schema-conformance training. But the cost-benefit is worse than fine-tuning Qwen3-VL-8B from a stronger prior.

### 6.2 Refiner heuristics (`refiners.py`) at scale

**What.** Per-field-type span cleaners that strip label words, isolate contiguous runs, pick best-row clusters. Three configurations tested:

| Config | KILE 500d | Δ vs v2 |
|---|---|---|
| V5b (original refiner) | 41.79% | -2.82pp |
| V5b + guard mode (only edit if removed words are in label set) | 43.64% | -0.97pp |
| V5b + bbox_verify (3-pass evidence verifier) | 42.75% | -1.86pp |

**Result.** Net negative at 500-doc scale across every configuration tested. Per-field analysis (`predictions/refiner_per_field_report.md`) shows a real split: 17 of 36 fields benefit from refining, 13 regress. Refiner helps on harder-to-find blocks (delivery/other addresses, vendor names, vendor_address, tax_detail_rate) but hurts on exact-value codes and standard billing blocks (bank_num, account_num, dates, billing address, amount_total_*).

**Root cause.** Refiner's heuristics over-trim valid extractions on the long-tail of docs that don't appear in 50-doc samples. Sonnet is already precise on the fields refining hurts; refining only helps where Sonnet over-includes context (multi-line addresses).

**Where it might still apply.** A selective-refiner gate (refine only the ~6 helping fields, skip the rest) could plausibly recover +0.5-1pp. The per-field map is preserved at `predictions/refiner_per_field_report.md` for that future implementation.

### 6.3 Snap geometry tightening (margin=3 vs default margin=6 at 200 DPI)

**What.** Rebuild the OCR snapped_geometry cache with a tighter min_char_width margin. Hypothesis was that tighter snap → tighter ground-truth bboxes → more precise PCC matching.

**Result.** -2.96pp KILE at 500-doc. Verified bit-for-bit that extract and eval read from the same snap source — no methodology bug. Tighter snap is genuinely worse for Sonnet's bbox–PCC alignment.

**Root cause.** Tighter character-width margin shrinks word bboxes; PCCs are at character centers within bboxes. With tighter bboxes, predicted bboxes need to land inside a smaller target area to satisfy PCC inclusion, increasing borderline misses.

**Where it might still apply.** Could help models that produce high-precision bboxes (a fine-tuned grounding VLM) but hurts models that produce slightly loose bboxes (Sonnet via word_id selection). May be worth re-testing if path B lands.

### 6.4 Cross-model ensemble: Gemini-3-Flash-Preview added as 4th voice

**What.** Build a Gemini extractor mirroring the Sonnet pipeline, add as 4th variant in the ensemble.

**Result.** Gemini-only run: 41.56% KILE on 250d (vs Sonnet's 44.95%). 4-way ensemble: 46.68% (vs 3-way 47.33%, -0.65pp). Buried.

**Root cause.** Gemini-3-Flash with `thinking_budget=0` (required to fit output budget) is strictly weaker than Sonnet on schema-conformant extraction (precision 0.59 vs 0.62). With thinking enabled, ~$50+ for the 250-doc run — out of budget.

**Where it might still apply.** Stronger affordable Gemini variant (Gemini-3-Pro at lower per-token cost, or a future Flash with cheaper thinking) could compose. The extractor code (`gemini_extract.py`) is preserved for retry. Cross-model diversity via Mistral, Llama-4, DeepSeek-VL2 also untested.

### 6.5 Oracle extraction integration (regex+checksum as pre-pass and post-pass)

**What.** `oracle_extract.py` produces high-confidence candidates for ~10 fields (IBAN mod-97 validated, BIC regex, VAT IDs with country-specific formats, account numbers, etc.). Two integration shapes tried:
- **Post-pass:** replace v2's prediction with oracle's where oracle fired with score=1.0.
- **Pre-pass:** inject oracle hints as a prefix block in the user message (`ORACLE CANDIDATES (verify and use only if correct): ...`).

**Result.**
- Post-pass: 0.00pp delta (v2 already extracted IBAN at oracle's level on these docs).
- Pre-pass single run with all candidates: -6.28pp KILE.
- Pre-pass single run with strict (score=1.0 only) candidates: -3.97pp KILE.
- Pre-pass strict in 4-way ensemble: +0.48pp (within noise).

**Root cause.**
- Post-pass: gap is recall not precision; oracle replacement doesn't help on fields v2 already gets right.
- Pre-pass: hint injection narrows Sonnet's attention to the hinted fields, degrading coverage on the other ~26 fields. Even with strict (high-precision) candidates, the attention cost dominates the precision benefit.

**Where it might still apply.**
- Oracle as a presence-detection signal for AOL recall augmentation (§4.3) rather than as a candidate hint.
- Oracle as a confidence input in the auditable-confidence ensemble (§4.5) — when oracle agrees with v2_ensemble at score=1.0, that prediction can auto-accept.
- Oracle as a hard validator that rejects model output failing a checksum (only fire on definitively-invalid outputs, never on valid ones).

### 6.6 Conservative AOL — score-modifying verifier (calc tool checking gross = net + tax)

**What.** After v2_ensemble extraction, run a calc verifier on the amount fields. If `|gross - (net + tax)| / gross > tolerance`, demote the score of the affected fields by a factor (×0.5 or ×0.95). Tested with strict tolerance (3%) and relaxed (10%).

**Result.** All four configurations regress: -4.20pp at ×0.5/3%, -1.45pp at ×0.95/10% (and identical to ×0.95/3% — the math tolerance had zero effect). Buried.

**Root cause.** Precision and recall at threshold are identical between baseline and AOL (verifier doesn't drop or add anything). The AP loss is purely from rank disturbance: demoting valid TPs shoves them below FPs from other docs in the AP curve. The verifier fires on parse failures (EU number formats `1.234,56`, partial display, rounding) — not on borderline math — so even a tighter math threshold doesn't reduce false-firing.

**Where it might still apply.** As a HITL escalation signal (flag for review, don't modify score) rather than a score-modifier. As a confidence input in the auditable-confidence ensemble.

### 6.7 V6 ReAct agentic loop (per-field tool-use Sonnet) — original implementation

**What.** Multi-agent stack: triage agent decides which fields to extract; per-field extractor agents focus on individual fields with tool access; cross-field verifier reconciles overlaps.

**Result.** 22.7% KILE (catastrophic regression).

**Root cause.** The triage gate dropped 48% of fields by skipping them entirely (over-aggressive "skip if uncertain"). The cross-field verifier deleted whole field arrays whenever predicted fields shared word_ids (delete-on-overlap semantics). Both gates were one-sided risk: their action was always "remove."

**Where it might still apply.** The agentic loop architecture itself is not broken — the gate semantics were. Per-field architectural lessons from later experiments:
- Triage must be a *priority hint*, never a gate (always extract; use triage only to allocate effort).
- Cross-field verification must be ADD-only or HITL-flag, never delete or modify.
- See §4.3 (recall-augmentation AOL) for the working shape.

### 6.8 Self-consistency 3-sample voting (per-field intersection of three independent runs)

**Result.** -2.7pp KILE.

**Root cause.** Mathematical: the intersection of three random-sample correct field sets shrinks faster than the intersection of incorrect sets. Recall drops more than precision gains.

**Where it might still apply.** Union-based ensembling is the right shape (and is what `v2_ensemble` does to gain +1.87pp). Intersection-based voting is structurally wrong for AP-recall.

### 6.9 DSPy MIPROv2 prompt optimization with Haiku evaluator

**What.** DSPy 3.1.3 (Feb 2026) MIPROv2 with 61 trials, Haiku as eval, Sonnet as instruction proposer.

**Result.** Haiku proxy showed +2pp on 10 docs; 50-doc Sonnet eval showed -0.77pp KILE / -3.26pp LIR.

**Root cause.** Haiku needs more explicit guidance than Sonnet. Instructions optimized for Haiku embed redundant clarifications that hurt Sonnet's already-strong baseline.

**Where it might still apply.** DSPy MIPROv2 with Sonnet (or a stronger model) as the eval target. Estimated cost ~$30 for one full optimization run.

### 6.10 Field-instructions dict (per-field guidance text appended to prompt)

**What.** Augment the system prompt with explicit per-field instructions ("amount_net is the pre-tax total..."). Two phrasings tested: original force-classification language and FORMAT-only language.

**Result.** -8.9pp KILE (original); -10.82pp KILE on 50d (FORMAT-only rewrite).

**Root cause.** Detailed per-field instructions cause Sonnet to over-constrain and miss valid fields. The concise system prompt lets the model use its general business-document semantics.

**Where it might still apply.** As a LoRA fine-tuning signal (turn instructions into training examples) rather than as runtime prompt content.

### 6.11 Text-aligner pattern (LLM outputs text only; aligner finds word_ids)

**What.** LLM outputs `{fieldtype, text}` only; a separate alignment module (exact match → NFKC normalization → fuzzy match → format-constrained match) maps text back to OCR word_ids.

**Result.** 30-34% KILE across three iterations of alignment-fix attempts. Below v2 by 7-12pp on every iteration.

**Root cause.** PCC-IoU=1.0 demands spatial grounding. Direct word_id selection (Sonnet looking at row-grouped OCR + image) outperforms text → fuzzy align because:
- Financial codes don't appear verbatim in OCR (the value as displayed is not exactly what OCR transcribed).
- Invoices with shared date cells fail occurrence-based assignment.
- Multi-line addresses fail text matching when OCR introduces line breaks.

**Where it might still apply.** Nowhere for PCC-IoU=1.0. Would work for text-only metrics (CER/WER on the extracted string).

### 6.12 Classifier-as-candidate-generator (sklearn MLPs proposing additional fields)

**What.** Run the trained MLPs over every OCR word; emit candidates for any word the MLP scored above threshold; merge with Sonnet's predictions.

**Result.** 22.58% KILE (-19pp regression).

**Root cause.** Classifiers were trained with random negatives. On val, score ≥ 0.85 means "text looks like this field" not "this doc has this field." Result: ~300 false positives per doc.

**Where it might still apply.** With a doc-level presence-detection gate on top (only emit candidates for fields the doc actually contains). Or as a feature in a feature-stack reranker.

### 6.13 LayoutLMv3-base full fine-tune

**What.** Fine-tune LayoutLMv3-base (layoutlmv3-large OOMed on 32 GB) for 20 epochs on DocILE train as a per-token classifier. Training token F1 = 0.582 (P=0.862, R=0.439).

**Result.** Inference produces ~0% AP at IoU=1.0 and 0.001 AP at relaxed IoU=0.25. Training token-level metrics did not transfer to field-level evaluation.

**Root cause.** Token-level F1 ≠ field-level AP. Model finds entity tokens but bbox boundaries don't match the gold span boundaries. Plus DocILE rules technically forbid IIT-CDIP-pretrained models for leaderboard submission (which LayoutLMv3 is) — irrelevant for the zero-shot research target but relevant if anyone repurposes this work for actual DocILE submission.

**Where it might still apply.** A 2026-era small bidirectional model trained as a span-correction head (input = candidate spans + OCR features, output = corrected spans) is the recommended successor — see §5.7. The "predict tokens from scratch" approach is the wrong factorization; refinement is the right one.

### 6.14 Donut / OCR-free generative VLMs

**What.** Donut, Qwen3-VL-2B zero-shot, similar OCR-free models that produce text outputs with no access to DocILE's OCR word grid.

**Result.** 0.29% to 2.8% KILE on 50d.

**Root cause.** PCC-IoU=1.0 requires alignment to DocILE's snapped OCR words. OCR-free models produce token coordinates from their own internal OCR, which never match the dataset's hardcoded snap geometry.

**Where it might still apply.** Nowhere for PCC-IoU=1.0. Would work for text-only or visual-grounding metrics on a different benchmark.

### 6.15 Code-Factory Python scripts (LLM generates extraction Python per cluster)

**What.** Sonnet generates a Python extraction function per cluster; the function runs deterministically over the OCR text.

**Result.** 20.9% KILE on 50d.

**Root cause.** Generated Python doesn't have access to DocILE's exact OCR word grid object — it operates on text strings. PCC-aligned bbox output is structurally impossible from this design.

**Where it might still apply.** As a feature/heuristic generator for fields with extremely regular templates within a cluster, fed into a downstream alignment step. Not as a primary extractor.

### 6.16 RapidTable (pre-2026 TSR) for LIR

**What.** Two configurations: (a) full-page table-structure recognition; (b) cropped-region table-structure recognition with YOLOv12-DocLayNet for region detection.

**Result.** -38pp LIR (full-page), -21pp LIR (cropped).

**Root cause.** RapidTable's TSR doesn't generalize to DocILE's invoice diversity. Full-page mode mistakes the entire invoice for one table. Cropped mode fails on tables with merged cells, multi-row line items, and non-standard headers.

**Where it might still apply.** Newer 2026-era TSR models (TableMaster-2026, latest StructEqTable) on cropped table regions could plausibly recover meaningful LIR F1. Untested.

### 6.17 PP-DocLayoutV3 region scoping for Sonnet (tag-mode and word-filter mode)

**What.** Use PP-DocLayoutV3 layout regions either as (a) tag annotations in the prompt or (b) word-set filters limiting Sonnet's selection space.

**Result.** Both -0 to -9pp at 50d.

**Where it might still apply.** The third untested mode — semantic-region-crop → re-prompt Sonnet on the cropped image — is the version GLM-OCR's SDK implements internally. It's the deterministic equivalent of the original document's Phase 2 (diffusion VLM as semantic segmenter). Estimated +5-15pp if it tightens Sonnet's word selection on multi-word fields. Not yet tested.

### 6.18 GLiNER post-processing on Sonnet output

**What.** Use GLiNER (small bidirectional NER) to disambiguate Sonnet's field-type assignments.

**Result.** ±0pp.

**Root cause.** GLiNER is designed to choose between competing candidates. Sonnet emits a single categorical pick per field — there are no competing candidates to disambiguate.

**Where it might still apply.** Composing with classifier reranking (§4.1) where multiple candidates per field exist — not over Sonnet directly.

### 6.19 Targeted second API call for financial codes

**What.** A second focused Sonnet pass dedicated to high-value-but-rare fields (IBAN, BIC, VAT IDs).

**Result.** Marginal (<0.5pp on overall AP).

**Root cause.** These fields together carry <2% of the AP weight. Even perfect recall on them moves overall AP <0.5pp.

**Where it might still apply.** As a cost-saving measure when only the financial codes are needed (e.g., a payments-focused subset of the schema).

---

## 6. Architectural Lessons

These are the durable findings — true regardless of which model or pipeline is in use.

### 7.1 The PCC-IoU=1.0 metric is the dominant constraint

Any architecture that does not output bboxes pre-aligned to DocILE's snapped OCR word grid will score near zero. This is not a soft preference — it is a binary set-equality predicate. Practical implication: **every extractor must either (a) select from the existing OCR word grid, or (b) post-snap its outputs to that grid via deterministic IoU-based assignment.** Anything else is dead on arrival.

### 7.2 Specialization-fit dominates "modern" or "open" or "fits VRAM"

The fit-vs-cost matrix:

| Specialization to OUR exact task | Cheapest closing move | Examples |
|---|---|---|
| Already on-task (drop-in) | $0 | None today (VDInstruct would qualify if released) |
| Adjacent task, transferable | Prompt + few-shot, ~free | Claude Sonnet (current best) |
| Capable architecture, wrong training | LoRA fine-tune, ~$5 | Qwen3-VL-8B (path B) |
| Wrong-task specialization, right modality | Substantial retraining | GLM-OCR |
| Architectural mismatch | Cannot close at any cost | Donut, Code-Factory, pre-2025 models violating the modern-only rule |

Modern + open + fits-VRAM are necessary, not sufficient. A model SOTA on a different benchmark task does not transfer to this one without explicit work to close the specialization gap.

### 7.3 The recall ceiling for prompt-based extraction is around 47% KILE

Empirical finding from the recall-augmentation AOL test: when an external prior identified 280 cluster-expected fields that v2_ensemble didn't extract, Claude correctly returned empty for 96% of them — those fields are genuine absences, not extraction failures. The ~24.77pp remaining gap to GraphDoc cannot be closed by better prompting / re-prompting / retrieval. Closing it requires:
- A different model (path B Qwen3-VL FT, or VDInstruct, or a span-correction model).
- A fundamentally different mechanism (graph-based GraphDoc-style classification, OCR-grid bidirectional encoder).

### 7.4 Score-modifying verifiers on a high-AP baseline are pure one-sided risk

Demoting predictions always hurts AP whenever the demoted predictions are TPs (which is most of the time on a high-AP baseline). The only safe AOL shape is **ADD missed fields, never modify existing prediction scores.** This reframes the original V6 ReAct failure: the problem wasn't that the gates were over-eager — it was that the entire "verify then modify" concept is structurally wrong against a strong baseline.

### 7.5 Hint injection narrows attention

Pre-pass hints (oracle candidates, schema reminders, field-specific instructions) systematically degrade coverage on the non-hinted fields. The model's attention budget is finite. Adding context costs accuracy elsewhere. Implication: keep the system prompt concise; surface hints only via downstream layers (validators, classifiers, ensembling) that don't compete for attention.

### 7.6 50-doc evaluations are unreliable; gate at 250 docs at minimum

Multiple experiments showed 50-doc gains (+1 to +3pp) that fully reversed on 500-doc full evaluation. The standard subset used in this project (`v5b_50.json`) has a distribution that consistently overstates refiner-style gains. Practical rule: don't trust any 50-doc result for shipping decisions; gate on 250-doc (using the first 250 docids of the standing baseline so the comparison is free) before paying for 500-doc.

### 7.7 Cluster-based few-shot is the largest single lever

Built into the dataset (`doc.annotation.cluster_id`), 75% val coverage, +17pp KILE. CLIP/SigLIP visual retrieval underperformed the dataset's own template grouping. For the 25% no-match docs, the recall drop is ~11.5pp on those docs — a cluster-prediction model (`cluster_infer.py`) plus retrieval-ICL fallback (SAIL) is the natural fix.

### 7.8 Refiner heuristics help on hard fields, hurt on easy ones

Per-field analysis shows a real split (helps ~6 fields, hurts ~6 fields, neutral ~24). Net is negative because the hurts dominate on the long tail. Selective per-fieldtype refining is the recoverable form.

### 7.9 Rossum-architecture maps cleanly onto our pieces

The composition that maps closest to Rossum's hybrid stack: (path B VLM grounding) + (classifier reranker) + (auditable confidence ensemble) + (recall-augmentation AOL). All four components exist or are in active development. The original document's Phase 4 (AOL) fits cleanly here, in its corrected ADD-only shape.

---

## 7. Directions Identified But Not Pursued

Each entry: what it is, why it's promising, what would be required.

### 8.1 VDInstruct (highest priority watch)

VDInstruct (arXiv:2507.09531, Nguyen et al. KAIST, July 2025) is an OCR-free multimodal LLM with dual spatial+semantic encoders, trained on DocILE in-distribution data. Reports 74.2% F1. Declared CC BY 4.0. Weights have not been released as of this writing — paper still under review.

**If the weights drop, this preempts every other path** for the DocILE benchmark specifically. A direct request to the corresponding author (Dinh Son Nguyen) is the lowest-cost-highest-EV move. Watch HuggingFace / KAIST Visual AI Group / authors' personal pages.

### 8.2 Path B variants (after the initial Qwen3-VL-8B LoRA lands)

- **More epochs.** Initial run is 3 epochs; 5-7 epochs may improve convergence, +$2-3 each.
- **Different LoRA rank / target modules.** r=64 with cross-attn included; or r=16 with only LM target modules.
- **Quantized base + higher LoRA rank** (QLoRA r=128) — fits in less VRAM, allows larger effective rank.
- **Train on synthetic data augmentation** (DocILE 100K synthetic split) to broaden coverage.

### 8.3 Span-correction model (the LayoutLMv3 idea redone)

Train a small bidirectional encoder (Qwen3-1.5B with JPT-style trick, or a fresh embedding head) on DocILE train pairs `(extractor candidate spans, gold spans)`. Smaller training task than full KIE; targets exactly the multi-word PCC mismatch that is the dominant remaining error. Estimated +5-10pp on top of any Sonnet-based extractor. Estimated cost: $5-10 RunPod, ~half day implementation.

### 8.4 Full HSDP build (LLaDA-V regions + Qwen3-LM JPT classifier + AOL)

The original document's full architecture, implemented:
- LLaDA-V fine-tuned on DocILE for semantic-region segmentation (this would be the first published application of a diffusion VLM to document region detection — high research value, high implementation cost).
- Qwen3-LM with JPT bidirectional trick + classification head trained on DocILE schema.
- Recall-augmentation AOL with calc verifier and HITL escalation.

Estimated cost: $10-20+ RunPod, several days of implementation. High variance — every component would be a first-of-kind application. Reserved as a research direction.

### 8.5 S³ Schema Scaffolding on a future diffusion LM

If diffusion LMs become the substrate for KIE (currently they don't outperform AR for our task), S³ scaffolding is a real technique for improving structural adherence. Not useful today because Sonnet is already at ~99% structural adherence on our schema — S³ solves a problem the current pipeline doesn't have.

### 8.6 DSPy MIPROv2 prompt optimization with Sonnet evaluator

The previous attempt used Haiku as the eval model; instructions optimized for Haiku's needs hurt Sonnet's stronger baseline. With Sonnet as the eval model, MIPROv2 (or its successor GEPA, +11% improvement reported on financial NER benchmarks) could plausibly add +1-4pp KILE. Estimated cost: $30 Sonnet API per optimization run. Worth it only if v2_ensemble is the long-term frozen baseline.

### 8.7 Cluster-prediction + SAIL retrieval for the no-match population

Both modules are built (`cluster_infer.py` at 70% top-1 accuracy, SAIL index built on 3070). End-to-end never evaluated. Estimated +2-3pp dataset-wide if it closes the no-match drag. Free if 3070 is available.

### 8.8 Selective per-field refining

Apply `refiners.py` only to the ~6 field types where per-field analysis shows positive lift; skip the rest. Estimated +0.5-1pp. ~15 lines of code change in `extract.py` plus a 500-doc re-run.

### 8.9 LIR-only classifier reranking

Side finding: per-fieldtype classifiers improve LIR F1 +1.6pp at threshold=0.3 even when KILE collapses (-2.5pp). Line-item field text patterns are more regular than KILE field patterns. Worth implementing as an LIR-only post-pass.

### 8.10 RapidTable LIR with newer 2026 TSR backbone

Combine YOLOv12-DocLayNet table region detection with a newer TSR model (TableMaster-2026 class). Estimated +3-8pp LIR. Untested. LIR is the secondary metric so this was deprioritized.

### 8.11 Conservative AOL rebuild with HITL-flag instead of score-modify

The calc verifier (gross = net + tax) is real and catches real anomalies. Rebuild with `use_only_for_ap=True` predictions for flagged cases (excludes them from F1 but includes them in AP) — preserves the AP signal while surfacing the contradictions for HITL review. Untested; the principle is consistent with the architectural lesson from §7.4.

### 8.12 Cross-model ensemble with a stronger affordable VLM

The Gemini-3-Flash-Preview test was inconclusive because thinking-disabled mode is too weak and thinking-enabled mode is too expensive. A cheaper-thinking variant, Llama-4-Vision, Mistral Pixtral, or DeepSeek-VL2 could compose. The `gemini_extract.py` pattern generalizes to any chat-completion VLM with image input.

### 8.13 Per-page page-classification + page-targeted extraction

For multi-page documents, run a small page-classifier (which page contains line items vs vendor info vs payment terms) and route each page to a focused extraction prompt. Particularly useful for the LIR task. Untested.

### 8.14 Auditable confidence ensemble (the production-readiness build)

Compose v2_ensemble + bbox_verify + classifiers + oracle as a multi-source confidence stack with HITL escalation. Doesn't move KILE AP directly but is the production-grade shape. Maps directly to Rossum's commercial architecture.

### 8.15 Train a doc-level field-presence classifier

For the recall-augmentation AOL (§4.3) to work better, the per-doc prior needs to be much stronger than cluster-membership. A small classifier trained on `(doc image, fieldtype) → P(field present)` would let the AOL fire only on high-confidence misses. Lifts the 4% hit-rate ceiling from the initial recall-AOL test.

### 8.16 GraphDoc-style GNN distillation from Sonnet predictions

Use Sonnet's output as supervision to train a GraphDoc-style GNN (bypassing the need for human annotations). Cheaper data than full DocILE annotation; could replicate GraphDoc's PCC-by-construction architecture. Speculative.

---

## 8. Open Questions

- **Does path B (Qwen3-VL-8B LoRA) clear v2_ensemble?** Pending result. If yes, the prompt-based stack becomes a candidate ensemble voice rather than the primary; if no, the marginal options in §7 become the only remaining levers.
- **Does VDInstruct release its weights?** If yes, almost certainly preempts every other approach for DocILE specifically.
- **Is the 25% no-match population recoverable?** §4.6 design is plausible; never end-to-end tested. Estimated +2-3pp dataset-wide.
- **Is there a span-correction model worth training?** §4.7 / §7.3 design is plausible; never built. Estimated +5-10pp specifically against the multi-word PCC mismatch error.
- **Does cross-model ensemble work with a stronger affordable VLM?** Gemini-3-Flash-Preview at thinking_budget=0 is too weak; with thinking is too expensive. Open whether a stronger affordable model changes this.
- **Does proper conservative AOL (HITL-flag, not score-modify) recover any AP?** §6.11 design is consistent with the architectural lesson; untested.

---

## 9. The Single Sentence

**Prompt-based extraction with Claude Sonnet + cluster-based few-shot + row-grouped OCR + per-prompt-variant ensemble achieves 46.48% KILE / 50.77% LIR on the DocILE 500-doc validation set, against GraphDoc's 71.25% / 75.93% target; the remaining gap is structural and requires either a fine-tuned grounding-VLM (path B, in progress), a span-correction model trained on DocILE word grids, or release of VDInstruct's DocILE-trained weights.**
