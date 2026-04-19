# Primer — Background for KNOWLEDGE_BASE.md

This document is for an AI researcher who is not familiar with the document AI field. It explains the terms, models, and techniques that the rest of the project documentation assumes you already know. Read this first if you're new to this corner of the field; then read KNOWLEDGE_BASE.md for the technical findings.

Familiarity assumed: general ML / deep learning, transformers, fine-tuning concepts, Python tooling. No prior exposure to document-extraction benchmarks needed.

---

## 1. The Field — Document AI / KIE / IDP

**Document AI** (sometimes called Intelligent Document Processing, IDP) is the subfield of ML that takes business documents — invoices, purchase orders, receipts, forms, contracts — and extracts structured data from them. The "documents" are usually scanned PDFs or images of paper that has been printed and then digitized again (so OCR is involved); even when the document was born digital, the layout is the dominant signal.

The two canonical tasks:

- **Document Classification:** "what kind of document is this?" (invoice vs receipt vs contract, or a customer-specific schema bucket).
- **Key Information Extraction (KIE):** "find the specific values for these named fields" (invoice number, total amount, vendor name, line items, etc.). This is the task this project addresses.

KIE is harder than it sounds because:

1. **Layouts are 2D.** A "Total" label and the "$1,234.56" value next to it are visually adjacent but linearly far apart in any text serialization. Models that read text left-to-right miss this.
2. **Schemas are open-ended.** Real customers want to extract dozens of fields, and which fields exist varies by document type (and by customer).
3. **The values are not always literally in the OCR.** Currency symbols get clipped, numbers get formatted with thousands separators that vary by locale, dates appear in dozens of formats. The "answer" is often an interpretation, not a substring.
4. **Localization matters.** Many production use cases need the bounding box of the value, not just the string — for human review, for redaction, for highlighting in a UI.

The commercial leader in this space is **Rossum** (rossum.ai), which builds a hybrid OCR + ML + rules + human-in-the-loop pipeline. The reference paper for their adaptive schema generation approach is RASG (arXiv:2405.20245).

## 2. The Benchmark — DocILE

**DocILE** ("Document Information Localization and Extraction") is the standard academic benchmark for KIE on business documents. It was introduced at ICDAR 2023 with a competition, but the dataset and evaluation code remain the standard for the field.

What DocILE provides:

- **5,180 train documents**, **500 validation documents**, **1,000 test documents** (with held-out test labels — leaderboard submission required to score). Plus ~6,680 annotated documents and a 100K synthetic split for pretraining experiments.
- **OCR layer.** Pre-computed DocTR (Mindee's OCR library) output for every page. This is critical: the dataset *fixes* the OCR layer so all submissions compete on the same word grid.
- **Annotations.** Per-document, per-field-type bounding boxes for two tasks:
  - **KILE** — 36 page-level field types (`amount_total_gross`, `vendor_name`, `iban`, `date_due`, etc.).
  - **LIR** — 19 line-item-row field types (`line_item_quantity`, `line_item_unit_price_gross`, etc.) grouped by row via a `line_item_id`.
- **Cluster IDs.** Each train document is tagged with a `cluster_id` representing the document template it follows (e.g., "all invoices from this specific vendor template"). This is curated by the dataset authors, not derived from a model. Training documents in the same cluster look very similar to each other; this is a powerful (and sometimes underused) signal for in-context learning.

The metric is harsh, and that harshness shapes everything (see §3).

## 3. The Metric — PCC-IoU and Why It's Unforgiving

When you train a normal object detector and evaluate it, you score predictions by IoU (intersection-over-union) — how much your predicted box overlaps the ground truth box, with a typical threshold of 0.5 or 0.7. DocILE does not use plain IoU.

**Pseudo-Character Centers (PCCs).** For each *snapped* OCR word on the page, DocILE computes one center point per character (so the word "TOTAL" contributes 5 PCCs evenly spaced across its bounding box). The collection of all these per-character points across all words on the page is the PCC set for that page.

**PCC-IoU.** A predicted bounding box matches a gold bounding box for a given field if and only if:

- Same field type
- Same page
- The set of PCCs covered by your bbox equals the set of PCCs covered by the gold bbox **exactly** (PCC-IoU = 1.0 by default).

This is a binary set-equality predicate. One extra word inside your prediction's bbox = a different PCC set = zero. One missing word = zero. There is no partial credit.

**Why the dataset designers chose this.** Production document-extraction systems care about word-level precision. A "Total" extraction that includes the label "Total:" along with the value "$1,234.56" is wrong in production — the downstream system would record the wrong value. PCC-IoU=1.0 punishes this case.

**What this implies for model architecture.** Any model whose output bounding boxes do not align to DocILE's specific snapped OCR word grid scores ~0% regardless of how semantically correct the extracted value is. This single constraint rules out:

- OCR-free models (Donut, etc.) that have their own internal OCR — their token coordinates can never match the dataset's snapped grid.
- Generative models that output text and require post-hoc alignment back to OCR words — every alignment ambiguity becomes a zero.
- Pre-2025 layout-aware models that snap to their own OCR backbone (often LayoutLMv3's own pre-training OCR).

The only safe shapes are: (a) pick from the existing OCR word grid (predict word_ids, derive bboxes as the union of selected words' bboxes), or (b) output bounding boxes and post-snap to the OCR grid via deterministic IoU-based assignment.

**Snapping.** "Snapped" OCR words are produced by Otsu binarization and a two-phase margin-shrink algorithm on the page image crop, then cached on disk per document. The evaluator uses these cached snapped bounding boxes as the source for PCC computation. DocILE ships a 200 DPI snap cache; changing DPI without rebuilding the cache breaks alignment.

The two scoring tasks:
- **KILE** is scored as **mean Average Precision** (mAP), micro-averaged across the 36 field types.
- **LIR** is scored as **F1**, with a two-stage matching: first align predicted line-item-row groups to gold groups via maximum-weight bipartite matching, then run PCC-IoU field matching within each matched pair.

## 4. The Models You'll Read About

### 4.1 Vision-Language Models (VLMs)

A VLM is a transformer-based model that consumes both an image and text and produces text. The architecture is typically: a vision encoder (often SigLIP, CLIP-ViT, or a custom ViT) projects the image into "soft" tokens that are concatenated with text tokens and fed into a decoder LM. Examples relevant to this project:

- **Claude Sonnet (Anthropic)** — closed-source, multimodal, strong instruction-following. We use it as the primary extractor.
- **Qwen3-VL-8B** (Alibaba, Apache 2.0) — open-weights frontier VLM with native grounding tokens (`<|object_ref_start|>...<|object_ref_end|>` paired with `<|box_start|>(x1,y1),(x2,y2)<|box_end|>`). Trained to emit normalized 0–1000 coordinate bounding boxes inline with text. Good fine-tuning candidate.
- **Gemini 3 Flash** (Google) — closed-source, fast multimodal model. Has a "thinking" mode that consumes output tokens for reasoning.
- **GLM-OCR-0.9B** (Zhipu AI / zai-org) — small (0.9B parameters) VLM specifically tuned for OCR + layout + table-recognition tasks. Not specifically tuned for arbitrary KIE schemas, which matters (see §6).

### 4.2 Layout-aware encoder-decoder models

The previous generation of document AI models. They consume OCR text, OCR bounding boxes, and the page image jointly, with cross-modal attention. Examples:

- **LayoutLMv3** (Microsoft, 2022) — bidirectional encoder over text + bbox + image patches; pre-trained on IIT-CDIP. The dominant pre-2024 baseline. Considered outdated for new work in 2025-2026 but still appears in the literature.
- **DocFormer**, **DocLLM**, **mPLUG-DocOwl** — newer variants with similar architectures.
- **Donut** (Naver Clova) — OCR-free; reads the image directly with a Swin encoder and decodes structured text. Cannot align to externally-specified OCR grids by design.

### 4.3 Discriminative graph models

The DocILE SOTA holder is **GraphDoc**: a Graph Neural Network that builds a graph over OCR words (nodes) connected by spatial edges (adjacency, alignment), then runs message-passing and a per-word per-fieldtype classifier head. Output is a soft label per word per fieldtype; the predicted bounding box for each field is just the bbox of the chosen word(s). Because predictions are bound to the existing OCR grid by construction, GraphDoc satisfies PCC-IoU=1.0 trivially. Trained on DocILE train.

### 4.4 Diffusion LMs (and Diffusion VLMs)

A newer paradigm that replaces autoregressive (AR) left-to-right text generation with parallel denoising over masked tokens. Inspired by image diffusion models. Examples:

- **LLaDA** (arXiv:2502.09992) — text-only diffusion LM at 8B scale.
- **LLaDA-V** (arXiv:2505.16933) — multimodal extension; has GUI bounding-box grounding capability.
- **DiffusionVL** (arXiv:2512.15713) — converts AR VLMs to diffusion via fine-tuning.

The argument for diffusion in document AI: AR models linearize a 2D layout into a 1D sequence and can't revise earlier tokens once later context appears. Diffusion processes the entire sequence in parallel, in principle handling multi-column layouts and tables better. Empirically, no diffusion VLM has been evaluated on DocILE-class KIE tasks at the time of writing — the architectural argument is plausible but unvalidated for this domain.

### 4.5 Discriminative bidirectional decoders

Decoder-only LLMs (Llama, Qwen, Mistral families) have causal attention masks that prevent each token from attending to future tokens. For bidirectional tasks like NER token classification, this is a handicap. The **Just Pass Twice (JPT)** trick (arXiv:2604.05158) feeds `[input + input]` to a causal LM; in the second copy, every token can attend back to the full first copy, recovering bidirectional context without changing the model. Combined with a LoRA + classification head, this lets a frontier decoder LM do token classification competitively with bidirectional encoders.

The direct predecessor is **Echo Embeddings** (Springer et al., ICLR 2025, arXiv:2402.15449), which used the same trick for sentence-level embeddings.

### 4.6 The relevant DocILE-trained model

**VDInstruct** (arXiv:2507.09531, KAIST, July 2025) — an OCR-free multimodal LLM with dual spatial+semantic encoders, trained with DocILE in-distribution. Reports 74.2% F1. Declared CC BY 4.0, but weights have not been released as of this writing. If the weights ever drop, this model preempts most of the work in this repo for the DocILE benchmark specifically.

## 5. The Techniques

### 5.1 Zero-shot vs few-shot vs fine-tuned

- **Zero-shot.** Apply a pretrained model directly to the task with only a system prompt and the input. No examples, no gradient updates. Maximum generality, weakest task-specific accuracy.
- **Few-shot (in-context learning).** Include 1–N annotated examples in the prompt as demonstrations. The model imitates the format. No gradient updates. The "few-shot retrieval" question — *which* examples to include — is itself a subfield (CLIP/SigLIP retrieval, BM25, dataset-specific cluster matching).
- **Fine-tuning.** Update model weights on task-specific data. Full fine-tuning updates every parameter; LoRA fine-tuning updates only small low-rank adapter matrices (typically ~0.1-1% of the total parameter count) and is much cheaper.

In this project: the standing best is few-shot (Claude Sonnet + cluster-based example retrieval). The biggest active training experiment (path B) is LoRA fine-tuning Qwen3-VL-8B on DocILE train.

### 5.2 LoRA (Low-Rank Adaptation)

A fine-tuning technique that decomposes weight updates as `W' = W + B @ A` where `A` is `(rank, dim)` and `B` is `(dim, rank)`. With `rank=32` and `dim=4096`, the LoRA adapter has ~262K parameters per matrix vs the full ~16M. Vastly cheaper to train, vastly cheaper to store (adapters are usually 100-500 MB for a 7-8B base), and you can swap multiple LoRAs over the same base model.

Hyperparameters that matter:
- **`r` (rank).** Higher rank = more capacity but more memory and slower convergence. Typical values 8-128.
- **`alpha`.** Scales the LoRA contribution (`B @ A * (alpha / r)`). Usually `alpha = r` or `alpha = 2r`.
- **`target_modules`.** Which weight matrices get adapters. Common: query/key/value/output projections in attention; gate/up/down in MLP. Often vision encoders are frozen entirely for VLM fine-tuning.

### 5.3 Quantization (briefly)

Run inference with weights stored in 4-bit or 8-bit instead of 16-bit, trading minor accuracy for substantial VRAM savings. INT4 quantization (e.g., via `bitsandbytes`) cuts a 7B model from ~14 GB BF16 to ~4-5 GB. Necessary on small GPUs (8 GB cards). Does not significantly degrade extraction-task accuracy in our experience but adds latency.

### 5.4 Ensembling

Run the same task multiple times with different inputs (different prompts, different temperatures, different models entirely) and merge the outputs. The merge can be:
- **Voting** (intersection — only keep predictions all variants agree on). Tends to hurt recall.
- **Union** (keep anything any variant proposed). Tends to hurt precision.
- **Score-weighted merge** (per item, take the highest-confidence prediction across variants). The middle path — what `ensemble.py` does in this project.

Ensembling can compose models with different failure modes (one model misses Field X, another catches it). It compounds inference cost linearly with the number of variants.

### 5.5 Retrieval-Augmented In-Context Learning (RAG-ICL, SAIL)

Instead of using a fixed set of few-shot examples, retrieve the most relevant examples from a training pool per input. The retrieval can use embeddings (CLIP for images, sentence-BERT for text), entity-level similarity (SAIL), or hand-curated metadata (DocILE's `cluster_id`).

### 5.6 Prompt optimization (DSPy MIPROv2, GEPA)

A relatively new family of techniques that automatically optimize the system prompt and few-shot examples for a given task by repeatedly generating candidate prompts, scoring them on a held-out set, and propagating gradients-on-prompts. **DSPy** (Stanford) is the dominant framework. **MIPROv2** is its primary optimization algorithm; **GEPA** (July 2025) is a successor with reported +11% improvements on financial NER benchmarks.

These methods are expensive (they rerun the full pipeline N times during optimization) but can find non-obvious prompt structures that human iteration misses. The catch: they need a strong evaluator model to score candidates. Optimizing for a weak evaluator (Haiku) and deploying with a strong one (Sonnet) often regresses, because the model learns to game the weak evaluator.

### 5.7 Agentic loops, tool use, ReAct, AOL

A pattern where an LLM iteratively decides what to do next: extract a field, call a tool to verify it (calculator, regex, lookup), re-extract if the tool flagged an issue. The original ReAct paper introduced this for QA tasks. Variants for document extraction include "Agentic Orchestration Layers" (AOL), where one model orchestrates a pipeline of specialized sub-agents.

The architectural hazard: any verifier that can *modify or delete* predictions on a high-AP baseline tends to hurt scores, because the verifier's wrongness rate against true positives is non-zero and AP is sensitive to rank order. The only safe shape is verifiers that *add* missed predictions and never touch existing ones, OR verifiers that only flag for human review without changing the score.

### 5.8 Grounding tokens

In modern VLMs (Qwen-VL family, GLM-V family, some others), bounding boxes are emitted as special token sequences interleaved with text — e.g., `<|object_ref_start|>vendor_name<|object_ref_end|><|box_start|>(150, 220),(450, 245)<|box_end|>`. Coordinates are usually normalized to 0–1000. Models trained with grounding can output spatial answers natively; models without grounding training cannot, even if they "see" the page.

## 6. Concepts You'll Encounter Specific to This Project

### 6.1 The 50-doc trap

Early in the project, experiments were gated on a 50-document subset to save API cost. Multiple times an improvement showed +1 to +3 pp KILE on 50 docs, then reversed on the full 500. The fixed 50-doc subset had a distribution that systematically overstated certain kinds of gains. The lesson: gate on 250 docs minimum; ideally use the first 250 docids of the standing baseline so the comparison is free.

### 6.2 The "cluster_id" lever

The single largest non-architectural lever in this project. DocILE training annotations carry a `cluster_id` that groups documents by template. 75% of validation documents have at least one annotated training document in the same cluster. Picking that document as the few-shot example in the prompt accounts for ~+17 pp KILE over a no-cluster baseline. CLIP and SigLIP visual retrieval underperform the dataset's own clustering.

### 6.3 The no-match population

The other 25% of validation documents have no matching training cluster. They get a generic zero-shot fallback and lose ~11.5 pp KILE relative to the cluster-matched docs. Closing this gap is a known open direction — candidate fixes include training a cluster-prediction model from page images and SAIL-style entity-level retrieval across clusters.

### 6.4 The specialization-fit framework

Not every modern open-weights VLM that fits on your GPU is a useful starting point. The relevant question is "how specialized is this model for our exact task, and what's the cheapest move to close the gap?" Candidate categories:
- **Already on-task** (drop in, $0).
- **Adjacent task transferable** (prompt + few-shot, ~free).
- **Capable architecture, wrong training** (LoRA fine-tune, ~$5).
- **Wrong-task specialization** (substantial retraining needed; cost worse than fine-tuning a stronger base).
- **Architectural mismatch** (cannot close at any cost).

A model that is SOTA on a different benchmark task does not transfer to this one without explicit work to close the specialization gap. This is the lesson that buried the GLM-OCR experiment in this project — GLM-OCR is #1 on OmniDocBench (text recognition + reading order + table structure) but does not extract schema-conformant fields well, because it was never trained for that.

### 6.5 "Refiner" / "Validator" / "Verifier"

These three terms refer to post-extraction modules in this codebase:
- **Refiner** — modifies an extracted prediction (trims label words, adjusts span boundaries, picks a contiguous run from a noisy selection). `refiners.py`.
- **Validator** — scores an extracted prediction's format validity (IBAN mod-97 checksum, BIC regex, date parsing, amount parsing). Returns a confidence; doesn't modify. `validators.py`.
- **Verifier** — runs a separate extraction pass to confirm a prediction (e.g., re-prompt the LLM with focused context to check the answer). `bbox_verify.py`.

All three are off in the production-best system. The first two were tested at scale and confirmed net-negative; the third was within-noise.

### 6.6 "Path A" / "Path B" / "Path C"

Internal shorthand for the strategic options considered:
- **Path A** — drop in an off-the-shelf 2026 SOTA OCR/doc model (GLM-OCR-0.9B). Tested, buried.
- **Path B** — LoRA fine-tune a strong general VLM (Qwen3-VL-8B) on DocILE train. In progress.
- **Path C** — build the full HSDP+AOL architecture from `idea_dllms.md` (diffusion VLM regions + JPT classifier + agentic loop). Reserved as a research direction; not pursued for cost reasons.

### 6.7 "v2" vs "v2_ensemble" vs "V5b"

Project iteration shorthand:
- **V1** — initial zero-shot Claude pipeline. ~27% KILE.
- **V2** — V1 + cluster-based few-shot retrieval + row-grouped OCR words. 44.61% KILE / 50.89% LIR on 500 docs. Long the standing best.
- **V5b** — V2 + refiner + validator. Regressed on full-scale eval.
- **v2_ensemble** — V2 run three times with prompt/temperature variants, score-weighted merged. 46.48% KILE / 50.77% LIR. The current standing best.

## 7. Reading Order

If you've read this primer, the suggested next reads:

1. **`README.md`** at project root — quick orientation, how to run things.
2. **`KNOWLEDGE_BASE.md`** — the full technical record. Sections of greatest value:
   - §1.2 (the metric) and §2 (standing best) — for grounding.
   - §6 (what was tried and didn't work) and §7 (architectural lessons) — the load-bearing technical content.
   - §8 (directions identified but not pursued) — the open questions.
3. **`EVAL_SPEC.md`** — the authoritative reference for the DocILE eval mechanics, with code-line citations.
4. **`idea_dllms.md`** — the original architectural proposal that shaped the recent work, with appendices showing how each component was verified or buried.

For the codebase, every Python module in `src/beat_docile/` carries a status banner (`[ACTIVE]`, `[EXPERIMENTAL]`, `[RESEARCH-BURIED]`, `[ARCHIVED]`) and a cross-reference to the relevant KNOWLEDGE_BASE.md section. The README has a module map.

## 8. References

The papers most worth knowing for this domain:

- **DocILE benchmark** — Šimsa et al. *DocILE Benchmark for Document Information Localization and Extraction*. ICDAR 2023. (`references/papers/docile-benchmark.pdf`)
- **GraphDoc** — the DocILE SOTA holder. See the DocILE leaderboard for the citation.
- **RASG (Rossum)** — *Schema-Agnostic Document Extraction*. arXiv:2405.20245. The reference for how a commercial production stack handles per-customer adaptation + auditable confidence.
- **LayoutLMv3** — Huang et al. ACL 2022 / arXiv:2204.08387. The dominant pre-2024 baseline.
- **Qwen3-VL** — arXiv:2511.21631. The current open-weights frontier we're fine-tuning in path B.
- **VDInstruct** — Nguyen et al. arXiv:2507.09531. DocILE-trained model; weights pending release.
- **LLaDA** — Nie et al. arXiv:2502.09992. The discrete text diffusion LM at 8B scale.
- **JPT (Just Pass Twice)** — Ewais et al. arXiv:2604.05158. The bidirectional-decoder trick for token classification.
- **Echo Embeddings** — Springer et al. arXiv:2402.15449. The direct predecessor of JPT for sentence embeddings.
- **DSPy** — Khattab et al. *DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines*. ICLR 2024. The prompt-optimization framework.

For dataset-related papers (DocVQA, FUNSD, CORD, SROIE, OmniDocBench), see the DocILE paper's related work section.
