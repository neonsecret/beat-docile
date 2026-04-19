# Re-architecting Zero-Shot Document Information Extraction: A Hybrid Agentic-Diffusion Pipeline

> **Update 2026-04-19 evening — current best:** `v2_ensemble` (3-variant Sonnet ensemble: T=1.0 + T=0.3 + alt-prompt, per-field merge via existing `ensemble.py`) = **46.48% KILE / 50.77% LIR on 500 val docs**. Replaces v2 (44.61%) as standing baseline. Held through 50→250→500 gate progression. Cost: 3× Sonnet API at inference. Predictions: `predictions/v2_ensemble_500.json`. Path B (Qwen3-VL-8B LoRA) still cooking on RunPod — ETA 22:20 UTC.

---

### Abstract
The evolution of Document AI from early multimodal encoders (e.g., LayoutLMv3) to massive Vision-Language Models (VLMs) has successfully generalized zero-shot Key Information Extraction (KIE). However, relying on Autoregressive (AR) generation to serialize 2D spatial layouts into 1D text introduces fundamental topological constraints and exposure bias. This manifests as mathematically incorrect extractions and hallucinated entities, an unacceptable margin of error for enterprise systems (e.g., Rossum's use-cases). 

This document proposes a **Hybrid Spatial-Discriminative Pipeline (HSDP) wrapped in an Agentic Orchestration Layer (AOL)**. By decoupling global spatial reasoning (via Diffusion VLMs) from dense token extraction (via Bidirectional Token Classification), we can achieve the zero-shot reasoning capabilities of frontier models (Qwen 3.6, Gemini 3.x, DeepSeek) while mathematically eliminating generative textual hallucinations.

---

### 1. Problem Formulation: The Limits of AR Generation in 2D Topologies
The transition away from early CNN/Transformer pipelines (like LayoutLMv3’s linear patch projection) was driven by the need for zero-shot generalization. However, substituting these pipelines with generative AR models introduces a severe architectural mismatch:

1.  **Topological Serialization:** AR models decode sequences left-to-right. Documents (especially multi-column tables and nested line items evaluated in DocILE) are 2D grids. Forcing an AR model to linearize a table creates alignment failures.
2.  **Exposure Bias & Reversal Curse:** If an AR model misinterprets a layout early in its sequence, it cannot look ahead or revise past tokens. This leads to cascading hallucination.
3.  **The "Generative" Risk:** Asking a model to *generate* a numerical total (e.g., `$15,400.00`) carries a probabilistic risk of token permutation. For high-stakes KIE, the optimal extraction error rate is strictly 0%.

---

### 2. Recent Architectural Shifts (2025–2026)
To solve these bottlenecks, the research community has validated two distinct architectural paradigms that bypass standard causal AR generation:

#### 2.1 Discrete Text Diffusion (dLLMs) for Global Spatial Awareness
Models like **LLaDA** (Large Language Diffusion with mAsking) [1] and **DiffusionVL** [2] have demonstrated that parallel denoising is vastly superior for layout-heavy tasks. 
*   **Rationale:** Rather than generating text sequentially, diffusion models start with a masked sequence and iteratively unmask all tokens simultaneously. This allows the model to process the 2D document context globally, anchoring extracted numbers to their broader layout.
*   **Scaffolding:** Frameworks like **S³ (Self-Adaptive Schema Scaffolding)** [3] inject rigid JSON schemas directly into the diffusion context. Because the sequence refines globally, the schema acts as a strict structural anchor, improving structural adherence by over 60% compared to AR baselines.

#### 2.2 Pseudo-Bidirectional Token Classification
If 100% deterministic text accuracy is required, generation must be replaced with strict Extractive Question Answering (Token Classification). Causal LLMs historically fail here because their attention masks prevent them from looking "forward" at trailing context.
*   **The JPT Solution:** The **Just Pass Twice (JPT)** framework [4] solves this by concatenating the input sequence `[Input + Input]`. In the second pass, the causal mask allows the target token to attend backward to the entirety of the first sequence. This extracts rich, bidirectionally-aware hidden states from massive pre-trained LLMs, allowing them to accurately classify deterministically extracted OCR tokens.

---

### 3. Proposed Architecture: Hybrid Spatial-Discriminative Pipeline (HSDP)
We propose synthesizing these findings into a unified, non-generative extraction pipeline.

**Phase 1: Deterministic Token Graphing (Ground Truth)**
*   Extract unalterable digital text, exact font attributes (metadata), and bounding boxes from the PDF binary. Apply modern high-fidelity OCR solely for rasterized regions.
*   *Output:* A spatial graph $T = \{ (t_1, x_1, y_1), ..., (t_n, x_n, y_n) \}$. By forcing all downstream models to select from $T$, we mathematically bound textual hallucination to 0%.

**Phase 2: Global Spatial Planning (Diffusion VLM)**
*   Instead of asking a VLM (e.g., Qwen 3.6-VL) to extract text, we task it purely with semantic segmentation.
*   *Mechanism:* The VLM analyzes the image layout and outputs semantic bounding boxes (e.g., $B_{table}$, $B_{total\_block}$, $B_{supplier\_address}$). Diffusion-based spatial planners inherently process the 2D topology, avoiding cascading alignment errors.

**Phase 3: Bidirectional Extractive Decoding (Classification)**
*   Isolate the raw tokens from $T$ that fall within the semantic bounding box $B_{table}$.
*   Feed these tokens through a text-only LLM utilizing the **JPT framework**.
*   *Mechanism:* A linear classification head maps the bidirectional hidden states of the specific tokens to the target schema (e.g., tagging token at `(150, 400)` as `[B-LINE_ITEM_QTY]`).

---

### 4. The Agentic Orchestration Layer (AOL)
A static pipeline fails on unstructured enterprise data (e.g., 50-page highly variable payloads with distractor pages). We wrap the HSDP in an active perception layer driven by an orchestrating LLM.

**A. Macro-Triage & Scouting (Compute Optimization)**
*   The Agent ingests low-resolution thumbnail grids or metadata representations of the payload.
*   Using tool calls (e.g., `drop_page(idx)`, `crop(x,y,w,h)`), it filters out terms-and-conditions or marketing inserts *before* invoking the heavy HSDP models, significantly reducing latency and false-positive exposure.

**B. Deterministic Tool-Augmented Verification**
*   Once HSDP outputs a structured JSON, the Agent runs verification using deterministic external tools.
*   *Example:* The Agent reads the classified `Line Items` and invokes `calc(sum(lines))`. If the calculated sum does not match the classified `Total`, the Agent detects an anomaly.

**C. Iterative Self-Correction**
*   Upon detecting an anomaly, the Agent does not crash; it adjusts the pipeline's hyperparameters.
*   *Action:* The Agent may re-invoke Phase 2 with a localized high-resolution crop (`re_run_planner(region, hi_res=True)`) or append a negative constraint to Phase 3. It mimics human cognitive "re-reading" when math fails.

---

### 5. Research Implications & Rationale for Benchmarks (e.g., DocILE)
This architecture maps directly to the complexities evaluated in benchmarks like DocILE, which emphasizes Line Item Extraction (LIE) and complex table parsing:

1.  **Resolves the Multi-Page Context Bottleneck:** Generative LLMs lose context over 50-page sequences. The Agentic Scout isolates the context dynamically.
2.  **Solves Table Serialization:** By isolating the table via a spatial VLM and extracting via coordinate-aware token classification, the model natively respects the grid structure, solving the alignment issues of 1D autoregression.
3.  **Auditable Confidence:** Because the classification is tied to deterministic coordinates, the system leaves an auditable trail (e.g., "The system flagged this Total because its bounding box was isolated in Phase 2, and Phase 3 classified it with 98% confidence"). This is vital for Human-In-The-Loop (HITL) UI design in platforms like Rossum.

---

### References & Foundational Literature
1. **Nie et al. (Feb 2025).** *Large Language Diffusion Models (LLaDA)*. Demonstrates non-autoregressive, parallel denoising matching LLaMA-3 scale, establishing the baseline for global-context spatial text processing.
2. **Wang et al. (Dec 2025).** *DiffusionVL: Translating Autoregressive Models into Diffusion Vision Language Models*. Validates the superiority of block-decoding and diffusion fine-tuning for multimodal visual-spatial reasoning.
3. **Ye et al. (2025).** *Unveiling the Potential of Diffusion Large Language Model in Controllable Generation*. Outlines the Self-Adaptive Schema Scaffolding (S³) framework, proving diffusion models vastly outperform AR models in rigid structural schema adherence.
4. **Ewais, Hashish, Ali (WitnessAI, April 2026).** *Just Pass Twice: Efficient Token Classification with LLMs for Zero-Shot NER*. arXiv:2604.05158. Outlines the sequence-concatenation methodology for extracting bidirectional representations from causal decoder-only models. Note: requires trained LoRA adapters + classification head — "zero-shot" refers to unseen entity types, not zero training. Direct prior art: Echo Embeddings (Springer et al., ICLR 2025, arXiv:2402.15449).

---

## Appendix — Verification & Reassessment (2026-04-19)

After spawning a 6-scout research team to verify every citation and survey the April-2026 doc-AI landscape, the architectural thesis (discriminative extraction over a deterministic OCR grid is the right way to escape generative AR's word-set mismatch) is **confirmed**. The specific recipe (Diffusion VLM for spatial planning + JPT for classification + agentic loop) is **only partially viable** off-the-shelf.

### Citation verification

| Cited | Status | Note |
|---|---|---|
| LLaDA (arXiv:2502.09992) | ✅ Real, MIT, ICML/NeurIPS 2025 | Repo: github.com/ML-GSAI/LLaDA. **Zero document-AI evaluation exists.** 8B FP16 doesn't fit on 8GB 3070; MoE variant might. |
| LLaDA-V (arXiv:2505.16933) | ✅ Real, CVPR 2026 poster | Multimodal extension. Has GUI-bbox grounding capability, but no document-region segmentation training. Would require fine-tuning. |
| DiffusionVL (arXiv:2512.15713) | ✅ Real, HUST, Dec 2025 | Outputs text, not coordinates. Does not natively do document semantic segmentation. |
| S³ Schema Scaffolding (arXiv:2507.04504) | ✅ Real, USC, July 2025 | Authorship was wrong ("Ye" — actual: Xiong/Cai/Li/Wang). Solves structural-adherence on diffusion LMs. **Sonnet already at ~99% structural adherence; this is solving a problem we don't have.** |
| JPT (arXiv:2604.05158) | ✅ Real, WitnessAI, April 2026 | Not anonymous. Requires LoRA + bilinear classifier head trained on target schema. Qwen3-4B INT4 fits 3070 inference; LoRA training needs 5090. |

### Where the proposed pipeline falls down

- **Phase 2 (Diffusion VLM as semantic segmenter):** No published model performs document-region semantic segmentation via diffusion VLM zero-shot. LLaDA-V's GUI-bbox grounding shows the *capability is plausible*, but applying it to invoices/POs would be the first publication. High variance, no baseline.
- **Phase 3 (JPT):** Real and applicable, but not "zero-shot" in the colloquial sense — requires LoRA fine-tune on DocILE train + classification head over 36 KILE field types. Workable, several days of work, ~$5 RunPod.
- **Phase 4 (Agentic loop):** **Considered untested at proper tuning.** A first-iteration v6 ReAct stack hit 22.7% KILE, but the failure modes were a triage gate that dropped 48% of fields (over-aggressive "skip if uncertain" threshold) and a cross-field verifier that deleted whole field arrays on word_id overlap (delete-on-conflict semantics). Both are tunable prompt/threshold parameters, not fundamental architecture limits. The right knobs would be: gates skewed toward keep-by-default, verifier demotes-confidence rather than delete-both, fewer per-field loops. v5b refiner+validator at 41.79% is a milder version of the same pattern and shows the approach can land near baseline without nuking recall. **Reserved for revisit after path A/B; if a better baseline exists from path B, an AOL-style verification layer on top of it is genuinely promising (the document's section 4 still applies).**

### What the landscape sweep uncovered (NOT in the original document)

The original idea overlooked off-the-shelf 2026 doc-AI models that already implement the discriminative-extraction philosophy:

- **GLM-OCR 0.9B** (zai-org, Mar 2026, MIT, **4GB VRAM**) — claims #1 on OmniDocBench (94.62), beating Qwen3-VL-235B and Gemini-3-Pro. Has KIE training + region grounding. Fits the 3070 trivially. Repo: github.com/zai-org/GLM-OCR. **Top spike candidate — claim needs validation on DocILE.**
- **VDInstruct** (Jul 2025, CC BY 4.0, arXiv:2507.09531) — **literally trained on DocILE data**. Highest-prior zero-shot model for our exact benchmark, conditional on weights actually being released.
- **LayTextLLM-Zero** (ACL 2025) — +15.2% KIE vs prior OCR-LLM SOTA, slots into our existing OCR pipeline.
- **Qwen3-VL-8B** (Apache 2.0, Nov 2025) — DocVQA 97% zero-shot, native bbox grounding. 4-6h LoRA on RunPod 5090 (~$3-5) is the highest-EV custom-training move.

### Reassessed plan (April 2026 baseline = v2 at 44.61% KILE / 50.89% LIR)

| # | Path | Est. lift | Cost | Risk |
|---|---|---|---|---|
| A | **GLM-OCR-0.9B drop-in spike on local GPU** | +10-25pp (if claim transfers) | ~free | Verify benchmark before committing — this is starting now |
| B | LoRA-FT Qwen3-VL-8B on DocILE train | +15-25pp | ~$5 RunPod, 4-6h | Medium, known recipe |
| C | Build full HSDP from this document (LLaDA-V regions + Qwen3 JPT classifier + agent) | unknown — could be 0 | $10-20+, days | High — every component would be the first published doc-AI application |

**Decision:** proceed with A first. C remains a research-direction option after A and B are exhausted.

---

## Appendix B — Negatives Audit (2026-04-19)

After the user pointed out that "every regression we got actually could've been fine but the agents didn't find bugs" and that we've spent days burying approaches without proper testing, this audit re-classifies every previously-killed idea. The new house rule: **negative results are bugs until proven otherwise** — declared dead requires architectural impossibility, not "first try didn't work."

### Truly dead (architectural impossibility — keep dead)

| Approach | Why it fundamentally cannot work |
|---|---|
| **Code-Factory Python scripts** | No access to DocILE's exact OCR word grid → produced bboxes can never PCC-align. Fundamental input mismatch, no tuning fixes it. |
| **Donut / OCR-free models** | Generated token coordinates don't match DocILE's pre-computed snapped word bboxes. PCC-IoU=1.0 is binary on word-set match. |
| **Non-snapped or non-DocTR bbox origins** | Evaluator hardcodes 200 DPI snapped geometry. Anything else scores ~0% regardless of correctness. |
| **GLiNER post-processing on Claude output** | Claude makes one categorical pick; GLiNER needs competing candidates to disambiguate. Wrong layer by design. |
| **LayoutLMv3-base/large fine-tune** | Pre-2025 architecture violates the modern-models-only hard rule (Jan 2026 retrospective). Even if inference threshold bug were fixed, would not be a strategic bet. **Pod stopped, checkpoint abandoned.** |
| **Self-consistency 3-sample voting** | Recall drops > precision gains by mathematical design (intersection of correct fields shrinks faster than wrong fields). |
| **Targeted 2nd API call for financial codes** | These fields together carry <2% of AP weight; even perfect recall on them moves overall <0.5pp. Math is correct. |

### Reclassified — needs proper testing, ranked by expected impact

These were declared dead after a single failed configuration. Each has a tunable knob that wasn't properly explored.

| # | Approach | Original verdict | Real status | Impact estimate | Cost to retest |
|---|---|---|---|---|---|
| 1 | **PP-DocLayoutV3 region scoping (semantic-crop mode)** | "Negative both modes" | Only tag-mode and word-filter modes tested. Never tried the GLM-OCR-style "crop region → re-prompt Claude on cropped region." Path A is essentially testing this approach with GLM-OCR's own KIE; same idea applies to Claude. | **+5-15pp KILE possible** if regions tighten Claude's word-id selection | Medium — needs GPU for layout model + Sonnet API |
| 2 | **V6 ReAct (agentic loop)** | "Catastrophic, 22.7% KILE" | Triage gate dropped 48% of fields by skipping; cross-field verifier deleted on word_id overlap. Both tunable thresholds. **Confirmed today: this is one prompt iteration's failure, not a tech failure.** | **+5-15pp possible** with keep-skewed gates + demote-confidence verifier | Medium — Sonnet API only, no GPU. ~1 day of prompt eng. |
| 3 | **Field-instructions dict** | "-8.9pp KILE — confused amount_net ↔ amount_gross" | One specific instruction wording was bad. Per-field guidance is otherwise valuable. | **+2-5pp possible** with better wording (avoid forcing classification, only describe format) | Low — Sonnet A/B on 50 docs, ~2h |
| 4 | **V5b refiner+validator tuning** | "41.79% KILE — regressed 2.8pp from v2" | Single config tested. Refiner is heuristic (per-field span cleaners) and validator is regex-based. Both have many knobs unexplored. | **+3-8pp possible** if per-field refiners stop over-cleaning addresses/multi-word fields | Low — modify `refiners.py`, A/B on 50 docs |
| 5 | **RapidTable LIR with newer 2026 TSR backbone** | "-21pp LIR cropped" | One TSR backbone tested. Newer models (TableMaster-2026, latest StructEqTable) untried. Plus the cropping was generic-region not table-region. | **+3-8pp LIR possible** with better TSR + table-region cropping | Medium — research + integration |
| 6 | **Aggressive contiguity in `_merge_bboxes`** | "-8pp KILE" | One algorithm (greedy contiguous-run merge). Span-aware variants (only merge if same y-row + same predicted text) untried. | **+0-3pp possible** | Low — algorithm change in `extract.py`, A/B on 50 docs |
| 7 | **"ONLY ONE: gross OR net" prompt rule** | "Made Claude skip valid amounts" | Rule was binary. Conditional variants ("when both visible, prefer gross over net unless explicitly labeled NET") untried. | **+0-2pp possible** | Low — single prompt edit |
| 8 | **bbox_verify (3-pass evidence verifier)** | "+1.36pp on 50d, within noise" | Built but never run on 500-doc to confirm. Default off. May be a real win that 50-doc noise hid. | **+1-3pp possible**, may already exist | Low — flip env flag, run 500-doc |

### Reclassified items — actual verdicts after testing (2026-04-19 evening)

| # | Item | Test result | Verdict |
|---|---|---|---|
| 3 | Field-instructions FORMAT-only | 50d KILE 34.05% vs guard 44.87% (-10.82pp) | ❌ **BURIED** — same failure mode as original; concise system prompt wins |
| 4 | V5b refiner guard mode | 50d +3.01pp vs V5b → **500d 43.64%** vs v2 44.61% (-0.97pp) | ❌ **BURIED** — 50d gain was sampling artifact; better than V5b but still below v2 |
| 6 | Span-aware contiguity in `_merge_bboxes` | Already implemented in `refiners.py` (`_to_rows`, `_largest_contiguous_run`) | ✅ **NOT NEEDED** — was already done |
| 7 | Conditional dedup ("ONLY ONE" softened) | Only 0.5% duplicates in v2 output (8/1493) | ❌ **BURIED** — not a real problem |
| 8 | bbox_verify on 500-doc | 500d KILE 42.75% vs v2 44.61% (-1.86pp) | ❌ **BURIED** — within-noise 50d gain didn't transfer |

### Structural finding from the audit

**The refiner is net negative at full scale across every configuration tested.** v2 (no refiner) > V5b (refiner) > V5b+guard > V5b+bbox_verify. Three different refiner strategies, all worse than raw Claude on 500-doc. The refiner's heuristics help on the 50-doc sample but hurt on the long-tail of docs we never sampled. **Production config locked: `BD_USE_REFINER=0` (= v2).**

### Per-field breakdown — selective-refiner consolation option

A 500-doc per-field analysis (artifact at `predictions/refiner_per_field_report.md` and `.json`) shows the net-negative result hides a real split: 17 of 36 KILE fields actually benefit from refining; 13 regress. Pattern:

- **Refiner HELPS:** delivery/other addresses, vendor names+IDs, vendor_address, tax_detail_rate (~6 clear fields). Interpretation: the refiner's row-isolation works on harder-to-find blocks where Claude over-includes neighboring text.
- **Refiner HURTS:** billing address, bank_num/account_num, dates, amount_total_tax/gross (~6 clear fields). Interpretation: refiner over-trims exact-value codes and standard billing blocks where Claude was already precise.
- **Asymmetry across address subtypes** (delivery wins, billing loses) is the most interesting structural finding.

**Selective refiner ceiling: +0.5-1pp.** Real but marginal — exactly the "no marginal gains" anti-pattern. **Not pursued today** to preserve API budget for path B inference. Per-field map preserved as a consolation move if path B fails to clear v2; can be implemented in `extract.py` as a per-fieldtype gate around the refiner call (~15 lines of code + 500-doc validation).

### The 50→500 trap recurrence

The v5b_50.json subset has a distribution that consistently overstates refiner benefit. **House rule: stop trusting any 50-doc result computed on the v5b_50 subset.** Future small-scale gates either pick fresh random 50, or skip 50-doc entirely and gate on 500.

### Items still untested

| # | Item | Why not tested today |
|---|---|---|
| 1 | PP-DocLayoutV3 semantic-region cropping (with Claude) | Needs GPU; GLM-OCR now buried, lower priority since path B running |
| 2 | V6 ReAct redesign with conservative gates | Highest implementation cost of the queued items; deferred until after path B settles |
| 5 | RapidTable LIR with newer 2026 TSR backbone | Needs research + integration; LIR is secondary metric; deferred |

### Updated active status

| Approach | Owner | Status |
|---|---|---|
| GLM-OCR with vLLM + layout stage | local GPU | ❌ **BURIED** — 4.23% KILE ceiling, see Appendix C |
| Qwen3-VL-8B LoRA fine-tune | RunPod | Pod live, smoke + training in progress |
| Reclassified items audit | revisit-runner | 5 of 8 buried with evidence; 3 deferred (need GPU or higher cost) |
| v2 (44.61% / 50.89%) | — | **Confirmed best after exhaustive Sonnet+heuristics tuning** |

### Honest accounting (revised)

Original claim: of ~15 buried approaches, 8 were "untested at proper tuning" and could yield +15-50pp. **Today's tests showed: of those 8, 5 are now properly buried with full-scale evidence, 1 was already implemented, and 3 remain untested.** The "premature burial" pattern was real (V5b deserved guard mode + 500-doc test, GLM-OCR deserved bug investigation), but the actual recoverable lift from refining the existing Sonnet stack appears to be **near zero**. The 26pp gap will only close with a model change (path B) or an architectural change (path C / V6 ReAct done right).

---

## Appendix C — GLM-OCR investigation closed (2026-04-19 PM)

Path A is buried with proper evidence after thorough investigation. Three runs showed the trajectory as bugs were fixed:

| Run | KILE | Recall | Precision | Notes |
|---|---|---|---|---|
| Initial 50-doc | 1.27% | 0.56% | 19.76% | Broken integration: PaddlePaddle missing, glmocr.parse silently returned `[]`, 36-field schema caused attention-to-middle skipping |
| After bug fixes (batched KIE 9-fields × 4) | 3.80% | 16.59% | 21.21% | Real model behavior; layout still missing |
| After full SDK + vLLM (21-doc partial) | **4.23%** | 16.78% | 20.96% | Layout regions detected, KIE still recall-limited |

**The lesson is NOT "drop-in 2026 SOTA failed."** Per user correction: any model not specialized for the EXACT task probably needs fine-tuning/adjustment. The only useful question is fit-vs-cost. GLM-OCR specifically failed because:

- OmniDocBench (where GLM-OCR is #1 at 94.62) measures text-recognition + reading-order + table-structure — "did you transcribe?"
- DocILE KILE measures schema-conformant field localization with PCC-IoU=1.0 bboxes — "which words belong to fieldtype X?"
- These are related but different tasks. SOTA on one ≠ SOTA on the other.
- GLM-OCR's training had 3 task prompts: "Text Recognition", "Table Recognition", "Formula Recognition." "Information Extraction" with custom JSON schema is an emergent capability bolted on, not a primary objective. That explains precision 21% (when it fires it's right) + recall 17% (refuses to fire on most fields — no schema-conformance training).

**Specialization-fit framework** (saved as `feedback_specialization_fit_framework.md`):

| Specialization to OUR task | Closing the gap | Examples |
|---|---|---|
| Already on-task | $0, drop in | None today (VDInstruct would qualify if released) |
| Adjacent task, transferable | Prompt + few-shot, ~free | Claude Sonnet (current v2 = 44.61%) |
| Capable architecture, wrong training | LoRA fine-tune, ~$5 | Qwen3-VL-8B (path B, in progress) |
| Wrong-task specialization, right modality | Substantial retraining | GLM-OCR (cost worse than alternatives) |
| Architectural mismatch | Cannot close at any cost | Donut, Code-Factory, pre-2025 models |

Modern/open/fits-VRAM are **necessary, not sufficient**. Future scout briefs must include the fit classification — not just exists/license/VRAM. GLM-OCR taught us this empirically; cheapest possible falsification of the "drop-in 2026 SOTA" hypothesis.

---

## Appendix D — Composition opportunities: what we have × GraphDoc / Rossum architectures

Brainstorm 2026-04-19 evening: where does our existing built infrastructure compose with the two reference architectures (GraphDoc SOTA on DocILE; Rossum's commercial backend) to potentially yield meaningful lift?

### Reference architectures

**GraphDoc (DocILE SOTA, 71.25% KILE / 75.93% LIR):**
OCR word grid → graph construction with spatial edges → GNN with per-word field-type classifier head → softmax over 36+19 field types per word. Trained on DocILE train. PCC-friendly by construction (predicts on existing OCR words, no generation, no bbox alignment problem). Strength: precise spatial grounding via the OCR grid. Weakness: rigid field set, no zero-shot generalization to unseen schemas.

**Rossum commercial backend (RASG paper, arXiv:2405.20245):**
Hybrid stack — OCR + VLM features + per-field classifier + checksum validators + per-customer schema adaptation + HITL escalation. Built around auditable confidence and graceful degradation. Multi-source per-field confidence drives auto-accept vs human-in-the-loop. Strength: production-grade, extensible, every prediction has a confidence trail. Weakness: needs per-customer training data; not a single end-to-end model.

### What of our work composes

| # | Combination | Maps to | Expected lift | Cost / blocker |
|---|---|---|---|---|
| 1 | **classifiers.py reranker over Claude** — 51 trained sklearn MLPs (F1>0.85, NEVER integrated) score each Claude candidate; reject below threshold | GraphDoc-light (per-word per-field classifier as veto layer) | **+3-5pp** by killing FPs that flooded V5b+refiner | Low — <1 day code, models on disk |
| 2 | **Qwen3-VL FT (path B) + Sonnet schema reasoning** — VLM proposes spatial regions per fieldtype, Sonnet decides accept/reject via mutex/conditional rules | Rossum-style hybrid (VLM features + LLM schema layer) | **+10-20pp** if path B works as projected | Conditional on path B success; medium integration |
| 3 | **Auditable confidence ensemble** — v2 + bbox_verify + classifiers.py + oracle_extract → multi-source per-field confidence; ≥3 agree → auto-accept; disagree → HITL flag | Rossum HITL + RASG confidence scoring | Doesn't move KILE AP directly but **is the production-readiness story** | Low — wires existing modules together; the build for shipping vs benchmark |
| 4 | **SAIL retrieval + cluster_infer for no-match docs** — cluster_infer (70% top-1) predicts cluster from image; SAIL retrieves entity-similar docs as fallback. Closes the documented -2.7pp no-match drag | Rossum-style adaptive few-shot per customer | **+2-3pp dataset-wide** | Free (CPU-feasible for SAIL retrieval); was item #2/#3 from Appendix B |
| 5 | **Span-correction model (the LayoutLMv3 idea redone with a 2026 model)** — small bidirectional model trained ONLY to predict exact span given v2's candidates + OCR grid. Much smaller than full KIE | GraphDoc-light (refines existing predictions to PCC-aligned spans) | **+5-10pp** by fixing multi-word PCC mismatch directly | Higher — requires a fresh small-model training; needs GPU but no contention with path B |
| 6 | **Conservative AOL (V6 ReAct redesigned)** — Sonnet extracts → calc tool verifies (gross = net + tax) → if math fails, conservative re-prompt only on affected fields | Rossum-style auditable verification + this document's section 4 done right | **+5-15pp** if gates properly tuned (keep-skewed not delete-skewed, see Phase 4 reclassification above) | Sonnet API only, no GPU; medium implementation |

### The actually high-EV stack (no GPU, no training)

**1 + 4 + 6 stacked.** Three independent attacks on three different failure modes:
- **#1 (classifier reranker)** → kills false-positive flooding (our biggest precision tax)
- **#4 (no-match coverage)** → fills the 25% of val docs with no cluster match
- **#6 (conservative AOL)** → catches semantic-coherence failures (amount sums, date ordering, mutex violations)

Each independently cheap; combined plausibly adds **+10-15pp** without touching the model architecture. This is the closest we can get to a Rossum-grade production stack with the pieces already on disk.

### The radical stack (with path B)

**Path B (Qwen3-VL FT) + #1 (classifier reranker) + #6 (AOL-light) + #3 (auditable confidence merge).**
Path B handles the model gap (better grounding than Sonnet). #1 reranks across multiple sources. #6 handles semantic verification. #3 wraps the whole thing in confidence trails. This composition is essentially a Rossum-architecture clone with our 2026 components — it's also where the document's original section 4 (AOL) finally fits cleanly.

### What does NOT compose

- **GraphDoc proper** — would require us to train a GNN from scratch on DocILE word features. High implementation cost, our budget doesn't justify, and #1 above gets most of the benefit cheaper.
- **Per-customer fine-tuning à la Rossum** — requires per-customer training data we don't have. Cluster-based few-shot (already in v2) is the cheap proxy.

---

## Appendix E — Complete options menu (everything we've enumerated)

Master list across all brainstorms 2026-04-19. Status as of evening: v2 = 44.61% KILE / 50.89% LIR baseline; path B (Qwen3-VL LoRA) training on RunPod; revisit-runner+glm-implementer running 250-doc gates on snap-margin and ensemble variants. **Active items in the live experiment streams are not duplicated here; this is the dormant menu.**

### Tier 1 — Composition opportunities (Appendix D, ranked by EV-per-effort, no-GPU)

| ID | Item | Lift est. | Cost | Status |
|---|---|---|---|---|
| D1 | classifiers.py reranker over Claude (51 trained MLPs, on disk) | +3-5pp | <1 day code | DORMANT, GraphDoc-light |
| D4 | SAIL retrieval + cluster_infer for no-match docs | +2-3pp | Free, CPU SAIL retrieval | DORMANT, closes -2.7pp drag |
| D6 | Conservative AOL (V6 ReAct redesigned, keep-skewed gates) | +5-15pp | Sonnet API only, medium impl | DORMANT, was Phase 4 of original idea |
| D2 | Qwen3-VL FT + Sonnet schema reasoning hybrid | +10-20pp | Conditional on path B | WAITING on path B result |
| D3 | Auditable confidence ensemble (v2 + bbox_verify + classifiers + oracle) | Production-readiness; small AP move | Wires existing modules | DORMANT, Rossum-style HITL stack |
| D5 | Span-correction model — modern small bidirectional, not LayoutLMv3 | +5-10pp | GPU + small training | DORMANT, needs free GPU |

### Tier 2 — No-GPU, no-pod menu (from earlier brainstorm)

| ID | Item | Lift est. | Cost | Status |
|---|---|---|---|---|
| M5 | oracle_extract pre-pass (NOT post-pass) | +0.5-2pp on financial codes | Sonnet API + 1 day code | Post-pass version was BURIED today; pre-pass version untested |
| M6 | Snap margin=3 (200 DPI) | -2.96pp KILE on 500d (was projected +0.83pp) | Free CPU + Sonnet re-extract | ❌ **BURIED 2026-04-19** — geometry verified bit-for-bit identical between extract and eval; tighter snap genuinely makes bbox–PCC alignment harder for Claude. Old projection falsified. |
| M7 | Different VLM in ensemble (Gemini-3-Flash-Preview) | -0.65pp 4-way vs 3-way (250d) | Anthropic API | ❌ **BURIED 2026-04-19** — Gemini at our cost tier (thinking_budget=0) is strictly weaker than Sonnet (precision 0.59 vs 0.62); thinking-enabled mode would cost ~$50+ on 250d, out of budget. Cross-model diversity hypothesis falsified at this cost tier. `gemini_extract.py` preserved for future stronger-Gemini retry. |
| M4 | Ensemble v2 with prompt+temperature variants (T=0, T=0.3, alt-prompt) | **+1.87pp KILE on 500d** | 3x Sonnet API | ✅ **APPLIED — NEW BASELINE** v2_ensemble_500.json = 46.48% KILE / 50.77% LIR |

### Tier 3 — GPU-bound (need local GPU, no training)

| ID | Item | Lift est. | Cost | Status |
|---|---|---|---|---|
| G1 | PP-DocLayoutV3 region-crop → re-prompt Claude on cropped region | +5-15pp | Free, GPU + Sonnet API | DORMANT — was Item #1 in Appendix B; the deterministic-region version of this document's Phase 2 |
| G2 | cluster_infer.py for no-match docs (Qwen3-VL embedding, 70% top-1 built) | +2-3pp | Free, GPU once + cached | DORMANT — composes with D4/SAIL |

### Tier 4 — Pod-bound (need RunPod budget, single-purpose)

| ID | Item | Lift est. | Cost | Status |
|---|---|---|---|---|
| P1 | Path B — Qwen3-VL-8B LoRA fine-tune on DocILE train | +15-25pp | $3.50-6.00, 4-8h | RUNNING NOW (pod live) |
| P2 | Path B variants — more epochs / different LoRA rank / different target modules | +1-5pp on top of P1 | $2-5 each | DORMANT, only worth after P1 lands |
| P3 | Train a 2026-era span-correction model on DocILE train | +5-10pp | $5-10 | DORMANT (= D5 above with budget) |
| P4 | LoRA fine-tune Qwen3-VL-2B (smaller, would fit 3070 inference) | TBD vs 8B | $2-3 | DORMANT — earlier 2B FT got 19.77% KILE, but architecture changes might help |

### Tier 5 — External / contingent on other parties

| ID | Item | Lift est. | Cost | Status |
|---|---|---|---|---|
| E1 | VDInstruct (literally trained on DocILE) — request weights from authors | very large | Email, time-only | WATCHING (`project_vdinstruct_lead.md`) |
| E2 | Watch for new doc-AI releases (Qwen4-VL? next-gen Gemini?) | unknown | Time-only | Passive monitoring |

### Tier 6 — BURIED with proper evidence (do not retry without new information)

| ID | Item | Final number | Why |
|---|---|---|---|
| B1 | GLM-OCR drop-in (Path A, full SDK + vLLM) | 4.23% KILE on 21d | Wrong-task specialization; OCR-trained, not KIE-tuned |
| B2 | V5b refiner (any tuning: stock, guard, bbox_verify) | 41.79% / 43.64% / 42.75% on 500d | Refiner net negative on long-tail distribution |
| B3 | Field-instructions FORMAT-only rewording | 34.05% on 50d | Same regression as original -8.9pp |
| B4 | Conditional ONLY ONE prompt rule | 0.5% duplicate rate | Not a real problem; evaluator handles it |
| B5 | Span-aware contiguity in `_merge_bboxes` | N/A | Already implemented in `refiners.py` |
| B6 | bbox_verify on 500d | 42.75% on 500d | Within-noise 50d gain didn't transfer |
| B7 | Oracle as POST-pass replacement | 44.61% (no change) | Architecturally wrong layer; pre-pass might still work |
| B8 | LayoutLMv3-base/large fine-tune | 0.05% inference | Outdated 2022 architecture, violates modern-only rule |
| B9 | Self-consistency 3-sample voting | -2.7pp | Recall drops by mathematical design |
| B10 | Donut / OCR-free models | 2.8% on 50d | PCC-IoU=1.0 misalignment, fundamental |
| B11 | Code-Factory Python scripts | 20.9% on 50d | No access to OCR grid, fundamental |
| B12 | RapidTable LIR (full-page and cropped) | -38pp / -21pp | TSR fails on invoice diversity |
| B13 | DSPy MIPROv2 with Haiku evaluator | -0.77pp / -3.26pp | Haiku evaluator hurts when applied to Sonnet |
| B14 | Targeted 2nd API call for financial codes | marginal | Field weights too small to move overall AP |
| B15 | Snap margin=3 at 200 DPI (full v2 re-extraction) | 41.65% KILE on 500d (-2.96pp) | Geometry verified identical between extract+eval; tighter snap genuinely worse — harder for Claude to land bbox–PCC alignment than wider margin=6 |
| B16 | classifiers.py reranker over v2_ensemble (D1) | 250d KILE 40.58-44.80% (-2.53 to -6.75pp) | Classifiers trained on "is this token a typical instance of field X" (random OCR vs gold spans). Claude's predictions aren't random — at 47% AP they're already filtered. The classifier prior just disturbs the rank ordering. Even threshold=0.0 (no drops) loses -2.53pp from rank disturbance alone. **Side finding: LIR F1 +1.6pp at threshold=0.3 — classifiers may help LIR-only reranking; revisit if LIR becomes a focus.** |
| B17 | Conservative AOL — calc verifier with score demotion (D6) | 250d KILE 43.13-45.88% (-1.45 to -4.20pp at any demote factor or math tolerance) | Tested ×0.5 demote, ×0.95 demote, 3% math tol, 10% math tol — all lose. Tolerance change had ZERO effect (same pages flagged regardless of strictness; verifier fires on parse failures, not borderline math). Even ×0.95 demote costs -1.45pp because the flagged fields ARE valid TPs. **Architectural lesson: score-modifying verifiers on a high-AP baseline are pure one-sided risk.** Any future AOL must be RECALL-AUGMENTATION ONLY (add missed fields, never touch existing prediction scores). This reframes V6 ReAct's failure: not just "over-eager gates" but the entire "verify then modify" concept is wrong against a high-AP baseline. |
| B18 | Gemini-3-Flash-Preview added to v2 ensemble (M7) | 4-way -0.65pp vs 3-way (250d) | Gemini single-run 41.56% KILE (vs Sonnet's 44.95%); precision 0.59 vs Sonnet 0.62. Thinking-enabled mode would cost $50+ on 250d (out of budget); thinking-disabled mode is strictly weaker than Sonnet. **Cross-model diversity hypothesis falsified at this cost tier.** Code preserved for retry if a stronger affordable Gemini variant ships. |
| B19 | Oracle prepass — both noisy and strict variants (M5) | Single -3.97 to -6.28pp; 4-way ensemble +0.29-0.48pp (within noise) | Tested score=0.7+1.0 (noisy) and score=1.0-only (strict IBAN+BIC+VAT). Both single-run variants regress materially. **Architectural lesson: pre-pass hint injection narrows Claude's attention to the hinted fields while DEGRADING coverage on the other ~26 KILE field types.** 4-way ensemble partially recovers only because un-hinted variants outvote the hinted one. Both pre-pass and post-pass oracle strategies now exhausted. |
| B20 | Recall-augmentation AOL (ADD-only, cluster-prior re-prompting) | +0.02pp KILE / +0.00pp LIR on 250d (8 fields added across 187 re-prompts; 4% hit rate) | Design worked correctly (LIR flat = no FPs, confirming ADD-only premise is safe). But cluster prior fired on 280 missing-field slots; Claude correctly resisted 96% of them and returned empty — the missing fields are GENUINE ABSENCES, not extraction failures. **Most important structural finding of the day: prompt-based extraction has hit its recall ceiling around v2_ensemble's 47% KILE on this benchmark.** The remaining ~24.77pp gap to GraphDoc cannot be closed by better prompting / re-prompting / retrieval — the misses don't exist to recover. Requires model change (path B Qwen3-VL FT) or fundamentally different mechanism. |

### Tier 7 — Awaiting reconsideration (reclassified-but-not-tested-yet)

| ID | Item | Why deferred |
|---|---|---|
| R1 | RapidTable LIR with newer 2026 TSR backbone | Research + integration cost; LIR is secondary metric |
| R2 | DSPy MIPROv2 with Sonnet evaluator | $30/run; only worth if v2 is the long-term frozen baseline |

### Decision summary

- **Immediate live work (next hours):** P1 (path B), M4+M6 (ensemble + snap, 250-doc gates running)
- **High-EV dormant, no-GPU, ready to dispatch:** D1 (classifier reranker) + D4 (no-match coverage) + D6 (conservative AOL)
- **Highest-EV dormant overall:** G1 (region-crop with GPU)
- **Highest-EV speculative:** P1 (path B) ± E1 (VDInstruct release)
- **Production-readiness path:** D3 (auditable confidence ensemble)
