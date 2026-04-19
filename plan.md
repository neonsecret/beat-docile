# Beating Rossum on DocILE: Research & Strategy

---

## 1. The Benchmark: DocILE

Rossum published **DocILE** (Document Information Localization and Extraction) — currently the world's largest research dataset for business document understanding.

**Link:** https://docile.rossum.ai/ | GitHub: https://github.com/rossumai/docile
**Live leaderboard:** https://rrc.cvc.uab.es/?ch=26

### What it tests

Two tasks:

| Task | What it measures |
|---|---|
| **KILE** (Key Information Localization & Extraction) | Detect fields by category (invoice_number, total_amount, vendor_name, date, etc.), localize them on the page (bounding box), and extract text |
| **LIR** (Line Item Recognition) | Same as KILE, but additionally group related fields into line item tuples — e.g., (description, quantity, unit_price, amount) per table row |

### Dataset scale

- **6,680 fully annotated** business documents (invoices, POs)
- **100,000 synthetically generated** documents for pre-training
- **~1M unlabeled documents** with pre-computed OCR (for self-supervised pretraining)
- OCR pre-computed via DocTR library with word-box snapping
- Documents come from UCSF Industry Documents Library and FCC Public Inspection Files

### Evaluation metric

**Average Precision (AP)** is the primary metric (COCO-style, non-increasing precision-recall curve).
A prediction matches ground truth if it has the **same field type AND spatial overlap** (via OCR word center containment, not raw IoU — so bounding box precision matters a lot).

Secondary metrics: F1, Precision, Recall.

---

## 2. Known Scores & Leaderboard State

| Method | KILE AP | LIR AP | Notes |
|---|---|---|---|
| **Regex only** | ~20.3% | — | Deterministic, no ML |
| **RoBERTa BASE + Synth** | ~mid-range | — | Best score among rules-compliant baselines |
| **LayoutLMv3 BASE** | Best baseline overall | — | Pre-trained on IIT-CDIP (prohibited in official track) |
| **GraphDoc** (USTC/iFLYTEK) | **Winner ICDAR 2023** | **Winner ICDAR 2023** | Transformer + word-combination learning + heuristics |
| **Code Factory** (2025 paper) | **80.0%** | **80.4%** | LLM calls compiled as code artifacts; 2.3× faster than direct LLM |
| **Direct LLM** | ~80.0% | ~78% (est.) | GPT-class model, slower, higher cost |

**Key insight:** The Code Factory paper (2025) bridges from ~20% regex to ~80% KILE by framing extraction as compilable code, not raw prompting. This is the current publicly known SOTA for the DocILE KILE metric. The leaderboard accepts ongoing submissions — it is theoretically beatable.

---

## 3. Prominent Approaches & Why They Work (or Don't)

### 3.1 Template/Rule-Based (Regex)
- **Score:** ~20.3% KILE
- Extract fields using regex patterns (dates, amounts, invoice numbers)
- **Fails** on format variation: `01/15/2024` vs `15 Jan 2024` vs `2024-01-15` are all the same date
- Completely fails on vendor names, line items, ambiguous fields

### 3.2 OCR + Sequence Labeling (LayoutLM family)
- Feed document as (text token, bounding box) pairs to a transformer
- Token classification: each token labeled as field type or O (outside)
- **LayoutLM (v1):** Text + 2D position embeddings, fine-tuned on downstream tasks
- **LayoutLMv2:** Adds visual features (CNN on document image patches)
- **LayoutLMv3 (2022):** Unified tri-modal — text tokens + ViT image patches + 2D positional embeddings. Pre-training: MLM + Masked Image Modeling + Word-Patch Alignment. FUNSD F1=90.81, CORD F1=98.48
- **Why strong:** Layout IS semantics in business documents. "Top-right, bold, `$12,345`" = total amount even without reading surrounding labels
- **Weakness:** Still OCR-dependent (errors propagate), struggles with novel document layouts not seen in pre-training

### 3.3 GraphDoc (ICDAR 2023 Winner)
- Graph-based transformer that models spatial relationships between text nodes
- Key innovation: learns **which adjacent words to merge** to form complete field values (e.g., "Acme" + "Corporation" + "Ltd" → one vendor_name entity)
- Uses heuristics derived from dataset statistics
- **Why it won:** Explicitly models the merging problem that sequence labelers treat implicitly

### 3.4 LLM-Centric / Code Factory (Current SOTA ~80%)
- Pass OCR text + document structure to an LLM, ask it to extract fields in JSON
- **Code Factory variant:** Instead of prompting the LLM to directly extract each time, prompt it to *write extraction code* (Python) for the document type, cache the code, run it deterministically
- Result: first-run cost of LLM inference, subsequent runs at regex speed — 2.3× lower latency at equivalent accuracy
- **Why it works:** LLMs generalize across format variations without explicit template engineering

### 3.5 Vision-Language Models (VLMs) — Emerging SOTA
- Don't use OCR at all. Feed the raw document image directly to a VLM
- **Qwen3-VL-235B:** 256K context, 32-language OCR, rivals GPT-5 on DocVQA (96.4 with Qwen2.5-VL-72B, Qwen3-VL expected higher)
- **GPT-5 / GPT-5.2:** 400K context, 6.2% hallucination rate (40% reduction vs prior)
- **Advantage:** No OCR error propagation, sees layout visually the way a human does
- **Weakness:** Slower and more expensive per document; still struggles with very long documents (many pages) and precise bounding box localization

---

## 4. The Gap & Where to Attack

The **80% ceiling** from Code Factory is the current known public ceiling for DocILE KILE. Here's why it's not yet 95%+:

1. **OCR errors propagate** — DocILE documents have "degraded OCR quality" (explicitly noted in the paper). A field value mangled by OCR can't be reconstructed by any downstream model without seeing the image
2. **Semantic ambiguity** — Is "Net 30" an `invoice_number`, a `payment_term` (→ `none`), or a line item qualifier? Context is required
3. **Bounding box precision** — AP requires spatial overlap. Getting the right text with a wrong bounding box = zero score
4. **LIR grouping is hard** — Associating extracted fields into line item tuples requires understanding table structure, which pure NLP approaches handle poorly

---

## 5. Our Strategy to Beat It

### Core Hypothesis
The Code Factory at 80% uses OCR text + LLM. We can beat it by combining:
1. **Visual grounding** (VLM sees the raw document image for OCR-robust extraction)
2. **Structured reasoning** (agent decomposes the problem, self-verifies)
3. **Localization accuracy** (dedicated bounding-box refinement step)

### Architecture: LLMCompiler Agent + VLM

#### Why LLMCompiler over ReAct

**ReAct** (Reason + Act): sequential — think, act, observe, think, act... Each step waits for the previous. Good for single-document deep reasoning. Poor for throughput.

**LLMCompiler** (Plan → DAG → Parallel Execute): the planner emits a directed acyclic graph of tool calls. A Task Fetching Unit executes independent tasks concurrently. For a multi-page invoice:
- Page 1 header extraction and Page 3 line items table extraction can run **in parallel**
- Then a merge/verification step runs after both complete

For DocILE's scale (6,680+ documents) LLMCompiler's parallelism is critical. For a single document's internal reasoning, we embed a ReAct-style self-correction loop **within** each LLMCompiler task node.

#### The Pipeline

```
Document (PDF/Image)
        │
        ├──[Task A] VLM Field Detector (Qwen3-VL)
        │     → sees raw image, outputs: field_type, rough_bbox, text_value
        │     runs per page in parallel
        │
        ├──[Task B] OCR Structural Parser (DocTR / PaddleOCR)
        │     → word-level bboxes + reading order
        │     runs in parallel with Task A
        │
        └──[Task C, depends on A+B] Fusion & Localization Agent
              → aligns VLM field predictions onto OCR word grid
              → resolves conflicts between VLM text and OCR text
              → snaps predicted bboxes to exact OCR word boundaries
              → outputs: final (field_type, page, bbox, text) tuples
                         and line item groupings for LIR

              [Internal ReAct loop within this agent]:
              1. Draft extraction
              2. Self-verify: "Does total_amount == sum of line item amounts?"
              3. Cross-check: "Is invoice_date before due_date?"
              4. If conflict → re-query VLM on the specific region
              5. Output final JSON
```

#### Why this beats Code Factory

| Dimension | Code Factory (80%) | Our Approach |
|---|---|---|
| OCR errors | Accepts OCR text as-is | VLM reads raw image → independent OCR-free signal |
| Localization | Inherits OCR bbox | Fusion step snaps to OCR grid with VLM guidance |
| Self-verification | None | Internal ReAct loop catches semantic inconsistencies |
| Throughput | 2.3× faster than direct LLM | LLMCompiler parallelism across pages |
| LIR (grouping) | Weak | VLM sees full table visually → natural row grouping |

---

## 6. What Rossum Actually Built — and Why It Matters

This is the most important section for understanding the competitive landscape.

### Aurora's Architecture: Discriminative, Not Generative

Rossum explicitly calls Aurora a **Transactional LLM (T-LLM)** and states a critical design decision:

> *"Rossum employs a discriminative decoder that operates within the confines of the input document."*

This is architecturally different from GPT-5, Qwen3-VL, or any generative model. Here's what it means:

| Property | Generative (GPT/Qwen) | Discriminative (Aurora) |
|---|---|---|
| Output space | Any token in vocabulary | Only tokens/spans present in the document |
| Hallucination risk | High — can fabricate values | Zero — constrained to document content |
| Format normalization | Yes — can output "2024-01-15" from "Jan 15, 2024" | No — outputs the literal span as it appears |
| Training data needed | Huge (pretraining + fine-tune) | Can reach SOTA with 10× fewer examples (Rossum claim) |
| Localization | Approximate bbox | Exact span from OCR word grid |

**Why discriminative wins for document extraction:** If you're extracting `total_amount` from an invoice, the value is literally on the page. You don't need a model that can *generate* "$1,234.56" — you need one that can *point to* where it is. A discriminative model makes this guarantee formally: it selects a span from the input rather than sampling from a distribution. This eliminates an entire class of failure modes.

Aurora was trained on **11 million transactional documents** and achieves 92.5% average accuracy across customers, processing **$700B/year** in transactions. This is their moat — the data, not the architecture.

### The "8 A100s" Context — What's Trainable

**Donut** (OCR-free Document Understanding Transformer, ECCV 2022) is the closest publicly documented architecture to what Rossum likely uses as an architectural template:
- **SwinTransformer encoder** (visual) → reads document image patches
- **BART decoder** (text) → generates structured output autoregressively
- Donut-proto was trained on **8 V100 GPUs** (~5 days) — equivalent to 8 A100s is faster
- Donut-base (the bigger model) needed **64 A100s** for 2.5 days

**The key finding:** An encoder-only Donut (Swin encoder + classification head, no BART decoder) achieves **~equal accuracy** to the full encoder-decoder model but runs **10× faster** at inference, because it avoids autoregressive decoding. This is exactly the discriminative architecture Rossum describes — point to a span, don't generate one.

On **8 A100s** (80GB each = 640GB total VRAM), you can realistically:

| What | Time estimate |
|---|---|
| Fine-tune LayoutLMv3-large on DocILE (6,680 docs) | ~4–8 hours |
| Train Donut-scale model from scratch on DocILE synth (100K docs) | ~3–5 days |
| Pre-train custom discriminative Swin+head on DocILE unlabeled (1M docs) | ~1–2 weeks |
| Fine-tune Qwen3-VL-7B (LoRA) for extraction | ~12–24 hours |

### Our Custom Architecture: Discriminative-Donut for DocILE

Instead of using a generative VLM (which hallucinates) or an OCR-dependent model (which inherits OCR errors), we train a custom architecture inspired by Rossum's own design choices:

```
Document Image (full page, 2048×1536)
        │
   Swin Transformer Encoder
   (hierarchical, 4 stages: {2,2,18,2} layers)
        │
   Visual Feature Map (H/32 × W/32 × 1024)
        │
   ┌────────────────────────────────────────────┐
   │  Dual-head discriminative decoder          │
   │                                            │
   │  Head 1: Field Type Classifier             │
   │    → For each image region: which field    │
   │      type is it? (or None)                 │
   │    → Token classification over visual      │
   │      feature grid (no OCR needed)          │
   │                                            │
   │  Head 2: Span Boundary Predictor           │
   │    → Given a detected field region,        │
   │      predict start/end of text span        │
   │    → Used for precise KILE localization    │
   │      (maps to OCR word grid at eval time)  │
   │                                            │
   │  Head 3: Line Item Grouping (for LIR)      │
   │    → Predicts which field detections       │
   │      belong to the same table row          │
   │    → Uses relative position + field type   │
   └────────────────────────────────────────────┘
        │
   Output: (field_type, bbox, text_value) × N
           + line item groupings
```

**Training procedure on 8 A100s:**

1. **Stage 1 — Visual pretraining** (3 days, 8 A100s): Train Swin encoder on DocILE's 100K synthetic documents + 1M unlabeled documents. Objective: masked image region prediction + text reading (like Donut pretraining but on domain-specific business documents).

2. **Stage 2 — Task fine-tuning** (1 day, 8 A100s): Fine-tune all three heads jointly on DocILE's 6,680 annotated training documents. Mixed loss: classification cross-entropy (Head 1) + span boundary BCE (Head 2) + grouping contrastive loss (Head 3).

3. **Stage 3 — OCR alignment** (2 hours): Post-hoc snap predicted bboxes to DocTR OCR word boundaries. This step is purely deterministic and doesn't require training — it just aligns our visual predictions to the word grid that DocILE's AP metric uses for matching.

**Why this beats the current 80% ceiling:**
- No OCR errors propagating (unlike LayoutLMv3, Code Factory)
- Discriminative = zero hallucination (unlike GPT/VLM approaches)
- Trained on DocILE's own synthetic data (in-distribution)
- LIR grouping is a dedicated head, not a post-processing afterthought

---

## 7. Model Choices (2025/2026)

### Primary extraction model
**Qwen3-VL-235B-A22B** (open source, Apache 2.0 via Alibaba)
- Rivals GPT-5 on multimodal benchmarks
- 256K token context (handles multi-page documents natively)
- Native 32-language OCR in-model
- Can be self-hosted — avoids per-token API cost at scale
- Sparse MoE architecture: 235B total params, 22B active per token → cost-efficient inference

**Alternative:** GPT-5 via API — highest general reasoning, lowest hallucination rate (6.2%), 400K context, but proprietary and expensive at DocILE scale

### Bounding box localization
**LayoutLMv3-large** fine-tuned on DocILE training set
- Takes (VLM-predicted field type, OCR word grid) → outputs precise word-level span
- Strong on FUNSD/CORD, proven on similar token classification tasks
- Serves as a "snap to grid" refinement on top of VLM's coarse localization

### OCR engine
**DocTR** (already used by DocILE's official pipeline) + **PaddleOCR** as ensemble for degraded documents
- Using the same OCR as the benchmark's ground truth generation avoids systematic mismatches in word boundary definitions

### Agent orchestration
**LangGraph** (LangChain's graph-based agent framework) implementing the LLMCompiler pattern
- Nodes: VLM extractor, OCR parser, Fusion agent (with internal ReAct), JSON validator
- Edges: dependency graph with parallel execution for independent page tasks

---

## 7. Key Technical Bets

### Bet 1: VLM-OCR fusion beats either alone
The failure mode of pure OCR approaches is degraded scan quality. The failure mode of pure VLM approaches is imprecise bounding boxes. Fusing both signals addresses both weaknesses simultaneously.

### Bet 2: Self-verification catches systematic errors
Invoices have internal consistency constraints:
- `total_amount` ≈ `subtotal` + `tax`
- `invoice_date` < `due_date`
- Line item `amount` ≈ `quantity` × `unit_price`

A ReAct loop that checks these constraints and re-queries on failures can catch a class of errors no single-pass model can fix.

### Bet 3: Few-shot document-type conditioning
Many DocILE documents come from the same vendor template. If we cluster documents by layout similarity (using LayoutLMv3 embeddings) and provide few-shot examples of the same template to the VLM, extraction accuracy on that template cluster should jump significantly. This is essentially what Code Factory does with code — we generalize it to visual templates.

### Bet 4: LIR via table structure understanding
Qwen3-VL natively handles table parsing. Instead of trying to group line item fields post-hoc (hard), instruct the VLM to output the full line item table as a structured object directly from the image — then map the cells back to OCR words for localization.

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Qwen3-VL hallucination on ambiguous fields | Self-verification loop + confidence threshold → flag for human review |
| VLM bounding boxes imprecise → low AP | LayoutLMv3 snap-to-grid refinement |
| Cost of running 235B model at scale | Qwen3-VL MoE: only 22B params active per token; batch inference on A100/H100 |
| DocILE OCR errors in ground truth annotations | Same DocTR OCR pipeline as benchmark → systematic errors cancel |
| LIR table row grouping ambiguity | Visual table parsing (VLM) + position-based clustering fallback |

---

## 9. Comparison to What Rossum Actually Does

Rossum's **Aurora Engine** uses:
- Proprietary transactional LLM trained on millions of business documents
- Template-free extraction (no manual field mapping)
- Continuous learning from human corrections (active learning loop)
- 276 language support

Our approach differs in that we're:
1. Building on top of open-weight models (Qwen3-VL) — reproducible, no black box
2. Using an explicit agentic verification loop vs. single-pass extraction
3. Targeting DocILE specifically — their own benchmark, their own data distribution

The benchmark rules prohibit using IIT-CDIP pre-trained checkpoints in the official track. Qwen3-VL was pre-trained on general web/document data not IIT-CDIP — likely compliant (verify before submission).

---

## 10. Proposed Experiment Plan

### Phase 1: Baseline reproduction
- Download DocILE dataset
- Run LayoutLMv3-base fine-tuned on DocILE training set
- Establish our internal KILE/LIR AP baseline on validation set
- Target: reproduce ~mid-range baseline scores

### Phase 2: VLM zero-shot extraction
- Run Qwen3-VL-7B (small, fast) on DocILE validation documents zero-shot
- Prompt: "Extract all key fields from this invoice. Return JSON with field_type, bounding_box, text."
- Measure KILE AP — establish VLM zero-shot ceiling without fine-tuning

### Phase 3: Fusion pipeline
- Align VLM predictions to DocTR OCR word grid
- Implement bbox snap: for each VLM-predicted bbox, find the minimal spanning set of OCR words whose centers fall inside it
- Re-measure KILE AP — expect improvement from more precise localization

### Phase 4: Self-verification loop
- Add internal ReAct agent with arithmetic consistency checks
- Re-run and measure delta

### Phase 5: Few-shot template clustering
- Cluster documents by LayoutLMv3 embeddings
- For each cluster, retrieve 3 nearest neighbors with ground-truth annotations as few-shot examples for VLM
- Re-run and measure delta

### Phase 6: LIR
- Instruct VLM to output table structure as JSON
- Map VLM table cells back to OCR spans for localization
- Measure LIR AP

### Phase 7: Scale up
- Move from Qwen3-VL-7B to Qwen3-VL-72B or -235B
- Measure quality/cost tradeoff

---

## Sources

- [DocILE Benchmark — Rossum](https://docile.rossum.ai/)
- [DocILE GitHub](https://github.com/rossumai/docile)
- [DocILE Paper (arXiv)](https://arxiv.org/abs/2302.05658)
- [DocILE ICDAR 2023 Overview (CEUR-WS)](https://ceur-ws.org/Vol-3497/paper-049.pdf)
- [ICDAR 2023 Winners Announcement](https://www.prnewswire.com/news-releases/shaping-the-future-of-document-processing-rossum-presents-the-winners-of-the-docile-competition-301936238.html)
- [Qwen3-VL GitHub](https://github.com/QwenLM/Qwen3-VL)
- [Qwen2.5-VL Technical Report](https://arxiv.org/pdf/2502.13923)
- [LayoutLMv3 — EmergentMind](https://www.emergentmind.com/topics/layoutlmv3)
- [Code Factory / LLM-Centric Pipeline Paper (HAL)](https://hal.science/hal-04772570v1/document)
- [RolmOCR — MarkTechPost](https://www.marktechpost.com/2025/04/05/reducto-ai-released-rolmocr-a-sota-ocr-model-built-on-qwen-2-5-vl-fully-open-source-and-apache-2-0-licensed-for-advanced-document-understanding/)
- [LLMCompiler vs ReAct — DEV Community](https://dev.to/jamesli/react-vs-plan-and-execute-a-practical-comparison-of-llm-agent-patterns-4gh9)
- [OmniDocBench (CVPR 2025)](https://github.com/opendatalab/OmniDocBench)
- [Awesome Document Understanding](https://github.com/tstanislawek/awesome-document-understanding)
- [Rossum Aurora — Official Page](https://rossum.ai/aurora-advanced-ai/)
- [Rossum Launches Its Own LLM — Deep Analysis](https://www.deep-analysis.net/rossum-launches-its-own-llm/)
- [Aurora 1.5 Press Release](https://www.prnewswire.com/news-releases/rossum-unveils-aurora-1-5-ai-engine--copilot-to-accelerate-document-processing-for-global-enterprises-in-276-languages-302288467.html)
- [Donut: OCR-Free Document Understanding Transformer (ECCV 2022)](https://arxiv.org/abs/2111.15664)
- [Donut GitHub (training hardware details)](https://github.com/clovaai/donut)
- [Encoder-only Donut — 10× inference speedup](https://www.qantev.com/post/fine-tuning-donut-transformer-for-document-classification)
- [Discriminative vs Generative for Document Extraction (ScienceDirect 2025)](https://www.sciencedirect.com/science/article/pii/S1467089525000260)
- [Qwen3-VL GitHub](https://github.com/QwenLM/Qwen3-VL)

