# Beat DocILE — Plan V2: The Precision Stack

**Goal:** beat GraphDoc's 71.25% KILE AP on DocILE.

**Current state:**
- Track A (Claude + few-shot) plateaued at 44.6% KILE AP
- Track B (LayoutLMv3-base fine-tune on 3070) running, ~12h remaining
- $13 RunPod budget (5090 at $0.69/hr ≈ 18h)
- Agentic systems are our leverage — not custom models

**Theory of the 26pp gap (44.6% → 71.25%):**
1. **Bbox precision** (~8pp): Claude's word_ids are semantically right but spatially imprecise. PCC-IoU=1.0 punishes any overrun.
2. **Rare fields** (~5pp): IBAN, BIC, reg_id, tax_id — generic prompting doesn't activate them.
3. **Semantic confusion** (~3pp): billing vs delivery addresses; net vs gross amounts.
4. **Model capacity** (~5pp): LayoutLMv3-base vs -large; no synthetic pretraining.
5. **Approach diversity** (~5pp): single generator vs ensemble of specialized heads.

The V2 plan addresses each gap.

---

## Architecture: 4-layer stack

```
Document → OCR (DocILE's DocTR)
         ↓
[L1] Claude extraction (v2 baseline, already working)
         ↓
[L2] Span refinement — deterministic spatial rules per field type
         ↓
[L3] Regex/classifier validators — reject or correct impossible values
         ↓
[L4] Ensemble with LayoutLMv3-large (RunPod trained) + LayoutLMv3-base (3070 trained)
         ↓
Ranked predictions (scored by layer agreement)
```

Each layer is additive — turn any off and the rest keeps working.

---

## Phase breakdown & task ownership

### Phase 1 — Claude-side bug fixes (file owner: `extract.py`)
Agent: **fixer**
- Fix `_merge_bboxes` contiguity: split non-contiguous word_ids, keep dominant visual block, emit bbox from that alone. Expected: +2-3pp on address fields.
- Prompt update: add "billing and delivery addresses must have DISJOINT word sets" rule.
- Prompt update: add "net and gross amounts are different values — don't output both unless both are on the document."
- Acceptance: re-run on 50 val docs, KILE AP ≥ 46%.

### Phase 2 — Span refiners (new file: `src/beat_docile/refiners.py`)
Agent: **refiner**
- For each field type, a `refine(word_ids, words, field_type) -> word_ids` function.
- Deterministic rules per type:
  - **addresses**: sort by (row, col), detect row-cluster gaps, keep largest cohesive block.
  - **amounts**: strip surrounding label words; normalize decimal/thousand separators.
  - **dates**: validate format, strip labels.
  - **names**: single row, capitalized sequence, stop at row break.
  - **IDs** (document_id, order_id, invoice_num): single token, alphanumeric.
- Apply refiners to Claude output in `_parse_response`.
- Acceptance: re-run on 50 val docs, KILE AP ≥ 48%.

### Phase 3 — Regex validators (new file: `src/beat_docile/validators.py`)
Agent: **validator**
- Regex library per field type:
  - IBAN: `^[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}$`
  - BIC: `^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$`
  - VAT ID (per country): EU patterns (DE+9d, GB+9d, FR+11chars...)
  - Amount: `^[\d,.]+$` with decimal/thousand logic
  - Date: multiple format support (ISO, US, EU)
- `validate(field) -> confidence_multiplier`: 1.0 if matches, 0.0 if clearly wrong, 0.5 if ambiguous.
- Apply as score multiplier in final output — wrong-format predictions get demoted, not dropped.
- Acceptance: re-run, precision up, AP ≥ 49%.

### Phase 4 — LayoutLMv3-large on RunPod 5090 (new dir: `runpod/`)
Agent: **trainer** (requires RunPod pod setup by user)
- Setup script: install deps on fresh pod, download DocILE dataset.
- Training script: adapt our existing `train_layoutlmv3.sh` for:
  - Model: `microsoft/layoutlmv3-large` (358M params, fits on 5090)
  - Batch size: 8 (vs 2 on 3070) — 5090 has 32GB
  - Epochs: 20 (converges faster with larger model + bigger batch)
  - Estimated time: ~8-10h
- Inference script: run on val + test, save predictions JSON.
- Budget: ~$7 of the $13.
- Acceptance: LayoutLMv3-large KILE AP ≥ 55% standalone.

### Phase 5 — Ensemble (new file: `src/beat_docile/ensemble.py`)
Agent: **ensembler** (runs after Phase 1-4 produce predictions)
- Per-field ensemble strategy:
  - Load predictions from: refined-Claude, LayoutLMv3-base, LayoutLMv3-large.
  - For each (docid, page, fieldtype, bbox-region):
    - If all agree (high IoU): take highest-score prediction, boost score.
    - If 2/3 agree: take majority-score.
    - If disagree: keep all separately with their scores (let the evaluator decide via greedy matching).
  - Precision-optimized variant: only emit predictions with ≥2 source agreement.
  - Recall-optimized variant: emit union.
- Try both variants on val, pick winner.
- Acceptance: Ensemble KILE AP ≥ 65%.

### Phase 6 — Optional: Word-merge classifier (if budget allows)
Agent: **graph-doc** (only runs if Phase 4 saves budget)
- Inspired by GraphDoc's word-combination learning.
- Input: pairs of adjacent OCR words + their features.
- Output: "merge" or "don't merge" for this pair.
- Apply to refine Claude's field boundaries.
- Training: ~2h on 5090 (BiLSTM + features).
- Expected gain: +3-5pp on complex fields.

---

## Coordination rules

**File ownership (no agent writes to another's files):**
| File | Owner |
|---|---|
| `src/beat_docile/extract.py` | fixer |
| `src/beat_docile/refiners.py` | refiner (new) |
| `src/beat_docile/validators.py` | validator (new) |
| `src/beat_docile/ensemble.py` | ensembler (new) |
| `runpod/` (all files) | trainer |
| `src/beat_docile/merge_classifier.py` | graph-doc (new, optional) |
| `src/beat_docile/cli.py` | fixer (owns, others request additions via messages) |

**Integration points (need coordination):**
- `_parse_response` in extract.py calls refiners.py functions — fixer must add imports after refiner is ready.
- CLI commands for new eval flags — fixer adds them.

**Evaluation discipline:**
- After each phase, the phase owner runs `bd extract --limit 50 --split val` and reports KILE AP.
- Full 500-doc eval only after all phases merge.

---

## Budget allocation

| Item | Cost | Time |
|---|---|---|
| RunPod 5090 LayoutLMv3-large training | $7 (~10h) | Phase 4 |
| RunPod 5090 word-merge classifier | $1.40 (~2h) | Phase 6 |
| RunPod buffer (re-runs, inference) | $3 (~4h) | Phase 4-6 |
| **Total RunPod** | **~$11.40** | ~16h |
| Claude API | not budgeted (free to iterate) | Phases 1-3, 5 |
| Local GPU 3070 (sunk) | $0 | Phase 4 backup (base model) |

---

## Success criteria

- **Minimum**: Beat our own v2 baseline (44.6% → 55%+ KILE AP).
- **Target**: Beat GraphDoc (71.25%+ KILE AP).
- **Stretch**: Top 1 on DocILE leaderboard.

For LIR F1: similar uplift expected from ensemble. Target: 60%+ (current best 50.9%).

---

## What's NOT in this plan

- No Qwen3-VL or other non-compliant VLMs — compliance ambiguity.
- No from-scratch pretraining — 18h budget is too small.
- No GraphDoc direct reimplementation — the actual paper's architecture is non-trivial. The word-merge classifier (Phase 6) is an approximation.
- No data augmentation on train — synthetic pretraining would be ideal but doesn't fit budget.
