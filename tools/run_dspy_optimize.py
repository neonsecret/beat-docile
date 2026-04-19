#!/usr/bin/env python3
"""Run DSPy MIPROv2 optimization on DocILE V5b extraction instruction.

Uses 300 labeled val docs as trainset (first 300), 100 as eval (next 100).
MIPROv2 proposes and tests instruction variants using Bayesian optimization.
Saves compiled program to models/dspy/compiled_miprov2.json.
Saves optimized instruction text to src/beat_docile/optimized_prompt.py.

Cost strategy (default):
  task_model=haiku  — 1,250 evaluation calls (~$2 total vs ~$30 all-Sonnet)
  prompt_model=sonnet — instruction proposal generation (~20 calls, worth quality)
  Instructions found on Haiku transfer well to Sonnet — same family, same format.

Usage:
    DATA_ROOT=data uv run python tools/run_dspy_optimize.py
    DATA_ROOT=data uv run python tools/run_dspy_optimize.py --num-trials 75 --auto heavy
    DATA_ROOT=data uv run python tools/run_dspy_optimize.py --task-model claude-sonnet-4-6

Optimization strategy: instruction-only (max_bootstrapped_demos=0).
Reason: page images are ~200KB each — serializing them as bootstrapped demos would
add ~1MB per demo to every future prompt. Pure instruction search avoids this.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
os.environ.setdefault("DATA_ROOT", str(DATA_DIR))

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from beat_docile.config import DATA_ROOT, VERTEX_PROJECT_ID, VERTEX_LOCATION, DEFAULT_MODEL
from beat_docile.data import load_split
from beat_docile.dspy_optimizer import (
    DocILEExtractionModule,
    build_dspy_examples,
    kile_metric,
)
import dspy
import litellm

litellm.drop_params = True


def _make_lm(model: str, max_tokens: int = 8192) -> dspy.LM:
    os.environ.setdefault("VERTEXAI_PROJECT", VERTEX_PROJECT_ID)
    os.environ.setdefault("VERTEXAI_LOCATION", VERTEX_LOCATION)
    return dspy.LM(f"vertex_ai/{model}", max_tokens=max_tokens, temperature=1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="DSPy MIPROv2 instruction optimization")
    parser.add_argument("--train-n", type=int, default=300)
    parser.add_argument("--eval-n", type=int, default=100)
    parser.add_argument("--auto", choices=["light", "medium", "heavy"], default="medium")
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--minibatch-size", type=int, default=25)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--num-candidates", type=int, default=20)
    parser.add_argument("--task-model", type=str, default="claude-haiku-4-5",
                        help="Model for evaluating candidates (default: haiku, ~$2 total)")
    parser.add_argument("--prompt-model", type=str, default=DEFAULT_MODEL,
                        help="Model for generating instruction proposals (default: sonnet)")
    args = parser.parse_args()

    task_lm = _make_lm(args.task_model)
    prompt_lm = _make_lm(args.prompt_model)
    dspy.configure(lm=task_lm)

    print("=== DSPy MIPROv2 Instruction Optimization ===")
    print(f"  DATA_ROOT:    {DATA_ROOT}")
    print(f"  Train docs:   {args.train_n}")
    print(f"  Eval docs:    {args.eval_n}")
    print(f"  Task model:   {args.task_model}  (evaluation — {args.num_trials}×{args.minibatch_size} calls)")
    print(f"  Prompt model: {args.prompt_model}  (instruction proposals — ~{args.num_candidates} calls)")
    print(f"  Auto mode:    {args.auto}")
    print(f"  Trials:       {args.num_trials} | Minibatch: {args.minibatch_size} | Threads: {args.num_threads}")

    print("\nLoading val dataset...")
    val_ds = load_split("val")
    all_val_docs = list(val_ds)
    print(f"  Total val docs: {len(all_val_docs)}")

    train_docs = all_val_docs[: args.train_n]
    eval_docs = all_val_docs[args.train_n : args.train_n + args.eval_n]

    print(f"\nBuilding trainset ({len(train_docs)} docs)...")
    trainset = build_dspy_examples(train_docs, max_docs=args.train_n)

    print(f"\nBuilding evalset ({len(eval_docs)} docs)...")
    evalset = build_dspy_examples(eval_docs, max_docs=args.eval_n)

    print(f"\nTrainset: {len(trainset)} | Evalset: {len(evalset)}")

    module = DocILEExtractionModule()

    print("\n--- Baseline eval (first 20 evalset examples, task model) ---")
    evaluator = dspy.Evaluate(
        devset=evalset[:20],
        metric=kile_metric,
        num_threads=min(args.num_threads, 4),
        display_progress=True,
    )
    t0 = time.time()
    baseline_score = float(evaluator(module))
    print(f"  Baseline KILE field F1 proxy: {baseline_score:.4f} ({time.time()-t0:.0f}s)")

    # auto= conflicts with explicit num_candidates/num_trials in DSPy 3.x — use explicit only
    print(f"\n--- Running MIPROv2 (explicit: {args.num_candidates} candidates, {args.num_trials} trials) ---")
    optimizer = dspy.MIPROv2(
        metric=kile_metric,
        prompt_model=prompt_lm,
        task_model=task_lm,
        auto=None,  # must be None to allow explicit num_candidates + num_trials
        num_candidates=args.num_candidates,
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
        num_threads=args.num_threads,
    )

    t0 = time.time()
    optimized = optimizer.compile(
        student=module,
        trainset=trainset,
        num_trials=args.num_trials,
        minibatch=True,
        minibatch_size=args.minibatch_size,
    )
    opt_time = time.time() - t0
    print(f"\n  MIPROv2 finished in {opt_time:.0f}s ({opt_time/60:.1f}min)")

    # Save immediately after compile — before any post-eval that might crash
    models_dir = PROJECT_ROOT / "models" / "dspy"
    models_dir.mkdir(parents=True, exist_ok=True)
    compiled_path = models_dir / "compiled_miprov2.json"
    optimized.save(str(compiled_path))
    print(f"\nSaved → {compiled_path}")

    # Post-eval: use 2 threads to avoid SQLite cache contention under parallelism
    print("\n--- Optimized eval (all evalset examples, 2 threads) ---")
    eval_evaluator = dspy.Evaluate(
        devset=evalset,
        metric=kile_metric,
        num_threads=2,
        display_progress=True,
    )
    optimized_score = float(eval_evaluator(optimized))
    print(f"  Optimized KILE field F1 proxy: {optimized_score:.4f}")
    print(f"  Delta: {optimized_score - baseline_score:+.4f}")

    instruction = _extract_instruction(optimized)
    _save_optimized_prompt(
        compiled_path=compiled_path,
        instruction=instruction,
        optimizer_name="MIPROv2",
        baseline_score=baseline_score,
        optimized_score=optimized_score,
    )

    print("\n=== MIPROv2 Complete ===")
    print(f"  Baseline:  {baseline_score:.4f}  Optimized: {optimized_score:.4f}  Delta: {optimized_score-baseline_score:+.4f}")
    print(f"\nNext: run GEPA — DATA_ROOT=data uv run python tools/run_dspy_gepa.py")
    print(f"Then eval: DATA_ROOT=data uv run python tools/run_dspy_eval.py")


def _extract_instruction(mod: dspy.Module) -> str | None:
    for attr in ("extended_signature", "signature"):
        try:
            sig = getattr(mod.extract, attr)
            if hasattr(sig, "instructions"):
                return sig.instructions
        except AttributeError:
            continue
    return None


def _save_optimized_prompt(
    compiled_path: Path,
    instruction: str | None,
    optimizer_name: str,
    baseline_score: float,
    optimized_score: float,
) -> None:
    output_path = PROJECT_ROOT / "src" / "beat_docile" / "optimized_prompt.py"

    if instruction:
        safe = instruction.replace('"""', r'\"\"\"')
        body = f'OPTIMIZED_INSTRUCTION = """{safe}"""'
    else:
        body = "OPTIMIZED_INSTRUCTION = None  # check compiled JSON manually"

    output_path.write_text(f'''"""DSPy {optimizer_name} optimized extraction instruction.

Auto-generated by tools/run_dspy_optimize.py — do not edit manually.
To load: DocILEExtractionModule().load(COMPILED_PATH)
To regenerate: DATA_ROOT=data uv run python tools/run_dspy_optimize.py
"""

{body}
COMPILED_PATH = "{compiled_path}"
OPTIMIZER = "{optimizer_name}"
OPTIMIZATION_SCORE = {optimized_score:.6f}
BASELINE_SCORE = {baseline_score:.6f}
''')
    print(f"Saved optimized instruction → {output_path}")

    if instruction:
        preview = instruction[:500] + "..." if len(instruction) > 500 else instruction
        print(f"\n--- {optimizer_name} optimized instruction ---")
        print(preview)
        print("─" * 60)


if __name__ == "__main__":
    main()
