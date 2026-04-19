#!/usr/bin/env python3
"""Run DSPy GEPA optimization on DocILE V5b extraction instruction.

GEPA (Genetic-Pareto Reflective Evolution) outperforms MIPROv2 by up to +11%
on structured extraction tasks. It reflects on program trajectories and proposes
targeted instruction improvements using text feedback alongside numeric scores.

Runs on the same 300-doc trainset as MIPROv2. Saves to compiled_gepa.json.
Compare results with run_dspy_eval.py --compiled models/dspy/compiled_gepa.json.

Usage:
    DATA_ROOT=data uv run python tools/run_dspy_gepa.py
    DATA_ROOT=data uv run python tools/run_dspy_gepa.py --auto medium
    DATA_ROOT=data uv run python tools/run_dspy_gepa.py --max-metric-calls 600
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
    parser = argparse.ArgumentParser(description="DSPy GEPA instruction optimization")
    parser.add_argument("--train-n", type=int, default=300)
    parser.add_argument("--eval-n", type=int, default=100)
    parser.add_argument("--auto", choices=["light", "medium", "heavy"], default="light",
                        help="GEPA search budget (default: light — faster than MIPROv2 medium)")
    parser.add_argument("--max-metric-calls", type=int, default=None,
                        help="Override max LM calls for evaluation (default: GEPA auto)")
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--task-model", type=str, default="claude-haiku-4-5")
    parser.add_argument("--reflection-model", type=str, default=DEFAULT_MODEL,
                        help="Model for GEPA reflections/proposals (default: sonnet)")
    args = parser.parse_args()

    task_lm = _make_lm(args.task_model)
    reflection_lm = _make_lm(args.reflection_model)
    dspy.configure(lm=task_lm)

    print("=== DSPy GEPA Instruction Optimization ===")
    print(f"  DATA_ROOT:       {DATA_ROOT}")
    print(f"  Train docs:      {args.train_n}")
    print(f"  Auto mode:       {args.auto}")
    print(f"  Task model:      {args.task_model}  (evaluation)")
    print(f"  Reflection model: {args.reflection_model}  (trajectory reflection + proposals)")
    print(f"  Threads:         {args.num_threads}")

    print("\nLoading val dataset...")
    val_ds = load_split("val")
    all_val_docs = list(val_ds)
    train_docs = all_val_docs[: args.train_n]
    eval_docs = all_val_docs[args.train_n : args.train_n + args.eval_n]

    print(f"\nBuilding trainset ({len(train_docs)} docs)...")
    trainset = build_dspy_examples(train_docs, max_docs=args.train_n)

    print(f"\nBuilding evalset ({len(eval_docs)} docs)...")
    evalset = build_dspy_examples(eval_docs, max_docs=args.eval_n)

    print(f"\nTrainset: {len(trainset)} | Evalset: {len(evalset)}")

    module = DocILEExtractionModule()

    print("\n--- Baseline eval (first 20 evalset examples) ---")
    evaluator = dspy.Evaluate(
        devset=evalset[:20],
        metric=kile_metric,
        num_threads=min(args.num_threads, 4),
        display_progress=True,
    )
    t0 = time.time()
    baseline_score = float(evaluator(module))
    print(f"  Baseline KILE field F1 proxy: {baseline_score:.4f} ({time.time()-t0:.0f}s)")

    print(f"\n--- Running GEPA ({args.auto} mode) ---")
    # GEPA requires a 5-argument metric: (gold, pred, trace, pred_name, pred_trace)
    def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        return kile_metric(gold, pred, trace)

    gepa_kwargs: dict = dict(
        metric=gepa_metric,
        auto=args.auto,
        reflection_lm=reflection_lm,
        num_threads=args.num_threads,
    )
    if args.max_metric_calls is not None:
        gepa_kwargs["max_metric_calls"] = args.max_metric_calls

    optimizer = dspy.GEPA(**gepa_kwargs)

    t0 = time.time()
    optimized = optimizer.compile(
        student=module,
        trainset=trainset,
    )
    opt_time = time.time() - t0
    print(f"\n  GEPA finished in {opt_time:.0f}s ({opt_time/60:.1f}min)")

    # Save immediately — before post-eval that might crash
    models_dir = PROJECT_ROOT / "models" / "dspy"
    models_dir.mkdir(parents=True, exist_ok=True)
    compiled_path = models_dir / "compiled_gepa.json"
    optimized.save(str(compiled_path))
    print(f"\nSaved → {compiled_path}")

    # 2 threads to avoid SQLite cache contention
    print("\n--- Optimized eval (all evalset examples, 2 threads) ---")
    eval_evaluator = dspy.Evaluate(
        devset=evalset,
        metric=kile_metric,
        num_threads=2,
        display_progress=True,
    )
    optimized_score = float(eval_evaluator(optimized))
    print(f"  Optimized KILE field F1 proxy: {optimized_score:.4f}")
    print(f"  Delta vs baseline: {optimized_score - baseline_score:+.4f}")

    instruction = _extract_instruction(optimized)
    _save_if_better(
        compiled_path=compiled_path,
        instruction=instruction,
        optimized_score=optimized_score,
        baseline_score=baseline_score,
    )

    print("\n=== GEPA Complete ===")
    print(f"  Baseline:  {baseline_score:.4f}  Optimized: {optimized_score:.4f}  Delta: {optimized_score-baseline_score:+.4f}")
    print(f"\nEval with full KILE AP:")
    print(f"  MIPROv2: DATA_ROOT=data uv run python tools/run_dspy_eval.py --compiled models/dspy/compiled_miprov2.json")
    print(f"  GEPA:    DATA_ROOT=data uv run python tools/run_dspy_eval.py --compiled models/dspy/compiled_gepa.json")


def _extract_instruction(mod: dspy.Module) -> str | None:
    for attr in ("extended_signature", "signature"):
        try:
            sig = getattr(mod.extract, attr)
            if hasattr(sig, "instructions"):
                return sig.instructions
        except AttributeError:
            continue
    return None


def _save_if_better(
    compiled_path: Path,
    instruction: str | None,
    optimized_score: float,
    baseline_score: float,
) -> None:
    """Update optimized_prompt.py only if GEPA beat MIPROv2."""
    output_path = PROJECT_ROOT / "src" / "beat_docile" / "optimized_prompt.py"

    # Check if existing optimized_prompt has a better score
    existing_score = None
    if output_path.exists():
        try:
            ns: dict = {}
            exec(output_path.read_text(), ns)
            existing_score = ns.get("OPTIMIZATION_SCORE")
        except Exception:
            pass

    if existing_score is not None and existing_score >= optimized_score:
        print(f"MIPROv2 score ({existing_score:.4f}) ≥ GEPA score ({optimized_score:.4f}) — keeping MIPROv2 instruction")
        return

    print(f"GEPA ({optimized_score:.4f}) beat existing ({existing_score}) — updating optimized_prompt.py")

    if instruction:
        safe = instruction.replace('"""', r'\"\"\"')
        body = f'OPTIMIZED_INSTRUCTION = """{safe}"""'
    else:
        body = "OPTIMIZED_INSTRUCTION = None"

    output_path.write_text(f'''"""DSPy GEPA optimized extraction instruction.

Auto-generated by tools/run_dspy_gepa.py — do not edit manually.
To load: DocILEExtractionModule().load(COMPILED_PATH)
"""

{body}
COMPILED_PATH = "{compiled_path}"
OPTIMIZER = "GEPA"
OPTIMIZATION_SCORE = {optimized_score:.6f}
BASELINE_SCORE = {baseline_score:.6f}
''')
    print(f"Updated optimized_prompt.py with GEPA instruction")

    if instruction:
        preview = instruction[:500] + "..." if len(instruction) > 500 else instruction
        print(f"\n--- GEPA optimized instruction ---")
        print(preview)
        print("─" * 60)


if __name__ == "__main__":
    main()
