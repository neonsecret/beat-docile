"""[ARCHIVED] Typer CLI for V6 extraction pipeline.

Status: ARCHIVED — CLI for the buried V6 ReAct pipeline (22.7% KILE).
See KNOWLEDGE_BASE.md §6.7. Kept for code-archaeology only.

Original commands: beat_docile-v6 extract / eval.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from rich.console import Console

from .v6_pipeline import evaluate_v6, run_v6_on_docids

app = typer.Typer(name="beat_docile-v6", help="V6 ReAct extraction pipeline.")
console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@app.command("extract")
def extract_cmd(
    docids_file: Path = typer.Option(..., "--docids-file", help="JSON file with list of docids."),  # noqa: B008
    out: Path = typer.Option(..., "--out", help="Output predictions JSON path."),  # noqa: B008
    no_haiku_verify: bool = typer.Option(
        False, "--no-haiku-verify", help="Disable Haiku verifier."
    ),
    no_classifier_tool: bool = typer.Option(
        False, "--no-classifier-tool", help="Disable classifier tool."
    ),
) -> None:
    """Extract V6 predictions for a set of document IDs."""
    docids: list[str] = json.loads(docids_file.read_text())
    console.print(
        f"[bold]V6 extract[/bold]: {len(docids)} docs → {out}  "
        f"haiku_verify={'off' if no_haiku_verify else 'on'}  "
        f"classifier={'off' if no_classifier_tool else 'on'}"
    )
    run_v6_on_docids(
        docids=docids,
        output_path=out,
        use_haiku_verify=not no_haiku_verify,
        use_classifier_tool=not no_classifier_tool,
        progress=True,
    )
    console.print(f"[green]Done.[/green] Predictions written to {out}")


@app.command("eval")
def eval_cmd(
    predictions: Path = typer.Argument(..., help="Path to predictions JSON file."),  # noqa: B008
) -> None:
    """Evaluate a predictions JSON file against DocILE val annotations."""
    console.print(f"[bold]V6 eval[/bold]: {predictions}")
    metrics = evaluate_v6(predictions)
    console.print(
        f"KILE AP={metrics['kile_ap']:.4f}  P={metrics['kile_p']:.3f}  R={metrics['kile_r']:.3f}"
    )
    console.print(
        f"LIR  F1={metrics['lir_f1']:.4f}  P={metrics['lir_p']:.3f}  R={metrics['lir_r']:.3f}"
    )
