#!/usr/bin/env python
"""CLIP cluster inference pipeline.

Build train embeddings → infer val/test clusters → validate → save mappings.

Usage
-----
# Step 1 — build train embeddings (run on Mac with MPS, or RunPod CPU)
uv run python tools/run_clip_cluster.py build-train \\
    --output models/clip_train_embeddings.npz \\
    --device mps

# Step 1b — same, from a RunPod-style path without MPS
python tools/run_clip_cluster.py build-train \\
    --output /workspace/clip_train_embeddings.npz \\
    --device cpu \\
    --data-root /workspace/data

# Step 2 — infer clusters for val (has ground truth → reports accuracy)
uv run python tools/run_clip_cluster.py infer-val \\
    --train-npz models/clip_train_embeddings.npz \\
    --output predictions/val_inferred_clusters.json \\
    --device mps

# Step 3 — infer clusters for test (no ground truth)
uv run python tools/run_clip_cluster.py infer-test \\
    --train-npz models/clip_train_embeddings.npz \\
    --output predictions/test_inferred_clusters.json \\
    --device mps
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

# Allow running from project root without installing the package
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from beat_docile.cluster_infer import (  # noqa: E402
    build_train_embeddings,
    infer_clusters_batch,
    validate_val_accuracy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_clip_cluster")

console = Console()
app = typer.Typer(help=__doc__, add_completion=False)

_DEFAULT_DATA_ROOT = Path.home() / "beat_docile" / "data"
_PROJECT_DATA_ROOT = _PROJECT_ROOT / "data"


def _resolve_data_root(data_root: Path | None) -> Path:
    if data_root is not None:
        return data_root
    env_val = os.environ.get("DATA_ROOT")
    if env_val:
        return Path(env_val)
    if _PROJECT_DATA_ROOT.exists():
        return _PROJECT_DATA_ROOT
    return _DEFAULT_DATA_ROOT


def _load_split_docs(split: str, data_root: Path, load_annotations: bool = True):
    """Load a DocILE split, filtering to docids with local PDFs."""
    from docile.dataset import Dataset

    index_file = data_root / f"{split}.json"
    if not index_file.exists():
        raise FileNotFoundError(f"Split index {index_file} not found — check DATA_ROOT")

    with open(index_file) as f:
        all_docids = json.load(f)

    pdf_dir = data_root / "pdfs"
    local_pdfs = set(p.stem for p in pdf_dir.glob("*.pdf"))
    available = [d for d in all_docids if d in local_pdfs]

    if len(available) < len(all_docids):
        logger.warning(
            "Split %s: %d/%d docids have local PDFs — embedding subset only",
            split, len(available), len(all_docids),
        )

    # docile Dataset raises if docids don't match the index; use smoke_subset trick
    ds = Dataset(
        "smoke_subset",
        data_root,
        load_annotations=load_annotations,
        load_ocr=False,
        docids=available,
    )
    return list(ds), all_docids


@app.command()
def build_train(
    output: Annotated[Path, typer.Option(help="Output .npz path for train embeddings")] = Path(
        "models/clip_train_embeddings.npz"
    ),
    device: Annotated[str, typer.Option(help="Torch device: mps (Mac), cpu, or cuda")] = "mps",
    data_root: Annotated[Path | None, typer.Option(help="Override DATA_ROOT env var")] = None,
) -> None:
    """Build CLIP embeddings for ALL available train documents and save to .npz."""
    root = _resolve_data_root(data_root)
    console.print(f"[bold]DATA_ROOT:[/bold] {root}")
    console.print(f"[bold]Output:[/bold] {output}")
    console.print(f"[bold]Device:[/bold] {device}")

    train_docs, all_train_ids = _load_split_docs("train", root, load_annotations=True)
    console.print(
        f"Embedding [cyan]{len(train_docs)}[/cyan] train docs "
        f"(of {len(all_train_ids)} total in split)"
    )

    stats = build_train_embeddings(train_docs, Path(output), device=device)
    console.print("\n[green]Done.[/green] Stats:")
    for k, v in stats.items():
        console.print(f"  {k}: {v}")


@app.command()
def infer_val(
    train_npz: Annotated[Path, typer.Option(help="Path to train embeddings .npz")],
    output: Annotated[Path, typer.Option(help="Output JSON path")] = Path(
        "predictions/val_inferred_clusters.json"
    ),
    k: Annotated[int, typer.Option(help="k-NN neighbourhood size")] = 1,
    device: Annotated[str, typer.Option(help="Torch device: mps, cpu, or cuda")] = "mps",
    data_root: Annotated[Path | None, typer.Option(help="Override DATA_ROOT env var")] = None,
    gate: Annotated[float, typer.Option(help="Minimum top-1 accuracy gate")] = 0.60,
) -> None:
    """Infer clusters for val docs and report accuracy against ground truth."""
    root = _resolve_data_root(data_root)
    train_npz = Path(train_npz)
    if not train_npz.exists():
        console.print(f"[red]ERROR:[/red] train NPZ not found: {train_npz}")
        raise typer.Exit(1)

    val_docs, _ = _load_split_docs("val", root, load_annotations=True)
    console.print(f"Inferring clusters for [cyan]{len(val_docs)}[/cyan] val docs (k={k})")

    preds = infer_clusters_batch(val_docs, train_npz, Path(output), device=device, k=k)

    # Reload val docs for ground-truth comparison (docs are consumed after embedding)
    val_docs2, _ = _load_split_docs("val", root, load_annotations=True)
    accuracy = validate_val_accuracy(preds, val_docs2)

    _print_accuracy_report(accuracy, gate=gate)
    console.print(f"\nCluster map saved → [bold]{output}[/bold]")


@app.command()
def infer_test(
    train_npz: Annotated[Path, typer.Option(help="Path to train embeddings .npz")],
    output: Annotated[Path, typer.Option(help="Output JSON path")] = Path(
        "predictions/test_inferred_clusters.json"
    ),
    k: Annotated[int, typer.Option(help="k-NN neighbourhood size")] = 1,
    device: Annotated[str, typer.Option(help="Torch device: mps, cpu, or cuda")] = "mps",
    data_root: Annotated[Path | None, typer.Option(help="Override DATA_ROOT env var")] = None,
) -> None:
    """Infer clusters for test docs (no ground truth) and report confidence distribution."""
    root = _resolve_data_root(data_root)
    train_npz = Path(train_npz)
    if not train_npz.exists():
        console.print(f"[red]ERROR:[/red] train NPZ not found: {train_npz}")
        raise typer.Exit(1)

    test_docs, _ = _load_split_docs("test", root, load_annotations=False)
    console.print(f"Inferring clusters for [cyan]{len(test_docs)}[/cyan] test docs (k={k})")

    preds = infer_clusters_batch(test_docs, train_npz, Path(output), device=device, k=k)
    _print_confidence_distribution(preds)
    console.print(f"\nCluster map saved → [bold]{output}[/bold]")


def _print_accuracy_report(accuracy: dict, gate: float = 0.60) -> None:
    t = Table(title="Val Cluster Accuracy")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", style="bold")

    top1 = accuracy.get("top1_accuracy", 0)
    t.add_row("Top-1 accuracy", f"{top1:.1%}")
    t.add_row("Correct / Total", f"{accuracy.get('n_correct_top1')} / {accuracy.get('n_docs')}")
    t.add_row("Confidence mean", f"{accuracy.get('confidence_mean', 0):.3f}")
    t.add_row("Confidence median", f"{accuracy.get('confidence_median', 0):.3f}")
    t.add_row(
        "Confidence P25/P75",
        f"{accuracy.get('confidence_p25', 0):.3f} / {accuracy.get('confidence_p75', 0):.3f}",
    )
    t.add_row("High-conf frac (>0.8)", f"{accuracy.get('high_confidence_frac', 0):.1%}")
    console.print(t)

    if top1 < gate:
        console.print(
            f"\n[red bold]GATE FAILED[/red bold]: top-1 accuracy {top1:.1%} < {gate:.0%}. "
            "CLIP isn't picking up cluster signal — consider text-based clustering."
        )
    else:
        console.print(
            f"\n[green bold]GATE PASSED[/green bold]: top-1 accuracy {top1:.1%} ≥ {gate:.0%}. "
            "Proceeding to test inference is safe."
        )


def _print_confidence_distribution(preds: dict) -> None:
    confs = np.array([p.confidence for p in preds.values()])
    t = Table(title="Test Confidence Distribution")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", style="bold")
    t.add_row("Docs inferred", str(len(preds)))
    t.add_row("Confidence mean", f"{confs.mean():.3f}")
    t.add_row("Confidence median", f"{np.median(confs):.3f}")
    t.add_row(
        "Confidence P25/P75",
        f"{np.percentile(confs, 25):.3f} / {np.percentile(confs, 75):.3f}",
    )
    t.add_row("High-conf (>0.8)", f"{(confs > 0.8).sum()} ({(confs > 0.8).mean():.1%})")
    t.add_row("Low-conf (<0.6)", f"{(confs < 0.6).sum()} ({(confs < 0.6).mean():.1%})")
    console.print(t)


if __name__ == "__main__":
    app()
