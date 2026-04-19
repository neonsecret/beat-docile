"""[ACTIVE] Main CLI entry point — commands: smoke, extract, eval, vlm-extract.

Status: ACTIVE — used in current best (v2_ensemble).
See KNOWLEDGE_BASE.md §3 for the architecture map.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console

from .config import DATA_ROOT, DEFAULT_MODEL
from .data import load_split

app = typer.Typer(help="beat_docile — DocILE pipeline CLI")
console = Console()


@app.command()
def smoke(
    model: str = typer.Option(DEFAULT_MODEL, help="Claude model ID"),
    limit: int = typer.Option(5, help="Number of val docs to process"),
    few_shot: bool = typer.Option(True, help="Use cluster-based few-shot examples"),
) -> None:
    """Run zero-shot extraction on N val docs and print eval scores."""
    from .eval import print_scores, run_eval
    from .extract import extract_documents

    console.print(f"[bold]Loading val split[/bold] (first {limit} docs)...")
    dataset = load_split("val")
    docs = list(dataset)[:limit]
    docids = [d.docid for d in docs]
    console.print(f"Docs: {docids}")

    train_index = None
    if few_shot:
        from .fewshot import _build_cluster_index

        console.print("[bold]Building train cluster index for few-shot...[/bold]")
        train_index = _build_cluster_index("train")
        console.print(f"Loaded {len(train_index)} clusters from train split.")

    console.print("[bold]Extracting...[/bold]")
    kile_preds, lir_preds = asyncio.run(extract_documents(docs, model, train_index=train_index))

    total_kile = sum(len(v) for v in kile_preds.values())
    total_lir = sum(len(v) for v in lir_preds.values())
    console.print(f"Extracted KILE fields: {total_kile}, LIR fields: {total_lir}")

    for docid in docids:
        k = len(kile_preds.get(docid, []))
        n_lir = len(lir_preds.get(docid, []))
        console.print(f"  {docid}: {k} KILE, {n_lir} LIR")

    # Build subset dataset for eval using a custom split name (avoids index-mismatch error)
    from docile.dataset import Dataset

    subset_dataset = Dataset(
        split_name="smoke_subset",
        dataset_path=DATA_ROOT,
        load_annotations=True,
        load_ocr=False,
        docids=docids,
    )

    console.print("[bold]Evaluating...[/bold]")
    result = run_eval(subset_dataset, kile_preds, lir_preds)
    scores = print_scores(result)
    console.print(f"\n[green]Scores: {scores}[/green]")


@app.command()
def extract(
    split: str = typer.Option("val", help="Dataset split"),
    limit: int | None = typer.Option(None, help="Max docs to process"),
    model: str = typer.Option(DEFAULT_MODEL, help="Claude model ID"),
    out: Path = typer.Option(Path("predictions.json"), help="Output JSON path"),  # noqa: B008
    few_shot: bool = typer.Option(True, help="Use cluster-based few-shot examples"),
    targeted: bool = typer.Option(
        True, help="Run targeted second pass for financial/registration fields"
    ),
    sc: bool = typer.Option(False, help="Use self-consistency (3 samples)"),
    cluster_map: Path | None = typer.Option(  # noqa: B008
        None, help="JSON {docid: cluster_id} override (for test docs without annotated cluster_id)"
    ),
) -> None:
    """Run extraction on a split and write predictions JSON."""
    from .extract import extract_documents

    dataset = load_split(split)
    docs = list(dataset)
    if limit is not None:
        docs = docs[:limit]

    cluster_override: dict[str, int] | None = None
    if cluster_map is not None:
        cluster_override = json.loads(cluster_map.read_text())
        console.print(
            f"Loaded cluster override for {len(cluster_override)} docs from {cluster_map}"
        )

    train_index = None
    if few_shot:
        from .fewshot import _build_cluster_index

        console.print("[bold]Building train cluster index for few-shot...[/bold]")
        train_index = _build_cluster_index("train")
        console.print(f"Loaded {len(train_index)} clusters from train split.")

    console.print(
        f"Extracting {len(docs)} docs from {split} split (few_shot={few_shot}, targeted={targeted}, sc={sc})..."
    )
    kile_preds, lir_preds = asyncio.run(
        extract_documents(
            docs,
            model,
            train_index=train_index,
            targeted_pass=targeted,
            self_consistency=sc,
            cluster_override=cluster_override,
        )
    )

    # Serialize to DocILE prediction format: {docid: [field_dict, ...]}
    # KILE + LIR merged per doc; KILE fields have no line_item_id
    output: dict[str, list[dict]] = {}
    for docid in [d.docid for d in docs]:
        fields_out = []
        for f in kile_preds.get(docid, []):
            fields_out.append(f.to_dict())
        for f in lir_preds.get(docid, []):
            fields_out.append(f.to_dict())
        output[docid] = fields_out

    out.write_text(json.dumps(output, indent=2))
    console.print(f"Written to {out}")


@app.command(name="eval")
def eval_cmd(
    split: str = typer.Option("val", help="Dataset split"),
    predictions: Path = typer.Option(..., help="Predictions JSON file"),  # noqa: B008
) -> None:
    """Evaluate a predictions JSON against ground truth."""
    from docile.dataset import Field

    from .eval import print_scores, run_eval

    dataset = load_split(split)

    raw = json.loads(predictions.read_text())
    kile_preds: dict[str, list[Field]] = {}
    lir_preds: dict[str, list[Field]] = {}

    for docid, fields in raw.items():
        kile_preds[docid] = []
        lir_preds[docid] = []
        for fd in fields:
            f = Field.from_dict(fd)
            if f.line_item_id is not None:
                lir_preds[docid].append(f)
            else:
                kile_preds[docid].append(f)

    result = run_eval(dataset, kile_preds, lir_preds)
    print_scores(result)


@app.command(name="vlm-extract")
def vlm_extract_cmd(
    split: str = typer.Option("val", help="Dataset split"),
    limit: int | None = typer.Option(50, help="Max docs to process"),
    model_dir: Path = typer.Option(  # noqa: B008
        Path.home() / "qwen3vl_docile", help="Path to merged VLM checkpoint"  # noqa: B008
    ),
    out: Path | None = typer.Option(None, help="Save predictions JSON"),  # noqa: B008
) -> None:
    """Run fine-tuned Qwen3-VL-2B inference on a split and print KILE AP."""
    from docile.dataset import Dataset

    from .eval import print_scores, run_eval
    from .vlm_extract import extract_documents as vlm_extract_documents

    console.print(f"[bold]VLM extract:[/bold] {split} split, {limit} docs, model={model_dir}")
    kile_preds = vlm_extract_documents(split, model_dir=model_dir, limit=limit)

    if out:
        import json as _json

        out.write_text(
            _json.dumps(
                {
                    k: [
                        {
                            "bbox": list(f.bbox.to_tuple()),
                            "page": f.page,
                            "text": f.text,
                            "fieldtype": f.fieldtype,
                            "score": f.score,
                        }
                        for f in v
                    ]
                    for k, v in kile_preds.items()
                },
                indent=2,
            )
        )
        console.print(f"Saved to {out}")

    docids = list(kile_preds.keys())
    subset_dataset = Dataset(
        split_name=f"vlm_{split}_subset",
        dataset_path=DATA_ROOT,
        load_annotations=True,
        load_ocr=False,
        docids=docids,
    )
    result = run_eval(subset_dataset, kile_preds, lir_preds={})
    print_scores(result)
