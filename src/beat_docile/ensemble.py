"""[ACTIVE] Per-field merge module — groups overlapping predictions across sources.

Status: ACTIVE — used in current best (v2_ensemble).
See KNOWLEDGE_BASE.md §3 for the architecture map.

PCC-IoU=1.0 scoring penalises duplicate boxes hard, so the merger groups
overlapping same-fieldtype predictions per doc and emits one combined field
per group. KILE buckets by (fieldtype); LIR buckets by (fieldtype, line_item_id).

Schema of input/output prediction JSONs matches `cli.py extract`:
    {docid: [{"bbox": [l,t,r,b], "page": int, "score": float,
              "fieldtype": str, "line_item_id": int|null, "text": str|null,
              "use_only_for_ap": bool}, ...]}
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import typer
from docile.dataset import BBox, Field

from .extract import _KILE_TYPES, _LIR_TYPES

# ── Bbox geometry ─────────────────────────────────────────────────────────────


def _iou(b1: BBox, b2: BBox) -> float:
    """Standard Intersection-over-Union for two bboxes (any units, normalized [0,1] here)."""
    ix1, iy1 = max(b1.left, b2.left), max(b1.top, b2.top)
    ix2, iy2 = min(b1.right, b2.right), min(b1.bottom, b2.bottom)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0.0:
        return 0.0
    a1 = max(0.0, b1.right - b1.left) * max(0.0, b1.bottom - b1.top)
    a2 = max(0.0, b2.right - b2.left) * max(0.0, b2.bottom - b2.top)
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


# ── Score combination ─────────────────────────────────────────────────────────

_COMBINERS = {"weighted_max", "weighted_mean", "vote"}


def _combine_scores(scores: list[float], weights: list[float], mode: str, n_sources: int) -> float:
    """Combine per-source (score, weight) pairs into a single ensemble score in [0, 1]."""
    if not scores:
        return 0.0
    if mode == "weighted_max":
        return min(1.0, max(s * w for s, w in zip(scores, weights, strict=False)))
    if mode == "weighted_mean":
        wsum = sum(weights)
        if wsum <= 0:
            return 0.0
        return min(1.0, sum(s * w for s, w in zip(scores, weights, strict=False)) / wsum)
    if mode == "vote":
        # Fraction of (weighted) sources that voted for this prediction.
        # Normalised against total weight across all input sources so a
        # prediction backed by every source approaches 1.0.
        total_weight = sum(weights) if n_sources == len(weights) else None
        if total_weight is None:
            # n_sources is the count of input sources; use uniform total
            total_weight = float(n_sources)
        return min(1.0, sum(weights) / total_weight if total_weight > 0 else 0.0)
    raise ValueError(f"unknown score_combine: {mode!r} (expected one of {_COMBINERS})")


# ── Core merge ────────────────────────────────────────────────────────────────


def _bucket_key(field: Field) -> tuple:
    """Group key: KILE buckets by fieldtype; LIR additionally by line_item_id.

    Different pages also kept separate so we never collapse cross-page boxes.
    """
    if field.line_item_id is not None:
        return ("lir", field.fieldtype, field.line_item_id, field.page)
    return ("kile", field.fieldtype, field.page)


def _group_overlapping(
    items: list[tuple[Field, int]], iou_threshold: float
) -> list[list[tuple[Field, int]]]:
    """Greedy single-link clustering of (field, source_idx) by bbox IoU.

    Two items belong to the same group if their bboxes have IoU >= threshold.
    Order-stable: groups are seeded in input order.
    """
    groups: list[list[tuple[Field, int]]] = []
    for item in items:
        field = item[0]
        placed = False
        for grp in groups:
            # Match if it overlaps any member of the group (single-link)
            if any(_iou(field.bbox, g[0].bbox) >= iou_threshold for g in grp):
                grp.append(item)
                placed = True
                break
        if not placed:
            groups.append([item])
    return groups


def _merge_group(
    group: list[tuple[Field, int]],
    weights: list[float],
    score_combine: str,
    n_sources: int,
) -> Field:
    """Collapse a group of overlapping fields into a single Field.

    Bbox: take from the highest weighted-score contributor (tie → first).
    Score: combined per `score_combine`. Other attrs inherited from that contributor.
    """
    weighted = [(f, w_idx, f.score * weights[w_idx]) for f, w_idx in group]
    # pick representative by weighted score, ties broken by original order
    best_idx = max(range(len(weighted)), key=lambda i: (weighted[i][2], -i))
    rep_field, _rep_src, _ = weighted[best_idx]

    # Per-source dedup: if multiple fields from same source land in one group,
    # take the strongest one before combining (avoids double-counting weight).
    by_source: dict[int, float] = {}
    for f, src in group:
        prev = by_source.get(src, -1.0)
        if f.score > prev:
            by_source[src] = f.score

    scores = list(by_source.values())
    grp_weights = [weights[s] for s in by_source]
    new_score = _combine_scores(scores, grp_weights, score_combine, n_sources)

    return Field(
        bbox=rep_field.bbox,
        page=rep_field.page,
        fieldtype=rep_field.fieldtype,
        score=new_score,
        line_item_id=rep_field.line_item_id,
        text=rep_field.text,
    )


def merge_predictions(
    sources: list[dict[str, list[Field]]],
    weights: list[float] | None = None,
    iou_threshold: float = 0.5,
    score_combine: str = "weighted_max",
) -> dict[str, list[Field]]:
    """Merge per-doc, per-fieldtype predictions across sources.

    For each (docid, fieldtype) bucket:
      - Group fields whose bboxes overlap above iou_threshold.
      - Within a group, combine scores per `score_combine` and emit ONE merged field.
      - Singletons survive scaled by their source weight.
      - For LIR fields: also key by line_item_id (group only within same li_id).
    """
    if not sources:
        return {}
    if score_combine not in _COMBINERS:
        raise ValueError(f"unknown score_combine: {score_combine!r}")
    if weights is None:
        weights = [1.0] * len(sources)
    if len(weights) != len(sources):
        raise ValueError(f"weights length {len(weights)} != sources length {len(sources)}")

    n_sources = len(sources)
    all_docids: set[str] = set()
    for src in sources:
        all_docids.update(src.keys())

    merged: dict[str, list[Field]] = {}
    for docid in sorted(all_docids):
        # Collect (field, source_idx) for this doc, grouped by bucket key
        buckets: dict[tuple, list[tuple[Field, int]]] = defaultdict(list)
        for src_idx, src in enumerate(sources):
            for f in src.get(docid, []):
                buckets[_bucket_key(f)].append((f, src_idx))

        out: list[Field] = []
        for items in buckets.values():
            for group in _group_overlapping(items, iou_threshold):
                out.append(_merge_group(group, weights, score_combine, n_sources))
        merged[docid] = out
    return merged


# ── Serialization helpers ─────────────────────────────────────────────────────


def load_predictions(path: Path | str) -> dict[str, list[Field]]:
    """Load a predictions JSON (cli.py extract format) into {docid: [Field, ...]}."""
    raw = json.loads(Path(path).read_text())
    out: dict[str, list[Field]] = {}
    for docid, fields in raw.items():
        out[docid] = [Field.from_dict(fd) for fd in fields]
    return out


def save_predictions(preds: dict[str, list[Field]], path: Path | str) -> None:
    """Serialize {docid: [Field, ...]} back to the cli.py extract JSON format."""
    out = {docid: [f.to_dict() for f in fields] for docid, fields in preds.items()}
    Path(path).write_text(json.dumps(out, indent=2))


def _iter_field_types(preds: Iterable[Field]) -> tuple[int, int]:
    n_kile = sum(1 for f in preds if f.fieldtype in _KILE_TYPES and f.line_item_id is None)
    n_lir = sum(1 for f in preds if f.fieldtype in _LIR_TYPES and f.line_item_id is not None)
    return n_kile, n_lir


# ── CLI ────────────────────────────────────────────────────────────────────────

app = typer.Typer(help="Ensemble multiple prediction sources into one.")


@app.command()
def merge(
    inputs: list[Path] = typer.Option(  # noqa: B008
        ..., "--inputs", "-i", help="Prediction JSON files to merge."
    ),
    weights: list[float] = typer.Option(  # noqa: B008
        None, "--weights", "-w", help="Per-source weight (defaults to uniform)."
    ),
    iou_threshold: float = typer.Option(0.5, help="IoU threshold for grouping overlapping bboxes."),
    score_combine: str = typer.Option("weighted_max", help="weighted_max | weighted_mean | vote"),
    out: Path = typer.Option(Path("merged.json"), "--out", "-o", help="Output JSON path."),  # noqa: B008
) -> None:
    """Merge two or more predictions JSONs and write a combined file."""
    if len(inputs) < 1:
        raise typer.BadParameter("at least one --inputs path required")
    if weights and len(weights) != len(inputs):
        raise typer.BadParameter("--weights count must match --inputs count")

    sources = [load_predictions(p) for p in inputs]
    merged = merge_predictions(
        sources,
        weights=weights or None,
        iou_threshold=iou_threshold,
        score_combine=score_combine,
    )
    save_predictions(merged, out)

    total = sum(len(v) for v in merged.values())
    typer.echo(f"Merged {len(inputs)} sources → {len(merged)} docs, {total} fields → {out}")


if __name__ == "__main__":
    app()
