"""[EXPERIMENTAL] Qwen3-VL-Embedding-2B cluster inference for DocILE test documents.

Status: EXPERIMENTAL — built, 70.4% top-1 accuracy on val, not yet wired into
production pipeline. See KNOWLEDGE_BASE.md §3.4 for details. Intended use:
fill the 25% no-match val docs with a predicted cluster so they get few-shot
coverage (estimated +2-3pp dataset-wide, see §5.6 / §8.7).

Train docs have annotated cluster_id; test docs do not. Builds 2048-dim image
embeddings via Qwen3-VL-Embedding-2B, then uses 1-NN cosine similarity.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

_MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
_EMBED_DIM = 2048


@dataclass
class ClusterPrediction:
    docid: str
    inferred_cluster_id: int
    confidence: float  # cosine similarity to nearest train neighbor
    nearest_train_docid: str
    nearest_distance: float  # 1 - cosine_similarity


def _resolve_device(device: str) -> str:
    """Return a valid torch device string, falling back to cpu if unavailable."""
    if device == "mps" and torch.backends.mps.is_available():
        return "mps"
    if device == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_qwen3vl_model(device: str = "mps"):
    """Load Qwen3-VL-Embedding-2B via sentence-transformers. Returns (model, resolved_device)."""
    from sentence_transformers import SentenceTransformer

    dev = _resolve_device(device)
    logger.info("Loading %s on %s", _MODEL_NAME, dev)
    model = SentenceTransformer(
        _MODEL_NAME,
        trust_remote_code=True,
        device=dev,
    )
    return model, dev


def embed_doc_qwen3vl(doc, model, device: str = "mps") -> np.ndarray:
    """Single-doc Qwen3-VL image embedding using page 0.

    Returns L2-normalised float32 embedding of shape (2048,).
    The caller must open ``doc`` with a context manager before calling.
    """
    from PIL import Image as PILImage

    image: PILImage.Image = doc.page_image(0)
    emb = model.encode([image], convert_to_numpy=True, show_progress_bar=False)
    emb_np = emb[0].astype(np.float32)
    norm = np.linalg.norm(emb_np)
    return emb_np / (norm + 1e-8)


def build_train_embeddings(
    train_docs: list,
    output_path: Path,
    device: str = "mps",
) -> dict:
    """Embed all train docs (with cluster_id) and save to a .npz file.

    The .npz contains three arrays:
      - ``embeddings``:  float32 (N, 2048)
      - ``cluster_ids``: int32   (N,)
      - ``docids``:      object  (N,) — UTF-8 strings

    Returns stats dict with n_embedded, n_failed, model_name, embedding_dim,
    elapsed_sec, output_path.
    """
    model, _dev = load_qwen3vl_model(device)
    embeddings: list[np.ndarray] = []
    cluster_ids: list[int] = []
    docids: list[str] = []
    n_failed = 0
    t0 = time.monotonic()

    for i, doc in enumerate(train_docs):
        try:
            with doc:
                cid = doc.annotation.cluster_id
                emb = embed_doc_qwen3vl(doc, model, device)
            embeddings.append(emb)
            cluster_ids.append(int(cid))
            docids.append(doc.docid)
        except Exception:
            logger.warning("Failed to embed doc %s", doc.docid, exc_info=True)
            n_failed += 1

        if (i + 1) % 100 == 0:
            elapsed = time.monotonic() - t0
            rate = (i + 1) / elapsed
            eta = (len(train_docs) - i - 1) / rate
            logger.info(
                "Embedded %d/%d docs (%.1f/s, ETA %.0fmin)",
                i + 1,
                len(train_docs),
                rate,
                eta / 60,
            )

    if not embeddings:
        raise RuntimeError("No embeddings produced — check data path and doc list")

    matrix = np.stack(embeddings)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        embeddings=matrix.astype(np.float32),
        cluster_ids=np.array(cluster_ids, dtype=np.int32),
        docids=np.array(docids, dtype=object),
    )
    elapsed = time.monotonic() - t0
    stats = {
        "n_embedded": len(embeddings),
        "n_failed": n_failed,
        "model_name": _MODEL_NAME,
        "embedding_dim": int(matrix.shape[1]),
        "elapsed_sec": round(elapsed, 1),
        "output_path": str(output_path),
    }
    logger.info("Train embeddings saved: %s", stats)
    return stats


def _load_train_npz(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load train .npz. Returns (embeddings float32 (N,D), cluster_ids int32 (N,), docids)."""
    data = np.load(path, allow_pickle=True)
    return (
        data["embeddings"].astype(np.float32),
        data["cluster_ids"].astype(np.int32),
        [str(d) for d in data["docids"]],
    )


def infer_cluster(
    query_doc,
    train_embeddings_path: Path,
    model,
    device: str = "mps",
    k: int = 1,
) -> ClusterPrediction:
    """1-NN (or k-NN majority-vote) cluster lookup by cosine similarity.

    Args:
        query_doc: A Document object (not yet open).
        train_embeddings_path: Path to the .npz built by build_train_embeddings.
        model: Loaded SentenceTransformer model.
        device: Torch device string.
        k: k=1 → nearest neighbour; k>1 → majority vote across top-k.
    """
    train_embs, train_cids, train_docids = _load_train_npz(train_embeddings_path)
    with query_doc:
        query_emb = embed_doc_qwen3vl(query_doc, model, device)

    sims = train_embs @ query_emb  # (N,) cosine sims — both sides L2-normalised
    top_k_idx = np.argsort(sims)[::-1][:k]
    nearest_idx = int(top_k_idx[0])
    nearest_sim = float(sims[nearest_idx])

    if k == 1:
        inferred_cid = int(train_cids[nearest_idx])
    else:
        vote_counts = Counter(int(train_cids[i]) for i in top_k_idx)
        inferred_cid = vote_counts.most_common(1)[0][0]

    return ClusterPrediction(
        docid=query_doc.docid,
        inferred_cluster_id=inferred_cid,
        confidence=nearest_sim,
        nearest_train_docid=train_docids[nearest_idx],
        nearest_distance=1.0 - nearest_sim,
    )


def infer_clusters_batch(
    docs: list,
    train_embeddings_path: Path,
    output_path: Path,
    device: str = "mps",
    k: int = 1,
) -> dict[str, ClusterPrediction]:
    """Batch-embed query docs and 1-NN lookup against train embeddings.

    Saves ``{docid: cluster_id}`` JSON to ``output_path``.
    """
    model, _dev = load_qwen3vl_model(device)
    train_embs, train_cids, train_docids = _load_train_npz(train_embeddings_path)

    query_embs: list[np.ndarray] = []
    query_docids: list[str] = []
    n_failed = 0
    t0 = time.monotonic()

    logger.info("Embedding %d query docs for cluster inference", len(docs))
    for i, doc in enumerate(docs):
        try:
            with doc:
                emb = embed_doc_qwen3vl(doc, model, device)
            query_embs.append(emb)
            query_docids.append(doc.docid)
        except Exception:
            logger.warning("Failed to embed query doc %s", doc.docid, exc_info=True)
            n_failed += 1

        if (i + 1) % 100 == 0:
            elapsed = time.monotonic() - t0
            logger.info("Embedded %d/%d query docs (%.1fs)", i + 1, len(docs), elapsed)

    if not query_embs:
        raise RuntimeError("No query embeddings produced — check data path and doc list")

    q_matrix = np.stack(query_embs).astype(np.float32)  # (n_query, 2048)
    sims = q_matrix @ train_embs.T  # (n_query, n_train)

    results: dict[str, ClusterPrediction] = {}
    for i, docid in enumerate(query_docids):
        top_k_idx = np.argsort(sims[i])[::-1][:k]
        nearest_idx = int(top_k_idx[0])
        nearest_sim = float(sims[i, nearest_idx])

        if k == 1:
            inferred_cid = int(train_cids[nearest_idx])
        else:
            vote_counts = Counter(int(train_cids[j]) for j in top_k_idx)
            inferred_cid = vote_counts.most_common(1)[0][0]

        results[docid] = ClusterPrediction(
            docid=docid,
            inferred_cluster_id=inferred_cid,
            confidence=nearest_sim,
            nearest_train_docid=train_docids[nearest_idx],
            nearest_distance=1.0 - nearest_sim,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cluster_map = {docid: pred.inferred_cluster_id for docid, pred in results.items()}
    with open(output_path, "w") as f:
        json.dump(cluster_map, f, indent=2)

    elapsed = time.monotonic() - t0
    logger.info(
        "Saved %d cluster predictions to %s (%d failed, %.1fs)",
        len(results),
        output_path,
        n_failed,
        elapsed,
    )
    return results


def validate_val_accuracy(
    val_preds: dict[str, ClusterPrediction],
    val_docs,
    k_values: tuple[int, ...] = (1, 3),
) -> dict:
    """Compute top-1 cluster accuracy on val docs that have ground-truth cluster_id.

    Returns dict with top1_accuracy, n_correct_top1, n_docs, confidence stats.
    """
    correct_top1 = 0
    total = 0
    confidences: list[float] = []

    for doc in val_docs:
        docid = doc.docid
        if docid not in val_preds:
            continue
        pred = val_preds[docid]
        true_cid = doc.annotation.cluster_id
        if pred.inferred_cluster_id == true_cid:
            correct_top1 += 1
        confidences.append(pred.confidence)
        total += 1

    if total == 0:
        return {"error": "no val docs found in predictions"}

    conf_arr = np.array(confidences)
    return {
        "top1_accuracy": correct_top1 / total,
        "n_correct_top1": correct_top1,
        "n_docs": total,
        "confidence_mean": float(conf_arr.mean()),
        "confidence_p25": float(np.percentile(conf_arr, 25)),
        "confidence_median": float(np.median(conf_arr)),
        "confidence_p75": float(np.percentile(conf_arr, 75)),
        "high_confidence_frac": float((conf_arr > 0.8).mean()),
    }
