"""[EXPERIMENTAL] CLIP-based document embeddings (used only by sail_retrieval, not v2_ensemble).

Status: EXPERIMENTAL — imported by sail_retrieval.py only, not in production path.
See KNOWLEDGE_BASE.md §3 for the architecture map.

Computes ViT-B/32 embeddings of first-page thumbnails for all docs in a split.
Run once per split, cache to .npy; retrieval uses cosine similarity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from rich.progress import track

from .data import load_split

_MODEL_NAME = "ViT-B-32"
_PRETRAINED = "openai"
_BATCH_SIZE = 16


def compute_embeddings(split: str, out: Path) -> dict[str, np.ndarray]:
    """Embed first page of every doc in split. Saves {docid: embedding} to out."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        _MODEL_NAME, pretrained=_PRETRAINED
    )
    model = model.eval().to(device)

    dataset = load_split(split)
    docids: list[str] = []
    vectors: list[np.ndarray] = []

    batch_imgs: list[Image.Image] = []
    batch_ids: list[str] = []

    def flush_batch() -> None:
        if not batch_imgs:
            return
        tensors = torch.stack([preprocess(img) for img in batch_imgs]).to(device)
        with torch.no_grad():
            embs = model.encode_image(tensors).cpu().float().numpy()
        for docid, emb in zip(batch_ids, embs, strict=False):
            docids.append(docid)
            vectors.append(emb / (np.linalg.norm(emb) + 1e-8))
        batch_imgs.clear()
        batch_ids.clear()

    for doc in track(dataset, description=f"Embedding {split}"):
        with doc:
            img = doc.page_image(0)
        batch_imgs.append(img)
        batch_ids.append(doc.docid)
        if len(batch_imgs) >= _BATCH_SIZE:
            flush_batch()
    flush_batch()

    out.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.stack(vectors)
    np.savez(out, docids=np.array(docids), embeddings=matrix)
    return dict(zip(docids, vectors, strict=False))


def load_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    """Load saved embeddings. Returns (docids, matrix) where matrix[i] = embedding for docids[i]."""
    data = np.load(path, allow_pickle=False)
    return list(data["docids"]), data["embeddings"]


def find_neighbors(
    query_docids: list[str],
    query_embeds: Path | tuple[list[str], np.ndarray],
    corpus_embeds: Path | tuple[list[str], np.ndarray],
    k: int = 3,
) -> dict[str, list[str]]:
    """For each query docid, return top-k nearest corpus docids by cosine similarity."""
    if isinstance(query_embeds, Path):
        query_embeds = load_embeddings(query_embeds)
    if isinstance(corpus_embeds, Path):
        corpus_embeds = load_embeddings(corpus_embeds)

    q_ids, q_matrix = query_embeds
    c_ids, c_matrix = corpus_embeds

    q_idx = {d: i for i, d in enumerate(q_ids)}
    sims = q_matrix @ c_matrix.T  # (n_query, n_corpus)

    result: dict[str, list[str]] = {}
    for docid in query_docids:
        if docid not in q_idx:
            result[docid] = []
            continue
        i = q_idx[docid]
        top_k = np.argsort(sims[i])[::-1][:k]
        result[docid] = [c_ids[j] for j in top_k]
    return result
