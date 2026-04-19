"""Tests for cluster_infer module (Qwen3-VL-Embedding-2B backend)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from beat_docile.cluster_infer import (
    ClusterPrediction,
    _load_train_npz,
    _resolve_device,
    build_train_embeddings,
    infer_cluster,
    infer_clusters_batch,
    validate_val_accuracy,
)

_DIM = 2048


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_npz(tmp_path: Path, n: int = 10, dim: int = _DIM) -> Path:
    rng = np.random.default_rng(42)
    embs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs /= norms + 1e-8
    cids = np.arange(n, dtype=np.int32) % 3
    docids = np.array([f"train_{i:04d}" for i in range(n)], dtype=object)
    path = tmp_path / "train_embs.npz"
    np.savez(path, embeddings=embs, cluster_ids=cids, docids=docids)
    return path


def _make_mock_doc(docid: str, cluster_id: int) -> MagicMock:
    from PIL import Image
    doc = MagicMock()
    doc.docid = docid
    doc.annotation.cluster_id = cluster_id
    doc.page_image.return_value = Image.new("RGB", (224, 224))
    doc.__enter__ = MagicMock(return_value=doc)
    doc.__exit__ = MagicMock(return_value=False)
    return doc


def _make_mock_st_model(dim: int = _DIM) -> MagicMock:
    """Mock SentenceTransformer that returns a random L2-normalised embedding."""
    model = MagicMock()
    call_counter = [0]

    def fake_encode(inputs, **kwargs):
        rng = np.random.default_rng(call_counter[0])
        call_counter[0] += 1
        emb = rng.standard_normal((len(inputs), dim)).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / (norms + 1e-8)

    model.encode.side_effect = fake_encode
    return model


# ── Unit tests ────────────────────────────────────────────────────────────────

class TestResolveDevice:
    def test_cpu_always_resolves(self):
        assert _resolve_device("cpu") == "cpu"

    def test_invalid_device_falls_back_to_cpu(self):
        assert _resolve_device("tpu") == "cpu"


class TestLoadTrainNpz:
    def test_roundtrip(self, tmp_path: Path):
        npz_path = _make_npz(tmp_path, n=5)
        embs, cids, docids = _load_train_npz(npz_path)
        assert embs.shape == (5, _DIM)
        assert embs.dtype == np.float32
        assert cids.dtype == np.int32
        assert docids[0] == "train_0000"

    def test_embeddings_normalised(self, tmp_path: Path):
        npz_path = _make_npz(tmp_path, n=4)
        embs, _, _ = _load_train_npz(npz_path)
        np.testing.assert_allclose(np.linalg.norm(embs, axis=1), np.ones(4), atol=1e-5)


class TestClusterPredictionDataclass:
    def test_fields_present(self):
        p = ClusterPrediction("abc", 7, 0.92, "train_001", 0.08)
        assert p.docid == "abc"
        assert p.inferred_cluster_id == 7
        assert p.confidence == pytest.approx(0.92)
        assert p.nearest_distance == pytest.approx(0.08)


# ── Integration tests (mock SentenceTransformer) ──────────────────────────────

class TestBuildTrainEmbeddings:
    def test_saves_correct_shape(self, tmp_path: Path):
        mock_model = _make_mock_st_model()
        docs = [_make_mock_doc(f"doc_{i}", cluster_id=i % 3) for i in range(6)]

        with patch("beat_docile.cluster_infer.load_qwen3vl_model",
                   return_value=(mock_model, "cpu")):
            stats = build_train_embeddings(docs, tmp_path / "out.npz", device="cpu")

        assert stats["n_embedded"] == 6
        assert stats["n_failed"] == 0
        assert stats["embedding_dim"] == _DIM
        embs, _cids, docids = _load_train_npz(tmp_path / "out.npz")
        assert embs.shape == (6, _DIM)
        assert len(docids) == 6

    def test_skips_failed_docs(self, tmp_path: Path):
        mock_model = _make_mock_st_model()
        docs = [_make_mock_doc(f"doc_{i}", cluster_id=i) for i in range(4)]
        docs[2].__enter__.side_effect = RuntimeError("bad PDF")

        with patch("beat_docile.cluster_infer.load_qwen3vl_model",
                   return_value=(mock_model, "cpu")):
            stats = build_train_embeddings(docs, tmp_path / "out.npz", device="cpu")

        assert stats["n_failed"] == 1
        assert stats["n_embedded"] == 3

    def test_creates_parent_dirs(self, tmp_path: Path):
        mock_model = _make_mock_st_model()
        docs = [_make_mock_doc("doc_0", cluster_id=1)]
        nested = tmp_path / "a" / "b" / "embs.npz"

        with patch("beat_docile.cluster_infer.load_qwen3vl_model",
                   return_value=(mock_model, "cpu")):
            build_train_embeddings(docs, nested, device="cpu")

        assert nested.exists()


class TestInferCluster:
    def test_returns_cluster_prediction(self, tmp_path: Path):
        npz_path = _make_npz(tmp_path, n=9)
        mock_model = _make_mock_st_model()
        doc = _make_mock_doc("query_0", cluster_id=0)

        with patch("beat_docile.cluster_infer.load_qwen3vl_model",
                   return_value=(mock_model, "cpu")):
            pred = infer_cluster(doc, npz_path, mock_model, device="cpu", k=1)

        assert isinstance(pred, ClusterPrediction)
        assert pred.docid == "query_0"
        assert pred.inferred_cluster_id in {0, 1, 2}

    def test_k3_majority_vote(self, tmp_path: Path):
        npz_path = _make_npz(tmp_path, n=9)
        mock_model = _make_mock_st_model()
        doc = _make_mock_doc("query_k3", cluster_id=0)

        with patch("beat_docile.cluster_infer.load_qwen3vl_model",
                   return_value=(mock_model, "cpu")):
            pred = infer_cluster(doc, npz_path, mock_model, device="cpu", k=3)

        assert pred.inferred_cluster_id in {0, 1, 2}


class TestInferClustersBatch:
    def test_saves_json_mapping(self, tmp_path: Path):
        npz_path = _make_npz(tmp_path, n=6)
        mock_model = _make_mock_st_model()
        docs = [_make_mock_doc(f"val_{i}", cluster_id=i % 3) for i in range(4)]
        out_json = tmp_path / "clusters.json"

        with patch("beat_docile.cluster_infer.load_qwen3vl_model",
                   return_value=(mock_model, "cpu")):
            preds = infer_clusters_batch(docs, npz_path, out_json, device="cpu")

        assert out_json.exists()
        mapping = json.loads(out_json.read_text())
        assert set(mapping.keys()) == {f"val_{i}" for i in range(4)}
        assert all(isinstance(v, int) for v in mapping.values())
        assert len(preds) == 4

    def test_all_predictions_valid_cluster(self, tmp_path: Path):
        npz_path = _make_npz(tmp_path, n=6)
        mock_model = _make_mock_st_model()
        docs = [_make_mock_doc(f"q_{i}", cluster_id=i % 3) for i in range(5)]
        out_json = tmp_path / "out.json"

        with patch("beat_docile.cluster_infer.load_qwen3vl_model",
                   return_value=(mock_model, "cpu")):
            preds = infer_clusters_batch(docs, npz_path, out_json, device="cpu")

        for pred in preds.values():
            assert pred.inferred_cluster_id in {0, 1, 2}
            assert pred.nearest_train_docid.startswith("train_")


class TestValidateValAccuracy:
    def test_perfect_accuracy(self):
        preds = {
            "doc_0": ClusterPrediction("doc_0", 5, 0.95, "train_x", 0.05),
            "doc_1": ClusterPrediction("doc_1", 3, 0.90, "train_y", 0.10),
        }
        docs = [_make_mock_doc("doc_0", 5), _make_mock_doc("doc_1", 3)]
        result = validate_val_accuracy(preds, docs)
        assert result["top1_accuracy"] == pytest.approx(1.0)
        assert result["n_correct_top1"] == 2

    def test_zero_accuracy(self):
        preds = {"doc_0": ClusterPrediction("doc_0", 99, 0.5, "t", 0.5)}
        result = validate_val_accuracy(preds, [_make_mock_doc("doc_0", 42)])
        assert result["top1_accuracy"] == pytest.approx(0.0)

    def test_skips_docs_not_in_preds(self):
        preds = {"doc_0": ClusterPrediction("doc_0", 5, 0.95, "t", 0.05)}
        docs = [_make_mock_doc("doc_0", 5), _make_mock_doc("doc_1", 3)]
        result = validate_val_accuracy(preds, docs)
        assert result["n_docs"] == 1

    def test_empty_preds_returns_error(self):
        result = validate_val_accuracy({}, [_make_mock_doc("doc_0", 1)])
        assert "error" in result

    def test_confidence_stats_present(self):
        preds = {
            f"doc_{i}": ClusterPrediction(f"doc_{i}", i, float(i) * 0.1, "t", 0.9)
            for i in range(1, 6)
        }
        docs = [_make_mock_doc(f"doc_{i}", i) for i in range(1, 6)]
        result = validate_val_accuracy(preds, docs)
        assert "confidence_mean" in result
        assert "high_confidence_frac" in result
