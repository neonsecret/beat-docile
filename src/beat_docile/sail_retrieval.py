"""[EXPERIMENTAL] SAIL (Structure-Aware ICL) retrieval for DocILE.

Selects training examples for ICL by combining:
  (a) visual layout similarity — Qwen3-VL-Embedding-2B cosine similarity
  (b) entity-level similarity — field-type presence vector cosine similarity

For train docs: entity vectors come from gold field annotations.
For val/test docs: entity vectors are estimated via OCR keyword patterns.

Public API
----------
select_few_shot(val_doc, k=3) -> list[FewShotExample]
    Drop-in companion to fewshot.load_few_shot_examples, returning a flat
    list of FewShotExample for any doc regardless of cluster_id.

Requires models/sail_index/sail_index.npz built by tools/build_sail_index.py.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DATA_ROOT
from .fewshot import FewShotExample

logger = logging.getLogger(__name__)

# ── Field type ordering (fixed 55-element index) ──────────────────────────────

_KILE_TYPES: list[str] = sorted(
    [
        "account_num",
        "amount_due",
        "amount_paid",
        "amount_total_gross",
        "amount_total_net",
        "amount_total_tax",
        "bank_num",
        "bic",
        "currency_code_amount_due",
        "customer_billing_address",
        "customer_billing_name",
        "customer_delivery_address",
        "customer_delivery_name",
        "customer_id",
        "customer_order_id",
        "customer_other_address",
        "customer_other_name",
        "customer_registration_id",
        "customer_tax_id",
        "date_due",
        "date_issue",
        "document_id",
        "iban",
        "order_id",
        "payment_reference",
        "payment_terms",
        "tax_detail_gross",
        "tax_detail_net",
        "tax_detail_rate",
        "tax_detail_tax",
        "vendor_address",
        "vendor_email",
        "vendor_name",
        "vendor_order_id",
        "vendor_registration_id",
        "vendor_tax_id",
    ]
)  # 36

_LIR_TYPES: list[str] = sorted(
    [
        "line_item_amount_gross",
        "line_item_amount_net",
        "line_item_code",
        "line_item_currency",
        "line_item_date",
        "line_item_description",
        "line_item_discount_amount",
        "line_item_discount_rate",
        "line_item_hts_number",
        "line_item_order_id",
        "line_item_person_name",
        "line_item_position",
        "line_item_quantity",
        "line_item_tax",
        "line_item_tax_rate",
        "line_item_unit_price_gross",
        "line_item_unit_price_net",
        "line_item_units_of_measure",
        "line_item_weight",
    ]
)  # 19

ALL_FIELD_TYPES: list[str] = _KILE_TYPES + _LIR_TYPES  # 55 total
FIELD_TYPE_IDX: dict[str, int] = {ft: i for i, ft in enumerate(ALL_FIELD_TYPES)}

# ── Keyword patterns for val doc entity estimation ────────────────────────────
# Each tuple: (field_type, patterns) — any pattern found in lowercase OCR text → bit = 1

_ENTITY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("account_num", ("account no", "account number", "konto", "compte no", "conto")),
    ("amount_due", ("amount due", "balance due", "total due", "pay now")),
    ("amount_paid", ("amount paid", "already paid", "bezahlt")),
    ("amount_total_gross", ("total", "brutto", "gross total", "ttc")),
    ("amount_total_net", ("net total", "netto", "subtotal", "sub-total")),
    ("amount_total_tax", ("vat amount", "tax amount", "total tax", "total vat")),
    ("bank_num", ("sort code", "blz", "routing", "bankleitzahl")),
    ("bic", ("bic", "swift")),
    ("currency_code_amount_due", ("€", "$", "£", "eur", "usd", "gbp")),
    ("customer_billing_address", ("bill to", "billing address", "invoice to")),
    ("customer_billing_name", ("bill to", "billing name", "invoiced to")),
    ("customer_delivery_address", ("ship to", "deliver to", "delivery address")),
    ("customer_delivery_name", ("deliver to", "consignee")),
    ("customer_id", ("customer id", "customer no", "client id", "client no")),
    ("customer_order_id", ("customer order", "po number", "purchase order", "your order")),
    ("customer_other_address", ("customer address",)),
    ("customer_other_name", ("customer name",)),
    ("customer_registration_id", ("customer reg", "buyer reg")),
    ("customer_tax_id", ("customer vat", "buyer vat", "customer tax id")),
    ("date_due", ("due date", "payment due", "pay by", "fälligkeitsdatum")),
    ("date_issue", ("invoice date", "issue date", "date of issue")),
    ("document_id", ("invoice no", "invoice number", "invoice #")),
    ("iban", ("iban",)),
    ("order_id", ("order no", "order number", "order id")),
    ("payment_reference", ("payment reference", "reference no", "remittance")),
    ("payment_terms", ("payment terms", "net 30", "net 60", "days net")),
    ("tax_detail_gross", ("gross",)),
    ("tax_detail_net", ("net",)),
    ("tax_detail_rate", ("%", "tax rate", "vat rate")),
    ("tax_detail_tax", ("tax",)),
    ("vendor_address", ("from:", "seller address")),
    ("vendor_email", ("@",)),
    ("vendor_name", ("invoice from", "seller:")),
    ("vendor_order_id", ("our order", "vendor order", "supplier order")),
    ("vendor_registration_id", ("reg no", "registration no", "kvk", "hrb", "ičo", "abn")),
    ("vendor_tax_id", ("vat no", "vat reg", "mwst-idnr", "uid-nr", "our vat")),
    ("line_item_amount_gross", ("gross",)),
    ("line_item_amount_net", ("net",)),
    ("line_item_code", ("code", "sku", "item no", "article no")),
    ("line_item_currency", ("currency",)),
    ("line_item_date", ("date",)),
    ("line_item_description", ("description", "desc", "product")),
    ("line_item_discount_amount", ("discount", "rabatt", "remise")),
    ("line_item_discount_rate", ("disc %", "discount %")),
    ("line_item_hts_number", ("hts", "hs code", "tariff")),
    ("line_item_order_id", ("order no", "line order")),
    ("line_item_person_name", ("name", "person")),
    ("line_item_position", ("pos.", "line no", "item #")),
    ("line_item_quantity", ("qty", "quantity", "menge", "pcs")),
    ("line_item_tax", ("tax", "vat")),
    ("line_item_tax_rate", ("tax %", "vat %")),
    ("line_item_unit_price_gross", ("unit price", "gross price", "price")),
    ("line_item_unit_price_net", ("net price", "unit net")),
    ("line_item_units_of_measure", ("unit", "pcs", "each", "per piece")),
    ("line_item_weight", ("weight", "kg", "lb")),
]

_DEFAULT_INDEX_PATH = (
    Path(__file__).parent.parent.parent / "models" / "sail_index" / "sail_index.npz"
)


# ── Entity vector helpers ─────────────────────────────────────────────────────


def entity_vec_from_gold(fields: list, li_fields: list) -> np.ndarray:
    """Binary presence vector (55,) from gold field annotations (train side)."""
    vec = np.zeros(len(ALL_FIELD_TYPES), dtype=np.float32)
    for f in fields:
        idx = FIELD_TYPE_IDX.get(f.fieldtype)
        if idx is not None:
            vec[idx] = 1.0
    for f in li_fields:
        idx = FIELD_TYPE_IDX.get(f.fieldtype)
        if idx is not None:
            vec[idx] = 1.0
    return vec


def entity_vec_from_ocr(ocr_text: str) -> np.ndarray:
    """Estimate presence vector (55,) from lowercase OCR text via keyword matching (val side)."""
    text = ocr_text.lower()
    vec = np.zeros(len(ALL_FIELD_TYPES), dtype=np.float32)
    for ft, patterns in _ENTITY_KEYWORDS:
        idx = FIELD_TYPE_IDX.get(ft)
        if idx is None:
            continue
        if any(pat in text for pat in patterns):
            vec[idx] = 1.0
    return vec


def _l2_normalize_rows(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.maximum(norms, 1e-8)


# ── Index dataclass ───────────────────────────────────────────────────────────


@dataclass
class SAILIndex:
    visual_embs: np.ndarray  # (N, 2048) float32, L2-normalised
    entity_vecs: np.ndarray  # (N, 55)   float32, L2-normalised
    docids: list[str]  # length N
    cluster_ids: np.ndarray  # (N,) int32
    gold_jsons: list[str]  # length N — compact JSON for FewShotExample
    words_layouts: list[str]  # length N — row-grouped word layout strings


def load_sail_index(path: Path) -> SAILIndex:
    """Load and L2-normalise SAIL index from NPZ."""
    data = np.load(path, allow_pickle=True)
    return SAILIndex(
        visual_embs=_l2_normalize_rows(data["visual_embs"].astype(np.float32)),
        entity_vecs=_l2_normalize_rows(data["entity_vecs"].astype(np.float32)),
        docids=[str(d) for d in data["docids"]],
        cluster_ids=data["cluster_ids"].astype(np.int32),
        gold_jsons=[str(s) for s in data["gold_jsons"]],
        words_layouts=[str(s) for s in data["words_layouts"]],
    )


# ── Retriever ─────────────────────────────────────────────────────────────────


def _image_to_b64(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


class SAILRetriever:
    """Combines visual + entity similarity for ICL training-example selection."""

    def __init__(
        self,
        index_path: Path = _DEFAULT_INDEX_PATH,
        device: str = "mps",
        alpha: float = 0.7,
    ) -> None:
        self._index_path = Path(index_path)
        self._device = device
        self._alpha = alpha
        self._index: SAILIndex | None = None
        self._model = None
        self._train_doc_lookup: dict[str, object] | None = None

    # ── Lazy loaders ──────────────────────────────────────────────────────────

    def _ensure_index(self) -> SAILIndex:
        if self._index is None:
            logger.info("Loading SAIL index from %s", self._index_path)
            self._index = load_sail_index(self._index_path)
            logger.info("SAIL index ready: %d train docs", len(self._index.docids))
        return self._index

    def _ensure_model(self):
        if self._model is None:
            from .cluster_infer import load_qwen3vl_model

            self._model, self._device = load_qwen3vl_model(self._device)
        return self._model

    def _ensure_train_docs(self) -> dict[str, object]:
        if self._train_doc_lookup is None:
            from docile.dataset import Dataset

            logger.info("Building train doc lookup for SAIL example loading")
            train_ds = Dataset("train", DATA_ROOT, load_annotations=False, load_ocr=False)
            self._train_doc_lookup = {doc.docid: doc for doc in train_ds}
            logger.info("Train doc lookup: %d docs", len(self._train_doc_lookup))
        return self._train_doc_lookup

    # ── FewShotExample construction ───────────────────────────────────────────

    def _load_fewshot_example(
        self,
        docid: str,
        cluster_id: int,
        gold_json: str,
        words_layout: str,
    ) -> FewShotExample | None:
        """Open train doc, render page 0 image, return FewShotExample."""
        doc_lookup = self._ensure_train_docs()
        doc = doc_lookup.get(docid)
        if doc is None:
            logger.warning("Train doc %s not found in lookup", docid)
            return None
        try:
            with doc:
                image = doc.page_image(0)
                image_b64 = _image_to_b64(image)
            return FewShotExample(
                docid=docid,
                cluster_id=cluster_id,
                image_b64=image_b64,
                words_layout=words_layout,
                gold_json=gold_json,
            )
        except Exception:
            logger.warning("Failed to render image for %s", docid, exc_info=True)
            return None

    # ── Core retrieval ────────────────────────────────────────────────────────

    def select_few_shot(self, val_doc, k: int = 3) -> list[FewShotExample]:
        """Select K training examples for val_doc via SAIL similarity.

        Combines visual (layout) + entity-level similarity.
        Works for any val doc, including NO_MATCH and UNANALYZABLE buckets.

        val_doc is opened internally; caller must not have it open already.
        """
        index = self._ensure_index()
        model = self._ensure_model()

        from .cluster_infer import embed_doc_qwen3vl

        # Embed val doc and estimate entity features from OCR
        with val_doc:
            visual_emb = embed_doc_qwen3vl(val_doc, model, self._device)  # (2048,) L2-normed
            try:
                page0_words = val_doc.ocr.get_all_words(page=0, snapped=False)
                ocr_text = " ".join(w.text for w in page0_words)
            except Exception:
                ocr_text = ""

        entity_vec = entity_vec_from_ocr(ocr_text)

        # L2-normalise entity query (visual_emb already normed by embed_doc_qwen3vl)
        ev_norm = np.linalg.norm(entity_vec)
        entity_vec_n = entity_vec / (ev_norm + 1e-8) if ev_norm > 1e-8 else entity_vec

        # Combined similarity score
        visual_sims = index.visual_embs @ visual_emb  # (N,)
        entity_sims = index.entity_vecs @ entity_vec_n  # (N,)
        combined = self._alpha * visual_sims + (1.0 - self._alpha) * entity_sims  # (N,)

        top_k_idx = np.argsort(combined)[::-1][:k]

        examples: list[FewShotExample] = []
        for idx in top_k_idx:
            ex = self._load_fewshot_example(
                docid=index.docids[idx],
                cluster_id=int(index.cluster_ids[idx]),
                gold_json=index.gold_jsons[idx],
                words_layout=index.words_layouts[idx],
            )
            if ex is not None:
                examples.append(ex)

        return examples


# ── Module-level singleton API ────────────────────────────────────────────────

_retriever: SAILRetriever | None = None


def get_retriever(
    index_path: Path = _DEFAULT_INDEX_PATH,
    device: str = "mps",
    alpha: float = 0.7,
) -> SAILRetriever:
    """Return the module-level SAILRetriever singleton (lazy-initialised)."""
    global _retriever
    if _retriever is None:
        _retriever = SAILRetriever(index_path=index_path, device=device, alpha=alpha)
    return _retriever


def select_few_shot(val_doc, k: int = 3) -> list[FewShotExample]:
    """Select K ICL training examples for val_doc via SAIL retrieval.

    Drop-in alternative to cluster-based few-shot selection; works for ALL docs
    including NO_MATCH and UNANALYZABLE buckets.

    Requires models/sail_index/sail_index.npz (build with tools/build_sail_index.py).
    """
    return get_retriever().select_few_shot(val_doc, k=k)
