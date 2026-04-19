"""[RESEARCH-BURIED] Per-field sklearn MLP classifiers for DocILE field type scoring.

Status: RESEARCH-BURIED — as a reranker over v2_ensemble, buried at 250-doc
(-2.5pp KILE) due to rank disturbance. See KNOWLEDGE_BASE.md §6.6 for details.

Side finding: LIR F1 improved +1.6pp at threshold=0.3 even when KILE collapsed
(see §5.1, §8.9). Architecture: sklearn Pipeline (StandardScaler + MLPClassifier)
per field type. Training: gold annotations as positives, random OCR spans as
negatives.
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from joblib import dump, load
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import WordBox
from .extract import _KILE_TYPES, _LIR_TYPES

_LOG = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_ALL_FIELDTYPES: list[str] = sorted(_KILE_TYPES | _LIR_TYPES)

_LABEL_VOCAB: list[str] = [
    "invoice",
    "bill",
    "total",
    "amount",
    "due",
    "date",
    "address",
    "name",
    "number",
    "no",
    "code",
    "tax",
    "vat",
    "net",
    "gross",
    "bank",
    "account",
    "iban",
    "bic",
    "swift",
    "customer",
    "vendor",
    "buyer",
    "order",
    "reference",
    "payment",
    "terms",
    "billing",
    "delivery",
    "email",
    "registration",
    "currency",
    "quantity",
    "price",
    "unit",
    "discount",
    "rate",
    "description",
    "subtotal",
    "balance",
]  # 40 items — fixed vocabulary for one-hot encoding

_LABEL_VOCAB_INDEX: dict[str, int] = {v: i for i, v in enumerate(_LABEL_VOCAB)}
_LABEL_VOCAB_SET: frozenset[str] = frozenset(_LABEL_VOCAB)

# Feature vector layout (total 71 dimensions):
#   [0:15]  scalar features
#   [15:31] neighbor text features (4 neighbors x 4 stats)
#   [31:71] label vocabulary one-hot (40 items)
_FEATURE_DIM: int = 15 + 4 * 4 + len(_LABEL_VOCAB)  # 71

# Regex patterns for feature matching
_IBAN_PATTERN: re.Pattern[str] = re.compile(r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}\b")
_DATE_PATTERN: re.Pattern[str] = re.compile(
    r"\b\d{1,4}[-./]\d{1,2}[-./]\d{1,4}\b"
    r"|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b",
    re.IGNORECASE,
)
_AMOUNT_PATTERN: re.Pattern[str] = re.compile(
    r"[$€£¥₹₽]?\s*\d[\d,.\s]*\d|\d[\d,.\s]*\d\s*[$€£¥₹₽%]"
)
_CURRENCY_SYMBOL_PATTERN: re.Pattern[str] = re.compile(r"[$€£¥₹₽₩₪₦₴₺₱฿]")

_MIN_POSITIVE_EXAMPLES: int = 10
_ROW_GAP_FRAC: float = 0.015  # words within this vertical distance are on the same row


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class SpanFeatures:
    """Features extracted from a candidate span for classification."""

    text: str
    char_count: int
    word_count: int
    has_digits: bool
    digit_ratio: float
    has_letters: bool
    has_currency_symbol: bool
    matches_iban_pattern: bool
    matches_date_pattern: bool
    matches_amount_pattern: bool
    bbox_left_frac: float  # span bbox left / page width (already normalized → equals bbox.left)
    bbox_top_frac: float  # span bbox top / page height
    bbox_width_frac: float
    bbox_height_frac: float
    left_neighbor_text: str  # word immediately to the left, or ""
    right_neighbor_text: str
    above_neighbor_text: str
    below_neighbor_text: str
    nearest_label_phrase: str  # nearest vocab word from _LABEL_VOCAB, or ""
    nearest_label_distance_frac: float  # Euclidean distance (centre-to-centre) in page fracs


@dataclass
class DocRecord:
    """Lightweight document record for classifier training.

    Loaded from OCR + annotation JSON files — no PDF or image needed.
    """

    docid: str
    pages: list[list[WordBox]]  # pages[page_idx] = list of WordBox (id = position in list)
    kile_fields: list[dict]  # {"bbox": [l,t,r,b], "fieldtype": str, "page": int}
    lir_fields: list[dict]  # same + "line_item_id": int


# ── Module-level classifier cache ─────────────────────────────────────────────

_CLASSIFIER_CACHE: dict[tuple[str, Path], Any] = {}


# ── OCR / annotation loading ──────────────────────────────────────────────────


def _parse_ocr_words(ocr_data: dict) -> list[list[WordBox]]:
    """Parse raw OCR JSON into per-page WordBox lists."""
    pages_out: list[list[WordBox]] = []
    for page_data in ocr_data.get("pages", []):
        words: list[WordBox] = []
        word_idx = 0
        for block in page_data.get("blocks", []):
            for line in block.get("lines", []):
                for word in line.get("words", []):
                    geo = word.get("snapped_geometry") or word.get("geometry")
                    if geo is None:
                        continue
                    (left, top), (right, bottom) = geo[0], geo[1]
                    page_idx = page_data.get("page_idx", 0)
                    words.append(
                        WordBox(
                            id=word_idx,
                            text=word.get("value", ""),
                            bbox=(float(left), float(top), float(right), float(bottom)),
                            page=page_idx,
                        )
                    )
                    word_idx += 1
        pages_out.append(words)
    return pages_out


def _load_doc_record(docid: str, data_dir: Path) -> DocRecord | None:
    """Load a DocRecord from OCR + annotation JSON files.

    Returns None if either file is missing.
    """
    ocr_path = data_dir / "ocr" / f"{docid}.json"
    ann_path = data_dir / "annotations" / f"{docid}.json"
    if not ocr_path.exists() or not ann_path.exists():
        return None
    try:
        with ocr_path.open() as f:
            ocr_data = json.load(f)
        with ann_path.open() as f:
            ann_data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        _LOG.warning("Failed to load doc %s: %s", docid, exc)
        return None

    pages = _parse_ocr_words(ocr_data)
    kile_fields = [
        {"bbox": item["bbox"], "fieldtype": item["fieldtype"], "page": item["page"]}
        for item in ann_data.get("field_extractions", [])
        if "bbox" in item and "fieldtype" in item and "page" in item
    ]
    lir_fields = [
        {
            "bbox": item["bbox"],
            "fieldtype": item["fieldtype"],
            "page": item["page"],
            "line_item_id": item.get("line_item_id", 0),
        }
        for item in ann_data.get("line_item_extractions", [])
        if "bbox" in item and "fieldtype" in item and "page" in item
    ]
    return DocRecord(docid=docid, pages=pages, kile_fields=kile_fields, lir_fields=lir_fields)


def load_doc_records(docids: list[str], data_dir: Path) -> list[DocRecord]:
    """Load DocRecord objects from OCR + annotation JSON files.

    Silently skips docids where either file is missing.
    """
    records: list[DocRecord] = []
    for docid in docids:
        record = _load_doc_record(docid, data_dir)
        if record is not None:
            records.append(record)
    _LOG.info("Loaded %d / %d doc records from %s", len(records), len(docids), data_dir)
    return records


# ── Feature extraction helpers ────────────────────────────────────────────────


def _words_in_annotation(
    words: list[WordBox], ann_bbox: tuple[float, float, float, float], margin: float = 0.005
) -> list[int]:
    """Return word ids whose centre falls within ann_bbox (with optional margin)."""
    left, top, right, bottom = ann_bbox
    result: list[int] = []
    for w in words:
        wl, wt, wr, wb = w.bbox
        cx = (wl + wr) / 2.0
        cy = (wt + wb) / 2.0
        if (left - margin) <= cx <= (right + margin) and (top - margin) <= cy <= (bottom + margin):
            result.append(w.id)
    return result


def _span_bbox(span_word_ids: list[int], words: list[WordBox]) -> tuple[float, float, float, float]:
    """Compute the bounding box enclosing all span words."""
    id_map = {w.id: w for w in words}
    bboxes = [id_map[wid].bbox for wid in span_word_ids if wid in id_map]
    if not bboxes:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    )


def _span_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _neighbour_stats(text: str) -> tuple[float, float, float, float]:
    """Compute 4 stats for a neighbour text string.

    Returns: (length_norm, digit_ratio, has_currency, is_label_keyword).
    """
    if not text:
        return (0.0, 0.0, 0.0, 0.0)
    length_norm = min(len(text) / 30.0, 1.0)
    digits = sum(1 for c in text if c.isdigit())
    digit_ratio = digits / max(len(text), 1)
    has_currency = 1.0 if _CURRENCY_SYMBOL_PATTERN.search(text) else 0.0
    is_keyword = 1.0 if text.strip().lower() in _LABEL_VOCAB_SET else 0.0
    return (length_norm, digit_ratio, has_currency, is_keyword)


def _find_neighbours(
    span_bbox_val: tuple[float, float, float, float],
    words: list[WordBox],
    span_ids: set[int],
) -> tuple[str, str, str, str]:
    """Find the nearest word in each cardinal direction from the span.

    Returns: (left_text, right_text, above_text, below_text).
    """
    sl, st, sr, sb = span_bbox_val
    s_mid_y = (st + sb) / 2.0

    best_left: tuple[float, str] = (float("inf"), "")
    best_right: tuple[float, str] = (float("inf"), "")
    best_above: tuple[float, str] = (float("inf"), "")
    best_below: tuple[float, str] = (float("inf"), "")

    for w in words:
        if w.id in span_ids:
            continue
        wl, wt, wr, wb = w.bbox
        w_mid_x = (wl + wr) / 2.0
        w_mid_y = (wt + wb) / 2.0

        on_same_row = abs(w_mid_y - s_mid_y) <= _ROW_GAP_FRAC * 3

        if on_same_row:
            if w_mid_x < sl:
                dist = sl - wr
                if dist < best_left[0]:
                    best_left = (dist, w.text)
            elif w_mid_x > sr:
                dist = wl - sr
                if dist < best_right[0]:
                    best_right = (dist, w.text)
        else:
            if w_mid_y < st:
                dist = st - wb
                if dist < best_above[0]:
                    best_above = (dist, w.text)
            elif w_mid_y > sb:
                dist = wt - sb
                if dist < best_below[0]:
                    best_below = (dist, w.text)

    return (best_left[1], best_right[1], best_above[1], best_below[1])


def _find_nearest_label(
    span_center: tuple[float, float],
    words: list[WordBox],
    span_ids: set[int],
) -> tuple[str, float]:
    """Find the nearest word in _LABEL_VOCAB_SET.

    Returns: (phrase, distance_frac) — distance is Euclidean between centres.
    """
    best_phrase = ""
    best_dist = float("inf")
    scx, scy = span_center

    for w in words:
        if w.id in span_ids:
            continue
        if w.text.strip().lower() not in _LABEL_VOCAB_SET:
            continue
        wl, wt, wr, wb = w.bbox
        wcx = (wl + wr) / 2.0
        wcy = (wt + wb) / 2.0
        dist = ((scx - wcx) ** 2 + (scy - wcy) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_phrase = w.text.strip().lower()

    return (best_phrase, min(best_dist, 2.0) if best_dist < float("inf") else 2.0)


# ── Public feature API ────────────────────────────────────────────────────────


def extract_features(
    span_word_ids: list[int],
    words: list[WordBox],
    page_width: float,
    page_height: float,
) -> SpanFeatures:
    """Extract a feature vector for one candidate span.

    Pure function — no side effects.
    page_width and page_height are the page dimensions; since WordBox bboxes are
    already normalised to [0, 1], these are typically 1.0. Kept as parameters for
    forward compatibility with callers that supply pixel-space dimensions.
    """
    if not span_word_ids:
        return SpanFeatures(
            text="",
            char_count=0,
            word_count=0,
            has_digits=False,
            digit_ratio=0.0,
            has_letters=False,
            has_currency_symbol=False,
            matches_iban_pattern=False,
            matches_date_pattern=False,
            matches_amount_pattern=False,
            bbox_left_frac=0.0,
            bbox_top_frac=0.0,
            bbox_width_frac=0.0,
            bbox_height_frac=0.0,
            left_neighbor_text="",
            right_neighbor_text="",
            above_neighbor_text="",
            below_neighbor_text="",
            nearest_label_phrase="",
            nearest_label_distance_frac=2.0,
        )

    id_map = {w.id: w for w in words}
    span_words = [id_map[wid] for wid in span_word_ids if wid in id_map]
    if not span_words:
        return extract_features([], words, page_width, page_height)

    text = " ".join(w.text for w in span_words)
    char_count = len(text)
    word_count = len(span_words)

    digits = sum(1 for c in text if c.isdigit())
    letters = sum(1 for c in text if c.isalpha())
    has_digits = digits > 0
    digit_ratio = digits / max(char_count, 1)
    has_letters = letters > 0
    has_currency_symbol = bool(_CURRENCY_SYMBOL_PATTERN.search(text))

    text_upper = text.upper().replace(" ", "")
    matches_iban_pattern = bool(_IBAN_PATTERN.search(text_upper))
    matches_date_pattern = bool(_DATE_PATTERN.search(text))
    matches_amount_pattern = bool(_AMOUNT_PATTERN.search(text))

    sbbox = _span_bbox(span_word_ids, words)
    sl, st, sr, sb = sbbox

    if page_width > 0:
        bbox_left_frac = sl / page_width
        bbox_width_frac = (sr - sl) / page_width
    else:
        bbox_left_frac = sl
        bbox_width_frac = sr - sl

    if page_height > 0:
        bbox_top_frac = st / page_height
        bbox_height_frac = (sb - st) / page_height
    else:
        bbox_top_frac = st
        bbox_height_frac = sb - st

    span_ids = set(span_word_ids)
    left_text, right_text, above_text, below_text = _find_neighbours(sbbox, words, span_ids)

    scenter = _span_center(sbbox)
    nearest_label, label_dist = _find_nearest_label(scenter, words, span_ids)

    return SpanFeatures(
        text=text,
        char_count=char_count,
        word_count=word_count,
        has_digits=has_digits,
        digit_ratio=digit_ratio,
        has_letters=has_letters,
        has_currency_symbol=has_currency_symbol,
        matches_iban_pattern=matches_iban_pattern,
        matches_date_pattern=matches_date_pattern,
        matches_amount_pattern=matches_amount_pattern,
        bbox_left_frac=bbox_left_frac,
        bbox_top_frac=bbox_top_frac,
        bbox_width_frac=bbox_width_frac,
        bbox_height_frac=bbox_height_frac,
        left_neighbor_text=left_text,
        right_neighbor_text=right_text,
        above_neighbor_text=above_text,
        below_neighbor_text=below_text,
        nearest_label_phrase=nearest_label,
        nearest_label_distance_frac=label_dist,
    )


def featurize_for_sklearn(features: SpanFeatures) -> np.ndarray:
    """Convert SpanFeatures → dense float vector of shape (_FEATURE_DIM,).

    Encoding layout (71 dimensions total):
      Indices  0-14  — scalar features (see inline comments)
      Indices 15-30  — neighbour text stats: 4 neighbours x 4 stats each
                       order: left, right, above, below;
                       stats per neighbour: length_norm, digit_ratio, has_currency, is_label_keyword
      Indices 31-70  — label vocabulary one-hot (40 items, order = _LABEL_VOCAB)
    """
    vec = np.zeros(_FEATURE_DIM, dtype=np.float32)

    # Scalar features [0:15]
    vec[0] = min(features.char_count / 100.0, 1.0)
    vec[1] = min(features.word_count / 10.0, 1.0)
    vec[2] = float(features.has_digits)
    vec[3] = features.digit_ratio
    vec[4] = float(features.has_letters)
    vec[5] = float(features.has_currency_symbol)
    vec[6] = float(features.matches_iban_pattern)
    vec[7] = float(features.matches_date_pattern)
    vec[8] = float(features.matches_amount_pattern)
    vec[9] = features.bbox_left_frac
    vec[10] = features.bbox_top_frac
    vec[11] = features.bbox_width_frac
    vec[12] = features.bbox_height_frac
    vec[13] = min(features.nearest_label_distance_frac / 2.0, 1.0)
    # Page region: top third (header area, common for vendor name, document_id, etc.)
    vec[14] = 1.0 if features.bbox_top_frac < 0.33 else 0.0

    # Neighbour text features [15:31]
    neighbours = [
        features.left_neighbor_text,
        features.right_neighbor_text,
        features.above_neighbor_text,
        features.below_neighbor_text,
    ]
    for i, nb_text in enumerate(neighbours):
        stats = _neighbour_stats(nb_text)
        vec[15 + i * 4 : 15 + i * 4 + 4] = stats

    # Label vocabulary one-hot [31:71]
    phrase = features.nearest_label_phrase.strip().lower()
    if phrase in _LABEL_VOCAB_INDEX:
        vec[31 + _LABEL_VOCAB_INDEX[phrase]] = 1.0

    return vec


# ── Training data construction ────────────────────────────────────────────────


def _sample_negative_spans(
    words: list[WordBox],
    positive_ids: set[int],
    n: int,
    rng: random.Random,
    max_span_words: int = 4,
) -> list[list[int]]:
    """Sample n non-overlapping random spans from words, avoiding positive_ids.

    Spans are contiguous runs of 1-max_span_words words in reading order.
    """
    if len(words) < 2:
        return []

    sorted_words = sorted(words, key=lambda w: (w.bbox[1], w.bbox[0]))
    word_ids = [w.id for w in sorted_words]
    n_words = len(word_ids)

    negatives: list[list[int]] = []
    attempts = 0
    max_attempts = n * 20

    while len(negatives) < n and attempts < max_attempts:
        attempts += 1
        span_len = rng.randint(1, min(max_span_words, n_words))
        start = rng.randint(0, n_words - span_len)
        span = word_ids[start : start + span_len]
        if not any(wid in positive_ids for wid in span):
            negatives.append(span)

    return negatives


def build_training_set(
    fieldtype: str,
    train_docs: list[DocRecord],
    n_negatives_per_doc: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Build feature matrix X and label vector y for a single fieldtype.

    Positives: each gold annotation of this fieldtype → SpanFeatures + label 1.
    Negatives: random non-overlapping spans from the same doc page → label 0.

    Returns: (X, y) where X has shape (n_samples, _FEATURE_DIM) and y in {0, 1}.
    """
    rng = random.Random(42)
    x_rows: list[np.ndarray] = []
    y_vals: list[int] = []

    all_fields: list[dict]
    for doc in train_docs:
        all_fields = doc.kile_fields + doc.lir_fields
        doc_fields = [f for f in all_fields if f["fieldtype"] == fieldtype]

        for page_idx, page_words in enumerate(doc.pages):
            page_fields = [f for f in doc_fields if f["page"] == page_idx]
            if not page_fields and not page_words:
                continue

            positive_ids: set[int] = set()
            for ann in page_fields:
                bbox: tuple[float, float, float, float] = tuple(ann["bbox"])  # type: ignore[assignment]
                span_ids = _words_in_annotation(page_words, bbox)
                if not span_ids:
                    continue
                positive_ids.update(span_ids)
                feats = extract_features(span_ids, page_words, 1.0, 1.0)
                x_rows.append(featurize_for_sklearn(feats))
                y_vals.append(1)

            if not page_words:
                continue
            neg_spans = _sample_negative_spans(page_words, positive_ids, n_negatives_per_doc, rng)
            for span_ids_neg in neg_spans:
                feats = extract_features(span_ids_neg, page_words, 1.0, 1.0)
                x_rows.append(featurize_for_sklearn(feats))
                y_vals.append(0)

    if not x_rows:
        return np.zeros((0, _FEATURE_DIM), dtype=np.float32), np.zeros(0, dtype=np.int32)

    x_mat = np.vstack(x_rows).astype(np.float32)
    y = np.array(y_vals, dtype=np.int32)
    return x_mat, y


# ── Training ──────────────────────────────────────────────────────────────────


def _build_pipeline() -> Pipeline:
    """Build a fresh sklearn Pipeline: StandardScaler + small MLP."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    max_iter=500,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=20,
                    verbose=False,
                ),
            ),
        ]
    )


def train_classifier(
    fieldtype: str,
    train_docs: list[DocRecord],
    model_dir: Path,
) -> dict:
    """Train a binary classifier for fieldtype and save to model_dir.

    Returns a metrics dict with keys:
      fieldtype, n_pos, n_neg, val_precision, val_recall, val_f1, val_auc.
    Returns None values for metric keys if fewer than _MIN_POSITIVE_EXAMPLES positives exist.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    x_mat, y = build_training_set(fieldtype, train_docs)

    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())

    metrics: dict = {
        "fieldtype": fieldtype,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "val_precision": None,
        "val_recall": None,
        "val_f1": None,
        "val_auc": None,
    }

    if n_pos < _MIN_POSITIVE_EXAMPLES:
        _LOG.info(
            "Skipping %s: only %d positive examples (min %d)",
            fieldtype,
            n_pos,
            _MIN_POSITIVE_EXAMPLES,
        )
        return metrics

    try:
        x_train, x_val, y_train, y_val = train_test_split(
            x_mat, y, test_size=0.2, stratify=y, random_state=42
        )
    except ValueError:
        x_train, x_val, y_train, y_val = train_test_split(x_mat, y, test_size=0.2, random_state=42)

    pipeline = _build_pipeline()
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_val)
    y_prob = pipeline.predict_proba(x_val)[:, 1]

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, average="binary", zero_division=0
    )

    try:
        auc = float(roc_auc_score(y_val, y_prob))
    except ValueError:
        auc = 0.5

    model_path = model_dir / f"{fieldtype}.joblib"
    dump(pipeline, model_path)
    _LOG.info(
        "Trained %s: n_pos=%d n_neg=%d val_f1=%.3f val_auc=%.3f → %s",
        fieldtype,
        n_pos,
        n_neg,
        f1,
        auc,
        model_path,
    )

    metrics.update(
        val_precision=float(prec), val_recall=float(rec), val_f1=float(f1), val_auc=float(auc)
    )
    return metrics


# ── Inference ─────────────────────────────────────────────────────────────────


def load_classifier(fieldtype: str, model_dir: Path) -> Any | None:
    """Load classifier for fieldtype from model_dir, with module-level caching.

    Returns None if the model file does not exist.
    """
    cache_key = (fieldtype, model_dir)
    if cache_key in _CLASSIFIER_CACHE:
        return _CLASSIFIER_CACHE[cache_key]

    model_path = model_dir / f"{fieldtype}.joblib"
    if not model_path.exists():
        return None

    pipeline = load(model_path)
    _CLASSIFIER_CACHE[cache_key] = pipeline
    return pipeline


def classifier_score(
    fieldtype: str,
    span_word_ids: list[int],
    words: list[WordBox],
    page_width: float,
    page_height: float,
    model_dir: Path,
) -> float:
    """Return p(this span is an instance of fieldtype) in [0, 1].

    Falls back to 0.5 if no model is trained for fieldtype.
    Target inference time: < 1 ms (pure sklearn, no API calls).
    """
    pipeline = load_classifier(fieldtype, model_dir)
    if pipeline is None:
        return 0.5

    feats = extract_features(span_word_ids, words, page_width, page_height)
    vec = featurize_for_sklearn(feats).reshape(1, -1)
    prob: float = float(pipeline.predict_proba(vec)[0, 1])
    return prob


# ── Batch training ────────────────────────────────────────────────────────────


def train_all_fields(
    train_docs: list[DocRecord],
    model_dir: Path,
    fieldtypes: list[str] | None = None,
) -> dict[str, dict]:
    """Train binary classifiers for all fieldtypes and save to model_dir.

    Skips fieldtypes with fewer than _MIN_POSITIVE_EXAMPLES positives.

    Returns: {fieldtype: metrics_dict} — metrics_dict is None for skipped types.
    """
    if fieldtypes is None:
        fieldtypes = _ALL_FIELDTYPES

    results: dict[str, dict] = {}
    for ft in fieldtypes:
        _LOG.info("Training classifier for: %s", ft)
        metrics = train_classifier(ft, train_docs, model_dir)
        results[ft] = metrics

    n_trained = sum(1 for m in results.values() if m.get("val_f1") is not None)
    n_skipped = len(results) - n_trained
    _LOG.info(
        "train_all_fields complete: %d trained, %d skipped (low positives)", n_trained, n_skipped
    )
    return results
