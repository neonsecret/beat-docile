"""Tests for oracle_extract.py — deterministic KILE field extraction."""

from __future__ import annotations

from beat_docile.data import WordBox
from beat_docile.oracle_extract import (
    SUPPORTED_FIELDTYPES,
    OracleMatch,
    oracle_extract_doc,
    oracle_extract_field,
)

# ── Test fixture helpers ──────────────────────────────────────────────────────

_ROW_Y = 0.1  # baseline y for all single-row word helpers


def words_row(texts: list[str], start_id: int = 0, y: float = _ROW_Y) -> list[WordBox]:
    """Create a horizontal row of WordBox objects with sequential IDs."""
    ws: list[WordBox] = []
    x = 0.05
    for i, text in enumerate(texts):
        w = len(text) * 0.012
        ws.append(WordBox(id=start_id + i, text=text, bbox=(x, y, x + w, y + 0.02), page=0))
        x += w + 0.01
    return ws


def single_word(text: str, wid: int = 0, y: float = _ROW_Y) -> WordBox:
    return WordBox(id=wid, text=text, bbox=(0.1, y, 0.3, y + 0.02), page=0)


# ── Known-valid test values ───────────────────────────────────────────────────

# Wikipedia example IBANs (mod-97 verified)
VALID_IBAN_UK = "GB29NWBK60161331926819"
VALID_IBAN_DE = "DE89370400440532013000"

# Invalid: wrong check digits
INVALID_IBAN = "GB00NWBK60161331926819"

VALID_BIC_8 = "DEUTDEDB"
VALID_BIC_11 = "DEUTDEDBXXX"

# ── IBAN tests ────────────────────────────────────────────────────────────────


def test_iban_single_word_valid():
    """Valid IBAN as single OCR token → 1 match, score 1.0, correct word_id."""
    words = [single_word(VALID_IBAN_UK, wid=0)]
    results = oracle_extract_field("iban", words, 0)
    assert len(results) == 1
    m = results[0]
    assert m.fieldtype == "iban"
    assert m.word_ids == [0]
    assert m.score == 1.0


def test_iban_split_across_words():
    """IBAN split into space-separated groups → joined and matched."""
    parts = ["GB29", "NWBK", "6016", "1331", "9268", "19"]
    words = words_row(parts)
    results = oracle_extract_field("iban", words, 0)
    assert len(results) == 1
    m = results[0]
    assert m.word_ids == list(range(6))
    assert m.score == 1.0


def test_iban_invalid_checksum_returns_empty():
    """IBAN with wrong check digits (mod-97 fails) → 0 matches."""
    words = [single_word(INVALID_IBAN)]
    results = oracle_extract_field("iban", words, 0)
    assert results == []


def test_iban_german_valid():
    """German IBAN passes mod-97."""
    words = [single_word(VALID_IBAN_DE)]
    results = oracle_extract_field("iban", words, 0)
    assert len(results) == 1
    assert results[0].score == 1.0


def test_iban_with_label_still_score_1():
    """Even with label context, IBAN is always 1.0 (checksum-verified)."""
    label_word = WordBox(id=0, text="IBAN:", bbox=(0.05, _ROW_Y, 0.15, _ROW_Y + 0.02), page=0)
    iban_word = WordBox(id=1, text=VALID_IBAN_UK, bbox=(0.16, _ROW_Y, 0.60, _ROW_Y + 0.02), page=0)
    results = oracle_extract_field("iban", [label_word, iban_word], 0)
    assert len(results) == 1
    assert results[0].score == 1.0
    assert results[0].word_ids == [1]


# ── BIC tests ─────────────────────────────────────────────────────────────────


def test_bic_8char_found():
    """8-character BIC matched and returned."""
    words = words_row(["BIC:", VALID_BIC_8])
    results = oracle_extract_field("bic", words, 0)
    bics = [m for m in results if m.text.replace(" ", "") == VALID_BIC_8]
    assert len(bics) == 1
    assert bics[0].score == 1.0  # label "BIC:" is nearby


def test_bic_11char_found():
    """11-character BIC with branch code matched."""
    words = words_row(["SWIFT:", VALID_BIC_11])
    results = oracle_extract_field("bic", words, 0)
    bics = [m for m in results if VALID_BIC_11 in m.text.replace(" ", "")]
    assert len(bics) == 1
    assert bics[0].score == 1.0  # "SWIFT:" triggers label


def test_bic_6char_not_matched():
    """6-character string does NOT match BIC (must be 8 or 11)."""
    words = words_row(["DEUTDE"])  # 6 chars — invalid
    results = oracle_extract_field("bic", words, 0)
    assert all(m.text.replace(" ", "") != "DEUTDE" for m in results)


def test_bic_no_label_score_07():
    """BIC without nearby label phrase → score 0.7."""
    # Surround with irrelevant words
    words = words_row(["Amount:", "1234.56", VALID_BIC_8, "EUR"])
    results = oracle_extract_field("bic", words, 0)
    bics = [m for m in results if m.text.replace(" ", "") == VALID_BIC_8]
    assert len(bics) == 1
    assert bics[0].score == 0.7


# ── document_id tests ─────────────────────────────────────────────────────────


def test_document_id_with_label_score_1():
    """Invoice number near 'Invoice No:' label → score 1.0."""
    words = words_row(["Invoice", "No:", "INV-2024-001"])
    results = oracle_extract_field("document_id", words, 0)
    ids = [m for m in results if "INV" in m.text]
    assert len(ids) >= 1
    assert ids[0].score == 1.0


def test_document_id_no_label_score_07():
    """Same invoice number but no label nearby → score 0.7."""
    # Only the invoice number, surrounded by unrelated content
    words = words_row(["Total:", "100.00", "INV-2024-001", "EUR", "due"])
    results = oracle_extract_field("document_id", words, 0)
    ids = [m for m in results if "INV" in m.text]
    assert len(ids) >= 1
    assert ids[0].score == 0.7


def test_document_id_pure_alpha_filtered():
    """Pure alphabetic strings are not matched as document IDs."""
    words = words_row(["Invoice", "No:", "TOTAL", "Amount"])
    results = oracle_extract_field("document_id", words, 0)
    # "TOTAL" and "Amount" are pure alpha (or short) → should not appear
    pure_alpha = [m for m in results if not any(c.isdigit() for c in m.text)]
    assert pure_alpha == []


# ── oracle_extract_doc dedup tests ───────────────────────────────────────────


def test_oracle_extract_doc_dedup_keeps_highest_score():
    """Same IBAN on two pages → only highest-score match returned."""
    label = WordBox(id=0, text="IBAN:", bbox=(0.05, 0.1, 0.15, 0.12), page=0)
    iban_w = WordBox(id=1, text=VALID_IBAN_UK, bbox=(0.16, 0.1, 0.60, 0.12), page=0)
    page0 = [label, iban_w]  # has label → score 1.0

    iban_w2 = WordBox(id=0, text=VALID_IBAN_UK, bbox=(0.16, 0.1, 0.60, 0.12), page=1)
    page1 = [iban_w2]  # no label → score 1.0 (IBAN is always 1.0)

    results = oracle_extract_doc({0: page0, 1: page1}, fieldtypes={"iban"})
    iban_results = [m for m in results if m.fieldtype == "iban"]
    assert len(iban_results) == 1  # deduplicated


def test_oracle_extract_doc_multi_page():
    """oracle_extract_doc finds fields across multiple pages."""
    bic_w = WordBox(id=0, text=VALID_BIC_8, bbox=(0.1, 0.1, 0.3, 0.12), page=0)
    iban_w = WordBox(id=0, text=VALID_IBAN_DE, bbox=(0.1, 0.1, 0.5, 0.12), page=1)

    results = oracle_extract_doc({0: [bic_w], 1: [iban_w]}, fieldtypes={"bic", "iban"})
    fieldtypes_found = {m.fieldtype for m in results}
    assert "bic" in fieldtypes_found
    assert "iban" in fieldtypes_found


def test_oracle_extract_doc_none_fieldtypes_runs_all():
    """When fieldtypes=None, all SUPPORTED_FIELDTYPES are run."""
    iban_w = WordBox(id=0, text=VALID_IBAN_UK, bbox=(0.1, 0.1, 0.5, 0.12), page=0)
    results = oracle_extract_doc({0: [iban_w]}, fieldtypes=None)
    iban_results = [m for m in results if m.fieldtype == "iban"]
    assert len(iban_results) == 1


# ── Unsupported fieldtype tests ───────────────────────────────────────────────


def test_unsupported_fieldtype_returns_empty():
    """Calling oracle_extract_field with an unsupported type → empty list."""
    words = words_row(["some", "text"])
    assert oracle_extract_field("vendor_name", words, 0) == []
    assert oracle_extract_field("amount_due", words, 0) == []
    assert oracle_extract_field("date_issue", words, 0) == []
    assert oracle_extract_field("line_item_code", words, 0) == []


# ── account_num tests ─────────────────────────────────────────────────────────


def test_account_num_with_label():
    """Account number digits near 'account' label → score 1.0."""
    words = words_row(["Account", "No:", "12345678"])
    results = oracle_extract_field("account_num", words, 0)
    accs = [m for m in results if "12345678" in m.text.replace(" ", "")]
    assert len(accs) >= 1
    assert accs[0].score == 1.0


def test_account_num_multi_word():
    """Account number split across multiple words is joined and matched."""
    # "1234" alone is only 4 digits (too short); joining "1234"+"5678" gives 8 digits → valid
    words = words_row(["Konto:", "1234", "5678", "9012"])
    results = oracle_extract_field("account_num", words, 0)
    # Extractor finds the first valid window: "1234 5678" (8 digits). Multi-word join works.
    accs = [m for m in results if "12345678" in m.text.replace(" ", "")]
    assert len(accs) >= 1
    # word_ids span at least 2 words (multi-word detection)
    assert len(accs[0].word_ids) >= 2


def test_bank_num_with_sort_code_label():
    """Routing number near 'sort code' label → score 1.0."""
    words = words_row(["Sort", "Code:", "12345678"])
    results = oracle_extract_field("bank_num", words, 0)
    banks = [m for m in results if "12345678" in m.text.replace(" ", "")]
    assert len(banks) >= 1
    assert banks[0].score == 1.0


# ── tax_id tests ──────────────────────────────────────────────────────────────


def test_vendor_tax_id_eu_format():
    """EU-style VAT number (2-letter prefix + digits) near 'VAT' label."""
    words = words_row(["VAT", "No:", "DE123456789"])
    results = oracle_extract_field("vendor_tax_id", words, 0)
    vats = [m for m in results if "DE123456789" in m.text.replace(" ", "")]
    assert len(vats) >= 1
    assert vats[0].score == 1.0


def test_customer_tax_id_digit_only():
    """Digit-only tax ID near 'tax no' label."""
    words = words_row(["Tax", "No:", "1234567890"])
    results = oracle_extract_field("customer_tax_id", words, 0)
    taxes = [m for m in results if "1234567890" in m.text.replace(" ", "")]
    assert len(taxes) >= 1


# ── payment_reference tests ───────────────────────────────────────────────────


def test_payment_reference_with_label():
    """Payment reference near 'Ref.' label → at least one match with score 1.0."""
    words = words_row(["Ref.", "PAY-2024-XYZ1"])
    results = oracle_extract_field("payment_reference", words, 0)
    refs = [m for m in results if "PAY" in m.text]
    assert len(refs) >= 1
    # The single-word match "PAY-2024-XYZ1" at pos=1 sees "Ref." in its ±3 context → score 1.0
    assert any(m.score == 1.0 for m in refs)


# ── registration_id tests ─────────────────────────────────────────────────────


def test_vendor_registration_id_with_label():
    """Registration ID near 'KvK' label → score 1.0."""
    words = words_row(["KvK", "Nr:", "12345678"])
    results = oracle_extract_field("vendor_registration_id", words, 0)
    regs = [m for m in results if "12345678" in m.text.replace(" ", "")]
    assert len(regs) >= 1
    assert regs[0].score == 1.0


# ── SUPPORTED_FIELDTYPES coverage ────────────────────────────────────────────


def test_supported_fieldtypes_set():
    """SUPPORTED_FIELDTYPES contains exactly the expected 10 types."""
    expected = {
        "iban", "bic", "account_num", "bank_num",
        "customer_tax_id", "vendor_tax_id",
        "vendor_registration_id", "customer_registration_id",
        "payment_reference", "document_id",
    }
    assert expected == SUPPORTED_FIELDTYPES


def test_oracle_match_is_dataclass():
    """OracleMatch can be constructed and fields accessed."""
    m = OracleMatch(word_ids=[1, 2], text="test", score=0.9, fieldtype="iban")
    assert m.word_ids == [1, 2]
    assert m.text == "test"
    assert m.score == 0.9
    assert m.fieldtype == "iban"
