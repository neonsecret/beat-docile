"""Tests for beat_docile.validators — format_confidence function."""

from beat_docile.validators import format_confidence

# ── IBAN ──────────────────────────────────────────────────────────────────────

class TestIBAN:
    def test_valid_gb_iban(self):
        # GB29 NWBK 6016 1331 9268 19 — well-known test IBAN
        assert format_confidence("iban", "GB29NWBK60161331926819") == 1.0

    def test_valid_de_iban(self):
        # DE89 3704 0044 0532 0130 00
        assert format_confidence("iban", "DE89370400440532013000") == 1.0

    def test_valid_iban_with_spaces(self):
        # Spaces should be stripped
        assert format_confidence("iban", "GB29 NWBK 6016 1331 9268 19") == 1.0

    def test_invalid_iban_bad_checksum(self):
        # Modify last digit to break checksum
        assert format_confidence("iban", "GB29NWBK60161331926818") == 0.0

    def test_invalid_iban_not_country_code(self):
        # Starts with digits, not letters
        assert format_confidence("iban", "1234567890123456") == 0.0

    def test_invalid_iban_too_short(self):
        assert format_confidence("iban", "GB29NWB") == 0.0


# ── BIC / SWIFT ───────────────────────────────────────────────────────────────

class TestBIC:
    def test_valid_bic_8_chars(self):
        assert format_confidence("bic", "DEUTDEDB") == 1.0

    def test_valid_bic_11_chars(self):
        assert format_confidence("bic", "NWBKGB2LXXX") == 1.0

    def test_invalid_bic_wrong_length(self):
        assert format_confidence("bic", "DEUT") == 0.0

    def test_valid_bic_lowercase_accepted(self):
        # Lowercase BIC normalised to uppercase — accepted conservatively
        assert format_confidence("bic", "deutdedb") == 1.0


# ── VAT / Tax IDs ─────────────────────────────────────────────────────────────

class TestTaxID:
    def test_valid_de_vat(self):
        assert format_confidence("vendor_tax_id", "DE123456789") == 1.0

    def test_valid_gb_vat(self):
        assert format_confidence("customer_tax_id", "GB123456789") == 1.0

    def test_invalid_tax_id_with_garbage_chars(self):
        # @ character is garbage
        assert format_confidence("vendor_tax_id", "DE123@456") == 0.0

    def test_invalid_tax_id_empty(self):
        assert format_confidence("vendor_tax_id", "") == 0.0

    def test_ambiguous_tax_id_with_space(self):
        # Space inside tax ID is suspicious → 0.5
        assert format_confidence("customer_tax_id", "DE 123 456 789") == 0.5


# ── Registration IDs ──────────────────────────────────────────────────────────

class TestRegistrationID:
    def test_valid_reg_id_alphanumeric(self):
        assert format_confidence("vendor_registration_id", "HRB12345") == 1.0

    def test_valid_reg_id_with_slash(self):
        assert format_confidence("customer_registration_id", "KvK/12345678") == 1.0

    def test_invalid_reg_id_empty(self):
        assert format_confidence("vendor_registration_id", "") == 0.0

    def test_uncertain_reg_id_very_long(self):
        # Very long string is unlikely to be a reg ID
        result = format_confidence("customer_registration_id", "A" * 25)
        assert result <= 0.5


# ── Amounts ───────────────────────────────────────────────────────────────────

class TestAmounts:
    def test_valid_amount_usd(self):
        assert format_confidence("amount_due", "$1,234.56") == 1.0

    def test_valid_amount_euro_eu_format(self):
        assert format_confidence("amount_total_gross", "€1.234,56") == 1.0

    def test_valid_amount_plain(self):
        assert format_confidence("amount_paid", "1234.56") == 1.0

    def test_valid_line_item_amount(self):
        assert format_confidence("line_item_amount_net", "100.00") == 1.0

    def test_invalid_amount_text(self):
        assert format_confidence("amount_due", "one hundred dollars") == 0.0

    def test_invalid_amount_empty(self):
        assert format_confidence("tax_detail_gross", "") == 0.0


# ── Tax rates ─────────────────────────────────────────────────────────────────

class TestRates:
    def test_valid_rate_with_percent(self):
        assert format_confidence("tax_detail_rate", "21%") == 1.0

    def test_valid_rate_without_percent(self):
        assert format_confidence("line_item_tax_rate", "19") == 1.0

    def test_valid_rate_decimal(self):
        assert format_confidence("line_item_discount_rate", "7.5%") == 1.0

    def test_invalid_rate_text(self):
        assert format_confidence("tax_detail_rate", "twenty percent") == 0.0

    def test_invalid_rate_over_100(self):
        # 200% is clearly wrong
        assert format_confidence("tax_detail_rate", "200%") == 0.0


# ── Dates ─────────────────────────────────────────────────────────────────────

class TestDates:
    def test_valid_iso_date(self):
        assert format_confidence("date_issue", "2024-01-15") == 1.0

    def test_valid_eu_date(self):
        assert format_confidence("date_due", "15.01.2024") == 1.0

    def test_valid_text_date(self):
        assert format_confidence("line_item_date", "15 Jan 2024") == 1.0

    def test_valid_us_date(self):
        assert format_confidence("date_issue", "01/15/2024") == 1.0

    def test_invalid_date_random_text(self):
        assert format_confidence("date_due", "not a date at all") == 0.0

    def test_invalid_date_empty(self):
        assert format_confidence("date_issue", "") == 0.0


# ── Currency codes ─────────────────────────────────────────────────────────────

class TestCurrencyCodes:
    def test_valid_iso_code_usd(self):
        assert format_confidence("currency_code_amount_due", "USD") == 1.0

    def test_valid_iso_code_eur(self):
        assert format_confidence("line_item_currency", "EUR") == 1.0

    def test_valid_currency_symbol(self):
        assert format_confidence("currency_code_amount_due", "$") == 1.0

    def test_invalid_currency_code_too_long(self):
        assert format_confidence("line_item_currency", "USDA") == 0.0

    def test_invalid_currency_empty(self):
        assert format_confidence("currency_code_amount_due", "") == 0.0


# ── Email ─────────────────────────────────────────────────────────────────────

class TestEmail:
    def test_valid_email(self):
        assert format_confidence("vendor_email", "info@example.com") == 1.0

    def test_valid_email_subdomain(self):
        assert format_confidence("vendor_email", "billing@company.co.uk") == 1.0

    def test_invalid_email_no_at(self):
        assert format_confidence("vendor_email", "notanemail.com") == 0.0

    def test_invalid_email_empty(self):
        assert format_confidence("vendor_email", "") == 0.0


# ── Quantity ──────────────────────────────────────────────────────────────────

class TestQuantity:
    def test_valid_integer_quantity(self):
        assert format_confidence("line_item_quantity", "5") == 1.0

    def test_valid_decimal_quantity(self):
        assert format_confidence("line_item_quantity", "2.5") == 1.0

    def test_invalid_quantity_text(self):
        assert format_confidence("line_item_quantity", "five") == 0.0

    def test_invalid_quantity_empty(self):
        assert format_confidence("line_item_quantity", "") == 0.0


# ── Position ──────────────────────────────────────────────────────────────────

class TestPosition:
    def test_valid_position_1(self):
        assert format_confidence("line_item_position", "1") == 1.0

    def test_valid_position_10(self):
        assert format_confidence("line_item_position", "10") == 1.0

    def test_invalid_position_text(self):
        assert format_confidence("line_item_position", "first") == 0.0

    def test_invalid_position_decimal(self):
        assert format_confidence("line_item_position", "1.5") == 0.0


# ── Free-text / unknown fields ────────────────────────────────────────────────

class TestFreeText:
    def test_unknown_fieldtype_returns_1(self):
        assert format_confidence("completely_unknown_field", "anything") == 1.0

    def test_vendor_name_no_penalty(self):
        assert format_confidence("vendor_name", "ACME Corporation Ltd.") == 1.0

    def test_vendor_address_no_penalty(self):
        assert format_confidence("vendor_address", "123 Main St, London, UK") == 1.0

    def test_payment_terms_no_penalty(self):
        assert format_confidence("payment_terms", "Net 30 days from invoice date") == 1.0

    def test_line_item_description_no_penalty(self):
        assert format_confidence("line_item_description", "Professional services rendered") == 1.0

    def test_document_id_no_penalty(self):
        assert format_confidence("document_id", "INV-2024-001234") == 1.0
