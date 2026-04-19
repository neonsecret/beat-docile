"""[RESEARCH-BURIED] Per-field-type extraction instructions for the DocILE prompt.

Status: RESEARCH-BURIED — -8.9pp KILE (original), -10.82pp KILE (FORMAT-only
rewrite on 50-doc). See KNOWLEDGE_BASE.md §6.10 for details. Root cause: detailed
per-field instructions cause Sonnet to over-constrain and miss valid fields.
Concise system prompt is better; these instructions may be useful as LoRA
fine-tuning signals (see §6.10) rather than runtime prompt content.
"""

from __future__ import annotations

FIELD_INSTRUCTIONS: dict[str, str] = {
    # ── KILE fields (36) ────────────────────────────────────────────────────────
    "account_num": (
        "Find the bank account number near labels like 'Account No', 'Konto Nr', "
        "'Compte', 'Číslo účtu'. NOT the IBAN (starts with 2-letter country code) "
        "and NOT the BIC/SWIFT code. Include only the number itself."
    ),
    "amount_due": (
        "Find the total amount still owed, near 'Amount Due', 'Zu zahlen', "
        "'Montant dû', 'Balance Due', 'Saldo'. Do NOT confuse with amount_total_gross "
        "or amount_paid. Include only the numeric value and any adjacent currency symbol."
    ),
    "amount_paid": (
        "Find the amount already paid, near 'Amount Paid', 'Vorauszahlung', "
        "'Acompte', 'Deposit', 'Paid'. This is a credit against the invoice total. "
        "Omit if absent. Include only the numeric value."
    ),
    "amount_total_gross": (
        "Find the total gross amount including tax, near 'Total', 'Grand Total', "
        "'Gesamt', 'Bruttobetrag', 'Total TTC'. Must include tax. "
        "Do NOT confuse with net total. Include only the numeric value."
    ),
    "amount_total_net": (
        "Find the total net amount before tax, near 'Subtotal', 'Netto', "
        "'Total HT', 'Net Total', 'Before VAT'. Do NOT include the tax amount. "
        "Include only the numeric value."
    ),
    "amount_total_tax": (
        "Find the aggregate tax amount across all line items, near 'VAT', 'Tax', "
        "'MwSt', 'TVA', 'GST'. Do NOT confuse with per-rate tax_detail_tax. "
        "Include only the numeric value."
    ),
    "bank_num": (
        "Find the bank routing/sort code near 'Sort Code', 'BLZ', 'Routing Number', "
        "'ABA', 'Bankleitzahl', 'BSB'. Short numeric code identifying the bank branch. "
        "NOT the full account number, NOT the IBAN."
    ),
    "bic": (
        "Find the BIC/SWIFT code — exactly 8 or 11 uppercase alphanumeric characters "
        "(e.g., DEUTDEDB, NWBKGB2LXXX). Near labels 'BIC', 'SWIFT', 'BIC/SWIFT'. "
        "Do NOT include the label itself or surrounding colons."
    ),
    "currency_code_amount_due": (
        "Find the currency symbol (€, $, £) or ISO 4217 3-letter code (EUR, USD, GBP) "
        "for the amount due. Usually adjacent to or on the same row as amount_due. "
        "Include ONLY the symbol or 3-letter code, not the amount."
    ),
    "customer_billing_address": (
        "Find the full billing address block for the customer near 'Bill To', "
        "'Invoice To', 'Rechnungsadresse'. Include ALL lines: street, city, postal code, "
        "country. Do NOT include the label words or the customer name line above."
    ),
    "customer_billing_name": (
        "Find the customer/buyer name in the billing section near 'Bill To', "
        "'Invoice To', 'Rechnungsadresse'. Include the name only, "
        "NOT the address lines below it."
    ),
    "customer_delivery_address": (
        "Find the delivery/ship-to address for the customer near 'Ship To', "
        "'Delivery Address', 'Lieferadresse'. Include ALL address lines: street, city, "
        "postal code, country. Do NOT include the label or delivery name."
    ),
    "customer_delivery_name": (
        "Find the recipient name at the delivery/ship-to address near 'Ship To', "
        "'Deliver To', 'Lieferadresse'. Include the name only, not the address lines below."
    ),
    "customer_id": (
        "Find the customer identifier near 'Customer No', 'Client ID', 'Kundennummer', "
        "'Customer Code', 'Debtor No'. Include only the identifier value, not the label."
    ),
    "customer_order_id": (
        "Find the order ID issued BY the customer (buyer's PO) near 'Customer PO', "
        "'Your Order No', 'PO Number', 'Ihre Bestellnr'. "
        "Do NOT confuse with vendor_order_id or document_id."
    ),
    "customer_other_address": (
        "Find any additional customer address that is neither billing nor delivery — "
        "e.g., a correspondence or registered address. "
        "Include ALL address lines; omit if no such address is present."
    ),
    "customer_other_name": (
        "Find any additional customer name that is neither billing nor delivery name — "
        "e.g., 'Contact', 'Attention', 'c/o'. Include the name only. "
        "Omit if absent."
    ),
    "customer_registration_id": (
        "Find the customer's company registration/business ID near 'Reg No', 'KvK', "
        "'HRB', 'IČO', 'ABN', 'CRN', 'Company No'. "
        "Include only the ID value, not the label."
    ),
    "customer_tax_id": (
        "Find the customer's VAT/tax ID near 'VAT No', 'USt-IdNr', 'DIČ', "
        "'TVA', 'NIF', 'BTW', 'ΑΦΜ'. "
        "Do NOT include the vendor's VAT ID. Include only the ID value."
    ),
    "date_due": (
        "Find the payment due date near 'Due Date', 'Payment Due', 'Fälligkeitsdatum', "
        "'Échéance', 'Płatność do'. "
        "Do NOT confuse with date_issue (invoice creation date). Include the date value only."
    ),
    "date_issue": (
        "Find the invoice issue/creation date near 'Invoice Date', 'Date', "
        "'Rechnungsdatum', 'Datum', 'Date de facture'. "
        "Do NOT confuse with date_due (payment due date). Include the date value only."
    ),
    "document_id": (
        "Find the invoice/document number near 'Invoice No', 'Rechnung Nr', 'Faktura', "
        "'Document ID', 'Bill No', 'Inv #', 'Numer faktury'. "
        "Include only the alphanumeric identifier, not the label."
    ),
    "iban": (
        "Find the IBAN starting with a 2-letter country code then 2 check digits and up "
        "to 30 alphanumeric chars (e.g., DE89 3704 0044...). "
        "Select word_ids covering the ENTIRE IBAN including all space-separated groups."
    ),
    "order_id": (
        "Find a general order identifier not specifically attributed to customer or vendor, "
        "near 'Order No', 'Order ID', 'Auftragsnummer'. "
        "Use customer_order_id / vendor_order_id if attribution is clear."
    ),
    "payment_reference": (
        "Find the payment reference/remittance string near 'Payment Reference', "
        "'Reference', 'Verwendungszweck', 'Komunikat', 'Creditor Reference'. "
        "This is what the payer includes with the bank transfer. Include the full string."
    ),
    "payment_terms": (
        "Find payment terms text near 'Payment Terms', 'Terms', 'Zahlungsbedingungen', "
        "'Conditions de paiement'. Examples: 'Net 30', '14 days', '2/10 net 30', "
        "'Due on receipt'. Include the complete terms string."
    ),
    "tax_detail_gross": (
        "In a tax breakdown table, find the gross amount (net + tax) for ONE tax-rate row. "
        "Extract one entry per rate row. Near column header 'Gross', 'Brutto'. "
        "Do NOT confuse with amount_total_gross."
    ),
    "tax_detail_net": (
        "In a tax breakdown table, find the net base amount for ONE tax-rate row. "
        "Extract one entry per rate row. Near column header 'Net', 'Netto', 'Base'. "
        "Do NOT confuse with amount_total_net."
    ),
    "tax_detail_rate": (
        "In a tax breakdown table, find the tax rate percentage for ONE row "
        "(e.g., 19%, 21%, 7%, 0%). Extract one entry per rate row. "
        "Include the number with or without % sign as shown."
    ),
    "tax_detail_tax": (
        "In a tax breakdown table, find the tax amount for ONE tax-rate row. "
        "Extract one entry per rate row. Near column header 'Tax', 'VAT', 'MwSt'. "
        "Do NOT confuse with amount_total_tax."
    ),
    "vendor_address": (
        "Find the full vendor/supplier address block, typically at top of invoice. "
        "Include ALL lines: street, city, postal code, country. "
        "Do NOT include vendor_name (the name line above) or phone/email/web."
    ),
    "vendor_email": (
        "Find the vendor's email address in format user@domain.tld. "
        "Near labels 'Email', 'E-mail', 'Mail'. "
        "Include the email address only — must contain '@'."
    ),
    "vendor_name": (
        "Find the vendor/issuer company or person name — usually the most prominent "
        "text at the top of the invoice. "
        "Do NOT include address lines below the name."
    ),
    "vendor_order_id": (
        "Find the order ID issued BY the vendor near 'Our Order No', 'Vendor PO', "
        "'Our Reference', 'Auftragsnr'. "
        "Do NOT confuse with customer_order_id."
    ),
    "vendor_registration_id": (
        "Find the vendor's company registration/business ID near 'Reg No', 'KvK', "
        "'HRB', 'IČO', 'ABN', 'CRN', 'Company Registration'. "
        "Include only the ID value, not the label."
    ),
    "vendor_tax_id": (
        "Find the vendor's VAT/tax ID near 'VAT No', 'USt-IdNr', 'MwSt-IdNr', "
        "'DIČ', 'TVA No', 'NIF', 'Steuernummer'. "
        "Do NOT include the customer's VAT ID. Include only the ID value."
    ),
    # ── LIR fields (19) ────────────────────────────────────────────────────────
    "line_item_amount_gross": (
        "For this table row, find the gross amount (qty x unit gross price) in the "
        "amount column. Near header 'Gross', 'Amount', 'Betrag brutto'. "
        "Include only the numeric value."
    ),
    "line_item_amount_net": (
        "For this table row, find the net amount (before tax) in the amount column. "
        "Near header 'Net Amount', 'Betrag netto', 'Prix HT', 'Subtotal'. "
        "Include only the numeric value."
    ),
    "line_item_code": (
        "For this table row, find the product code, SKU, article or item number. "
        "Near column header 'Code', 'SKU', 'Item No', 'Art. No', 'Artikelnummer'. "
        "Include only the code, NOT the description."
    ),
    "line_item_currency": (
        "For this table row, find the currency code (EUR, USD) or symbol (€, $) if "
        "explicitly listed per row. Near header 'Currency'. "
        "Include only the symbol or 3-letter code."
    ),
    "line_item_date": (
        "For this table row, find a date specifically tied to this line item — "
        "e.g., service date, delivery date, period. "
        "Include only the date value, not surrounding text."
    ),
    "line_item_description": (
        "For this table row, find the full product/service description. May span multiple "
        "words or wrap within the same row. Include ALL description words for this row; "
        "do NOT spill into adjacent rows."
    ),
    "line_item_discount_amount": (
        "For this table row, find the discount in absolute monetary value. "
        "Near header 'Discount', 'Rabatt', 'Remise'. "
        "Include only the numeric value."
    ),
    "line_item_discount_rate": (
        "For this table row, find the discount as a percentage. "
        "Near header 'Disc %', 'Rabatt %', 'Remise %'. "
        "Include only the rate value (e.g., 10%)."
    ),
    "line_item_hts_number": (
        "For this table row, find the Harmonized Tariff Schedule (HS/HTS) number. "
        "Near header 'HS Code', 'HTS', 'Tariff Code'. "
        "Format: typically 6-10 digits with dots (e.g., 8471.30.00)."
    ),
    "line_item_order_id": (
        "For this table row, find an order ID specific to this line item. "
        "Near header 'Order Ref', 'PO Ref'. "
        "Include only the identifier."
    ),
    "line_item_person_name": (
        "For this table row, find a person name associated with this line item — "
        "e.g., consultant, technician, employee name. "
        "Include the name only."
    ),
    "line_item_position": (
        "For this table row, find the row/position number — a small integer in the "
        "leftmost column (1, 2, 3...). Near headers 'Pos', '#', 'Nr', 'Item'. "
        "Include only the integer."
    ),
    "line_item_quantity": (
        "For this table row, find the quantity value. Near header 'Qty', 'Quantity', "
        "'Menge', 'Anzahl'. Include only the numeric value (integer or decimal). "
        "Do NOT include the unit of measure."
    ),
    "line_item_tax": (
        "For this table row, find the tax amount. Near header 'Tax', 'VAT', 'MwSt', "
        "'TVA'. Include only the numeric value."
    ),
    "line_item_tax_rate": (
        "For this table row, find the applicable tax rate percentage. "
        "Near header 'Tax Rate', 'VAT %', 'MwSt %'. "
        "Include only the rate value (e.g., 19%, 21%)."
    ),
    "line_item_unit_price_gross": (
        "For this table row, find the unit price including tax. "
        "Near header 'Unit Price (Gross)', 'Einzelpreis brutto', 'Prix unitaire TTC'. "
        "Include only the numeric value."
    ),
    "line_item_unit_price_net": (
        "For this table row, find the unit price excluding tax. "
        "Near header 'Unit Price', 'Price', 'Einzelpreis netto', 'Prix unitaire HT'. "
        "Include only the numeric value."
    ),
    "line_item_units_of_measure": (
        "For this table row, find the unit of measure. Near header 'Unit', 'UOM', "
        "'Einheit'. Examples: pcs, kg, hrs, ea, m², box, l. "
        "Include only the unit abbreviation."
    ),
    "line_item_weight": (
        "For this table row, find the weight value. Near header 'Weight', 'Gewicht', "
        "'Poids'. Include the numeric value; include the unit (kg, lbs) "
        "only if it appears as part of the same word/token."
    ),
}


def get_field_instruction(fieldtype: str) -> str | None:
    """Return the extraction instruction for this field type, or None if not defined."""
    return FIELD_INSTRUCTIONS.get(fieldtype)


# Pre-built guidance block for all 55 field types (used in extraction prompt).
# Computed once at module load to avoid per-call overhead.
ALL_FIELD_GUIDANCE: str = "\n".join(
    f"- {ft}: {instr}" for ft, instr in sorted(FIELD_INSTRUCTIONS.items())
)
