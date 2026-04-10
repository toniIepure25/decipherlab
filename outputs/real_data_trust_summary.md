# Real Data Trust Summary

## Main Takeaways

- The Historical Newspapers benchmark now has a stronger trust chain: OCR-derived labels, direct visual audit, and a gold-style adjudicated pass.
- Only two tokens changed across the full evaluated test split, and all grouped metrics remained unchanged after both the audit and gold-style passes.
- The observed corrections are narrow OCR substitutions, not broad sequence-level annotation failures.

## Error Taxonomy

- `ocr_substitution:den->der`: `1`. Observed gold-style correction count.
- `ocr_substitution:der->des`: `1`. Observed gold-style correction count.

## Gold Style

- `pass_agreement_rate`: `1.000`. Pass A vs Pass B agreement on gold-style pass.
- `ocr_to_gold_token_error_rate`: `0.016`. OCR-derived token disagreement against adjudicated labels.
- `ocr_to_gold_sequence_error_rate`: `0.033`. Grouped sequence disagreement rate.

## Metric Stability Audit

- `cluster_distance_validated_minus_original_conformal_exact_delta`: `0.000`. Metric drift after first audit pass.
- `calibrated_classifier_validated_minus_original_conformal_exact_delta`: `0.000`. Metric drift after first audit pass.

## Metric Stability Gold

- `cluster_distance_gold_minus_original_conformal_exact_delta`: `0.000`. Metric drift after gold-style adjudication.
- `calibrated_classifier_gold_minus_original_conformal_exact_delta`: `0.000`. Metric drift after gold-style adjudication.

## Trust Support

- `qc_contact_sheet_count`: `6`. Saved QC sheets covering the reviewed Historical subset.

## Validation Audit

- `audited_tokens`: `126`. Full Historical Newspapers test split visual audit.
- `audited_sequences`: `30`. Grouped sequences reviewed in the audit pass.
- `changed_tokens`: `2`. Tokens changed relative to OCR-derived labels.
