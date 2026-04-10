# Historical Newspapers Gold Protocol

## Goal

Upgrade the current real grouped benchmark from OCR-derived labels to a stronger gold-style evaluation slice without changing the structured-uncertainty methods.

## Scope

Chosen gold-style scope:

- full `test` split
- `30` grouped sequences
- `126` tokens

This keeps the comparison clean:

- training and validation remain unchanged
- only the evaluated grouped labels become more trustworthy

## Protocol

1. Start from the validated grouped manifest.
2. Export full-token annotation rows for the test split.
3. Perform a second in-session visual pass over the entire test split.
4. Record pass-A and pass-B labels plus the adjudicated label.
5. Materialize a gold-style grouped manifest with the adjudicated labels.
6. Rerun the grouped pack unchanged on the gold-style manifest.

## Agreement Tracking

Tracked quantities:

- pass-A / pass-B agreement
- OCR-to-gold token error rate
- OCR-to-gold sequence error rate
- error-type breakdown for corrected labels

## Caveat

This is a **gold-style** subset, not a true independent multi-annotator gold campaign.

Why the wording matters:

- it is stronger than the OCR-derived benchmark
- it is stronger than the first light audit
- it is still weaker than an externally validated annotation effort

## Output Artifacts

- [gold_annotations.csv](/home/tonystark/Desktop/decipher/data/processed/historical_newspapers_grouped_words/gold_annotations.csv)
- [gold_subset_manifest.yaml](/home/tonystark/Desktop/decipher/data/processed/historical_newspapers_grouped_words/gold_subset_manifest.yaml)
- [gold_agreement_summary.md](/home/tonystark/Desktop/decipher/data/processed/historical_newspapers_grouped_words/gold_agreement_summary.md)
- [real_grouped_strengthened_summary.md](/home/tonystark/Desktop/decipher/outputs/real_grouped_strengthened_summary.md)
