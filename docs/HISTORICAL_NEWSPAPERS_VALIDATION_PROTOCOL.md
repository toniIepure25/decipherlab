# Historical Newspapers Validation Protocol

## Goal

Measure whether the current real grouped result depends materially on OCR-derived token labels.

## Validation Design

Chosen validation scope:

- the full `test` split of the grouped benchmark
- `30` grouped sequences
- `126` tokens

Why this scope was chosen:

- it is still tractable for direct visual review
- it avoids a smaller-sample subset confound
- it lets the grouped pack be rerun on the exact same evaluation split after corrections

## Review Procedure

1. Generate grouped sequence contact sheets from the cropped token images.
2. Inspect each test-sequence strip visually.
3. Compare the OCR-derived normalized label against the observed token image.
4. Record only explicit corrections in `validation_corrections.csv`.
5. Materialize a validated manifest in which train/val remain OCR-derived and the full test split carries the audited labels.
6. Rerun the grouped pack unchanged on the validated manifest.

## Current Review Mode

The current pass is:

- a curated visual audit performed in-session
- not an independent human annotation campaign

That means the result is useful as a robustness check, but should still be described conservatively in any paper draft.

## Output Artifacts

- [validation_corrections.csv](/home/tonystark/Desktop/decipher/data/processed/historical_newspapers_grouped_words/validation_corrections.csv)
- [validation_subset_annotations.csv](/home/tonystark/Desktop/decipher/data/processed/historical_newspapers_grouped_words/validation_subset_annotations.csv)
- [validation_subset_manifest.yaml](/home/tonystark/Desktop/decipher/data/processed/historical_newspapers_grouped_words/validation_subset_manifest.yaml)
- [validation_label_noise_summary.md](/home/tonystark/Desktop/decipher/data/processed/historical_newspapers_grouped_words/validation_label_noise_summary.md)
- [validation_qc](/home/tonystark/Desktop/decipher/data/processed/historical_newspapers_grouped_words/validation_qc)

## Scientific Interpretation

This protocol can answer one narrow question:

- does the current real grouped result survive a direct label audit on the evaluated grouped sequences?

It does not answer:

- whether the benchmark is equivalent to gold manual annotation
- whether the result replicates on a second real grouped corpus
