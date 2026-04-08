# Limitations And Threats

## Current Technical Limitations

- The posterior model is a calibrated baseline built from PCA embeddings plus logistic classification or prototype distance. It is not a full OCR/HTR recognizer.
- Downstream family scoring remains heuristic.
- The old heuristic posterior path is intentionally simple; it is a comparison baseline, not a strong production model.
- Real-data support currently assumes glyph-crop manifests, not full document layout or line segmentation.
- Bootstrap intervals are sequence-level uncertainty estimates for the implemented protocol, not population guarantees about all future corpora.

## External Validity Threats

- Fixture-scale or small manifest experiments do not justify historical generalization.
- Dataset-specific crop quality, label consistency, and split quality can materially affect conclusions.
- Family-level labels may be unavailable or weak for many real corpora, limiting downstream evaluation.
- Controlled ambiguity injection is informative but does not capture every failure mode of archival imagery.
- Multi-seed stability on one dataset does not imply stability across distinct scripts, hands, or acquisition pipelines.

## Interpretation Threats

- Better symbol top-k recovery does not automatically imply better decipherment.
- Lower calibration error on one dataset does not imply global calibration robustness.
- Null-resistant structural behavior should not be conflated with semantic interpretation.
- If the calibrated posterior falls back to prototype behavior due to low supervision, claims about calibration must be weakened.
- If uncertainty improves top-k retention without changing downstream ranking, the claim must remain symbol-level rather than decipherment-level.

## What Would Strengthen Publication Evidence

- A sequence-rich manuscript or cipher dataset with stable train/val/test splits and richer grouped structure than the current three corpora
- Additional datasets with different hands, symbol inventories, and degradation profiles
- Comparison against stronger OCR/HTR baselines
- Better downstream inference engines that can actually exploit preserved alternatives
- Multi-seed experiments and confidence intervals on the headline results
- Real downstream tasks with enough family/group labels to test whether symbol-level rescue translates into structural or hypothesis-level gains
