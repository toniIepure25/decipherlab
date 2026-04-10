# Reviewer Risk Checklist

## 1. “This is still too synthetic.”

- Best evidence-based answer: the paper now separates the evidence ladder explicitly and includes two real grouped/token-aligned corpora plus a redesigned real downstream task; only the process-family result remains synthetic-from-real.
- Evidence artifact: [outputs/real_vs_synthetic_bridge_summary.csv](/home/tonystark/Desktop/decipher/outputs/real_vs_synthetic_bridge_summary.csv)
- Manuscript location: introduction, scope section, limitations
- Coverage status: covered

## 2. “The downstream result is mixed.”

- Best evidence-based answer: yes, and that is part of the contribution. The redesigned task removes much of the coverage objection, and the paper explains the remaining mixed outcome rather than hiding it.
- Evidence artifact: [outputs/real_downstream_redesigned_summary.csv](/home/tonystark/Desktop/decipher/outputs/real_downstream_redesigned_summary.csv), [outputs/statistical_robustness_summary.csv](/home/tonystark/Desktop/decipher/outputs/statistical_robustness_summary.csv), [outputs/propagation_regime_summary.csv](/home/tonystark/Desktop/decipher/outputs/propagation_regime_summary.csv)
- Manuscript location: redesigned downstream results, propagation analysis, limitations
- Coverage status: covered

## 3. “Why not just build a stronger decoder?”

- Best evidence-based answer: the branch already evaluated trigram and CRF-style decoders, and neither produced a universal improvement. The remaining bottleneck is propagation conditions, not decoder count.
- Evidence artifact: [docs/SEQUENCE_RESULTS_NOTES.md](/home/tonystark/Desktop/decipher/docs/SEQUENCE_RESULTS_NOTES.md)
- Manuscript location: methods and discussion
- Coverage status: covered

## 4. “Isn’t this just OCR calibration?”

- Best evidence-based answer: no. Calibration is mixed across datasets, while the more stable finding is uncertainty preservation and its propagation boundary.
- Evidence artifact: [outputs/real_vs_synthetic_bridge_summary.csv](/home/tonystark/Desktop/decipher/outputs/real_vs_synthetic_bridge_summary.csv), [docs/NEW_CLAIMS_AUDIT.md](/home/tonystark/Desktop/decipher/docs/NEW_CLAIMS_AUDIT.md)
- Manuscript location: methods, grouped results, limitations
- Coverage status: covered

## 5. “Why does exact recovery not improve more?”

- Best evidence-based answer: because grouped rescue is necessary but not sufficient. The downstream rescue gap conditioned on grouped rescue is large, but it is still support-regime-dependent rather than automatic.
- Evidence artifact: [outputs/statistical_robustness_summary.csv](/home/tonystark/Desktop/decipher/outputs/statistical_robustness_summary.csv), [outputs/propagation_model_summary.csv](/home/tonystark/Desktop/decipher/outputs/propagation_model_summary.csv), [outputs/propagation_case_studies.md](/home/tonystark/Desktop/decipher/outputs/propagation_case_studies.md)
- Manuscript location: propagation analysis and discussion
- Coverage status: covered

## 6. “What is actually new here?”

- Best evidence-based answer: the paper contributes a support-aware uncertainty propagation analysis grounded in both synthetic-from-real and real grouped data, with replicated real grouped top-k transfer and an explanatory account of why higher-level propagation often fails.
- Evidence artifact: [outputs/propagation_model_summary.csv](/home/tonystark/Desktop/decipher/outputs/propagation_model_summary.csv), [outputs/propagation_thresholds.csv](/home/tonystark/Desktop/decipher/outputs/propagation_thresholds.csv), [outputs/propagation_regime_plot.png](/home/tonystark/Desktop/decipher/outputs/propagation_regime_plot.png)
- Manuscript location: introduction and contributions
- Coverage status: covered, but wording should remain especially crisp

## 7. “The real grouped corpora are not real decipherment data.”

- Best evidence-based answer: correct, and the manuscript says so explicitly. The paper is positioned as grouped recognition under ambiguity, not semantic decipherment.
- Evidence artifact: [submission_bplus/claims_check.md](/home/tonystark/Desktop/decipher/submission_bplus/claims_check.md)
- Manuscript location: framing and scope, limitations
- Coverage status: covered

## 8. “Why should reviewers trust the Historical Newspapers labels?”

- Best evidence-based answer: the benchmark survived a full-split audit and a gold-style adjudicated pass with only two token corrections, narrow OCR-substitution errors, and zero metric drift.
- Evidence artifact: [outputs/real_data_trust_summary.csv](/home/tonystark/Desktop/decipher/outputs/real_data_trust_summary.csv), [docs/REAL_DATA_TRUST_HARDENING.md](/home/tonystark/Desktop/decipher/docs/REAL_DATA_TRUST_HARDENING.md)
- Manuscript location: real grouped corpus section and appendix
- Coverage status: covered
