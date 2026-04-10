# Appendix

## A. Reproducibility

The stronger sequence branch is manifest-backed and config-driven throughout. The main artifacts referenced in the paper are generated from deterministic experiment configs and fixed-seed runs already recorded in the repository outputs.

Key evidence artifacts:

- [outputs/real_vs_synthetic_bridge_summary.csv](/home/tonystark/Desktop/decipher/outputs/real_vs_synthetic_bridge_summary.csv)
- [outputs/propagation_model_summary.csv](/home/tonystark/Desktop/decipher/outputs/propagation_model_summary.csv)
- [outputs/propagation_regime_summary.csv](/home/tonystark/Desktop/decipher/outputs/propagation_regime_summary.csv)
- [outputs/real_grouped_replication_summary.csv](/home/tonystark/Desktop/decipher/outputs/real_grouped_replication_summary.csv)
- [outputs/real_downstream_redesigned_summary.csv](/home/tonystark/Desktop/decipher/outputs/real_downstream_redesigned_summary.csv)
- [outputs/statistical_robustness_summary.csv](/home/tonystark/Desktop/decipher/outputs/statistical_robustness_summary.csv)
- [outputs/real_data_trust_summary.csv](/home/tonystark/Desktop/decipher/outputs/real_data_trust_summary.csv)

## B. Additional Boundary Analysis

The strongest boundary artifact is the redesigned real downstream task. It materially improves coverage relative to exact transcript-bank overlap, but exact downstream gains remain selective rather than replicated.

This matters because it rules out the simplest excuse for the weak downstream story while still preserving an honest limit.

## C. Real-Data Trust Hardening

The Historical Newspapers grouped benchmark was hardened with:

- a full-split visual audit
- a gold-style two-pass adjudicated label pass
- explicit disagreement and error-taxonomy reporting
- metric-stability checks after each label tier

The resulting trust summary is [outputs/real_data_trust_summary.md](/home/tonystark/Desktop/decipher/outputs/real_data_trust_summary.md).

## D. Why The Paper Does Not Need Another Decoder

The branch already compares:

- `fixed_greedy`
- `uncertainty_beam`
- `conformal_beam`
- `uncertainty_trigram_beam`
- `conformal_trigram_beam`
- a CRF-style exact decoder

Those comparisons show that the remaining bottleneck is not decoder count. The stronger paper contribution is explanatory: when rescue propagates and when it does not.

## E. Suggested Supplementary Artifacts

- real grouped replication plot
- real downstream redesign summary
- statistical robustness summary
- propagation thresholds and regime summaries
- failure summaries for grouped and downstream settings

## F. Limit Statement For Supplement

The paper’s strongest positive higher-level evidence remains synthetic-from-real, while the real downstream evidence is selective and support-regime-dependent. This limitation is not hidden; it is part of the paper’s scientific conclusion.
