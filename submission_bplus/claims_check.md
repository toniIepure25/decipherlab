# Claims Check

## Central Claim

Preserved transcription uncertainty improves real grouped top-k recovery across two token-aligned corpora, but higher-level propagation is support-gated: grouped rescue predicts downstream success only in specific support regimes, and exact real downstream gains remain selective rather than replicated.

## Secondary Supported Claims

1. Preserving transcription uncertainty improves symbol-level retention under ambiguity on real glyph corpora.
Evidence: [outputs/real_vs_synthetic_bridge_summary.csv](/home/tonystark/Desktop/decipher/outputs/real_vs_synthetic_bridge_summary.csv), [docs/NEW_CLAIMS_AUDIT.md](/home/tonystark/Desktop/decipher/docs/NEW_CLAIMS_AUDIT.md)

2. Grouped top-k rescue replicates across Historical Newspapers and ScaDS.AI.
Evidence: [outputs/real_grouped_replication_summary.csv](/home/tonystark/Desktop/decipher/outputs/real_grouped_replication_summary.csv), [outputs/real_grouped_replication_plot.png](/home/tonystark/Desktop/decipher/outputs/real_grouped_replication_plot.png), [outputs/statistical_robustness_summary.csv](/home/tonystark/Desktop/decipher/outputs/statistical_robustness_summary.csv)

3. The redesigned real downstream task removes much of the prior coverage collapse.
Evidence: [outputs/real_downstream_coverage_analysis.csv](/home/tonystark/Desktop/decipher/outputs/real_downstream_coverage_analysis.csv), [outputs/real_downstream_redesigned_summary.csv](/home/tonystark/Desktop/decipher/outputs/real_downstream_redesigned_summary.csv)

4. Exact real downstream gains appear only selectively after the redesign.
Evidence: [outputs/real_downstream_redesigned_summary.csv](/home/tonystark/Desktop/decipher/outputs/real_downstream_redesigned_summary.csv), [outputs/real_vs_synthetic_bridge_summary.csv](/home/tonystark/Desktop/decipher/outputs/real_vs_synthetic_bridge_summary.csv)

5. Symbol rescue is the clearest predictor of grouped rescue.
Evidence: [outputs/propagation_model_summary.csv](/home/tonystark/Desktop/decipher/outputs/propagation_model_summary.csv), [outputs/statistical_robustness_summary.csv](/home/tonystark/Desktop/decipher/outputs/statistical_robustness_summary.csv)

6. Grouped rescue is necessary but not sufficient for downstream success, and conformal helps mainly in specific support regimes.
Evidence: [outputs/propagation_model_summary.csv](/home/tonystark/Desktop/decipher/outputs/propagation_model_summary.csv), [outputs/propagation_regime_summary.csv](/home/tonystark/Desktop/decipher/outputs/propagation_regime_summary.csv), [outputs/propagation_thresholds.csv](/home/tonystark/Desktop/decipher/outputs/propagation_thresholds.csv), [outputs/propagation_case_studies.md](/home/tonystark/Desktop/decipher/outputs/propagation_case_studies.md)

7. The Historical Newspapers grouped benchmark is trustworthy enough for a bounded grouped-recognition claim.
Evidence: [outputs/real_data_trust_summary.csv](/home/tonystark/Desktop/decipher/outputs/real_data_trust_summary.csv), [docs/REAL_DATA_TRUST_HARDENING.md](/home/tonystark/Desktop/decipher/docs/REAL_DATA_TRUST_HARDENING.md)

## Explicit Non-Claims

- We do not claim semantic decipherment.
- We do not claim plaintext recovery.
- We do not claim universal downstream gains.
- We do not claim a universal propagation law.
- We do not claim that conformal is always better than raw uncertainty.
- We do not claim that the stronger decoders are the main novelty.

## Wording Softened To Avoid Overclaiming

- “replicated grouped top-k transfer” instead of “replicated structured recovery”
- “selective real downstream gains” instead of “real downstream improvement”
- “support-gated propagation” instead of “predictable propagation law”
- “better-covered real downstream task” instead of “real downstream benchmark solved”
- “bounded empirical paper” instead of “general uncertainty framework”

## Consistency Rule

If a sentence implies exact real downstream improvement across both real corpora, it is too strong and should be weakened.
