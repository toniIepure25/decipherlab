# Figure And Table Plan

## Main Figure

- Filename: [cross_dataset_effects_plot.png](/home/tonystark/Desktop/decipher/outputs/cross_dataset_effects_plot.png)
- What it shows: cross-dataset uncertainty-effect curves over ambiguity for Omniglot, scikit-learn digits, and Kuzushiji-49 under the matched posterior families
- Why it is central: it visualizes the paper's main result directly, which is the direction and magnitude of symbol top-k rescue under ambiguity across all three real datasets
- Placement: main paper

## Main Table

- Filename: [cross_dataset_tables_with_ci.csv](/home/tonystark/Desktop/decipher/outputs/cross_dataset_tables_with_ci.csv)
- What it shows: per-dataset condition-wise top-k, NLL, and ECE values with confidence intervals for the full 2x2 comparison
- Why it is central: it provides the most paper-ready numerical view of the frozen protocol across all three datasets and makes the calibration inconsistency visible alongside the replicated uncertainty effect
- Placement: main paper

## Secondary Figure

- Filename: [comparison_symbol_topk.png](/home/tonystark/Desktop/decipher/outputs/runs/20260407T201933Z_kuzushiji49_balanced_subset_paper_pack_evaluation/comparison_symbol_topk.png)
- What it shows: Kuzushiji-49 condition-wise symbol top-k comparison with uncertainty bands
- Why it is central: it anchors the paper in the historically grounded dataset and shows that the positive effect survives on a manuscript-adjacent corpus, not only on Omniglot or the simpler digits dataset
- Placement: main paper if space permits, otherwise appendix

## Failure-Analysis Artifact

- Filename: [cross_dataset_failure_summary.csv](/home/tonystark/Desktop/decipher/outputs/cross_dataset_failure_summary.csv)
- What it shows: recurring failure categories across all three datasets, including top-1 collapse with top-k rescue and calibration instability
- Why it is central: it keeps the paper honest by showing where uncertainty helps only at symbol level and where calibration remains unstable
- Placement: appendix
