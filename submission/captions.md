# Final Captions

## Figure 1

**File:** `figures/fig1_cross_dataset_effects_plot.png`

**Caption:**  
Cross-dataset uncertainty-effect curves over ambiguity for Omniglot, scikit-learn digits, and Kuzushiji-49 under matched posterior families. The direction of the uncertainty-rescue effect is positive on all three datasets, but its magnitude varies substantially by corpus.

## Table 1

**File:** `tables/table1_cross_dataset_with_ci.csv`

**Caption:**  
Condition-wise cross-dataset comparison for the frozen 2x2 protocol. The table reports top-k accuracy, negative log-likelihood, and expected calibration error with confidence intervals for Omniglot, scikit-learn digits, and Kuzushiji-49.

## Figure 2

**File:** `figures/fig2_kuzushiji49_symbol_topk.png`

**Caption:**  
Kuzushiji-49 condition-wise symbol top-k comparison under the frozen 2x2 protocol. This figure anchors the paper in the historically grounded corpus and shows that the symbol-level uncertainty-rescue effect remains positive on a manuscript-adjacent dataset.

## Appendix Artifact

**File:** `tables/tableA1_cross_dataset_failure_summary.csv`

**Caption:**  
Cross-dataset failure summary, including top-1 collapse with top-k rescue, uncertainty-helped-symbols-not-downstream cases, over-diffuse posteriors, and calibration instability. The recurring failure modes justify the paper's conservative symbol-level framing.
