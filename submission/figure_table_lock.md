# Figure And Table Lock

## Main Figure

- File: `submission/figures/fig1_cross_dataset_effects_plot.png`
- Role: main paper
- Essential: yes
- Why included: it shows the core empirical result directly, namely the replicated uncertainty-rescue effect across all three datasets under ambiguity

## Main Table

- File: `submission/tables/table1_cross_dataset_with_ci.csv`
- Role: main paper
- Essential: yes
- Why included: it provides the full condition-wise 2x2 numeric comparison with confidence intervals and preserves calibration/NLL context alongside top-k accuracy

## Secondary Figure

- File: `submission/figures/fig2_kuzushiji49_symbol_topk.png`
- Role: main paper if space permits, otherwise appendix
- Essential: no
- Why included: it highlights the historically grounded corpus and makes the manuscript-adjacent result concrete rather than leaving it only in the cross-dataset aggregate

## Appendix Artifact

- File: `submission/tables/tableA1_cross_dataset_failure_summary.csv`
- Role: appendix
- Essential: yes for trustworthiness, no for headline narrative
- Why included: it documents the recurring failure modes that keep the paper's claims narrow and prevents overinterpretation of the positive averages
