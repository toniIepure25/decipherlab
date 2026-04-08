# Appendix

## A. Locked Artifact Inventory

- Main figure: [figures/fig1_cross_dataset_effects_plot.png](figures/fig1_cross_dataset_effects_plot.png)
- Main table source: [tables/table1_cross_dataset_with_ci.csv](tables/table1_cross_dataset_with_ci.csv)
- Secondary figure: [figures/fig2_kuzushiji49_symbol_topk.png](figures/fig2_kuzushiji49_symbol_topk.png)
- Failure-analysis appendix artifact: [tables/tableA1_cross_dataset_failure_summary.csv](tables/tableA1_cross_dataset_failure_summary.csv)

## B. Dataset Notes

- Omniglot: deterministic within-character split, `32,460` crops, `1,623` classes, `50` alphabet groups.
- scikit-learn digits: deterministic per-class cap split, `1,797` crops, `10` classes, no grouping.
- Kuzushiji-49: full OpenML download with a balanced-cap evaluation manifest, `22,050` crops, `49` classes, no grouping in the current manifest.

The dataset preparation notes are maintained in:

- [docs/DATASET_SELECTION.md](/home/tonystark/Desktop/decipher/docs/DATASET_SELECTION.md)
- [docs/DATASET_SELECTION_SECONDARY.md](/home/tonystark/Desktop/decipher/docs/DATASET_SELECTION_SECONDARY.md)
- [docs/DATASET_SELECTION_TERTIARY.md](/home/tonystark/Desktop/decipher/docs/DATASET_SELECTION_TERTIARY.md)
- [docs/DATASET_PREPARATION.md](/home/tonystark/Desktop/decipher/docs/DATASET_PREPARATION.md)

## C. Reproducibility Notes

- Omniglot and Kuzushiji-49 runs use a two-seed sweep `{23, 29}`.
- The scikit-learn digits run uses a three-seed sweep `{23, 29, 31}` from the already completed evidence pack.
- Bootstrap confidence intervals use `500` trials and `0.95` confidence.
- Cross-dataset synthesis is built from completed results packs rather than rerunning the evaluation logic.

The key run directories are:

- `outputs/runs/20260407T150327Z_omniglot_paper_pack_evaluation`
- `outputs/runs/20260407T191829Z_sklearn_digits_paper_pack_evaluation`
- `outputs/runs/20260407T201933Z_kuzushiji49_balanced_subset_paper_pack_evaluation`

## D. Failure Analysis Notes

The appendix failure artifact preserves the same pattern across all three datasets:

- `top1_collapse_but_topk_rescue` recurs on Omniglot, scikit-learn digits, and Kuzushiji-49.
- `uncertainty_helped_symbols_not_downstream` recurs on all three datasets, which is why the paper remains symbol-level rather than making downstream decipherment claims.
- `calibration_worsened_or_unstable` also recurs across datasets, which is why calibration is described as inconsistent rather than universally beneficial.

## E. Transfer Notes

This appendix is designed to move directly into supplementary material or a workshop appendix section. The artifact links above already point to the copied files under `submission/`.
