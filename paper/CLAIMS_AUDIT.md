# Claims Audit

## Claims Supported On All Three Datasets

- Preserving transcription uncertainty improves symbol-level retention of correct alternatives relative to hard transcript collapse under matched posterior families.
- The four-condition protocol separates the effects of uncertainty retention and posterior calibration cleanly enough to show that they are not interchangeable.
- The direction of the uncertainty effect is consistent across three real handwritten-symbol corpora with different visual statistics.
- Failure analysis exposes recurring `top1_collapse_but_topk_rescue` cases and recurring `uncertainty_helped_symbols_not_downstream` cases on all three datasets.
- Bootstrap intervals, multi-seed summaries, and cross-dataset comparisons can now be reported from completed real-data runs rather than only from synthetic or fixture-scale evidence.

## Claims Supported On A Subset Of Datasets

- Calibration improved symbol-level metrics on Omniglot and Kuzushiji-49, but not on scikit-learn digits.
- Omniglot alone supports grouped dataset characterization through alphabet labels; the other two datasets do not add comparable grouping.
- The uncertainty effect is materially larger on scikit-learn digits and Kuzushiji-49 than on Omniglot, so effect magnitude is not stable even though effect direction is.
- Kuzushiji-49 specifically strengthens the paper's historical relevance by showing the effect on a manuscript-adjacent corpus rather than only on Omniglot and a generic digit dataset.

## Claims Still Unsupported

- Full decipherment of unknown scripts
- Semantic translation or plaintext recovery
- Broad historical generalization
- Universal benefit from calibrated posteriors
- Reliable downstream family-ranking gains across datasets
- Claims that symbol-level rescue consistently improves higher-level inference

## Evidence Required For Stronger Claims

- A sequence-rich historical manuscript or cipher corpus with richer grouped structure than the current three datasets
- Dataset structure with defensible grouped or family labels that align with the implemented downstream hypothesis metrics
- Cross-dataset replication on additional corpora with materially different acquisition and labeling conditions
- Evidence that symbol-level rescue affects grouped, structural, or family-level reasoning rather than only top-k retention
- Comparison against stronger recognizers if the paper intends to make broader modeling claims rather than protocol claims
