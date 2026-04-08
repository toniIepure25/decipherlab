# Final Claims Check

## 1. Exact Central Supported Claim

Preserving transcription uncertainty improves symbol-level retention of correct alternatives under ambiguous observations relative to hard transcript collapse, and this effect replicates across three real handwritten-symbol datasets.

## 2. Secondary Supported Claims

- The direction of the uncertainty-rescue effect is positive on Omniglot, scikit-learn digits, and Kuzushiji-49.
- The effect magnitude is dataset-dependent: smallest on Omniglot, largest on scikit-learn digits, and intermediate on Kuzushiji-49.
- Calibration is dataset-dependent rather than universally helpful: it helps on Omniglot and Kuzushiji-49, but not on scikit-learn digits.
- The recurring `top1_collapse_but_topk_rescue` pattern appears on all three datasets.
- The current framework can report bootstrap intervals, seed summaries, and failure analyses from completed real-data runs rather than only from synthetic or fixture-scale experiments.

## 3. Explicit Non-Claims

- This paper does not claim full decipherment of unknown scripts.
- This paper does not claim semantic translation or plaintext recovery.
- This paper does not claim broad historical generalization.
- This paper does not claim reliable downstream family-level gains.
- This paper does not claim that calibration is universally beneficial.
- This paper does not claim that the system solves undeciphered manuscripts.

## 4. Wording Softened During Final Editing

- Softened from “improves downstream decipherment-related reasoning” to “improves symbol-level retention of correct alternatives” because downstream-family evidence is still weak.
- Softened from “historical generalization” to “historically grounded corpus” because Kuzushiji-49 strengthens relevance but does not justify broad generalization.
- Softened from “calibration helps” to “calibration remains inconsistent across datasets” because the digits run is a clear negative case.
- Softened from “manuscript evidence” to “manuscript-adjacent evidence” because the current Kuzushiji-49 manifest remains crop-level and not sequence-rich.
