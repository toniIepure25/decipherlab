# Tertiary Dataset Selection

## Goal

Add one more historically grounded or manuscript-adjacent corpus to the frozen DecipherLab paper pack so the cross-dataset story extends beyond Omniglot and scikit-learn digits.

The selection criteria were:

- more historically grounded or manuscript-like than scikit-learn digits
- compatible with the existing glyph-crop manifest workflow
- visually and statistically distinct from both Omniglot and digits
- feasible to integrate without architecture churn
- small enough to stay well within the `40 GB` storage budget

## Candidates Considered

| Dataset | Why It Was Considered | Strengths | Weaknesses |
| --- | --- | --- | --- |
| Kuzushiji-49 | Preferred historical corpus for this phase | Real historical Japanese characters, 49 classes, much closer to manuscript conditions than digits, accessible through OpenML in this environment | Strong class imbalance, no native document/group labels, full-manifest evaluation would be heavier than needed for the frozen multi-seed paper pack |
| Kuzushiji-MNIST | Practical fallback if Kuzushiji-49 proved too heavy | Same historical grounding, easier download and smaller footprint | Only 10 classes, materially weaker than Kuzushiji-49 for a meaningful third-dataset comparison |
| EMNIST ByClass | Large real handwritten character corpus | Easy symbol-level framing, strong label coverage | Not historically grounded, closer to digits than to manuscript or cipher conditions, weaker scientific complement to Omniglot |

## Chosen Dataset

**Kuzushiji-49** won.

It is the strongest scientifically aligned third corpus that was actually accessible in this environment. OpenML provided a reliable acquisition path, which let us integrate the preferred historical dataset without changing the frozen evaluation protocol.

## Why It Strengthens The Paper

- It is more historically grounded than scikit-learn digits.
- It complements Omniglot with a real historical character corpus rather than another generic handwritten-symbol set.
- It tests whether the symbol-level uncertainty-rescue effect survives on a manuscript-adjacent dataset with different stroke statistics, class imbalance, and label structure.

## Dataset Accounting

- Dataset name: `Kuzushiji-49`
- Source: `OpenML` dataset `41991`
- OpenML URL: `https://www.openml.org/d/41991`
- Approximate full downloaded size: `100113985` bytes (`Kuzushiji-49.arff.gz` in the OpenML cache)
- Downloaded size in this environment: `100113985` bytes
- Download mode: **full dataset download**
- Evaluation mode: **balanced subset manifest preserving all 49 classes**

## Evaluation Subset Strategy

The full dataset was downloaded, but the experiment manifest uses a deterministic balanced cap:

- train: `300` examples per class
- val: `75` examples per class
- test: `75` examples per class

This yields `22050` labeled crops while preserving all `49` classes. The subset is not arbitrary; it is designed to:

- keep every class in the experiment
- control Kuzushiji-49's strong class imbalance
- keep the frozen multi-seed 2x2 paper pack tractable
- make the resulting corpus comparable in scale to Omniglot rather than overwhelmingly larger

## Known Limitations

- The evaluation manifest is a balanced subset, not a full-manifest sweep.
- The dataset provides no document-level grouping or decipherment-family labels aligned with the current downstream metrics.
- The strongest expected evidence remains symbol-level rather than family-level or semantic.
