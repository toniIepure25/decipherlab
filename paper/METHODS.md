# Methods

## System Overview

DecipherLab separates image evidence, transcription uncertainty, structural diagnostics, and downstream hypothesis scoring into distinct modules. The implemented study uses the existing glyph-crop protocol rather than full page layout or semantic translation.

## Data Interface

The primary real-data input is a split-aware glyph-crop manifest with one record per crop. Each record specifies `sequence_id`, `position`, `image_path`, and `split`, with optional `transcription`, `family`, `group_id`, and metadata fields. Dataset-specific preparation is kept outside the core pipeline.

The current paper uses three manifest-backed corpora:

1. Omniglot, converted to a deterministic within-character `12/4/4` train/val/test split because the official background/evaluation partition holds out whole character classes.
2. scikit-learn digits, converted from the bundled `load_digits` corpus into crop PNGs and split deterministically with `100` train and `30` validation examples per class, leaving the remainder for test.
3. Kuzushiji-49, downloaded through OpenML in full and converted into a deterministic balanced-cap manifest with `300/75/75` train/val/test examples per class so the frozen multi-seed paper pack remains tractable while preserving all `49` classes.

The third dataset was chosen to strengthen the paper with a more historically grounded corpus than scikit-learn digits without changing the crop-level protocol.

## Posterior Conditions

We compare four controlled conditions:

1. Fixed transcript + heuristic posterior
2. Fixed transcript + calibrated posterior
3. Uncertainty-aware + heuristic posterior
4. Uncertainty-aware + calibrated posterior

The heuristic posterior is a prototype or distance-based baseline. The calibrated posterior is a PCA-plus-classifier baseline with temperature scaling. Both are explicitly baseline methods rather than full OCR/HTR recognizers.

## Uncertainty Representation

The uncertainty-aware path preserves a top-k posterior over candidate glyph identities at each position. The fixed path collapses the posterior to the top-1 candidate before downstream structural analysis. This isolates the effect of preserving alternatives under otherwise matched conditions.

## Structural And Downstream Analysis

Downstream modules compute entropy-based diagnostics, index of coincidence, repeat statistics, compression-based signals, and family-level heuristic rankings. These outputs are treated as evidence summaries rather than proofs of decipherment.

In practice, the current paper remains symbol-level because none of the three external datasets supplies decipherment-family labels aligned with the implemented hypothesis families. Omniglot provides alphabet grouping, which is useful for dataset characterization, but not for family-accuracy claims. scikit-learn digits and Kuzushiji-49 provide no natural grouping beyond class identity in the current manifests.

## Statistical Reporting

The framework reports bootstrap confidence intervals on the main symbol-level metrics and supports multi-seed sweeps without changing the train/validation/evaluation protocol. Failure analysis remains a first-class output so that gains in symbol recovery can be distinguished from gains in downstream reasoning.

Cross-dataset synthesis is built from completed results packs rather than rerunning or reshaping the evaluation code. This keeps the three-dataset comparison evidence-bound: each dataset is first evaluated independently under the frozen paper pack, then summarized only from the saved tables, intervals, and failure reports.
