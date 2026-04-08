# Dataset Selection

## Goal

Select one real labeled crop dataset that best fits the current DecipherLab thesis:

> Preserving transcription uncertainty can improve symbol-level retention of correct alternatives under ambiguous observations relative to hard transcript collapse.

The selection had to satisfy four constraints at once:

- genuine relevance to ambiguous symbol or script recognition
- crop-level labels suitable for symbol recovery
- compatibility with the existing manifest-backed 2x2 protocol
- total local storage at or below `40 GB`

## Candidates Considered

| Dataset | Why It Was Considered | Strengths | Weaknesses |
| --- | --- | --- | --- |
| Omniglot | Real handwritten glyph crops from many alphabets; already aligned with the current crop protocol | Highly script-like, per-image PNG crops, alphabet grouping, easy manifest conversion, tiny storage footprint | Not historical manuscript material; original official split is not label-preserving for the current symbol-recovery task |
| Kuzushiji-49 | Historical Japanese cursive character crops and a stronger historical-script match | Real historical script, larger sample count than Omniglot, good symbol labels | Requires array-to-image conversion for the current pipeline, lacks natural alphabet-style grouping, and would add adapter churn during a methods-frozen phase |
| EMNIST ByClass | Large real handwritten symbol dataset with straightforward classification labels | Very large sample count, clean train/test structure, easy symbol-level evaluation | Less script-like, weak connection to unknown-script framing, and no natural higher-level grouping |

## Chosen Dataset

**Omniglot** remains the best fit for this phase.

It wins on the combination of script relevance, clean crop-level format, group metadata, low integration risk, and exact compatibility with the already-frozen fixed-vs-uncertainty protocol. Kuzushiji-49 is an important next candidate, but switching to it in this phase would spend effort on a new adapter rather than strengthening the evidence pack around the existing thesis.

## Why Omniglot Won

- It is the most script-like dataset that fits the current manifest workflow with essentially no protocol changes.
- It already provides one glyph crop per file, which minimizes conversion errors.
- It preserves meaningful higher-level grouping through alphabet identity.
- It supports symbol-level supervised evaluation directly through character labels.
- It fits comfortably under the `40 GB` storage cap even after extraction and manifest generation.

## Dataset Accounting

- Dataset name: `Omniglot`
- Source: `https://github.com/brendenlake/omniglot`
- Approximate official archive size: `15,927,098` bytes across `images_background.zip` and `images_evaluation.zip`
- Downloaded local size: `167M` under `data/raw/omniglot/`
- Download mode: **full dataset**
- Subset strategy: **none**

The local footprint is well below the project’s `40 GB` maximum dataset budget, so the full dataset is retained rather than a capped subset.

## Necessary Split Adjustment

The official Omniglot `images_background` versus `images_evaluation` partition holds out whole character classes. That split is valid for one-shot learning, but it is not valid for this paper’s label-preserving symbol-recovery protocol because train and test would not share symbol identities.

We therefore use a deterministic within-character split:

- `12` samples per class for train
- `4` samples per class for validation
- `4` samples per class for test

This keeps the core evaluation question intact: does preserving uncertainty help when the correct symbol remains one plausible alternative under visual ambiguity?

## Known Limitations

- Omniglot is a multi-script handwriting corpus, not a historical decipherment benchmark.
- Sequences are single-glyph crops, so grouped and structural downstream metrics remain limited.
- The resulting evidence should be interpreted as symbol-level ambiguity robustness, not semantic decipherment.
