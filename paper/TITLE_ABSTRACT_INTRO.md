# Title, Abstract, And Introduction

## Working Title

Preserving Transcription Uncertainty Improves Symbol-Level Reasoning Under Ambiguous Glyph Observations

## Abstract

Decipherment pipelines for rare scripts, cipher alphabets, and damaged artifacts often collapse uncertain visual evidence into a single transcript too early. This can discard correct alternatives before structural diagnostics or downstream reasoning have a chance to use them. We present DecipherLab, a research-grade framework for uncertainty-aware decipherment experiments that preserves top-k transcription posteriors, compares fixed versus uncertainty-aware inference, and reports explicit failure cases rather than only positive averages. The implemented evaluation protocol crosses fixed versus uncertainty-aware inference with heuristic versus calibrated posterior generation, and pairs this four-condition comparison with bootstrap confidence intervals, multi-seed summaries, and manifest-backed real glyph-crop datasets.

The current external evidence pack covers three real handwritten-symbol corpora with different visual statistics: Omniglot, scikit-learn digits, and Kuzushiji-49. Across all three datasets, preserving uncertainty improves symbol top-k retention relative to hard transcript collapse under matched posterior families. The effect size is dataset-dependent: it is smallest on Omniglot, largest on scikit-learn digits, and intermediate on the historically grounded Kuzushiji-49 corpus. Calibration remains inconsistent across datasets: it helps on Omniglot and Kuzushiji-49, but hurts on scikit-learn digits. We therefore support a narrow claim only: preserving transcription uncertainty improves symbol-level retention of correct alternatives under ambiguous observations relative to hard transcript collapse, and this effect replicates across three real handwritten-symbol datasets. We do not claim full decipherment, semantic recovery, or broad historical generalization.

## Introduction

Unknown-script analysis and historical cipher work are often bottlenecked by early commitment. In many practical settings, the evidence pipeline is effectively `image -> transcript -> structural analysis`, and the transcript step is treated as if it were known. This is risky when glyph inventories are ambiguous, crops are degraded, or multiple symbol identities remain plausible.

The core hypothesis of this paper is narrow and testable: preserving transcription uncertainty can improve decipherment-related reasoning under ambiguous observations. We operationalize this claim at the symbol and structural level rather than at the level of semantic translation. The goal is not to present a universal decipherment system, but to build and evaluate an auditable inference stack that carries ambiguity forward long enough to measure whether it matters.

DecipherLab implements this evaluation philosophy as a modular framework. It supports manifest-backed real crop datasets, calibrated and heuristic posterior generators, structural triage metrics, family-level hypothesis scoring, bootstrap uncertainty estimates, multi-seed robustness sweeps, and explicit failure analysis. The present study now spans three real handwritten-symbol corpora:

- Omniglot: `32,460` labeled glyph crops, `1,623` character classes, `50` alphabet groups
- Kuzushiji-49 balanced subset: `22,050` labeled glyph crops, `49` historical character classes, no native grouping in the current manifest
- scikit-learn digits: `1,797` labeled digit crops, `10` classes, no native grouping

These datasets are deliberately different. Omniglot is multi-script and visually heterogeneous; Kuzushiji-49 is historically grounded and manuscript-adjacent; scikit-learn digits is low-resolution, single-domain, and numerically constrained. This lets us ask a precise question: does the symbol-level uncertainty effect persist across corpora with different class structure and visual statistics, including one more historical corpus? The answer from the current evidence is yes in direction, but not in magnitude or calibration behavior. That narrower conclusion is the basis of the paper.
