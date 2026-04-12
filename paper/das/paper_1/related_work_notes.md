# Related Work Notes

## Purpose

This note records the literature-positioning choices for the DAS manuscript so the related-work section stays concise and evidence-bound.

## Literature blocks now covered in the paper

1. Uncertainty and calibration

- `guo2017calibration`
- `angelopoulos2021conformal`

These references support the paper's claim that local confidence quality and set-valued prediction matter, while also clarifying that our contribution is not another calibration paper.

2. Structured decoding in sequence recognition

- `graves2006ctc`
- `scheidl2018wordbeam`

These references anchor the manuscript in the sequence-decoding tradition where multiple local alternatives are preserved so higher-level constraints can act.

3. Handwriting and document-recognition context

- `plamondon2000handwriting`

This survey reference gives the manuscript a field-facing bridge to handwriting recognition without turning the section into a survey.

4. Benchmark grounding

- `lake2015omniglot`
- `pedregosa2011scikit`
- `clanuwat2018kuzushiji`
- `historical_newspapers_gt`
- `scadsai_grouped_words`

These references clarify which corpora are standard published resources and which are public dataset records used through DecipherLab manifests.

## Deliberate exclusions

- No broad decipherment literature was added because the paper does not make decipherment claims.
- No large OCR/HTR survey block was added beyond the handwriting-recognition survey because the paper is not a survey and DAS page economy is tight.
- No extra conformal references were added beyond the tutorial-style grounding because the manuscript uses conformal prediction as a bounded component, not as its primary theoretical contribution.
