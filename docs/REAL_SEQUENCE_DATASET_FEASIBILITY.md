# Real Sequence Dataset Feasibility

## Goal

The narrow question for this phase was:

> Can the current sequence branch add one real grouped or sequence-rich corpus without turning the project into a segmentation or full HTR rewrite?

## Feasibility Threshold

A dataset counted as feasible only if it provided:

- real grouped sequence structure
- token- or symbol-level visual units
- labels aligned to those units
- compatibility with the current manifest-backed structured-uncertainty workflow
- modest integration cost

## Candidates Considered

### Historical Newspapers Ground Truth

Outcome: feasible and selected.

Why:

- public page images plus ALTO OCR token boxes
- real line grouping
- word-level token coordinates that can be cropped directly into the existing glyph-crop manifest format
- no segmentation rewrite required

Main caveat:

- labels are OCR-derived rather than manually verified

### TextBite

Outcome: promising but not selected for this phase.

Why:

- stronger layout/document structure story than the chosen dataset
- less direct token-aligned fit to the current decoder stack
- slower and riskier integration path in this environment

### ICDAR2019 HDRC

Outcome: infeasible for this phase.

Why:

- grouped line structure is real and relevant
- accessible supervision is line-level rather than token-aligned
- integrating it cleanly would require segmentation/alignment work outside scope

### M5HisDoc

Outcome: infeasible for this phase.

Why:

- scientifically attractive historical manuscript corpus
- not immediately reproducible to integrate in the current environment and phase budget

## Feasibility Decision

This phase successfully integrated a real grouped/token-aligned public dataset:

- `Historical Newspapers Ground Truth` (Zenodo record 2583866)

The integration is deliberately narrow:

- grouped structure is real
- token alignment is real
- labels are OCR-derived

So the branch now has **preliminary real grouped evidence**, but not a gold-token manuscript result.

## What This Changed

The branch is no longer blocked by a total absence of real grouped data.

It can now test:

- whether symbol-level rescue transfers to grouped top-k behavior on real data
- whether structured decoding or conformal pruning helps grouped exact recovery at all on a real grouped benchmark

## What Still Remains Missing

The stronger missing asset is still:

- a real grouped manuscript or cipher-like corpus with gold or manually validated token labels

That is the dataset class that would materially upgrade the branch from preliminary real grouped evidence to a stronger publication result.
