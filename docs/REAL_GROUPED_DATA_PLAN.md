# Real Grouped Data Plan

## Decision Gate

This phase started with a hard choice:

1. integrate one public real grouped/token-aligned dataset if it met the current manifest and confusion-network interface
2. otherwise create a small pilot dataset

The required interface came from [REQUIRED_REAL_DATA_SPEC.md](/home/tonystark/Desktop/decipher/docs/REQUIRED_REAL_DATA_SPEC.md):

- ordered grouped sequences
- token- or symbol-level visual units
- one label per unit
- stable train/val/test split
- no full segmentation or HTR rewrite

## Public Candidates Considered

### Historical Newspapers Ground Truth (Zenodo 2583866)

Why it was feasible:

- public and small enough to integrate quickly
- page images plus ALTO OCR with `TextLine` and `String` token boxes
- token-level coordinates compatible with the current crop manifest workflow
- real grouped structure at the text-line level

Why it won:

- it was the cleanest path to real grouped evidence without broadening the branch
- the existing structured-uncertainty stack could consume it with only dataset-specific preparation code
- it supports a real grouped transfer check immediately, even though the labels are OCR-derived

### TextBite

Why it was promising:

- historically grounded document understanding dataset
- grouped layout structure and realistic page images

Why it was not chosen for this phase:

- larger download and slower practical integration path in this environment
- the available structure is more layout-centric than token-aligned for the current decoder stack
- it would have consumed the phase budget without giving a cleaner grouped token interface than the newspapers corpus

### ICDAR2019 HDRC / Family Records

Why it was promising:

- real historical grouped page and line structure
- manuscript-adjacent and scientifically relevant

Why it was rejected for this phase:

- accessible data exposes line-level transcriptions rather than token-aligned crops
- using it cleanly would require a segmentation/alignment rewrite that exceeds this phase

### M5HisDoc

Why it was promising:

- historically grounded manuscript corpus with richer structure ambitions

Why it was rejected for this phase:

- access and reproducibility constraints made same-phase integration impractical here

## Chosen Path

This phase chose **public integration**, not pilot dataset creation.

Selected dataset:

- `Historical Newspapers Ground Truth` (Zenodo record 2583866)

Why this was the highest-leverage move:

- it breaks the external-validity bottleneck with the smallest serious real grouped dataset integration
- it reuses the existing manifest and confusion-network pipeline directly
- it gives a preliminary but genuinely real grouped evidence point without destabilizing the branch

## Current Scientific Boundary

The result is a **real grouped/token-aligned benchmark with OCR-derived labels**, not a gold manuscript sequence corpus.

That means it can strengthen the paper in one specific way:

- test whether symbol/top-k rescue and structured decoding transfer at all to real grouped sequences

It does **not** yet support:

- gold-token grouped sequence claims
- manuscript-native downstream family claims
- semantic decipherment claims
