# Next Paper Plan

## Research Question

Can structured transcription uncertainty, when decoded with explicit sequence constraints, improve higher-level inference under ambiguity in low-resource glyph and manuscript-like settings?

## Difference From The Current Workshop Paper

The workshop paper is symbol-level and externally grounded on three real datasets.

The next paper should test whether uncertainty becomes more useful once it is:

1. represented structurally rather than as flat top-k rows
2. decoded jointly with sequence constraints
3. evaluated on sequence-level tasks where downstream utility is measurable
4. controlled with principled uncertainty sets rather than only raw calibration

## Branch Hypothesis

The main method hypothesis is:

> Sequence constraints can convert symbol-level ambiguity rescue into measurable sequence-level recovery gains.

The accompanying reliability hypothesis is:

> Risk-controlled prediction sets can make these gains safer and easier to interpret than raw posterior truncation alone.

## Minimal Serious Baseline

The first publishable baseline in this branch is:

- real glyph crops as the visual source
- synthetic Markov-style sequences built from those crops
- confusion-network uncertainty representation
- bigram transition model with beam search
- split conformal prediction sets for coverage-aware pruning

## Evidence Needed For A Stronger Paper

- ambiguity-sweep results showing where structured decoding helps most
- sequence-level gains beyond plain symbol top-k rescue
- failure analysis showing when structural decoding still fails
- calibration versus conformal comparisons under the same benchmark
- comparisons across modern and historically grounded glyph corpora

## Immediate Deliverables

- branch scaffold and configs
- benchmark implementation
- first structured decoder baseline
- first risk-control baseline
- sequence-level metrics and tests
