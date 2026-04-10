# Propagation Framework

## Purpose

This framework defines how DecipherLab measures whether uncertainty rescue travels from local symbol retention to higher inference levels.

The goal is explanatory, not architectural. We keep the existing decoder stack fixed and ask when preserved uncertainty becomes useful at grouped or downstream levels.

## Level 1: Symbol Rescue

Symbol rescue means the uncertainty-aware method preserves the correct local alternative when the hard-collapse baseline does not.

Core indicators:

- symbol top-k inclusion
- symbol top-k delta relative to the paired baseline
- posterior entropy
- top-1 margin
- prediction-set size
- conformal set size and singleton rate

Level-1 success is present when:

- `symbol_rescue = 1`, or
- local top-k retention improves relative to the paired baseline

## Level 2: Grouped Rescue

Grouped rescue means local uncertainty preservation produces a measurable grouped-sequence advantage.

Core indicators:

- grouped top-k recovery
- grouped exact recovery
- grouped token accuracy
- grouped token delta relative to the paired baseline
- grouped plausibility support from train-derived resources when available

Level-2 success is present when:

- `grouped_topk_rescue = 1`, or
- grouped exact recovery improves relative to the paired baseline

## Level 3: Downstream Recovery

Downstream recovery means grouped uncertainty becomes useful for a higher structural target.

In this branch there are two downstream settings:

- synthetic-from-real family identification
- real grouped `train_supported_ngram_path` recovery

Core indicators:

- downstream exact success
- downstream partial success when defined
- downstream exact delta relative to the paired baseline
- downstream support coverage and upper bound

Level-3 success is present when:

- `downstream_exact_rescue = 1`, or
- a positive paired downstream exact or partial delta appears

## What Counts As Propagation

Propagation is directional.

Level 1 -> Level 2:

- a symbol-rescue event produces grouped top-k rescue, grouped exact rescue, or a positive grouped token delta

Level 2 -> Level 3:

- a grouped-rescue event produces downstream exact or partial rescue on the evaluation task

Propagation is therefore stronger than rescue alone. A system can preserve local alternatives without converting them into grouped or downstream success.

## Candidate Gating Factors

This branch treats propagation as support-gated rather than decoder-gated.

Measured gating factors:

- ambiguity level
- posterior entropy
- top-1 margin
- prediction-set size
- conformal set size
- sequence length
- corpus identity
- decoder family
- posterior family
- train-support coverage
- support upper bound
- grouped top-k delta
- bigram-support availability on the redesigned real downstream task

## What A Strong Propagation Result Would Look Like

A strong propagation result would show all three of the following:

- symbol rescue strongly predicts grouped rescue across datasets
- grouped rescue predicts downstream success under measurable support conditions
- those support conditions are stable enough to define transferable regimes rather than one-off corpus anecdotes

## What Failure To Propagate Means

Failure to propagate does not mean preserved uncertainty is useless.

It means one of two things:

- the preserved alternatives are locally correct but structurally insufficient for the higher task
- the higher task lacks enough support, coverage, or constraint quality to convert local rescue into exact recovery

Scientifically, this branch now treats failure-to-propagate as an empirical limit that should be measured and explained, not hidden behind stronger decoders.
