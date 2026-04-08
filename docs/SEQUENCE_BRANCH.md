# Sequence Branch

## Purpose

This branch is intentionally separate from the frozen workshop paper package.

The workshop paper supports a symbol-level claim:

> Preserving transcription uncertainty improves symbol-level retention of correct alternatives under ambiguity relative to hard transcript collapse.

The sequence branch asks a different question:

> Can structured transcription uncertainty, coupled with explicit sequence constraints, improve higher-level inference under ambiguity?

## What Is New Here

- synthetic-from-real sequence tasks built from real glyph corpora
- structured uncertainty represented as confusion networks rather than only flat top-k lists
- a first structural decoder that combines visual evidence with a learned bigram prior
- a risk-control path based on split conformal prediction sets

## What Is Not Changing

- the frozen workshop submission package under `paper/` and `submission/`
- the existing 2x2 symbol-level paper pack
- the narrow non-semantic framing of the project

## Current Branch Scope

Version 1 of the sequence branch contains:

- `src/decipherlab/sequence/benchmark.py`
  - builds reproducible synthetic sequence tasks from real manifest-backed glyph crops
- `src/decipherlab/structured_uncertainty/confusion_network.py`
  - converts posterior rows into serializable confusion networks
- `src/decipherlab/decoding/beam_search.py`
  - applies a bigram transition model with beam search
- `src/decipherlab/risk_control/conformal.py`
  - builds split-conformal prediction sets and reports coverage/set-size behavior
- `src/decipherlab/sequence/runner.py`
  - runs the branch end to end from config

## Success Criteria For This Branch

The new branch is successful when it can support a stronger, still narrow paper claim such as:

- structured uncertainty plus sequence constraints improves sequence-level recovery under ambiguity on synthetic-from-real glyph benchmarks
- risk-controlled candidate sets provide better or more stable downstream behavior than raw truncation alone

These claims must remain separate from any semantic-decipherment claim.
