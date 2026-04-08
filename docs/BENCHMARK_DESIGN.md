# Benchmark Design

## Design Principle

The next-paper branch keeps real-data and synthetic-task claims separate.

- Real-data claims:
  - visual glyph recognition under ambiguity
  - source corpora remain Omniglot, scikit-learn Digits, Kuzushiji-49, and future manifest-backed corpora
- Synthetic downstream claims:
  - structured sequence reconstruction from real glyph crops
  - sequence-level decoding under explicit transition constraints

## Benchmark Added In This Branch

### Task: Real-Glyph Markov Sequences

The current branch adds a synthetic-from-real benchmark:

1. start from a manifest-backed real glyph-crop dataset
2. select labels that have sufficient support in train, val, and test splits
3. sample synthetic sequences from a configurable Markov transition process
4. instantiate each symbol in the sequence with a real glyph crop drawn from the corresponding split
5. inject ambiguity only at evaluation time
6. compare fixed collapse, raw uncertainty decoding, and conformal-filtered decoding

## Why This Benchmark Exists

The existing workshop package shows symbol-level rescue, but real external datasets do not yet provide honest downstream decipherment-family targets.

This benchmark creates a controlled sequence task where:

- the visual side still comes from real glyph corpora
- the structural side is explicit and reproducible
- higher-level utility of uncertainty can actually be measured

## Metrics

The sequence branch reports:

- symbol top-1
- symbol top-k
- symbol NLL
- symbol ECE
- sequence exact match
- sequence token accuracy
- sequence top-k recovery
- sequence CER-like edit rate
- prediction-set coverage
- prediction-set average size
- prediction-set rescue rate

## Current Limitations

- the transition process is synthetic, not read from real manuscripts
- no semantic content is implied
- downstream claims in this branch are therefore synthetic-from-real, not historical-semantic claims
