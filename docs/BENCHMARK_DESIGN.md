# Benchmark Design

## Design Principle

The sequence branch keeps real-data and synthetic-task claims separate.

- Real-data component:
  - glyph images come from real manifest-backed corpora
- Synthetic-task component:
  - sequence structure and downstream objectives are generated

This lets the branch study higher-level utility of uncertainty without implying semantic decipherment.

## Benchmarks Currently Implemented

### 1. `real_glyph_markov_sequences`

This is the baseline sequence-reconstruction benchmark.

Protocol:

1. select symbol classes with sufficient support in train, val, and test
2. sample synthetic sequences from a configurable grouped Markov process
3. instantiate each symbol with a real glyph crop from the corresponding split
4. inject ambiguity only at evaluation time
5. compare:
   - `fixed_greedy`
   - `uncertainty_beam`
   - `conformal_beam`
   - `uncertainty_trigram_beam`
   - `conformal_trigram_beam`
   - `uncertainty_crf_viterbi`
   - `conformal_crf_viterbi`

Supported metrics:

- sequence exact match
- sequence top-k recovery
- token accuracy
- CER-like sequence error
- symbol-level top-k, NLL, ECE
- coverage, set size, singleton rate, rescue rate

### 2. `real_glyph_process_family_sequences`

This is the first decipherment-style synthetic downstream task.

Families currently implemented:

- `sticky_markov`
- `alternating_markov`
- `motif_repeat`

The downstream objective is family/rule identification from decoded sequences.

Scientifically, this acts as a controlled analogue of structural family inference rather than plaintext recovery.

## What The Benchmarks Now Show

### Markov Reconstruction

- calibrated `uncertainty_beam` yields modest but replicated sequence-level gains across all three datasets
- the strongest exact-match gains appear on Kuzushiji-49
- sequence top-k gains are more stable than exact-match gains
- trigram decoding is not a universal improvement on this task
- CRF-style exact decoding is mostly a null result on this task:
  - mean CRF exact-match deltas are zero across the current cross-dataset pack
  - CRF top-k deltas are usually negative or neutral

### Process-Family Identification

- downstream family-identification gains now appear on Omniglot, Digits, and Kuzushiji-49 under at least one posterior setting
- the strongest downstream gains are on Kuzushiji-49
- the Digits calibrated setting remains mixed, so the downstream result is replicated but bounded
- trigram decoding helps most clearly on Kuzushiji-49 with calibrated posteriors and is weaker or mixed elsewhere
- CRF-style decoding contributes only tiny family-level deltas in a few settings and does not currently change the core downstream story

## What Is Still Synthetic

- all sequence structure
- all process-family labels
- all downstream family/rule targets

The only real-data component is the glyph imagery and its train/val/test split discipline.

## Current Limitations

- sequence structure is synthetic rather than manuscript-native
- no real decipherment-family labels are present
- no real grouped sequence corpus is integrated in the current branch
- the strongest downstream gains are still task- and decoder-sensitive
- failure cases where symbol rescue does not propagate remain common
