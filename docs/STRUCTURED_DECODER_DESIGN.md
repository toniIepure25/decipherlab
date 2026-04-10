# Structured Decoder Design

## Decoder Families In The Current Evidence Pack

The branch now compares three deliberately simple and auditable structural decoders:

1. bigram beam decoder
2. trigram beam decoder
3. CRF-style exact Viterbi decoder

Both operate over the same confusion-network uncertainty representation.

## Scoring Decomposition

### Bigram Beam

`sum_t log p_visual(y_t | x_t) + lambda * sum_t log p_bigram(y_t | y_{t-1})`

### Trigram Beam

`sum_t log p_visual(y_t | x_t) + lambda * sum_t log p_trigram(y_t | y_{t-2}, y_{t-1})`

For short prefixes, the trigram decoder falls back to explicit first- and second-position priors. All probabilities come from smoothed counts fit on the synthetic-from-real training sequences.

### CRF-Style Exact Decoder

`sum_t log p_visual(y_t | x_t) + lambda * sum_t log p_bigram(y_t | y_{t-1})`

This decoder uses the same unary and pairwise factors as the bigram beam baseline, but solves the sequence exactly with dynamic programming rather than approximate beam search. In the current branch it is an auditable CRF-style baseline, not a separately trained CRF.

## Why These Baselines Matter

They are the smallest decoder family that can test the main structural hypothesis:

> preserved alternatives become more useful when decoded jointly with explicit sequence constraints.

The trigram decoder is intentionally modest. It is meant to test whether a stronger but still interpretable structural prior turns uncertainty rescue into better higher-level recovery.

The CRF-style decoder tests a different question:

> if the factorization stays simple, is exact sequence inference enough to convert preserved symbol uncertainty into stronger higher-level gains?

## Measured Results

### Markov Reconstruction

- bigram `uncertainty_beam` remains the stronger baseline for the main cross-dataset sequence claim
- trigram decoding is usually neutral or negative for mean sequence exact match on this task
- this is strong evidence that simply increasing decoder order does not automatically improve structural recovery
- CRF-style exact decoding is also mostly null on this task:
  - mean CRF exact-match deltas are `0.0` across the current cross-dataset summary
  - CRF sequence top-k deltas are neutral or negative
- this suggests the current branch is limited more by factor quality than by approximate search alone

### Process-Family Identification

- trigram decoding becomes more useful on the downstream task than on plain reconstruction
- the clearest positive setting is Kuzushiji-49 with calibrated posteriors:
  - mean trigram family delta over `uncertainty_beam`: `+0.094`
- smaller positive trigram family deltas also appear on:
  - Omniglot with cluster-distance posteriors
  - Digits with calibrated posteriors

This supports a narrower decoder claim:

> a slightly stronger structural prior can help downstream structural family identification in selected settings, even when it does not improve basic sequence reconstruction.

CRF-style decoding is weaker here:

- CRF family deltas are small and inconsistent across datasets
- the largest conformal-over-CRF family lift is still modest
- exact inference with the current factors does not reproduce the clearest trigram downstream win

## Risk-Control Component

The branch uses split conformal prediction sets to prune each confusion-network position to a coverage-aware candidate set.

Current measured behavior:

- on the Markov benchmark, conformal mostly reduces coverage and set size relative to raw `uncertainty_beam`
- it rarely improves sequence exact match there
- on the process-family task, conformal can improve downstream family identification in selected settings, but not consistently

So conformal is currently best justified as a reliability and downstream-analysis component, not as the main decoder improvement.

## Recurring Failure Modes

- `symbol_rescue_without_sequence_rescue`
- `decoder_trapped_by_transition_prior`
- `high_ambiguity_diffuse_uncertainty`
- `family_rescue_without_sequence_rescue`
- `trigram_amplifies_bad_prior`
- `crf_amplifies_bad_prior`
- `conformal_over_pruning`

## Current Design Conclusion

The bigram decoder is still the most stable mainline method. The trigram decoder is scientifically useful because it exposes a real structural question and yields a few genuine downstream gains. The CRF-style decoder is scientifically useful as a negative control: it shows that exact inference over the current simple factors is not enough by itself.
