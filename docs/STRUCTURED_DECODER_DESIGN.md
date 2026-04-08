# Structured Decoder Design

## Version 1 Decoder

The first structured decoder is intentionally simple and auditable:

- uncertainty representation: confusion network
- structural model: smoothed bigram transition model
- search algorithm: beam search

This gives a serious baseline without introducing opaque modeling assumptions.

## Visual-Structural Factorization

For a candidate path `y_1 ... y_T`, version 1 scores:

`score = sum_t log p_visual(y_t | x_t) + lambda * sum_t log p_transition(y_t | y_{t-1})`

where:

- `p_visual` comes from the existing posterior model
- `p_transition` is estimated from synthetic training sequences
- `lambda` is the structural weight

## Why This Is The Right First Step

- it directly tests whether structural decoding can exploit preserved alternatives
- it is easy to ablate against hard collapse
- it can be upgraded later to CRF or WFST-style decoding without changing the benchmark framing

## Risk-Control Path

The current risk-control baseline uses split conformal prediction sets:

- fit on validation posteriors
- derive a probability threshold with empirical coverage semantics
- prune each confusion-network position to a prediction set
- decode over the filtered graph

This is still approximate because the visual posteriors themselves are approximate, but the set-construction rule is explicit and auditable.

## Planned Extensions

- sequence-conditioned posterior reweighting
- higher-order transition models
- CRF-style decoding
- WFST-style rescoring
- iterative EM-like refinement between visual beliefs and structural consistency
