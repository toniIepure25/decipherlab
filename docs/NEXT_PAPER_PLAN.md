# Next Paper Plan

## Updated Research Question

Can structured transcription uncertainty, when decoded with explicit sequence constraints, improve grouped recovery under ambiguity beyond symbol-level top-k rescue, and does any part of that advantage survive on real grouped data?

The paper is now better framed as a propagation question:

- under what measurable support conditions does symbol rescue propagate to grouped rescue and then to downstream recovery?

## Current Empirical State

The branch now has three evidence layers:

1. real symbol-level results from the frozen workshop package
2. synthetic-from-real grouped results on three real glyph corpora
3. one strengthened real grouped/token-aligned benchmark from historical newspapers
4. one second real grouped/token-aligned benchmark from ScaDS.AI
5. one real downstream structural task built from train-derived transcript banks on those corpora
6. one small full-test-split visual audit over Historical Newspapers
7. one gold-style adjudicated upgrade of that same Historical Newspapers test split

## Strongest Supported Pattern

The cleanest cross-dataset synthetic pattern remains:

- with calibrated posteriors, `uncertainty_beam` improves mean sequence exact match over `fixed_greedy` on Omniglot, Digits, and Kuzushiji-49
- sequence top-k gains are more stable than exact-match gains
- the strongest synthetic grouped gains remain on Kuzushiji-49

## Strongest New Real Grouped Pattern

The branch now has two real grouped transfer checks:

- Historical Newspapers: grouped top-k rescue is positive, symbol top-k rescue is positive, raw exact-match gain is negative on average, and conformal gives the clearest exact-match gain.
- ScaDS.AI: grouped top-k rescue is again positive, symbol top-k rescue is again positive, and raw exact-match gain becomes positive under `cluster_distance`, but conformal exact-match gains do not carry over cleanly.

This is materially stronger than a one-corpus grouped paper because grouped top-k rescue now replicates across two real grouped corpora.

## Real Downstream Structural Result

The branch now also has one real downstream structural test:

- the original exact transcript-bank task was coverage-limited and mostly negative
- the redesigned train-supported n-gram-path task is much better covered on both corpora
- positive real downstream gains now appear, but selectively:
  - ScaDS.AI shows a positive raw uncertainty exact downstream gain under `cluster_distance`
  - Historical Newspapers shows the clearest conformal downstream rescue
- across both corpora, exact downstream recovery still does not replicate cleanly

This is stronger than the original negative downstream boundary, but still not a clean positive replicated real downstream result.

## Validation Outcome

The first robustness check on the real grouped benchmark is now complete:

- full test split audited: `30` sequences / `126` tokens
- corrected tokens: `2`
- token error rate: `0.016`
- grouped metrics unchanged after correction

So the current grouped finding looks stable to a small curated audit, but not yet upgraded to a gold-label result.

## Gold-Style Outcome

The benchmark now also has a gold-style strengthened slice:

- full test split upgraded to pass-A / pass-B adjudicated labels
- pass agreement: `1.000`
- OCR-to-gold token error rate: `0.016`
- grouped metrics unchanged relative to the OCR-derived and audited runs

This makes the Historical Newspapers result more trustworthy, but not sufficient by itself.

## Best Current Claim

The strongest evidence-bound claim now available is:

> Structured uncertainty helps reliably at symbol level on real data, yields modest grouped gains on synthetic-from-real sequence tasks, and now shows replicated grouped top-k transfer across two real grouped/token-aligned corpora, with mixed exact-match behavior.

The strongest explanatory extension is:

> Rescue propagation is support-gated. Symbol rescue strongly predicts grouped rescue, but grouped rescue becomes real downstream success only in specific support regimes rather than under a universal rule.

## What Still Blocks A Stronger Submission Tier

- exact-match behavior is still mixed across the two real grouped corpora
- the redesigned real downstream task still gives mixed exact gains rather than a replicated positive effect
- the propagation regimes are informative but still partly corpus-specific
- the Historical Newspapers gold-style upgrade is still a repeated in-session review rather than an independent multi-annotator gold dataset
- real downstream family/process evidence is still missing
- the stronger decoders remain selective rather than universally helpful

## Highest-Leverage Next Step

The best next step is no longer another decoder. It is one of:

1. add a real grouped downstream target with cleaner semantics than local n-gram support and non-trivial coverage
2. replace the current Historical Newspapers gold-style test split with an independent multi-annotator gold annotation pass

Either step would directly test whether the current grouped transfer story can move from a strong boundary result to a stronger real downstream paper.

If no additional data step is feasible, the branch is now strong enough for a bounded paper organized around:

1. replicated real grouped top-k transfer
2. a better-covered but still mixed real downstream test
3. a support-aware explanation of why rescue propagates in some regimes and fails in others
