# External Validity Gap

## What The Sequence Branch Can Defend Today

The branch now supports three different evidence levels:

1. **Real symbol-level evidence** from the frozen workshop package
2. **Synthetic-from-real grouped evidence** from the Markov and process-family tasks
3. **Strengthened real grouped evidence** from the Historical Newspapers grouped-token benchmark
4. **Replicated real grouped evidence** from a second grouped/token-aligned corpus: ScaDS.AI
5. **A redesigned real downstream structural recovery check** via train-supported n-gram-path decoding on both real grouped corpora
6. **A small robustness check** from a full test-split visual audit over Historical Newspapers
7. **A gold-style adjudicated check** over that same Historical Newspapers test split

## What The New Real Grouped Benchmarks Add

Historical Newspapers closes one part of the gap:

- grouped structure is real
- token alignment is real
- the current confusion-network and decoder stack runs on it directly

What the two real grouped corpora now support:

- symbol/top-k rescue survives onto real grouped benchmarks
- grouped top-k rescue now replicates across two real grouped corpora
- conformal pruning can improve grouped exact match on one strengthened corpus, but that exact-match pattern does not replicate cleanly
- the Historical Newspapers result is stable to a small curated correction pass over the evaluated test split
- the Historical Newspapers result is also stable to a stronger gold-style adjudicated test-split upgrade

What it does **not** yet support:

- gold-token manuscript claims
- real downstream family-identification claims
- broad grouped-sequence generalization
- replicated exact real downstream recovery claims

## What Remains Synthetic-Only

These claims still depend on synthetic-from-real task construction:

- sequence exact-match replication across Omniglot, Digits, and Kuzushiji-49
- downstream process-family identification gains
- family-sensitivity findings such as the strength of `alternating_markov`

The glyph imagery is real, but the grouped structure or family labels are generated in those tasks.

## Why The Gap Still Matters

The gap still matters for three reasons:

- the current replicated claim is strongest for grouped top-k, not grouped exact match
- Historical Newspapers still depends on OCR-derived labels plus an in-session gold-style upgrade rather than an independent multi-annotator gold campaign
- the redesigned real downstream n-gram-path task improves coverage materially, but the downstream gains are still selective:
  - Historical Newspapers is improved but still mixed
  - ScaDS.AI shows one positive raw uncertainty exact setting, not a clean replicated pattern
  - the new propagation analysis explains part of this limit, but it does not erase it

## Strongest Honest Boundary

The strongest evidence-bound framing is now:

> structured uncertainty clearly helps on real symbol-level tasks, helps on synthetic-from-real grouped tasks, and now shows replicated grouped top-k transfer across two real grouped/token-aligned corpora. A redesigned real downstream structural test is now available and better covered, but its gains are still selective and do not yet yield a replicated exact downstream recovery claim.

The strongest explanatory extension is:

> propagation failure is not random noise. It is partly support-gated and becomes more likely when rescue never reaches grouped top-k or when grouped gains remain too small to support exact downstream recovery.

## What Would Change The Paper Most

One dataset or target would still change the branch most:

- a real grouped manuscript or cipher-like corpus
- token- or symbol-aligned visual units
- gold or manually validated labels
- enough grouped structure or repeated transcript patterns to measure exact downstream recovery with cleaner semantics than local n-gram support
