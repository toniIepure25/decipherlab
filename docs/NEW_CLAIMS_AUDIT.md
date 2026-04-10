# New Claims Audit

## Supported On Real Glyph Corpora At Symbol Level

These claims come from the frozen workshop paper package:

- Preserving transcription uncertainty improves symbol-level retention of correct alternatives under ambiguity relative to hard transcript collapse.
- That symbol-level effect replicates across Omniglot, scikit-learn Digits, and Kuzushiji-49.

## Supported On Synthetic-From-Real Grouped Tasks

### Markov Sequence Benchmark

- Structured uncertainty can produce modest sequence-level gains beyond hard collapse on synthetic-from-real benchmarks.
- With calibrated posteriors, `uncertainty_beam` improves mean sequence exact match over `fixed_greedy` on Omniglot, Digits, and Kuzushiji-49.
- Sequence top-k recovery gains are more stable than exact-match gains.

### Downstream Process-Family Benchmark

- Preserved uncertainty can improve a downstream synthetic structural objective beyond plain sequence reconstruction.
- Downstream family-identification gains appear on all three datasets under at least one posterior setting.
- The strongest downstream gains remain on Kuzushiji-49.
- These gains are selective rather than universal:
  - Digits with calibrated posteriors remains mixed
  - `alternating_markov` benefits more than `motif_repeat`

### Decoder Family Findings

- A higher-order trigram decoder can improve downstream family identification in selected settings.
- A CRF-style exact decoder is mainly a negative result:
  - exact inference over the current simple factors does not materially improve grouped recovery

## Supported On One Strengthened Real Grouped Corpus

These claims come from the Historical Newspapers grouped-token benchmark after the audit and gold-style strengthening pass:

- Preserved uncertainty still improves grouped top-k recovery.
- Preserved uncertainty still improves symbol top-k recovery inside grouped sequences.
- Raw `uncertainty_beam` does **not** improve grouped exact match on average.
- Conformal pruning provides the clearest grouped exact-match gain on this corpus, especially with the noisier `cluster_distance` posterior family.
- The grouped metrics were unchanged across OCR-derived, audited, and gold-style labels.

## Supported Across Two Real Grouped Corpora

These claims now come from Historical Newspapers plus ScaDS.AI:

- Grouped top-k rescue replicates across two real grouped/token-aligned corpora.
- Symbol top-k rescue also remains positive on both real grouped corpora.
- Raw grouped exact-match gains do **not** replicate cleanly across the two corpora.
- Conformal exact-match gains are mixed across the two corpora rather than cleanly replicated.

## Supported On A Real Downstream Structural Task

These claims now come from the redesigned `train_supported_ngram_path` benchmark built from the two real grouped corpora:

- A better-covered real downstream structural task can be defined from the existing grouped corpora using train-derived ordered n-gram paths rather than exact train-transcript overlap.
- The negative result on exact train-transcript-bank recovery was partly a coverage artifact:
  - the redesigned task raises the approximate exact upper bound from `0.000` to `0.667` on Historical Newspapers and from `0.111` to `1.000` on ScaDS.AI.
- Real downstream gains do appear under the redesigned task, but only selectively:
  - ScaDS.AI with `cluster_distance` shows a positive raw uncertainty exact downstream delta (`+0.111`)
  - Historical Newspapers shows the clearest conformal downstream rescue (`+0.222` over raw uncertainty under `cluster_distance`)
- The strongest supported real downstream claim is still bounded:
  - better coverage reveals selective real structural gains, but exact downstream recovery does **not** yet replicate cleanly across the two corpora.

## Supported As An Explanatory Propagation Claim

These claims now come from the propagation analysis over synthetic-from-real and real grouped tasks:

- Symbol rescue is the clearest predictor of grouped rescue.
- Grouped rescue is a stronger prerequisite for downstream success than raw symbol rescue alone.
- Better support helps, but propagation does **not** obey one simple monotonic rule across corpora.
- Conformal is best understood as support-regime-specific:
  - it helps most in limited-support, low-entropy regions on the current real downstream task
  - it is not a universal exact-recovery improvement
- The branch now supports a stronger explanatory claim:
  - uncertainty rescue propagates only under measurable support conditions, and failure to propagate is itself a stable empirical finding.

## Partially Supported Claims

- There is an ambiguity-regime effect:
  - Omniglot and Digits benefit mainly at low ambiguity
  - Kuzushiji-49 retains stronger synthetic grouped gains into medium ambiguity
  - on the real newspapers benchmark, conformal helps grouped exact match more reliably than raw uncertainty beam across the ambiguity sweep
- Conformal prediction sets can help grouped recovery in selected settings, but they are not a universal win.
- The higher-order decoder may be genuinely useful for downstream synthetic structural objectives, but it is not consistently helpful across datasets and tasks.

## Unsupported Claims

- semantic decipherment
- plaintext recovery
- broad historical generalization
- universal calibration superiority
- universal conformal superiority
- universal higher-order decoder superiority
- a universal propagation law
- a compact propagation index with stable cross-corpus validity
- replicated exact real downstream transcript recovery
- real downstream family-identification gains
- gold-token manuscript or cipher grouped claims

## Evidence Required For A Stronger-Tier Publication

- one real grouped manuscript or cipher-like corpus with token- or symbol-aligned visual units
- independent gold or manually validated grouped labels
- enough grouped supervision or repeated transcript structure to measure exact downstream recovery with non-trivial coverage and cleaner semantics than local n-gram support
- ideally one real downstream grouped target beyond grouped exact/top-k recovery

## Main Remaining Gaps

- the strongest higher-level gains are still synthetic-from-real
- the redesigned real downstream structural task is better-covered but still mixed rather than replicated
- real grouped replication now exists, but only across two corpora and only for grouped recovery rather than downstream structure
- downstream family/process claims still do not transfer to real grouped data
- `symbol_rescue_without_sequence_rescue` remains frequent
