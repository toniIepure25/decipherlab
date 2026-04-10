# Support-Aware Uncertainty Propagation in Low-Resource Glyph and Grouped Sequence Recognition

*A bounded empirical paper on when preserved transcription uncertainty propagates from symbol rescue to grouped and downstream recovery.*

## Abstract

Early collapse of uncertain visual evidence can discard correct alternatives before grouped or downstream constraints have a chance to use them. We study this problem in low-resource glyph and grouped-sequence recognition using DecipherLab, an auditable uncertainty-preserving pipeline built around confusion-network posteriors, constrained decoding, and explicit failure analysis. The evidence spans three synthetic-from-real glyph benchmarks, two real grouped/token-aligned corpora, and a redesigned real downstream task with substantially improved train-support coverage.

Across Omniglot, scikit-learn Digits, and Kuzushiji-49, preserving uncertainty improves symbol-level retention and yields modest grouped gains, with grouped top-k improvements more stable than exact-match improvements. On two real grouped corpora, Historical Newspapers and ScaDS.AI handwriting, grouped top-k rescue and symbol top-k rescue both transfer, establishing a replicated real grouped result. The Historical Newspapers benchmark also survived a full-split audit and a gold-style adjudicated pass with only `2/126` token corrections and no metric drift. However, exact grouped recovery and exact downstream recovery remain mixed: a better-covered real downstream task reveals selective gains, including conformal gains on Historical Newspapers and raw uncertainty gains on ScaDS.AI, but not a clean replicated exact downstream win.

We therefore shift the paper’s contribution from “more decoding” to explanation. A support-aware propagation analysis shows that symbol rescue is the strongest predictor of grouped rescue, while grouped rescue is necessary but not sufficient for downstream success. The grouped-rescue advantage conditioned on symbol rescue is stable, and real downstream rescue becomes much more likely once grouped top-k gain reaches at least one recovered alternative. Propagation depends on support regime rather than a single monotonic law: raw uncertainty helps most in rare high-support, high-entropy regions, whereas conformal helps more in limited-support, low-entropy regions. The strongest evidence-bound claim is therefore narrow: preserved uncertainty reliably improves local and grouped alternative retention, and its higher-level usefulness is gated by measurable support conditions. This makes the paper a bounded empirical study of uncertainty propagation under ambiguity rather than a decipherment or semantic-recovery claim.

## 1. Introduction

Systems that read ambiguous glyphs or grouped handwritten tokens often commit too early. The common pipeline is still effectively `image -> top-1 transcript -> higher-level reasoning`, which means correct alternatives may be discarded before grouped or structural constraints can act on them. That failure mode matters in low-resource settings, where ambiguity is high and labeled grouped data are scarce.

This paper studies a narrow version of that problem. We do not ask whether uncertainty solves decipherment, semantic recovery, or manuscript interpretation. We ask when preserved uncertainty becomes useful at higher levels of inference, and when it does not.

The central claim is:

> Preserved transcription uncertainty improves real grouped top-k recovery across two token-aligned corpora, but higher-level propagation is support-gated: grouped rescue predicts downstream success only in specific support regimes, and exact real downstream gains remain selective rather than replicated.

This is a stronger paper than a decoder-comparison study because it explains a measured limit. The branch now has real grouped replication, a redesigned real downstream task with much better coverage, and an analysis layer that formalizes why rescue often fails to propagate beyond grouped top-k.

## 2. Scope And Positioning

This is a support-aware uncertainty propagation paper for low-resource glyph and grouped-sequence recognition.

It is not:

- a decipherment breakthrough
- a semantic recovery paper
- a universal uncertainty framework
- a general OCR calibration paper

The real evidence is separated from the synthetic-from-real evidence throughout:

1. real symbol-level evidence from the frozen workshop package
2. synthetic-from-real grouped and downstream evidence on Omniglot, Digits, and Kuzushiji-49
3. real grouped evidence on Historical Newspapers and ScaDS.AI
4. a real downstream structural recovery test derived from those grouped corpora
5. a propagation analysis explaining when rescue moves across levels

## 3. Methods

### 3.1 Uncertainty Representation And Decoding

The branch uses explicit confusion-network style uncertainty representations rather than hard top-1 collapse. Decoders consume per-position candidate structure and compare:

- `fixed_greedy`
- `uncertainty_beam`
- `conformal_beam`
- `uncertainty_trigram_beam`
- `conformal_trigram_beam`
- a CRF-style exact decoder

The decoder comparison is part of the evidence base, but the present paper’s main contribution is not a new decoder family. The strongest architectural result is actually modest: the additional decoders are selective tools, not universal improvements.

### 3.2 Propagation Framework

We formalize three levels of rescue:

- Level 1: symbol rescue
- Level 2: grouped rescue
- Level 3: downstream recovery

Propagation is directional:

- symbol rescue -> grouped rescue
- grouped rescue -> downstream success

The framework is defined in [docs/PROPAGATION_FRAMEWORK.md](/home/tonystark/Desktop/decipher/docs/PROPAGATION_FRAMEWORK.md). The goal is to measure when higher-level gains occur under explicit support conditions rather than to claim they should always occur.

### 3.3 Explanatory Features

For each paired evaluation unit, we extract a compact support representation including:

- symbol and grouped rescue indicators
- downstream exact or partial success where defined
- ambiguity level
- posterior entropy
- top-1 margin
- prediction-set size
- conformal set size
- sequence length
- corpus identity
- decoder identity
- posterior family
- train-support coverage
- grouped top-k delta

This keeps the analysis interpretable and auditable.

## 4. Datasets And Tasks

### 4.1 Synthetic-From-Real Tasks

The synthetic branch uses real glyph imagery from:

- Omniglot
- scikit-learn Digits
- Kuzushiji-49

These tasks support grouped and downstream evaluation while remaining clearly synthetic-from-real in structure.

### 4.2 Real Grouped Corpora

The real grouped evidence uses:

- Historical Newspapers grouped word sequences with ALTO token boxes
- ScaDS.AI grouped word sequences

These corpora provide real grouped/token-aligned data and support the existing confusion-network pipeline directly.

The Historical Newspapers benchmark also has a hardened trust chain:

- full evaluated split visual audit: `126` tokens across `30` grouped sequences
- only `2` corrected tokens
- gold-style pass agreement: `1.000`
- zero grouped-metric drift after the audit and gold-style passes

### 4.3 Real Downstream Task

The real downstream structural task is the redesigned `train_supported_ngram_path` benchmark. It replaces exact transcript-bank matching, which suffered from extreme train/test coverage collapse, with a train-supported ordered n-gram-path target derived from training transcripts only.

This redesign materially improves coverage:

- Historical Newspapers full-path upper bound: `0.000 -> 0.667`
- ScaDS.AI full-path upper bound: `0.111 -> 1.000`

The task is still bounded and local in semantics, but it is honest, reproducible, and far better suited for testing real downstream propagation.

## 5. Results

### 5.1 Real Symbol-Level And Synthetic Higher-Level Evidence

The frozen workshop package remains the strongest fully real result at symbol level: preserving uncertainty improves symbol-level retention across Omniglot, Digits, and Kuzushiji-49.

The stronger sequence branch adds two synthetic-from-real findings:

- grouped sequence gains appear across the three glyph corpora, with grouped top-k more stable than exact match
- downstream process-family gains appear selectively, strongest on Kuzushiji-49

These results motivate the grouped and downstream question, but they are not the core real-data claim.

### 5.2 Replicated Real Grouped Transfer

The cleanest new real result is grouped top-k replication:

- Historical Newspapers mean grouped top-k delta: `+0.056`
- ScaDS.AI mean grouped top-k delta: `+0.306`
- two-corpus mean grouped top-k delta: `+0.181`

This establishes that preserved uncertainty does transfer beyond local symbol retention on real grouped/token-aligned corpora. What does not replicate cleanly is exact grouped recovery:

- raw uncertainty exact deltas are mixed across the two corpora
- conformal exact gains remain corpus-dependent

The grouped top-k result is statistically stable under example-level bootstrapping:

- two-corpus mean grouped top-k delta `+0.181`
- bootstrap interval `[+0.122, +0.236]`
- bootstrap probability of a positive mean effect `1.000`

### 5.3 Redesigned Real Downstream Task

The redesigned real downstream task removes the strongest coverage objection, but it does not create a clean replicated exact downstream win.

Selective positive cases do appear:

- ScaDS.AI with `cluster_distance`: raw exact downstream delta `+0.111`
- Historical Newspapers with `cluster_distance`: conformal exact downstream delta `+0.222` over raw uncertainty

Across both real corpora, however:

- mean raw downstream exact delta remains `-0.028`
- mean conformal downstream exact delta is `+0.059`
- raw downstream exact interval crosses zero `[-0.073, +0.017]`
- conformal downstream exact interval remains positive `[+0.024, +0.097]`

The paper therefore supports a bounded real downstream claim only: once coverage is improved, downstream gains become visible, but they remain selective rather than cleanly replicated.

### 5.4 Propagation Analysis

The propagation analysis is the main analytical contribution.

The strongest measured findings are:

- `symbol_rescue` is the clearest grouped-rescue predictor, with odds ratio about `18.4`
- on the real downstream task, `grouped_topk_delta` is the strongest positive downstream predictor, with odds ratio about `6.1`
- grouped rescue is therefore necessary but not sufficient for downstream success
- grouped-rescue rate rises by about `0.085` when symbol rescue is present, with bootstrap interval `[0.072, 0.099]`
- downstream exact-rescue rate rises by about `0.375` when grouped rescue is present, with bootstrap interval `[0.275, 0.475]`

Threshold analysis gives two defensible summary rules:

- the most stable downstream threshold is `grouped_topk_delta >= 1`, which holds across posterior families
- grouped-rescue entropy thresholds are informative but less stable across posterior families

The regimes are informative but not universal:

- raw uncertainty helps most in a rare high-support, high-entropy regime
- conformal helps most in a limited-support, low-entropy regime
- no single monotonic support law explains both real grouped corpora

## 6. Contributions

This paper makes six bounded contributions.

1. It extends the original symbol-level uncertainty result into grouped and downstream evaluation.
2. It establishes replicated real grouped top-k rescue across two real grouped/token-aligned corpora.
3. It builds and evaluates a better-covered real downstream structural task rather than stopping at a coverage-limited negative result.
4. It shows that exact real downstream gains remain selective even after coverage improves.
5. It formalizes a support-aware propagation framework spanning symbol, grouped, and downstream levels.
6. It explains the limit: grouped rescue is a real gain, but downstream propagation is support-gated rather than automatic.

## 7. Limitations

The paper’s main limitation is not hidden. The strongest higher-level positive evidence is still partly synthetic-from-real, and the real downstream structural result remains mixed rather than replicated.

More specifically:

- real grouped top-k rescue replicates, but exact grouped recovery does not
- the redesigned real downstream task has much better coverage, but exact gains remain selective
- conformal helps in some real support regimes, not universally
- Historical Newspapers has much stronger validation than before, but still not an independent multi-annotator gold campaign
- no real downstream family/process target is available yet

These limitations do not negate the paper’s contribution. They define its boundary. The paper’s strongest value is that it now explains why propagation often fails instead of simply reporting that it fails.

## 8. Conclusion

The strongest evidence-bound conclusion is not that uncertainty always improves structured recovery. It is narrower and more useful:

- preserved uncertainty reliably improves real symbol and grouped alternative retention
- grouped rescue can replicate on real grouped corpora
- downstream propagation depends on measurable support conditions and often fails even after grouped rescue occurs

That makes the paper best understood as a support-aware uncertainty propagation study for low-resource glyph and grouped sequence recognition. Its novelty lies in showing not only where preserved uncertainty helps, but also the measurable conditions under which that help does and does not propagate.
