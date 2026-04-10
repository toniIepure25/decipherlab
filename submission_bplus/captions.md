# Captions

## Main Figure

**Figure 1. Propagation regimes on the real downstream task.**
Real downstream rescue rates separated by method family, support regime, and entropy regime. The figure shows the paper’s core explanatory result: raw uncertainty and conformal help in different parts of the support space, and no single monotonic support rule explains propagation across both real grouped corpora. The caption should state clearly that this figure explains the mixed downstream result rather than claiming universal downstream improvement.

## Main Table

**Table 1. Real-to-synthetic evidence bridge.**
Summary of the strongest evidence layers in the branch: fully real symbol-level results, synthetic-from-real grouped/downstream results, replicated real grouped transfer, and the redesigned real downstream structural task. The table shows which claims are fully real, which remain synthetic-from-real, and where exact downstream gains remain selective rather than replicated. This is the main reviewer-facing boundary table.

## Secondary Figure

**Figure 2. Real grouped replication across Historical Newspapers and ScaDS.AI.**
Comparison of grouped exact and grouped top-k deltas across the two real grouped/token-aligned corpora. The figure shows that grouped top-k rescue replicates across both corpora even though exact grouped gains remain mixed. The text should pair this figure with the Historical trust-hardening result.

## Secondary Table

**Table 2. Statistical robustness summary.**
Bootstrap intervals, effect directions, and positive-probability summaries for the main real and synthetic claims. This table shows that the replicated real grouped top-k result is statistically stable, while the real downstream exact result remains selective and partly sign-indefinite under raw uncertainty.

## Appendix Artifact

**Appendix Figure/Table A1. Real downstream redesign and coverage boundary.**
Coverage and performance summary for the redesigned `train_supported_ngram_path` task. This artifact documents that the coverage problem was materially reduced, but that downstream exact gains remained selective. It answers the reviewer concern that the downstream result is negative only because of trivial train/test mismatch.
