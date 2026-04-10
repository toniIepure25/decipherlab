# Sequence Results Notes

## Scope

These notes belong to the sequence-focused branch, not the frozen workshop paper package.

The branch now has five evidence layers:

1. real symbol-level evidence from the frozen workshop package
2. synthetic-from-real grouped sequence evidence
3. strengthened real grouped evidence from Historical Newspapers
4. replicated real grouped evidence from a second corpus: ScaDS.AI
5. one real downstream structural recovery check built from the two grouped corpora

It now also has one explanatory layer:

6. support-aware propagation analysis across symbol, grouped, and downstream levels

## Synthetic-From-Real Task Packs

### Markov Reconstruction

Cross-dataset pack:

- [summary CSV](/home/tonystark/Desktop/decipher/outputs/sequence_cross_dataset_summary.csv)
- [summary Markdown](/home/tonystark/Desktop/decipher/outputs/sequence_cross_dataset_summary.md)
- [effects plot](/home/tonystark/Desktop/decipher/outputs/sequence_cross_dataset_effects_plot.png)
- [aggregate table](/home/tonystark/Desktop/decipher/outputs/sequence_cross_dataset_tables.csv)
- [failure summary](/home/tonystark/Desktop/decipher/outputs/sequence_cross_dataset_failure_summary.csv)
- [ambiguity regime table](/home/tonystark/Desktop/decipher/outputs/sequence_cross_dataset_ambiguity_regime_table.csv)

Strongest pattern:

- with calibrated posteriors, `uncertainty_beam` improves mean sequence exact match over `fixed_greedy` on Omniglot, Digits, and Kuzushiji-49
- sequence top-k gains are more stable than exact-match gains
- the strongest synthetic grouped gains remain on Kuzushiji-49

### Process-Family Identification

Cross-dataset pack:

- [aggregate CSV](/home/tonystark/Desktop/decipher/outputs/sequence_process_family_cross_dataset.csv)
- [summary Markdown](/home/tonystark/Desktop/decipher/outputs/sequence_process_family_cross_dataset.md)
- [effects plot](/home/tonystark/Desktop/decipher/outputs/sequence_process_family_cross_dataset_effects_plot.png)
- [aggregate table](/home/tonystark/Desktop/decipher/outputs/sequence_process_family_cross_dataset_tables.csv)
- [failure summary](/home/tonystark/Desktop/decipher/outputs/sequence_process_family_cross_dataset_failure_summary.csv)
- [ambiguity regime table](/home/tonystark/Desktop/decipher/outputs/sequence_process_family_cross_dataset_ambiguity_regime_table.csv)

Supported synthetic pattern:

- downstream family-identification gains appear on all three datasets under at least one posterior setting
- the strongest and cleanest downstream gains remain on Kuzushiji-49
- the gains are selective rather than family-universal

Family-sensitivity outputs:

- [summary CSV](/home/tonystark/Desktop/decipher/outputs/sequence_family_sensitivity_summary.csv)
- [summary Markdown](/home/tonystark/Desktop/decipher/outputs/sequence_family_sensitivity_summary.md)

Measured pattern:

- `alternating_markov` carries the clearest positive family-identification gains
- `motif_repeat` is usually the hardest case and is often neutral or negative

## Real Grouped Benchmarks

Historical Newspapers integration:

- [grouped manifest README](/home/tonystark/Desktop/decipher/data/processed/historical_newspapers_grouped_words/README.md)
- [manifest summary](/home/tonystark/Desktop/decipher/data/processed/historical_newspapers_grouped_words/manifest.summary.md)
- [dataset plan](/home/tonystark/Desktop/decipher/docs/REAL_GROUPED_DATA_PLAN.md)

Real grouped pack:

- [summary CSV](/home/tonystark/Desktop/decipher/outputs/real_grouped_historical_newspapers/sequence_cross_dataset_summary.csv)
- [summary Markdown](/home/tonystark/Desktop/decipher/outputs/real_grouped_historical_newspapers/sequence_cross_dataset_summary.md)
- [effects plot](/home/tonystark/Desktop/decipher/outputs/real_grouped_historical_newspapers/sequence_cross_dataset_effects_plot.png)
- [aggregate table](/home/tonystark/Desktop/decipher/outputs/real_grouped_historical_newspapers/sequence_cross_dataset_tables.csv)
- [failure summary](/home/tonystark/Desktop/decipher/outputs/real_grouped_historical_newspapers/sequence_cross_dataset_failure_summary.csv)
- [ambiguity regime table](/home/tonystark/Desktop/decipher/outputs/real_grouped_historical_newspapers/sequence_cross_dataset_ambiguity_regime_table.csv)
- [robustness summary](/home/tonystark/Desktop/decipher/outputs/real_grouped_robustness_summary.md)
- [strengthened summary](/home/tonystark/Desktop/decipher/outputs/real_grouped_strengthened_summary.md)

What Historical Newspapers shows:

- raw `uncertainty_beam` improves grouped top-k recovery on average
- raw `uncertainty_beam` improves symbol top-k recovery on average inside the grouped benchmark
- raw `uncertainty_beam` does not improve grouped exact match on average
- conformal pruning provides the clearest preliminary grouped exact-match gain

This supports a strengthened one-corpus grouped transfer claim, not a gold-token manuscript claim.

Second real grouped corpus:

- [search note](/home/tonystark/Desktop/decipher/docs/SECOND_REAL_GROUPED_CORPUS_SEARCH.md)
- [grouped manifest README](/home/tonystark/Desktop/decipher/data/processed/scadsai_grouped_words/README.md)
- [manifest summary](/home/tonystark/Desktop/decipher/data/processed/scadsai_grouped_words/manifest.summary.md)
- [summary CSV](/home/tonystark/Desktop/decipher/outputs/real_grouped_scadsai/sequence_cross_dataset_summary.csv)
- [summary Markdown](/home/tonystark/Desktop/decipher/outputs/real_grouped_scadsai/sequence_cross_dataset_summary.md)
- [effects plot](/home/tonystark/Desktop/decipher/outputs/real_grouped_scadsai/sequence_cross_dataset_effects_plot.png)
- [aggregate table](/home/tonystark/Desktop/decipher/outputs/real_grouped_scadsai/sequence_cross_dataset_tables.csv)
- [failure summary](/home/tonystark/Desktop/decipher/outputs/real_grouped_scadsai/sequence_cross_dataset_failure_summary.csv)
- [ambiguity regime table](/home/tonystark/Desktop/decipher/outputs/real_grouped_scadsai/sequence_cross_dataset_ambiguity_regime_table.csv)
- [replication summary](/home/tonystark/Desktop/decipher/outputs/real_grouped_replication_summary.md)
- [bridge summary](/home/tonystark/Desktop/decipher/outputs/real_vs_synthetic_bridge_summary.md)

What ScaDS.AI shows:

- raw `uncertainty_beam` improves grouped exact match under `cluster_distance`
- raw `uncertainty_beam` improves grouped top-k strongly under both posterior families
- symbol top-k rescue is substantially larger than on Historical Newspapers
- conformal exact-match gain does not replicate cleanly on this corpus

## Real Grouped Validation And Gold-Style Check

Validation artifacts:

- [decision note](/home/tonystark/Desktop/decipher/docs/REAL_GROUPED_NEXT_STEP_DECISION.md)
- [strengthening decision](/home/tonystark/Desktop/decipher/docs/REAL_GROUPED_STRENGTHENING_DECISION.md)
- [validation protocol](/home/tonystark/Desktop/decipher/docs/HISTORICAL_NEWSPAPERS_VALIDATION_PROTOCOL.md)
- [gold protocol](/home/tonystark/Desktop/decipher/docs/HISTORICAL_NEWSPAPERS_GOLD_PROTOCOL.md)
- [validation subset README](/home/tonystark/Desktop/decipher/data/processed/historical_newspapers_grouped_words/validation_subset_README.md)
- [label-noise summary](/home/tonystark/Desktop/decipher/data/processed/historical_newspapers_grouped_words/validation_label_noise_summary.md)
- [gold agreement summary](/home/tonystark/Desktop/decipher/data/processed/historical_newspapers_grouped_words/gold_agreement_summary.md)

Measured validation result:

- full `test` split audited: `30` sequences / `126` tokens
- corrected tokens: `2`
- corrected sequences: `1`
- token error rate: `0.016`
- sequence error rate: `0.033`

Robustness result:

- the grouped metrics were unchanged after applying the audited corrections
- raw uncertainty still improves grouped top-k and symbol top-k
- conformal still provides the clearest grouped exact-match gain

This makes the current real grouped result more trustworthy, but still not equivalent to a gold independent annotation study.

Gold-style result:

- full test split promoted to a gold-style adjudicated subset
- pass-A / pass-B agreement: `1.000`
- OCR-to-gold token error rate: `0.016`
- OCR-to-gold sequence error rate: `0.033`
- original, audited, and gold-style grouped metrics were identical in the strengthened comparison pack

Interpretation:

- the current Historical Newspapers result is now stronger than a one-pass OCR-derived benchmark
- it is now complemented by a second real grouped corpus
- but the two-corpus replication is strongest for grouped top-k rescue, not for conformal exact-match gains

## Decoder Findings

### Trigram Decoder

- trigram decoding is not a general win on the Markov benchmark
- it remains a selective structural aid rather than a universal improvement

### CRF-Style Decoder

- the CRF-style exact decoder remains mostly a null result
- exact inference over the current factors is not enough by itself to unlock larger grouped gains

## Ambiguity Regimes

Synthetic tasks:

- Omniglot and Digits benefit mostly at low ambiguity
- Kuzushiji-49 retains stronger gains into medium ambiguity

Real grouped task:

- raw uncertainty beam is mixed to negative for grouped exact match across most regimes
- conformal pruning gives the clearest grouped exact-match improvement across low, medium, and high ambiguity
- raw uncertainty still improves grouped top-k most clearly in higher ambiguity for the noisier cluster-distance posterior
- on ScaDS.AI, grouped top-k rescue is stronger overall and raw exact-match gains appear only for the noisier `cluster_distance` posterior family

## External Validity Boundary

The branch now has replicated real grouped recovery evidence, but the strongest higher-level results are still synthetic-from-real.

Supported on real grouped data:

- grouped top-k transfer
- symbol top-k transfer inside grouped sequences
- robustness of the Historical Newspapers findings to a small full-test-split visual audit
- robustness of the Historical Newspapers findings to a gold-style adjudicated test split
- grouped top-k replication across Historical Newspapers and ScaDS.AI

Still synthetic-only:

- process-family downstream claims
- family-sensitivity claims
- the cross-dataset sequence exact-match replication story

Still unsupported:

- gold-token manuscript or cipher grouped claims
- real grouped downstream family/process claims
- semantic decipherment or plaintext recovery

## Real Downstream Structural Recovery

Task design:

- [task design note](/home/tonystark/Desktop/decipher/docs/REAL_DOWNSTREAM_TASK_DESIGN.md)
- [task redesign note](/home/tonystark/Desktop/decipher/docs/REAL_DOWNSTREAM_TASK_REDESIGN.md)
- [summary CSV](/home/tonystark/Desktop/decipher/outputs/real_grouped_downstream_summary.csv)
- [summary Markdown](/home/tonystark/Desktop/decipher/outputs/real_grouped_downstream_summary.md)
- [effects plot](/home/tonystark/Desktop/decipher/outputs/real_grouped_downstream_plot.png)
- [coverage analysis CSV](/home/tonystark/Desktop/decipher/outputs/real_downstream_coverage_analysis.csv)
- [coverage analysis Markdown](/home/tonystark/Desktop/decipher/outputs/real_downstream_coverage_analysis.md)
- [redesigned summary CSV](/home/tonystark/Desktop/decipher/outputs/real_downstream_redesigned_summary.csv)
- [redesigned summary Markdown](/home/tonystark/Desktop/decipher/outputs/real_downstream_redesigned_summary.md)
- [redesigned plot](/home/tonystark/Desktop/decipher/outputs/real_downstream_redesigned_plot.png)

Task definition:

- derive a transcript bank from the `train` split only
- restrict candidate transcripts to the same transcript length as the evaluation example
- rerank those real train transcripts with the existing uncertainty-aware decoder stack

Measured downstream pattern:

- Historical Newspapers has zero exact transcript-bank coverage on the evaluated test split, so exact downstream recovery is impossible under this task.
- ScaDS.AI has low but non-zero transcript-bank coverage (`0.111` mean across the two posterior settings).
- Raw `uncertainty_beam` does not improve exact downstream recovery on either real corpus.
- The only positive exact downstream gain is a small selective conformal gain on ScaDS.AI with `calibrated_classifier` (`+0.056`).
- Grouped top-k rescue therefore persists more clearly than exact real transcript recovery.

Interpretation:

- the branch now has a real downstream structural task, not only grouped recovery
- that task sharpens the boundary more than it expands the claim
- the current real grouped corpora do not yet show replicated exact downstream recovery from preserved uncertainty

Redesigned downstream result:

- the exact transcript-bank task was replaced by a better-covered train-supported n-gram-path task
- the redesigned task raises the exact-recovery upper bound materially:
  - Historical Newspapers: `0.000 -> 0.667`
  - ScaDS.AI: `0.111 -> 1.000`
- positive real downstream gains now appear, but selectively rather than cleanly:
  - ScaDS.AI / `cluster_distance`: raw uncertainty exact downstream delta `+0.111`
  - Historical Newspapers / `cluster_distance`: conformal exact downstream delta `+0.222` over raw uncertainty
- the two-corpus average exact downstream delta for raw uncertainty remains slightly negative, so the real downstream claim is still bounded

## Propagation Analysis

Propagation artifacts:

- [framework note](/home/tonystark/Desktop/decipher/docs/PROPAGATION_FRAMEWORK.md)
- [results note](/home/tonystark/Desktop/decipher/docs/PROPAGATION_RESULTS_NOTES.md)
- [feature summary](/home/tonystark/Desktop/decipher/outputs/propagation_features.md)
- [model summary](/home/tonystark/Desktop/decipher/outputs/propagation_model_summary.md)
- [threshold summary](/home/tonystark/Desktop/decipher/outputs/propagation_thresholds.md)
- [regime summary](/home/tonystark/Desktop/decipher/outputs/propagation_regime_summary.md)
- [regime plot](/home/tonystark/Desktop/decipher/outputs/propagation_regime_plot.png)
- [cross-dataset synthesis](/home/tonystark/Desktop/decipher/outputs/propagation_cross_dataset_summary.md)

Strongest explanatory pattern:

- symbol rescue is the clearest predictor of grouped rescue
- grouped rescue is necessary but not sufficient for downstream success
- grouped top-k improvement is the strongest positive downstream predictor on the current real downstream task
- conformal helps mainly in specific support regimes rather than universally

Threshold and regime result:

- grouped rescue falls off sharply once entropy becomes too diffuse
- downstream rescue becomes much more likely once grouped top-k delta reaches at least one recovered alternative
- no single monotonic support law explains both real grouped corpora

Interpretation:

- the branch is now stronger because it explains the limit, not only because it measures it
- the current best framing is support-aware uncertainty propagation under ambiguity
