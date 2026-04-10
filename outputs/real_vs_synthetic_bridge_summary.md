# Real vs Synthetic Bridge Summary

## Main Boundary

- Real symbol-level evidence remains the strongest fully real claim in the repository.
- Synthetic-from-real sequence and downstream tasks still carry most of the higher-level structural evidence.
- The branch now has one strengthened OCR-grounded grouped benchmark and one second real grouped handwriting benchmark.

## What Transfers

- Real grouped raw uncertainty exact-match delta: `-0.090`.
- Real grouped raw uncertainty top-k delta: `0.056`.
- Real grouped conformal exact-match delta: `0.139`.
- Interpretation: symbol/top-k rescue transfers to real grouped data more clearly than grouped exact match.

## Two-Corpus Real Grouped Replication

- Historical Newspapers mean grouped top-k delta: `0.056`.
- ScaDS.AI mean grouped top-k delta: `0.306`.
- Two-corpus mean grouped top-k delta: `0.181`.
- Two-corpus mean conformal exact delta: `0.059`.
- Interpretation: grouped top-k rescue now replicates across two real grouped corpora, while conformal exact-match gains remain corpus-dependent.

## Real Downstream Structural Recovery

- Two-corpus mean raw downstream exact delta: `-0.028`.
- Two-corpus mean conformal downstream exact delta: `0.059`.
- Two-corpus mean full-path coverage upper bound: `0.833`.
- Historical Newspapers downstream coverage fraction: `0.878`.
- ScaDS.AI downstream coverage fraction: `1.000`.
- Interpretation: the redesigned real downstream task fixes much of the coverage collapse, but exact downstream gains are still mixed rather than cleanly replicated.

## Gold-Style Check

- Gold-style pass agreement rate: `1.000`.
- Gold-style OCR-to-label error rate: `0.016`.
- Mean change in conformal exact delta after gold-style upgrade: `0.000`.
- Interpretation: the current real grouped result appears stable to a stronger two-pass gold-style review, but it is still not equivalent to an independent multi-annotator gold annotation campaign.

## What Remains Synthetic-Only

- Synthetic process-family downstream family delta: `0.048`.
- Downstream structural family-identification remains synthetic-from-real only.

## Scope Rows

- `real_symbol_level` / `frozen_workshop_symbol_level` / `mean_calibrated_uncertainty_topk_delta`: `0.224`. Fully real symbol-level evidence across Omniglot, Digits, and Kuzushiji-49.
- `synthetic_from_real_sequence` / `real_glyph_markov_sequences` / `mean_calibrated_sequence_exact_delta`: `0.019`. Synthetic-from-real sequence exact-match gain with calibrated posteriors.
- `synthetic_from_real_sequence` / `real_glyph_markov_sequences` / `mean_calibrated_sequence_topk_delta`: `0.073`. Synthetic-from-real grouped top-k recovery is more stable than exact match.
- `synthetic_from_real_downstream` / `real_glyph_process_family_sequences` / `mean_uncertainty_family_delta`: `0.048`. Synthetic-from-real downstream family gains are positive on average but selective.
- `real_grouped_sequence` / `real_grouped_manifest_sequences` / `mean_uncertainty_sequence_exact_delta`: `-0.090`. Raw uncertainty beam does not improve grouped exact match on average on the strengthened Historical Newspapers grouped benchmark.
- `real_grouped_sequence` / `real_grouped_manifest_sequences` / `mean_uncertainty_sequence_topk_delta`: `0.056`. Raw uncertainty does improve grouped top-k recovery on the strengthened Historical Newspapers grouped benchmark.
- `real_grouped_sequence` / `real_grouped_manifest_sequences` / `mean_uncertainty_symbol_topk_delta`: `0.063`. Symbol-level rescue still transfers into the strengthened Historical Newspapers grouped benchmark.
- `real_grouped_sequence` / `real_grouped_manifest_sequences` / `mean_conformal_sequence_exact_delta`: `0.139`. Conformal pruning gives the clearest grouped exact-match gain on the strengthened Historical Newspapers benchmark.
- `real_grouped_replication` / `historical_newspapers_real_grouped_gold` / `mean_uncertainty_sequence_topk_delta`: `0.056`. Historical Newspapers retains positive grouped top-k rescue after the gold-style upgrade.
- `real_grouped_replication` / `scadsai_real_grouped` / `mean_uncertainty_sequence_topk_delta`: `0.306`. ScaDS.AI also shows positive grouped top-k rescue under the unchanged grouped decoder pack.
- `real_grouped_replication` / `two_real_grouped_corpora` / `mean_uncertainty_sequence_topk_delta`: `0.181`. Grouped top-k rescue is positive across both real grouped corpora.
- `real_grouped_replication` / `two_real_grouped_corpora` / `mean_conformal_sequence_exact_delta`: `0.059`. Conformal exact-match gains are mixed across the two real grouped corpora rather than cleanly replicated.
- `real_grouped_downstream` / `historical_newspapers_train_supported_ngram_path` / `mean_downstream_coverage_fraction`: `0.878`. Historical Newspapers now has substantial downstream coverage under the train-supported n-gram-path task.
- `real_grouped_downstream` / `historical_newspapers_train_supported_ngram_path` / `mean_uncertainty_downstream_exact_delta`: `-0.090`. On Historical Newspapers, raw uncertainty still does not improve exact n-gram-path recovery on average despite the better-covered task.
- `real_grouped_downstream` / `scadsai_train_supported_ngram_path` / `mean_downstream_coverage_fraction`: `1.000`. ScaDS.AI has effectively full downstream coverage under the train-supported n-gram-path task.
- `real_grouped_downstream` / `scadsai_train_supported_ngram_path` / `mean_uncertainty_downstream_exact_delta`: `0.035`. ScaDS.AI shows a selective positive raw uncertainty exact downstream gain under the noisier cluster-distance setting.
- `real_grouped_downstream` / `two_real_grouped_corpora_train_supported_ngram_path` / `mean_uncertainty_downstream_exact_delta`: `-0.028`. Across both real grouped corpora, raw uncertainty does not improve exact n-gram-path recovery on average despite the redesigned higher-coverage task.
- `real_grouped_downstream` / `two_real_grouped_corpora_train_supported_ngram_path` / `mean_conformal_downstream_exact_delta`: `0.059`. Across both real grouped corpora, conformal remains the clearest exact downstream rescue mechanism on average, but the gain is still selective rather than replicated.
- `real_grouped_downstream` / `two_real_grouped_corpora_train_supported_ngram_path` / `mean_full_path_coverage_upper_bound`: `0.833`. The redesigned task substantially improves the real downstream upper bound relative to exact transcript-bank overlap.
- `real_grouped_validation` / `historical_newspapers_gold_style_subset` / `gold_pass_agreement_rate`: `1.000`. Gold-style two-pass in-session review produced full pass agreement on the adjudicated test split.
- `real_grouped_validation` / `historical_newspapers_gold_style_subset` / `gold_ocr_to_label_error_rate`: `0.016`. The gold-style subset retains a low OCR-to-label disagreement rate.
- `real_grouped_validation` / `historical_newspapers_gold_style_subset` / `gold_minus_original_conformal_sequence_exact_delta`: `0.000`. The real grouped metrics were unchanged after upgrading from OCR-derived to gold-style adjudicated labels.
