# Real Downstream Coverage Analysis

## Main Finding

- Exact train-transcript-bank recovery is coverage-limited on both real grouped corpora.
- Train-supported n-gram-path recovery materially improves coverage on both corpora, especially on ScaDS.AI.

## Rows

- `historical_newspapers_real_grouped_gold` / `train_transcript_bank` / `valid_candidate_rate`: `0.944`. Fraction of test examples with any same-length train transcript candidate.
- `historical_newspapers_real_grouped_gold` / `train_transcript_bank` / `exact_target_coverage`: `0.000`. Fraction of test examples whose exact transcript appears in the train bank.
- `historical_newspapers_real_grouped_gold` / `train_transcript_bank` / `approximate_exact_upper_bound`: `0.000`. Under exact transcript-bank decoding, exact recovery cannot exceed exact train/test overlap.
- `historical_newspapers_real_grouped_gold` / `train_transcript_bank` / `mean_candidate_size`: `13.389`. Average number of same-length train transcript candidates per test example.
- `historical_newspapers_real_grouped_gold` / `train_supported_ngram_path` / `valid_candidate_rate`: `1.000`. Fraction of test examples with at least one train-supported n-gram in the gold path.
- `historical_newspapers_real_grouped_gold` / `train_supported_ngram_path` / `full_path_coverage`: `0.667`. Fraction of test examples whose full gold n-gram path is supported by train n-grams.
- `historical_newspapers_real_grouped_gold` / `train_supported_ngram_path` / `mean_supported_fraction`: `0.878`. Average fraction of each gold n-gram path covered by the train n-gram inventory.
- `historical_newspapers_real_grouped_gold` / `train_supported_ngram_path` / `approximate_exact_upper_bound`: `0.667`. Exact n-gram-path recovery cannot exceed the fraction with full train-supported n-gram coverage.
- `historical_newspapers_real_grouped_gold` / `train_supported_ngram_path` / `mean_candidate_size`: `2.889`. Average number of supported gold n-grams per test example.
- `historical_newspapers_real_grouped_gold` / `train_supported_ngram_path` / `train_inventory_size`: `25.000`. Distinct train n-gram inventory size used by the redesigned task.
- `scadsai_real_grouped` / `train_transcript_bank` / `valid_candidate_rate`: `1.000`. Fraction of test examples with any same-length train transcript candidate.
- `scadsai_real_grouped` / `train_transcript_bank` / `exact_target_coverage`: `0.111`. Fraction of test examples whose exact transcript appears in the train bank.
- `scadsai_real_grouped` / `train_transcript_bank` / `approximate_exact_upper_bound`: `0.111`. Under exact transcript-bank decoding, exact recovery cannot exceed exact train/test overlap.
- `scadsai_real_grouped` / `train_transcript_bank` / `mean_candidate_size`: `32.333`. Average number of same-length train transcript candidates per test example.
- `scadsai_real_grouped` / `train_supported_ngram_path` / `valid_candidate_rate`: `1.000`. Fraction of test examples with at least one train-supported n-gram in the gold path.
- `scadsai_real_grouped` / `train_supported_ngram_path` / `full_path_coverage`: `1.000`. Fraction of test examples whose full gold n-gram path is supported by train n-grams.
- `scadsai_real_grouped` / `train_supported_ngram_path` / `mean_supported_fraction`: `1.000`. Average fraction of each gold n-gram path covered by the train n-gram inventory.
- `scadsai_real_grouped` / `train_supported_ngram_path` / `approximate_exact_upper_bound`: `1.000`. Exact n-gram-path recovery cannot exceed the fraction with full train-supported n-gram coverage.
- `scadsai_real_grouped` / `train_supported_ngram_path` / `mean_candidate_size`: `3.056`. Average number of supported gold n-grams per test example.
- `scadsai_real_grouped` / `train_supported_ngram_path` / `train_inventory_size`: `24.000`. Distinct train n-gram inventory size used by the redesigned task.
