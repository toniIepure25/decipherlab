# Sequence Cross-Dataset Summary

These results come from a real grouped manifest-backed sequence dataset and should be read as preliminary grouped evidence, not semantic decipherment evidence.

## Aggregate Effects
- historical_newspapers_real_grouped_validated / real_grouped_manifest_sequences (calibrated_classifier): mean uncertainty exact delta = -0.056, mean uncertainty top-k delta = 0.056, mean conformal coverage delta = -0.024, mean CRF exact delta = 0.000, peak exact delta at ambiguity 0.000.
- historical_newspapers_real_grouped_validated / real_grouped_manifest_sequences (cluster_distance): mean uncertainty exact delta = -0.125, mean uncertainty top-k delta = 0.056, mean conformal coverage delta = -0.063, mean CRF exact delta = 0.000, peak exact delta at ambiguity 0.450.

## Ambiguity Regimes
- historical_newspapers_real_grouped_validated cluster_distance high: mean uncertainty exact delta = 0.056, mean uncertainty top-k delta = 0.333, mean conformal coverage delta = -0.111, mean CRF exact delta = 0.000.
- historical_newspapers_real_grouped_validated cluster_distance low: mean uncertainty exact delta = -0.250, mean uncertainty top-k delta = -0.111, mean conformal coverage delta = -0.042, mean CRF exact delta = 0.000.
- historical_newspapers_real_grouped_validated cluster_distance medium: mean uncertainty exact delta = -0.056, mean uncertainty top-k delta = 0.111, mean conformal coverage delta = -0.056, mean CRF exact delta = 0.000.
- historical_newspapers_real_grouped_validated calibrated_classifier high: mean uncertainty exact delta = -0.056, mean uncertainty top-k delta = 0.056, mean conformal coverage delta = -0.042, mean CRF exact delta = 0.000.
- historical_newspapers_real_grouped_validated calibrated_classifier low: mean uncertainty exact delta = -0.028, mean uncertainty top-k delta = 0.083, mean conformal coverage delta = -0.021, mean CRF exact delta = 0.000.
- historical_newspapers_real_grouped_validated calibrated_classifier medium: mean uncertainty exact delta = -0.111, mean uncertainty top-k delta = 0.000, mean conformal coverage delta = -0.014, mean CRF exact delta = 0.000.

## Recurring Failure Modes
- historical_newspapers_real_grouped_validated: `conformal_over_expansion` count = 30.
- historical_newspapers_real_grouped_validated: `conformal_over_pruning` count = 15.
- historical_newspapers_real_grouped_validated: `decoder_trapped_by_transition_prior` count = 30.
- historical_newspapers_real_grouped_validated: `high_ambiguity_diffuse_uncertainty` count = 9.
- historical_newspapers_real_grouped_validated: `sequence_rescue_only_crf_not_conformal` count = 3.
- historical_newspapers_real_grouped_validated: `sequence_rescue_only_trigram_not_conformal` count = 14.
- historical_newspapers_real_grouped_validated: `sequence_rescue_only_uncertainty_beam` count = 15.
- historical_newspapers_real_grouped_validated: `sequence_rescue_uncertainty_beam` count = 21.
- historical_newspapers_real_grouped_validated: `symbol_rescue_without_sequence_rescue` count = 17.
- historical_newspapers_real_grouped_validated: `trigram_amplifies_bad_prior` count = 41.
- historical_newspapers_real_grouped_validated: `trigram_sequence_rescue` count = 4.