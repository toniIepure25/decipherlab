# Sequence Cross-Dataset Summary

These results come from a real grouped manifest-backed sequence dataset and should be read as preliminary grouped evidence, not semantic decipherment evidence.

## Aggregate Effects
- scadsai_real_grouped / real_grouped_manifest_sequences (calibrated_classifier): mean uncertainty exact delta = -0.042, mean uncertainty top-k delta = 0.250, mean conformal coverage delta = -0.017, mean CRF exact delta = 0.000, peak exact delta at ambiguity 0.300.
- scadsai_real_grouped / real_grouped_manifest_sequences (cluster_distance): mean uncertainty exact delta = 0.111, mean uncertainty top-k delta = 0.361, mean conformal coverage delta = -0.119, mean CRF exact delta = 0.000, peak exact delta at ambiguity 0.000.

## Ambiguity Regimes
- scadsai_real_grouped cluster_distance high: mean uncertainty exact delta = 0.111, mean uncertainty top-k delta = 0.389, mean conformal coverage delta = -0.122, mean CRF exact delta = 0.000.
- scadsai_real_grouped cluster_distance low: mean uncertainty exact delta = 0.111, mean uncertainty top-k delta = 0.333, mean conformal coverage delta = -0.115, mean CRF exact delta = 0.000.
- scadsai_real_grouped cluster_distance medium: mean uncertainty exact delta = 0.111, mean uncertainty top-k delta = 0.389, mean conformal coverage delta = -0.122, mean CRF exact delta = 0.000.
- scadsai_real_grouped calibrated_classifier high: mean uncertainty exact delta = -0.056, mean uncertainty top-k delta = 0.278, mean conformal coverage delta = -0.039, mean CRF exact delta = 0.000.
- scadsai_real_grouped calibrated_classifier low: mean uncertainty exact delta = -0.111, mean uncertainty top-k delta = 0.250, mean conformal coverage delta = -0.007, mean CRF exact delta = 0.000.
- scadsai_real_grouped calibrated_classifier medium: mean uncertainty exact delta = 0.111, mean uncertainty top-k delta = 0.222, mean conformal coverage delta = -0.014, mean CRF exact delta = 0.000.

## Recurring Failure Modes
- scadsai_real_grouped: `conformal_over_expansion` count = 22.
- scadsai_real_grouped: `conformal_over_pruning` count = 15.
- scadsai_real_grouped: `decoder_trapped_by_transition_prior` count = 29.
- scadsai_real_grouped: `high_ambiguity_diffuse_uncertainty` count = 20.
- scadsai_real_grouped: `sequence_rescue_only_crf_not_conformal` count = 4.
- scadsai_real_grouped: `sequence_rescue_only_trigram_not_conformal` count = 5.
- scadsai_real_grouped: `sequence_rescue_only_uncertainty_beam` count = 15.
- scadsai_real_grouped: `sequence_rescue_uncertainty_beam` count = 46.
- scadsai_real_grouped: `symbol_rescue_without_sequence_rescue` count = 61.
- scadsai_real_grouped: `trigram_amplifies_bad_prior` count = 58.