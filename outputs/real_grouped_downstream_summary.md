# Real Grouped Downstream Summary

## Train-Transcript-Bank Task

- `historical_newspapers_real_grouped_gold` / `cluster_distance`: bank coverage `0.000`, fixed exact `0.000`, uncertainty exact `0.000`, uncertainty downstream exact delta `0.000`, uncertainty downstream top-k delta `0.000`, conformal downstream exact delta `0.000`, grouped top-k delta `0.056`, `grouped_topk_rescue_without_downstream_exact` failures `16`.
- `historical_newspapers_real_grouped_gold` / `calibrated_classifier`: bank coverage `0.000`, fixed exact `0.000`, uncertainty exact `0.000`, uncertainty downstream exact delta `0.000`, uncertainty downstream top-k delta `0.000`, conformal downstream exact delta `0.000`, grouped top-k delta `0.056`, `grouped_topk_rescue_without_downstream_exact` failures `5`.
- `scadsai_real_grouped` / `cluster_distance`: bank coverage `0.111`, fixed exact `0.042`, uncertainty exact `0.042`, uncertainty downstream exact delta `0.000`, uncertainty downstream top-k delta `-0.042`, conformal downstream exact delta `0.000`, grouped top-k delta `0.361`, `grouped_topk_rescue_without_downstream_exact` failures `27`.
- `scadsai_real_grouped` / `calibrated_classifier`: bank coverage `0.111`, fixed exact `0.056`, uncertainty exact `0.056`, uncertainty downstream exact delta `0.000`, uncertainty downstream top-k delta `0.000`, conformal downstream exact delta `0.056`, grouped top-k delta `0.250`, `grouped_topk_rescue_without_downstream_exact` failures `18`.

## Interpretation

- This summary asks whether replicated real grouped top-k rescue propagates into a real downstream structural target: train-transcript-bank recovery.
- Positive grouped top-k deltas with weak or negative downstream exact deltas indicate that uncertainty is preserving useful alternatives without reliably resolving full real transcripts.
