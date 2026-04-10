# Real Downstream Redesigned Summary

## Train-Supported N-Gram-Path Task

- `historical_newspapers_real_grouped_gold` / `cluster_distance`: coverage fraction `0.878`, full-path coverage `0.667`, fixed exact `0.639`, uncertainty exact `0.514`, uncertainty downstream exact delta `-0.125`, uncertainty downstream token delta `-0.174`, conformal downstream exact delta `0.222`, grouped top-k delta `0.056`, `grouped_topk_rescue_without_downstream_exact` failures `9`.
- `historical_newspapers_real_grouped_gold` / `calibrated_classifier`: coverage fraction `0.878`, full-path coverage `0.667`, fixed exact `0.819`, uncertainty exact `0.764`, uncertainty downstream exact delta `-0.056`, uncertainty downstream token delta `-0.049`, conformal downstream exact delta `0.056`, grouped top-k delta `0.056`, `grouped_topk_rescue_without_downstream_exact` failures `4`.
- `scadsai_real_grouped` / `cluster_distance`: coverage fraction `1.000`, full-path coverage `1.000`, fixed exact `0.125`, uncertainty exact `0.236`, uncertainty downstream exact delta `0.111`, uncertainty downstream token delta `0.093`, conformal downstream exact delta `-0.042`, grouped top-k delta `0.361`, `grouped_topk_rescue_without_downstream_exact` failures `17`.
- `scadsai_real_grouped` / `calibrated_classifier`: coverage fraction `1.000`, full-path coverage `1.000`, fixed exact `0.278`, uncertainty exact `0.236`, uncertainty downstream exact delta `-0.042`, uncertainty downstream token delta `0.013`, conformal downstream exact delta `0.000`, grouped top-k delta `0.250`, `grouped_topk_rescue_without_downstream_exact` failures `17`.

## Interpretation

- This redesigned task asks whether decoder outputs recover the gold train-supported n-gram path, not only the raw grouped transcript.
- Positive downstream token or exact deltas indicate that grouped top-k rescue is propagating into a better-covered real structural target.
