# Profile-Aware Interactive Decoding Summary

This summary compares explicit operating profiles rather than forcing a single adaptive policy to serve every archival workflow.

## Results

- `historical_newspapers_real_grouped` / `cluster_distance` / `rescue_first` / `k=2`: grouped top-k delta `0.056`, downstream exact delta `n/a`, recall@budget `0.583`, shortlist utility `0.618`, set size `3.047`, defer rate `0.000`.
- `historical_newspapers_real_grouped` / `calibrated_classifier` / `rescue_first` / `k=2`: grouped top-k delta `0.028`, downstream exact delta `n/a`, recall@budget `0.847`, shortlist utility `0.847`, set size `1.132`, defer rate `0.000`.
- `historical_newspapers_real_grouped` / `cluster_distance` / `rescue_first` / `k=3`: grouped top-k delta `0.056`, downstream exact delta `n/a`, recall@budget `0.625`, shortlist utility `0.618`, set size `3.047`, defer rate `0.000`.
- `historical_newspapers_real_grouped` / `calibrated_classifier` / `rescue_first` / `k=3`: grouped top-k delta `0.028`, downstream exact delta `n/a`, recall@budget `0.847`, shortlist utility `0.847`, set size `1.132`, defer rate `0.000`.
- `historical_newspapers_real_grouped` / `cluster_distance` / `rescue_first` / `k=5`: grouped top-k delta `0.056`, downstream exact delta `n/a`, recall@budget `0.694`, shortlist utility `0.618`, set size `3.047`, defer rate `0.000`.
- `historical_newspapers_real_grouped` / `calibrated_classifier` / `rescue_first` / `k=5`: grouped top-k delta `0.028`, downstream exact delta `n/a`, recall@budget `0.847`, shortlist utility `0.847`, set size `1.132`, defer rate `0.000`.
- `historical_newspapers_real_grouped` / `cluster_distance` / `shortlist_first` / `k=2`: grouped top-k delta `0.111`, downstream exact delta `n/a`, recall@budget `0.750`, shortlist utility `0.750`, set size `1.184`, defer rate `0.000`.
- `historical_newspapers_real_grouped` / `calibrated_classifier` / `shortlist_first` / `k=2`: grouped top-k delta `0.028`, downstream exact delta `n/a`, recall@budget `0.833`, shortlist utility `0.836`, set size `1.115`, defer rate `0.000`.
- `historical_newspapers_real_grouped` / `cluster_distance` / `shortlist_first` / `k=3`: grouped top-k delta `0.111`, downstream exact delta `n/a`, recall@budget `0.750`, shortlist utility `0.750`, set size `1.184`, defer rate `0.000`.
- `historical_newspapers_real_grouped` / `calibrated_classifier` / `shortlist_first` / `k=3`: grouped top-k delta `0.028`, downstream exact delta `n/a`, recall@budget `0.833`, shortlist utility `0.836`, set size `1.115`, defer rate `0.000`.
- `historical_newspapers_real_grouped` / `cluster_distance` / `shortlist_first` / `k=5`: grouped top-k delta `0.097`, downstream exact delta `n/a`, recall@budget `0.736`, shortlist utility `0.736`, set size `1.212`, defer rate `0.000`.
- `historical_newspapers_real_grouped` / `calibrated_classifier` / `shortlist_first` / `k=5`: grouped top-k delta `0.028`, downstream exact delta `n/a`, recall@budget `0.847`, shortlist utility `0.836`, set size `1.115`, defer rate `0.000`.
- `historical_newspapers_real_downstream` / `cluster_distance` / `rescue_first` / `k=2`: grouped top-k delta `0.056`, downstream exact delta `-0.125`, recall@budget `0.569`, shortlist utility `0.611`, set size `2.945`, defer rate `0.097`.
- `historical_newspapers_real_downstream` / `calibrated_classifier` / `rescue_first` / `k=2`: grouped top-k delta `0.028`, downstream exact delta `-0.014`, recall@budget `0.847`, shortlist utility `0.847`, set size `1.132`, defer rate `0.000`.
- `historical_newspapers_real_downstream` / `cluster_distance` / `rescue_first` / `k=3`: grouped top-k delta `0.056`, downstream exact delta `-0.125`, recall@budget `0.625`, shortlist utility `0.611`, set size `2.945`, defer rate `0.097`.
- `historical_newspapers_real_downstream` / `calibrated_classifier` / `rescue_first` / `k=3`: grouped top-k delta `0.028`, downstream exact delta `-0.014`, recall@budget `0.847`, shortlist utility `0.847`, set size `1.132`, defer rate `0.000`.
- `historical_newspapers_real_downstream` / `cluster_distance` / `rescue_first` / `k=5`: grouped top-k delta `0.056`, downstream exact delta `-0.125`, recall@budget `0.694`, shortlist utility `0.622`, set size `2.772`, defer rate `0.000`.
- `historical_newspapers_real_downstream` / `calibrated_classifier` / `rescue_first` / `k=5`: grouped top-k delta `0.028`, downstream exact delta `-0.014`, recall@budget `0.847`, shortlist utility `0.847`, set size `1.132`, defer rate `0.000`.
- `historical_newspapers_real_downstream` / `cluster_distance` / `shortlist_first` / `k=2`: grouped top-k delta `0.083`, downstream exact delta `0.056`, recall@budget `0.708`, shortlist utility `0.711`, set size `1.441`, defer rate `0.097`.
- `historical_newspapers_real_downstream` / `calibrated_classifier` / `shortlist_first` / `k=2`: grouped top-k delta `0.028`, downstream exact delta `-0.014`, recall@budget `0.833`, shortlist utility `0.836`, set size `1.115`, defer rate `0.000`.
- `historical_newspapers_real_downstream` / `cluster_distance` / `shortlist_first` / `k=3`: grouped top-k delta `0.083`, downstream exact delta `0.056`, recall@budget `0.708`, shortlist utility `0.711`, set size `1.441`, defer rate `0.097`.
- `historical_newspapers_real_downstream` / `calibrated_classifier` / `shortlist_first` / `k=3`: grouped top-k delta `0.028`, downstream exact delta `-0.014`, recall@budget `0.833`, shortlist utility `0.836`, set size `1.115`, defer rate `0.000`.
- `historical_newspapers_real_downstream` / `cluster_distance` / `shortlist_first` / `k=5`: grouped top-k delta `0.097`, downstream exact delta `0.042`, recall@budget `0.736`, shortlist utility `0.729`, set size `1.268`, defer rate `0.000`.
- `historical_newspapers_real_downstream` / `calibrated_classifier` / `shortlist_first` / `k=5`: grouped top-k delta `0.028`, downstream exact delta `-0.014`, recall@budget `0.847`, shortlist utility `0.836`, set size `1.115`, defer rate `0.000`.
- `scadsai_real_grouped` / `cluster_distance` / `rescue_first` / `k=2`: grouped top-k delta `0.333`, downstream exact delta `n/a`, recall@budget `0.292`, shortlist utility `0.338`, set size `3.934`, defer rate `0.542`.
- `scadsai_real_grouped` / `calibrated_classifier` / `rescue_first` / `k=2`: grouped top-k delta `0.250`, downstream exact delta `n/a`, recall@budget `0.347`, shortlist utility `0.408`, set size `2.792`, defer rate `0.000`.
- `scadsai_real_grouped` / `cluster_distance` / `rescue_first` / `k=3`: grouped top-k delta `0.333`, downstream exact delta `n/a`, recall@budget `0.333`, shortlist utility `0.338`, set size `3.934`, defer rate `0.542`.
- `scadsai_real_grouped` / `calibrated_classifier` / `rescue_first` / `k=3`: grouped top-k delta `0.250`, downstream exact delta `n/a`, recall@budget `0.431`, shortlist utility `0.408`, set size `2.792`, defer rate `0.000`.
- `scadsai_real_grouped` / `cluster_distance` / `rescue_first` / `k=5`: grouped top-k delta `0.333`, downstream exact delta `n/a`, recall@budget `0.458`, shortlist utility `0.338`, set size `3.934`, defer rate `0.000`.
- `scadsai_real_grouped` / `calibrated_classifier` / `rescue_first` / `k=5`: grouped top-k delta `0.250`, downstream exact delta `n/a`, recall@budget `0.528`, shortlist utility `0.408`, set size `2.785`, defer rate `0.000`.
- `scadsai_real_grouped` / `cluster_distance` / `shortlist_first` / `k=2`: grouped top-k delta `0.292`, downstream exact delta `n/a`, recall@budget `0.347`, shortlist utility `0.365`, set size `2.885`, defer rate `0.542`.
- `scadsai_real_grouped` / `calibrated_classifier` / `shortlist_first` / `k=2`: grouped top-k delta `0.208`, downstream exact delta `n/a`, recall@budget `0.347`, shortlist utility `0.400`, set size `2.601`, defer rate `0.000`.
- `scadsai_real_grouped` / `cluster_distance` / `shortlist_first` / `k=3`: grouped top-k delta `0.292`, downstream exact delta `n/a`, recall@budget `0.361`, shortlist utility `0.365`, set size `2.885`, defer rate `0.542`.
- `scadsai_real_grouped` / `calibrated_classifier` / `shortlist_first` / `k=3`: grouped top-k delta `0.208`, downstream exact delta `n/a`, recall@budget `0.431`, shortlist utility `0.400`, set size `2.601`, defer rate `0.000`.
- `scadsai_real_grouped` / `cluster_distance` / `shortlist_first` / `k=5`: grouped top-k delta `0.250`, downstream exact delta `n/a`, recall@budget `0.375`, shortlist utility `0.319`, set size `1.771`, defer rate `0.000`.
- `scadsai_real_grouped` / `calibrated_classifier` / `shortlist_first` / `k=5`: grouped top-k delta `0.208`, downstream exact delta `n/a`, recall@budget `0.486`, shortlist utility `0.400`, set size `2.604`, defer rate `0.000`.
- `scadsai_real_downstream` / `cluster_distance` / `rescue_first` / `k=2`: grouped top-k delta `0.333`, downstream exact delta `0.111`, recall@budget `0.292`, shortlist utility `0.338`, set size `3.934`, defer rate `0.542`.
- `scadsai_real_downstream` / `calibrated_classifier` / `rescue_first` / `k=2`: grouped top-k delta `0.250`, downstream exact delta `-0.042`, recall@budget `0.347`, shortlist utility `0.408`, set size `2.792`, defer rate `0.000`.
- `scadsai_real_downstream` / `cluster_distance` / `rescue_first` / `k=3`: grouped top-k delta `0.333`, downstream exact delta `0.111`, recall@budget `0.333`, shortlist utility `0.338`, set size `3.934`, defer rate `0.542`.
- `scadsai_real_downstream` / `calibrated_classifier` / `rescue_first` / `k=3`: grouped top-k delta `0.250`, downstream exact delta `-0.042`, recall@budget `0.431`, shortlist utility `0.408`, set size `2.792`, defer rate `0.000`.
- `scadsai_real_downstream` / `cluster_distance` / `rescue_first` / `k=5`: grouped top-k delta `0.333`, downstream exact delta `0.111`, recall@budget `0.458`, shortlist utility `0.338`, set size `3.934`, defer rate `0.000`.
- `scadsai_real_downstream` / `calibrated_classifier` / `rescue_first` / `k=5`: grouped top-k delta `0.250`, downstream exact delta `-0.042`, recall@budget `0.528`, shortlist utility `0.408`, set size `2.785`, defer rate `0.000`.
- `scadsai_real_downstream` / `cluster_distance` / `shortlist_first` / `k=2`: grouped top-k delta `0.292`, downstream exact delta `0.111`, recall@budget `0.347`, shortlist utility `0.365`, set size `2.885`, defer rate `0.542`.
- `scadsai_real_downstream` / `calibrated_classifier` / `shortlist_first` / `k=2`: grouped top-k delta `0.208`, downstream exact delta `-0.042`, recall@budget `0.347`, shortlist utility `0.400`, set size `2.601`, defer rate `0.000`.
- `scadsai_real_downstream` / `cluster_distance` / `shortlist_first` / `k=3`: grouped top-k delta `0.292`, downstream exact delta `0.111`, recall@budget `0.361`, shortlist utility `0.365`, set size `2.885`, defer rate `0.542`.
- `scadsai_real_downstream` / `calibrated_classifier` / `shortlist_first` / `k=3`: grouped top-k delta `0.208`, downstream exact delta `-0.042`, recall@budget `0.431`, shortlist utility `0.400`, set size `2.601`, defer rate `0.000`.
- `scadsai_real_downstream` / `cluster_distance` / `shortlist_first` / `k=5`: grouped top-k delta `0.278`, downstream exact delta `0.069`, recall@budget `0.403`, shortlist utility `0.336`, set size `1.823`, defer rate `0.000`.
- `scadsai_real_downstream` / `calibrated_classifier` / `shortlist_first` / `k=5`: grouped top-k delta `0.208`, downstream exact delta `-0.042`, recall@budget `0.486`, shortlist utility `0.400`, set size `2.604`, defer rate `0.000`.

## Operating-profile takeaway

- `shortlist_first` is the compact verification profile: it improves budgeted shortlist utility and usually returns smaller candidate sets.
- `rescue_first` is the fragile-manuscript profile: it protects grouped rescue and is more willing to preserve candidates or defer when pruning is risky.
