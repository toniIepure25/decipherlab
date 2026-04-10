# Sensitivity Summary

## Main Takeaways

- ScaDS.AI is more rescue-rich at grouped level (`0.132` grouped top-k delta) than Historical Newspapers (`0.031`).
- Downstream exact recovery does not follow a simple monotonic support pattern here: `high_support` gives `0.015`, while `limited_support` gives `0.021`.
- Larger candidate sets strengthen grouped top-k rescue (`0.167` vs `0.007`), but smaller sets give cleaner downstream exact behavior (`0.042` vs `-0.015`).

## Rows

- `axis`=`dataset`, `bucket`=`historical_newspapers`, `sample_count`=`288`, `mean_grouped_topk_delta`=`0.031`, `mean_downstream_exact_delta`=`0.024`, `grouped_topk_rescue_rate`=`0.128`, `downstream_exact_rescue_rate`=`0.108`
- `axis`=`dataset`, `bucket`=`scadsai`, `sample_count`=`288`, `mean_grouped_topk_delta`=`0.132`, `mean_downstream_exact_delta`=`0.007`, `grouped_topk_rescue_rate`=`0.191`, `downstream_exact_rescue_rate`=`0.045`
- `axis`=`method_family`, `bucket`=`raw_uncertainty`, `sample_count`=`288`, `mean_grouped_topk_delta`=`0.181`, `mean_downstream_exact_delta`=`-0.028`, `grouped_topk_rescue_rate`=`0.233`, `downstream_exact_rescue_rate`=`0.069`
- `axis`=`method_family`, `bucket`=`conformal`, `sample_count`=`288`, `mean_grouped_topk_delta`=`-0.017`, `mean_downstream_exact_delta`=`0.059`, `grouped_topk_rescue_rate`=`0.087`, `downstream_exact_rescue_rate`=`0.083`
- `axis`=`sequence_length_bin`, `bucket`=`short_or_equal_median`, `sample_count`=`496`, `mean_grouped_topk_delta`=`0.089`, `mean_downstream_exact_delta`=`0.014`, `grouped_topk_rescue_rate`=`0.169`, `downstream_exact_rescue_rate`=`0.075`
- `axis`=`sequence_length_bin`, `bucket`=`longer_than_median`, `sample_count`=`80`, `mean_grouped_topk_delta`=`0.037`, `mean_downstream_exact_delta`=`0.025`, `grouped_topk_rescue_rate`=`0.100`, `downstream_exact_rescue_rate`=`0.087`
- `axis`=`candidate_set_bin`, `bucket`=`small_or_equal_median`, `sample_count`=`306`, `mean_grouped_topk_delta`=`0.007`, `mean_downstream_exact_delta`=`0.042`, `grouped_topk_rescue_rate`=`0.098`, `downstream_exact_rescue_rate`=`0.082`
- `axis`=`candidate_set_bin`, `bucket`=`larger_than_median`, `sample_count`=`270`, `mean_grouped_topk_delta`=`0.167`, `mean_downstream_exact_delta`=`-0.015`, `grouped_topk_rescue_rate`=`0.230`, `downstream_exact_rescue_rate`=`0.070`
- `axis`=`support_regime`, `bucket`=`limited_support`, `sample_count`=`96`, `mean_grouped_topk_delta`=`0.031`, `mean_downstream_exact_delta`=`0.021`, `grouped_topk_rescue_rate`=`0.177`, `downstream_exact_rescue_rate`=`0.219`
- `axis`=`support_regime`, `bucket`=`high_support`, `sample_count`=`480`, `mean_grouped_topk_delta`=`0.092`, `mean_downstream_exact_delta`=`0.015`, `grouped_topk_rescue_rate`=`0.156`, `downstream_exact_rescue_rate`=`0.048`
- `axis`=`ambiguity_regime`, `bucket`=`high`, `sample_count`=`144`, `mean_grouped_topk_delta`=`0.104`, `mean_downstream_exact_delta`=`0.028`, `grouped_topk_rescue_rate`=`0.194`, `downstream_exact_rescue_rate`=`0.076`
- `axis`=`ambiguity_regime`, `bucket`=`low`, `sample_count`=`288`, `mean_grouped_topk_delta`=`0.066`, `mean_downstream_exact_delta`=`-0.007`, `grouped_topk_rescue_rate`=`0.149`, `downstream_exact_rescue_rate`=`0.069`
- `axis`=`ambiguity_regime`, `bucket`=`medium`, `sample_count`=`144`, `mean_grouped_topk_delta`=`0.090`, `mean_downstream_exact_delta`=`0.049`, `grouped_topk_rescue_rate`=`0.146`, `downstream_exact_rescue_rate`=`0.090`
- `axis`=`label_quality_tier`, `bucket`=`cluster_distance:original_vs_gold`, `sample_count`=`30`, `mean_grouped_topk_delta`=`0.056`, `mean_downstream_exact_delta`=`n/a`, `grouped_topk_rescue_rate`=`n/a`, `downstream_exact_rescue_rate`=`n/a`
- `axis`=`label_quality_tier`, `bucket`=`calibrated_classifier:original_vs_gold`, `sample_count`=`30`, `mean_grouped_topk_delta`=`0.056`, `mean_downstream_exact_delta`=`n/a`, `grouped_topk_rescue_rate`=`n/a`, `downstream_exact_rescue_rate`=`n/a`
