# Propagation Results Notes

## Main Supported Claim

The strongest supported propagation claim is:

- symbol rescue strongly predicts grouped rescue
- grouped rescue is necessary but not sufficient for downstream success
- downstream propagation depends on support regime rather than following a single monotonic rule

This turns the branch into an explanatory paper about support-aware uncertainty propagation, not only a mixed-results decoder comparison.

## Feature Layer

Primary export:

- [propagation features CSV](/home/tonystark/Desktop/decipher/outputs/propagation_features.csv)
- [propagation features Markdown](/home/tonystark/Desktop/decipher/outputs/propagation_features.md)

Current row counts:

- `synthetic_markov`: `2304`
- `synthetic_process_family`: `2304`
- `real_grouped_downstream_redesigned`: `576`

## Explanatory Models

Primary export:

- [model summary CSV](/home/tonystark/Desktop/decipher/outputs/propagation_model_summary.csv)
- [model summary Markdown](/home/tonystark/Desktop/decipher/outputs/propagation_model_summary.md)

Main findings:

- the clearest grouped-rescue predictor is `symbol_rescue`
- the grouped-rescue model assigns `symbol_rescue` an odds ratio of about `18.4`
- grouped rescue becomes less likely as entropy, ambiguity, set size, and sequence length increase
- on the real downstream task, the strongest positive predictor is `grouped_topk_delta`
- on the real downstream task, conformal has a positive average coefficient, but the effect is regime- and corpus-specific

Important caveat:

- some coefficients are clearly confounded by corpus and method composition
- the negative coverage coefficient in the real downstream model should not be read as “less support is better”

## Threshold And Regime Findings

Primary export:

- [threshold summary CSV](/home/tonystark/Desktop/decipher/outputs/propagation_thresholds.csv)
- [threshold summary Markdown](/home/tonystark/Desktop/decipher/outputs/propagation_thresholds.md)
- [regime summary CSV](/home/tonystark/Desktop/decipher/outputs/propagation_regime_summary.csv)
- [regime summary Markdown](/home/tonystark/Desktop/decipher/outputs/propagation_regime_summary.md)
- [regime plot](/home/tonystark/Desktop/decipher/outputs/propagation_regime_plot.png)

Most defensible threshold findings:

- grouped rescue falls off once mean confusion entropy rises beyond about `1.765`
- real downstream rescue is much more likely when grouped top-k delta reaches at least `1`

Measured regimes:

- raw uncertainty helps most in a rare high-support, high-entropy regime
- conformal helps most in a limited-support, low-entropy regime
- there is no single universal “more support always fixes propagation” rule in the current data

## Cross-Dataset Synthesis

Primary export:

- [cross-dataset summary CSV](/home/tonystark/Desktop/decipher/outputs/propagation_cross_dataset_summary.csv)
- [cross-dataset summary Markdown](/home/tonystark/Desktop/decipher/outputs/propagation_cross_dataset_summary.md)

Main synthesis:

- synthetic datasets are rescue-rich at symbol level but still grouped- and downstream-poor overall
- Kuzushiji-49 is the strongest synthetic higher-level setting
- Historical Newspapers has low symbol rescue but relatively strong grouped-to-downstream propagation once grouped rescue occurs
- ScaDS.AI has more symbol and grouped rescue than Historical Newspapers, but weaker downstream propagation from grouped rescue

This means corpus identity still matters after accounting for support.

## Compact Index Decision

A compact propagation index is not justified yet.

Reason:

- the current thresholds are useful but not stable enough to collapse into one score
- the real downstream regimes are still partly corpus-specific
- a single index would suggest a stronger general law than the evidence supports

## Scientific Boundary

What is now explained:

- why grouped rescue is much rarer than symbol rescue
- why downstream gains can remain mixed even after coverage improves
- why conformal can help exact recovery in some real-data regimes without being a universal win

What remains unexplained:

- a single transferable support law across both real grouped corpora
- a clean cross-corpus explanation for why Historical Newspapers and ScaDS.AI differ downstream
- a real downstream family/process result
