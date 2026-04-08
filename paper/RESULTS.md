# Results

This section summarizes only measured outputs from three completed real-data runs:

- Omniglot: `outputs/runs/20260407T150327Z_omniglot_paper_pack_evaluation`
- scikit-learn digits: `outputs/runs/20260407T191829Z_sklearn_digits_paper_pack_evaluation`
- Kuzushiji-49: `outputs/runs/20260407T201933Z_kuzushiji49_balanced_subset_paper_pack_evaluation`

## Omniglot

Omniglot remains the hardest and most heterogeneous corpus in the paper. The strongest condition, `D. Uncertainty-Aware + Calibrated Posterior`, reached mean symbol top-k accuracy `0.145` with `95%` bootstrap interval `[0.143, 0.147]`, compared with `0.045` for `A. Fixed Transcript + Heuristic Posterior`. Within matched posterior families, uncertainty retention improved symbol top-k by `0.053` on average for the heuristic path and by `0.070` for the calibrated path.

At the highest ambiguity level tested (`0.45`), the uncertainty effect persisted but was modest: the heuristic uncertainty delta was `0.009` and the calibrated uncertainty delta was `0.012`. Calibration helped on Omniglot in both the fixed and uncertainty-aware settings, and the combined condition improved NLL by `-1.640` and ECE by `-0.409` relative to the heuristic fixed baseline.

## scikit-learn Digits

The second dataset produced the largest symbol-level uncertainty effect in the current paper, despite being visually much simpler than the other corpora. `A. Fixed Transcript + Heuristic Posterior` reached mean symbol top-k accuracy `0.430` with `95%` bootstrap interval `[0.421, 0.440]`, while `C. Uncertainty-Aware + Heuristic Posterior` reached `0.904` with interval `[0.897, 0.911]`. The calibrated uncertainty condition `D` reached `0.715` with interval `[0.703, 0.725]`.

Within matched posterior families, uncertainty retention improved symbol top-k by `0.474` on average for the heuristic path and by `0.366` for the calibrated path. At ambiguity `0.45`, the corresponding deltas remained large: `0.597` for heuristic uncertainty and `0.450` for calibrated uncertainty. The combined condition improved NLL by `-1.109`, but calibration was not uniformly helpful on this dataset: fixed calibration reduced mean top-k by `-0.081`, and combined calibration increased ECE by `0.068`.

## Kuzushiji-49

The historically grounded Kuzushiji-49 run strengthens the paper beyond the earlier two-dataset story. `A. Fixed Transcript + Heuristic Posterior` reached mean symbol top-k accuracy `0.452` with `95%` bootstrap interval `[0.444, 0.460]`, while `D. Uncertainty-Aware + Calibrated Posterior` reached `0.773` with interval `[0.767, 0.779]`. The calibrated fixed condition `B` also outperformed the heuristic fixed baseline, reaching `0.537` with interval `[0.529, 0.544]`.

Within matched posterior families, uncertainty retention improved symbol top-k by `0.253` on average for the heuristic path and by `0.236` for the calibrated path. At ambiguity `0.45`, the deltas remained materially positive: `0.268` for heuristic uncertainty and `0.251` for calibrated uncertainty. The combined condition improved NLL by `-5.237` and ECE by `-0.175` relative to the heuristic fixed baseline. In other words, Kuzushiji-49 behaves more like Omniglot than digits with respect to calibration direction, but its uncertainty-rescue effect is much larger than Omniglot's.

## Cross-Dataset Interpretation

The narrow positive result now replicates across all three datasets: preserving uncertainty improved symbol-level top-k retention relative to hard transcript collapse on Omniglot, scikit-learn digits, and Kuzushiji-49. The direction of the effect was consistent, but the magnitude was not. Omniglot showed the smallest uncertainty deltas, scikit-learn digits the largest, and Kuzushiji-49 an intermediate but clearly positive effect.

The historically grounded dataset materially strengthens the paper because it closes the previous gap between a script-like corpus and a simple digit corpus. Kuzushiji-49 does not merely repeat the Omniglot story; it provides a manuscript-adjacent corpus where the uncertainty effect remains positive and calibration is again helpful. That makes the overall narrative stronger: the uncertainty-rescue effect appears more stable across corpora than the calibration effect does.

Calibration remained inconsistent across the three datasets. It helped on Omniglot and Kuzushiji-49, but hurt on scikit-learn digits. This is one of the clearest reasons to keep the paper's claim narrow: uncertainty retention appears robustly helpful at the symbol level, while calibration benefit remains dataset-dependent.

## Failure Analysis

The same qualitative failure mode appeared on all three datasets: correct classes often fell out of top-1 while remaining recoverable in top-k. Omniglot recorded `9,550` `top1_collapse_but_topk_rescue` cases, scikit-learn digits `5,008`, and Kuzushiji-49 `7,197`. Each dataset also recorded the same number of `uncertainty_helped_symbols_not_downstream` cases, which means the current evidence pack continues to show symbol-level rescue without downstream proof.

Calibration-related instability also recurred across all three datasets, though with different magnitudes: `50,001` `calibration_worsened_or_unstable` cases on Omniglot, `4,576` on scikit-learn digits, and `8,490` on Kuzushiji-49. Neither the new historical corpus nor the earlier datasets produced informative downstream family evidence under the current labels. The present results therefore support symbol-level ambiguity robustness, not semantic decipherment or reliable higher-level reasoning gains.
