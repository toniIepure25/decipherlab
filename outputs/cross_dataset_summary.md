# Cross-Dataset Summary

## Headline
- omniglot_character_crops: heuristic uncertainty top-k delta mean 0.053; calibrated uncertainty top-k delta mean 0.070; fixed-calibration top-k delta mean 0.030.
- sklearn_digits_crops: heuristic uncertainty top-k delta mean 0.474; calibrated uncertainty top-k delta mean 0.366; fixed-calibration top-k delta mean -0.081.
- kuzushiji49_balanced_crops: heuristic uncertainty top-k delta mean 0.253; calibrated uncertainty top-k delta mean 0.236; fixed-calibration top-k delta mean 0.085.

## Ambiguity Sweep
- kuzushiji49_balanced_crops: at ambiguity 0.450, heuristic uncertainty top-k delta = 0.268, calibrated uncertainty top-k delta = 0.251.
- omniglot_character_crops: at ambiguity 0.450, heuristic uncertainty top-k delta = 0.009, calibrated uncertainty top-k delta = 0.012.
- sklearn_digits_crops: at ambiguity 0.450, heuristic uncertainty top-k delta = 0.597, calibrated uncertainty top-k delta = 0.450.

## Failure Modes
- kuzushiji49_balanced_crops: calibration_worsened_or_unstable = 8490
- kuzushiji49_balanced_crops: overdiffuse_posterior = 12422
- kuzushiji49_balanced_crops: top1_collapse_but_topk_rescue = 7197
- kuzushiji49_balanced_crops: uncertainty_helped_symbols_not_downstream = 7197
- omniglot_character_crops: calibration_worsened_or_unstable = 50001
- omniglot_character_crops: overdiffuse_posterior = 83918
- omniglot_character_crops: top1_collapse_but_topk_rescue = 9550
- omniglot_character_crops: uncertainty_helped_symbols_not_downstream = 9550
- sklearn_digits_crops: calibration_worsened_or_unstable = 4576
- sklearn_digits_crops: overdiffuse_posterior = 3363
- sklearn_digits_crops: top1_collapse_but_topk_rescue = 5008
- sklearn_digits_crops: uncertainty_helped_symbols_not_downstream = 5008

## Guardrail
- These comparisons remain symbol-level and ambiguity-focused.
- They do not by themselves support semantic decipherment or broad historical-generalization claims.
