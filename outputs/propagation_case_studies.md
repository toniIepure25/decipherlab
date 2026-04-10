# Propagation Case Studies

## Raw Uncertainty Downstream Rescue

- dataset: `historical_newspapers`
- example_id: `00675743_line_0038`
- method_family: `raw_uncertainty`
- posterior: `cluster_distance`
- ambiguity: `0.000` (low)
- sequence_length: `4.000`
- mean_confusion_entropy: `0.710`
- prediction_set_avg_size: `3.250`
- support_coverage: `1.000`
- symbol_topk_delta: `0.250`
- grouped_topk_delta: `1.000`
- downstream_exact_delta: `1.000`
- run_dir: `outputs/runs/20260409T142627Z_sequence_historical_newspapers_real_downstream_redesigned_sequence_branch_cluster_distance`

## Conformal Downstream Rescue

- dataset: `historical_newspapers`
- example_id: `00675549_line_0027`
- method_family: `conformal`
- posterior: `cluster_distance`
- ambiguity: `0.000` (low)
- sequence_length: `4.000`
- mean_confusion_entropy: `0.000`
- prediction_set_avg_size: `1.000`
- support_coverage: `0.333`
- symbol_topk_delta: `0.000`
- grouped_topk_delta: `1.000`
- downstream_exact_delta: `1.000`
- run_dir: `outputs/runs/20260409T142627Z_sequence_historical_newspapers_real_downstream_redesigned_sequence_branch_cluster_distance`

## Grouped Rescue Without Downstream Rescue

- dataset: `scadsai`
- example_id: `0961-008`
- method_family: `raw_uncertainty`
- posterior: `cluster_distance`
- ambiguity: `0.450` (high)
- sequence_length: `4.000`
- mean_confusion_entropy: `1.325`
- prediction_set_avg_size: `4.000`
- support_coverage: `1.000`
- symbol_topk_delta: `0.750`
- grouped_topk_delta: `1.000`
- downstream_exact_delta: `0.000`
- run_dir: `outputs/runs/20260409T142630Z_sequence_scadsai_real_downstream_redesigned_sequence_branch_cluster_distance`

## Symbol Rescue Without Grouped Rescue

- dataset: `omniglot`
- example_id: `test_seq_0044`
- method_family: `raw_uncertainty`
- posterior: `cluster_distance`
- ambiguity: `0.150` (low)
- sequence_length: `12.000`
- mean_confusion_entropy: `1.270`
- prediction_set_avg_size: `5.917`
- support_coverage: `n/a`
- symbol_topk_delta: `1.000`
- grouped_topk_delta: `0.000`
- downstream_exact_delta: `0.000`
- run_dir: `outputs/runs/20260408T214508Z_sequence_omniglot_process_family_sequence_branch_cluster_distance`
