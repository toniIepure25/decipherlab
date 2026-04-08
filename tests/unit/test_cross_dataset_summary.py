from __future__ import annotations

import json
from pathlib import Path

from decipherlab.evaluation.cross_dataset import build_cross_dataset_outputs
from decipherlab.utils.io import write_csv


def _write_fake_run(run_dir: Path, dataset_name: str, topk_a: float, topk_b: float, topk_c: float, topk_d: float) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        run_dir / "main_comparison_with_ci.csv",
        [
            {
                "condition": "A. Fixed Transcript + Heuristic Posterior",
                "mean_symbol_top1_accuracy": topk_a / 2.0,
                "mean_symbol_topk_accuracy": topk_a,
                "mean_symbol_topk_accuracy_ci_lower": topk_a - 0.01,
                "mean_symbol_topk_accuracy_ci_upper": topk_a + 0.01,
                "mean_symbol_nll": 4.0,
                "mean_symbol_nll_ci_lower": 3.8,
                "mean_symbol_nll_ci_upper": 4.2,
                "mean_symbol_ece": 0.4,
                "mean_symbol_ece_ci_lower": 0.35,
                "mean_symbol_ece_ci_upper": 0.45,
                "mean_family_topk_accuracy": "",
            },
            {
                "condition": "B. Fixed Transcript + Calibrated Posterior",
                "mean_symbol_top1_accuracy": topk_b / 2.0,
                "mean_symbol_topk_accuracy": topk_b,
                "mean_symbol_topk_accuracy_ci_lower": topk_b - 0.01,
                "mean_symbol_topk_accuracy_ci_upper": topk_b + 0.01,
                "mean_symbol_nll": 3.5,
                "mean_symbol_nll_ci_lower": 3.3,
                "mean_symbol_nll_ci_upper": 3.7,
                "mean_symbol_ece": 0.3,
                "mean_symbol_ece_ci_lower": 0.25,
                "mean_symbol_ece_ci_upper": 0.35,
                "mean_family_topk_accuracy": "",
            },
            {
                "condition": "C. Uncertainty-Aware + Heuristic Posterior",
                "mean_symbol_top1_accuracy": topk_c / 2.0,
                "mean_symbol_topk_accuracy": topk_c,
                "mean_symbol_topk_accuracy_ci_lower": topk_c - 0.01,
                "mean_symbol_topk_accuracy_ci_upper": topk_c + 0.01,
                "mean_symbol_nll": 3.0,
                "mean_symbol_nll_ci_lower": 2.8,
                "mean_symbol_nll_ci_upper": 3.2,
                "mean_symbol_ece": 0.25,
                "mean_symbol_ece_ci_lower": 0.2,
                "mean_symbol_ece_ci_upper": 0.3,
                "mean_family_topk_accuracy": "",
            },
            {
                "condition": "D. Uncertainty-Aware + Calibrated Posterior",
                "mean_symbol_top1_accuracy": topk_d / 2.0,
                "mean_symbol_topk_accuracy": topk_d,
                "mean_symbol_topk_accuracy_ci_lower": topk_d - 0.01,
                "mean_symbol_topk_accuracy_ci_upper": topk_d + 0.01,
                "mean_symbol_nll": 2.5,
                "mean_symbol_nll_ci_lower": 2.3,
                "mean_symbol_nll_ci_upper": 2.7,
                "mean_symbol_ece": 0.2,
                "mean_symbol_ece_ci_lower": 0.15,
                "mean_symbol_ece_ci_upper": 0.25,
                "mean_family_topk_accuracy": "",
            },
        ],
    )
    write_csv(
        run_dir / "pairwise_effect_summary.csv",
        [
            {
                "ambiguity_level": 0.0,
                "heuristic_uncertainty_topk_delta": topk_c - topk_a,
                "calibrated_uncertainty_topk_delta": topk_d - topk_b,
                "fixed_calibration_topk_delta": topk_b - topk_a,
                "uncertainty_calibration_topk_delta": topk_d - topk_c,
                "combined_topk_delta": topk_d - topk_a,
                "combined_nll_delta": -1.0,
                "combined_ece_delta": -0.2,
            },
            {
                "ambiguity_level": 0.45,
                "heuristic_uncertainty_topk_delta": (topk_c - topk_a) / 2.0,
                "calibrated_uncertainty_topk_delta": (topk_d - topk_b) / 2.0,
                "fixed_calibration_topk_delta": (topk_b - topk_a) / 2.0,
                "uncertainty_calibration_topk_delta": (topk_d - topk_c) / 2.0,
                "combined_topk_delta": (topk_d - topk_a) / 2.0,
                "combined_nll_delta": -0.5,
                "combined_ece_delta": -0.1,
            },
        ],
    )
    write_csv(
        run_dir / "failure_case_summary.csv",
        [
            {
                "case_type": "top1_collapse_but_topk_rescue",
                "ambiguity_level": 0.0,
                "condition_group": "uncertainty_calibrated",
                "count": 10,
            },
            {
                "case_type": "calibration_worsened_or_unstable",
                "ambiguity_level": 0.45,
                "condition_group": "uncertainty",
                "count": 4,
            },
        ],
    )
    (run_dir / "dataset_summary.json").write_text(
        json.dumps(
            {
                "dataset_name": dataset_name,
                "sequence_count": 100,
                "record_count": 100,
                "split_group_counts": {"train": 0, "val": 0, "test": 0},
                "train_symbol_counts": {str(i): 10 for i in range(10)},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "experiment_metadata.json").write_text(
        json.dumps({"dataset_manifest": "dummy.yaml"}),
        encoding="utf-8",
    )


def test_build_cross_dataset_outputs_writes_expected_artifacts(tmp_path) -> None:
    run_a = tmp_path / "omniglot_run"
    run_b = tmp_path / "digits_run"
    run_c = tmp_path / "k49_run"
    _write_fake_run(run_a, "omniglot_character_crops", 0.1, 0.2, 0.3, 0.4)
    _write_fake_run(run_b, "sklearn_digits_crops", 0.2, 0.3, 0.35, 0.5)
    _write_fake_run(run_c, "kuzushiji49_balanced_crops", 0.05, 0.09, 0.14, 0.18)

    outputs = build_cross_dataset_outputs(
        datasets=[
            {"dataset_label": "omniglot_character_crops", "run_dir": run_a},
            {"dataset_label": "sklearn_digits_crops", "run_dir": run_b},
            {"dataset_label": "kuzushiji49_balanced_crops", "run_dir": run_c},
        ],
        output_root=tmp_path / "outputs",
    )

    assert Path(outputs["summary_csv"]).exists()
    assert Path(outputs["summary_md"]).exists()
    assert Path(outputs["effects_plot"]).exists()
    assert Path(outputs["failure_csv"]).exists()
    assert Path(outputs["ci_table_csv"]).exists()
    summary_text = Path(outputs["summary_md"]).read_text(encoding="utf-8")
    assert "omniglot_character_crops" in summary_text
    assert "sklearn_digits_crops" in summary_text
    assert "kuzushiji49_balanced_crops" in summary_text
    ci_rows = Path(outputs["ci_table_csv"]).read_text(encoding="utf-8")
    assert "mean_symbol_nll" in ci_rows
    assert "mean_symbol_ece" in ci_rows
