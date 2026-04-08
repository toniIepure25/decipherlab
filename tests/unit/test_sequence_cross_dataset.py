from __future__ import annotations

from pathlib import Path

from decipherlab.sequence.cross_dataset import build_sequence_cross_dataset_outputs
from decipherlab.sequence.runner import run_sequence_branch_experiment
from tests.helpers import build_test_config, create_real_manifest_fixture


def _run_fixture_sequence_branch(tmp_path: Path, name: str, strategy: str):
    manifest_path = create_real_manifest_fixture(tmp_path / name)
    base_config = build_test_config(tmp_path / name)
    config = base_config.model_copy(
        update={
            "experiment": base_config.experiment.model_copy(update={"name": name, "output_root": tmp_path / "runs"}),
            "dataset": base_config.dataset.model_copy(
                update={
                    "source": "manifest",
                    "manifest_path": manifest_path,
                    "generate_if_missing": False,
                    "train_split": "train",
                    "val_split": "val",
                    "evaluation_split": "test",
                }
            ),
            "evaluation": base_config.evaluation.model_copy(update={"ambiguity_levels": [0.0, 0.35], "top_k": 3}),
            "sequence_benchmark": {
                "enabled": True,
                "selected_symbol_count": 4,
                "min_instances_per_symbol": 2,
                "train_sequences": 6,
                "val_sequences": 3,
                "test_sequences": 3,
                "sequence_length": 6,
                "group_count": 2,
                "self_transition_bias": 3.0,
                "within_group_bias": 1.5,
                "cross_group_bias": 0.35,
                "transition_noise": 0.05,
                "sample_with_replacement": True,
            },
            "structured_uncertainty": {
                "enabled": True,
                "max_candidates_per_position": 3,
                "cumulative_probability_mass": 0.95,
                "min_probability": 1.0e-5,
                "include_top1_fallback": True,
            },
            "decoding": {
                "enabled": True,
                "beam_width": 6,
                "lm_weight": 1.2,
                "transition_smoothing": 0.1,
                "top_k_sequences": 3,
                "length_normalize": True,
            },
            "risk_control": {
                "enabled": True,
                "alpha": 0.1,
                "min_set_size": 1,
                "max_set_size": 3,
                "include_top1_fallback": True,
            },
        }
    )
    return run_sequence_branch_experiment(config, strategy_override=strategy)


def test_sequence_cross_dataset_outputs_build_from_run_bundles(tmp_path):
    dataset_runs = []
    for dataset_name in ["dataset_a", "dataset_b", "dataset_c"]:
        cluster = _run_fixture_sequence_branch(tmp_path, f"{dataset_name}_cluster", "cluster_distance")
        calibrated = _run_fixture_sequence_branch(tmp_path, f"{dataset_name}_calibrated", "calibrated_classifier")
        dataset_runs.append(
            {
                "dataset_label": dataset_name,
                "task_label": "real_glyph_markov_sequences",
                "cluster_distance_run": cluster["run_dir"],
                "calibrated_classifier_run": calibrated["run_dir"],
            }
        )

    outputs = build_sequence_cross_dataset_outputs(dataset_runs, output_root=tmp_path / "outputs")

    assert Path(outputs["summary_csv"]).exists()
    assert Path(outputs["summary_md"]).exists()
    assert Path(outputs["effects_plot"]).exists()
