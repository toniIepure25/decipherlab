from __future__ import annotations

from pathlib import Path

from decipherlab.config import (
    DecodingConfig,
    RiskControlConfig,
    SequenceBenchmarkConfig,
    StructuredUncertaintyConfig,
)
from decipherlab.sequence.cross_dataset import (
    build_sequence_cross_dataset_outputs,
    build_sequence_decoder_comparison_outputs,
)
from decipherlab.sequence.runner import run_sequence_branch_experiment
from tests.helpers import build_test_config, create_real_manifest_fixture


def _run_fixture_sequence_branch(tmp_path: Path, name: str, strategy: str, task_name: str = "real_glyph_markov_sequences"):
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
            "sequence_benchmark": SequenceBenchmarkConfig(
                enabled=True,
                task_name=task_name,
                selected_symbol_count=4,
                min_instances_per_symbol=2,
                train_sequences=9 if task_name == "real_glyph_process_family_sequences" else 6,
                val_sequences=6 if task_name == "real_glyph_process_family_sequences" else 3,
                test_sequences=6 if task_name == "real_glyph_process_family_sequences" else 3,
                sequence_length=6,
                group_count=2,
                self_transition_bias=3.0,
                within_group_bias=1.5,
                cross_group_bias=0.35,
                transition_noise=0.05,
                sample_with_replacement=True,
            ),
            "structured_uncertainty": StructuredUncertaintyConfig(
                enabled=True,
                max_candidates_per_position=3,
                cumulative_probability_mass=0.95,
                min_probability=1.0e-5,
                include_top1_fallback=True,
            ),
            "decoding": DecodingConfig(
                enabled=True,
                decoder_variants=["bigram_beam", "trigram_beam", "crf_viterbi"],
                beam_width=6,
                lm_weight=1.2,
                trigram_lm_weight=1.2,
                transition_smoothing=0.1,
                top_k_sequences=3,
                length_normalize=True,
            ),
            "risk_control": RiskControlConfig(
                enabled=True,
                alpha=0.1,
                min_set_size=1,
                max_set_size=3,
                include_top1_fallback=True,
            ),
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


def test_sequence_process_family_and_decoder_comparison_outputs_build_from_run_bundles(tmp_path):
    dataset_runs = []
    for dataset_name in ["dataset_a", "dataset_b", "dataset_c"]:
        cluster = _run_fixture_sequence_branch(
            tmp_path,
            f"{dataset_name}_cluster_process",
            "cluster_distance",
            task_name="real_glyph_process_family_sequences",
        )
        calibrated = _run_fixture_sequence_branch(
            tmp_path,
            f"{dataset_name}_calibrated_process",
            "calibrated_classifier",
            task_name="real_glyph_process_family_sequences",
        )
        dataset_runs.append(
            {
                "dataset_label": dataset_name,
                "task_label": "real_glyph_process_family_sequences",
                "cluster_distance_run": cluster["run_dir"],
                "calibrated_classifier_run": calibrated["run_dir"],
            }
        )

    process_outputs = build_sequence_cross_dataset_outputs(
        dataset_runs,
        output_root=tmp_path / "outputs",
        output_prefix="sequence_process_family_cross_dataset",
        markdown_title="Sequence Process-Family Cross-Dataset Summary",
        alias_stem="sequence_process_family_cross_dataset",
    )
    decoder_outputs = build_sequence_decoder_comparison_outputs(
        [
            {
                "task_label": "real_glyph_process_family_sequences",
                "effect_rows_csv": process_outputs["effect_rows_csv"],
                "failure_csv": process_outputs["failure_csv"],
            }
        ],
        output_root=tmp_path / "outputs",
    )

    assert Path(tmp_path / "outputs" / "sequence_process_family_cross_dataset.csv").exists()
    assert Path(tmp_path / "outputs" / "sequence_process_family_cross_dataset.md").exists()
    assert Path(decoder_outputs["summary_csv"]).exists()
    assert Path(decoder_outputs["plot"]).exists()
    assert Path(decoder_outputs["real_vs_synthetic_csv"]).exists()


def test_sequence_cross_dataset_summary_marks_real_grouped_scope(tmp_path):
    cluster = _run_fixture_sequence_branch(
        tmp_path,
        "real_grouped_cluster",
        "cluster_distance",
        task_name="real_grouped_manifest_sequences",
    )
    calibrated = _run_fixture_sequence_branch(
        tmp_path,
        "real_grouped_calibrated",
        "calibrated_classifier",
        task_name="real_grouped_manifest_sequences",
    )
    outputs = build_sequence_cross_dataset_outputs(
        [
            {
                "dataset_label": "real_grouped_fixture",
                "task_label": "real_grouped_manifest_sequences",
                "cluster_distance_run": cluster["run_dir"],
                "calibrated_classifier_run": calibrated["run_dir"],
            }
        ],
        output_root=tmp_path / "outputs",
    )

    summary_text = Path(outputs["summary_md"]).read_text(encoding="utf-8")

    assert "preliminary grouped evidence" in summary_text
    assert "synthetic-from-real sequence findings" not in summary_text
