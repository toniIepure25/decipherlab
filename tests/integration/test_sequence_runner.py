from __future__ import annotations

from pathlib import Path

from decipherlab.config import (
    DecodingConfig,
    RealDownstreamConfig,
    RiskControlConfig,
    SequenceBenchmarkConfig,
    StructuredUncertaintyConfig,
)
from decipherlab.sequence.runner import run_sequence_branch_experiment
from tests.helpers import build_test_config, create_real_manifest_fixture


def test_sequence_branch_runner_writes_expected_artifacts(tmp_path):
    manifest_path = create_real_manifest_fixture(tmp_path)
    base_config = build_test_config(tmp_path)
    config = base_config.model_copy(
        update={
            "experiment": base_config.experiment.model_copy(
                update={"name": "sequence_branch_fixture", "output_root": tmp_path / "runs"}
            ),
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
                selected_symbol_count=4,
                min_instances_per_symbol=2,
                train_sequences=6,
                val_sequences=3,
                test_sequences=3,
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
                decoder_variants=["bigram_beam", "crf_viterbi"],
                beam_width=6,
                lm_weight=1.2,
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
            "real_downstream": RealDownstreamConfig(
                enabled=True,
                task_name="train_supported_ngram_path",
                transcript_top_k=3,
                min_frequency=1,
                exact_length_only=True,
                missing_log_probability=1.0e-8,
                ngram_order=2,
                min_supported_ngrams=1,
            ),
        }
    )

    result = run_sequence_branch_experiment(config)
    methods = {row["method"] for row in result["summary_rows"]}

    assert methods == {
        "fixed_greedy",
        "uncertainty_beam",
        "conformal_beam",
        "uncertainty_crf_viterbi",
        "conformal_crf_viterbi",
    }
    assert Path(result["run_dir"], "sequence_branch_summary.csv").exists()
    assert Path(result["run_dir"], "sequence_branch_report.md").exists()
    assert Path(result["run_dir"], "sequence_pairwise_effects.csv").exists()
    assert Path(result["run_dir"], "sequence_ambiguity_regime_table.csv").exists()
    assert Path(result["run_dir"], "sequence_failure_summary.csv").exists()
    assert all("sequence_topk_recovery" in row for row in result["summary_rows"])


def test_sequence_branch_runner_supports_trigram_and_process_family_task(tmp_path):
    manifest_path = create_real_manifest_fixture(tmp_path / "process_family")
    base_config = build_test_config(tmp_path / "process_family")
    config = base_config.model_copy(
        update={
            "experiment": base_config.experiment.model_copy(
                update={"name": "sequence_branch_process_family_fixture", "output_root": tmp_path / "runs"}
            ),
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
            "evaluation": base_config.evaluation.model_copy(update={"ambiguity_levels": [0.15], "top_k": 3}),
            "sequence_benchmark": SequenceBenchmarkConfig(
                enabled=True,
                task_name="real_glyph_process_family_sequences",
                selected_symbol_count=4,
                min_instances_per_symbol=2,
                train_sequences=9,
                val_sequences=6,
                test_sequences=6,
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
            "real_downstream": RealDownstreamConfig(
                enabled=True,
                task_name="train_transcript_bank",
                transcript_top_k=3,
                min_frequency=1,
                exact_length_only=True,
                missing_log_probability=1.0e-8,
            ),
        }
    )

    result = run_sequence_branch_experiment(config, strategy_override="calibrated_classifier")
    methods = {row["method"] for row in result["summary_rows"]}

    assert methods == {
        "fixed_greedy",
        "uncertainty_beam",
        "conformal_beam",
        "uncertainty_trigram_beam",
        "conformal_trigram_beam",
        "uncertainty_crf_viterbi",
        "conformal_crf_viterbi",
    }
    assert any(row["family_identification_accuracy"] is not None for row in result["summary_rows"])


def test_sequence_branch_runner_supports_real_grouped_manifest_task(tmp_path):
    manifest_path = create_real_manifest_fixture(tmp_path / "real_grouped")
    base_config = build_test_config(tmp_path / "real_grouped")
    config = base_config.model_copy(
        update={
            "experiment": base_config.experiment.model_copy(
                update={"name": "sequence_branch_real_grouped_fixture", "output_root": tmp_path / "runs"}
            ),
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
                task_name="real_grouped_manifest_sequences",
                selected_symbol_count=4,
                min_instances_per_symbol=2,
                train_sequences=1,
                val_sequences=1,
                test_sequences=1,
                sequence_length=6,
                group_count=2,
                self_transition_bias=3.0,
                within_group_bias=1.5,
                cross_group_bias=0.35,
                transition_noise=0.05,
                sample_with_replacement=False,
                minimum_real_sequence_length=4,
                maximum_real_sequence_length=6,
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
                decoder_variants=["bigram_beam", "crf_viterbi"],
                beam_width=6,
                lm_weight=1.2,
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
            "real_downstream": RealDownstreamConfig(
                enabled=True,
                task_name="train_transcript_bank",
                transcript_top_k=3,
                min_frequency=1,
                exact_length_only=True,
                missing_log_probability=1.0e-8,
            ),
        }
    )

    result = run_sequence_branch_experiment(config, strategy_override="calibrated_classifier")
    report_text = Path(result["run_dir"], "sequence_branch_report.md").read_text(encoding="utf-8")

    assert result["benchmark"]["metadata"]["synthetic_from_real"] is False
    assert "preliminary grouped evidence" in report_text
    assert any(row["real_downstream_exact_match"] is not None for row in result["summary_rows"])
    assert any(row["real_downstream_bank_coverage"] is not None for row in result["summary_rows"])
