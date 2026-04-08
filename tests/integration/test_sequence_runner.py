from __future__ import annotations

from pathlib import Path

from decipherlab.config import (
    DecodingConfig,
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
        }
    )

    result = run_sequence_branch_experiment(config)
    methods = {row["method"] for row in result["summary_rows"]}

    assert methods == {"fixed_greedy", "uncertainty_beam", "conformal_beam"}
    assert Path(result["run_dir"], "sequence_branch_summary.csv").exists()
    assert Path(result["run_dir"], "sequence_branch_report.md").exists()
    assert Path(result["run_dir"], "sequence_pairwise_effects.csv").exists()
    assert Path(result["run_dir"], "sequence_ambiguity_regime_table.csv").exists()
    assert Path(result["run_dir"], "sequence_failure_summary.csv").exists()
    assert all("sequence_topk_recovery" in row for row in result["summary_rows"])
