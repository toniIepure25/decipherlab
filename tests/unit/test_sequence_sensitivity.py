from __future__ import annotations

from pathlib import Path

from decipherlab.sequence.sensitivity import build_process_family_sensitivity_rows


def test_build_process_family_sensitivity_rows_recovers_family_deltas(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    csv_path = run_dir / "sequence_branch_examples.csv"
    csv_path.write_text(
        "\n".join(
            [
                "task_name,dataset_name,posterior_strategy_requested,true_family,ambiguity_level,example_id,method,family_identification_accuracy,sequence_exact_match,sequence_token_accuracy",
                "real_glyph_process_family_sequences,demo,cluster_distance,alternating_markov,0.3,ex1,fixed_greedy,0,0,0.5",
                "real_glyph_process_family_sequences,demo,cluster_distance,alternating_markov,0.3,ex1,uncertainty_beam,1,1,0.75",
                "real_glyph_process_family_sequences,demo,cluster_distance,alternating_markov,0.3,ex1,uncertainty_trigram_beam,1,1,0.75",
                "real_glyph_process_family_sequences,demo,cluster_distance,alternating_markov,0.3,ex1,conformal_beam,1,1,0.75",
                "real_glyph_process_family_sequences,demo,cluster_distance,alternating_markov,0.3,ex1,uncertainty_crf_viterbi,0,1,0.75",
                "real_glyph_process_family_sequences,demo,cluster_distance,motif_repeat,0.3,ex2,fixed_greedy,1,1,0.75",
                "real_glyph_process_family_sequences,demo,cluster_distance,motif_repeat,0.3,ex2,uncertainty_beam,0,0,0.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = build_process_family_sensitivity_rows([run_dir])
    by_family = {row["true_family"]: row for row in rows}

    assert by_family["alternating_markov"]["mean_uncertainty_family_delta"] == 1.0
    assert by_family["alternating_markov"]["mean_uncertainty_sequence_exact_delta"] == 1.0
    assert by_family["alternating_markov"]["mean_trigram_family_delta"] == 0.0
    assert by_family["alternating_markov"]["mean_conformal_family_delta"] == 0.0
    assert by_family["alternating_markov"]["mean_crf_family_delta"] == -1.0
    assert by_family["motif_repeat"]["mean_uncertainty_family_delta"] == -1.0
