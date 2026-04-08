from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from decipherlab.utils.io import ensure_directory, write_text


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or stripped.lower() == "n/a":
        return None
    return float(stripped)


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _condition_lookup(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["condition"]: row for row in rows}


def _failure_count(rows: list[dict[str, str]], case_type: str) -> int:
    return sum(int(row["count"]) for row in rows if row["case_type"] == case_type)


def build_experiments_section(run_dir: str | Path) -> str:
    run_path = Path(run_dir)
    metadata = _read_json(run_path / "experiment_metadata.json")
    dataset_summary_path = run_path / "dataset_summary.json"
    dataset_summary = _read_json(dataset_summary_path) if dataset_summary_path.exists() else None
    lines = [
        "# Experiments",
        "",
        "## Protocol",
        "- We evaluate the existing four-condition protocol: fixed vs uncertainty-aware inference crossed with heuristic vs calibrated posterior generation.",
        f"- Ambiguity levels: `{metadata['ambiguity_levels']}`.",
        f"- Evaluation top-k: `{metadata['evaluation_top_k']}`.",
        f"- Bootstrap procedure: `{metadata['bootstrap_trials']}` trials at confidence `{metadata['bootstrap_confidence_level']}`.",
        f"- Seed sweep: base seed `{metadata['base_seed']}` with additional seeds `{metadata['seed_sweep']}`.",
        "",
        "## Dataset",
        f"- Source: `{metadata['dataset_source']}`.",
        f"- Manifest: `{metadata['dataset_manifest']}`.",
        f"- Splits: train=`{metadata['train_split']}`, val=`{metadata['val_split']}`, test=`{metadata['evaluation_split']}`.",
    ]
    if dataset_summary is not None:
        lines.extend(
            [
                f"- Sequence count: `{dataset_summary['sequence_count']}`.",
                f"- Record count: `{dataset_summary['record_count']}`.",
                f"- Split composition: `{dataset_summary['split_sequence_counts']}`.",
                f"- Split label coverage: `{dataset_summary['split_label_coverage']}`.",
                f"- Group counts by split: `{dataset_summary['split_group_counts']}`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Reporting",
            "- Primary symbol-level metrics: top-1, top-k, NLL, and ECE.",
            "- Downstream metrics are reported only when labels permit them.",
            "- Failure summaries remain part of the main evidence pack rather than an omitted appendix artifact.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_results_section(run_dir: str | Path) -> str:
    run_path = Path(run_dir)
    main_rows = _read_csv_rows(run_path / "main_comparison_with_ci.csv")
    pairwise_rows = _read_csv_rows(run_path / "pairwise_effect_summary.csv")
    failure_rows = _read_csv_rows(run_path / "failure_case_summary.csv")
    sequence_rows = _read_csv_rows(run_path / "sequence_level_summary.csv")

    by_condition = _condition_lookup(main_rows)
    baseline = by_condition.get("A. Fixed Transcript + Heuristic Posterior")
    combined = by_condition.get("D. Uncertainty-Aware + Calibrated Posterior")
    highest_pairwise = max(pairwise_rows, key=lambda row: _to_float(row["ambiguity_level"]) or 0.0) if pairwise_rows else None
    max_sequence_rescue = max((_to_float(row.get("symbol_rescue_sequences")) or 0.0 for row in sequence_rows), default=0.0)

    lines = [
        "# Results",
        "",
        f"This section was generated from `{run_path}` and summarizes only the measured outputs in the current evidence pack.",
        "",
        "## Main Comparison",
    ]
    if baseline is not None and combined is not None:
        lines.extend(
            [
                f"- `{baseline['condition']}` reached mean symbol top-k accuracy {_fmt(_to_float(baseline['mean_symbol_topk_accuracy']))} "
                f"[{_fmt(_to_float(baseline['mean_symbol_topk_accuracy_ci_lower']))}, {_fmt(_to_float(baseline['mean_symbol_topk_accuracy_ci_upper']))}].",
                f"- `{combined['condition']}` reached mean symbol top-k accuracy {_fmt(_to_float(combined['mean_symbol_topk_accuracy']))} "
                f"[{_fmt(_to_float(combined['mean_symbol_topk_accuracy_ci_lower']))}, {_fmt(_to_float(combined['mean_symbol_topk_accuracy_ci_upper']))}].",
                f"- The heuristic fixed baseline had mean symbol NLL {_fmt(_to_float(baseline['mean_symbol_negative_log_likelihood']))}, "
                f"while the combined condition had {_fmt(_to_float(combined['mean_symbol_negative_log_likelihood']))}.",
            ]
        )
    if highest_pairwise is not None:
        lines.extend(
            [
                "",
                "## Ambiguity Sensitivity",
                f"- At the highest ambiguity level tested ({_fmt(_to_float(highest_pairwise['ambiguity_level']))}), "
                f"the combined condition changed symbol top-k accuracy by {_fmt(_to_float(highest_pairwise['combined_topk_delta']))} on average across seeds.",
                f"- The same condition changed NLL by {_fmt(_to_float(highest_pairwise['combined_nll_delta']))}.",
            ]
        )
    lines.extend(
        [
            "",
            "## Downstream And Failure Analysis",
            f"- Maximum per-condition sequence rescue count was `{int(max_sequence_rescue)}` sequence(s).",
            f"- `top1_collapse_but_topk_rescue` cases observed: `{_failure_count(failure_rows, 'top1_collapse_but_topk_rescue')}`.",
            f"- `uncertainty_helped_symbols_not_downstream` cases observed: `{_failure_count(failure_rows, 'uncertainty_helped_symbols_not_downstream')}`.",
            f"- `calibration_worsened_or_unstable` cases observed: `{_failure_count(failure_rows, 'calibration_worsened_or_unstable')}`.",
            "",
            "These results support narrow statements about symbol-level ambiguity robustness under the tested protocol. They should not be read as evidence of historical semantic decipherment.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_limitations_section(run_dir: str | Path) -> str:
    run_path = Path(run_dir)
    main_rows = _read_csv_rows(run_path / "main_comparison_with_ci.csv")
    failure_rows = _read_csv_rows(run_path / "failure_case_summary.csv")
    family_supported = any((_to_float(row.get("mean_family_topk_accuracy")) or 0.0) > 0.0 for row in main_rows)
    lines = [
        "# Limitations",
        "",
        "- The present evidence supports symbol-level retention of correct alternatives under ambiguity, not full decipherment or semantic recovery.",
    ]
    if not family_supported:
        lines.append("- Downstream family-level gains are not established on the current run because family top-k accuracy remains absent or negligible.")
    if _failure_count(failure_rows, "calibration_worsened_or_unstable") > 0:
        lines.append("- Calibration is not uniformly beneficial: at least one measured condition showed worse or unstable calibration-related behavior.")
    if _failure_count(failure_rows, "uncertainty_helped_symbols_not_downstream") > 0:
        lines.append("- Symbol-level rescue does not automatically propagate to downstream reasoning; the evidence pack explicitly records such failure cases.")
    lines.extend(
        [
            "- The current conclusions are bounded by the supplied manifest labels, split quality, and crop selection.",
            "- Stronger claims require a second real dataset or a sequence-rich corpus with enough downstream labels or grouped structure to test whether preserved alternatives improve higher-level inference.",
        ]
    )
    return "\n".join(lines) + "\n"


def assemble_paper_sections(run_dir: str | Path, paper_dir: str | Path = "paper") -> dict[str, str]:
    output_dir = ensure_directory(paper_dir)
    experiments_text = build_experiments_section(run_dir)
    results_text = build_results_section(run_dir)
    limitations_text = build_limitations_section(run_dir)
    write_text(output_dir / "EXPERIMENTS.md", experiments_text)
    write_text(output_dir / "RESULTS.md", results_text)
    write_text(output_dir / "LIMITATIONS.md", limitations_text)
    return {
        "experiments": str(output_dir / "EXPERIMENTS.md"),
        "results": str(output_dir / "RESULTS.md"),
        "limitations": str(output_dir / "LIMITATIONS.md"),
    }
