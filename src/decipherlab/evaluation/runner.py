from __future__ import annotations

from pathlib import Path
from typing import Any

from decipherlab.config import DecipherLabConfig, dump_config, load_config
from decipherlab.evaluation.failure_analysis import analyze_failure_cases
from decipherlab.evaluation.results_pack import (
    build_pairwise_effect_rows,
    condition_label,
    write_results_pack,
)
from decipherlab.pipeline import run_pipeline
from decipherlab.utils.io import ensure_directory, write_json, write_text
from decipherlab.utils.logging import configure_logging
from decipherlab.utils.runtime import prepare_run_context


def _level_label(value: float) -> str:
    return f"{value:.2f}".replace(".", "_")


def _metric_value(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _resolve_ambiguity_levels(config: DecipherLabConfig) -> list[float]:
    if config.evaluation.ambiguity_levels:
        return config.evaluation.ambiguity_levels
    if config.evaluation.noise_levels:
        return config.evaluation.noise_levels
    return [0.0]


def _resolve_seed_sweep(config: DecipherLabConfig) -> list[int]:
    if config.experiment.seed_sweep:
        seen: list[int] = []
        for seed in [config.experiment.seed, *config.experiment.seed_sweep]:
            if seed not in seen:
                seen.append(seed)
        return seen
    return [config.experiment.seed]


def _build_summary_markdown(comparisons: list[dict[str, Any]]) -> str:
    lines = [
        "# DecipherLab Evaluation Summary",
        "",
        "## Comparison Sweep",
        "| Seed | Ambiguity | Condition | Symbol Top-1 | Symbol Top-k | NLL | ECE | Family Top-k | Structural Error |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for comparison in sorted(
        comparisons,
        key=lambda row: (row["seed"], row["ambiguity_level"], row["condition"]),
    ):
        lines.append(
            "| {seed} | {ambiguity:.2f} | {condition} | {top1} | {topk} | {nll} | {ece} | {family_topk} | {struct_error} |".format(
                seed=comparison["seed"],
                ambiguity=comparison["ambiguity_level"],
                condition=comparison["condition"],
                top1=_metric_value(comparison["symbol_top1_accuracy"]),
                topk=_metric_value(comparison["symbol_topk_accuracy"]),
                nll=_metric_value(comparison["symbol_negative_log_likelihood"]),
                ece=_metric_value(comparison["symbol_expected_calibration_error"]),
                family_topk=_metric_value(comparison["family_topk_accuracy"]),
                struct_error=_metric_value(comparison["mean_structural_recovery_error"]),
            )
        )
    return "\n".join(lines) + "\n"


def _build_paper_summary_markdown(
    pairwise_rows: list[dict[str, Any]],
    failure_payload: dict[str, Any],
) -> str:
    aggregated_pairwise: dict[float, list[dict[str, Any]]] = {}
    for row in pairwise_rows:
        aggregated_pairwise.setdefault(row["ambiguity_level"], []).append(row)
    lines = [
        "# Paper-Oriented Experiment Summary",
        "",
        "## Protocol",
        "- Compare fixed and uncertainty-aware inference under the same ambiguity sweep.",
        "- Cross both inference modes with two posterior-generation strategies: heuristic distance and calibrated classification.",
        "- Keep splits, ambiguity levels, and reporting schema fixed across all four conditions.",
        "- Aggregate robustness across configured seeds without changing the underlying train/validation/evaluation protocol.",
        "",
        "## Pairwise Effects",
        "| Ambiguity | Heuristic Uncertainty Top-k Delta | Calibrated Uncertainty Top-k Delta | Fixed Calibration Top-k Delta | Uncertainty Calibration Top-k Delta | Combined Top-k Delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for ambiguity_level in sorted(aggregated_pairwise):
        rows = aggregated_pairwise[ambiguity_level]
        lines.append(
            "| {ambiguity:.2f} | {hu} | {cu} | {fc} | {uc} | {combined} |".format(
                ambiguity=ambiguity_level,
                hu=_metric_value(
                    sum(row["heuristic_uncertainty_topk_delta"] for row in rows if row["heuristic_uncertainty_topk_delta"] is not None)
                    / len(rows)
                ),
                cu=_metric_value(
                    sum(row["calibrated_uncertainty_topk_delta"] for row in rows if row["calibrated_uncertainty_topk_delta"] is not None)
                    / len(rows)
                ),
                fc=_metric_value(
                    sum(row["fixed_calibration_topk_delta"] for row in rows if row["fixed_calibration_topk_delta"] is not None)
                    / len(rows)
                ),
                uc=_metric_value(
                    sum(row["uncertainty_calibration_topk_delta"] for row in rows if row["uncertainty_calibration_topk_delta"] is not None)
                    / len(rows)
                ),
                combined=_metric_value(
                    sum(row["combined_topk_delta"] for row in rows if row["combined_topk_delta"] is not None)
                    / len(rows)
                ),
            )
        )
    if failure_payload["summary_rows"]:
        lines.extend(
            [
                "",
                "## Failure Analysis Counts",
                "| Case Type | Ambiguity | Condition Group | Count |",
                "| --- | ---: | --- | ---: |",
            ]
        )
        for row in failure_payload["summary_rows"]:
            lines.append(
                "| {case_type} | {ambiguity:.2f} | {condition_group} | {count} |".format(
                    case_type=row["case_type"],
                    ambiguity=row["ambiguity_level"],
                    condition_group=row["condition_group"],
                    count=row["count"],
                )
            )
    lines.extend(
        [
            "",
            "## Guardrail",
            "- These outputs support narrow empirical claims about the tested ambiguity protocol and dataset labels.",
            "- They do not support broad historical-generalization claims without a larger real dataset.",
        ]
    )
    return "\n".join(lines) + "\n"


def _aggregate_failure_payloads(
    seed_cubes: dict[int, dict[float, dict[str, dict[str, dict[str, Any]]]]],
    top_k: int,
    overdiffuse_entropy_ratio: float,
) -> dict[str, Any]:
    failure_cases: list[dict[str, Any]] = []
    grouped: dict[tuple[str, float, str], int] = {}
    for seed, results_cube in sorted(seed_cubes.items()):
        payload = analyze_failure_cases(
            results_cube,
            top_k=top_k,
            overdiffuse_entropy_ratio=overdiffuse_entropy_ratio,
        )
        for case in payload["failure_cases"]:
            enriched = dict(case)
            enriched["seed"] = seed
            failure_cases.append(enriched)
        for row in payload["summary_rows"]:
            key = (row["case_type"], row["ambiguity_level"], row["condition_group"])
            grouped[key] = grouped.get(key, 0) + row["count"]
    summary_rows = [
        {
            "case_type": case_type,
            "ambiguity_level": ambiguity_level,
            "condition_group": condition_group,
            "count": count,
        }
        for (case_type, ambiguity_level, condition_group), count in sorted(grouped.items())
    ]
    return {"failure_cases": failure_cases, "summary_rows": summary_rows}


def run_ablation_suite(config: DecipherLabConfig | str | Path) -> dict[str, object]:
    resolved = load_config(config) if isinstance(config, (str, Path)) else config
    evaluation_context = prepare_run_context(resolved, suffix="evaluation")
    logger = configure_logging(evaluation_context.run_dir)
    dump_config(resolved, evaluation_context.run_dir / "config.yaml")

    comparisons: list[dict[str, Any]] = []
    comparison_cells: list[dict[str, Any]] = []
    seed_cubes: dict[int, dict[float, dict[str, dict[str, dict[str, Any]]]]] = {}

    for seed in _resolve_seed_sweep(resolved):
        seed_config = resolved.model_copy(deep=True)
        seed_config.experiment.seed = seed
        seed_config.experiment.output_root = evaluation_context.run_dir / "seed_runs"
        seed_config.experiment.name = f"{resolved.experiment.name}_seed_{seed}"
        seed_cubes[seed] = {}
        for ambiguity_level in _resolve_ambiguity_levels(seed_config):
            seed_cubes[seed][ambiguity_level] = {}
            for strategy in seed_config.evaluation.comparison_strategies:
                seed_cubes[seed][ambiguity_level][strategy] = {}
                for posterior_mode in ("fixed", "uncertainty"):
                    logger.info(
                        "Running %s mode with %s strategy at ambiguity %.2f for seed %s",
                        posterior_mode,
                        strategy,
                        ambiguity_level,
                        seed,
                    )
                    result = run_pipeline(
                        seed_config,
                        posterior_mode=posterior_mode,
                        suffix=f"{strategy}_{posterior_mode}_ambiguity_{_level_label(ambiguity_level)}",
                        ambiguity_level=ambiguity_level,
                        strategy_override=strategy,
                    )
                    summary = dict(result["summary"])
                    summary["seed"] = seed
                    summary["ambiguity_level"] = ambiguity_level
                    summary["posterior_strategy_requested"] = strategy
                    summary["condition"] = condition_label(strategy, posterior_mode)
                    summary["run_dir"] = str(result["run_dir"])
                    comparisons.append(summary)
                    comparison_cells.append(
                        {
                            "seed": seed,
                            "ambiguity_level": ambiguity_level,
                            "posterior_mode": posterior_mode,
                            "posterior_strategy_requested": strategy,
                            "condition": summary["condition"],
                            "run_dir": str(result["run_dir"]),
                            "summary": summary,
                            "example_payloads": result["example_payloads"],
                        }
                    )
                    seed_cubes[seed][ambiguity_level][strategy][posterior_mode] = result

    ensure_directory(evaluation_context.run_dir)
    pairwise_rows: list[dict[str, Any]] = []
    for seed, results_cube in sorted(seed_cubes.items()):
        for row in build_pairwise_effect_rows(results_cube):
            pairwise_rows.append({"seed": seed, **row})
    failure_payload = _aggregate_failure_payloads(
        seed_cubes,
        top_k=resolved.evaluation.top_k,
        overdiffuse_entropy_ratio=resolved.evaluation.overdiffuse_entropy_ratio,
    )

    write_json(evaluation_context.run_dir / "comparison_runs.json", comparisons)
    write_text(evaluation_context.run_dir / "evaluation_summary.md", _build_summary_markdown(comparisons))
    write_json(evaluation_context.run_dir / "pairwise_summary.json", pairwise_rows)
    write_text(
        evaluation_context.run_dir / "paper_experiment_summary.md",
        _build_paper_summary_markdown(pairwise_rows, failure_payload),
    )
    results_pack = write_results_pack(
        run_dir=evaluation_context.run_dir,
        config=resolved,
        comparisons=comparisons,
        comparison_cells=comparison_cells,
        pairwise_rows=pairwise_rows,
        failure_payload=failure_payload,
    )

    return {
        "run_dir": evaluation_context.run_dir,
        "comparisons": comparisons,
        "comparison_cells": comparison_cells,
        "pairwise_rows": pairwise_rows,
        "failure_payload": failure_payload,
        "results_pack": results_pack,
    }
