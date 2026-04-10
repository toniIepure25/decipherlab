from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from decipherlab.utils.io import ensure_directory, write_csv, write_text


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return float(stripped)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def discover_latest_process_family_runs(outputs_root: str | Path = "outputs/runs") -> list[Path]:
    root = Path(outputs_root)
    patterns = [
        "*sequence_omniglot_process_family_sequence_branch_cluster_distance",
        "*sequence_omniglot_process_family_sequence_branch_calibrated_classifier",
        "*sequence_sklearn_digits_process_family_sequence_branch_cluster_distance",
        "*sequence_sklearn_digits_process_family_sequence_branch_calibrated_classifier",
        "*sequence_kuzushiji_process_family_sequence_branch_cluster_distance",
        "*sequence_kuzushiji_process_family_sequence_branch_calibrated_classifier",
    ]
    run_dirs: list[Path] = []
    for pattern in patterns:
        candidates = sorted(root.glob(pattern))
        if candidates:
            run_dirs.append(candidates[-1])
    return run_dirs


def build_process_family_sensitivity_rows(run_dirs: list[Path]) -> list[dict[str, Any]]:
    grouped_rows: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for run_dir in run_dirs:
        csv_path = run_dir / "sequence_branch_examples.csv"
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                continue
            required = {
                "task_name",
                "dataset_name",
                "posterior_strategy_requested",
                "true_family",
                "example_id",
                "ambiguity_level",
                "method",
                "family_identification_accuracy",
                "sequence_exact_match",
                "sequence_token_accuracy",
            }
            if not required.issubset(set(reader.fieldnames)):
                continue
            rows = [dict(row) for row in reader]

        if not rows or rows[0]["task_name"] != "real_glyph_process_family_sequences":
            continue

        example_groups: dict[tuple[str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
        for row in rows:
            key = (
                row["posterior_strategy_requested"],
                row["ambiguity_level"],
                row["example_id"],
            )
            example_groups[key][row["method"]] = row

        for methods in example_groups.values():
            fixed = methods.get("fixed_greedy")
            uncertainty = methods.get("uncertainty_beam")
            if fixed is None or uncertainty is None:
                continue

            family = str(fixed["true_family"])
            dataset_name = str(fixed["dataset_name"])
            posterior_strategy = str(fixed["posterior_strategy_requested"])
            bucket = grouped_rows[(dataset_name, posterior_strategy, family)]

            fixed_family = _to_float(fixed.get("family_identification_accuracy")) or 0.0
            uncertainty_family = _to_float(uncertainty.get("family_identification_accuracy")) or 0.0
            fixed_exact = _to_float(fixed.get("sequence_exact_match")) or 0.0
            uncertainty_exact = _to_float(uncertainty.get("sequence_exact_match")) or 0.0
            fixed_token = _to_float(fixed.get("sequence_token_accuracy")) or 0.0
            uncertainty_token = _to_float(uncertainty.get("sequence_token_accuracy")) or 0.0

            bucket["uncertainty_family_delta"].append(uncertainty_family - fixed_family)
            bucket["uncertainty_sequence_exact_delta"].append(uncertainty_exact - fixed_exact)
            bucket["uncertainty_token_accuracy_delta"].append(uncertainty_token - fixed_token)

            trigram = methods.get("uncertainty_trigram_beam")
            if trigram is not None:
                trigram_family = _to_float(trigram.get("family_identification_accuracy")) or 0.0
                bucket["trigram_family_delta"].append(trigram_family - uncertainty_family)

            conformal = methods.get("conformal_beam")
            if conformal is not None:
                conformal_family = _to_float(conformal.get("family_identification_accuracy")) or 0.0
                bucket["conformal_family_delta"].append(conformal_family - uncertainty_family)

            crf = methods.get("uncertainty_crf_viterbi")
            if crf is not None:
                crf_family = _to_float(crf.get("family_identification_accuracy")) or 0.0
                bucket["crf_family_delta"].append(crf_family - uncertainty_family)

            bucket["example_count"].append(1.0)

    rows: list[dict[str, Any]] = []
    for (dataset_name, posterior_strategy, family), metrics in sorted(grouped_rows.items()):
        rows.append(
            {
                "dataset_name": dataset_name,
                "posterior_strategy_requested": posterior_strategy,
                "true_family": family,
                "example_count": int(sum(metrics["example_count"])),
                "mean_uncertainty_family_delta": _mean(metrics["uncertainty_family_delta"]),
                "mean_uncertainty_sequence_exact_delta": _mean(
                    metrics["uncertainty_sequence_exact_delta"]
                ),
                "mean_uncertainty_token_accuracy_delta": _mean(
                    metrics["uncertainty_token_accuracy_delta"]
                ),
                "mean_trigram_family_delta": _mean(metrics["trigram_family_delta"]),
                "mean_conformal_family_delta": _mean(metrics["conformal_family_delta"]),
                "mean_crf_family_delta": _mean(metrics["crf_family_delta"]),
            }
        )
    return rows


def build_process_family_sensitivity_outputs(
    run_dirs: list[Path],
    output_root: str | Path = "outputs",
) -> dict[str, str]:
    output_dir = ensure_directory(output_root)
    rows = build_process_family_sensitivity_rows(run_dirs)

    csv_path = output_dir / "sequence_family_sensitivity_summary.csv"
    md_path = output_dir / "sequence_family_sensitivity_summary.md"

    write_csv(csv_path, rows)

    alternating_rows = [
        row for row in rows if row["true_family"] == "alternating_markov"
    ]
    motif_rows = [row for row in rows if row["true_family"] == "motif_repeat"]
    sticky_rows = [row for row in rows if row["true_family"] == "sticky_markov"]

    def _fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    alternating_best = max(
        alternating_rows,
        key=lambda row: row["mean_uncertainty_family_delta"]
        if row["mean_uncertainty_family_delta"] is not None
        else float("-inf"),
        default=None,
    )
    motif_worst = min(
        motif_rows,
        key=lambda row: row["mean_uncertainty_family_delta"]
        if row["mean_uncertainty_family_delta"] is not None
        else float("inf"),
        default=None,
    )

    markdown = [
        "# Sequence Family Sensitivity Summary",
        "",
        "This analysis stays within the existing synthetic-from-real `real_glyph_process_family_sequences` benchmark.",
        "",
        "## Main Pattern",
        "",
        "- `alternating_markov` is the most uncertainty-friendly family in the current branch.",
        "- `motif_repeat` remains the hardest family and often turns symbol rescue into weak or negative family deltas.",
        "- `sticky_markov` is mixed: sometimes positive, sometimes close to zero, and occasionally negative.",
        "",
        "## Representative Extremes",
        "",
    ]
    if alternating_best is not None:
        markdown.append(
            f"- Strongest alternating-markov gain: `{alternating_best['dataset_name']}` / "
            f"`{alternating_best['posterior_strategy_requested']}` with mean uncertainty family delta "
            f"`{_fmt(alternating_best['mean_uncertainty_family_delta'])}`."
        )
    if motif_worst is not None:
        markdown.append(
            f"- Weakest motif-repeat result: `{motif_worst['dataset_name']}` / "
            f"`{motif_worst['posterior_strategy_requested']}` with mean uncertainty family delta "
            f"`{_fmt(motif_worst['mean_uncertainty_family_delta'])}`."
        )
    markdown.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The downstream gain is not family-universal.",
            "- The branch currently benefits most when the process family exposes alternating local structure.",
            "- Harder repeat-heavy families remain a boundary case where preserved symbol uncertainty often fails to become stable higher-level recovery.",
        ]
    )
    write_text(md_path, "\n".join(markdown) + "\n")

    return {"csv": str(csv_path), "markdown": str(md_path)}
