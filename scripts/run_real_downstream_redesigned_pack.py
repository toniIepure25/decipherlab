from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/decipherlab-sequence-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from decipherlab.sequence.runner import run_sequence_branch_experiment
from decipherlab.utils.io import ensure_directory, write_csv, write_text


CONFIGS = [
    (
        "historical_newspapers_real_grouped_gold",
        Path("configs/experiments/sequence_historical_newspapers_real_downstream_redesigned.yaml"),
    ),
    (
        "scadsai_real_grouped",
        Path("configs/experiments/sequence_scadsai_real_downstream_redesigned.yaml"),
    ),
]


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return float(stripped)


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _mean(values: list[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return float(sum(filtered) / len(filtered))


def _build_plot(rows: list[dict[str, object]], output_path: Path) -> None:
    labels = [f"{row['dataset']}:{row['posterior_strategy_requested']}" for row in rows]
    metrics = [
        ("mean_downstream_coverage_fraction", "Downstream Coverage"),
        ("mean_uncertainty_downstream_exact_delta", "Uncertainty Downstream Exact"),
        ("mean_uncertainty_downstream_token_accuracy_delta", "Uncertainty Downstream Token"),
        ("mean_conformal_downstream_exact_delta", "Conformal Downstream Exact"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), squeeze=False)
    x = np.arange(len(labels))
    colors = ["#4c78a8" if "historical" in str(label) else "#e45756" for label in labels]
    for axis, (metric_key, title) in zip(axes.flatten(), metrics):
        values = [0.0 if row[metric_key] is None else float(row[metric_key]) for row in rows]
        axis.bar(x, values, color=colors)
        axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=20, ha="right")
        axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    summary_rows: list[dict[str, object]] = []
    for dataset_label, config_path in CONFIGS:
        for strategy in ["cluster_distance", "calibrated_classifier"]:
            result = run_sequence_branch_experiment(config_path, strategy_override=strategy)
            summary = _read_csv(Path(result["run_dir"]) / "sequence_branch_summary.csv")
            pairwise = _read_csv(Path(result["run_dir"]) / "sequence_pairwise_effects.csv")
            failure_rows = _read_csv(Path(result["run_dir"]) / "sequence_failure_summary.csv")
            examples = _read_csv(Path(result["run_dir"]) / "sequence_branch_examples.csv")
            fixed_rows = [row for row in summary if row["method"] == "fixed_greedy"]
            uncertainty_rows = [row for row in summary if row["method"] == "uncertainty_beam"]
            summary_rows.append(
                {
                    "dataset": dataset_label,
                    "posterior_strategy_requested": strategy,
                    "mean_downstream_coverage_fraction": _mean([_to_float(row["real_downstream_bank_coverage"]) for row in fixed_rows]),
                    "mean_downstream_exact_if_fully_covered": _mean(
                        [_to_float(row["real_downstream_exact_match_if_covered"]) for row in uncertainty_rows]
                    ),
                    "mean_fixed_downstream_exact_match": _mean(
                        [_to_float(row["real_downstream_exact_match"]) for row in fixed_rows]
                    ),
                    "mean_uncertainty_downstream_exact_match": _mean(
                        [_to_float(row["real_downstream_exact_match"]) for row in uncertainty_rows]
                    ),
                    "mean_uncertainty_downstream_exact_delta": _mean(
                        [_to_float(row["uncertainty_downstream_exact_delta"]) for row in pairwise]
                    ),
                    "mean_uncertainty_downstream_topk_delta": _mean(
                        [_to_float(row["uncertainty_downstream_topk_delta"]) for row in pairwise]
                    ),
                    "mean_uncertainty_downstream_token_accuracy_delta": _mean(
                        [_to_float(row["uncertainty_downstream_token_accuracy_delta"]) for row in pairwise]
                    ),
                    "mean_conformal_downstream_exact_delta": _mean(
                        [_to_float(row["conformal_downstream_exact_delta"]) for row in pairwise]
                    ),
                    "mean_conformal_downstream_token_accuracy_delta": _mean(
                        [_to_float(row["conformal_downstream_token_accuracy_delta"]) for row in pairwise]
                    ),
                    "mean_grouped_topk_delta": _mean(
                        [_to_float(row["uncertainty_sequence_topk_delta"]) for row in pairwise]
                    ),
                    "mean_grouped_exact_delta": _mean(
                        [_to_float(row["uncertainty_sequence_exact_delta"]) for row in pairwise]
                    ),
                    "mean_full_coverage_rate": _mean(
                        [
                            1.0 if _to_float(row["real_downstream_exact_match_if_covered"]) is not None else 0.0
                            for row in examples
                            if row["method"] == "uncertainty_beam"
                        ]
                    ),
                    "grouped_topk_without_downstream_exact_failures": int(
                        sum(
                            int(float(row["count"]))
                            for row in failure_rows
                            if row["case_type"] == "grouped_topk_rescue_without_downstream_exact"
                        )
                    ),
                    "symbol_without_downstream_failures": int(
                        sum(
                            int(float(row["count"]))
                            for row in failure_rows
                            if row["case_type"] == "symbol_rescue_without_downstream_recovery"
                        )
                    ),
                }
            )

    output_root = ensure_directory("outputs")
    csv_path = output_root / "real_downstream_redesigned_summary.csv"
    md_path = output_root / "real_downstream_redesigned_summary.md"
    plot_path = output_root / "real_downstream_redesigned_plot.png"
    write_csv(csv_path, summary_rows)
    _build_plot(summary_rows, plot_path)

    lines = [
        "# Real Downstream Redesigned Summary",
        "",
        "## Train-Supported N-Gram-Path Task",
        "",
    ]
    for row in summary_rows:
        lines.append(
            (
                f"- `{row['dataset']}` / `{row['posterior_strategy_requested']}`: coverage fraction `{_fmt(row['mean_downstream_coverage_fraction'])}`, "
                f"full-path coverage `{_fmt(row['mean_full_coverage_rate'])}`, "
                f"fixed exact `{_fmt(row['mean_fixed_downstream_exact_match'])}`, "
                f"uncertainty exact `{_fmt(row['mean_uncertainty_downstream_exact_match'])}`, "
                f"uncertainty downstream exact delta `{_fmt(row['mean_uncertainty_downstream_exact_delta'])}`, "
                f"uncertainty downstream token delta `{_fmt(row['mean_uncertainty_downstream_token_accuracy_delta'])}`, "
                f"conformal downstream exact delta `{_fmt(row['mean_conformal_downstream_exact_delta'])}`, "
                f"grouped top-k delta `{_fmt(row['mean_grouped_topk_delta'])}`, "
                f"`grouped_topk_rescue_without_downstream_exact` failures `{row['grouped_topk_without_downstream_exact_failures']}`."
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This redesigned task asks whether decoder outputs recover the gold train-supported n-gram path, not only the raw grouped transcript.",
            "- Positive downstream token or exact deltas indicate that grouped top-k rescue is propagating into a better-covered real structural target.",
        ]
    )
    write_text(md_path, "\n".join(lines) + "\n")
    print(json.dumps({"csv": str(csv_path), "markdown": str(md_path), "plot": str(plot_path)}, indent=2))


if __name__ == "__main__":
    main()
