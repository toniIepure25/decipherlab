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

from decipherlab.ingest.historical_newspapers import materialize_historical_newspapers_validation_subset
from decipherlab.sequence.cross_dataset import build_sequence_cross_dataset_outputs
from decipherlab.sequence.runner import run_sequence_branch_experiment
from decipherlab.utils.io import ensure_directory, write_csv, write_text


ORIGINAL_CONFIG = Path("configs/experiments/sequence_historical_newspapers_real_grouped.yaml")
VALIDATED_CONFIG = Path("configs/experiments/sequence_historical_newspapers_real_grouped_validated.yaml")


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
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


def _run_pack(config_path: Path, dataset_label: str, output_root: str) -> dict[str, str]:
    strategy_runs: dict[str, str] = {}
    for strategy in ["cluster_distance", "calibrated_classifier"]:
        result = run_sequence_branch_experiment(config_path, strategy_override=strategy)
        strategy_runs[strategy] = str(result["run_dir"])
    outputs = build_sequence_cross_dataset_outputs(
        [
            {
                "dataset_label": dataset_label,
                "task_label": "real_grouped_manifest_sequences",
                "cluster_distance_run": strategy_runs["cluster_distance"],
                "calibrated_classifier_run": strategy_runs["calibrated_classifier"],
            }
        ],
        output_root=output_root,
    )
    return outputs


def _build_plot(rows: list[dict[str, object]], output_path: Path) -> None:
    strategies = [str(row["posterior_strategy_requested"]) for row in rows]
    metrics = [
        ("uncertainty_sequence_exact_delta", "Uncertainty Exact"),
        ("uncertainty_sequence_topk_delta", "Uncertainty Top-k"),
        ("uncertainty_symbol_topk_delta", "Symbol Top-k"),
        ("conformal_sequence_exact_delta", "Conformal Exact"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), squeeze=False)
    for axis, (metric_key, title) in zip(axes.flatten(), metrics):
        x = np.arange(len(strategies))
        width = 0.35
        original = [float(row[f"original_{metric_key}"]) for row in rows]
        validated = [float(row[f"validated_{metric_key}"]) for row in rows]
        axis.bar(x - width / 2, original, width, label="ocr-derived")
        axis.bar(x + width / 2, validated, width, label="validated")
        axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        axis.set_xticks(x)
        axis.set_xticklabels(strategies)
        axis.set_title(title)
    axes[0, 0].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    materialize_historical_newspapers_validation_subset(
        source_manifest_path="data/processed/historical_newspapers_grouped_words/manifest.yaml",
        corrections_csv_path="data/processed/historical_newspapers_grouped_words/validation_corrections.csv",
        output_manifest_path="data/processed/historical_newspapers_grouped_words/validation_subset_manifest.yaml",
        audit_csv_path="data/processed/historical_newspapers_grouped_words/validation_subset_annotations.csv",
        note_path="data/processed/historical_newspapers_grouped_words/validation_subset_README.md",
    )

    original_outputs = _run_pack(
        ORIGINAL_CONFIG,
        dataset_label="historical_newspapers_real_grouped",
        output_root="outputs/real_grouped_historical_newspapers",
    )
    validated_outputs = _run_pack(
        VALIDATED_CONFIG,
        dataset_label="historical_newspapers_real_grouped_validated",
        output_root="outputs/real_grouped_historical_newspapers_validated",
    )

    original_rows = {
        row["posterior_strategy_requested"]: row
        for row in _read_csv_rows(original_outputs["tables_csv"])
    }
    validated_rows = {
        row["posterior_strategy_requested"]: row
        for row in _read_csv_rows(validated_outputs["tables_csv"])
    }
    noise_summary = json.loads(
        Path("data/processed/historical_newspapers_grouped_words/validation_label_noise_summary.json").read_text(
            encoding="utf-8"
        )
    )

    robustness_rows: list[dict[str, object]] = []
    for strategy in ["cluster_distance", "calibrated_classifier"]:
        original_row = original_rows[strategy]
        validated_row = validated_rows[strategy]
        robustness_rows.append(
            {
                "posterior_strategy_requested": strategy,
                "audited_token_count": noise_summary["audited_token_count"],
                "changed_token_count": noise_summary["changed_token_count"],
                "token_error_rate": noise_summary["token_error_rate"],
                "original_uncertainty_sequence_exact_delta": _to_float(
                    original_row["mean_uncertainty_sequence_exact_delta"]
                ),
                "validated_uncertainty_sequence_exact_delta": _to_float(
                    validated_row["mean_uncertainty_sequence_exact_delta"]
                ),
                "validated_minus_original_uncertainty_sequence_exact_delta": (
                    _to_float(validated_row["mean_uncertainty_sequence_exact_delta"]) or 0.0
                )
                - (_to_float(original_row["mean_uncertainty_sequence_exact_delta"]) or 0.0),
                "original_uncertainty_sequence_topk_delta": _to_float(
                    original_row["mean_uncertainty_sequence_topk_delta"]
                ),
                "validated_uncertainty_sequence_topk_delta": _to_float(
                    validated_row["mean_uncertainty_sequence_topk_delta"]
                ),
                "validated_minus_original_uncertainty_sequence_topk_delta": (
                    _to_float(validated_row["mean_uncertainty_sequence_topk_delta"]) or 0.0
                )
                - (_to_float(original_row["mean_uncertainty_sequence_topk_delta"]) or 0.0),
                "original_uncertainty_symbol_topk_delta": _to_float(
                    original_row["mean_uncertainty_symbol_topk_delta"]
                ),
                "validated_uncertainty_symbol_topk_delta": _to_float(
                    validated_row["mean_uncertainty_symbol_topk_delta"]
                ),
                "validated_minus_original_uncertainty_symbol_topk_delta": (
                    _to_float(validated_row["mean_uncertainty_symbol_topk_delta"]) or 0.0
                )
                - (_to_float(original_row["mean_uncertainty_symbol_topk_delta"]) or 0.0),
                "original_conformal_sequence_exact_delta": _to_float(
                    original_row["mean_conformal_sequence_exact_delta"]
                ),
                "validated_conformal_sequence_exact_delta": _to_float(
                    validated_row["mean_conformal_sequence_exact_delta"]
                ),
                "validated_minus_original_conformal_sequence_exact_delta": (
                    _to_float(validated_row["mean_conformal_sequence_exact_delta"]) or 0.0
                )
                - (_to_float(original_row["mean_conformal_sequence_exact_delta"]) or 0.0),
            }
        )

    output_root = ensure_directory("outputs")
    summary_csv = output_root / "real_grouped_robustness_summary.csv"
    summary_md = output_root / "real_grouped_robustness_summary.md"
    plot_path = output_root / "real_grouped_robustness_plot.png"
    write_csv(summary_csv, robustness_rows)
    _build_plot(robustness_rows, plot_path)
    write_text(
        summary_md,
        "\n".join(
            [
                "# Real Grouped Robustness Summary",
                "",
                f"- Audited tokens: `{noise_summary['audited_token_count']}`",
                f"- Corrected tokens: `{noise_summary['changed_token_count']}`",
                f"- Token error rate: `{noise_summary['token_error_rate']:.3f}`",
                "",
                "## OCR-Derived vs Validated",
                "",
            ]
            + [
                (
                    f"- `{row['posterior_strategy_requested']}`: uncertainty exact "
                    f"`{_fmt(row['original_uncertainty_sequence_exact_delta'])}` -> `{_fmt(row['validated_uncertainty_sequence_exact_delta'])}`, "
                    f"uncertainty top-k `{_fmt(row['original_uncertainty_sequence_topk_delta'])}` -> `{_fmt(row['validated_uncertainty_sequence_topk_delta'])}`, "
                    f"symbol top-k `{_fmt(row['original_uncertainty_symbol_topk_delta'])}` -> `{_fmt(row['validated_uncertainty_symbol_topk_delta'])}`, "
                    f"conformal exact `{_fmt(row['original_conformal_sequence_exact_delta'])}` -> `{_fmt(row['validated_conformal_sequence_exact_delta'])}`."
                )
                for row in robustness_rows
            ]
            + [
                "",
                "## Interpretation",
                "",
                "- This comparison isolates label corrections on the real grouped benchmark while keeping the structured-uncertainty pipeline unchanged.",
                "- It should be read as a robustness check, not as a new corpus-level generalization claim.",
            ]
        )
        + "\n",
    )
    print(
        json.dumps(
            {
                "original_outputs": original_outputs,
                "validated_outputs": validated_outputs,
                "robustness_summary": {
                    "csv": str(summary_csv),
                    "markdown": str(summary_md),
                    "plot": str(plot_path),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
