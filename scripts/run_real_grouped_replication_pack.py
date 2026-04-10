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

from decipherlab.sequence.cross_dataset import build_sequence_cross_dataset_outputs
from decipherlab.sequence.runner import run_sequence_branch_experiment
from decipherlab.utils.io import ensure_directory, write_csv, write_text


HISTORICAL_GOLD_CONFIG = Path("configs/experiments/sequence_historical_newspapers_real_grouped_gold.yaml")
SCADS_CONFIG = Path("configs/experiments/sequence_scadsai_real_grouped.yaml")


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
    labels = [f"{row['dataset']}:{row['posterior_strategy_requested']}" for row in rows]
    metrics = [
        ("mean_uncertainty_sequence_exact_delta", "Uncertainty Exact"),
        ("mean_uncertainty_sequence_topk_delta", "Uncertainty Top-k"),
        ("mean_uncertainty_symbol_topk_delta", "Symbol Top-k"),
        ("mean_conformal_sequence_exact_delta", "Conformal Exact"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), squeeze=False)
    values_x = np.arange(len(labels))
    for axis, (metric_key, title) in zip(axes.flatten(), metrics):
        values = [float(row[metric_key]) for row in rows]
        axis.bar(values_x, values, color=["#4c78a8" if "historical" in str(row["dataset"]) else "#e45756" for row in rows])
        axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        axis.set_xticks(values_x)
        axis.set_xticklabels(labels, rotation=20, ha="right")
        axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    historical_outputs = _run_pack(
        HISTORICAL_GOLD_CONFIG,
        dataset_label="historical_newspapers_real_grouped_gold",
        output_root="outputs/real_grouped_historical_newspapers_gold",
    )
    scads_outputs = _run_pack(
        SCADS_CONFIG,
        dataset_label="scadsai_real_grouped",
        output_root="outputs/real_grouped_scadsai",
    )

    rows: list[dict[str, object]] = []
    for dataset_label, outputs in [
        ("historical_newspapers_real_grouped_gold", historical_outputs),
        ("scadsai_real_grouped", scads_outputs),
    ]:
        for row in _read_csv_rows(outputs["tables_csv"]):
            rows.append(
                {
                    "dataset": dataset_label,
                    "posterior_strategy_requested": row["posterior_strategy_requested"],
                    "mean_uncertainty_sequence_exact_delta": _to_float(row["mean_uncertainty_sequence_exact_delta"]),
                    "mean_uncertainty_sequence_topk_delta": _to_float(row["mean_uncertainty_sequence_topk_delta"]),
                    "mean_uncertainty_symbol_topk_delta": _to_float(row["mean_uncertainty_symbol_topk_delta"]),
                    "mean_conformal_sequence_exact_delta": _to_float(row["mean_conformal_sequence_exact_delta"]),
                    "peak_ambiguity_for_uncertainty_exact_delta": _to_float(row["peak_ambiguity_for_uncertainty_exact_delta"]),
                }
            )

    output_root = ensure_directory("outputs")
    summary_csv = output_root / "real_grouped_replication_summary.csv"
    summary_md = output_root / "real_grouped_replication_summary.md"
    plot_path = output_root / "real_grouped_replication_plot.png"
    write_csv(summary_csv, rows)
    _build_plot(rows, plot_path)
    write_text(
        summary_md,
        "\n".join(
            [
                "# Real Grouped Replication Summary",
                "",
                "## Historical Newspapers Gold vs ScaDS.AI",
                "",
            ]
            + [
                (
                    f"- `{row['dataset']}` / `{row['posterior_strategy_requested']}`: uncertainty exact "
                    f"`{_fmt(row['mean_uncertainty_sequence_exact_delta'])}`, uncertainty top-k "
                    f"`{_fmt(row['mean_uncertainty_sequence_topk_delta'])}`, symbol top-k "
                    f"`{_fmt(row['mean_uncertainty_symbol_topk_delta'])}`, conformal exact "
                    f"`{_fmt(row['mean_conformal_sequence_exact_delta'])}`, peak ambiguity "
                    f"`{_fmt(row['peak_ambiguity_for_uncertainty_exact_delta'])}`."
                )
                for row in rows
            ]
            + [
                "",
                "## Interpretation",
                "",
                "- This pack compares the strongest current Historical Newspapers grouped run against the second real grouped corpus using the unchanged grouped decoder family.",
                "- Grouped top-k rescue is the clearest replicated signal across both corpora.",
                "- Conformal exact-match gains are mixed rather than cleanly replicated across both corpora.",
            ]
        )
        + "\n",
    )
    print(
        json.dumps(
            {
                "historical_outputs": historical_outputs,
                "scads_outputs": scads_outputs,
                "replication_summary": {
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
