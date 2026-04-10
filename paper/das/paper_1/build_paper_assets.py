from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/decipherlab-das-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
PAPER_ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
FIGURES = PAPER_ROOT / "figures"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _to_float(value: str) -> float:
    return float(value)


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    pdf_path = FIGURES / f"{stem}.pdf"
    png_path = FIGURES / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_propagation_regime_figure() -> None:
    rows = _read_csv(OUTPUTS / "propagation_regime_summary.csv")
    regime_order = [
        ("limited_support", "low_entropy"),
        ("high_support", "low_entropy"),
        ("high_support", "high_entropy"),
    ]
    label_map = {
        ("limited_support", "low_entropy"): "Limited support\nLow entropy",
        ("high_support", "low_entropy"): "High support\nLow entropy",
        ("high_support", "high_entropy"): "High support\nHigh entropy",
    }
    family_map = {
        "conformal": ("Conformal", "#dd8452"),
        "raw_uncertainty": ("Raw uncertainty", "#4c78a8"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), sharex=True, constrained_layout=True)
    for ax, family_key in zip(axes, ["conformal", "raw_uncertainty"]):
        family_rows = { (row["support_regime"], row["entropy_regime"]): row for row in rows if row["method_family"] == family_key }
        plot_labels: list[str] = []
        plot_values: list[float] = []
        grouped_rates: list[float] = []
        counts: list[int] = []
        for key in regime_order:
            row = family_rows.get(key)
            if row is None:
                continue
            plot_labels.append(label_map[key])
            plot_values.append(_to_float(row["downstream_exact_rescue_rate"]))
            grouped_rates.append(_to_float(row["grouped_topk_rescue_rate"]))
            counts.append(int(row["count"]))
        y = np.arange(len(plot_labels))
        ax.barh(y, plot_values, color=family_map[family_key][1], height=0.6)
        ax.set_yticks(y, plot_labels)
        ax.invert_yaxis()
        ax.set_xlim(0.0, 0.5)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_title(family_map[family_key][0])
        for idx, (value, grouped_rate, count) in enumerate(zip(plot_values, grouped_rates, counts)):
            ax.text(min(value + 0.012, 0.48), idx, f"{value:.2f}  (grp {grouped_rate:.2f}, n={count})", va="center", fontsize=7.2)
    axes[0].set_ylabel("Support regime")
    axes[0].set_xlabel("Downstream exact rescue rate")
    axes[1].set_xlabel("Downstream exact rescue rate")
    fig.suptitle("Support regimes for real downstream rescue")
    _save(fig, "fig1_propagation_regime_plot")


def build_real_grouped_replication_figure() -> None:
    rows = _read_csv(OUTPUTS / "real_grouped_replication_summary.csv")
    label_map = {
        ("historical_newspapers_real_grouped_gold", "calibrated_classifier"): "Hist. / Cal.",
        ("historical_newspapers_real_grouped_gold", "cluster_distance"): "Hist. / Dist.",
        ("scadsai_real_grouped", "calibrated_classifier"): "ScaDS / Cal.",
        ("scadsai_real_grouped", "cluster_distance"): "ScaDS / Dist.",
    }
    color_map = {
        "historical_newspapers_real_grouped_gold": "#4c78a8",
        "scadsai_real_grouped": "#e45756",
    }
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            0 if row["dataset"] == "historical_newspapers_real_grouped_gold" else 1,
            0 if row["posterior_strategy_requested"] == "calibrated_classifier" else 1,
        ),
    )

    labels = [label_map[(row["dataset"], row["posterior_strategy_requested"])] for row in ordered_rows]
    topk_values = [_to_float(row["mean_uncertainty_sequence_topk_delta"]) for row in ordered_rows]
    raw_exact = [_to_float(row["mean_uncertainty_sequence_exact_delta"]) for row in ordered_rows]
    conformal_exact = [_to_float(row["mean_conformal_sequence_exact_delta"]) for row in ordered_rows]
    colors = [color_map[row["dataset"]] for row in ordered_rows]
    y = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.4), constrained_layout=True)

    axes[0].barh(y, topk_values, color=colors, height=0.6)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.0, 0.42)
    axes[0].set_xlabel("Grouped top-k delta")
    axes[0].set_title("Replicated grouped top-k rescue")
    for idx, value in enumerate(topk_values):
        axes[0].text(value + 0.01, idx, f"{value:.2f}", va="center", fontsize=7.5)

    bar_height = 0.32
    axes[1].barh(y - bar_height / 2, raw_exact, color="#9aa6b2", height=bar_height, label="Raw exact delta")
    axes[1].barh(y + bar_height / 2, conformal_exact, color="#dd8452", height=bar_height, label="Conformal exact delta")
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].axvline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlim(-0.18, 0.28)
    axes[1].set_xlabel("Exact-recovery delta")
    axes[1].set_title("Exact recovery remains mixed")
    axes[1].legend(loc="lower right", frameon=False)
    for idx, value in enumerate(raw_exact):
        offset = 0.012 if value >= 0 else -0.012
        axes[1].text(value + offset, idx - bar_height / 2, f"{value:.2f}", va="center", ha="left" if value >= 0 else "right", fontsize=7.2)
    for idx, value in enumerate(conformal_exact):
        offset = 0.012 if value >= 0 else -0.012
        axes[1].text(value + offset, idx + bar_height / 2, f"{value:.2f}", va="center", ha="left" if value >= 0 else "right", fontsize=7.2)

    _save(fig, "fig2_real_grouped_replication_plot")


def build_real_downstream_figure() -> None:
    rows = _read_csv(OUTPUTS / "real_downstream_redesigned_summary.csv")
    label_map = {
        ("historical_newspapers_real_grouped_gold", "calibrated_classifier"): "Hist. / Cal.",
        ("historical_newspapers_real_grouped_gold", "cluster_distance"): "Hist. / Dist.",
        ("scadsai_real_grouped", "calibrated_classifier"): "ScaDS / Cal.",
        ("scadsai_real_grouped", "cluster_distance"): "ScaDS / Dist.",
    }
    color_map = {
        "historical_newspapers_real_grouped_gold": "#4c78a8",
        "scadsai_real_grouped": "#e45756",
    }
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            0 if row["dataset"] == "historical_newspapers_real_grouped_gold" else 1,
            0 if row["posterior_strategy_requested"] == "calibrated_classifier" else 1,
        ),
    )
    labels = [label_map[(row["dataset"], row["posterior_strategy_requested"])] for row in ordered_rows]
    coverage = [_to_float(row["mean_downstream_coverage_fraction"]) for row in ordered_rows]
    raw_exact = [_to_float(row["mean_uncertainty_downstream_exact_delta"]) for row in ordered_rows]
    conformal_exact = [_to_float(row["mean_conformal_downstream_exact_delta"]) for row in ordered_rows]
    colors = [color_map[row["dataset"]] for row in ordered_rows]
    y = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.4), constrained_layout=True)
    axes[0].barh(y, coverage, color=colors, height=0.6)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.0, 1.08)
    axes[0].set_xlabel("Coverage fraction")
    axes[0].set_title("Redesigned task coverage")
    for idx, value in enumerate(coverage):
        axes[0].text(value + 0.02, idx, f"{value:.2f}", va="center", fontsize=7.5)

    bar_height = 0.32
    axes[1].barh(y - bar_height / 2, raw_exact, color="#9aa6b2", height=bar_height, label="Raw exact delta")
    axes[1].barh(y + bar_height / 2, conformal_exact, color="#dd8452", height=bar_height, label="Conformal exact delta")
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].axvline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlim(-0.18, 0.28)
    axes[1].set_xlabel("Downstream exact delta")
    axes[1].set_title("Selective downstream gains")
    axes[1].legend(loc="lower right", frameon=False)
    _save(fig, "figA1_real_downstream_redesigned_plot")


def main() -> None:
    _configure_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    build_propagation_regime_figure()
    build_real_grouped_replication_figure()
    build_real_downstream_figure()


if __name__ == "__main__":
    main()
