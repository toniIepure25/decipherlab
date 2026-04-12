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
    replication_rows = _read_csv(OUTPUTS / "real_grouped_replication_summary.csv")
    regime_rows = _read_csv(OUTPUTS / "propagation_regime_summary.csv")

    dataset_labels = {
        ("historical_newspapers_real_grouped_gold", "calibrated_classifier"): "Historical N. / calibrated",
        ("historical_newspapers_real_grouped_gold", "cluster_distance"): "Historical N. / distance",
        ("scadsai_real_grouped", "calibrated_classifier"): "ScaDS.AI / calibrated",
        ("scadsai_real_grouped", "cluster_distance"): "ScaDS.AI / distance",
    }
    dataset_colors = {
        "historical_newspapers_real_grouped_gold": "#4c78a8",
        "scadsai_real_grouped": "#e45756",
    }
    ordered_replication = sorted(
        replication_rows,
        key=lambda row: (
            0 if row["dataset"] == "historical_newspapers_real_grouped_gold" else 1,
            0 if row["posterior_strategy_requested"] == "calibrated_classifier" else 1,
        ),
    )

    regime_order = [
        ("limited_support", "low_entropy"),
        ("high_support", "low_entropy"),
        ("high_support", "high_entropy"),
    ]
    regime_labels = {
        ("limited_support", "low_entropy"): "Limited support,\nlow entropy",
        ("high_support", "low_entropy"): "High support,\nlow entropy",
        ("high_support", "high_entropy"): "High support,\nhigh entropy",
    }
    family_map = {
        "raw_uncertainty": ("Raw uncertainty", "#4c78a8"),
        "conformal": ("Conformal", "#dd8452"),
    }

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(7.1, 4.1), gridspec_kw={"height_ratios": [1.0, 1.08]})

    labels = [dataset_labels[(row["dataset"], row["posterior_strategy_requested"])] for row in ordered_replication]
    values = [_to_float(row["mean_uncertainty_sequence_topk_delta"]) for row in ordered_replication]
    colors = [dataset_colors[row["dataset"]] for row in ordered_replication]
    y = np.arange(len(labels))
    ax_top.barh(y, values, color=colors, height=0.58)
    ax_top.set_yticks(y, labels)
    ax_top.invert_yaxis()
    ax_top.set_xlim(0.0, 0.42)
    ax_top.set_xlabel("Grouped top-k delta")
    ax_top.set_title("A. Replicated grouped rescue", loc="left", fontweight="bold", fontsize=9.6)
    for idx, value in enumerate(values):
        ax_top.text(value + 0.011, idx, f"{value:.2f}", va="center", fontsize=7.0)

    regime_index = np.arange(len(regime_order))
    bar_height = 0.28
    for offset, family_key in [(-bar_height / 2, "raw_uncertainty"), (bar_height / 2, "conformal")]:
        family_rows = {(row["support_regime"], row["entropy_regime"]): row for row in regime_rows if row["method_family"] == family_key}
        plotted_keys = [key for key in regime_order if key in family_rows]
        plotted_positions = [regime_order.index(key) + offset for key in plotted_keys]
        plotted_values = [_to_float(family_rows[key]["downstream_exact_rescue_rate"]) for key in plotted_keys]
        ax_bottom.barh(
            plotted_positions,
            plotted_values,
            height=bar_height,
            color=family_map[family_key][1],
            label=family_map[family_key][0],
        )
        for pos, regime_key, value in zip(plotted_positions, plotted_keys, plotted_values):
            grouped_rate = _to_float(family_rows[regime_key]["grouped_topk_rescue_rate"])
            ax_bottom.text(min(value + 0.012, 0.47), pos, f"{value:.2f}", va="center", fontsize=6.8)
            ax_bottom.text(0.01, pos + 0.23, f"grouped {grouped_rate:.2f}", va="center", fontsize=6.0, color="#666666")

    ax_bottom.set_yticks(regime_index, [regime_labels[key] for key in regime_order])
    ax_bottom.invert_yaxis()
    ax_bottom.set_xlim(0.0, 0.5)
    ax_bottom.set_xlabel("Downstream exact rescue")
    ax_bottom.set_title("B. Support-gated downstream rescue", loc="left", fontweight="bold", fontsize=9.6, pad=18)
    ax_bottom.legend(
        loc="upper right",
        bbox_to_anchor=(0.99, 1.17),
        frameon=False,
        ncol=2,
        columnspacing=1.3,
        handlelength=1.6,
        borderaxespad=0.0,
    )
    ax_bottom.tick_params(axis="y", labelsize=7.6, pad=4)
    fig.subplots_adjust(left=0.24, right=0.98, top=0.93, bottom=0.12, hspace=0.50)
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

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55), gridspec_kw={"width_ratios": [1.1, 1.0, 1.0]})
    axes[0].barh(y, coverage, color=colors, height=0.6)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.0, 1.08)
    axes[0].set_xlabel("Coverage")
    axes[0].set_title("A. Coverage", fontsize=9.4)
    for idx, value in enumerate(coverage):
        axes[0].text(value + 0.02, idx, f"{value:.2f}", va="center", fontsize=7.1)

    axes[1].barh(y, raw_exact, color="#9aa6b2", height=0.6)
    axes[1].set_yticks(y, ["", "", "", ""])
    axes[1].invert_yaxis()
    axes[1].axvline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlim(-0.18, 0.28)
    axes[1].set_xlabel("Exact delta")
    axes[1].set_title("B. Raw uncertainty", fontsize=9.4)
    for idx, value in enumerate(raw_exact):
        offset = 0.012 if value >= 0 else -0.012
        axes[1].text(value + offset, idx, f"{value:.2f}", va="center", ha="left" if value >= 0 else "right", fontsize=6.9)

    axes[2].barh(y, conformal_exact, color="#dd8452", height=0.6)
    axes[2].set_yticks(y, ["", "", "", ""])
    axes[2].invert_yaxis()
    axes[2].axvline(0.0, color="black", linewidth=0.8)
    axes[2].set_xlim(-0.18, 0.28)
    axes[2].set_xlabel("Exact delta")
    axes[2].set_title("C. Conformal", fontsize=9.4)
    for idx, value in enumerate(conformal_exact):
        offset = 0.012 if value >= 0 else -0.012
        axes[2].text(value + offset, idx, f"{value:.2f}", va="center", ha="left" if value >= 0 else "right", fontsize=6.9)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.84, bottom=0.18, wspace=0.33)
    _save(fig, "figA1_real_downstream_redesigned_plot")


def main() -> None:
    _configure_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    build_propagation_regime_figure()
    build_real_grouped_replication_figure()
    build_real_downstream_figure()


if __name__ == "__main__":
    main()
