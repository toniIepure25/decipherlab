from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/decipherlab-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from decipherlab.evaluation.metrics import average_probabilities_by_true_family
from decipherlab.utils.io import write_json, write_text


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _plot_family_probabilities(path: Path, summary: dict[str, Any], example_payloads: list[dict[str, Any]]) -> None:
    if not example_payloads:
        return
    probabilities = average_probabilities_by_true_family(
        [payload["ranking_object"] for payload in example_payloads],
        [payload["true_family"] for payload in example_payloads],
    )
    if not probabilities:
        return
    families = sorted({family for payload in probabilities.values() for family in payload})
    true_families = sorted(probabilities)
    data = np.asarray(
        [[probabilities[true_family].get(candidate, 0.0) for candidate in families] for true_family in true_families],
        dtype=float,
    )
    fig, axis = plt.subplots(figsize=(8, 4))
    image = axis.imshow(data, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(families)), families, rotation=45, ha="right")
    axis.set_yticks(range(len(true_families)), true_families)
    axis.set_title("Average Family Probability By True Family")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_posterior_entropy(path: Path, example_payloads: list[dict[str, Any]]) -> None:
    if not example_payloads:
        return
    labels = [payload["example_id"] for payload in example_payloads]
    entropies = [payload["triage"]["mean_posterior_entropy"] for payload in example_payloads]
    fig, axis = plt.subplots(figsize=(max(8, len(labels) * 0.25), 4))
    axis.bar(range(len(labels)), entropies, color="#315b7c")
    axis.set_title("Mean Posterior Entropy Per Example")
    axis.set_ylabel("Entropy")
    axis.set_xticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _build_markdown_report(summary: dict[str, Any], example_payloads: list[dict[str, Any]]) -> str:
    top_examples = sorted(
        example_payloads,
        key=lambda payload: payload["ranking"]["evidences"][0]["probability"],
        reverse=True,
    )[:5]
    lines = [
        "# DecipherLab Evidence Report",
        "",
        "## Aggregate Metrics",
        f"- Examples: {summary['example_count']}",
        f"- Ambiguity level: {_fmt_metric(summary.get('ambiguity_level'))}",
        f"- Posterior strategy: `{summary.get('posterior_strategy', 'unknown')}`",
        f"- Family top-1 accuracy: {_fmt_metric(summary.get('family_top1_accuracy'))}",
        f"- Family top-{summary['evaluation_top_k']} accuracy: {_fmt_metric(summary.get('family_topk_accuracy'))}",
        f"- Mean Brier score: {_fmt_metric(summary.get('mean_brier_score'))}",
        f"- Family ECE: {_fmt_metric(summary.get('expected_calibration_error'))}",
        f"- Symbol top-1 accuracy: {_fmt_metric(summary.get('symbol_top1_accuracy'))}",
        f"- Symbol top-{summary['evaluation_top_k']} accuracy: {_fmt_metric(summary.get('symbol_topk_accuracy'))}",
        f"- Symbol NLL: {_fmt_metric(summary.get('symbol_negative_log_likelihood'))}",
        f"- Symbol ECE: {_fmt_metric(summary.get('symbol_expected_calibration_error'))}",
        f"- Glyph clustering ARI: {_fmt_metric(summary.get('glyph_clustering_ari'))}",
        f"- Mean structural recovery error: {_fmt_metric(summary.get('mean_structural_recovery_error'))}",
        "",
        "## Caveats",
        "- Family scores are heuristic and evidence-oriented, not proofs of decipherment.",
        "- The posterior model is a calibrated baseline, not a full OCR/HTR recognizer.",
        "- Transposition support remains intentionally lightweight in this MVP.",
        "- Real-data experiments are only as strong as the supplied manifest labels and split quality.",
        "",
        "## Example Snapshots",
    ]
    for payload in top_examples:
        best = payload["ranking"]["evidences"][0]
        lines.extend(
            [
                f"### {payload['example_id']}",
                f"- True family: `{payload['true_family']}`" if payload["true_family"] is not None else "- True family: `n/a`",
                f"- Best ranked family: `{best['family']}` with probability {best['probability']:.3f}",
                f"- Posterior entropy: {payload['triage']['mean_posterior_entropy']:.3f}",
                f"- Symbol top-1 accuracy: {_fmt_metric(payload['symbol_metrics']['top1_accuracy'])}",
                f"- Null comparison: repeat gain {payload['triage']['diagnostics']['repeat_gain']:.3f}, compression gain {payload['triage']['diagnostics']['compression_gain']:.3f}",
            ]
        )
    return "\n".join(lines) + "\n"


def write_run_artifacts(
    run_dir: str | Path,
    summary: dict[str, Any],
    example_payloads: list[dict[str, Any]],
    cluster_payload: dict[str, Any],
    posterior_payload: dict[str, Any],
) -> None:
    output_dir = Path(run_dir)
    serializable_examples = []
    for payload in example_payloads:
        serializable = dict(payload)
        serializable.pop("ranking_object", None)
        serializable_examples.append(serializable)

    write_json(output_dir / "metrics.json", summary)
    write_json(output_dir / "cluster_result.json", cluster_payload)
    write_json(output_dir / "posterior_model.json", posterior_payload)
    write_json(output_dir / "example_results.json", serializable_examples)
    write_text(output_dir / "report.md", _build_markdown_report(summary, serializable_examples))
    _plot_family_probabilities(output_dir / "family_probabilities.png", summary, example_payloads)
    _plot_posterior_entropy(output_dir / "posterior_entropy.png", example_payloads)
