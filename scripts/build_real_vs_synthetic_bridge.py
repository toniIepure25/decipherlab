from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from decipherlab.utils.io import write_csv, write_text


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


def _mean(values: list[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def main() -> None:
    symbol_rows = _read_csv("outputs/cross_dataset_summary.csv")
    synthetic_markov_rows = _read_csv("outputs/sequence_cross_dataset_tables.csv")
    synthetic_process_rows = _read_csv("outputs/sequence_process_family_cross_dataset_tables.csv")
    real_grouped_rows = _read_csv("outputs/real_grouped_historical_newspapers_gold/sequence_cross_dataset_tables.csv")
    real_grouped_replication_rows = _read_csv("outputs/real_grouped_replication_summary.csv")
    real_downstream_rows = _read_csv("outputs/real_downstream_redesigned_summary.csv")
    real_downstream_coverage_rows = _read_csv("outputs/real_downstream_coverage_analysis.csv")
    strengthened_rows = _read_csv("outputs/real_grouped_strengthened_summary.csv")
    gold_annotations = _read_csv("data/processed/historical_newspapers_grouped_words/gold_annotations.csv")
    grouped_by_dataset: dict[str, list[dict[str, str]]] = {}
    for row in real_grouped_replication_rows:
        grouped_by_dataset.setdefault(row["dataset"], []).append(row)
    downstream_by_dataset: dict[str, list[dict[str, str]]] = {}
    for row in real_downstream_rows:
        downstream_by_dataset.setdefault(row["dataset"], []).append(row)

    historical_rows = grouped_by_dataset["historical_newspapers_real_grouped_gold"]
    scads_rows = grouped_by_dataset["scadsai_real_grouped"]
    historical_downstream_rows = downstream_by_dataset["historical_newspapers_real_grouped_gold"]
    scads_downstream_rows = downstream_by_dataset["scadsai_real_grouped"]
    coverage_by_dataset_task_metric = {
        (row["dataset"], row["task_name"], row["coverage_metric"]): row for row in real_downstream_coverage_rows
    }

    bridge_rows = [
        {
            "evidence_scope": "real_symbol_level",
            "task": "frozen_workshop_symbol_level",
            "support_level": "supported",
            "key_metric": "mean_calibrated_uncertainty_topk_delta",
            "value": _mean(
                [_to_float(row["calibrated_uncertainty_topk_delta_mean"]) for row in symbol_rows]
            ),
            "note": "Fully real symbol-level evidence across Omniglot, Digits, and Kuzushiji-49.",
        },
        {
            "evidence_scope": "synthetic_from_real_sequence",
            "task": "real_glyph_markov_sequences",
            "support_level": "supported",
            "key_metric": "mean_calibrated_sequence_exact_delta",
            "value": _mean(
                [
                    _to_float(row["mean_uncertainty_sequence_exact_delta"])
                    for row in synthetic_markov_rows
                    if row["posterior_strategy_requested"] == "calibrated_classifier"
                ]
            ),
            "note": "Synthetic-from-real sequence exact-match gain with calibrated posteriors.",
        },
        {
            "evidence_scope": "synthetic_from_real_sequence",
            "task": "real_glyph_markov_sequences",
            "support_level": "supported",
            "key_metric": "mean_calibrated_sequence_topk_delta",
            "value": _mean(
                [
                    _to_float(row["mean_uncertainty_sequence_topk_delta"])
                    for row in synthetic_markov_rows
                    if row["posterior_strategy_requested"] == "calibrated_classifier"
                ]
            ),
            "note": "Synthetic-from-real grouped top-k recovery is more stable than exact match.",
        },
        {
            "evidence_scope": "synthetic_from_real_downstream",
            "task": "real_glyph_process_family_sequences",
            "support_level": "supported_but_bounded",
            "key_metric": "mean_uncertainty_family_delta",
            "value": _mean(
                [_to_float(row["mean_uncertainty_family_delta"]) for row in synthetic_process_rows]
            ),
            "note": "Synthetic-from-real downstream family gains are positive on average but selective.",
        },
        {
            "evidence_scope": "real_grouped_sequence",
            "task": "real_grouped_manifest_sequences",
            "support_level": "strengthened_real_grouped",
            "key_metric": "mean_uncertainty_sequence_exact_delta",
            "value": _mean(
                [_to_float(row["mean_uncertainty_sequence_exact_delta"]) for row in real_grouped_rows]
            ),
            "note": "Raw uncertainty beam does not improve grouped exact match on average on the strengthened Historical Newspapers grouped benchmark.",
        },
        {
            "evidence_scope": "real_grouped_sequence",
            "task": "real_grouped_manifest_sequences",
            "support_level": "strengthened_real_grouped",
            "key_metric": "mean_uncertainty_sequence_topk_delta",
            "value": _mean(
                [_to_float(row["mean_uncertainty_sequence_topk_delta"]) for row in real_grouped_rows]
            ),
            "note": "Raw uncertainty does improve grouped top-k recovery on the strengthened Historical Newspapers grouped benchmark.",
        },
        {
            "evidence_scope": "real_grouped_sequence",
            "task": "real_grouped_manifest_sequences",
            "support_level": "strengthened_real_grouped",
            "key_metric": "mean_uncertainty_symbol_topk_delta",
            "value": _mean(
                [_to_float(row["mean_uncertainty_symbol_topk_delta"]) for row in real_grouped_rows]
            ),
            "note": "Symbol-level rescue still transfers into the strengthened Historical Newspapers grouped benchmark.",
        },
        {
            "evidence_scope": "real_grouped_sequence",
            "task": "real_grouped_manifest_sequences",
            "support_level": "strengthened_real_grouped",
            "key_metric": "mean_conformal_sequence_exact_delta",
            "value": _mean(
                [_to_float(row["mean_conformal_sequence_exact_delta"]) for row in real_grouped_rows]
            ),
            "note": "Conformal pruning gives the clearest grouped exact-match gain on the strengthened Historical Newspapers benchmark.",
        },
        {
            "evidence_scope": "real_grouped_replication",
            "task": "historical_newspapers_real_grouped_gold",
            "support_level": "replicated_real_grouped_topk",
            "key_metric": "mean_uncertainty_sequence_topk_delta",
            "value": _mean(
                [_to_float(row["mean_uncertainty_sequence_topk_delta"]) for row in historical_rows]
            ),
            "note": "Historical Newspapers retains positive grouped top-k rescue after the gold-style upgrade.",
        },
        {
            "evidence_scope": "real_grouped_replication",
            "task": "scadsai_real_grouped",
            "support_level": "replicated_real_grouped_topk",
            "key_metric": "mean_uncertainty_sequence_topk_delta",
            "value": _mean(
                [_to_float(row["mean_uncertainty_sequence_topk_delta"]) for row in scads_rows]
            ),
            "note": "ScaDS.AI also shows positive grouped top-k rescue under the unchanged grouped decoder pack.",
        },
        {
            "evidence_scope": "real_grouped_replication",
            "task": "two_real_grouped_corpora",
            "support_level": "replicated_real_grouped_topk",
            "key_metric": "mean_uncertainty_sequence_topk_delta",
            "value": _mean(
                [_to_float(row["mean_uncertainty_sequence_topk_delta"]) for row in real_grouped_replication_rows]
            ),
            "note": "Grouped top-k rescue is positive across both real grouped corpora.",
        },
        {
            "evidence_scope": "real_grouped_replication",
            "task": "two_real_grouped_corpora",
            "support_level": "mixed_real_grouped_exact",
            "key_metric": "mean_conformal_sequence_exact_delta",
            "value": _mean(
                [_to_float(row["mean_conformal_sequence_exact_delta"]) for row in real_grouped_replication_rows]
            ),
            "note": "Conformal exact-match gains are mixed across the two real grouped corpora rather than cleanly replicated.",
        },
        {
            "evidence_scope": "real_grouped_downstream",
            "task": "historical_newspapers_train_supported_ngram_path",
            "support_level": "better_covered_real_downstream",
            "key_metric": "mean_downstream_coverage_fraction",
            "value": _mean(
                [_to_float(row["mean_downstream_coverage_fraction"]) for row in historical_downstream_rows]
            ),
            "note": "Historical Newspapers now has substantial downstream coverage under the train-supported n-gram-path task.",
        },
        {
            "evidence_scope": "real_grouped_downstream",
            "task": "historical_newspapers_train_supported_ngram_path",
            "support_level": "better_covered_real_downstream",
            "key_metric": "mean_uncertainty_downstream_exact_delta",
            "value": _mean(
                [_to_float(row["mean_uncertainty_downstream_exact_delta"]) for row in historical_downstream_rows]
            ),
            "note": "On Historical Newspapers, raw uncertainty still does not improve exact n-gram-path recovery on average despite the better-covered task.",
        },
        {
            "evidence_scope": "real_grouped_downstream",
            "task": "scadsai_train_supported_ngram_path",
            "support_level": "better_covered_real_downstream",
            "key_metric": "mean_downstream_coverage_fraction",
            "value": _mean(
                [_to_float(row["mean_downstream_coverage_fraction"]) for row in scads_downstream_rows]
            ),
            "note": "ScaDS.AI has effectively full downstream coverage under the train-supported n-gram-path task.",
        },
        {
            "evidence_scope": "real_grouped_downstream",
            "task": "scadsai_train_supported_ngram_path",
            "support_level": "selective_real_downstream",
            "key_metric": "mean_uncertainty_downstream_exact_delta",
            "value": _mean(
                [_to_float(row["mean_uncertainty_downstream_exact_delta"]) for row in scads_downstream_rows]
            ),
            "note": "ScaDS.AI shows a selective positive raw uncertainty exact downstream gain under the noisier cluster-distance setting.",
        },
        {
            "evidence_scope": "real_grouped_downstream",
            "task": "two_real_grouped_corpora_train_supported_ngram_path",
            "support_level": "bounded_real_downstream",
            "key_metric": "mean_uncertainty_downstream_exact_delta",
            "value": _mean(
                [_to_float(row["mean_uncertainty_downstream_exact_delta"]) for row in real_downstream_rows]
            ),
            "note": "Across both real grouped corpora, raw uncertainty does not improve exact n-gram-path recovery on average despite the redesigned higher-coverage task.",
        },
        {
            "evidence_scope": "real_grouped_downstream",
            "task": "two_real_grouped_corpora_train_supported_ngram_path",
            "support_level": "bounded_real_downstream",
            "key_metric": "mean_conformal_downstream_exact_delta",
            "value": _mean(
                [_to_float(row["mean_conformal_downstream_exact_delta"]) for row in real_downstream_rows]
            ),
            "note": "Across both real grouped corpora, conformal remains the clearest exact downstream rescue mechanism on average, but the gain is still selective rather than replicated.",
        },
        {
            "evidence_scope": "real_grouped_downstream",
            "task": "two_real_grouped_corpora_train_supported_ngram_path",
            "support_level": "coverage_improvement",
            "key_metric": "mean_full_path_coverage_upper_bound",
            "value": _mean(
                [
                    _to_float(
                        coverage_by_dataset_task_metric[(dataset, "train_supported_ngram_path", "full_path_coverage")]["value"]
                    )
                    for dataset in ("historical_newspapers_real_grouped_gold", "scadsai_real_grouped")
                ]
            ),
            "note": "The redesigned task substantially improves the real downstream upper bound relative to exact transcript-bank overlap.",
        },
        {
            "evidence_scope": "real_grouped_validation",
            "task": "historical_newspapers_gold_style_subset",
            "support_level": "gold_style_check",
            "key_metric": "gold_pass_agreement_rate",
            "value": _mean(
                [1.0 if row["annotator_agreement"].strip().lower() == "true" else 0.0 for row in gold_annotations]
            ),
            "note": "Gold-style two-pass in-session review produced full pass agreement on the adjudicated test split.",
        },
        {
            "evidence_scope": "real_grouped_validation",
            "task": "historical_newspapers_gold_style_subset",
            "support_level": "gold_style_check",
            "key_metric": "gold_ocr_to_label_error_rate",
            "value": _mean(
                [
                    1.0 if row["ocr_label"] != row["adjudicated_label"] else 0.0
                    for row in gold_annotations
                ]
            ),
            "note": "The gold-style subset retains a low OCR-to-label disagreement rate.",
        },
        {
            "evidence_scope": "real_grouped_validation",
            "task": "historical_newspapers_gold_style_subset",
            "support_level": "gold_style_check",
            "key_metric": "gold_minus_original_conformal_sequence_exact_delta",
            "value": _mean(
                [
                    _to_float(row["gold_minus_original_conformal_sequence_exact_delta"])
                    for row in strengthened_rows
                ]
            ),
            "note": "The real grouped metrics were unchanged after upgrading from OCR-derived to gold-style adjudicated labels.",
        },
    ]

    output_csv = Path("outputs/real_vs_synthetic_bridge_summary.csv")
    output_md = Path("outputs/real_vs_synthetic_bridge_summary.md")
    write_csv(output_csv, bridge_rows)

    real_uncertainty_exact = next(
        row
        for row in bridge_rows
        if row["evidence_scope"] == "real_grouped_sequence"
        and row["key_metric"] == "mean_uncertainty_sequence_exact_delta"
    )
    real_uncertainty_topk = next(
        row
        for row in bridge_rows
        if row["evidence_scope"] == "real_grouped_sequence"
        and row["key_metric"] == "mean_uncertainty_sequence_topk_delta"
    )
    real_conformal_exact = next(
        row
        for row in bridge_rows
        if row["evidence_scope"] == "real_grouped_sequence"
        and row["key_metric"] == "mean_conformal_sequence_exact_delta"
    )
    synthetic_family = next(
        row
        for row in bridge_rows
        if row["evidence_scope"] == "synthetic_from_real_downstream"
    )
    real_downstream_uncertainty = next(
        row
        for row in bridge_rows
        if row["evidence_scope"] == "real_grouped_downstream"
        and row["task"] == "two_real_grouped_corpora_train_supported_ngram_path"
        and row["key_metric"] == "mean_uncertainty_downstream_exact_delta"
    )
    real_downstream_conformal = next(
        row
        for row in bridge_rows
        if row["evidence_scope"] == "real_grouped_downstream"
        and row["task"] == "two_real_grouped_corpora_train_supported_ngram_path"
        and row["key_metric"] == "mean_conformal_downstream_exact_delta"
    )
    real_downstream_coverage = next(
        row
        for row in bridge_rows
        if row["evidence_scope"] == "real_grouped_downstream"
        and row["task"] == "two_real_grouped_corpora_train_supported_ngram_path"
        and row["key_metric"] == "mean_full_path_coverage_upper_bound"
    )
    gold_agreement = next(
        row
        for row in bridge_rows
        if row["evidence_scope"] == "real_grouped_validation"
        and row["key_metric"] == "gold_pass_agreement_rate"
    )
    gold_noise = next(
        row
        for row in bridge_rows
        if row["evidence_scope"] == "real_grouped_validation"
        and row["key_metric"] == "gold_ocr_to_label_error_rate"
    )
    gold_change = next(
        row
        for row in bridge_rows
        if row["evidence_scope"] == "real_grouped_validation"
        and row["key_metric"] == "gold_minus_original_conformal_sequence_exact_delta"
    )

    markdown = [
        "# Real vs Synthetic Bridge Summary",
        "",
        "## Main Boundary",
        "",
        "- Real symbol-level evidence remains the strongest fully real claim in the repository.",
        "- Synthetic-from-real sequence and downstream tasks still carry most of the higher-level structural evidence.",
        "- The branch now has one strengthened OCR-grounded grouped benchmark and one second real grouped handwriting benchmark.",
        "",
        "## What Transfers",
        "",
        f"- Real grouped raw uncertainty exact-match delta: `{_fmt(real_uncertainty_exact['value'])}`.",
        f"- Real grouped raw uncertainty top-k delta: `{_fmt(real_uncertainty_topk['value'])}`.",
        f"- Real grouped conformal exact-match delta: `{_fmt(real_conformal_exact['value'])}`.",
        "- Interpretation: symbol/top-k rescue transfers to real grouped data more clearly than grouped exact match.",
        "",
        "## Two-Corpus Real Grouped Replication",
        "",
        f"- Historical Newspapers mean grouped top-k delta: `{_fmt(_mean([_to_float(row['mean_uncertainty_sequence_topk_delta']) for row in historical_rows]))}`.",
        f"- ScaDS.AI mean grouped top-k delta: `{_fmt(_mean([_to_float(row['mean_uncertainty_sequence_topk_delta']) for row in scads_rows]))}`.",
        f"- Two-corpus mean grouped top-k delta: `{_fmt(_mean([_to_float(row['mean_uncertainty_sequence_topk_delta']) for row in real_grouped_replication_rows]))}`.",
        f"- Two-corpus mean conformal exact delta: `{_fmt(_mean([_to_float(row['mean_conformal_sequence_exact_delta']) for row in real_grouped_replication_rows]))}`.",
        "- Interpretation: grouped top-k rescue now replicates across two real grouped corpora, while conformal exact-match gains remain corpus-dependent.",
        "",
        "## Real Downstream Structural Recovery",
        "",
        f"- Two-corpus mean raw downstream exact delta: `{_fmt(real_downstream_uncertainty['value'])}`.",
        f"- Two-corpus mean conformal downstream exact delta: `{_fmt(real_downstream_conformal['value'])}`.",
        f"- Two-corpus mean full-path coverage upper bound: `{_fmt(real_downstream_coverage['value'])}`.",
        f"- Historical Newspapers downstream coverage fraction: `{_fmt(_mean([_to_float(row['mean_downstream_coverage_fraction']) for row in historical_downstream_rows]))}`.",
        f"- ScaDS.AI downstream coverage fraction: `{_fmt(_mean([_to_float(row['mean_downstream_coverage_fraction']) for row in scads_downstream_rows]))}`.",
        "- Interpretation: the redesigned real downstream task fixes much of the coverage collapse, but exact downstream gains are still mixed rather than cleanly replicated.",
        "",
        "## Gold-Style Check",
        "",
        f"- Gold-style pass agreement rate: `{_fmt(gold_agreement['value'])}`.",
        f"- Gold-style OCR-to-label error rate: `{_fmt(gold_noise['value'])}`.",
        f"- Mean change in conformal exact delta after gold-style upgrade: `{_fmt(gold_change['value'])}`.",
        "- Interpretation: the current real grouped result appears stable to a stronger two-pass gold-style review, but it is still not equivalent to an independent multi-annotator gold annotation campaign.",
        "",
        "## What Remains Synthetic-Only",
        "",
        f"- Synthetic process-family downstream family delta: `{_fmt(synthetic_family['value'])}`.",
        "- Downstream structural family-identification remains synthetic-from-real only.",
        "",
        "## Scope Rows",
        "",
    ]
    for row in bridge_rows:
        markdown.append(
            f"- `{row['evidence_scope']}` / `{row['task']}` / `{row['key_metric']}`: `{_fmt(row['value'])}`. {row['note']}"
        )
    write_text(output_md, "\n".join(markdown) + "\n")
    print(json.dumps({"csv": str(output_csv), "markdown": str(output_md)}, indent=2))


if __name__ == "__main__":
    main()
