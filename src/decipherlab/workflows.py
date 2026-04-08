from __future__ import annotations

from pathlib import Path

from decipherlab.config import DecipherLabConfig, load_config
from decipherlab.evaluation.runner import run_ablation_suite
from decipherlab.ingest.manifest import load_glyph_crop_manifest_dataset
from decipherlab.ingest.schema import GlyphCropManifest
from decipherlab.ingest.validation import format_manifest_summary_markdown, summarize_glyph_crop_manifest
from decipherlab.manuscript import assemble_paper_sections
from decipherlab.utils.io import write_json, write_text
import yaml


def summarize_real_manifest_dataset(config: DecipherLabConfig | str | Path) -> dict[str, object]:
    resolved = load_config(config) if isinstance(config, (str, Path)) else config
    if resolved.dataset.source != "manifest":
        raise ValueError("Real-manifest summary requires dataset.source='manifest'.")
    if resolved.dataset.manifest_path is None:
        raise ValueError("manifest_path is required for real-manifest summary.")
    manifest_path = Path(resolved.dataset.manifest_path)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest = GlyphCropManifest.model_validate(payload)
    summary = summarize_glyph_crop_manifest(
        manifest,
        manifest_path=manifest_path,
        dataset_config=resolved.dataset,
    )
    summary.update(
        {
            "manifest_path": str(manifest_path),
            "split_counts": summary.get("split_sequence_counts"),
            "label_coverage": summary.get("split_label_coverage"),
            "family_label_coverage": summary.get("split_family_coverage"),
        }
    )
    return summary


def run_real_manifest_paper_pack(
    config: DecipherLabConfig | str | Path,
    paper_dir: str | Path | None = "paper",
) -> dict[str, object]:
    resolved = load_config(config) if isinstance(config, (str, Path)) else config
    if resolved.dataset.source != "manifest":
        raise ValueError("run_real_manifest_paper_pack requires dataset.source='manifest'.")
    if resolved.dataset.manifest_path is None or not Path(resolved.dataset.manifest_path).exists():
        raise ValueError("The configured manifest_path does not exist.")

    dataset_summary = summarize_real_manifest_dataset(resolved)
    result = run_ablation_suite(resolved)
    run_dir = Path(result["run_dir"])
    write_json(run_dir / "dataset_summary.json", dataset_summary)
    write_text(run_dir / "dataset_summary.md", format_manifest_summary_markdown(dataset_summary))

    paper_outputs: dict[str, str] | None = None
    if paper_dir is not None:
        paper_outputs = assemble_paper_sections(run_dir=run_dir, paper_dir=paper_dir)

    return {
        **result,
        "dataset_summary": dataset_summary,
        "paper_outputs": paper_outputs,
    }
