from __future__ import annotations

from pathlib import Path

import typer

from decipherlab.benchmarks.synthetic import generate_synthetic_dataset, save_synthetic_dataset
from decipherlab.config import load_config
from decipherlab.ingest.preparation import build_glyph_crop_manifest_from_table
from decipherlab.evaluation.runner import run_ablation_suite
from decipherlab.ingest.manifest import load_glyph_crop_manifest_dataset, load_synthetic_manifest_dataset
from decipherlab.ingest.schema import GlyphCropManifest
from decipherlab.ingest.validation import summarize_glyph_crop_manifest
from decipherlab.manuscript import assemble_paper_sections
from decipherlab.pipeline import run_pipeline
from decipherlab.workflows import run_real_manifest_paper_pack, summarize_real_manifest_dataset

app = typer.Typer(help="DecipherLab research CLI.", no_args_is_help=True)
demo_app = typer.Typer(help="Reproducible demos.")
app.add_typer(demo_app, name="demo")


@app.command("generate-synthetic")
def generate_synthetic(
    config: Path = typer.Option(Path("configs/experiments/synthetic_mvp.yaml"), exists=True),
    output_dir: Path = typer.Option(Path("data/synthetic/generated/synthetic_mvp")),
) -> None:
    """Generate a synthetic benchmark dataset and manifest."""

    resolved = load_config(config)
    bundle = generate_synthetic_dataset(resolved.synthetic, seed=resolved.experiment.seed)
    manifest_path = save_synthetic_dataset(
        bundle,
        output_dir=output_dir,
        dataset_name=resolved.experiment.name,
        seed=resolved.experiment.seed,
    )
    typer.echo(f"Synthetic dataset written to {manifest_path}")


@app.command("run-pipeline")
def run_pipeline_command(
    config: Path = typer.Option(Path("configs/experiments/synthetic_mvp.yaml"), exists=True),
    posterior_mode: str = typer.Option("uncertainty", help="One of: uncertainty, fixed."),
    ambiguity_level: float = typer.Option(0.0, help="Amount of evaluation-time image ambiguity to inject."),
) -> None:
    """Run the image-to-hypothesis pipeline."""

    if posterior_mode not in {"uncertainty", "fixed"}:
        raise typer.BadParameter("posterior_mode must be 'uncertainty' or 'fixed'.")
    result = run_pipeline(config, posterior_mode=posterior_mode, ambiguity_level=ambiguity_level)  # type: ignore[arg-type]
    typer.echo(f"Run complete: {result['run_dir']}")


@app.command("evaluate")
def evaluate_command(
    config: Path = typer.Option(Path("configs/experiments/synthetic_mvp.yaml"), exists=True),
) -> None:
    """Run the fixed-vs-uncertainty ablation suite."""

    result = run_ablation_suite(config)
    typer.echo(f"Evaluation summary written to {result['run_dir']}")


@app.command("build-manifest")
def build_manifest_command(
    records_path: Path = typer.Option(..., exists=True, help="CSV/JSONL/JSON/YAML table of glyph-crop records."),
    output_path: Path = typer.Option(..., help="Destination manifest YAML path."),
    dataset_name: str = typer.Option(..., help="Dataset name to embed in the manifest."),
    image_root: Path | None = typer.Option(None, help="Optional root to prepend to relative image paths."),
) -> None:
    """Build a validated glyph-crop manifest from a flat records table."""

    manifest = build_glyph_crop_manifest_from_table(
        records_path=records_path,
        output_path=output_path,
        dataset_name=dataset_name,
        image_root=image_root,
    )
    typer.echo(f"Manifest written to {output_path}")
    typer.echo(f"Dataset: {manifest.dataset_name}")
    typer.echo(f"Records: {len(manifest.records)}")


@app.command("validate-manifest")
def validate_manifest_command(
    manifest_path: Path = typer.Option(..., exists=True),
    manifest_format: str = typer.Option("glyph_crop", help="One of: glyph_crop, synthetic_npz."),
) -> None:
    """Validate and summarize a manifest-backed dataset."""

    if manifest_format == "glyph_crop":
        import json
        import yaml

        if manifest_path.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        else:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = GlyphCropManifest.model_validate(payload)
        summary = summarize_glyph_crop_manifest(manifest, manifest_path=manifest_path)
        typer.echo(f"Dataset: {summary['dataset_name']}")
        typer.echo(f"Splits: {', '.join(summary['split_sequence_counts'])}")
        typer.echo(f"Sequences: {summary['sequence_count']}")
        typer.echo(f"Records: {summary['record_count']}")
        typer.echo(f"Warnings: {summary['warning_count']}")
        if summary["warnings"]:
            for warning in summary["warnings"]:
                typer.echo(f"- {warning}")
        return
    elif manifest_format == "synthetic_npz":
        dataset = load_synthetic_manifest_dataset(manifest_path)
    else:
        raise typer.BadParameter("manifest_format must be 'glyph_crop' or 'synthetic_npz'.")
    typer.echo(f"Dataset: {dataset.dataset_name}")
    typer.echo(f"Splits: {', '.join(dataset.split_names())}")
    typer.echo(f"Examples: {dataset.count_examples()}")
    typer.echo(f"Metadata: {dataset.metadata}")


@app.command("run-real-paper-pack")
def run_real_paper_pack_command(
    config: Path = typer.Option(Path("configs/experiments/real_manifest_large_paper_pack.yaml"), exists=True),
    paper_dir: Path = typer.Option(Path("paper"), help="Directory where manuscript drafts should be refreshed."),
) -> None:
    """Run the full real-data paper pack and refresh manuscript drafts."""

    summary = summarize_real_manifest_dataset(config)
    typer.echo(f"Dataset: {summary['dataset_name']}")
    typer.echo(f"Sequences: {summary['sequence_count']}")
    typer.echo(f"Split counts: {summary['split_sequence_counts']}")
    typer.echo(f"Label coverage: {summary['split_label_coverage']}")
    typer.echo(f"Groups: {summary['split_group_counts']}")
    if summary["warnings"]:
        typer.echo("Warnings:")
        for warning in summary["warnings"]:
            typer.echo(f"- {warning}")
    result = run_real_manifest_paper_pack(config, paper_dir=paper_dir)
    typer.echo(f"Paper pack complete: {result['run_dir']}")
    if result["paper_outputs"] is not None:
        typer.echo(f"Paper drafts refreshed under: {paper_dir}")


@app.command("assemble-paper")
def assemble_paper_command(
    run_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    paper_dir: Path = typer.Option(Path("paper"), help="Directory where manuscript drafts should be written."),
) -> None:
    """Assemble manuscript-ready drafts from an existing results pack."""

    outputs = assemble_paper_sections(run_dir=run_dir, paper_dir=paper_dir)
    typer.echo(f"Experiments draft: {outputs['experiments']}")
    typer.echo(f"Results draft: {outputs['results']}")
    typer.echo(f"Limitations draft: {outputs['limitations']}")


@demo_app.command("synthetic-uncertainty")
def demo_synthetic_uncertainty() -> None:
    """Run the default synthetic uncertainty-aware demo."""

    result = run_pipeline(Path("configs/experiments/synthetic_mvp.yaml"), posterior_mode="uncertainty")
    typer.echo(f"Demo run complete: {result['run_dir']}")


if __name__ == "__main__":
    app()
