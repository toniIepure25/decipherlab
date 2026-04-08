from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from decipherlab.cli import app
from decipherlab.config import dump_config
from decipherlab.evaluation.runner import run_ablation_suite

from tests.helpers import build_test_config, create_real_manifest_fixture


def test_cli_help_smoke() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "DecipherLab research CLI" in result.stdout


def test_cli_generate_and_run_pipeline(tmp_path: Path) -> None:
    runner = CliRunner()
    config = build_test_config(tmp_path, noise_std=0.2)
    config_path = tmp_path / "config.yaml"
    dump_config(config, config_path)

    generated_dir = tmp_path / "generated"
    generate_result = runner.invoke(
        app,
        ["generate-synthetic", "--config", str(config_path), "--output-dir", str(generated_dir)],
    )
    assert generate_result.exit_code == 0
    assert (generated_dir / "manifest.json").exists()

    run_result = runner.invoke(app, ["run-pipeline", "--config", str(config_path)])
    assert run_result.exit_code == 0
    assert "Run complete:" in run_result.stdout


def test_cli_validate_manifest(tmp_path: Path) -> None:
    runner = CliRunner()
    manifest_path = create_real_manifest_fixture(tmp_path)
    result = runner.invoke(
        app,
        ["validate-manifest", "--manifest-path", str(manifest_path), "--manifest-format", "glyph_crop"],
    )
    assert result.exit_code == 0
    assert "fixture_real_manifest" in result.stdout
    assert "Sequences:" in result.stdout


def test_cli_build_manifest(tmp_path: Path) -> None:
    import csv
    import json

    runner = CliRunner()
    manifest_path = create_real_manifest_fixture(tmp_path)
    import yaml

    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    records_path = tmp_path / "records.csv"
    fieldnames = [
        "sequence_id",
        "position",
        "image_path",
        "split",
        "example_id",
        "group_id",
        "family",
        "transcription",
        "metadata",
    ]
    with records_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in payload["records"]:
            writer.writerow(record | {"metadata": json.dumps(record.get("metadata", {}))})

    output_manifest = tmp_path / "built.yaml"
    result = runner.invoke(
        app,
        [
            "build-manifest",
            "--records-path",
            str(records_path),
            "--output-path",
            str(output_manifest),
            "--dataset-name",
            "cli_fixture",
            "--image-root",
            str(manifest_path.parent),
        ],
    )
    assert result.exit_code == 0
    assert output_manifest.exists()
    assert "cli_fixture" in result.stdout


def test_cli_assemble_paper(tmp_path: Path) -> None:
    runner = CliRunner()
    config = build_test_config(tmp_path, noise_std=0.2)
    result = run_ablation_suite(config)
    paper_dir = tmp_path / "paper"
    cli_result = runner.invoke(
        app,
        ["assemble-paper", "--run-dir", str(result["run_dir"]), "--paper-dir", str(paper_dir)],
    )
    assert cli_result.exit_code == 0
    assert (paper_dir / "RESULTS.md").exists()
