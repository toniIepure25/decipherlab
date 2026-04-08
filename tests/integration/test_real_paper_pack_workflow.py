from __future__ import annotations

from decipherlab.workflows import run_real_manifest_paper_pack

from tests.helpers import build_test_config, create_real_manifest_fixture


def test_real_manifest_paper_pack_writes_dataset_and_paper_artifacts(tmp_path) -> None:
    manifest_path = create_real_manifest_fixture(tmp_path)
    config = build_test_config(tmp_path, noise_std=0.1)
    config.dataset.source = "manifest"
    config.dataset.manifest_path = manifest_path
    config.dataset.manifest_format = "glyph_crop"
    config.dataset.train_split = "train"
    config.dataset.val_split = "val"
    config.dataset.evaluation_split = "test"
    config.experiment.seed_sweep = [17]
    paper_dir = tmp_path / "paper"

    result = run_real_manifest_paper_pack(config, paper_dir=paper_dir)
    run_dir = result["run_dir"]
    assert (run_dir / "dataset_summary.json").exists()
    assert (run_dir / "dataset_summary.md").exists()
    assert (paper_dir / "EXPERIMENTS.md").exists()
    assert (paper_dir / "RESULTS.md").exists()
    assert (paper_dir / "LIMITATIONS.md").exists()
