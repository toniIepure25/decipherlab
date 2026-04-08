from __future__ import annotations

from collections import Counter

import numpy as np
import yaml

from decipherlab.ingest import kuzushiji49 as k49


class _FakeBunch(dict):
    def __getattr__(self, name: str):
        return self[name]


def test_build_kuzushiji49_balanced_manifest_and_note(tmp_path, monkeypatch) -> None:
    labels = np.array(["0"] * 8 + ["1"] * 8 + ["2"] * 8, dtype=object)
    images = np.arange(labels.shape[0] * 28 * 28, dtype=np.uint8).reshape(labels.shape[0], 28 * 28)

    def _fake_fetch_openml(*args, **kwargs):
        return _FakeBunch(data=images, target=labels)

    monkeypatch.setattr(k49, "fetch_openml", _fake_fetch_openml)
    monkeypatch.setattr(k49, "KUZUSHIJI49_CACHE_FILE", tmp_path / "Kuzushiji-49.arff.gz")
    (tmp_path / "Kuzushiji-49.arff.gz").write_bytes(b"cache")

    dataset_root = tmp_path / "k49"
    manifest_path = dataset_root / "manifest.yaml"

    manifest = k49.build_kuzushiji49_balanced_manifest(
        output_dir=dataset_root,
        manifest_path=manifest_path,
        train_count_per_class=3,
        val_count_per_class=2,
        test_count_per_class=2,
        split_seed=11,
        image_subdir="images_test",
    )

    assert manifest.dataset_name == "kuzushiji49_balanced_crops"
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert len(payload["records"]) == 21
    assert payload["metadata"]["split_counts"] == {"test": 6, "train": 9, "val": 6}
    assert Counter(record["transcription"] for record in payload["records"]) == {"0": 7, "1": 7, "2": 7}

    summary = k49.summarize_kuzushiji49_local_artifacts(dataset_root=dataset_root, manifest_path=manifest_path)
    assert summary["cache_size"] == 5
    assert summary["manifest_summary"]["label_count"] == 3
    note = k49.format_kuzushiji49_integration_note(dataset_root=dataset_root, manifest_path=manifest_path)
    assert "balanced capped subset with all classes preserved" in note
    assert "- `21` total labeled crops" in note
