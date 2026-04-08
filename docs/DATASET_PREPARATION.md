# Dataset Preparation

## Manifest Format

DecipherLab currently supports a split-aware glyph-crop manifest with one record per crop.

Required fields per record:

- `sequence_id`: groups crops into one sequence/example
- `position`: zero-based order inside the sequence
- `image_path`: path relative to the manifest file
- `split`: one of `train`, `val`, `test`

Optional but recommended fields:

- `example_id`: stable crop identifier
- `group_id`: optional higher-level grouping for document, folio, or source-batch summaries
- `transcription`: symbol label for supervised posterior evaluation
- `family`: document/cipher family label for downstream family metrics
- `metadata`: arbitrary JSON/YAML object

Top-level manifest fields:

- `dataset_name`
- `unit_type: glyph_crop`
- `metadata`
- `records`

## Minimal YAML Example

```yaml
dataset_name: example_crops
unit_type: glyph_crop
metadata:
  source: "institution or collection"
records:
  - sequence_id: seq_001
    position: 0
    image_path: images/seq_001_000.png
    split: train
    example_id: seq_001_000
    group_id: doc_01
    transcription: g12
    family: monoalphabetic
```

## Preparation Rules

- Keep splits deterministic and document how they were created.
- Do not mix the same `sequence_id` across splits.
- Keep `position` contiguous within each sequence: `0..n-1`.
- Keep image paths relative to the manifest for portability.
- Preserve original crops outside the run output directories.
- If labels are partial, leave missing labels as absent or `null`; do not guess.
- Keep family labels consistent within a sequence if you intend to use downstream family metrics.
- Use `group_id` when you want grouped summaries later, but do not fabricate it if the source corpus has no meaningful grouping.

## Validation Workflow

Use the CLI validator before running experiments:

```bash
PYTHONPATH=src python3 -m decipherlab.cli validate-manifest --manifest-path path/to/manifest.yaml --manifest-format glyph_crop
```

This prints:

- dataset name
- available splits
- sequence and record counts
- warnings for underpowered splits, sparse train classes, or missing images

For larger datasets, the loader also records a validation summary inside dataset metadata so the evaluation pack can carry split and coverage context forward.

## Flat-Table To Manifest Workflow

If your crop metadata already exists as a flat CSV, JSONL, JSON, or YAML table, you can build a manifest directly:

```bash
PYTHONPATH=src python3 -m decipherlab.cli build-manifest \
  --records-path data/raw/my_dataset/records.csv \
  --output-path data/raw/my_dataset/manifest.yaml \
  --dataset-name my_dataset
```

Expected table columns:

- required: `sequence_id`, `position`, `image_path`, `split`
- optional: `example_id`, `group_id`, `transcription`, `family`, `metadata`

For CSV inputs, the `metadata` column should contain a JSON object string when used.

## Concrete External-Dataset Workflow

1. Export one flat records table from your dataset preparation environment.
2. Build `manifest.yaml` with the manifest builder command above.
3. Run `validate-manifest` and inspect split balance, label coverage, group coverage, and warnings.
4. Copy [real_manifest_large_paper_pack.yaml](/home/tonystark/Desktop/decipher/configs/experiments/real_manifest_large_paper_pack.yaml) and set `dataset.manifest_path`.
5. Adjust `dataset.min_*_warning` thresholds to match the scale of the corpus.
6. Launch the full evidence pack:

```bash
python3 scripts/run_real_manifest_paper_pack.py --config configs/experiments/real_manifest_large_paper_pack.yaml --paper-dir paper
```

7. Inspect `dataset_summary.md`, `main_comparison_with_ci.csv`, `seed_summary.csv`, and `failure_case_summary.csv` before drafting claims.

## Omniglot Workflow

Omniglot is the current selected external dataset for the paper pack because it offers real handwritten glyph crops, alphabet-level grouping, and clean compatibility with the existing manifest-based protocol.

If the Omniglot archives are already present locally, regenerate the manifest and integration note with:

```bash
python3 scripts/prepare_omniglot_manifest.py \
  --background-zip data/raw/omniglot/images_background.zip \
  --evaluation-zip data/raw/omniglot/images_evaluation.zip \
  --extract-dir data/raw/omniglot/extracted \
  --manifest-path data/raw/omniglot/manifest.yaml \
  --note-path data/raw/omniglot/README.md \
  --split-seed 23
```

Then validate the manifest and run the full evidence pack:

```bash
PYTHONPATH=src python3 -m decipherlab.cli validate-manifest \
  --manifest-path data/raw/omniglot/manifest.yaml \
  --manifest-format glyph_crop

python3 scripts/run_real_manifest_paper_pack.py \
  --config configs/experiments/omniglot_paper_pack.yaml \
  --paper-dir paper
```

This writes the standard paper-pack outputs under `outputs/runs/<timestamp>_omniglot_paper_pack_evaluation/` and refreshes the measured manuscript sections in `paper/`.

## scikit-learn Digits Workflow

The secondary dataset in the current cross-dataset paper pack is the bundled scikit-learn digits corpus. It is not a historical manuscript dataset, but it provides a real handwritten-symbol corpus with very different visual statistics from Omniglot and requires no network download in this environment.

Prepare the manifest and integration note with:

```bash
python3 scripts/prepare_sklearn_digits_manifest.py \
  --output-dir data/raw/sklearn_digits \
  --manifest-path data/raw/sklearn_digits/manifest.yaml \
  --note-path data/raw/sklearn_digits/README.md \
  --train-count-per-class 100 \
  --val-count-per-class 30 \
  --split-seed 23
```

Then validate and run the frozen paper pack:

```bash
PYTHONPATH=src python3 -m decipherlab.cli validate-manifest \
  --manifest-path data/raw/sklearn_digits/manifest.yaml \
  --manifest-format glyph_crop

python3 scripts/run_real_manifest_paper_pack.py \
  --config configs/experiments/sklearn_digits_paper_pack.yaml \
  --paper-dir paper
```

For the cross-dataset synthesis against Omniglot, use:

```bash
python3 scripts/build_cross_dataset_summary.py \
  --omniglot-run outputs/runs/20260407T150327Z_omniglot_paper_pack_evaluation \
  --secondary-run outputs/runs/<timestamp>_sklearn_digits_paper_pack_evaluation \
  --secondary-label sklearn_digits_crops \
  --output-root outputs
```

## Kuzushiji-49 Workflow

Kuzushiji-49 is the current historically grounded dataset in the paper pack. The full corpus is downloaded through OpenML, but the evaluation manifest uses a deterministic balanced cap so the frozen multi-seed 2x2 comparison remains tractable while preserving all `49` classes.

Prepare the manifest and integration note with:

```bash
python3 scripts/prepare_kuzushiji49_manifest.py \
  --output-dir data/raw/kuzushiji49 \
  --manifest-path data/raw/kuzushiji49/manifest_balanced_49x300_75_75.yaml \
  --note-path data/raw/kuzushiji49/README.md \
  --train-count-per-class 300 \
  --val-count-per-class 75 \
  --test-count-per-class 75 \
  --split-seed 23
```

Then validate and run the frozen paper pack:

```bash
PYTHONPATH=src python3 -m decipherlab.cli validate-manifest \
  --manifest-path data/raw/kuzushiji49/manifest_balanced_49x300_75_75.yaml \
  --manifest-format glyph_crop

python3 scripts/run_real_manifest_paper_pack.py \
  --config configs/experiments/kuzushiji49_balanced_subset_paper_pack.yaml \
  --paper-dir paper
```

To extend the cross-dataset synthesis with Omniglot, scikit-learn digits, and Kuzushiji-49, use:

```bash
python3 scripts/build_cross_dataset_summary.py \
  --omniglot-run outputs/runs/20260407T150327Z_omniglot_paper_pack_evaluation \
  --secondary-run outputs/runs/20260407T191829Z_sklearn_digits_paper_pack_evaluation \
  --secondary-label sklearn_digits_crops \
  --tertiary-run outputs/runs/<timestamp>_kuzushiji49_balanced_subset_paper_pack_evaluation \
  --tertiary-label kuzushiji49_balanced_crops \
  --output-root outputs
```

## Recommended Real-Data Protocol

1. Prepare a manifest with stable `train`, `val`, and `test` splits.
2. Put symbol labels in `transcription` wherever available.
3. Add family labels only when they are defensible.
4. Tune `dataset.min_*_warning` thresholds to match the scale of the corpus.
5. Use `experiment.seed_sweep` when you want repeated-run robustness checks without changing the split definition.
6. Run the comparison suite with [real_manifest_uncertainty.yaml](/home/tonystark/Desktop/decipher/configs/experiments/real_manifest_uncertainty.yaml) as a template.
7. Inspect failure summaries before drafting conclusions.
