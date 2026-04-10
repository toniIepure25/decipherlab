# DecipherLab

[![CI](https://github.com/toniIepure25/decipherlab/actions/workflows/ci.yml/badge.svg)](https://github.com/toniIepure25/decipherlab/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

DecipherLab is a research-grade, uncertainty-aware experimentation platform for deciphering unknown scripts, historical ciphers, and cryptic artifacts. It is intentionally framed as an evidence-ranking and structure-discovery system, not a universal translator.

## Portfolio Snapshot

- Reproducible synthetic-first vertical slice
- Uncertainty-preserving transcription and hypothesis ranking
- Auditable baselines with explicit null controls
- Paper-ready outputs and manuscript refresh scripts
- Sequence-branch paper package with real grouped replication and support-aware propagation analysis
- DAS 2026 LaTeX submission workspace under `paper/das/paper_1`
- Clear separation between structure discovery and semantic recovery

The first implemented milestone is a synthetic, fully reproducible pipeline:

1. Generate synthetic glyph sequences and image crops for cipher-like and pseudo-text families.
2. Preserve transcription uncertainty with posterior distributions over discovered glyph identities.
3. Cluster glyph allographs with simple, auditable image features.
4. Compute structural diagnostics and null comparisons.
5. Rank competing explanatory families and export evidence reports.

## Scientific Positioning

- Preserve uncertainty rather than collapsing to a single transcript too early.
- Separate structure discovery from semantic recovery.
- Rank hypotheses with explicit evidence and null controls.
- Prefer auditable baselines over opaque, overclaimed models.
- Treat failure to reject nulls as a meaningful outcome.

## Why This Repository Is Portfolio-Ready

- It demonstrates end-to-end engineering, from data ingest to reproducible outputs.
- It presents scientific restraint: the system reports evidence and uncertainty instead of pretending to solve the problem universally.
- It includes experiment configs, scripts, test coverage, and manuscript assets in one coherent package.
- It is designed to be legible to both engineering reviewers and research reviewers.

## Repository Layout

```text
configs/              Typed YAML experiment configs
data/                 Raw, interim, processed, and synthetic datasets
docs/                 Architecture, protocol, roadmap, and blueprint notes
outputs/runs/         Reproducible experiment outputs
paper/                Workshop-paper skeleton and run-refreshed manuscript drafts
scripts/              Convenience wrappers and future automation hooks
src/decipherlab/      Research code
tests/                Unit and integration coverage
```

## Current Paper Assets

- `submission_bplus/`: reviewer-facing Markdown package for the stronger sequence branch
- `paper/das/paper_1/`: IEEE/DAS LaTeX submission package built from the stronger sequence-branch evidence

The current stronger paper is positioned as a support-aware uncertainty propagation study for low-resource glyph and grouped recognition. Its strongest supported real-data claim is that preserved transcription uncertainty improves grouped top-k recovery across two token-aligned corpora, while higher-level propagation remains support-gated and selective.

## Quick Start

```bash
uv sync
uv run pytest
uv run decipherlab demo synthetic-uncertainty
```

If you prefer the system interpreter:

```bash
python3 -m pytest
PYTHONPATH=src python3 -m decipherlab.cli demo synthetic-uncertainty
```

To run the focused fixed-vs-uncertainty experiment on a manifest-backed crop dataset:

```bash
python3 scripts/run_core_claim_experiment.py --config configs/experiments/real_manifest_uncertainty.yaml
```

To validate a real glyph-crop manifest before running experiments:

```bash
PYTHONPATH=src python3 -m decipherlab.cli validate-manifest --manifest-path path/to/manifest.yaml --manifest-format glyph_crop
```

To build a manifest from a flat records table:

```bash
PYTHONPATH=src python3 -m decipherlab.cli build-manifest --records-path data/raw/my_dataset/records.csv --output-path data/raw/my_dataset/manifest.yaml --dataset-name my_dataset
```

To run the full real-data paper pack and refresh manuscript drafts:

```bash
python3 scripts/run_real_manifest_paper_pack.py --config configs/experiments/real_manifest_large_paper_pack.yaml --paper-dir paper
```

To run the current external-validation paper pack on Omniglot specifically:

```bash
python3 scripts/prepare_omniglot_manifest.py --manifest-path data/raw/omniglot/manifest.yaml --note-path data/raw/omniglot/README.md
python3 scripts/run_real_manifest_paper_pack.py --config configs/experiments/omniglot_paper_pack.yaml --paper-dir paper
```

To refresh manuscript sections from an existing results pack:

```bash
python3 scripts/assemble_paper_from_results.py --run-dir outputs/runs/<run_dir> --paper-dir paper
```

## First MVP Capabilities

- Synthetic benchmark generation for `monoalphabetic`, `homophonic`, `transposition`, and `pseudo_text`.
- Split-aware manifest ingest for external glyph-crop datasets.
- Uncertainty-aware transcription as a top-k posterior over calibrated glyph identity candidates.
- Structural triage with entropy, conditional entropy, index of coincidence, repeat structure, adjacency graphs, and compressibility.
- Hypothesis ranking over `unknown_script`, `monoalphabetic`, `homophonic`, `transposition_heuristic`, and `pseudo_text_null`.
- Evaluation harness for four-way comparison under injected ambiguity.
- Results-pack export with CSV/JSON tables, bootstrap confidence intervals, seed summaries, plots, figure captions, and failure summaries.
- Generic manifest-building helper for flat CSV/JSONL/JSON/YAML glyph-crop records.
- Real-data paper-pack workflow that writes dataset summaries and refreshes manuscript drafts.

## Environment And Reproducibility

- Python `>=3.10`
- `uv` for environment management
- Fixed seeds in configs and run manifests
- Structured outputs under `outputs/runs/<timestamp>_<experiment>/`
- MIT licensed with a citation file for reuse and attribution

Each run writes:

- a validated config snapshot
- a run manifest with seed and config hash
- machine-readable metrics
- a Markdown evidence report

## Current Limitations

- The current pipeline is synthetic-first. External historical datasets are not bundled.
- Hypothesis ranking is heuristic and evidence-oriented, not full Bayesian decipherment.
- The transposition family is intentionally lightweight in the first pass and is labeled heuristic in reports.
- No semantic translation layer is implemented or claimed.

## Core Documents

- [Research Blueprint](docs/RESEARCH_BLUEPRINT.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md)
- [Experiment Protocol](docs/EXPERIMENT_PROTOCOL.md)
- [Roadmap](docs/ROADMAP.md)
- [Working Agreements](AGENTS.md)
