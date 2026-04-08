# DecipherLab Architecture

## System Overview

DecipherLab is organized as a research platform with explicit boundaries between image evidence, uncertain symbolic representations, structural diagnostics, and ranked explanatory hypotheses. The design goal is to support credible experiments, ablations, and future paper writing without hiding assumptions inside one monolithic model.

## Data Flow

1. **Ingest**
   - Read split-aware manifest datasets or generate synthetic benchmarks with deterministic train/val/test assignment.
   - Normalize examples into shared `DatasetCollection`, `SequenceExample`, and `GlyphCrop` objects.
2. **Vision / Glyphs**
   - Render or load glyph crops.
   - Extract auditable image features.
   - Optionally corrupt only the evaluation split to simulate realistic ambiguity.
   - Cluster evaluation glyphs into a working inventory and estimate allograph structure.
3. **Transcription**
   - Fit a calibrated posterior model on the training split.
   - Use a semi-learned glyph embedding with a supervised probabilistic classifier when symbol labels are available.
   - Fall back to a prototype-based unsupervised posterior when labels are absent.
   - Preserve uncertainty rather than hard-collapsing to a single transcript.
4. **Structure**
   - Compute entropy, conditional entropy, index of coincidence, repeat structure, adjacency graphs, and null controls.
5. **Hypotheses / Scoring**
   - Score competing worlds: `unknown_script`, `monoalphabetic`, `homophonic`, `transposition_heuristic`, `pseudo_text_null`.
   - Produce comparable evidence objects with caveats, not opaque labels.
6. **Evaluation / Reporting**
   - Compare four controlled conditions: fixed/uncertainty-aware crossed with heuristic/calibrated posteriors.
   - Export machine-readable tables, failure summaries, plots, and manuscript-friendly Markdown artifacts.

## Module Boundaries

- `benchmarks`: synthetic data generation and known-ground-truth tasks
- `ingest`: dataset manifests, manifest preparation helpers, and adapters
- `vision`: synthetic glyph rendering, image corruption, and crop-level helpers
- `glyphs`: feature extraction, clustering, and inventory recovery
- `transcription`: posterior representation, calibrated posterior models, and transcript utilities
- `structure`: diagnostics and null baselines
- `hypotheses`: family-specific heuristic scorers
- `scoring`: ranking and report assembly
- `evaluation`: metrics, ablations, and benchmark orchestration
- `evaluation/failure_analysis`: structured negative-case analysis
- `evaluation/results_pack`: paper-ready tables, plots, captions, and draft result text
- `manuscript`: assembly of manuscript-facing sections from a completed results pack
- `workflows`: thin orchestration layer for manifest-backed paper-pack runs
- `utils`: configs, seeds, logging, artifact IO

## Dataset Adapter Boundary

The primary real-data interface is a glyph-crop manifest adapter:

- one manifest record per crop
- relative image paths
- sequence grouping via `sequence_id`
- explicit `train` / `val` / `test` splits
- optional `transcription` labels
- optional `family` metadata

Dataset-specific assumptions are kept inside `ingest/`. The rest of the pipeline only consumes normalized `DatasetCollection` objects and split names from config.

For external validation, the recommended workflow is:

1. prepare a flat records table outside the core pipeline
2. convert it into a validated glyph-crop manifest with the manifest builder
3. validate split/class/group coverage
4. run the real-data paper pack
5. refresh manuscript drafts from the saved results pack

## Uncertainty Propagation Concept

The primary implemented uncertainty object is a `TranscriptionPosterior`:

- per-position candidate glyph identities
- normalized log-probabilities
- optional boundary uncertainty
- deterministic serialization for downstream scoring

The current generator is a calibrated baseline, not a full lattice decoder. It uses:

- learned scaling and PCA embedding from the training split
- multinomial logistic classification when symbol supervision exists
- temperature tuning on the validation split
- top-k posterior truncation for downstream use

The interface is intentionally simple so future work can replace the posterior generator with stronger OCR/HTR or Bayesian components without changing downstream modules.

## Research vs Engineering Split

Research-facing code:

- synthetic benchmark generation
- triage diagnostics
- hypothesis scoring
- evaluation metrics and ablations

Engineering-facing code:

- config validation
- artifact management
- CLI entrypoints
- structured logging and manifests

The split is important because scientific credibility depends on both. Research logic should stay inspectable, while execution plumbing should make runs repeatable.

## Current Architecture Decisions

- Synthetic-first development, followed by a manifest-backed external validation run on Omniglot
- Manifest-backed real-data adapter for external glyph-crop datasets
- Image features plus a simple learned embedding, rather than a full neural recognizer
- Clean train/val/test split handling throughout the pipeline
- Heuristic family scorers before full Bayesian search
- Null comparisons built into the triage stage rather than added later
- Ambiguity injection on the evaluation split to test uncertainty sensitivity
- Report artifacts instead of a UI in the current phase

## Extension Paths

- Replace PCA-plus-logistic baselines with stronger OCR/HTR encoders
- Add real manuscript/cipher adapters under `ingest/` beyond glyph-crop manifests
- Upgrade heuristic scorers to probabilistic search engines
- Add scribal-hand clustering and layout analysis
- Add interactive analyst tooling on top of saved artifacts
