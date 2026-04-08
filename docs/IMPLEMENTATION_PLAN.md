# Implementation Plan

## Completed Foundation

- Establish a `src/` package layout and a validated config system.
- Add reproducible run manifests, artifact directories, and CLI entrypoints.
- Document the scientific framing and repository agreements in Markdown.

## Completed Scientific MVP

- Generate synthetic glyph-image datasets across multiple explanatory families.
- Build an auditable glyph clustering baseline from deterministic image features.
- Represent transcription uncertainty as a top-k posterior over discovered glyph identities.
- Compute structural diagnostics and shuffled null baselines.
- Rank competing hypotheses and export evidence reports.

## Current Empirical Step

- Add a split-aware real glyph-crop manifest adapter.
- Fit a calibrated posterior model on the training split and tune temperature on validation.
- Compare four conditions under controlled ambiguity injected into the evaluation split.
- Measure symbol-level recovery, calibration, entropy behavior, hypothesis ranking changes, and structural recovery.
- Export paper-ready tables, plots, and failure summaries.

## Deliverables In This Phase

- `configs/experiments/real_manifest_uncertainty.yaml`
- `scripts/run_core_claim_experiment.py`
- split-aware manifest loader and schema validation
- calibrated posterior model artifacts
- ambiguity-sweep evaluation plots and paper-style summary

## Boundaries In This Phase

- No claim of full semantic decipherment
- No bundled historical benchmark release in the repository
- No UI or service layer
- No end-to-end neural OCR/HTR stack yet
- No claim that uncertainty always helps; the new protocol is designed to test that question honestly

## Validation Standard

- Unit coverage for configs, manifest loading, posterior handling, clustering, diagnostics, and family scorers
- Integration coverage for synthetic and real-manifest pipeline paths
- Evaluation artifacts for ambiguity sweeps
- Demo or experiment commands that produce complete run directories
