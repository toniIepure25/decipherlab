# Paper Results Notes

## Supported Narrow Claim

The current codebase supports the following claim:

> Preserving transcription uncertainty can improve some downstream decipherment-related signals, especially symbol top-k recovery, under controlled ambiguity on manifest-backed glyph-crop datasets.

This claim is supported only under the implemented protocol:

- train/val/test split held fixed
- ambiguity injected only at evaluation time
- four-way comparison:
  - fixed + heuristic posterior
  - fixed + calibrated posterior
  - uncertainty-aware + heuristic posterior
  - uncertainty-aware + calibrated posterior
- deterministic bootstrap confidence intervals on the headline metrics
- optional seed sweeps with aggregate mean and standard deviation reporting
- reportable metrics for symbol recovery, calibration, entropy, and downstream family ranking where labels exist

The current external-validation evidence in the repository now spans three real corpora:

- Omniglot with a deterministic within-character split
- scikit-learn digits with deterministic per-class caps
- Kuzushiji-49 with a full OpenML download and a deterministic balanced evaluation subset

This means the supported claim is no longer grounded in a single external corpus or only in synthetic fixtures. It is still a symbol-level claim, but it now replicates across three real datasets, including one historically grounded corpus.

## Not Yet Supported

- broad historical-manuscript generalization
- improvement on full decipherment rather than decipherment-related inference
- superiority of the calibrated posterior on every dataset or ambiguity regime
- semantic recovery claims
- downstream family-ranking gains when the dataset does not carry enough family signal

## Main Reusable Artifacts

The comparison runner writes:

- `main_comparison_table.{csv,json,md}`
- `main_comparison_with_ci.{csv,json}`
- `ambiguity_sweep_table.{csv,json}`
- `ambiguity_sweep_with_ci.{csv,json}`
- `seed_summary.{csv,json}`
- `calibration_table.{csv,json}`
- `pairwise_effect_table.{csv,json}`
- `pairwise_effect_summary.{csv,json}`
- `failure_case_summary.{csv,json,md}`
- `results_section_draft.md`
- `limitations_section_draft.md`
- `figure_captions.md`
- manuscript-facing paper sections under `paper/` when refreshed with the paper assembly utility

## Paper Assembly Workflow

The recommended assembly path is:

1. run the full evidence pack on a manifest-backed dataset
2. inspect the saved tables and failure summaries
3. refresh manuscript sections with:

```bash
python3 scripts/assemble_paper_from_results.py --run-dir outputs/runs/<run_dir> --paper-dir paper
```

This updates `paper/EXPERIMENTS.md`, `paper/RESULTS.md`, and `paper/LIMITATIONS.md` from measured outputs rather than from hand-written impressions.

## Recommended Results Narrative

- Start with the four-condition comparison, not a single positive number.
- Emphasize top-k recovery and calibration together, but keep them analytically separate.
- Report where uncertainty helps and where it does not.
- Report the interval around the estimate, not only the point value.
- If multi-seed results are available, include them in the main table or appendix rather than only in notes.
- Include failure cases in the main text or appendix, not only the supplement.
- Phrase conclusions in terms of ambiguity robustness, not “decipherment solved.”
