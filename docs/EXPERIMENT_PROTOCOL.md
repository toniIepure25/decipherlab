# Experiment Protocol

## Benchmark Tasks

1. Glyph inventory recovery from synthetic image crops
2. Symbol posterior recovery on split-aware glyph-crop manifests
3. Hypothesis-family ranking over `monoalphabetic`, `homophonic`, `transposition`, and `pseudo_text`
4. Robustness under injected ambiguity on the evaluation split
5. Four-way comparison:
   - fixed + heuristic posterior
   - fixed + calibrated posterior
   - uncertainty-aware + heuristic posterior
   - uncertainty-aware + calibrated posterior
6. Rejection behavior on shuffled and pseudo-text controls
7. Bootstrap confidence intervals on symbol and downstream metrics
8. Optional multi-seed robustness sweeps with preserved per-run artifacts

## Metrics

- Symbol top-1 accuracy
- Symbol top-k accuracy
- Symbol negative log-likelihood
- Symbol expected calibration error
- Mean posterior entropy
- Mean entropy on correct vs incorrect symbol predictions
- Family classification accuracy when family labels exist
- Top-k hypothesis success when family labels exist
- Adjusted Rand Index for glyph clustering
- Structural metric recovery against clean references where applicable
- Family-level Expected Calibration Error
- Brier score on family probabilities
- Null rejection gap: score margin against shuffled controls
- Best-family change rate between fixed and uncertainty-aware inference
- Bootstrap confidence intervals for top-1, top-k, NLL, ECE, family top-k, and structural recovery
- Seed-level mean and standard deviation summaries when seed sweeps are configured

## Baselines

- Fixed-transcript pipeline using hard posterior collapse
- Uncertainty-aware pipeline using top-k posterior glyph identities
- Cluster-distance softmax as the old posterior baseline
- Calibrated classifier posterior as the current baseline
- Pseudo-text null baseline
- Shuffled transcript control

## Ablation Plan

- Remove uncertainty propagation
- Remove calibration and revert to distance softmax
- Remove null comparisons
- Reduce the posterior model to unsupervised prototype fallback
- Increase ambiguity on the evaluation split and observe degradation
- Replace a single seed with a seed sweep while preserving the same splits and ambiguity schedule

## Seed Policy

- Every config must specify a seed.
- The seed is written into the run manifest and reused for dataset generation, split handling, ambiguity injection, clustering initializations, and evaluation.
- If a study requires multiple seeds, set `experiment.seed_sweep` and preserve the base seed as the default single-run reference.
- Multi-seed tables should report both mean and dispersion, not only the best seed.

## Statistical Procedure

- Confidence intervals use deterministic bootstrap resampling with replacement.
- Symbol metrics are resampled at the sequence/example level, with labeled-position weighting inside each sequence for top-1, top-k, NLL, and ECE.
- Main comparison tables average across ambiguity cells after within-cell resampling.
- Seed summary tables report mean, standard deviation, and bootstrap intervals over seed-level point estimates.
- Bootstrap settings are config-driven through `evaluation.bootstrap_trials`, `evaluation.bootstrap_confidence_level`, and `evaluation.bootstrap_seed`.

## Reporting Format

Every run should produce:

- validated config snapshot
- metrics JSON
- example-level ranking JSON
- posterior model JSON
- Markdown evidence report
- ambiguity-sweep plots for symbol and family behavior
- confidence-interval tables for the main comparison and ambiguity sweep
- seed summary tables when a seed sweep is configured
- results and limitations markdown drafts grounded in measured outputs
- dataset integration summary files for manifest-backed real-data runs

Reports must include:

- explicit caveats for heuristic modules
- label coverage and split coverage
- null-control comparisons
- whether the correct family appeared in top-1 and top-k when labels exist
- whether uncertainty changed the best-ranked family
- whether uncertainty improved or worsened symbol recovery and calibration at each ambiguity level
- whether calibration helped, hurt, or was neutral under each ambiguity level
- whether the observed effect persists across seeds or is driven by a single run

For external real-data validation, the preferred execution path is the real-data paper-pack runner. The current repository-backed external run uses Omniglot:

```bash
python3 scripts/run_real_manifest_paper_pack.py --config configs/experiments/omniglot_paper_pack.yaml --paper-dir paper
```

## Anti-Hallucination Rules

- A fluent-looking or structured-looking output is not sufficient evidence.
- Every family score should be interpreted relative to relevant nulls.
- Unsupported cases should remain unresolved rather than forced into a confident narrative.
- Missing labels should reduce what is claimed, not be silently backfilled.
