# Evaluation Plan for Paper 2

## Core evaluation bed

Paper 2 reuses the strongest current real-data evaluation bed:

- Historical Newspapers grouped words
- ScaDS.AI grouped words
- redesigned real downstream task on both corpora

The synthetic-from-real tasks remain secondary. They can be used later only if they help explain controller behavior, not as the main evidence.

## Main baselines

The adaptive controller is evaluated against:

- `fixed_greedy`
- `uncertainty_beam`
- `conformal_beam`
- `uncertainty_trigram_beam`
- `conformal_trigram_beam`
- `uncertainty_crf_viterbi`
- `conformal_crf_viterbi`

The controller itself is:

- `adaptive_support_beam`

## Main metrics

The main paper_2 metrics are:

- grouped top-k recovery
- grouped exact recovery
- shortlist recall@2, @3, and @5
- shortlist utility under small review budgets
- real downstream exact recovery
- real downstream token accuracy
- prediction-set size as a proxy for candidate efficiency
- adaptive conformal selection rate
- adaptive mean beam width

## Method-facing questions

The method evaluation asks:

1. does the controller improve grouped recovery relative to fixed baselines?
2. does it improve real downstream recovery in the redesigned task?
3. does it match or beat the strongest fixed bigram baseline?
4. when it does not win, does it at least show a clear efficiency or failure-reduction pattern?
5. does it improve operator-facing shortlist quality under small review budgets?

## Current pack

The reproducible current pack is:

- script: `scripts/run_support_aware_adaptive_pack.py`
- summary: `paper/icdar_das/paper_2/experiments/adaptive_decoder_summary.csv`
- readable summary: `paper/icdar_das/paper_2/experiments/adaptive_decoder_summary.md`
- plot: `paper/icdar_das/paper_2/experiments/adaptive_decoder_plot.png`
- shortlist utility summary: `paper/icdar_das/paper_2/experiments/shortlist_utility_summary.csv`
- shortlist utility plot: `paper/icdar_das/paper_2/experiments/shortlist_utility_plot.png`

## Success threshold for paper readiness

For paper_2 to become clearly submission-worthy, the adaptive controller should satisfy at least one of:

- consistent grouped improvement with no meaningful downstream penalty
- selective but real downstream improvement over the fixed bigram baselines with a clear practical rationale
- comparable accuracy with a cleaner efficiency / failure-reduction story

If none of those hold, paper_2 should remain a development track rather than the primary submission.
