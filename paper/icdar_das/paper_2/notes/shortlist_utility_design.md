# Shortlist Utility Design

## Practical utility target

The adaptive controller is meant to optimize shortlist utility under small operator review budgets:

- keep the correct grouped transcript inside a shortlist of size 2, 3, or 5
- avoid unnecessary candidate inflation
- preserve rescue-rich cases where aggressive pruning would remove the only useful alternative

This is the practical system objective that matters most for a human-in-the-loop archival workflow.

## Why this is more DAS-relevant than plain exact recovery

In historical document pipelines, operators usually review a few candidates rather than blindly trusting a top-1 prediction. That means a system can still be highly useful when:

- exact top-1 recovery is mixed
- the correct answer is consistently retained in a very small shortlist
- the shortlist stays compact enough to review quickly

That makes shortlist utility more aligned with document-analysis systems practice than a pure exact-match objective.

## Connection to archives and historical documents

Historical grouped word crops are difficult because of:

- handwriting variability
- degradation and scan artifacts
- sparse train support
- grouping noise

For those cases, the right behavior is often:

- preserve a few plausible readings
- rank them well enough for fast verification
- avoid overwhelming the archivist with long noisy candidate lists

## Metrics representing shortlist utility

The shortlist utility layer uses:

- recall@2
- recall@3
- recall@5
- a weighted shortlist utility score:
  - `0.5 * recall@2 + 0.3 * recall@3 + 0.2 * recall@5`
- prediction-set average size as an operator-burden proxy

These metrics let us measure how useful the adaptive system is at the exact point where a human would review its output.

## Current lesson for paper_2

The shortlist utility layer reveals a stronger systems story than exact recovery alone:

- the unconstrained learned gate gives the strongest average shortlist utility with the most compact candidate burden
- the constrained controller is safer in rescue-rich settings
- no single policy dominates every regime

That tradeoff is exactly what makes `paper_2` feel like a document-analysis systems paper rather than just another adaptive-decoder benchmark.
