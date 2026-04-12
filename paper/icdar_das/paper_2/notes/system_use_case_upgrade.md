# Upgraded System Use Case

## User

The primary user is an archivist, historian, or document-analysis operator reviewing uncertain grouped word crops from a historical collection.

## Workflow

The workflow is:

1. the system receives a grouped crop
2. the recognizer produces a confusion network rather than a single collapsed transcript
3. the support-aware selector recommends `rescue_first` or `shortlist_first`
4. the operator or workflow specifies a review budget such as `k=2`, `k=3`, or `k=5`
5. the controller decides whether to preserve, prune, or defer
6. the system returns a shortlist for verification, retrieval, or downstream ranking
7. the operator confirms, rejects, or follows an explicit escalation path

## Where the adaptive controller sits

The adaptive controller sits between posterior construction and grouped/downstream decoding:

- after symbol-level uncertainty is available
- before candidate pruning or grouped assembly is finalized

That location is important because it lets the system preserve rescue when needed, prune when safe, and defer when the requested budget is too small for the current case.

## Why it is useful in a real document-analysis system

A static decoder forces the same behavior on every grouped example. That is a poor fit for archival data, where some crops are diffuse and fragile while others are compact and low risk.

The upgraded controller helps because it can:

- expose a rescue-preserving mode for ambiguous archival crops
- expose a shortlist-first mode for compact verification cases
- recommend the more suitable mode automatically from support features and the active review budget
- expose budget-aware defer behavior when the requested shortlist is unsafe
- improve the chance that the correct answer is already visible inside a shortlist of size 2, 3, or 5
- make the review-effort tradeoff measurable

## Why grouped top-k and downstream tradeoffs matter operationally

For a human-facing archive workflow, the best system is not always the one with the highest exact-match number. It is often the one that:

- keeps the correct answer in the shortlist
- avoids overwhelming the operator with noisy candidates
- does not over-prune the difficult cases
- knows when to defer instead of pretending a tiny shortlist is safe

That is why grouped top-k rescue, candidate-set size, downstream exact recovery, and defer behavior all matter together in `paper_2`.

The newer practical-utility layer strengthens that story further by adding:

- review efficiency
- effort-adjusted utility under explicit review budgets
- fragile-case utility
- defer rate when the requested budget is unsafe

The selector results strengthen the workflow claim further:

- the selector stays close to the best profile in hindsight for effort-adjusted utility (mean regret `0.011`)
- it routes Historical Newspapers `cluster_distance` cases mostly toward `shortlist_first`
- it routes tight-budget ScaDS.AI `cluster_distance` cases mostly toward `rescue_first`

Those metrics and routing behaviors are closer to what a real operator experiences than exact-match alone.
