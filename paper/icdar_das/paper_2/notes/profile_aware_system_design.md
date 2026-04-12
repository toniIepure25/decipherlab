# Profile-Aware System Design

## Core idea

`paper_2` no longer assumes that one adaptive policy should be globally optimal for every historical-document workflow. Instead, the system exposes explicit operating profiles that align uncertainty control with real operator goals, and it lets those profiles behave differently under explicit review budgets.

## Operating profiles

### 1. `rescue_first`

What it optimizes:

- grouped top-k rescue
- safer downstream-sensitive behavior in fragile regimes
- preserving alternatives when over-pruning would remove useful evidence

How it behaves:

- reuses the constrained gate
- is more willing to keep raw uncertainty and wider beams
- only allows conformal pruning when grouped-rescue guardrails are satisfied
- under tight budgets, can defer fragile cases rather than forcing an unsafe tiny shortlist

Best fit:

- fragile historical crops
- archive scenarios where missing the truth is more costly than reviewing a slightly longer shortlist

### 2. `shortlist_first`

What it optimizes:

- compact operator-facing shortlists
- shortlist utility under small review budgets
- pruning when support is good enough that the shortlist can be made smaller safely

How it behaves:

- reuses the learned gate
- is more willing to choose conformal pruning
- prefers smaller candidate sets and lower review burden
- under tight budgets, can still defer when support is too weak for safe shortlist compaction

Best fit:

- verification-heavy workflows
- archive or library pipelines where operators review many grouped crops quickly

## Review-budget layer

The profiles now interact with explicit budgets:

- `k=2`
- `k=3`
- `k=5`

This matters because the system should behave differently when the operator can inspect only two candidates than when the workflow allows five.

The empirical pattern is clear:

- tight budgets (`k=2` and `k=3`) activate defer on `14.8%` of profiled cases on average
- `k=5` removes defer on average and gives the strongest overall effort-adjusted operating point for `shortlist_first`

## Why there is no balanced mode by default

We intentionally keep the system to two modes for now.

- the evidence already supports two clean operational objectives
- adding a third "balanced" mode would blur the paper's main contribution unless it produced a distinct and useful operating point
- the current results do not justify that extra complexity yet

## Why the profiles differ

The current real-data evidence shows a real split rather than noise:

- `shortlist_first` gives the strongest average shortlist utility (`0.574`) and the smallest average set size (`1.689`)
- `rescue_first` gives the stronger average grouped top-k delta (`0.167`)
- `shortlist_first` at `k=5` gives the strongest mean effort-adjusted utility (`0.482`)
- `shortlist_first` also has the smaller mean regret to the best profile for shortlist utility (`0.007`)
- `rescue_first` has the smaller mean regret to the best profile for grouped rescue (`0.010`)

So the profiles are not arbitrary user-interface labels. They correspond to real differences in the rescue-versus-pruning frontier.

## Why explicit profiles are more realistic

Historical document systems are not used under one universal objective:

- an archivist checking rare, difficult material may want higher rescue even if the shortlist is longer
- a verification operator handling many routine crops may prefer a smaller shortlist with lower review cost

Treating those as the same optimization target is less realistic than exposing explicit operating modes.

Treating all review budgets as equivalent is also unrealistic. The same crop can justify a compact shortlist under `k=5` but a defer decision under `k=2`.

## Why this is DAS-relevant

This profile-aware design fits DAS directly because it is:

- a document analysis system component, not only an analysis result
- useful for archives and historical collections
- naturally human-in-the-loop
- benchmarkable with real grouped and downstream metrics
- auditable because the profiles select between already interpretable adaptive policies
- naturally suited to interactive verification under explicit review budgets

## Clean scientific claim boundary

The system does not claim that profile choice solves all downstream failures. It claims that support-aware uncertainty control should expose explicit operating modes and review-budget-aware control because different archival workflows value rescue and shortlist compactness differently.
