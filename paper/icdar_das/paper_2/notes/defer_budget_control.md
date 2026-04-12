# Defer and Budget Control Note

## What new decision options were added

The profile-aware controller now exposes three explicit control actions:

- `preserve`
- `prune`
- `defer`

It also accepts an explicit review budget:

- `k=2`
- `k=3`
- `k=5`

## Why this matters in historical verification workflows

Archive workflows are budgeted in practice. Some operators can inspect only two or three candidates before moving on, while other workflows allow a longer shortlist.

Without explicit budget handling, the system either:

- over-prunes fragile cases to satisfy a small budget, or
- keeps too many candidates in routine cases and wastes operator effort

The defer action makes that tradeoff honest. Instead of forcing a brittle shortlist on hard cases, the system can admit that the requested budget is too small and route the case to manual verification.

## How this interacts with `rescue_first` and `shortlist_first`

`rescue_first`

- is more willing to keep raw uncertainty
- prioritizes rescue preservation
- can defer fragile cases under tight budgets rather than forcing unsafe pruning

`shortlist_first`

- is more willing to prune when support is favorable
- prioritizes compact review
- can also defer when the requested budget is too small for safe shortlist compaction

## What changed empirically

The new layer produces a real budget-sensitive behavior:

- tight budgets (`k=2` and `k=3`) activate defer on `14.8%` of profiled cases on average
- `k=5` removes defer on average and gives the strongest overall effort-adjusted operating point for `shortlist_first`
- `shortlist_first` at `k=5` gives the strongest mean effort-adjusted utility (`0.482`) and the closest mean effort gap to the best fixed policy (`-0.036`)
- `rescue_first` preserves the stronger grouped operating point (`0.167` mean grouped top-k delta vs fixed)
- the selector now sits on top of those profiles and chooses between them with mean effort-adjusted regret `0.011` to the best profile in hindsight

The fragile-case layer shows why defer matters:

- on fragile Historical Newspapers `cluster_distance` downstream cases, both profiles defer completely at `k=2` and `k=3`
- at `k=5`, defer disappears and fragile-case effort-adjusted utility rises sharply
- on fragile ScaDS.AI `cluster_distance` cases, defer remains active at `k=2` and `k=3`, then disappears at `k=5`

## Why this strengthens the system story

The controller now behaves more like a real interactive verification system:

- it knows the review budget
- it adapts shortlist behavior to that budget
- it escalates cases that do not fit the requested budget safely
- it can route cases toward the better operating profile without asking the operator to guess the regime in advance

That is a stronger contribution than a decoder-only framing because it turns uncertainty control into an operator-facing system policy.
