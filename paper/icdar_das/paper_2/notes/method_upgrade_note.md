# Method Upgrade Note

## What was upgraded

The first learned-gate controller in `paper_2` improved some weak downstream settings, but it often became too conformal-heavy. The first upgrade added two pieces of structure without changing the decoder family:

1. constrained training targets
2. rescue-preserving inference vetoes

That turned the controller into a support-aware constrained gate rather than a plain learned selector.

The current upgrade goes one step further: instead of insisting on one globally best adaptive policy, the system now exposes explicit operating profiles, explicit review-budget-aware control, and a lightweight profile selector:

1. `rescue_first`
   - reuse the constrained gate
   - protect grouped rescue in fragile regimes
2. `shortlist_first`
   - reuse the unconstrained learned gate
   - optimize compact shortlist utility under small review budgets
3. budget-aware defer layer
   - preserves or prunes when the requested review budget is feasible
   - defers to human review when the case is too fragile for the requested budget
4. profile selector
   - routes examples toward `rescue_first` or `shortlist_first`
   - approximates the better profile choice under the current budget and support regime

This turns the split between adaptive behaviors into an explicit system feature rather than an unresolved weakness.

## What question the upgrade answers

The method upgrade asks:

- can a lightweight adaptive controller be shaped toward a practical rescue-versus-pruning operating point instead of simply learning whichever action produced the highest validation utility?
- can the same controller improve the quality of very small operator-facing shortlists rather than only raw exact recovery?
- should one static adaptive policy be assumed at all, or should the system expose profile-specific operating points and route cases between them?
- should the system force every case into the requested shortlist budget, or should it explicitly defer fragile cases?

That is a method/system question, not just a tuning question.

## What changed empirically

The profile-aware, budget-aware, and selector-aware view makes the empirical picture cleaner:

- `shortlist_first` is the strongest compact-review profile on average:
  - mean shortlist utility: `0.574`
  - mean set size: `1.689`
  - mean review efficiency: `0.419`
  - strongest mean effort-adjusted utility at `k=5`: `0.482`
  - mean downstream exact delta vs fixed: `+0.014`

- `rescue_first` is the safer rescue-preserving profile on average:
  - mean grouped top-k delta vs fixed: `0.167`
  - mean shortlist utility: `0.553`
  - mean fragile-case budgeted recall at `k=5`: `0.607`
  - mean set size: `2.690`

- the selector now gives a stronger compromise operating point:
  - mean effort-adjusted utility: `0.452`
  - mean grouped top-k delta vs fixed: `0.162`
  - mean recall@budget: `0.591`
  - mean effort-adjusted regret to the best profile in hindsight: `0.011`
  - mean grouped regret to the best profile: `0.015`

The split is also visible in the key corpus regimes:

- Historical Newspapers with `cluster_distance` downstream:
  - `shortlist_first`: downstream exact delta vs fixed `+0.042` at `k=5`
  - both profiles defer under the tightest fragile-case budgets
- ScaDS.AI with `cluster_distance` downstream:
  - `rescue_first`: downstream exact delta vs fixed `+0.111`
  - `shortlist_first`: stronger effort-adjusted behavior and smaller candidate sets
  - selector: routes mostly toward rescue at `k=2/3`, then shifts toward shortlist at `k=5`

- Historical Newspapers with `cluster_distance`:
  - selector: routes mostly toward `shortlist_first` (`0.778` to `0.847` shortlist selection rate, depending on task and budget)
  - direct selector-level defer remains rare, because most escalation is already handled inside the selected profile

The new practical-utility layer adds a stronger systems interpretation:

- `shortlist_first` at `k=5` has the closest mean effort gap to the best fixed policy (`-0.036`)
- tight budgets (`k=2` and `k=3`) trigger defer on `14.8%` of profiled cases on average
- the regret story is stronger for operator-facing objectives:
  - `shortlist_first` has mean effort-adjusted regret `0.000` across all three budgets
  - `rescue_first` keeps mean grouped regret `0.010`
  - the selector keeps mean effort-adjusted regret to the best profile at `0.011`
  - the selector can even exceed the best fixed single profile in the calibrated Historical Newspapers regimes

## What did not improve

The upgrade still does not create a universal winner:

- `shortlist_first` remains weaker on some rescue-rich ScaDS.AI settings
- `rescue_first` remains weaker on compact-review objectives
- the selector still trails the strongest fixed baseline on average effort-adjusted utility by about `0.056`
- selector-level direct defer is not the main gain; most of the practical value still comes from profile routing and profile-level defer behavior

## Honest interpretation

The upgrade makes `paper_2` a stronger systems paper because it turns a split result into an explicit operating-profile and budget-aware verification interface, then adds a lightweight selector that gets close to hindsight-best profile choice. The strongest claim is no longer about one adaptive policy winning everywhere; it is about exposing interpretable modes, routing between them intelligently, and using honest defer behavior under real review budgets. That is more realistic and more DAS-native, but it still stops short of a universal real-data accuracy win.
