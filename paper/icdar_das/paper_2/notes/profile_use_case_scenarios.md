# Profile-Aware Use-Case Scenarios

## Scenario 1: Archivist reviewing fragile historical material

User:

- archivist, curator, or historian working with uncertain grouped word crops from difficult manuscript or newspaper material

Workflow:

1. the system extracts a grouped crop
2. the posterior/confusion network preserves local uncertainty
3. the selector or operator chooses `rescue_first`
4. the workflow allows a larger budget such as `k=5`
5. the system preserves more alternatives and allows wider raw beams where pruning is risky
6. if the budget is too tight for the case, the system can defer rather than over-prune
7. the archivist reviews a longer shortlist or an escalated fragile case

Why this mode is preferable:

- missing the correct reading is costly
- rescue matters more than aggressive pruning
- grouped top-k and downstream-sensitive behavior matter more than absolute shortlist compactness

Best success metrics:

- grouped top-k recovery
- downstream exact or partial recovery on fragile cases
- fragile-case recall@budget
- avoidance of rescue destruction under pruning

## Scenario 2: Verification operator reviewing many crops quickly

User:

- transcription operator or digitization workflow reviewer handling a high volume of grouped word crops

Workflow:

1. the system extracts a grouped crop
2. the posterior/confusion network is built
3. the selector or operator chooses `shortlist_first`
4. the workflow requests a tight budget such as `k=2` or `k=3`
5. the system uses more aggressive shortlist compaction when support looks favorable
6. if the case is too fragile for that budget, the system explicitly defers instead of forcing unsafe pruning
7. the operator reviews a compact shortlist for routine cases and escalated fragile cases separately

Why this mode is preferable:

- review speed matters
- shortlist quality under small budgets matters more than preserving every alternative
- compact verification lists reduce operator burden

Best success metrics:

- recall@2
- recall@3
- effort-adjusted utility
- review-load proxy
- defer rate

## Why this is a document analysis system contribution

These scenarios make the adaptive controller a real system component:

- it sits inside a practical grouped-recognition workflow
- it lets the workflow choose between two measurable operating modes
- it can recommend the better operating mode automatically under the current review budget
- it connects uncertainty control directly to archive and verification needs

The contribution is therefore not only better decoding, but better uncertainty control for real document-analysis use.

The new utility layer makes this more concrete:

- `shortlist_first` is the better verification profile because it raises review efficiency while keeping shortlists small
- `rescue_first` is the better fragile-material profile because it protects grouped rescue and preserves the stronger rescue-rich ScaDS.AI operating point
- the budget layer makes the workflow more realistic because tight budgets can trigger defer, while `k=5` often recovers much stronger effort-adjusted utility
- the selector makes the workflow more practical because it gets close to the right profile choice without requiring the operator to guess the regime in advance
