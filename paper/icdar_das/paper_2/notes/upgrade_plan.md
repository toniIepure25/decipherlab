# Paper 2 Upgrade Plan

## Exact weakness of the current paper_2 track

The original single-policy learned-gate version made `paper_2` a real method paper, but it still had one central weakness:

- it improved some fragile downstream settings, but it did so by becoming too pruning-heavy
- that made the method harder to trust as a document-analysis system component
- it still did not produce a sufficiently clean win over the strongest fixed baseline

So the core weakness was not missing infrastructure. It was the assumption that one adaptive policy should serve every workflow.

## Highest-leverage way to improve it

The most efficient upgrade is to keep the controller small and interpretable, but expose explicit operating profiles:

- `shortlist-first` for compact review utility
- `rescue-first` for fragile grouped cases
- keep both profiles tied to the same auditable support features and decoder actions

This is higher leverage than adding another decoder because it turns a split empirical result into a practical system feature.

## What would make paper_2 clearly stronger than paper_1

`paper_2` would become clearly stronger than `paper_1` if it delivered one of these outcomes cleanly:

1. a replicated real-data improvement over the strongest fixed baseline on grouped or downstream recovery
2. a clearly superior practical operating point:
   - explicit rescue-first and shortlist-first modes
   - lower review burden in one mode and stronger rescue in the other
   - a clear system-level workflow advantage for human verification

The second path is more realistic for this project than a universal accuracy win.

## What would make paper_2 still too weak

`paper_2` is still too weak if:

- the upgraded controller merely matches the rule baseline without a meaningful efficiency or workflow advantage
- gains remain explainable only as parameter tuning
- the practical system story is weaker than the bounded explanatory strength already achieved in `paper_1`

## Decision criteria for whether paper_2 should become the main DAS submission

`paper_2` should only replace `paper_1` as the main DAS submission if the upgraded controller satisfies all three conditions:

1. it has a clear method identity beyond analysis
2. it shows either a real-data win or a clearly better operating point
3. the resulting paper reads as a stronger DAS system paper than the safe explanatory paper

If those conditions are only partially met, `paper_1` should stay the primary submission and `paper_2` should remain the higher-risk follow-up.
