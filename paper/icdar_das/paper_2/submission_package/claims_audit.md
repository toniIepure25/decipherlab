# Preliminary Claims Audit

## Claims that are currently supportable

- The adaptive controller is now a profile-aware and selector-aware system component rather than only a single adaptive policy.
- `shortlist-first` and `rescue-first` correspond to real operating-point differences on the real grouped/downstream bed, not just cosmetic mode labels.
- `shortlist-first` gives the strongest average compact-review behavior:
  - mean shortlist utility `0.574`
  - mean review efficiency `0.419`
  - mean set size `1.689`
- `rescue-first` gives the stronger rescue-preserving operating point:
  - mean grouped top-k delta vs fixed `0.167`
  - restores the strongest ScaDS.AI `cluster_distance` downstream exact delta `+0.111`
- the new selector gives a stronger compromise operating point:
  - mean effort-adjusted utility `0.452`
  - mean grouped top-k delta vs fixed `0.162`
  - mean recall@budget `0.591`
  - mean effort-adjusted regret to the best profile `0.011`
  - mean grouped regret to the best profile `0.015`
  - it nearly matches `shortlist-first` on effort utility while moving much closer to `rescue-first` on grouped rescue
- the selector shows meaningful regime specialization:
  - Historical Newspapers `cluster_distance` runs are mostly routed to `shortlist-first`
  - tight-budget ScaDS.AI `cluster_distance` runs are mostly routed to `rescue-first`
- The system is directly motivated by the propagation findings from `paper_1` and remains auditable through explicit support features, selector coefficients, and logged decisions.
- The practical-utility layer makes the method relevant to human-in-the-loop archival verification rather than only to exact-match benchmarking:
  - fragile downstream cases defer under `k=2/3` and recover utility at `k=5`
  - selector-level fragile-case effort regret to the best profile averages `0.009`

## Claims that are not yet supportable

- A single universally best adaptive policy.
- A claim that the selector beats the best profile or the strongest fixed baseline on average effort-adjusted utility.
- A clean replicated real downstream improvement across both real corpora under one profile.
- A claim that profile-aware control resolves the support boundary identified in `paper_1`.
- A claim that the selector's direct-defer recommendation is itself the main source of improvement.
- A claim that `paper_2` is unambiguously stronger than `paper_1`.

## Current central claim candidate

- Support-aware uncertainty-propagation findings can be operationalized into a lightweight interactive verification system with explicit rescue-first and shortlist-first profiles and a lightweight selector that routes cases near the right profile under real review budgets, improving operator-facing utility while preserving rescue-sensitive behavior, even though no single policy dominates every real-data regime.
