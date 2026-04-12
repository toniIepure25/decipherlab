# Preliminary Reviewer Risk Checklist

## Main risks

1. The method may still look like controller packaging rather than a real systems contribution.
2. Reviewers may say the paper avoids the hard problem by exposing multiple profiles instead of finding one best policy.
3. The real downstream gains remain selective.
4. The selector story could look like one more meta-heuristic on top of the profiles rather than a meaningful system design.
5. Reviewers may ask whether the selector is still too far from the strongest fixed baseline to matter.
6. Reviewers may ask whether direct defer is just a way of hiding failure.

## Best current answers

- The contribution is not a new decoder family; it is a support-aware interactive system component aligned with real archival workflows.
- The profiles correspond to measurable operating-point differences on real grouped/downstream data, not only to narrative framing.
- `shortlist-first` and `rescue-first` each dominate different practical objectives:
  - compact shortlist utility
  - rescue-preserving behavior in fragile regimes
- The selector strengthens that story by routing between the profiles rather than leaving mode choice entirely manual.
- The new practical-utility layer makes those objectives concrete:
  - `shortlist-first` gives mean shortlist utility `0.574` and mean review efficiency `0.419`
  - `rescue-first` gives mean grouped top-k delta `0.167` and stronger fragile-case protection
- The selector/regret analysis shows the system gets close to objective-specific hindsight choices:
  - selector mean effort-adjusted regret to the best profile is `0.011`
  - selector mean grouped regret to the best profile is `0.015`
  - selector mean effort-adjusted utility (`0.452`) stays close to `shortlist-first` (`0.462`) while selector mean grouped top-k delta (`0.162`) moves much closer to `rescue-first` (`0.167`)
  - `shortlist-first` and `rescue-first` remain the right anchors for their respective objectives
- The system is auditable because profile, selected decoder, beam width, and support indicators are logged for every decision.
- The selector is also auditable because it is a single logistic gate with exported coefficients and explicit thresholding.
- The paper is more DAS-native because it frames uncertainty control around operator workflows, not only around exact-match optimization.
- Direct defer is not hidden failure: it is explicit workflow control under tight budgets, and selector-level direct defer remains sparse because most escalation is already handled at the profile level.

## What still needs careful presentation

- make clear that multiple profiles plus selector routing are a realistic archival systems feature, not an excuse for weak accuracy
- foreground shortlist utility, review-budget relevance, and selector regret early
- show the ScaDS.AI rescue-rich advantage and the Historical Newspapers shortlist advantage side by side
- avoid implying that the selector beats every fixed baseline everywhere
- be explicit that the selector gets close to the right profile on average, but still trails the strongest fixed baseline on average effort-adjusted utility
