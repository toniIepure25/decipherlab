# Support-Aware Adaptive Decoder

## Overview

This note defines the method contribution for the more ambitious `paper_2` track. The controller does not add a new decoder family. Instead, it uses support features identified in the propagation analysis to choose between existing auditable decoding options at inference time.

The current version is no longer only an adaptive decoder. It is a support-aware interactive verification component with:

- explicit operating profiles
- a lightweight profile selector
- explicit review budgets
- explicit preserve / prune / defer behavior

## Method progression

The method now has six stages of increasing sophistication:

1. a hand-written rule controller
2. a shallow learned gate
3. a constrained learned gate with rescue-preserving guardrails
4. a profile-aware interactive controller exposing explicit operating modes
5. a budget-aware and defer-aware verification controller
6. a profile selector that routes cases toward the more suitable operating profile

The current `paper_2` method is the sixth version.

## What support features are used

The current controller uses only features available before final decoding:

- mean confusion entropy
- mean confusion-set size
- sequence length
- posterior family (`cluster_distance` or `calibrated_classifier`)
- length-support count from the train split for the current sequence length
- conformal availability

These are all already present in, or directly derivable from, the current sequence branch.

## What decisions are made

The controller makes four lightweight decisions:

1. decoder choice
   - `uncertainty_beam`
   - `conformal_beam`
2. beam width
   - narrowed beam in low-entropy compact regimes
   - default beam in ordinary regimes
   - widened beam in diffuse high-entropy regimes
3. control action
   - `preserve`
   - `prune`
   - `defer`
4. profile choice
   - `rescue_first`
   - `shortlist_first`

At the system level it also exposes:

- an explicit review budget
  - `k=2`
  - `k=3`
  - `k=5`

It does not add a new scoring model or a new opaque reranker.

## Practical objective

The practical system target is not only exact recovery. The controller is also designed to improve:

- shortlist recall under very small operator review budgets
- candidate usefulness per retained alternative
- review efficiency
- fragile-case behavior under explicit review budgets

It is also designed to escalate cases that do not fit a tight review budget safely.

## Policy form

The current method is a profile-aware controller built from two lightweight subpolicies plus a lightweight selector:

- `shortlist_first` reuses the learned gate
- `rescue_first` reuses the constrained gate
- a logistic selector routes examples toward the more suitable profile under the current budget
- both use the same support features and the same decoder actions
- profile selection remains explicit and operator-facing rather than hidden inside one universal policy

The budget layer then shapes behavior further:

- tighter budgets encourage more compact shortlist behavior when support is favorable
- fragile cases can trigger `defer` rather than unsafe pruning
- wider budgets allow more preserve-oriented behavior without violating workflow constraints

The selector then turns those profile choices into a system capability rather than a menu:

- it recommends `shortlist_first` when compact review is worth the rescue tradeoff
- it recommends `rescue_first` when fragile support regimes make pruning risky
- it can trigger a direct defer recommendation in tight-budget ambiguous cases, although most defer behavior still happens inside the profile controllers

## Why this is justified by paper_1

`paper_1` established three directly relevant findings:

1. grouped rescue is real and replicated
2. downstream propagation is support-gated
3. raw uncertainty and conformal help in different support regimes

The adaptive controller operationalizes those findings instead of leaving them as analysis only. It turns regime knowledge into a concrete inference policy and verification workflow.

## Why the method stays interpretable

The controller remains auditable because:

- it uses a small explicit feature set
- it uses logistic gates rather than a deep policy model
- it selects among existing decoders instead of inventing a new hidden model
- the profile mode is explicit rather than implicitly inferred from an opaque objective
- the review budget is explicit rather than hidden in a learned reward

Every adaptive decision can be logged as:

- selector shortlist probability
- operating profile
- review budget
- selected decoder
- beam width
- control action
- decision reason
- support flags
- gate probabilities
- exported feature coefficients

The verification layer is also auditable because it uses simple ranked-list metrics:

- recall@2
- recall@3
- recall@5
- weighted shortlist utility
- review efficiency
- effort-adjusted utility
- defer rate
- fragile-case utility
- regret to the best profile in hindsight

## Failure modes it is meant to reduce

The controller is specifically meant to reduce:

- conformal over-pruning in high-entropy regimes
- under-pruned noisy candidate inflation in low-entropy limited-support regimes
- grouped rescue that is lost because one static decoder is used everywhere
- operator regret from choosing the wrong profile for the current support regime
- unnecessary wide-beam computation in compact low-risk cases
- unsafe tiny shortlists in fragile cases that should instead be escalated to human review

It is not meant to solve:

- true support absence
- universal exact real downstream recovery
- semantic interpretation
- all corpus-specific downstream failures

## Intended scientific claim

If successful, the method supports a stronger `paper_2` claim than `paper_1`:

- support-aware regime findings can be converted into a lightweight interactive verification system that exposes explicit rescue-first and shortlist-first modes under real review budgets
- those same findings can be used by a lightweight selector to route cases near the best profile in hindsight for operator-facing objectives
- the result is a practical shortlist-control and defer-aware component for archival document analysis rather than post-hoc analysis only

The current evidence supports a narrow but stronger systems claim:

- mean effort-adjusted utility `0.452`
- mean grouped top-k delta vs fixed `0.162`
- mean recall@budget `0.591`
- mean effort-adjusted regret to the best profile in hindsight `0.011`

The claim remains intentionally narrow: the system improves operator-facing control of uncertainty and gets close to objective-specific best-profile behavior, not universal accuracy.
