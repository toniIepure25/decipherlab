# Profile Selector Design

## What the selector chooses

The selector sits on top of the existing profile-aware system and chooses among:

- `rescue_first`
- `shortlist_first`
- direct defer recommendation when the budget is tight and the case is both fragile and selector-uncertain

The selector does not invent a new decoder family. It only routes examples toward the already implemented operating profiles.

## Which features it uses

The selector uses already available support signals plus a small amount of profile-side context:

- mean confusion entropy
- mean confusion-set size
- sequence length
- length-support count
- limited-support flag
- posterior family flag
- conformal availability
- review budget `k`
- budget-tight flag
- fragile-signal count
- rescue-profile conformal / wide-beam probabilities
- shortlist-profile conformal / wide-beam probabilities
- rescue/shortlist defer and preserve/prune candidate flags

These are all available before scoring against ground truth and remain auditable.

## What kind of selector it is

The selector is a shallow learned router:

- one logistic gate predicts whether `shortlist_first` is preferable to `rescue_first`
- the decision threshold is explicit
- direct defer is handled by a small rule overlay rather than a second opaque model

This keeps the selector lightweight and reproducible while still letting it approximate hindsight profile choice.

## Why it remains interpretable

The selector remains interpretable because:

- it is linear in a small feature set
- the coefficients can be exported directly
- the threshold policy is explicit
- the final direct-defer rule is human-readable
- every example logs the selected profile, shortlist probability, and final control action

The system is therefore inspectable at both the model level and the example level.

## Why this is more powerful than exposing profiles only

Exposing profiles alone is useful, but it assumes a human or external workflow has to choose the operating mode in advance. The selector is stronger because it can:

- adapt profile choice case-by-case
- respond to review budget automatically
- stay near the best profile for the current objective without requiring manual switching
- recommend defer when the requested budget is too tight for safe shortlist control

That makes the contribution feel more like an interactive verification system than a menu of alternative controllers.

## What exact new claim it could support

The strongest claim this design can support is:

support-aware interactive verification should not only expose multiple operating profiles; it should also route cases toward the more suitable profile under explicit review budgets and fragile-case signals, using a lightweight selector that remains close to hindsight-best profile choice for operator-facing objectives.

## Clean claim boundary

The selector does not claim:

- universal superiority over all fixed strategies
- universal superiority over both manual profiles on every objective
- semantic recovery

It claims a lightweight, auditable approximation to the right profile choice under realistic archival review constraints.
