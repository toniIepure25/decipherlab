# System Contribution Upgrade

## Why the earlier paper_2 story was not yet strong enough

Earlier versions of `paper_2` had a real adaptive-method contribution, but the paper still risked reading like a careful decoder extension:

- one learned or constrained policy did not dominate every real regime
- the profile split was interesting, but not yet a full system contribution
- the operator-facing story was present, but explicit budget and escalation behavior were still missing

That meant the method was real, but the paper identity was not yet as strong as a memorable DAS systems paper.

## Stronger contribution introduced here

The upgraded contribution is a support-aware interactive verification system for low-resource historical/grouped document analysis.

The system now combines:

1. profile-aware control
   - `rescue_first`
   - `shortlist_first`
2. review-budget-aware behavior
   - explicit operation under `k=2`, `k=3`, and `k=5`
3. explicit preserve / prune / defer control
4. a lightweight profile selector
5. auditable decision logging

## Why this is more than a decoder policy

This is no longer only a policy that flips between raw and conformal decoding. It is a system component because it:

- conditions on operator objective
- conditions on review budget
- can route cases toward the more suitable operating profile
- can explicitly defer fragile cases instead of forcing unsafe tiny shortlists
- produces shortlists and verification behavior that map directly onto archive workflows

That shifts the contribution from "adaptive decoding" toward "interactive verification control."

## Why this is clearly DAS-relevant

The upgraded system fits DAS directly because it targets:

- historical document workflows
- human verification under uncertainty
- grouped recognition and candidate ranking
- review-budget constraints
- auditable system behavior rather than opaque model replacement

## Why this is novel within the current scope

Within the current project scope, the novelty is not a new neural architecture. The novelty is the combination of:

- support-aware control
- explicit operating profiles
- lightweight profile selection
- explicit review-budget handling
- explicit preserve / prune / defer actions
- regret-style analysis against hindsight profile choice

That makes the paper more distinct than "paper_1 plus another adaptive ablation" while staying lightweight and auditable.

## Clean claim boundary

The upgraded contribution does not claim:

- one universal best policy
- universal downstream improvement
- semantic recovery

It claims that uncertainty control for historical document verification should be explicit about operator objective and review budget, and that lightweight, interpretable logic can both expose the right operating profiles and route cases close to the right profile choice in practice.
