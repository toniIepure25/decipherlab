# Learned Gate Interpretability

## Main feature drivers

The exported gate coefficients show that the learned policy is still driven by the same support signals highlighted in `paper_1`.

For downstream settings, the largest average absolute coefficients for `prefer_conformal` are:

- `mean_confusion_set_size`: `3.511`
- `mean_confusion_entropy`: `3.440`
- `conformal_available`: `1.021`

For `prefer_wide_beam`, the strongest average absolute coefficients are:

- `conformal_available`: `1.286`
- `mean_confusion_entropy`: `0.771`
- `sequence_length`: `0.717`
- `length_support_count`: `0.717`

These values come from [learned_gate_coefficients.csv](/home/tonystark/Desktop/decipher/paper/icdar_das/paper_2/experiments/learned_gate_coefficients.csv).

## What regimes it prefers

The learned gate mostly uses conformal decoding in limited or compact-support regimes, and it becomes especially conformal-heavy when the validation split repeatedly rewards pruning. That behavior is visible in the operating-point summary:

- `rule`: mean conformal rate `0.163`, mean beam width `8.917`
- `learned`: mean conformal rate `0.773`, mean beam width `4.910`

So the learned gate is not a generic “preserve more uncertainty” controller. In its current form it behaves more like a support-aware pruning controller with selective raw fallback.

## When it helps

The learned gate helps most where the rule policy was too conservative about invoking conformal decoding:

- Historical Newspapers / `cluster_distance` / real downstream:
  - rule downstream exact delta vs fixed: `-0.125`
  - learned downstream exact delta vs fixed: `+0.042`
  - learned vs raw uncertainty downstream exact gain: `+0.167`

That is the clearest evidence that the learned gate improves on the hand-written policy.

## When it hurts

The learned gate hurts most where the validation objective pushes it into over-pruning:

- ScaDS.AI / `cluster_distance` / real downstream:
  - rule downstream exact delta vs fixed: `+0.111`
  - learned downstream exact delta vs fixed: `+0.069`
  - learned vs raw uncertainty downstream exact delta: `-0.042`

In that setting the learned gate chose conformal essentially all the time, which reduced grouped top-k and downstream exact relative to the stronger raw-uncertainty operating point.

## Comparison to the original rules

The learned gate partly rediscovers the original rule logic:

- entropy and candidate-set size remain central
- support scarcity still matters
- conformal is preferred in more fragile regimes than raw uncertainty

But it also changes the policy meaningfully:

- it uses soft probabilities instead of hard thresholds
- it learns corpus- and posterior-specific gates
- it is much more willing to choose conformal than the original rule controller

That last change is why it improves the weakest Historical Newspapers setting but gives back some of the rule controller's best ScaDS.AI result.

## Honest takeaway

The learned gate is not a black box, and it is not merely reprinting the hand-written rules. It is a real policy refinement. The current evidence suggests that it improves the rule-based controller's worst failure mode, but it does not yet balance rescue and pruning well enough to dominate the strongest fixed baseline across the full real-data bed.
