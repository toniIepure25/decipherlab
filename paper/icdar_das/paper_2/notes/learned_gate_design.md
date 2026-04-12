# Learned Gate Design

## Why the rule-based controller was not strong enough

The original rule-based controller established that support-aware control is implementable and useful, but it remained too rigid:

- it hard-coded a small set of thresholds
- it did not adapt those thresholds to corpus- and posterior-specific support patterns
- it matched or selectively improved fixed baselines, but did not produce a cleaner method win

That made the systems idea credible, but not yet strong enough for a method paper centered on adaptive decoding.

## What the learned gate decides

The upgraded controller replaces the fixed thresholds with two tiny learned policy heads:

1. `prefer_conformal`
   - predicts whether the example should use `conformal_beam` instead of raw `uncertainty_beam`
2. `prefer_wide_beam`
   - predicts whether the raw uncertainty branch should widen the beam rather than keep the default width

At inference time the controller therefore chooses one of three lightweight actions:

- conformal beam with the compact beam width
- raw uncertainty beam with the default beam width
- raw uncertainty beam with the wider beam width

## Support features used by the gate

The learned gate uses only features already motivated by `paper_1`:

- mean confusion entropy
- mean confusion-set size
- sequence length
- train-support count for the sequence length
- limited-support indicator
- posterior-family indicator (`is_calibrated`)
- conformal-availability indicator

These features are available before the final decoding choice, so the controller does not leak test labels into inference.

## Training target

The gate is trained on validation examples only. For each validation example, the pipeline evaluates the small action set above and scores each action with a utility function that combines:

- grouped exact and grouped top-k recovery
- downstream exact and downstream top-k recovery when downstream evaluation is available
- token accuracy and CER-like penalties
- a mild prediction-set-size penalty to discourage useless candidate inflation

The learned gate then predicts which action had the highest validation utility.

## Why the gate remains interpretable

The learned controller stays auditable because it is intentionally small:

- it uses logistic gates rather than a large opaque model
- feature weights are exported for every dataset/posterior setting
- per-example conformal and wide-beam probabilities are logged
- every adaptive decision is written to the run outputs with its selected action and gate scores

This means the learned gate can be inspected as a policy, not just as an accuracy number.

## Empirical question it is meant to answer

The learned-gate upgrade asks a narrow method question:

- can the support-aware propagation findings from `paper_1` be turned into a lightweight learned policy that improves on the hand-written controller and produces a clearer real grouped or real downstream win than fixed decoder selection alone?

The current answer is selective rather than universal, which is precisely why the gate needs explicit operating-point and interpretability analysis.
