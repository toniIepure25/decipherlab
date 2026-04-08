# Uncertainty Experiment Template

## Research Question

Does preserving transcription uncertainty improve downstream structural or decipherment-related inference under ambiguous observations?

## Dataset

- Manifest path:
- Unit type:
- Split policy:
- Label coverage:

## Posterior Model

- Strategy:
- Embedding:
- Calibration:
- Top-k support:

## Ambiguity Protocol

- Ambiguity levels:
- Corruption types:
- Train split corruption:
- Evaluation split corruption:

## Primary Metrics

- Symbol top-1 / top-k recovery
- Symbol calibration error
- Mean posterior entropy
- Family top-k accuracy
- Structural recovery error
- Best-family change rate between fixed and uncertainty-aware inference

## Interpretation Notes

- Report gains and regressions at each ambiguity level.
- Do not claim improvement unless the uncertainty-aware condition beats the fixed baseline on the relevant metric with a clear margin.
- Note label coverage and any partial-supervision caveats explicitly.
