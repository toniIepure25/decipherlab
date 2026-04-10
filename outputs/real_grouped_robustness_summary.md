# Real Grouped Robustness Summary

- Audited tokens: `126`
- Corrected tokens: `2`
- Token error rate: `0.016`

## OCR-Derived vs Validated

- `cluster_distance`: uncertainty exact `-0.125` -> `-0.125`, uncertainty top-k `0.056` -> `0.056`, symbol top-k `0.084` -> `0.084`, conformal exact `0.222` -> `0.222`.
- `calibrated_classifier`: uncertainty exact `-0.056` -> `-0.056`, uncertainty top-k `0.056` -> `0.056`, symbol top-k `0.042` -> `0.042`, conformal exact `0.056` -> `0.056`.

## Interpretation

- This comparison isolates label corrections on the real grouped benchmark while keeping the structured-uncertainty pipeline unchanged.
- It should be read as a robustness check, not as a new corpus-level generalization claim.
