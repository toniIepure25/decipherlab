# Real Grouped Strengthened Summary

## Original vs Audited vs Gold-Style

- `cluster_distance`: uncertainty exact `-0.125` / `-0.125` / `-0.125`, uncertainty top-k `0.056` / `0.056` / `0.056`, symbol top-k `0.084` / `0.084` / `0.084`, conformal exact `0.222` / `0.222` / `0.222`.
- `calibrated_classifier`: uncertainty exact `-0.056` / `-0.056` / `-0.056`, uncertainty top-k `0.056` / `0.056` / `0.056`, symbol top-k `0.042` / `0.042` / `0.042`, conformal exact `0.056` / `0.056` / `0.056`.

## Interpretation

- The strengthened pack compares the original OCR-derived run, the first audited run, and the gold-style adjudicated subset without changing the decoder family.
- Stable numbers across the three versions increase trust in the current real grouped result, but they do not substitute for replication on a second real grouped corpus.
