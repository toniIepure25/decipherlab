# Real Grouped Replication Summary

## Historical Newspapers Gold vs ScaDS.AI

- `historical_newspapers_real_grouped_gold` / `calibrated_classifier`: uncertainty exact `-0.056`, uncertainty top-k `0.056`, symbol top-k `0.042`, conformal exact `0.056`, peak ambiguity `0.000`.
- `historical_newspapers_real_grouped_gold` / `cluster_distance`: uncertainty exact `-0.125`, uncertainty top-k `0.056`, symbol top-k `0.084`, conformal exact `0.222`, peak ambiguity `0.450`.
- `scadsai_real_grouped` / `calibrated_classifier`: uncertainty exact `-0.042`, uncertainty top-k `0.250`, symbol top-k `0.264`, conformal exact `0.000`, peak ambiguity `0.300`.
- `scadsai_real_grouped` / `cluster_distance`: uncertainty exact `0.111`, uncertainty top-k `0.361`, symbol top-k `0.342`, conformal exact `-0.042`, peak ambiguity `0.000`.

## Interpretation

- This pack compares the strongest current Historical Newspapers grouped run against the second real grouped corpus using the unchanged grouped decoder family.
- Grouped top-k rescue is the clearest replicated signal across both corpora.
- Conformal exact-match gains are mixed rather than cleanly replicated across both corpora.
