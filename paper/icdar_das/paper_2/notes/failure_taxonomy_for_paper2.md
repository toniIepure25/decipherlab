# Failure Taxonomy for Paper 2

## Reused categories from the current branch

Paper 2 inherits several failure categories from the existing sequence analysis:

- local miss
- grouped assembly failure
- grouped top-k rescue without downstream exact recovery
- conformal over-pruning
- diffuse uncertainty without grouped recovery

## Extended categories for the adaptive controller

The adaptive controller adds method-specific categories:

- wrong regime choice
  - the controller chose conformal when preserving alternatives would have helped
- unnecessary wide search
  - the controller widened the beam without improving grouped or downstream recovery
- missed pruning opportunity
  - the controller stayed raw in a compact low-entropy regime where conformal would have improved exact recovery
- support absence
  - no decoder choice could help because the relevant downstream support was not present
- rule mismatch
  - grouped rescue happened, but downstream structural recovery still failed

## Failure modes the current adaptive pack reduces

Current observed reductions are limited but real:

- avoided conformal-over-pruning cases appear in the Historical Newspapers and ScaDS.AI cluster-distance runs
- the controller preserves the strongest grouped top-k signal in the diffuse ScaDS.AI cluster-distance regime

## Failure modes that still remain

The current controller still fails to reduce:

- corpus-specific downstream mismatch on Historical Newspapers cluster-distance
- selective grouped exact losses when the controller defaults to the raw branch
- universal downstream exact recovery

## Practical interpretation

This taxonomy matters because paper_2 should not claim a generic decoder win. The stronger systems claim is narrower:

- the controller can reduce specific failure types in some support regimes, but it does not eliminate the support boundary identified in paper_1.
