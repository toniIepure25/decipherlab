# Real Data Trust Hardening

## Goal

Strengthen reviewer trust in the real grouped benchmarks without expanding the project into a new annotation campaign.

## Chosen Strategy

The strongest feasible option in this environment was a stricter adjudication and audit package for Historical Newspapers rather than a new independent blind annotation study.

Why:

- the Historical Newspapers grouped benchmark is central to the real-data story
- the evaluated `test` split is small enough to review completely
- repeating a controlled visual review and tracking disagreement is feasible
- a truly independent multi-annotator campaign is not feasible in-session

## Hardening Chain

The benchmark now has four explicit trust layers:

1. OCR-derived grouped labels
2. direct visual audit over the full evaluated split
3. gold-style two-pass adjudicated labels
4. grouped-metric stability checks after each label tier

## What The Hardening Shows

From [real_data_trust_summary.md](/home/tonystark/Desktop/decipher/outputs/real_data_trust_summary.md):

- `126` audited tokens across `30` grouped sequences
- only `2` corrected tokens
- pass-A / pass-B agreement `1.000`
- OCR-to-gold token error rate `0.016`
- OCR-to-gold sequence error rate `0.033`
- zero drift in the grouped exact-match conformal deltas after the audit and after the gold-style pass

## Error Taxonomy

The observed corrections are narrow OCR substitutions:

- `den -> der`
- `der -> des`

There is no evidence here of broad segmentation failure, label collapse, or systematic grouped-sequence corruption.

## Reviewer-Facing Value

This is still not an independent multi-annotator gold campaign, and the paper should keep saying so.

But it is substantially stronger than an OCR-derived benchmark:

- the full evaluated split was visually checked
- disagreements were quantified
- the error types were narrow
- the main grouped metrics were unchanged after label strengthening

## Honest Boundary

The strongest wording supported by this hardening is:

- the Historical Newspapers result is trustworthy enough for a bounded grouped-recognition claim
- it is not yet the basis for a gold-token manuscript claim
