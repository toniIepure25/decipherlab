# Real Grouped Next-Step Decision

## Options Considered

### 1. Manual Validation Of Historical Newspapers Labels

Feasibility in this environment:

- high
- the grouped benchmark already exists
- the evaluation split is only `30` sequences / `126` tokens
- the vocabulary is small enough for a direct visual audit
- the existing grouped pipeline can be rerun unchanged after corrections

Expected scientific value:

- directly tests whether the current real grouped result is an OCR-label artifact
- sharpens trust in the existing benchmark without adding scope
- enables an apples-to-apples before/after comparison

### 2. Add A Second Real Grouped/Token-Aligned Corpus

Feasibility in this environment:

- lower
- still blocked by the same token-alignment and access constraints documented in the feasibility notes
- likely to require more dataset-specific preparation and a longer search loop than this phase allows

Expected scientific value:

- potentially stronger than validation if a clean second corpus existed
- but higher risk of stalling without producing a finished result in this phase

## Decision

This phase chose **manual validation of the Historical Newspapers grouped benchmark**.

## Why This Was The Highest-Leverage Move

- it strengthens the only current real grouped benchmark directly
- it produces a clean robustness test without changing the structured-uncertainty methods
- it reduces one specific external-validity concern immediately: OCR label noise

## What This Decision Can And Cannot Strengthen

It can strengthen:

- trust in the current real grouped transfer result
- confidence that the conformal exact-match finding is not just an OCR-label artifact

It cannot by itself establish:

- cross-corpus grouped replication
- gold-token manuscript claims
- real downstream family-identification claims
