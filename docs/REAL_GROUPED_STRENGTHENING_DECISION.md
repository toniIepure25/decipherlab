# Real Grouped Strengthening Decision

## Options Considered

### 1. Integrate A Second Real Grouped/Token-Aligned Corpus

Expected scientific value:

- strongest option if successful, because it would test replication rather than only label quality

Feasibility in this environment:

- still weak
- the same token-alignment and access constraints remain
- likely to consume the phase without producing a finished real grouped comparison

Expected engineering cost:

- moderate to high
- dataset search, preparation, validation, and pack integration

### 2. Upgrade Historical Newspapers To A Gold-Style Evaluation Subset

Expected scientific value:

- lower than true second-corpus replication
- but directly strengthens trust in the one real grouped result we already have

Feasibility in this environment:

- high
- the grouped benchmark already exists
- the test split is small enough to review completely
- the grouped pack can be rerun unchanged

Expected engineering cost:

- low to moderate
- annotation protocol, agreement tracking, manifest materialization, and comparison scripts

## Decision

This phase chose **gold-style upgrade of the Historical Newspapers grouped benchmark**.

## Why This Was The Highest-Leverage Path

- it improves the trustworthiness of the only current real grouped result immediately
- it does not require another risky dataset integration cycle
- it preserves the current structured-uncertainty methods unchanged

## Boundary

This choice does **not** create a replicated real grouped result across corpora.

What it does create is:

- a stronger label-quality story for the current real grouped benchmark
- a more defensible bridge from OCR-derived labels to a gold-style adjudicated evaluation slice
