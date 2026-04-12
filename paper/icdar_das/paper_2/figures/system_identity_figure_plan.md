# System Identity Figure Plan

## Purpose

This figure should make `paper_2` look like a real interactive document-analysis system paper rather than an analysis-only paper.

## Intended message

The figure should communicate one idea quickly:

- uncertainty is first represented explicitly
- support features and the review budget are read together
- a lightweight selector chooses the operating profile that best matches the current regime
- the controller then preserve/prune/defer routes the case into an archival verification workflow

## Figure structure

Recommended pipeline:

1. grouped word crop
2. posterior / confusion network
3. support feature extractor
4. support + budget features
5. lightweight profile selector
   - recommend `rescue_first`
   - recommend `shortlist_first`
6. adaptive control action
   - preserve
   - prune
   - defer
7. grouped shortlist / candidate ranking
8. downstream verification / retrieval / escalation
9. operator-facing outcome

## Why this helps the paper

This figure gives `paper_2` a clear system identity:

- it situates the selector and controller inside a real workflow
- it explains why grouped top-k, shortlist utility, regret, and defer matter together
- it makes review-budget control visible as the central practical objective
- it makes the selector feel like a small but real system contribution rather than another decoder tweak
