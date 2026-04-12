# Preliminary Contributions

1. A lightweight support-aware interactive verification system that exposes `rescue-first` and `shortlist-first` operating profiles for grouped historical-document workflows.
2. A profile selector that routes examples toward the more suitable operating profile under explicit review budgets and support regimes without adding a new opaque decoder family.
3. Budget-aware preserve / prune / defer logic that makes the controller behave like a real verification component rather than a decoder-only heuristic.
4. A practical utility evaluation layer for human-in-the-loop archival verification, based on review-budget recall, shortlist utility, review efficiency, fragile-case utility, and operator-effort proxies.
5. An oracle/regret analysis showing how close the selector gets to the best profile in hindsight for budgeted verification objectives, and how it nearly matches `shortlist-first` on effort utility while moving closer to `rescue-first` on grouped rescue.
6. An auditable systems analysis that logs profile selection, decoder choice, beam width, budget, defer decisions, and support indicators for every example.
7. A practical document-analysis framing for archival transcription assistance and candidate verification under uncertainty.
