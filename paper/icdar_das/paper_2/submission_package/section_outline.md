# Section Outline

1. Introduction
   - one decoding policy is not realistic for every archival workflow
   - support-aware uncertainty control should expose profiles and select among them under review budgets
2. Related Work
   - uncertainty-aware recognition
   - conformal and risk-aware prediction
   - interactive and archive-oriented document analysis systems
3. Support-Aware Interactive Verification System
   - support features
   - available preserve / prune / defer actions
   - `rescue-first` profile
   - `shortlist-first` profile
   - lightweight profile selector
   - budget-aware routing and direct-defer recommendation
   - auditable implementation
4. Evaluation Bed
   - two real grouped/token-aligned corpora
   - redesigned real downstream task
   - shortlist utility under review budgets
   - review efficiency, fragile-case utility, and oracle regret
5. Results
   - grouped and downstream profile comparison
   - shortlist utility and review-efficiency comparison
   - profile-selector gap to best profile and gap to best fixed policy
   - fragile-case routing and defer behavior
   - failure cases and selector rationale
6. Practical DAS Workflows
   - archivist / historian rescue-first scenario
   - high-throughput verification shortlist-first scenario
   - selector-assisted profile recommendation under explicit review budgets
7. Limitations
   - no universal best policy
   - selector remains close to oracle rather than universally dominant
8. Conclusion
