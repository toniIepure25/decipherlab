# Paper 2 Workspace

This directory holds the more ambitious follow-up paper track for DAS 2026.

## Goal

`paper_2` is a method/system paper built on the findings from `paper_1`. Its current method contribution is a support-aware interactive verification system for low-resource historical/grouped recognition, with explicit operating profiles and a lightweight profile selector.

## Current status

- rule-based, learned-gate, constrained-gate, profile-aware, and profile-selector policies implemented in the main pipeline
- real grouped and real downstream evaluation packs executed
- shortlist-utility, practical-utility, profile-aware, selector, fragile-case, and regret evaluation layers executed
- practical use-case and system-identity notes drafted
- bootstrap submission package created
- go/no-go audit still rates this track as riskier than `paper_1`, but materially stronger than the earlier single-policy paper_2 draft

## Key artifacts

- `plan.md`: concept note and success criteria
- `experiments/evaluation_plan.md`: evaluation bed and metrics
- `experiments/adaptive_decoder_summary.csv`: main current method summary
- `experiments/adaptive_decoder_summary.md`: readable interpretation of current results
- `experiments/profile_aware_summary.csv`: explicit rescue-first vs shortlist-first comparison
- `experiments/profile_aware_summary.md`: profile-aware operating-mode interpretation
- `experiments/profile_aware_plot.png`: compact profile tradeoff figure
- `experiments/profile_selector_summary.csv`: lightweight selector summary against the profile family
- `experiments/profile_selector_summary.md`: readable selector/oracle interpretation
- `experiments/profile_selector_plot.png`: selector operating-point and oracle-gap figure
- `experiments/practical_utility_summary.csv`: review-efficiency and fragile-case utility summary
- `experiments/practical_utility_summary.md`: readable operator-facing utility interpretation
- `experiments/practical_utility_plot.png`: shortlist efficiency tradeoff figure
- `experiments/operator_effort_dominance.csv`: selector effort comparison against profiles and fixed strategies
- `experiments/operator_effort_dominance.md`: readable operator-effort interpretation
- `experiments/fragile_case_selector_summary.csv`: selector behavior on fragile cases
- `experiments/fragile_case_selector_summary.md`: readable fragile-case selector interpretation
- `experiments/profile_regret_summary.csv`: regret to best profile and gap to best fixed policy
- `experiments/profile_regret_summary.md`: oracle-gap interpretation
- `experiments/operating_point_summary.md`: compact tradeoff summary for rule vs learned control
- `notes/profile_aware_system_design.md`: operating-profile design note
- `notes/profile_selector_design.md`: lightweight selector design note
- `notes/profile_use_case_scenarios.md`: archive and operator workflow scenarios
- `notes/learned_gate_interpretability.md`: feature-driven interpretation of the learned policy
- `notes/practical_use_case.md`: original operational DAS framing
- `notes/failure_taxonomy_for_paper2.md`: failure categories targeted by the controller
- `notes/go_no_go_audit.md`: current viability audit
- `submission_package/`: early paper bootstrap

## Current honest takeaway

The strongest current `paper_2` story is no longer that one adaptive policy wins everywhere. It is that support-aware uncertainty control exposes meaningful operating profiles for real archival workflows, and that a lightweight selector can route cases close to the right profile under real review budgets. On average, the selector reaches mean effort-adjusted utility `0.452`, mean grouped top-k delta `0.162`, and mean effort regret `0.011` to the best profile in hindsight. That makes `paper_2` more memorable and more DAS-native than before, but it is still a higher-risk track than `paper_1`.
