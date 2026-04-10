# Claims Map

## Central Claim

- Claim: Preserved transcription uncertainty improves real grouped top-k recovery across two token-aligned corpora, but higher-level propagation is support-gated: grouped rescue predicts downstream success only in specific support regimes, and exact real downstream gains remain selective rather than replicated.
- Manuscript location: `abstract.tex`, `sections/01_introduction.tex`, `sections/05_results.tex`, `sections/07_conclusion.tex`
- Supporting figure/table: `Figure 2`, `Table 1`
- Repo artifacts: `outputs/propagation_regime_plot.png`, `outputs/real_vs_synthetic_bridge_summary.csv`

## Major Supporting Claims

- Claim: Real grouped top-k rescue replicates across Historical Newspapers and ScaDS.AI.
- Manuscript location: `sections/05_results.tex`
- Supporting figure/table: `Figure 1`, `Appendix Table A1`
- Repo artifacts: `outputs/real_grouped_replication_plot.png`, `outputs/statistical_robustness_summary.csv`

- Claim: Historical Newspapers is trustworthy enough for a bounded grouped-recognition claim.
- Manuscript location: `sections/04_experimental_setup.tex`, `sections/06_limitations.tex`, `sections/appendix.tex`
- Supporting figure/table: `Table A2`
- Repo artifacts: `outputs/real_data_trust_summary.csv`, `docs/REAL_DATA_TRUST_HARDENING.md`

- Claim: The redesigned real downstream task reduces the prior coverage objection but still yields selective exact gains.
- Manuscript location: `sections/04_experimental_setup.tex`, `sections/05_results.tex`, `sections/appendix.tex`
- Supporting figure/table: `Table 1`, `Figure A1`
- Repo artifacts: `outputs/real_downstream_coverage_analysis.csv`, `outputs/real_downstream_redesigned_summary.csv`

- Claim: Symbol rescue is the strongest predictor of grouped rescue, and grouped rescue is necessary but not sufficient for downstream success.
- Manuscript location: `sections/03_method.tex`, `sections/05_results.tex`
- Supporting figure/table: `Figure 1`, `Appendix Table A1`
- Repo artifacts: `outputs/propagation_model_summary.csv`, `outputs/statistical_robustness_summary.csv`

- Claim: Conformal helps mainly in specific support regimes instead of universally.
- Manuscript location: `sections/05_results.tex`, `sections/06_limitations.tex`
- Supporting figure/table: `Figure 1`, `Figure A1`
- Repo artifacts: `outputs/propagation_regime_summary.csv`, `outputs/propagation_case_studies.md`, `outputs/real_downstream_redesigned_summary.csv`

- Claim: Historical Newspapers is strong enough for a bounded real grouped-recognition claim after trust hardening.
- Manuscript location: `sections/04_experimental_setup.tex`, `sections/appendix.tex`
- Supporting figure/table: `Appendix Table A2`
- Repo artifacts: `outputs/real_data_trust_summary.csv`, `docs/REAL_DATA_TRUST_HARDENING.md`
