# Figure And Table Lock

## Main Figure

- Filename: [propagation_regime_plot.png](/home/tonystark/Desktop/decipher/outputs/propagation_regime_plot.png)
- What it shows: real downstream rescue rates by method family, support regime, and entropy regime
- Why it is central: it turns the paper from a mixed-results decoder paper into an explanatory paper
- Where it belongs: main results section, after the redesigned real downstream task
- Reviewer concern addressed: “is there any coherent explanation for the mixed downstream result?”

## Main Table

- Filename: [real_vs_synthetic_bridge_summary.csv](/home/tonystark/Desktop/decipher/outputs/real_vs_synthetic_bridge_summary.csv)
- What it shows: the evidence ladder from real symbol-level results to synthetic grouped/downstream results to real grouped replication and redesigned real downstream results
- Why it is central: it makes the claim boundary explicit and keeps the paper honest
- Where it belongs: end of the main results section or beginning of the discussion
- Reviewer concern addressed: “what exactly is real, replicated, synthetic-only, or still mixed?”

## Secondary Artifact 1

- Filename: [real_grouped_replication_plot.png](/home/tonystark/Desktop/decipher/outputs/real_grouped_replication_plot.png)
- What it shows: grouped exact and grouped top-k behavior across Historical Newspapers and ScaDS.AI
- Why it is central: it establishes the strongest real-data positive result in the branch
- Where it belongs: main results section before the downstream task
- Reviewer concern addressed: “is the real grouped result just a one-corpus artifact?”

## Secondary Artifact 2

- Filename: [statistical_robustness_summary.csv](/home/tonystark/Desktop/decipher/outputs/statistical_robustness_summary.csv)
- What it shows: bootstrap intervals and directionality for the main real and synthetic claims
- Why it is central: it hardens the paper against “mixed but under-supported” reviewer reactions
- Where it belongs: main results section or short robustness section after the bridge table
- Reviewer concern addressed: “are the paper’s main claims statistically stable enough to trust?”

## Appendix Boundary Artifact

- Filename: [real_downstream_redesigned_summary.csv](/home/tonystark/Desktop/decipher/outputs/real_downstream_redesigned_summary.csv)
- What it shows: redesigned downstream coverage and selective exact gains across the two real grouped corpora
- Why it is central: it proves that the remaining downstream limit is not just a trivial coverage collapse
- Where it belongs: appendix or supplement boundary-analysis section
- Reviewer concern addressed: “is the negative downstream result still just a bad task design artifact?”

## Supplementary Trust Artifact

- Filename: [real_data_trust_summary.csv](/home/tonystark/Desktop/decipher/outputs/real_data_trust_summary.csv)
- What it shows: Historical Newspapers audit counts, gold-style agreement, error taxonomy, and zero metric drift across label tiers
- Why it is central: it hardens the trust story for the most scrutinizable real grouped corpus
- Where it belongs: appendix or supplement data-quality section
- Reviewer concern addressed: “why should we trust the Historical Newspapers labels?”
