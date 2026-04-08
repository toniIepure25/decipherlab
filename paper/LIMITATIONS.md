# Limitations

- The present evidence supports symbol-level retention of correct alternatives under ambiguity, not decipherment, translation, or semantic recovery.
- The third dataset materially strengthens the paper, but the historical corpus is still evaluated through a balanced subset manifest rather than a full-manifest sweep. That choice is documented and principled, but it remains a scope-limiting decision.
- None of the three external datasets provides decipherment-family labels aligned with the current downstream hypothesis families. Omniglot offers alphabet grouping for characterization, while scikit-learn digits and Kuzushiji-49 provide no meaningful document-level grouping in the current manifests. Downstream family claims therefore remain unproven.
- Calibration is not stable across datasets. It helped on Omniglot and Kuzushiji-49, but hurt on scikit-learn digits in several conditions, so the paper cannot frame calibration as a universally reliable addition.
- Symbol-level rescue does not automatically propagate to downstream reasoning. All three datasets contain explicit `uncertainty_helped_symbols_not_downstream` cases, and those failures are part of the main evidence pack rather than hidden in appendix-only artifacts.
- The cross-dataset story now spans three corpora, but only one of them is explicitly historical and none is sequence-rich in a way that supports grouped decipherment-style evaluation.
- Stronger publication claims require a manuscript or cipher corpus with richer grouped structure, defensible higher-level labels, and enough signal to test whether rescued alternatives improve reasoning beyond symbol top-k retention.
