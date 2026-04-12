# Paper 2 Plan

## Central contribution

`paper_2` is a method/system paper built on top of `paper_1`. Its central contribution is a lightweight support-aware adaptive decoder that uses measured support features to decide how aggressively uncertainty should be preserved or pruned at inference time.

## Why this is different from paper_1

`paper_1` is primarily explanatory:

- it establishes replicated grouped top-k rescue
- it shows that downstream propagation is selective
- it explains when propagation fails

`paper_2` is operational:

- it converts the propagation findings into an explicit decoding policy
- it evaluates that policy as a method
- it frames the result as a document-analysis system contribution rather than only an explanatory paper

This makes `paper_2` more than “paper_1 plus one more ablation.”

## Exact problem it solves

The current pipeline forces the user to choose one decoder strategy globally. That is a mismatch with the evidence from `paper_1`, where raw uncertainty and conformal help in different support regimes. The method problem for `paper_2` is therefore:

- how to choose the right decoding behavior per example without introducing a large opaque model

## Likely central claim

The strongest plausible central claim is:

- a support-aware adaptive decoder can use measured support signals to choose between preserving and pruning uncertainty, improving real grouped and selectively improving real downstream recovery relative to fixed decoder choices while remaining interpretable and auditable

## What would count as success

Success for `paper_2` would mean:

- the adaptive controller improves grouped top-k or grouped exact on the two real grouped corpora
- and/or it improves the redesigned real downstream task relative to the strongest fixed bigram baselines
- the controller reduces at least one recurrent failure mode
- the method remains simple enough to explain with a compact pipeline/controller figure
- the paper clearly feels like a DAS system paper

## What would count as failure

The paper_2 track is not strong enough if:

- the adaptive controller only matches the best baseline without a compensating efficiency or robustness story
- gains appear only on synthetic tasks
- the policy becomes complicated enough that it no longer looks lightweight or auditable
- the result is just another decoder comparison without a practical workflow interpretation
