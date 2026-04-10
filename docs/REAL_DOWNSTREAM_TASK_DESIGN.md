# Real Downstream Task Design

## Chosen Task

The real downstream task for this phase is `train_transcript_bank` decoding.

For each real grouped corpus:

- build a bank of exact grouped transcripts from the `train` split only
- partition that bank by transcript length
- for each evaluation line, score same-length train transcripts with the current posterior and decoder-family structural prior
- rank the bank transcripts and evaluate exact/top-k grouped transcript recovery

## Why This Task

This is the highest-leverage real downstream task available from the current corpora because it uses:

- real grouped sequences
- real transcript labels
- real sequence constraints derived only from training data
- the existing confusion-network and decoder stack

It creates a real structural target beyond grouped token top-k rescue without requiring a new dataset hunt or a segmentation rewrite.

## What It Can Strengthen

If it works, it can strengthen the claim from:

- grouped top-k rescue transfers to real grouped data

to:

- preserved uncertainty can improve constrained grouped transcript recovery on real grouped data

That is still not a semantic-decipherment claim. It is a real structured-recovery claim under a train-derived transcript bank constraint.

## What Counts As Success

Success would mean at least one of the following on the real grouped corpora:

- positive exact grouped transcript recovery delta for `uncertainty_beam` over `fixed_greedy`
- stronger exact grouped transcript recovery for `conformal_beam` than for raw uncertainty
- positive top-k grouped transcript recovery under the transcript-bank task that is larger than the exact-match gain and helps explain the remaining gap

The strongest outcome would be replication of any exact downstream gain across both corpora.

## What Counts As Failure

Failure would mean:

- grouped top-k rescue remains positive, but exact transcript-bank recovery does not improve
- or gains appear only when the true transcript is already covered by the train bank and disappear otherwise

That would still be scientifically useful, because it would show a real limit: grouped uncertainty rescue does not automatically become exact structured recovery on real grouped corpora.

## Main Caveat

This task is coverage-bounded by the training transcript bank:

- if a test transcript is absent from the train bank, exact recovery is impossible under this task

So coverage must be reported alongside downstream accuracy rather than hidden.
