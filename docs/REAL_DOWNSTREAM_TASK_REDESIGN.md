# Real Downstream Task Redesign

## Why The Original Task Was Too Weak

The original real downstream task, `train_transcript_bank`, asked the system to recover an exact grouped transcript by ranking same-length transcripts seen in the `train` split.

That task was honest, but too coverage-limited:

- Historical Newspapers: exact train/test transcript overlap was `0.000`
- ScaDS.AI: exact train/test transcript overlap was only `0.111`

So the negative result on exact recovery could not distinguish:

- a true limit of preserved uncertainty
- from a task design that made exact recovery impossible on most test examples

## Candidate Redesigns Considered

### 1. Exact Transcript Bank

- strongest lexical target
- already implemented
- rejected as the primary downstream task because exact overlap is too low

### 2. Edit-Distance-Bounded Transcript Bank

- could raise candidate availability
- but still depends on train-lexicon overlap and introduces a heuristic radius choice
- rejected for this phase because it adds more arbitrary retrieval design than evidence

### 3. Train-Supported Trigram Path

- structurally meaningful
- better covered than exact transcript retrieval
- but on Historical Newspapers the actual benchmarked trigram support was still too sparse
- rejected in favor of a slightly more permissive n-gram target

### 4. Train-Supported Bigram Path

- still derived only from real train transcripts
- preserves local structural order
- materially improves coverage on both corpora
- compatible with the current decoder outputs without adding a new model family

## Selected Task

The selected redesigned task is `train_supported_ngram_path`, instantiated here as a train-supported bigram path.

For each real grouped corpus:

- build the inventory of character bigrams from the `train` split only
- convert each gold grouped transcript into its ordered path of train-supported bigrams
- convert each decoder output into the same ordered bigram path representation
- evaluate exact path recovery, top-k path recovery, token accuracy, and CER on that derived structural target

## Why This Is Stronger

This task is scientifically stronger than exact transcript-bank decoding because:

- it still uses real grouped transcripts and train-only resources
- it keeps the target structural and ordered rather than collapsing to bag-of-symbol overlap
- it avoids the extreme train/test lexical overlap problem
- it tests whether preserved uncertainty helps recover real local transcript structure, not only raw grouped top-k alternatives

## Coverage Improvement

On the actual benchmarked grouped sequences:

- Historical Newspapers:
  - exact transcript-bank upper bound: `0.000`
  - train-supported bigram-path upper bound: `0.667`
- ScaDS.AI:
  - exact transcript-bank upper bound: `0.111`
  - train-supported bigram-path upper bound: `1.000`

So the redesigned task is a substantially better downstream test.

## What Counts As Success

Success would mean any of the following:

- positive exact bigram-path recovery delta for `uncertainty_beam` over `fixed_greedy` on both corpora
- positive exact bigram-path recovery for `conformal_beam` that is not confined to one corpus
- positive downstream token-accuracy gains that replicate across both corpora

## What Counts As Failure

Failure would mean:

- the redesigned task improves coverage substantially
- but exact or partial downstream gains remain mixed, one-corpus-specific, or strategy-specific

That would still be useful, because it would show that the remaining bottleneck is not only coverage. It would imply a real limit in how far grouped top-k rescue currently propagates into higher-level real transcript structure.
