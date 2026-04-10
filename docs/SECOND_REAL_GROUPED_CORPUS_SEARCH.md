# Second Real Grouped Corpus Search

## Goal

Find one second real grouped/token-aligned corpus that can feed the existing manifest-backed confusion-network pipeline directly, without a segmentation or HTR rewrite.

## Candidates Considered

### ScaDS.AI German Line- And Word-Level Handwriting Dataset

Status: selected and integrated.

Why it was feasible:

- public archive with manageable size (`232 MB`)
- explicit `page_id`, `line_id`, and `word_id` fields in `word_annotations.csv`
- matching line metadata in `line_annotations.csv`
- pre-extracted word crops in `images/words/`
- XML structure that agrees with the CSV identifiers
- no new segmentation path required

Why it complements Historical Newspapers:

- grouped structure is still real, but the labels are public word annotations rather than OCR-derived ALTO tokens
- it is handwriting-focused rather than OCR-grounded newspaper print
- it tests whether the real grouped result transfers to a second corpus with a different visual/noise profile

Claim it can strengthen if integration succeeds:

- grouped top-k rescue is not only a one-corpus Historical Newspapers artifact

### EPARCHOS

Status: promising but not selected this phase.

Why it was only partially feasible:

- historically grounded and manuscript-adjacent
- line and word annotations appear to exist
- but the practical path here looked less direct than ScaDS because it would require a new page/polygon crop integration pass before we could reuse the current grouped manifest workflow

Why it lost to ScaDS this phase:

- higher engineering cost for the same immediate scientific question
- weaker leverage-to-scope ratio than a corpus that already ships word images keyed to line IDs

### IAM And Similar HTR Corpora

Status: rejected for this phase.

Why they were infeasible or lower leverage:

- some require access friction or nontrivial acquisition steps in this environment
- several are modern handwriting corpora without the same manuscript/OCR-grounded relevance
- they did not beat ScaDS on the combination of direct token alignment, public access, and low integration cost

## Selection Decision

ScaDS.AI was the best complement to Historical Newspapers because it gave the branch exactly what was missing:

- a second real grouped/token-aligned corpus
- public word-level labels
- direct line grouping
- zero segmentation rewrite

## Compatibility Note

One small compatibility adjustment was required after integration:

- the grouped benchmark could support `5` well-covered labels under the existing `min_instances_per_symbol = 4` rule, rather than `6`

This was a dataset-specific coverage adjustment, not a change to the grouped evaluation logic.
