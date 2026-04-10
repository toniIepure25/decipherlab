# Sequence Dataset Search

## Goal

Identify the highest-leverage path to real grouped/sequence evidence for the sequence branch without triggering a segmentation or full HTR rewrite.

## Early Search Outcome

The first candidate wave showed a common problem:

- many real manuscript or handwriting datasets provide line images and transcriptions
- the current branch needs token- or symbol-aligned visual units

That ruled out several otherwise strong corpora for the current phase.

## Candidates Considered

### IAM / Bentham / George Washington style corpora

Why they mattered:

- real handwritten lines
- clear grouped sequence structure

Why they were not chosen:

- line-level or word-level recognition is the practical entry point
- the current branch would need segmentation/alignment machinery it does not have

### ICDAR2019 HDRC / M5HisDoc

Why they mattered:

- historically grounded manuscript-like corpora
- stronger historical relevance than Omniglot or Digits

Why they were not chosen:

- HDRC still lacked directly usable token-aligned units
- M5HisDoc was not a clean reproducible integration path in this environment

### Historical Newspapers Ground Truth

Why it ultimately won:

- public and lightweight enough to integrate immediately
- ALTO OCR provides grouped text lines plus token boxes
- token crops can be represented directly in the existing manifest and confusion-network workflow

## Final Decision

The branch moved from pure search into real grouped public integration with:

- `Historical Newspapers Ground Truth` (Zenodo 2583866)

The resulting benchmark is real grouped data, but with OCR-derived token labels.

## Practical Boundary

The search therefore ended with a mixed conclusion:

- a clean **gold-token manuscript** dataset still was not found for this phase
- a **preliminary real grouped/token-aligned** public corpus was found and integrated successfully

That is why the branch now supports preliminary grouped transfer claims, but not stronger gold-token manuscript claims.
