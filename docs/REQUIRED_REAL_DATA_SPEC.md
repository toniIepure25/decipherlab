# Required Real Data Specification

## Purpose

This document specifies the minimum real dataset that would unlock the next empirical step for the sequence branch.

The goal is not semantic decipherment. The goal is to test whether the current structured-uncertainty advantage survives on real grouped sequence data.

## Minimum Required Data Interface

Each example should support:

- a `group_id` or `sequence_id`
- an ordered sequence of token- or symbol-level visual units
- one label per unit
- a stable train/val/test split

In practice, the easiest acceptable forms are:

1. pre-segmented symbol crops grouped into real lines or records
2. token crops with token labels and stable sequence order
3. line images plus trusted token or symbol segmentation points

## Minimum Metadata Needed

- ordered position within a sequence
- source page or document id
- split membership
- label inventory
- sequence length

Helpful but optional:

- line transcription
- family or document-type metadata
- token confidence or annotator agreement

## What Is Not Enough

The following are not sufficient for the current branch on their own:

- page images plus transcriptions only
- line images plus transcriptions only
- word images without token alignment
- character crops without any real grouped sequence structure

Those resources are valuable, but they would require a broader segmentation or recognizer rewrite first.

## Minimal Annotation That Would Unlock The Branch

If an otherwise good historical corpus only has line images, the smallest extra annotation that would make it usable here is:

- token or symbol segmentation points or boxes
- ordered token labels aligned to those points or boxes

That would let the current manifest workflow represent real sequences directly and would avoid a full OCR/HTR model detour.

## Expected Engineering Cost

If a dataset already satisfies the requirements above, integration should be modest:

- a dataset-specific preparation script
- manifest generation
- validation checks
- one experiment config

If it lacks token alignment, the cost becomes much larger:

- segmentation
- alignment logic
- new uncertainty interfaces for real line images

That is likely a multi-week branch expansion rather than a clean next experiment.
