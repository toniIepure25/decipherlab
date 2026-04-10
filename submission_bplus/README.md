# Submission B+ Package

This directory is the reviewer-facing manuscript package for the stronger sequence branch.

It is optimized for a bounded empirical paper on support-aware uncertainty propagation in low-resource glyph and grouped-sequence recognition.

## Contents

- `venue_fit.md`: practical venue-fit analysis by paper type
- `title_options.md`: five title options, one recommendation, and one positioning line
- `abstract.md`: final abstract
- `manuscript.md`: main manuscript draft in structured markdown
- `appendix.md`: appendix-ready supplementary material
- `captions.md`: locked captions for the selected figures and tables
- `figure_table_lock.md`: final artifact selection and rationale
- `claims_check.md`: exact central claim, secondary claims, non-claims, and softened wording
- `reviewer_risk_checklist.md`: likely reviewer objections and evidence-based responses
- `reproducibility_checklist.md`: exact commands and dependencies for rebuilding the main submission artifacts
- `figures/`: locked figure assets or source pointers
- `tables/`: locked table assets or source pointers

## What This Package Is Optimized For

- a document-analysis, handwriting, or grouped-recognition venue
- a bounded empirical methods paper
- a paper whose novelty comes from support-aware explanation, not a new black-box model

## What This Package Is Not Trying To Be

- a decipherment breakthrough submission
- a semantic recovery paper
- a universal uncertainty framework paper
- a general large-model or foundation-model paper

## Final Transfer Workflow

1. Use `manuscript.md` as the source text for the target venue template.
2. Use `abstract.md` and `title_options.md` as the final title/abstract source.
3. Keep the artifact selection fixed using `figure_table_lock.md`.
4. Re-check every sentence-level claim against `claims_check.md`.
5. Use `reviewer_risk_checklist.md` to strengthen the introduction, limitations, and discussion before submission.
6. Use `reproducibility_checklist.md` when drafting the final appendix or supplement reproducibility block.
