# Submission Package

This directory is a venue-agnostic manuscript package assembled from the frozen DecipherLab evidence pack.

## Contents

- `title_options.md`: final title shortlist with one recommended title
- `abstract.md`: final abstract
- `manuscript.md`: workshop-ready manuscript draft in structured markdown
- `appendix.md`: appendix and reproducibility notes
- `captions.md`: final captions for locked figures and tables
- `figure_table_lock.md`: final inclusion rationale for the selected artifacts
- `final_claims_check.md`: last-pass claims audit in submission language
- `checklist.md`: final manual tasks before submission
- `figures/`: copied submission figures
- `tables/`: copied submission tables

## Recommended Transfer Workflow

1. Choose the target workshop template.
2. Copy the text from `manuscript.md` into the venue template sections.
3. Copy `appendix.md` into the venue appendix or supplement section.
4. Import the locked figures from `figures/`.
5. Format the main table from `tables/table1_cross_dataset_with_ci.csv`.
6. Use `captions.md` as the source of final figure and table captions.
7. Re-check wording against `final_claims_check.md` after any template edits.
8. Complete the missing manual metadata from `checklist.md`.

## What This Package Is Optimized For

- a narrow workshop paper on symbol-level ambiguity robustness
- a conservative uncertainty-aware inference framing
- minimal friction when moving from repo drafts into a formal template

## What Still Needs Manual Input

- authors and affiliations
- acknowledgements or funding
- workshop-specific formatting
- bibliography and venue citation style
- any ethics, data, or conflict-of-interest statements required by the venue
