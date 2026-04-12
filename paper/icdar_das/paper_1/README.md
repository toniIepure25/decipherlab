# DAS 2026 LNCS Submission Package

This directory contains the anonymous Springer LNCS submission package for DAS 2026 built from the stronger sequence-branch paper content. The current state is intended as a submission-finished review package rather than a draft transfer.

## Key files

- `main.tex`: anonymous LNCS manuscript entry point
- `sections/`: manuscript body split into clean sections
- `figures/`: locked paper figures
- `tables/`: main and appendix tables
- `bib/references.bib`: active bibliography entries
- `template_completion_audit.md`: LNCS template completion record
- `captions.md`: final caption lock for the LNCS package
- `anonymity_checklist.md`: double-blind submission verification
- `camera_ready_metadata.md`: withheld author metadata for later restoration
- `page_budget_notes.md`: page-count and compression notes

## Build

Run from `paper/icdar_das/paper_1`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Positioning

The paper is positioned as a bounded empirical study of support-aware uncertainty propagation in low-resource glyph and grouped recognition. It is not positioned as decipherment, semantic recovery, or a universal uncertainty framework.
