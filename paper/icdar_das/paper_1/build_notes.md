# Build Notes

## Template source of truth

- Class file: `paper/icdar_das/template/llncs.cls`
- Bibliography style: `paper/icdar_das/template/splncs04.bst`
- Conventions reference: `paper/icdar_das/template/llncsdoc.tex`

## Local compile setup

- `paper/icdar_das/paper_1/llncs.cls` is a symlink to `../template/llncs.cls`
- `paper/icdar_das/paper_1/splncs04.bst` is a symlink to `../template/splncs04.bst`

## Compile commands

```bash
cd paper/icdar_das/paper_1
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Current build status

- `main.pdf` builds successfully.
- The manuscript is anonymous for submission.
- The build uses real LNCS support files, not hand-written stand-ins.
- The current title, captions, and table wording reflect the final micro-polish pass for submission review.

## Remaining minor build warnings

- small keyword-line overfull warning on page 1
- a few `Underfull \hbox` warnings from compact appendix table text
- BibTeX sorts dataset records as `misc` entries; these compile correctly but could be enriched later if a better archival dataset citation is preferred
