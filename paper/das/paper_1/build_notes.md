# Build Notes

## Template Used

- Base template: `paper/das/template/IEEE-conference-template-062824/IEEE-conference-template-062824.tex`
- Local manuscript entry point: `paper/das/paper_1/main.tex`

## Compile Commands

Run from `paper/das/paper_1`:

```bash
python3 build_paper_assets.py
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## DAS / IEEE Adjustments Made

- Preserved the IEEE conference document class and standard package set.
- Split the manuscript into section files for maintainability.
- Converted locked CSV results into compact IEEE-ready LaTeX tables.
- Kept the title and abstract free of math, footnotes, and special formatting to match the template warning.

## Remaining Manual Completion

- Replace author and affiliation placeholders.
- Upgrade any bibliography placeholders in `bib/references.bib` with fuller metadata if the final submission requires it.
- Decide whether the venue requires a separate data availability or acknowledgment wording.
