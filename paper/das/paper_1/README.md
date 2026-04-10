# DAS 2026 Paper Package

This directory contains the DAS 2026 LaTeX transfer of the stronger sequence-branch paper package.

## Template Source

- Template source of truth: `paper/das/template/IEEE-conference-template-062824/IEEE-conference-template-062824.tex`
- Local working entry point: `paper/das/paper_1/main.tex`

## Main Files

- `main.tex`: IEEE conference manuscript entry point
- `abstract.tex`: abstract body
- `build_paper_assets.py`: rebuilds publication-quality paper figures from locked CSV outputs
- `sections/`: numbered paper sections plus appendix
- `figures/`: locked figure assets copied from `submission_bplus/`
- `tables/`: locked CSV artifacts plus paper-ready LaTeX tables
- `bib/`: IEEE bibliography style and placeholder project bibliography

## Notes

- Scientific content is transferred from `submission_bplus/` without expanding the paper’s claim boundary.
- Bibliography wiring is active with conservative local dataset/source entries.
- Author, affiliation, and acknowledgment lines are placeholders for the final submission pass.
