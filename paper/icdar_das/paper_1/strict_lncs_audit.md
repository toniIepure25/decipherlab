# Strict LNCS Audit

This audit records the concrete IEEE-like or non-LNCS behaviors that were still visible in the draft PDF and the final LNCS corrections applied in the local submission package.

## 1. Page 1 front matter and abstract block

- Earlier issue on page 1: the front matter still read like a converted conference draft, with named metadata in the workflow and a front-page style that did not fully behave like a blind LNCS submission.
- LNCS correction: the submission now uses LNCS `\title`, `\author`, `\authorrunning`, `\institute`, `\maketitle`, and `abstract` conventions in [main.tex](/home/tonystark/Desktop/decipher/paper/icdar_das/paper_1/main.tex). The visible front matter is now anonymous and compact, and the keyword line appears inside the LNCS abstract block rather than as IEEE-style `Index Terms`.

## 2. Page 1 double-blind compliance

- Earlier issue on page 1 and in source notes: author metadata and project-identifying wording were still close enough to the manuscript path to merit an explicit double-blind audit.
- LNCS correction: the submission manuscript contains no author name, no email, no affiliation, no acknowledgements, and no repository-identifying wording. Camera-ready metadata is isolated in [camera_ready_metadata.md](/home/tonystark/Desktop/decipher/paper/icdar_das/paper_1/camera_ready_metadata.md).

## 3. Pages 1--7 section heading style

- Earlier issue across the body: the paper content came from an IEEE-oriented package and needed to be checked for Roman-numeral or all-caps heading carryover.
- LNCS correction: sectioning now uses native LNCS numbered headings (`1`, `2`, `3`, etc.) through `\section` and `\subsection`, with no Roman numerals or IEEE-style all-caps headings in the manuscript.

## 4. Page 8 Figure 1 caption and numbering style

- Earlier issue on the main results page: the main-figure caption still referenced nonexistent panels (`B and C`) after the figure layout had been simplified.
- LNCS correction: the caption in [05_results.tex](/home/tonystark/Desktop/decipher/paper/icdar_das/paper_1/sections/05_results.tex) now refers only to Panels A and B, matching the rendered figure exactly and using sentence-style LNCS captioning.

## 5. Page 5 Table 1 caption and density

- Earlier issue on the evidence-boundary page: the main table still felt like a dense internal summary artifact, closer to a dashboard than to an LNCS main-paper table.
- LNCS correction: Table 1 now appears as a lean LNCS-style table with sentence-style captioning, compact headers, and short claim-oriented rows rather than prose-heavy cells.

## 6. Page 8 Figure 1 layout

- Earlier issue on page 8: panel B had in-plot annotation clutter and a legend competing with the data region, which made the strongest figure feel cramped.
- LNCS correction: Figure 1 is now a clean two-panel figure with the legend outside the plotting area, no overlaid grouped-rate annotations, and panel spacing adjusted for LNCS scale.

## 7. Page 9 Figure 2 layout

- Earlier issue on page 9: the redesigned downstream figure's right-side comparison was visually cramped and vulnerable to legend overlap.
- LNCS correction: Figure 2 now uses three labeled panels for coverage, raw exact deltas, and conformal exact deltas, eliminating legend overlap and improving left/right balance.

## 8. Pages 7--10 appendix and bibliography behavior

- Earlier issue near the references boundary: appendix floats risked drifting around the bibliography pages, which read like an IEEE-style float spill rather than a deliberate LNCS supplement.
- LNCS correction: the bibliography now appears before the appendix in [main.tex](/home/tonystark/Desktop/decipher/paper/icdar_das/paper_1/main.tex), and appendix material begins only after a forced page break. The trust and robustness tables now remain on appendix pages rather than in the references area.

## 9. Bibliography style

- Earlier issue in the first LNCS transfer: bibliography setup was still partly inherited from the IEEE package.
- LNCS correction: the manuscript now uses Springer LNCS BibTeX style `splncs04` from the completed local template directory, with real bibliography entries in LNCS style.
