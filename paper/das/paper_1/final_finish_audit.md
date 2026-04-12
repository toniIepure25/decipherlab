# Final Finish Audit

This note records the last draft-like issues identified in the DAS package and the concrete fixes applied in the final finishing pass.

## 1. Title block and author block

- Page 1, title block: the earlier title stacked awkwardly across three lines and made the front page feel like a converted internal report.
- Page 1, author block: `Affiliation pending final metadata` looked explicitly provisional.
- Fix applied: the title was tightened to `Support-Aware Uncertainty Propagation in Low-Resource Glyph and Grouped Recognition`, and the author block now uses a final-looking single-author identity:
  - `Iepure Antoniu`
  - `Independent Researcher`
  - `iepuretoni533@gmail.com`

## 2. Figure 1 problems

- Main results page, Figure 1 panel B: the legend sat on top of the plotting region, and the earlier regime explanation felt like an analytics dashboard rather than a paper figure.
- Main results page, Figure 1 overall: the previous layout did not make the paper's main conclusion legible within a few seconds.
- Fix applied: Figure 1 was rebuilt so the grouped-replication signal and the support-gated rescue story are visually separated, the legend no longer sits inside the data region, and the caption now states the figure's conclusion directly.

## 3. Figure 2 problems

- Appendix figure, earlier right panel: the legend floated over the plotted values, which made the selective downstream comparison hard to read at conference scale.
- Fix applied: the appendix downstream figure now separates coverage, raw exact deltas, and conformal exact deltas into distinct panels, eliminating legend overlap entirely.

## 4. Table I problems

- Results section, Table I: the older table tried to summarize too much of the paper at once and read more like an internal summary dashboard than a conference table.
- Fix applied: Table I was reduced to a compact evidence-boundary table with shorter headers, four rows, and no trust-audit prose inside the main table body.

## 5. Float economy and page balance

- Earlier draft state: the main-paper float stack felt crowded because the evidence-boundary table, the main figure, and appendix-support material were visually too close together.
- Fix applied: the appendix now starts on a fresh page, the redesigned downstream figure lives in the appendix, and the main paper keeps only the strongest figure and the lean boundary table.

## 6. Related work and bibliography

- Earlier draft state: the bibliography was too repo-facing and too light on field-facing grounding.
- Fix applied: the active bibliography now includes calibration, conformal prediction, handwriting recognition, structured decoding, and benchmark-grounding references, and the related-work section now positions the paper against those literatures rather than only against local artifacts.

## 7. Prose and conference tone

- Earlier draft state: parts of the paper still sounded like a careful project report, especially in transitions around results and limitations.
- Fix applied: the results arc now argues one clearer story:
  1. grouped rescue replicates;
  2. downstream gains remain selective even after coverage improves; and
  3. support-aware propagation explains why.

## 8. Remaining minor issues

- `main.log` still contains a few small `Underfull \hbox` warnings from compact table text, but there are no blocking compilation errors.
- The only remaining human-choice item is whether `Independent Researcher` should stay as the final affiliation line or be replaced with a confirmed institutional affiliation before submission.
