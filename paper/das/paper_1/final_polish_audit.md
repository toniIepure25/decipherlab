# Final Polish Audit

## 1. Title-Page Problems

- Page 1 still shows template placeholders instead of submission metadata:
  - `Author Name Placeholder`
  - `Coauthor Placeholder`
  - placeholder affiliation lines
  - placeholder email addresses
- The title is too long for the current layout and breaks into three lines, which makes the first page look more like a converted draft than a finished conference submission.
- The abstract is visually too long for the first page in its current form. It dominates the page and leaves the beginning of the introduction cramped at the bottom.
- The first page still carries a dual-author block even though the package should now be a single-author submission.

## 2. Figure Readability Problems

- Page 4, current real-grouped replication figure:
  - subplot labels are tiny at conference scale
  - x-axis labels are long, diagonal, and unreadable in the two-column PDF
  - the main message is not immediate because the reader has to decode four separate panels with dataset-strategy strings
- Page 5, current propagation-regime figure:
  - category labels are compressed and partially diagonal
  - the bar ordering does not visually separate raw uncertainty from conformal clearly enough
  - the main explanatory story is present, but the visual hierarchy is weak
- Page 6, appendix downstream figure:
  - the same tiny-label / diagonal-label problem reappears
  - the figure looks like an export from an internal analysis notebook rather than a final appendix artifact
- Page 6, appendix sensitivity figure:
  - readable, but visually plain relative to the rest of the paper
  - looks more like a diagnostic plot than a final paper figure

## 3. Table Readability Problems

- Page 4, current bridge table:
  - too much prose is embedded inside cells
  - it reads like a CSV summary converted directly to LaTeX
  - line-by-line interpretation text makes the table visually dense and slows scanning
- Page 5, current robustness table:
  - numerically useful, but visually heavy
  - long row labels dominate the table
  - the table feels like a supplemental artifact rather than a polished main-paper table
- Page 5, current appendix tables:
  - too many tables stack vertically on the same page
  - the trust table is important, but visually it should be appendix support, not part of a crowded artifact wall

## 4. Float Placement / Page-Balance Problems

- Page 4 is overloaded:
  - dense bridge table at the top
  - real-grouped replication figure below
  - the page feels crowded and top-heavy
- Page 5 is overloaded:
  - robustness table at the top
  - main propagation figure in the middle
  - downstream appendix table below
  - trust appendix table below that
- The current float order makes the paper feel assembled rather than composed.
- The main figure is present near the correct discussion, but the surrounding float stack weakens its impact.

## 5. Bibliography / Reference Problems

- The PDF currently has no real bibliography section.
- Page 4 ends with a placeholder references paragraph, which is one of the strongest remaining draft signals in the package.
- The bibliography assets are staged locally, but they are not yet wired into a real conference-style references block.

## 6. Prose / Tone Problems

- The paper is already careful and scientifically honest, but parts of it still read like a polished project report rather than a finished conference manuscript.
- The introduction and results occasionally restate the same boundary claims more than once.
- Some paragraph openings are longer than necessary and could be more direct.
- The limitations section is strong in content, but it can be shortened and made more conference-crisp.

## 7. Remaining Template Inconsistencies

- The title page still exposes IEEE template placeholder structure.
- The references placeholder paragraph is not conference-ready.
- The current PDF still contains minor underfull-box artifacts from placeholder or overly long lines.
- The current artifact mix in the main body is too close to the original exported package layout and not yet tuned to the final DAS presentation.
