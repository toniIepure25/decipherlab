# Template Audit

1. Main LaTeX entry point in `paper/das/template`

   - `paper/das/template/IEEE-conference-template-062824/IEEE-conference-template-062824.tex`

2. Bibliography style file to use

   - `paper/das/template/IEEEtran.bst`
   - The template bundle README identifies `IEEEtran.bst` as the standard IEEE bibliography style for IEEE conference work.

3. Bibliography source files present

   - `paper/das/template/IEEEabrv.bib`
   - `paper/das/template/IEEEexample.bib`
   - `paper/das/template/IEEEfull.bib`

4. Section structure expected by the template

   - Standard IEEE conference structure with `\maketitle`, abstract, keywords, numbered sections, optional unnumbered acknowledgments, and references.
   - The template demonstrates standard figure and table placement conventions: figure captions below figures, table captions above tables, and figures/tables cited after first mention.

5. DAS-specific formatting constraints visible in the template bundle

   - The LaTeX source of truth is the IEEE conference template using `\documentclass[conference]{IEEEtran}`.
   - The template abstract warns: do not use symbols, special characters, footnotes, or math in the paper title or abstract.
   - `paper/das/template/das-ieee_paper_format_2026.docx` appears to provide DAS formatting guidance, but no DAS-specific LaTeX class or extra macros are present in the bundle.
   - I therefore treat the DAS Word material as guidance only and the IEEE conference LaTeX template as the operative source of truth.

6. What is reused directly vs adapted minimally

   - Reused directly:
     - `IEEEtran.cls`
     - IEEE conference sectioning and layout conventions
     - `IEEEtran.bst`
     - `IEEEabrv.bib`
   - Adapted minimally:
     - paper content is split into section files under `paper/das/paper_1/sections/`
     - locked figures and table assets from `submission_bplus/` are copied into `paper/das/paper_1/`
     - compact LaTeX tables are created from locked CSV outputs for IEEE-ready inclusion
