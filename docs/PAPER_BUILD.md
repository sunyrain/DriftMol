# Manuscript Build Notes

This file documents how to build the current manuscript sources from a clean
checkout. Generated PDFs and LaTeX auxiliary files are local artifacts and are
ignored by Git.

## Conference-Template Draft

`docs/PAPER_AAAI.tex` is the active conference-template source. It currently
uses the local `aaai2026` class:

```tex
\documentclass[letterpaper]{article}
\usepackage[draft]{aaai2026}
```

Keep the matching template and bibliography files with the source:

- `docs/aaai2026.sty`
- `docs/aaai2026.bst`
- `docs/references_aaai.bib`

Build from the repository root:

```bash
TEXINPUTS=docs//: BSTINPUTS=docs//: latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=DriftingMol_AAAI docs/PAPER_AAAI.tex
```

The source keeps main result tables inline. If `results/tables/*.tex` is
refreshed, manually synchronize the inline manuscript tables before building a
submission package.

Primary figures are stored in `docs/figures/`. Quantitative figures should be
regenerated from tracked scripts and result artifacts:

```bash
python scripts/collect_results.py
python scripts/export_latex_tables.py
python scripts/plot_result_figures.py
```

The reproducibility checklist source is:

- `docs/AAAI_REPRODUCIBILITY_CHECKLIST.tex`

The supplemental source is staged as:

- `docs/SUPPLEMENT_FAITHFUL_DRIFTING.tex`
- `docs/SUPPLEMENT_FAITHFUL_DRIFTING_INLINED.tex`
- `docs/SUPPLEMENT_FAITHFUL_DRIFTING_AAAI.tex`

Regenerate the inlined supplement source with:

```bash
python scripts/render_faithful_supplement.py
```

Build the standalone supplement with:

```bash
TEXINPUTS=docs//: BSTINPUTS=docs//: latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=DriftingMol_AAAI_FaithfulSupplement docs/SUPPLEMENT_FAITHFUL_DRIFTING_AAAI.tex
```

## Springer Nature Draft

`docs/PAPER_DRAFT.md` is a LaTeX manuscript source despite the `.md`
extension. It uses the Springer Nature `sn-jnl` document class:

```tex
\documentclass[pdflatex,sn-nature]{sn-jnl}
```

Local template files:

- `docs/sn-jnl.cls`
- `docs/bst/sn-nature.bst`

Build from the repository root:

```bash
TEXINPUTS=docs//: BSTINPUTS=docs/bst//: pdflatex -interaction=nonstopmode -halt-on-error -jobname=DriftingMol docs/PAPER_DRAFT.md
TEXINPUTS=docs//: BSTINPUTS=docs/bst//: pdflatex -interaction=nonstopmode -halt-on-error -jobname=DriftingMol docs/PAPER_DRAFT.md
```

If the Springer Nature template is updated upstream, replace the local class
and bibliography style from the official Springer Nature LaTeX package rather
than from an unrelated mirror.

## Packaging Rule

Before sharing or submitting an artifact, include only source files needed to
compile the manuscript: TeX sources, template files, bibliography files,
figures, and any required inlined table content. Do not commit generated PDFs,
`.aux`, `.bbl`, `.blg`, `.fls`, `.fdb_latexmk`, or local submission zip files.
