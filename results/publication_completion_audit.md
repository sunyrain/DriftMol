# Publication Completion Audit

Objective: read and organize the repository, run the missing publication experiments in parallel, fully use compute resources, finish experiments, manuscript writing, and paper figures.

| Requirement | Evidence | Status |
|---|---|---|
| Result CSV, Markdown summary, status JSON, and LaTeX tables exist and are non-empty | all expected result artifacts present and non-empty | PASS |
| Generated LaTeX tables are synchronized to CSV/status sources | QED main, fair multi4, and QED 3-seed table rows match source artifacts | PASS |
| Publication manifest exists and defines every required experiment entry | 24 required manifest entries present | PASS |
| Publication manifest entries reference existing config files and matching commands | all manifest configs exist and commands reference their config | PASS |
| No required publication experiment is pending or incomplete | pending_or_incomplete=0 | PASS |
| F/A6/A8/G4 QED variants have complete 3-seed coverage | all key variants have 3/3 seeds | PASS |
| z-diversity Pareto sweep is complete | all z-div points complete | PASS |
| Fair G4 multi4 publication run is complete and passes the quality gate | complete_pass | PASS |
| Paper-faithfulness audit P1-P6 is complete | P1-P6 complete | PASS |
| Inference throughput and 1-NFE claims have benchmark evidence | nfe=1, generator_mol_per_s=47533.2, end_to_end_mol_per_s=4604.2 | PASS |
| AAAI manuscript source exists with final author metadata | title, authors, corresponding author, affiliation, and AAAI style present | PASS |
| AAAI source references required final result tables | AAAI table labels and refs present | PASS |
| AAAI source references all final paper figures | AAAI figure includes, labels, and refs present | PASS |
| AAAI LaTeX build prerequisites are documented and available | build notes present; TeX engine=latexmk; aaai2026 style/bst/bib present | PASS |
| AAAI source reflects inference benchmark throughput | generator/end-to-end throughput and sample count reflected in AAAI source | PASS |
| AAAI citations resolve against references_aaai.bib | 19 cited keys resolved by references_aaai.bib | PASS |
| Manuscript computational-cost table is synchronized to benchmark JSON | benchmark rates already reflected in manuscript | PASS |
| Manuscript exists and references required final result tables | required table labels and refs present | PASS |
| All final paper figures exist as non-empty PDF and PNG files | all required figure files present and non-empty | PASS |
| Manuscript references all final paper figures | all required figure includes, labels, and refs present | PASS |
| Manuscript LaTeX build prerequisites are documented and available | build notes present; TeX engine=pdflatex; sn-jnl.cls resolved via docs/ | PASS |
| Repository result documentation uses the current publication protocol | README and FULL_RESULTS point to generated publication tables and fair multi4 v2 | PASS |
| Manuscript citations and bibliography are internally consistent | 20 cited keys all resolved | PASS |
| Unit test suite passes | OK (95 tests) | PASS |

Overall: PASS. The publication package satisfies the audited completion gate.
