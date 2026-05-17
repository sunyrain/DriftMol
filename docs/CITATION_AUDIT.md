# Citation Audit

Date: 2026-05-17 UTC

Scope: `docs/PAPER_AAAI.tex` and `docs/references_aaai.bib`.

## Summary

- All 19 citation keys used by `docs/PAPER_AAAI.tex` resolve in `docs/references_aaai.bib`.
- The LaTeX/BibTeX build completes with no undefined citations or BibTeX warnings.
- Removed unsafe metadata patterns: no incomplete author lists and no stale citation keys remain in the active AAAI source.
- Corrected high-risk entries:
  - Drifting Models: authors are Mingyang Deng, He Li, Tianhong Li, Yilun Du, Kaiming He; year is 2026; arXiv ID is 2602.04770.
  - FREED reference: replaced unsupported ICLR 2024 metadata with the NeurIPS 2021 paper "Hit and Lead Discovery with Explorative RL and Fragment-based Molecule Generation".
  - LIMO author list: corrected Kunyang Sun and Michael Gilson.
  - GraphNVP author spelling follows the arXiv metadata: Katushiko Ishiguro.

## DOI and Source Status

| Key | Status | Verified source |
|---|---|---|
| `deng2026drifting` | arXiv DOI `10.48550/arXiv.2602.04770` | https://arxiv.org/abs/2602.04770 |
| `digress` | arXiv DOI `10.48550/arXiv.2209.14734`; accepted ICLR 2023 URL included | https://openreview.net/forum?id=UaAD-Nu86WX |
| `moflow` | DOI `10.1145/3394486.3403104` | Crossref / ACM DOI record |
| `jtvae` | no proceedings DOI found; PMLR URL included | https://proceedings.mlr.press/v80/jin18a.html |
| `graphvae` | DOI `10.1007/978-3-030-01418-6_41` | Crossref / Springer DOI record |
| `graphnvp` | arXiv DOI `10.48550/arXiv.1905.11600` | https://arxiv.org/abs/1905.11600 |
| `gdss` | arXiv DOI `10.48550/arXiv.2202.02514`; PMLR URL included | https://proceedings.mlr.press/v162/jo22a.html |
| `irwin2012zinc` | DOI `10.1021/ci3001277` | Crossref / ACS DOI record |
| `krenn2020selfies` | DOI `10.1088/2632-2153/aba947` | Crossref / IOP DOI record |
| `krenn2022selfies` | DOI `10.1016/j.patter.2022.100588` | Crossref / Elsevier DOI record |
| `ho2022classifierfree` | arXiv DOI `10.48550/arXiv.2207.12598`; workshop URL included | https://openreview.net/forum?id=qw8AKxfYbI |
| `peebles2023scalable` | DOI `10.1109/ICCV51070.2023.00387` | Crossref / IEEE DOI record |
| `gomez2018automatic` | DOI `10.1021/acscentsci.7b00572` | Crossref / ACS DOI record |
| `olivecrona2017molecular` | DOI `10.1186/s13321-017-0235-x` | Crossref / Springer DOI record |
| `eckmann2022limo` | no PMLR DOI found; PMLR URL included | https://proceedings.mlr.press/v162/eckmann22a.html |
| `zhou2019optimization` | DOI `10.1038/s41598-019-47148-x` | Crossref / Nature DOI record |
| `yang2021freed` | no NeurIPS DOI found; NeurIPS proceedings URL included | https://papers.nips.cc/paper_files/paper/2021/hash/41da609c519d77b29be442f8c1105647-Abstract.html |
| `lee2023mood` | no PMLR DOI found; PMLR URL included | https://proceedings.mlr.press/v202/lee23f.html |
| `lipman2023flow` | arXiv DOI `10.48550/arXiv.2210.02747`; accepted ICLR 2023 URL included | https://openreview.net/forum?id=PqvMRDCJT9t |

## Local Verification Commands

```bash
TEXINPUTS=docs//: BSTINPUTS=docs//: latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=DriftingMol_AAAI docs/PAPER_AAAI.tex
grep -n "Warning\\|undefined\\|Citation\\|Overfull\\|!" DriftingMol_AAAI.log
python scripts/audit_publication_completion.py --run-tests
python scripts/audit_generative_baselines.py
```
