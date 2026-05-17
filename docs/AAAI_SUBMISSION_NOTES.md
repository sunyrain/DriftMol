# AAAI Submission Notes

Updated: 2026-05-17 UTC

## Local AAAI Source

- Main AAAI draft: `docs/PAPER_AAAI.tex`
- Bibliography: `docs/references_aaai.bib`
- AAAI style files: `docs/aaai2026.sty`, `docs/aaai2026.bst`
- Refined image-2 raster mechanism figure: `docs/figures/fig1_main.png`
- Separate reproducibility checklist: `docs/AAAI_REPRODUCIBILITY_CHECKLIST.tex`
- Faithful-Drifting supplement skeleton:
  `docs/SUPPLEMENT_FAITHFUL_DRIFTING.tex`
- Inlined faithful-Drifting supplement source:
  `docs/SUPPLEMENT_FAITHFUL_DRIFTING_INLINED.tex`
- Standalone faithful-Drifting supplement source/PDF:
  `docs/SUPPLEMENT_FAITHFUL_DRIFTING_AAAI.tex`,
  `DriftingMol_AAAI_FaithfulSupplement.pdf`
- Same-backbone generative-model baseline plan:
  `docs/GEN_MODEL_BASELINE_PLAN.md`,
  `configs/publication_ext/generative_baselines_manifest.json`,
  `scripts/train_latent_generative_baseline.py`

The AAAI source uses `article` plus `\usepackage[draft]{aaai2026}` for the
current visible-author draft. Authors are Jiangjie Qiu, Yijun Li, Wentao Li,
and Xiaonan Wang, all affiliated with Beijing Key Laboratory of Artificial
Intelligence for Advanced Chemical Engineering Materials; Xiaonan Wang is
marked as corresponding author. The existing Springer/Nature long draft remains
at `docs/PAPER_DRAFT.md` and is not overwritten.

The AAAI source keeps the result tables inline. This is intentional: the AAAI
2026 LaTeX instructions allow BibTeX as a separate source file but disallow
additional `\input`/`\include` source files. After the final result refresh,
copy updated rows from `results/tables/*.tex` into `docs/PAPER_AAAI.tex`.
For the faithful-Drifting supplement, run
`python scripts/render_faithful_supplement.py` to refresh the inlined source
after `scripts/collect_faithful_drifting_results.py`.

Current clean submission/source bundle:
`submission/driftingmol_submission_source.zip`. It contains only the active main
source, AAAI style/bibliography files, and the five figures used by the main
paper; it was verified by zip integrity testing and by compiling from a fresh
temporary unzip directory.
The approved image-2 figure is kept at
`docs/figures/image2_assets/fig1_main_image2_final.png`; the submission source
references `docs/figures/fig1_main.png`. The data figures are generated from
scripts and experiment artifacts rather than image-2 assets. The earlier teaser
figure is no longer used by the AAAI draft. The AAAI draft now includes a
mathematical-analysis section with six formal propositions and proofs covering
decoder gradient coupling, the induced pullback metric, bounded drift targets,
conditional mean-shift drift, z-diversity repulsion, SELFIES validity, and
one-forward-pass inference. The current quantitative figures use descriptive
mechanism labels, with Fig.2 shown as a script-generated ablation ranking plus
mechanism-group summary, Fig.3 as z-diversity sensitivity, Fig.4 as seed-level
QED stability, and Fig.5 as fair four-property conditioning.
Same-backbone CVAE, WGAN-GP, DDPM, and Flow-Matching generative baselines are
complete and tracked separately as the proper generator-family comparison.
The current compiled main PDF is 9 pages. The AAAI reproducibility checklist is
kept as a separate source/PDF rather than appended after the references. Public
paper-facing tables and figures use descriptive model/control names; internal
configuration IDs such as A6/A8/F/G4 remain only in result provenance files.

## Build Command

Build from the repository root:

```bash
TEXINPUTS=docs//: BSTINPUTS=docs//: latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=DriftingMol_AAAI docs/PAPER_AAAI.tex
```

Build the faithful-Drifting supplement:

```bash
python scripts/render_faithful_supplement.py
TEXINPUTS=docs//: BSTINPUTS=docs//: latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=DriftingMol_AAAI_FaithfulSupplement docs/SUPPLEMENT_FAITHFUL_DRIFTING_AAAI.tex
```

Clean only generated build products if needed:

```bash
latexmk -C -jobname=DriftingMol_AAAI docs/PAPER_AAAI.tex
```

## Current Gating Items

For the stable 9-page main paper, there are no remaining experiment gates. The
main publication queue is complete:
75/75 experiments are collected, the inference benchmark exists, and
`python scripts/audit_publication_completion.py --run-tests` reports PASS.
The reviewer-faithful Drifting supplement is now complete for the strict core
and allocation sweep. VAE sensitivity, downstream alternative-VAE drifting,
destructive ablations, the three-seed fixed linear property-guidance baseline,
next-wave QED controls, reviewer-extra replicates, and the full graph stress
route are also complete. The graph route is summarized as representation-stress
evidence rather than as a competing graph-generation claim.

Completed audit checkpoints:

- `pub_P1_paper_tau_batch_lambda_qed_s42`: complete/pass, best `alpha=5.0`,
  QED `rho=0.295`, uniqueness `98.2%`, MAE `0.206`, slope `0.450`. This is
  below the decoder-coupled top tier and is currently treated as a negative
  audit control rather than a replacement result.
- `pub_P2_paper_tau_fixed_lambda_qed_s42`: complete/pass, best `alpha=2.0`,
  QED `rho=0.298`, uniqueness `98.5%`, MAE `0.208`, slope `0.450`. This
  matches P1's negative-control pattern, so the audit evidence still supports
  the decoder-coupled mechanism claim rather than a temperature/lambda
  bookkeeping explanation.
- `pub_P3_no_cfg_qed_s42`: complete/pass, best `alpha=1.5`, QED `rho=0.196`,
  uniqueness `98.5%`, MAE `0.207`, slope `0.289`. Removing CFG weakens control
  while preserving diversity, supporting the guidance part of the mechanism.
- `pub_P4_no_zdiv_qed_s42`: complete/pass, best `alpha=5.0`, QED `rho=0.306`,
  uniqueness `77.8%`, MAE `0.196`, slope `0.477`.
- `pub_P5_y_only_norm_qed_s42`: complete/pass, best `alpha=5.0`, QED
  `rho=0.377`, uniqueness `98.1%`, MAE `0.210`, slope `0.564`.
- `pub_P6_no_cross_norm_qed_s42`: complete/pass, best `alpha=3.0`, QED
  `rho=-0.001`, uniqueness `89.9%`, MAE `0.515`, slope `-0.004`.

Completed z-diversity checkpoints for the balanced multi-layer setting:

- `pub_G4_qed_zdiv0p0_s42`: complete/pass, best `alpha=5.0`, QED `rho=0.434`,
  uniqueness `96.1%`, MAE `0.168`, slope `0.690`.
- `pub_G4_qed_zdiv0p5_s42`: complete/pass, best `alpha=5.0`, QED `rho=0.454`,
  uniqueness `98.1%`, MAE `0.175`, slope `0.689`.
- `pub_G4_qed_zdiv1p0_s42`: complete/pass, best `alpha=3.0`, QED `rho=0.433`,
  uniqueness `98.1%`, MAE `0.172`, slope `0.654`.
- `pub_G4_qed_zdiv2p0_s42`: complete/pass, best `alpha=5.0`, QED `rho=0.438`,
  uniqueness `98.2%`, MAE `0.173`, slope `0.663`.
- `pub_G4_qed_zdiv4p0_s42`: complete/pass, best `alpha=3.0`, QED `rho=0.456`,
  uniqueness `98.1%`, MAE `0.172`, slope `0.698`.

Fair balanced multi-layer multi4:

- `pub_G4_multi4_v2_s42`: complete/pass, best `alpha=5.0`, average rho
  `0.474`, QED `0.252`, SA `0.378`, LogP `0.561`, MolWt `0.705`, min
  uniqueness `98.6%`.

Benchmark:

- `results/inference_benchmark.json`: 20,000 samples, 1 NFE, generator
  `47,533` latents/s, end-to-end `4,604` molecules/s on an idle RTX 4090D.

Final refresh command sequence:

```bash
python scripts/collect_results.py
python scripts/export_latex_tables.py
python scripts/plot_result_figures.py
python scripts/benchmark_inference.py
python scripts/update_manuscript_benchmark.py
python scripts/audit_publication_completion.py --run-tests
```

The AAAI text should keep validity framed carefully: SELFIES gives validity by
construction, so validity is a representation guarantee; the paper's empirical
claim is decoder-coupled drift improving property control while preserving
diversity and one-forward-pass inference.
