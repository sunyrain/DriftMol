# DriftingMol Publication Audit

Updated: 2026-05-17 UTC

This audit separates the already-complete AAAI package from the now-complete
reviewer extension evidence pack.

## Audit Scope

### Completed AAAI Package

The current AAAI submission package is complete and internally consistent:

- Main manuscript: `docs/PAPER_AAAI.tex`
- Main PDF: `DriftingMol_AAAI.pdf`
- Reproducibility checklist: `docs/AAAI_REPRODUCIBILITY_CHECKLIST.tex`
- Checklist PDF: `DriftingMol_AAAI_Checklist.pdf`
- Submission bundle: `submission/driftingmol_submission_source.zip`
- Completion audit: `python scripts/audit_publication_completion.py --run-tests`
  reports PASS and covers the AAAI source, final figures, tables, authors,
  bibliography, benchmark claims, and build prerequisites.
- Unit tests: `python -m unittest discover -s tests` reports 95/95 PASS.

### Extension Phase

The reviewer extension phase is complete. It is defined in
`docs/PUBLICATION_PLAN.md` and covers:

- LogP, SA-score, and multi-property generalization,
- reviewer-extra robustness replicates,
- graph representation stress tests,
- next-wave fixed-guidance baselines,
- and same-backbone CVAE/WGAN-GP/DDPM/Flow-Matching generative baselines.

The archived graph-line metrics have been recovered into
`docs/GRAPH_STRESS_TEST.md`, `results/graph_stress_test.json`, and
`results/tables/tab_graph_stress.tex`. This is a diagnostic snapshot. The full
graph stress-test work package is now closed by fresh graph-route artifacts.
The diagnostic bottleneck figure exists at
`docs/figures/fig_graph_bottleneck.pdf`; the full graph stress-test status is
recorded in `results/graph_stress_full_status.json`.

Destructive drifting ablations, VAE architecture sensitivity, downstream
alternative-VAE drifting, strict faithful-Drifting reproduction, allocation
sweeps, the three-seed fixed linear property-guidance baseline, reviewer-extra
robustness, next-wave property-guidance controls, full graph stress, and the
same-backbone generative-model baselines are now complete.
`scripts/audit_extension_completion.py --strict` reports PASS and records the
broader extension status in `results/extension_completion_audit.md`.
Same-backbone generative baselines are complete and are tracked by
`configs/publication_ext/generative_baselines_manifest.json`.

## Checklist

| Item | Evidence | Status |
|---|---|---|
| AAAI manuscript exists and compiles | `DriftingMol_AAAI.pdf` | PASS |
| Reproducibility checklist exists and compiles | `DriftingMol_AAAI_Checklist.pdf` | PASS |
| Publication audit gate passes | `results/publication_completion_audit.md` and audit script output | PASS |
| Unit test suite passes | 95 tests | PASS |
| Current plan is written down | `docs/PUBLICATION_PLAN.md` | PASS |
| Extension completion audit exists | `results/extension_completion_audit.md` | PASS |
| Parallel extension runner exists | `scripts/run_manifest_parallel.py` dry-run writes `outputs/publication_ext/parallel_runner_status.json` | PASS |
| Extension monitor exists | `python scripts/monitor_extension.py` reports extension and GPU status | PASS |
| Minimum protocol-matched baselines are complete | `results/matched_baselines_qed.json` contains retrieval, VAE-jitter, and bin-Gaussian | PASS |
| Trained stronger baselines are complete | three-seed fixed linear latent-property guidance baseline complete | PASS |
| Graph stress diagnostic snapshot exists | `docs/GRAPH_STRESS_TEST.md` and `results/graph_stress_test.json` | PASS |
| Graph bottleneck diagnostic figure exists | `docs/figures/fig_graph_bottleneck.pdf` | PASS |
| Full graph stress test is complete | graph QED, LogP, no-drift, raw/repaired validity, and graph-vs-SELFIES comparison complete | PASS |
| Destructive ablation configs are runnable | `configs/publication_ext/manifest.json` dry-run succeeds | PASS |
| Destructive ablation collector exists | `results/destructive_ablation.csv`, `results/destructive_ablation_status.json`, and `results/tables/tab_destructive_ablation.tex` | PASS |
| Destructive ablations are complete | 7/7 rows complete | PASS |
| VAE sensitivity configs and collector exist | `configs/publication_ext/vae_sensitivity/*.yaml` and `results/vae_sensitivity_status.json` | PASS |
| VAE sensitivity is complete | 4/4 VAE rows complete; downstream VAE-drift 4/4 complete | PASS |
| Generalization queue is complete | 4/4 complete | PASS |
| Reviewer-extra queue is complete | 4/4 complete/pass | PASS |
| Next-wave property-guidance baselines are complete | 4/4 complete/pass | PASS |
| Same-backbone generative baselines are complete | `results/generative_baselines_qed.json`, `results/tables/tab_generative_baselines_qed.tex`, and 12 `outputs/publication_ext/generative_baselines/*/final_metrics.json` files | PASS |

## Current Interpretation

The paper itself is ready as an AAAI submission package, and the reviewer-facing
extension package is closed under the current audit gates. The quantitative
generator-family comparison should use the same-backbone conditional latent VAE,
WGAN-GP, DDPM, and Flow-Matching baselines.
