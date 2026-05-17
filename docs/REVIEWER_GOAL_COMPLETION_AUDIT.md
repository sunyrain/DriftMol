# Reviewer Goal Completion Audit

Updated: 2026-05-15 UTC

Objective: plan and organize reviewer-facing experiments, especially evidence
that the Drifting algorithm reproduction is real and reliable before relying on
DriftingMol-specific decoder coupling.

## Deliverables And Evidence

| Deliverable | Evidence | Status | Notes |
|---|---|---|---|
| Comprehensive reviewer experiment plan | `docs/REVIEWER_EXPERIMENT_MATRIX.md`, `docs/PUBLICATION_PLAN.md`, `docs/REVIEWER_FEEDBACK_SYNTHESIS.md` | PASS | Maps likely reviewer objections to concrete artifacts and claim rules. |
| Prompt-to-artifact checklist | `docs/REVIEWER_PROMPT_TO_ARTIFACT_CHECKLIST.md` | PASS | Tracks faithful reproduction, VAE sensitivity, graph stress, downstream VAE-drift, trained baselines, and checkpoint safety status. |
| Equation-to-code audit for Drifting Algorithm 2 | `docs/DRIFTING_ALGORITHM_AUDIT.md`, `scripts/audit_drifting_faithfulness.py`, `results/drifting_faithfulness_status.json` | PASS | Tensor equivalence reports `max_abs_diff=0.0`. |
| Strict faithful latent-feature Drifting runs | `configs/reviewer_faithful/core/*.yaml`, `results/tables/tab_faithful_drifting_core.tex` | PASS | 4/4 complete: plain phi, property-aware phi, random phi, and z-space controls. |
| Original-paper-style sample allocation checks | `configs/reviewer_faithful/allocation/*.yaml`, `results/tables/tab_faithful_drifting_allocation.tex` | PASS | 6/6 complete; allocation improves diversity but not strong QED target tracking. |
| Faithful reproduction audit gate | `python scripts/audit_drifting_faithfulness.py` | PASS | Overall faithful Drifting audit is closed for the 10-run strict package. |
| Anti-symmetry and normalization mechanism evidence | `results/destructive_ablation.csv`, `results/tables/tab_destructive_ablation.tex` | PASS | 7/7 destructive drift ablations complete. |
| SELFIES validity limitation addressed | `docs/GRAPH_STRESS_TEST.md`, `results/graph_stress_test.json`, `docs/figures/fig_graph_bottleneck.pdf`, `docs/GRAPH_STRESS_EXECUTION_PLAN.md`, `configs/publication_ext/graph_stress_manifest.json`, `results/graph_archive_launchability_status.json`, `results/graph_stress_full_status.json` | PASS | Diagnostic snapshot exists, E36/E40 archived metrics are audited, raw-vs-repaired decoding is summarized, and the fresh graph stress package is complete with QED, LogP, no-drift, raw/repaired validity, and graph-vs-SELFIES artifacts. |
| VAE architecture sensitivity | `configs/publication_ext/vae_sensitivity/*.yaml`, `results/vae_sensitivity_status.json`, `results/tables/tab_vae_sensitivity.tex` | PASS | 4/4 complete; low-beta, high-beta, DEC6, and latent128 all have final reconstruction and prior-sampling metrics. |
| Downstream drifting under alternative VAEs | `configs/publication_ext/vae_drift_manifest.json`, `scripts/collect_vae_drift_results.py`, `results/vae_drift_downstream_status.json` | PASS | 4/4 complete; final QED rho values are low-beta `0.437`, high-beta `0.282`, latent128 `0.421`, and DEC6 `0.272`. |
| Stronger reviewer baseline package | `configs/publication_ext/baseline_manifest.json`, `scripts/collect_trained_baselines.py`, `results/trained_baseline_status.json`, `results/tables/tab_trained_baseline_qed.tex` | PASS | 3/3 complete; three-seed fixed linear guidance mean QED rho is `0.046 +/- 0.150`. |
| Generalization beyond QED | `configs/publication_ext/generalization_manifest.json`, `scripts/collect_generalization_results.py`, `results/generalization_status.json`, `results/tables/tab_generalization.tex` | PASS | 4/4 complete; LogP and both multi4 seeds are complete/pass, and SA-score is complete/pass. |
| Extra reviewer robustness queue | `configs/publication_ext/reviewer_extra_manifest.json`, `scripts/collect_reviewer_extra_results.py`, `results/reviewer_extra_status.json`, `results/tables/tab_reviewer_extra.tex` | PASS | 4/4 complete/pass; continuous QED, LogP seed 43, SA-score seed 43, and low-beta VAE-drift seed 43 all have final metrics. |
| Next-wave baseline planning and launch | `docs/NEXT_WAVE_EXPERIMENT_PLAN.md`, `configs/publication_ext/next_wave_manifest.json`, `scripts/collect_next_wave_results.py`, `results/next_wave_status.json`, `results/tables/tab_next_wave.tex`, `outputs/publication_ext/parallel_runner_status_next_wave_gpu0.json` | PASS | 4/4 complete/pass, including continuous-QED seed 43 and the fixed linear LogP, SA-score, and multi-property guidance controls. |
| Checkpoint and disk safety | `scripts/report_checkpoint_cleanup_candidates.py`, `results/checkpoint_cleanup_deleted.md`, `results/checkpoint_cleanup_candidates.md` | PASS | Completed-run `last.pt` checkpoints have been deleted only after confirming sibling `best.pt` and valid `final_metrics.json`, excluding active output directories; the latest deletion log records 53 files reclaimed and about 13G free. |
| Baseline and extension postprocess automation | `outputs/publication_ext/vae_drift_postprocess.pid`, `outputs/publication_ext/generalization_postprocess.pid`, `outputs/publication_ext/reviewer_extra_postprocess.pid` | PASS | Watchers are alive and will refresh tables/audits as final metrics appear. |
| Regression safety | `python -m unittest discover -s tests` | PASS | Last full run: 95 tests OK. |

## Current Blocking Items

No reviewer-facing experiment blocker remains under the current objective:
`results/reviewer_extra_status.json`, `results/next_wave_status.json`,
`results/extension_completion_status.json`, and
`results/generative_baselines_status.json` all report complete/pass states.

## Current Claim Boundary

It is now supported to claim that the strict Drifting reproduction was
implemented, audited, and executed in molecular latent space. The broader
reviewer-facing extension package is complete; the paper should still avoid
claiming external molecular-generation SOTA beyond the reproduced protocol.
