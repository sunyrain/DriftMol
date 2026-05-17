# Reviewer Experiment Matrix

Updated: 2026-05-15 UTC

This matrix maps likely reviewer objections to concrete experiments and
artifacts. It is a working checklist for moving the paper toward a stronger
AAAI submission without overclaiming.

For a prompt-to-artifact audit of the current objective, see
`docs/REVIEWER_PROMPT_TO_ARTIFACT_CHECKLIST.md`.

Latest experiment refresh (UTC): reviewer-extra, next-wave, and full graph
stress queues are complete. Same-backbone generative baselines for CVAE,
WGAN-GP, DDPM, and Flow Matching are complete.

| Reviewer objection | Experiment / evidence | Artifact | Current status | Claim rule |
|---|---|---|---|---|
| The method may not faithfully reproduce Drifting Models. | Algorithm-2 tensor audit comparing implementation to direct pseudocode transcription plus equation-to-code mapping. | `docs/DRIFTING_ALGORITHM_AUDIT.md`, `scripts/audit_drifting_faithfulness.py`, `results/drifting_faithfulness_status.json` | PASS; strict core 4/4 and allocation 6/6 complete | We can claim the drift-field computation matches Algorithm 2 and that the strict molecular reproduction package is complete, while interpreting the measured control signal conservatively. |
| The molecular result may be a decoder heuristic, not Drifting. | Strict latent-MAE phi run with generated negatives, QED bins, `Nc=64`, `Npos=64`, `Nneg=64`, `tau={0.02,0.05,0.2}`, no z-diversity. | `results/tables/tab_faithful_drifting_core.tex`, `configs/reviewer_faithful/core/rf_FD_STRICT_PLAIN_PHI_QED_s42.yaml` | COMPLETE, modest signal | Present direct faithful reproduction as measurable but weak; reserve stronger claims for decoder-coupled DriftingMol. |
| Feature extractors may not matter. | Random-phi and z-space controls under the same strict protocol. | `results/tables/tab_faithful_drifting_core.tex` | COMPLETE | Random phi and z-space are similar to or stronger than plain phi; weaken feature-space superiority claims. |
| The original paper reports feature-quality sensitivity; molecules should too. | Plain latent-MAE phi versus property-aware latent-MAE phi. | `results/tables/tab_faithful_drifting_core.tex` | COMPLETE | Property-aware phi improves over plain phi but remains modest; emphasize representation limits and decoder alignment. |
| Positive/negative sample estimation may differ from original Drifting. | Table-2-style allocation sweep over `Npos` and `Nneg` at fixed effective batch. | `configs/reviewer_faithful/allocation/*.yaml`, `outputs/reviewer_faithful/allocation_status.json`, `results/tables/tab_faithful_drifting_allocation.tex` | COMPLETE; 6/6 rows complete | Allocation improves diversity but not strong target tracking, so describe the direct molecule port as representation-limited. |
| Anti-symmetry may not be necessary. | Destructive attraction/repulsion and normalization ablations. | `results/destructive_ablation.csv`, `results/tables/tab_destructive_ablation.tex` | COMPLETE: 7/7 | Use as supplemental mechanism evidence; distinguish expected failures from pass/fail quality gates. |
| SELFIES makes validity trivial. | Graph representation stress test and limitations wording. | `docs/GRAPH_STRESS_TEST.md`, `docs/GRAPH_STRESS_EXECUTION_PLAN.md`, `docs/GRAPH_NAMESPACE_ADAPTER_PLAN.md`, `configs/publication_ext/graph_stress_manifest.json`, `results/graph_stress_test.json`, `results/graph_stress_full_status.json`, `results/graph_archive_launchability_status.json` | COMPLETE as representation-stress evidence; graph VAE, graph latent cache, graph Latent-MAE, fresh graph QED/LogP drifting, no-drift graph ablation, raw-vs-repaired validity, and graph-vs-SELFIES comparison are complete | State validity is inherited from SELFIES; graph decoding remains a representation bottleneck, so graph results are used as stress evidence rather than a competing graph-generator claim. |
| Results may depend on one lucky VAE. | VAE beta, latent-size, and decoder-depth sensitivity, followed by downstream QED drifting on the alternative VAE checkpoints. | `configs/publication_ext/vae_sensitivity/*.yaml`, `configs/publication_ext/vae_drift_manifest.json`, `results/vae_sensitivity_status.json`, `results/vae_drift_downstream_status.json` | COMPLETE; VAE sensitivity 4/4 and downstream VAE-drift 4/4 complete | Use as architecture-robustness evidence: low-beta and latent-128 preserve moderate QED control, while high-beta and decoder-6 expose sensitivity to reconstruction/decoder geometry. |
| Baselines are too weak. | Retrieval, VAE jitter, bin Gaussian, three-seed fixed linear latent-property guidance, and same-backbone CVAE/WGAN-GP/DDPM/Flow-Matching generative baselines. | `results/matched_baselines_qed.json`, `configs/publication_ext/baseline_manifest.json`, `results/trained_baseline_status.json`, `results/tables/tab_trained_baseline_qed.tex`, `docs/TRAINED_BASELINE_EXECUTION_CHECKLIST.md`, `docs/GEN_MODEL_BASELINE_PLAN.md`, `configs/publication_ext/generative_baselines_manifest.json`, `scripts/train_latent_generative_baseline.py` | COMPLETE; lightweight references, fixed linear guidance, and same-backbone generative baselines are complete | Use the trained linear baseline to show a simple property head does not replace drift. Use the same-backbone generative baselines as the real generator-family comparison. |
| Property-transfer gains may be explainable by a simple property head. | Next-wave fixed linear latent-property baselines for LogP, SA-score, and four-property no-binning control. | `docs/NEXT_WAVE_EXPERIMENT_PLAN.md`, `configs/publication_ext/next_wave_manifest.json`, `scripts/collect_next_wave_results.py`, `results/next_wave_status.json`, `results/tables/tab_next_wave.tex` | COMPLETE; 4/4 pass | Use as extra reviewer evidence that fixed property-guidance controls do not explain the drift results. |
| Results may be QED-specific. | Extra four-property no-binning seeds plus LogP and SA-score single-property controls. | `configs/publication_ext/generalization_manifest.json`, `results/generalization_status.json`, `results/tables/tab_generalization.tex` | PASS; LogP is complete/pass with rho `0.639`, both extra multi-property seeds are complete/pass with mean `0.486`, and SA-score is complete/pass with rho `0.455` | Use as broader molecular-property transfer evidence; if mixed, still present QED as the strongest validated target, but not the only validated target. |
| QED control may be an artifact of discretized target bins. | Continuous-QED control with the same balanced multi-layer protocol but quantile binning disabled. | `configs/publication_ext/reviewer_extra_manifest.json`, `results/reviewer_extra_status.json`, `results/tables/tab_reviewer_extra.tex` | COMPLETE; continuous QED seed 42 rho `0.344`, seed 43 rho `0.340` | Present binning as a validated protocol choice, while continuous controls show the signal does not disappear without target bins. |
| Property-transfer and VAE-robustness evidence may be single-seed. | Second LogP and SA-score transfer seeds plus a second low-beta downstream VAE-drift seed. | `configs/publication_ext/reviewer_extra/*.yaml`, `results/reviewer_extra_status.json`, `results/tables/tab_reviewer_extra.tex` | COMPLETE; 4/4 pass | Use as secondary robustness evidence for property transfer and alternative-VAE downstream drifting. |
| Figures look like an internal report. | Use an image-2 raster conceptual figure and regenerate quantitative figures with clean labels. | `docs/figures/*`, `scripts/plot_result_figures.py`, `results/pdf_preview/page_*.png`, `results/tables/tab_graph_raw_vs_repaired.tex` | PASS for the current 8-page draft; raw-vs-repaired decoding is now summarized alongside the diagnostic graph figures | Keep main text figures sparse; move dense details to supplement. |

## Launch Priority

1. Keep the main paper concise and leave dense baseline evidence in the
   reviewer artifact package.
2. Refresh the submission bundle whenever reviewer-facing result artifacts are
   added or regenerated.

## Completion Gate

The reviewer-risk package is not complete until:

1. `python scripts/audit_drifting_faithfulness.py` reports `PASS`.
2. `python scripts/audit_extension_completion.py --strict` reports `PASS`.
3. The final paper or supplement maps each accepted claim to a completed
   artifact in this matrix.
