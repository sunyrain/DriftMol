# Reviewer Prompt-to-Artifact Checklist

Updated: 2026-05-15 UTC

This checklist maps the current reviewer-facing objective to concrete
artifacts and gates. It is intentionally conservative: planned configs,
passing script audits, or active jobs do not count as completed experiments
until final metrics are collected.

## Objective

Plan and organize experiments that answer likely reviewer questions, especially
whether DriftingMol faithfully reproduces the original Drifting Models
algorithm before adding molecule-specific decoder coupling.

## Success Criteria

| Requirement from objective | Concrete artifact / command | Evidence status | Remaining gap |
|---|---|---|---|
| Comprehensive reviewer experiment planning | `docs/REVIEWER_EXPERIMENT_MATRIX.md`, `docs/PUBLICATION_PLAN.md`, `docs/REVIEWER_FEEDBACK_SYNTHESIS.md`, `docs/REVIEWER_GOAL_COMPLETION_AUDIT.md` | PASS | Keep synchronized with new completed results. |
| Explicit proof route for faithful Drifting reproduction | `docs/DRIFTING_FAITHFULNESS_PLAN.md` | PASS | Promote claims conservatively; strict core and allocation are complete, but measured target tracking is modest. |
| Equation-to-code audit for the original drift algorithm | `docs/DRIFTING_ALGORITHM_AUDIT.md`, `scripts/audit_drifting_faithfulness.py` | PASS for tensor equivalence | This proves the drift-field implementation path, not final experimental behavior. |
| Strict latent-feature Drifting experiments | `configs/reviewer_faithful/core/*.yaml`; `results/faithful_drifting.csv` | PASS | 4/4 core metrics collected; strongest strict core is ZSPACE with rho `0.127`, U `97.8%`, MAE `0.221`. |
| Feature extractor controls | `rf_FD_STRICT_PLAIN_PHI_QED_s42`, `rf_FD_STRICT_PROP_PHI_QED_s42`, `rf_FD_STRICT_RANDOM_PHI_QED_s42`, `rf_FD_STRICT_ZSPACE_QED_s42` | PASS | Core controls are complete; interpret as modest but concrete faithful-Drifting evidence. |
| Positive/negative sample allocation tests analogous to the original paper | `configs/reviewer_faithful/allocation/*.yaml`, `outputs/reviewer_faithful/allocation_status.json`, `results/tables/tab_faithful_drifting_allocation.tex` | PASS | 6/6 complete; allocation recovers diversity but remains weak in QED tracking, with the best negative-allocation rows at rho about `0.076`. |
| Anti-symmetry destructive evidence | `results/destructive_ablation.csv`, `results/tables/tab_destructive_ablation.tex` | PASS | 7/7 complete; use as supplemental mechanism evidence. |
| SELFIES validity limitation and VAE sensitivity | `configs/publication_ext/vae_sensitivity/*.yaml`, `results/vae_sensitivity_status.json`, `results/tables/tab_vae_sensitivity.tex`, `docs/GRAPH_STRESS_TEST.md`, `docs/GRAPH_STRESS_EXECUTION_PLAN.md`, `docs/GRAPH_NAMESPACE_ADAPTER_PLAN.md`, `configs/publication_ext/graph_stress_manifest.json`, `results/graph_stress_full_status.json`, `results/tables/tab_graph_raw_vs_repaired.tex` | PASS | VAE sensitivity is 4/4 complete: Low-beta exact recon `96.9%`, High-beta `18.4%`, DEC6 `56.5%`, Latent-128 `94.3%`; all prior VUN values are at least `0.980`. Full graph stress is complete as representation-stress evidence, including fresh graph QED/LogP drifting, no-drift graph ablation, raw-vs-repaired validity, and graph-vs-SELFIES comparison. |
| Downstream drifting under alternative VAEs | `configs/publication_ext/vae_drift_manifest.json`, `configs/publication_ext/vae_drift/*.yaml`, `scripts/collect_vae_drift_results.py`, `scripts/collect_vae_drift_live_snapshot.py`, `results/vae_drift_downstream_status.json`, `results/tables/tab_vae_drift_downstream.tex` | PASS | 4/4 complete. Final QED rho values: low-beta `0.437`, high-beta `0.282`, latent-128 `0.421`, DEC6 `0.272`. This closes the alternative-VAE downstream check; `results/vae_drift_live_snapshot.json` remains monitoring-only. |
| Additional trained baseline for reviewer comparison | `configs/publication_ext/baselines/ext_B_LINEAR_PROP_QED_s4*.yaml`, `configs/publication_ext/baseline_manifest.json`, `scripts/collect_trained_baselines.py`, `results/trained_baseline_status.json`, `results/tables/tab_trained_baseline_qed.tex`, `docs/TRAINED_BASELINE_EXECUTION_CHECKLIST.md` | PASS | Linear latent-property guidance is complete across seeds 42/43/44, with mean QED rho `0.046 +/- 0.150`, U `92.9% +/- 2.6%`, MAE `0.335 +/- 0.096`; it is weak evidence for fixed-guidance control, not a replacement for drift. |
| Same-backbone generative-model baselines | `scripts/train_latent_generative_baseline.py`, `docs/GEN_MODEL_BASELINE_PLAN.md`, `configs/publication_ext/generative_baselines_manifest.json`, `outputs/publication_ext/parallel_runner_status_generative_baselines.json`, `results/generative_baselines_qed.json`, `results/tables/tab_generative_baselines_qed.tex` | PASS | CVAE, WGAN-GP, DDPM, and Flow Matching are complete for seeds 42/43/44. Required final artifacts are `results/generative_baselines_qed.json`, `results/tables/tab_generative_baselines_qed.tex`, and 12 `outputs/publication_ext/generative_baselines/*/final_metrics.json` files. |
| Generalization beyond QED | `configs/publication_ext/generalization_manifest.json`, `scripts/collect_generalization_results.py`, `results/generalization_status.json`, `results/tables/tab_generalization.tex`, `outputs/publication_ext/generalization_postprocess.pid` | PASS | 4/4 complete; LogP is complete/pass with rho `0.639`, both multi4 seeds are complete/pass with mean `0.486`, and SA-score is complete/pass with rho `0.455`. |
| Reviewer-extra bin and seed robustness | `configs/publication_ext/reviewer_extra_manifest.json`, `configs/publication_ext/reviewer_extra/*.yaml`, `scripts/collect_reviewer_extra_results.py`, `scripts/watch_reviewer_extra_postprocess.py`, `results/reviewer_extra_status.json`, `results/tables/tab_reviewer_extra.tex` | PASS | 4/4 complete/pass. The table records continuous-QED seed 42 rho `0.344`, continuous-QED seed 43 rho `0.340`, LogP seed 43 rho `0.634`, SA-score seed 43 rho `0.440`, and low-beta VAE-drift seed 43 QED rho `0.429`. |
| Next-wave property-transfer baselines | `docs/NEXT_WAVE_EXPERIMENT_PLAN.md`, `configs/publication_ext/next_wave_manifest.json`, `configs/publication_ext/next_wave/*.yaml`, `scripts/generate_next_wave_configs.py`, `scripts/collect_next_wave_results.py`, `results/next_wave_status.json`, `results/tables/tab_next_wave.tex` | PASS | 4/4 complete/pass, including continuous-QED seed 43 and the fixed linear property-guidance controls. |
| Disk and checkpoint safety for continued reviewer runs | `scripts/report_checkpoint_cleanup_candidates.py`, `results/checkpoint_cleanup_deleted.md`, `results/checkpoint_cleanup_candidates.md` | PASS | 45 completed-run `last.pt` checkpoints have been deleted only after confirming sibling `best.pt` and valid `final_metrics.json`, while all active output directories were excluded. Current follow-up report has 0 reclaimable candidates and `/root/autodl-tmp` has about 18G free. |
| Launch automation for strict faithful runs | `scripts/defer_faithful_core_after_destructive.py`, `outputs/reviewer_faithful/deferred_faithful_core_launcher.pid`, `outputs/reviewer_faithful/core_status.json`, `tests/test_defer_faithful_core.py` | COMPLETED + TESTED | Automatic launch succeeded and `core_status.json` is `completed`; launcher exit is now treated as normal after core completion. |
| Safe long-run re-scheduling without duplicate outputs | `scripts/run_manifest_parallel.py`, `tests/test_run_manifest_parallel.py` | PASS | Applies to future launches; already-running runners keep their in-memory queue. |
| Result collection and LaTeX table export | `scripts/collect_faithful_drifting_results.py`, `scripts/render_faithful_supplement.py`, `docs/SUPPLEMENT_FAITHFUL_DRIFTING.tex`, `docs/SUPPLEMENT_FAITHFUL_DRIFTING_INLINED.tex`, `docs/SUPPLEMENT_FAITHFUL_DRIFTING_AAAI.tex`, `DriftingMol_AAAI_FaithfulSupplement.pdf`, `results/tables/tab_faithful_drifting_core.tex`, `results/tables/tab_faithful_drifting_allocation.tex` | PASS for faithful package | Faithful core and allocation tables are populated and now packaged as a standalone AAAI supplement PDF; downstream extension tables are still refreshed by their own watchers. |
| Completion verifier | `scripts/audit_reviewer_experiment_readiness.py`, `scripts/audit_drifting_faithfulness.py`, `scripts/audit_graph_archive_launchability.py`, `scripts/audit_extension_completion.py --strict` | PASS | Faithfulness, graph launchability, reviewer readiness, and strict extension completion all report PASS. |
| Regression safety | `python -m unittest discover -s tests` | PASS | Last run: 95 tests passed through `python scripts/audit_publication_completion.py --run-tests`. |

## Current Blocking Items

1. Same-backbone generative baselines are complete; use them for the full
   generative-model comparison.
2. Future optional external public graph generators can be added later, but
   they should be separated from the same-backbone quantitative comparison
   unless the exact target-bin protocol is reproduced.

## Commands

Monitor current jobs:

```bash
python scripts/monitor_extension.py
```

Refresh result tables and faithfulness status:

```bash
python scripts/collect_extension_results.py
python scripts/collect_vae_drift_results.py
python scripts/collect_reviewer_extra_results.py
python scripts/collect_next_wave_results.py
python scripts/collect_faithful_drifting_results.py
python scripts/audit_drifting_faithfulness.py
python scripts/audit_reviewer_experiment_readiness.py
```

Final completion gate:

```bash
python scripts/audit_drifting_faithfulness.py
python scripts/audit_extension_completion.py --strict
python scripts/audit_reviewer_experiment_readiness.py
```

All three must report a closed/pass state before the reviewer-facing
faithfulness package is treated as complete.
