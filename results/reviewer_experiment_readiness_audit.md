# Reviewer Experiment Readiness Audit

Objective: verify the planning and execution state for reviewer-facing
experiments, especially faithful reproduction of Drifting Models.

| Requirement | Evidence | Status | Note |
|---|---|---|---|
| Comprehensive reviewer experiment plan exists | `docs/PUBLICATION_PLAN.md; docs/REVIEWER_EXPERIMENT_MATRIX.md; docs/REVIEWER_GOAL_COMPLETION_AUDIT.md` | PASS |  |
| Faithful Drifting plan exists | `docs/DRIFTING_FAITHFULNESS_PLAN.md` | PASS |  |
| Prompt-to-artifact checklist exists | `docs/REVIEWER_PROMPT_TO_ARTIFACT_CHECKLIST.md` | PASS |  |
| Equation-to-code algorithm audit exists | `docs/DRIFTING_ALGORITHM_AUDIT.md` | PASS |  |
| Algorithm 2 tensor equivalence passes | `results/drifting_faithfulness_status.json` | PASS | max_abs_diff=0.0 |
| Strict reviewer-faithful config protocol passes | `results/drifting_faithfulness_status.json` | PASS | checked=10, failures=[] |
| Reviewer-faithful manifest has strict core and allocation configs | `configs/reviewer_faithful/manifest.json` | PASS | entries=10, groups={'faithful_core': 4, 'faithful_allocation': 6}, missing_configs=[] |
| Faithful result collection artifacts exist | `results/faithful_drifting.csv; results/tables/tab_faithful_drifting_core.tex; docs/SUPPLEMENT_FAITHFUL_DRIFTING_AAAI.tex; DriftingMol_AAAI_FaithfulSupplement.pdf` | PASS |  |
| Strict faithful core runs are complete | `results/faithful_drifting_status.json` | PASS | complete=4/4 |
| Strict faithful allocation sweeps are complete | `results/faithful_drifting_status.json; results/tables/tab_faithful_drifting_allocation.tex` | PASS | complete=6/6 |
| Deferred faithful launcher completed or is active | `outputs/reviewer_faithful/deferred_faithful_core_launcher.pid; outputs/reviewer_faithful/core_status.json` | PASS | pid_alive=False, core_state=completed, core_complete=True |
| Destructive anti-symmetry evidence has minimum completed runs | `results/destructive_ablation_status.json` | PASS | complete=7/7 |
| VAE sensitivity has at least one completed alternative | `results/vae_sensitivity_status.json` | PASS | complete=4/4 |
| Graph archive launchability preflight is recorded | `scripts/audit_graph_archive_launchability.py; results/graph_archive_launchability_status.json; results/graph_archive_launchability_audit.md` | PASS | complete=True, missing_artifacts=0, blockers=0, archived_metrics=2, graph_cache=True, legacy_latent_caches=2 |
| Graph stress prepared manifest exists | `configs/publication_ext/graph_stress_manifest.json; docs/GRAPH_NAMESPACE_ADAPTER_PLAN.md` | PASS | entries=9, preconditions=5, launch_now=True, namespace_plan=True |
| Downstream VAE-drift queue and collector are organized | `configs/publication_ext/vae_drift_manifest.json; scripts/collect_vae_drift_results.py; results/vae_drift_downstream_status.json` | PASS | entries=4, complete=4/4, launchers_alive=0, automation_present=True, postprocess_ok=True, missing_configs=[] |
| Generalization queue and collector are organized | `configs/publication_ext/generalization_manifest.json; scripts/collect_generalization_results.py; results/generalization_status.json` | PASS | entries=4, complete=4/4, launchers_alive=0, postprocess_ok=True, missing_configs=[] |
| Reviewer-extra queue and collector are organized | `configs/publication_ext/reviewer_extra_manifest.json; scripts/collect_reviewer_extra_results.py; results/reviewer_extra_status.json` | PASS | entries=4, complete=4/4, launchers_alive=0, postprocess_ok=True, missing_configs=[] |
| Next-wave reviewer experiments are prepared or running | `docs/NEXT_WAVE_EXPERIMENT_PLAN.md; configs/publication_ext/next_wave_manifest.json; scripts/collect_next_wave_results.py; results/next_wave_status.json; outputs/publication_ext/parallel_runner_status_next_wave_gpu0.json` | PASS | entries=4, complete=4/4, runner_state=completed, missing_configs=[] |
| Trained baseline queue and collector are organized | `configs/publication_ext/baseline_manifest.json; scripts/collect_trained_baselines.py; results/trained_baseline_status.json` | PASS | entries=3, complete=3/3, missing_configs=[] |
| Overall extension audit is closed | `results/extension_completion_status.json` | PASS | complete=True |

Overall: PASS
