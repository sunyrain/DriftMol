# Objective Audit

Updated: 2026-05-15 UTC

Objective: plan reviewer-facing experiments and prove the Drifting
reproduction is real and reliable, with faithful latent-space drifting
experiments as supplementary evidence.

## Success Criteria

| Criterion | Evidence | Status |
|---|---|---|
| Faithful latent-feature Drifting reproduction exists and is audited | [docs/DRIFTING_FAITHFULNESS_PLAN.md](/root/autodl-tmp/DriftingMol/docs/DRIFTING_FAITHFULNESS_PLAN.md), [results/faithful_drifting_status.json](/root/autodl-tmp/DriftingMol/results/faithful_drifting_status.json), [results/tables/tab_faithful_drifting_core.tex](/root/autodl-tmp/DriftingMol/results/tables/tab_faithful_drifting_core.tex) | PASS |
| Sample-allocation variants of the faithful reproduction are complete | [results/tables/tab_faithful_drifting_allocation.tex](/root/autodl-tmp/DriftingMol/results/tables/tab_faithful_drifting_allocation.tex) | PASS |
| Mechanism ablations show the drift objective matters | [results/destructive_ablation_status.json](/root/autodl-tmp/DriftingMol/results/destructive_ablation_status.json), [results/tables/tab_destructive_ablation.tex](/root/autodl-tmp/DriftingMol/results/tables/tab_destructive_ablation.tex) | PASS |
| VAE sensitivity is tested | [results/vae_sensitivity_status.json](/root/autodl-tmp/DriftingMol/results/vae_sensitivity_status.json), [results/tables/tab_vae_sensitivity.tex](/root/autodl-tmp/DriftingMol/results/tables/tab_vae_sensitivity.tex) | PASS |
| Alternative-VAE downstream drifting is tested | [results/vae_drift_downstream_status.json](/root/autodl-tmp/DriftingMol/results/vae_drift_downstream_status.json), [results/tables/tab_vae_drift_downstream.tex](/root/autodl-tmp/DriftingMol/results/tables/tab_vae_drift_downstream.tex) | PASS |
| Reviewer comparison baselines are complete | [results/trained_baseline_status.json](/root/autodl-tmp/DriftingMol/results/trained_baseline_status.json), [results/tables/tab_trained_baseline_qed.tex](/root/autodl-tmp/DriftingMol/results/tables/tab_trained_baseline_qed.tex), [results/generative_baselines_qed.json](/root/autodl-tmp/DriftingMol/results/generative_baselines_qed.json), [results/tables/tab_generative_baselines_qed.tex](/root/autodl-tmp/DriftingMol/results/tables/tab_generative_baselines_qed.tex) | PASS |
| Reviewer-facing generalization is complete | [results/generalization_status.json](/root/autodl-tmp/DriftingMol/results/generalization_status.json), [results/tables/tab_generalization.tex](/root/autodl-tmp/DriftingMol/results/tables/tab_generalization.tex) | PASS |
| Reviewer-facing robustness extensions are complete | [results/reviewer_extra_status.json](/root/autodl-tmp/DriftingMol/results/reviewer_extra_status.json) | PASS |
| Next-wave baselines are complete | [results/next_wave_status.json](/root/autodl-tmp/DriftingMol/results/next_wave_status.json), [outputs/publication_ext/parallel_runner_status_next_wave_gpu0.json](/root/autodl-tmp/DriftingMol/outputs/publication_ext/parallel_runner_status_next_wave_gpu0.json) | PASS |
| Graph stress is complete with the archive-local namespace adapter and fresh graph-route artifacts | [docs/GRAPH_NAMESPACE_ADAPTER_PLAN.md](/root/autodl-tmp/DriftingMol/docs/GRAPH_NAMESPACE_ADAPTER_PLAN.md), [results/graph_archive_launchability_status.json](/root/autodl-tmp/DriftingMol/results/graph_archive_launchability_status.json), [results/graph_stress_full_status.json](/root/autodl-tmp/DriftingMol/results/graph_stress_full_status.json) | PASS |
| Main AAAI package is ready | [results/publication_completion_audit.md](/root/autodl-tmp/DriftingMol/results/publication_completion_audit.md), [DriftingMol_AAAI.pdf](/root/autodl-tmp/DriftingMol/DriftingMol_AAAI.pdf); audit now covers the AAAI source, figures, tables, authors, bibliography, build files, benchmark claims, and 95 tests | PASS |

## Current Gaps

No current experiment gap remains under this objective. The remaining work is
paper judgment: keep claims scoped to the reproduced protocol and avoid
non-protocol-matched external baseline claims.

## Latest Completed Progress

Reviewer-extra robustness is 4/4 complete, next-wave controls are 4/4 complete,
graph stress is complete, and same-backbone generative baselines are complete
for conditional latent VAE, WGAN-GP, DDPM, and Flow Matching across seeds
42/43/44.

## Current Interpretation

The faithful Drifting reproduction is already supported by completed
experiments and audits.

The broader reviewer-facing extension package is closed under the current audit
gates. The repository should still avoid broad external SOTA claims unless
external public graph/string generators are re-run under their own matched
protocols.
