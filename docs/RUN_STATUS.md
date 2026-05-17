# Run Status

Updated: 2026-05-17 UTC; live log check 2026-05-17

This file separates the stable main-paper run state from the completed AAAI
extension evidence pack.

## Stable Main-Paper Package

The main 9-page AAAI package is complete and audited:

- Main manuscript: `docs/PAPER_AAAI.tex`
- Main PDF: `DriftingMol_AAAI.pdf`
- Reproducibility checklist: `docs/AAAI_REPRODUCIBILITY_CHECKLIST.tex`
- Checklist PDF: `DriftingMol_AAAI_Checklist.pdf`
- Faithful-Drifting supplement source: `docs/SUPPLEMENT_FAITHFUL_DRIFTING_AAAI.tex`
- Faithful-Drifting supplement PDF: `DriftingMol_AAAI_FaithfulSupplement.pdf`
- Clean submission/source bundle: `submission/driftingmol_submission_source.zip`
- Main audit gate: `python scripts/audit_publication_completion.py --run-tests`
  reports PASS and now explicitly checks the AAAI source, final figure/table
  references, author metadata, AAAI style files, references, benchmark
  throughput markers, and the 95-test suite.
- The main PDF and clean submission/source bundle were refreshed on 2026-05-17 after
  the expert-review language pass, citation audit, title update, and
  AAAI-source audit pass.

The main results already collected include:

- QED conditional generation on ZINC250K.
- 15-condition QED mechanism ablation.
- Three-seed QED stability for key settings.
- z-diversity sweep for the layer-balanced decoder-drift setting.
- Four-property v2 no-binning evaluation.
- Inference benchmark on an idle RTX 4090D.

## Extension Queue

The extension phase is complete under the current reviewer-facing gates. The
current extension and reviewer readiness audits report:

- Overall extension completion: PASS
- Destructive drift ablations: 7/7 complete, 0 running, 0 queued
- VAE sensitivity: 4/4 complete; BETA_LOW, BETA_HIGH, DEC6, and LATENT128 all
  have final reconstruction and prior-sampling metrics
- Downstream VAE drifting: 4/4 complete; low-beta, high-beta, latent128, and
  DEC6 all have final QED metrics
- Generalization queue: 4/4 complete; LogP is complete/pass with rho `0.639`,
  both multi4 seeds are complete/pass with mean `0.486`, and SA-score is
  complete/pass with rho `0.455`
- Reviewer-extra queue: 4/4 complete/pass. The synchronized table records
  continuous QED seed 42 rho `0.344`, LogP seed 43 rho `0.634`, SA-score seed
  43 rho `0.440`, and low-beta VAE-drift seed 43 QED rho `0.429`.
- Next-wave reviewer experiments: 4/4 complete/pass. The refreshed
  `results/tables/tab_next_wave.tex` includes continuous-QED seed 43 with
  final alpha `3.0`, QED rho `0.340`, U `98.7%`, and MAE `0.208`, plus the
  three completed linear property-guidance baselines.
- Strict reviewer-faithful core: 4/4 complete after automatic deferred launch
- Strict reviewer-faithful allocation sweeps: 6/6 complete
- Reviewer-faithful protocol audit: PASS for Algorithm 2 tensor equivalence,
  strict config semantics, manifest grouping, artifact presence, and all 10
  faithful reproduction runs
- Full graph stress package: COMPLETE. The graph VAE checkpoint, graph cache,
  graph latent cache, graph Latent-MAE checkpoint, fresh graph QED, fresh graph
  LogP, no-drift graph ablation, raw-vs-repaired validity, and graph-vs-SELFIES
  comparison all finished with no active failures.
- Protocol-matched lightweight baselines: PASS for three existing references
- Trained property-guidance baseline: seeds 42, 43, and 44 are complete; the
  three-seed baseline has mean QED rho `0.046 +/- 0.150`
- Same-backbone generative baselines: conditional latent VAE, WGAN-GP, DDPM,
  and Flow Matching are complete for seeds 42/43/44. These are the proper
  generator-family baselines.
- Current compute availability: all four GPUs are idle after the next-wave and
  graph-route queues completed. Disk has about `13G` free on
  `/root/autodl-tmp` after confirmed-reclaimable completed-run `last.pt`
  checkpoint cleanup.
- Latest completed 30-minute monitor log:
  `outputs/publication_ext/monitor_logs/monitor_30min_20260514T213644Z.log`.
  The current 30-minute monitor was launched after that completed cycle.
- Current 30-minute monitor: PID `231292`, log
  `outputs/publication_ext/monitor_logs/monitor_30min_20260514T224722Z.log`.
  It is sleeping before the next scheduled collection.
- Checkpoint cleanup: 53 completed-run `last.pt` checkpoints were deleted only
  after confirming sibling `best.pt` and valid `final_metrics.json`, excluding
  all active output directories; see `results/checkpoint_cleanup_deleted.md`.
- Latest audit refresh:
  - `python scripts/audit_publication_completion.py --run-tests` reports PASS
    with 95 unit tests
  - `python scripts/audit_drifting_faithfulness.py` reports PASS
  - `python scripts/audit_extension_completion.py --strict` reports PASS
  - `python scripts/audit_reviewer_experiment_readiness.py` reports PASS
  - `python scripts/audit_graph_archive_launchability.py` reports PASS

Completed destructive runner:

```bash
setsid python scripts/run_manifest_parallel.py \
  --manifest configs/publication_ext/manifest.json \
  --group destructive_drift \
  --devices 0,1,2,3 \
  --poll-seconds 30 \
  --status-file outputs/publication_ext/parallel_runner_status.json \
  --log-dir outputs/publication_ext/parallel_logs \
  > outputs/publication_ext/destructive_parallel_runner.log 2>&1 < /dev/null &
```

Runner artifacts:

- PID file: `outputs/publication_ext/destructive_parallel_runner.pid`
- Runner status: `outputs/publication_ext/parallel_runner_status.json`
- Logs: `outputs/publication_ext/parallel_logs/*.log`

Most recent stable completed snapshot:

| Run | GPU | Progress | Status |
|---|---:|---:|---|
| `ext_D_ATTR_qed_s42` | done | final | complete_fail |
| `ext_D_REPL_qed_s42` | done | final | complete_pass |
| `ext_D_BROKEN_ATTR_qed_s42` | done | final | complete_fail |
| `ext_D_BROKEN_REPL_qed_s42` | done | final | complete_pass |
| `ext_D_NOCROSS_qed_s42` | done | final | complete_pass |
| `ext_D_YONLY_qed_s42` | done | final | complete_pass |
| `ext_D_NONORM_qed_s42` | done | final | complete_pass |
| `ext_V_BETA_LOW_vae_s42` | done | final | complete_pass; exact recon `96.9%`, prior VUN `0.988` |
| `ext_V_BETA_HIGH_vae_s42` | done | final | complete_pass; exact recon `18.4%`, prior VUN `0.986` |
| `ext_V_LATENT128_vae_s42` | done | final | complete_pass; exact recon `94.3%`, prior VUN `0.992` |
| `ext_V_DEC6_vae_s42` | done | final | complete_pass; exact recon `56.5%`, prior VUN `0.980` |

Completed downstream VAE-drifting runs:

| Run | GPU | Progress | Final/live rho | Status |
|---|---:|---:|---:|---|
| `ext_vae_lowbeta_drift_qed_s42` | done | final | 0.437 | complete_pass; final metrics collected |
| `ext_vae_highbeta_drift_qed_s42` | done | final | 0.282 | complete_pass; final metrics collected |
| `ext_vae_latent128_drift_qed_s42` | done | final | 0.421 | complete_pass; final metrics collected |
| `ext_vae_dec6_drift_qed_s42` | done | final | 0.272 | complete_pass; final metrics collected |

Live VAE-drift values are parsed from logs by
`scripts/collect_vae_drift_live_snapshot.py`; they are monitoring signals only
and must not be used as final paper evidence until `final_metrics.json` exists.

Additional trained baseline:

| Run | GPU | Progress | Status |
|---|---:|---:|---|
| `ext_B_LINEAR_PROP_QED_s42` | done | final | complete; best alpha `5.0`, rho `0.211`, U `89.9%`, MAE `0.439` |
| `ext_B_LINEAR_PROP_QED_s43` | done | final | complete; best alpha `2.0`, rho `0.008`, U `94.5%`, MAE `0.250` |
| `ext_B_LINEAR_PROP_QED_s44` | done | final | complete; best alpha `5.0`, rho `-0.081`, U `94.2%`, MAE `0.315` |
| baseline postprocess | watcher | done | three-seed table collected in `results/tables/tab_trained_baseline_qed.tex` |

The completed destructive rows are available in `results/destructive_ablation.csv`.
All destructive drift ablations have final metrics.

Runtime note: the destructive, faithful, VAE-sensitivity, downstream VAE-drift,
generalization, trained-baseline, reviewer-extra, next-wave, and graph-route
queues have completed. The same-backbone generative-baseline queue also completed CVAE, WGAN-GP,
DDPM, and Flow Matching for seeds 42/43/44.
`scripts/run_manifest_parallel.py` now re-checks completion immediately before
launching a queued entry, so future controlled re-scheduling can avoid duplicate
output-directory launches if another process completed a queued experiment.
The completed destructive set provides the anti-symmetry/cross-normalization
evidence table for the supplement.

Faithful core completed automatically after destructive completion:

| Run | GPU | Progress | Status |
|---|---:|---:|---|
| `rf_FD_STRICT_PLAIN_PHI_QED_s42` | done | final | complete_pass |
| `rf_FD_STRICT_PROP_PHI_QED_s42` | done | final | complete_pass |
| `rf_FD_STRICT_RANDOM_PHI_QED_s42` | done | final | complete_pass |
| `rf_FD_STRICT_ZSPACE_QED_s42` | done | final | complete_pass |

Faithful-core final metrics are collected. `PLAIN`, `PROP`, and `RANDOM` pass
the mechanical completion gate, but their conditional rho remains weak
(`0.055`, `0.105`, and `0.122`, respectively). `ZSPACE` also passes and is the
strongest strict core result so far: best alpha `5.0`, QED Spearman rho
`0.127`, uniqueness `97.8%`, MAE `0.221`, and slope `0.169`. This is useful
as a conservative faithful-Drifting supplement result, but it is still a
modest-control signal rather than a strong main-paper claim.

Allocation sweeps are complete:

| Run | GPU | Progress | Status |
|---|---:|---:|---|
| `rf_FD_ALLOC_POS01_QED_s42` | done | final | complete_pass; rho `0.028`, U `68.7%` |
| `rf_FD_ALLOC_POS16_QED_s42` | done | final | complete_pass; rho `0.053`, U `76.7%` |
| `rf_FD_ALLOC_POS32_QED_s42` | done | final | complete_pass; rho `0.062`, U `79.4%` |
| `rf_FD_ALLOC_POS64_QED_s42` | done | final | complete_pass; rho `0.055`, U `81.4%` |
| `rf_FD_ALLOC_NEG16_QED_s42` | done | final | complete_pass; rho `0.076`, U `79.4%` |
| `rf_FD_ALLOC_NEG32_QED_s42` | done | final | complete_pass; rho `0.076`, U `79.9%` |

Reviewer-faithful artifacts:

- Plan: `docs/DRIFTING_FAITHFULNESS_PLAN.md`
- Prompt-to-artifact checklist: `docs/REVIEWER_PROMPT_TO_ARTIFACT_CHECKLIST.md`
- Supplement skeleton: `docs/SUPPLEMENT_FAITHFUL_DRIFTING.tex`
- Inlined supplement source: `docs/SUPPLEMENT_FAITHFUL_DRIFTING_INLINED.tex`
- Standalone supplement source/PDF:
  `docs/SUPPLEMENT_FAITHFUL_DRIFTING_AAAI.tex`,
  `DriftingMol_AAAI_FaithfulSupplement.pdf`
- Audit: `results/drifting_faithfulness_audit.md`
- Readiness audit: `results/reviewer_experiment_readiness_audit.md`
- Deferred launcher PID: `outputs/reviewer_faithful/deferred_faithful_core_launcher.pid`
  currently records PID `479910`
- Allocation postprocess watcher log:
  `outputs/reviewer_faithful/allocation_postprocess.log`
- VAE postprocess watcher log:
  `outputs/publication_ext/vae_postprocess.log`
- Trained baseline checklist:
  `docs/TRAINED_BASELINE_EXECUTION_CHECKLIST.md`
- Trained baseline collector:
  `scripts/collect_trained_baselines.py`, writing
  `results/trained_baseline_qed.csv`,
  `results/trained_baseline_status.json`, and
  `results/tables/tab_trained_baseline_qed.tex`
- Downstream VAE-drift manifest and collector:
  `configs/publication_ext/vae_drift_manifest.json`,
  `scripts/collect_vae_drift_results.py`, writing
  `results/vae_drift_downstream.csv`,
  `results/vae_drift_downstream_status.json`, and
  `results/tables/tab_vae_drift_downstream.tex`
- Downstream VAE-drift live monitor:
  `scripts/collect_vae_drift_live_snapshot.py`, writing
  `results/vae_drift_live_snapshot.json` and
  `results/tables/tab_vae_drift_live_snapshot.tex`; this is a monitoring
  artifact only, not final paper evidence
- Full live monitor snapshot:
  `python scripts/monitor_extension.py --json > results/live_monitor_snapshot.json`;
  this captures current queue, audit, GPU, disk, launcher, and watcher state as
  a machine-readable monitoring artifact
- Graph archive launchability audit:
  `scripts/audit_graph_archive_launchability.py`, writing
  `results/graph_archive_launchability_status.json` and
  `results/graph_archive_launchability_audit.md`; current status is PASS after
  the graph route artifacts and namespace adapter were verified
- Graph namespace adapter plan:
  `docs/GRAPH_NAMESPACE_ADAPTER_PLAN.md`, recording the exact archive-local
  `src.*` mapping and smoke-test imports to run before any graph GPU launch
- Checkpoint cleanup reports:
  `results/checkpoint_cleanup_candidates.md` and
  `results/checkpoint_cleanup_deleted.md`
- Downstream VAE-drift postprocess watcher:
  `outputs/publication_ext/vae_drift_postprocess.pid`, writing
  `outputs/publication_ext/vae_drift_postprocess.log`
- Generalization manifest and collector:
  `configs/publication_ext/generalization_manifest.json`,
  `scripts/collect_generalization_results.py`, writing
  `results/generalization_results.csv`,
  `results/generalization_status.json`, and
  `results/tables/tab_generalization.tex`
- Generalization deferred launchers:
  `outputs/publication_ext/generalization_launcher_gpu0.pid` through
  `outputs/publication_ext/generalization_launcher_gpu3.pid`
- Generalization postprocess watcher:
  `outputs/publication_ext/generalization_postprocess.pid`, writing
  `outputs/publication_ext/generalization_postprocess.log`
- Reviewer-extra manifest and collector:
  `configs/publication_ext/reviewer_extra_manifest.json`,
  `scripts/collect_reviewer_extra_results.py`, writing
  `results/reviewer_extra_results.csv`,
  `results/reviewer_extra_status.json`, and
  `results/tables/tab_reviewer_extra.tex`
- Reviewer-extra deferred launchers:
  `outputs/publication_ext/reviewer_extra_launcher_gpu0.pid` through
  `outputs/publication_ext/reviewer_extra_launcher_gpu3.pid`
- Reviewer-extra postprocess watcher:
  `outputs/publication_ext/reviewer_extra_postprocess.pid`, writing
  `outputs/publication_ext/reviewer_extra_postprocess.log`
- Next-wave experiment plan and configs:
  `docs/NEXT_WAVE_EXPERIMENT_PLAN.md`,
  `configs/publication_ext/next_wave_manifest.json`,
  `scripts/generate_next_wave_configs.py`, and
  `scripts/collect_next_wave_results.py`; these rows are pending and
  intentionally outside the current strict extension completion gate
- Disk cleanup:
  `scripts/report_checkpoint_cleanup_candidates.py`, writing
  `results/checkpoint_cleanup_candidates.md` and
  `results/checkpoint_cleanup_deleted.md`; cleanup has deleted 53
  completed-run `last.pt` checkpoints totaling about 12.7G across four batches
  after excluding active output directories. Current follow-up dry-run reports
  0 candidates.
- Graph stress follow-up plan:
  `docs/GRAPH_STRESS_EXECUTION_PLAN.md`; fresh graph training is intentionally
  not launched while compute is occupied and the graph checkpoints/cache needed
  by the archived graph line remain missing.

Robust deferred-launcher start command:

```bash
setsid python scripts/defer_faithful_core_after_destructive.py \
  --watch-status outputs/publication_ext/parallel_runner_status.json \
  --faithful-status outputs/reviewer_faithful/core_status.json \
  --devices 0,2,3 \
  --poll-seconds 60 \
  --log-dir outputs/reviewer_faithful/logs \
  --pid-file outputs/reviewer_faithful/deferred_faithful_core_launcher.pid \
  > outputs/reviewer_faithful/deferred_faithful_core_launcher.log 2>&1 < /dev/null &
```

## Monitoring Commands

Use:

```bash
python scripts/monitor_extension.py
python scripts/collect_extension_results.py
python scripts/collect_trained_baselines.py
python scripts/collect_vae_drift_results.py
python scripts/collect_generalization_results.py
python scripts/collect_reviewer_extra_results.py
python scripts/collect_next_wave_results.py
python scripts/audit_extension_completion.py
python scripts/collect_faithful_drifting_results.py
python scripts/audit_drifting_faithfulness.py
python scripts/audit_reviewer_experiment_readiness.py
python scripts/render_faithful_supplement.py
```

The old VAE queue-manager process was stopped after verifying that the active
BETA_LOW training child continued on GPU1. Current independent VAE status files
are:

- `outputs/publication_ext/parallel_runner_status_vae.json` for the retained
  BETA_LOW child/log; BETA_LOW is complete.
- `outputs/publication_ext/parallel_runner_status_vae_beta_high.json`
- `outputs/publication_ext/parallel_runner_status_vae_latent128.json`
- `outputs/publication_ext/parallel_runner_status_vae_dec6.json`; the DEC6
  VAE sensitivity run is complete. The downstream DEC6 drift runner in
  `outputs/publication_ext/parallel_runner_status_vae_drift_gpu1.json` is also
  complete.

Do not restart the full `vae_sensitivity` group while these independent
runners are active.

## Runtime Diagnosis

No training queue remains. The earlier slow period was caused by full
downstream generation jobs occupying every GPU, not by a stalled collector or
data-preparation process. The reviewer-extra, next-wave, graph-route, and
same-backbone generative-baseline queues are now complete, and `nvidia-smi`
shows all four RTX 4090D GPUs idle.

The faithful allocation sweep is now complete. Its purpose is no longer
scheduling; it is evidence for the supplement that direct faithful Drifting in
the molecule setting remains measurable but modest unless decoder coupling is
added.

`rf_FD_ALLOC_POS16_QED_s42` finished with the same qualitative conclusion:
the allocation change recovers diversity but does not recover strong
conditional tracking. Its best alpha is `5.0`, with QED Spearman rho `0.053`,
uniqueness `76.7%`, MAE `0.246`, and slope `0.065`; generated QED values still
cluster near `0.49-0.52` across targets. This supports a
negative/boundary-condition interpretation for allocation sensitivity unless a
larger positive or negative allocation row changes the trend.

`rf_FD_ALLOC_POS32_QED_s42` finished with the same trajectory: best alpha
`5.0`, QED Spearman rho `0.062`, uniqueness `79.4%`, MAE `0.244`, and slope
`0.078`. Across `Npos=1,16,32`, larger positive allocation repairs diversity
faster than target tracking, leaving the faithful Drifting molecular analogue
as a modest supplemental control rather than a main performance result.

`rf_FD_ALLOC_POS64_QED_s42` finished with the same pattern: best alpha `3.0`,
QED Spearman rho `0.055`, uniqueness `81.4%`, MAE `0.247`, and slope `0.064`.
The full positive-allocation sequence now supports the same conservative
interpretation: larger positive allocation improves diversity but does not
rescue target tracking.

`rf_FD_ALLOC_NEG16_QED_s42` finished with best alpha `5.0`, QED Spearman rho
`0.076`, uniqueness `79.4%`, MAE `0.246`, and slope `0.092`. It is the
joint strongest allocation row, but still below the `0.1` rho threshold and
therefore supports the conservative interpretation.

`rf_FD_ALLOC_NEG32_QED_s42` finished with best alpha `5.0`, QED Spearman rho
`0.076`, uniqueness `79.9%`, MAE `0.246`, and slope `0.091`. The complete
allocation sweep therefore supports the same conservative interpretation:
allocation helps diversity more than target tracking.

## Next Actions

1. Keep the main manuscript within the 8-page limit.
2. Use conditional latent VAE, WGAN-GP, DDPM, and Flow Matching as the
   generator-family baseline comparison.
3. Move any further optional external graph/string generator comparisons to
   supplement or a future version unless their protocols are fully reproduced.

## Graph Route Completion

Updated: 2026-05-15 09:15 UTC

The graph route is complete as a representation stress-test chain rather than
a replacement for the SELFIES main route.

- Rebuilt `archive/graph_vae_line/data/qm9/qm9_from_cache.smi` and
  `archive/graph_vae_line/data/cache/qm9_graph_cache.pt` from the recovered QM9
  SMILES list: 128,056 graph tensors, split as 102,444 train / 12,805 val /
  12,807 test.
- Added archive publication configs for graph VAE recovery, graph Latent-MAE,
  fresh QED graph drifting, fresh LogP graph drifting, and no-drift graph
  ablation.
- Completed graph VAE recovery:
  `archive/graph_vae_line/outputs/vae_v3_valence/final_metrics.json`. Final
  test reconstruction has edge accuracy `0.913`, bond accuracy `0.894`, and
  final sampled generation reports validity `1.000`, uniqueness `0.7455`, and
  novelty `0.7636` over 10,000 samples.
- Started `scripts/watch_graph_route_chain.py` with 30-minute polling. The
  watcher is currently building `qm9_latent_cache_v3.pt`; the VAE loader was
  fixed to infer the archived 4-class bond head from checkpoint weights before
  retrying the cache build.
- The latent-cache build encoded 102,444 train, 12,805 validation, and 12,807
  test molecules into 128-dimensional graph latents and wrote
  `archive/graph_vae_line/data/cache/qm9_latent_cache_v3.pt`.
- Graph Latent-MAE recovery completed with best validation loss `0.2687`.
  The learned feature head reports R2 values of `0.626` for QED, `0.599` for
  LogP, and `0.802` for MolWt.
- Fresh graph QED, graph LogP, and no-drift generator runs completed through
  `outputs/publication_ext/parallel_runner_status_graph_stress_retry.json`.
  The first launch exposed two archive-interface issues: zero-drift CFG skipped
  the decoder-feature pool, and `discretize_logits` did not accept the
  `temperature` argument used by graph evaluation. Both paths are patched, and
  `python -m unittest discover -s tests -p 'test_graph_*.py'` passes 4 graph
  tests including the new temperature-decoding smoke test.
- Fresh graph QED ends with validity `1.000`, uniqueness `0.131`, novelty
  `0.754`, and best QED rho `0.019`. Fresh graph LogP ends with validity
  `0.9997`, uniqueness `0.319`, novelty `0.755`, and best LogP rho `0.327`.
  The no-drift graph ablation ends with validity `1.000`, uniqueness `0.124`,
  novelty `0.743`, and best QED rho `0.046`.
- Raw-vs-repaired decode diagnostics and graph-vs-SELFIES comparison completed.
  The raw graph decode validity drops from about `0.76--0.78` at tau `0.0` to
  `0.18--0.23` at tau `1.0`, while repaired validity is `1.000` across the
  tested temperatures. `results/graph_stress_full_status.json` now reports
  `complete=true`.

Current graph launchability audit: PASS.

## Current Queue Snapshot

Updated: 2026-05-15 UTC

- Next-wave experiments are complete. The refreshed table is
  `results/tables/tab_next_wave.tex`; continuous-QED seed 43 finishes with
  alpha `3.0`, QED Spearman `0.340`, uniqueness `98.7%`, and MAE `0.208`.
  The trained linear-property baselines remain weak compared with
  decoder-coupled drift, supporting the mechanism interpretation.
- Reviewer-extra SA seed 43 is complete and synchronized into
  `results/tables/tab_reviewer_extra.tex`: best alpha `3.0`, SA Spearman
  `0.440`, uniqueness `98.9%`, and MAE `0.947`.
- Graph VAE recovery is complete. Final evaluation gives validity `1.000`,
  uniqueness `0.7455`, novelty `0.7636`, edge accuracy `0.913`, and bond
  accuracy `0.894`.
- The graph watcher's first fresh-generator stage recorded the initial failed
  launch, but the replacement runner completed all fresh graph rows and the
  deferred postprocess completed both graph postprocess rows with no failures.

The AAAI PDF rebuild is currently clean with the correct AAAI style/bibliography
search paths and is now 9 pages after the expert-review polish pass. The manuscript describes graph experiments
as a representation stress test rather than as a replacement for the SELFIES
main route.

## Final Paper Refresh

Updated: 2026-05-17 UTC

- Rebuilt quantitative Fig.2 and Fig.5 from `scripts/plot_result_figures.py`
  and switched the AAAI source to `fig2_qed_ablation.pdf` and
  `fig5_zdiv_pareto.pdf`; the data figures no longer reference image-2
  outputs.
- Recompiled `DriftingMol_AAAI.pdf`; the final main draft is 9 pages.
- Refreshed the clean source-only arXiv bundle at
  `submission/driftingmol_submission_source.zip`.
- Verified `python scripts/audit_publication_completion.py --run-tests`,
  `python scripts/audit_extension_completion.py --strict`,
  `python scripts/audit_drifting_faithfulness.py`, graph launchability, and
  graph unittest discovery.

## Figure Layout Fix

Updated: 2026-05-15 22:40 UTC

- Reworked Fig.2 as a single full-width, script-generated QED ablation ranking
  without the crowded mechanism-group subpanel.
- Reworked Fig.3 as two aligned traces for QED rho and uniqueness instead of a
  dual-axis plot with overlapping end labels.
- Reworked Fig.4 with short method labels and wider numeric spacing.
- Rebuilt `DriftingMol_AAAI.pdf`; the PDF remains 8 pages. Rendered page 6 to
  `results/pdf_preview/current_page_06.png` and visually checked that
  Fig.2--Fig.4 no longer overlap.
- Re-ran `python scripts/audit_publication_completion.py --run-tests`: PASS.

## Manuscript Polish

Updated: 2026-05-15 23:05 UTC

- Tightened the abstract and introduction around the mechanism-focused claim:
  decoder-coupled drift in a SELFIES latent space, not a broad SOTA molecular
  optimization claim.
- Rewrote the evaluation-protocol and matched-reference paragraphs to clarify
  the fixed guidance grid, quality gates, retrieval/jitter references, and
  trained linear property-head baseline.
- Condensed the limitations section so the SELFIES validity guarantee,
  protocol-matched comparison boundary, and graph-route stress-test boundary are
  stated directly.
- Rebuilt `DriftingMol_AAAI.pdf`; it remains 8 pages. Rendered polished page
  previews to `results/pdf_preview/polished_page_*.png`.
- Re-ran `python scripts/audit_publication_completion.py --run-tests`: PASS.
