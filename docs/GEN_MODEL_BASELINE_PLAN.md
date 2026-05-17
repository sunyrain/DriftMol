# Generative Model Baseline Plan

Updated: 2026-05-16 UTC

## Motivation

The publication-grade comparison needs representative generative families
evaluated under the same ZINC250K, target-conditioning, and metric protocol as
DriftingMol.

## Baseline Tiers

### Tier 1: Same-Backbone Generative Baselines

These are the primary fair baselines because they share the same SELFIES VAE
decoder, latent cache, QED target bins, 10,000-sample evaluation, and RDKit
metric code.

| Family | Implemented baseline | Artifact |
|---|---|---|
| VAE | Conditional latent VAE prior, p(z \| y) | `scripts/train_latent_generative_baseline.py --method cvae` |
| GAN | Conditional latent WGAN-GP | `scripts/train_latent_generative_baseline.py --method gan` |
| Diffusion | Conditional latent DDPM | `scripts/train_latent_generative_baseline.py --method diffusion` |
| Flow Matching | Conditional latent flow matching with Euler sampling | `scripts/train_latent_generative_baseline.py --method flow_matching` |

Shared manifest:

```bash
python scripts/run_manifest_parallel.py \
  --manifest configs/publication_ext/generative_baselines_manifest.json \
  --devices 0,1,2,3 \
  --poll-seconds 30 \
  --status-file outputs/publication_ext/parallel_runner_status_generative_baselines.json \
  --log-dir outputs/publication_ext/parallel_logs_generative_baselines
```

Expected outputs:

- `results/generative_baselines_qed.json`
- `results/tables/tab_generative_baselines_qed.tex`
- `results/generative_baselines/*_qed_s42.json`
- `outputs/publication_ext/generative_baselines/*/final_metrics.json`

Current execution:

- Seeds 42, 43, and 44 are complete for all four families.
- The LaTeX table aggregates available seeds by method and switches to
  mean/std notation when multiple seeds are present.

## Completed Three-Seed Result

| Family | Seeds | NFE | QED rho | MAE | U | N | IntDiv |
|---|---:|---:|---:|---:|---:|---:|---:|
| Conditional latent VAE | 3 | 1 | 0.014 +/- 0.026 | 0.237 +/- 0.004 | 16.8 +/- 0.7% | 100.0% | 0.713 +/- 0.017 |
| Conditional latent WGAN-GP | 3 | 1 | 0.151 +/- 0.013 | 0.251 +/- 0.000 | 98.2 +/- 0.2% | 100.0% | 0.913 +/- 0.001 |
| Conditional latent DDPM | 3 | 100 | 0.048 +/- 0.015 | 0.240 +/- 0.002 | 98.0 +/- 0.2% | 100.0% | 0.902 +/- 0.000 |
| Conditional latent Flow Matching | 3 | 50 | 0.080 +/- 0.004 | 0.252 +/- 0.002 | 98.2 +/- 0.2% | 100.0% | 0.905 +/- 0.002 |

### Tier 2: Public External Generative References

These should be discussed separately because most public implementations do
not natively use the DriftingMol QED target-bin protocol or SELFIES latent
backbone.

| Family | Representative public model | Use in paper |
|---|---|---|
| VAE | JT-VAE | External graph/string VAE reference; not same-backbone unless reimplemented. |
| GAN | MolGAN | Classic graph GAN reference; likely limited by small-graph setup and mode collapse. |
| Flow | MoFlow, GraphAF, GraphDF | Public graph normalizing-flow references for molecule generation. |
| Diffusion | DiGress, GDSS, CDGS | Public graph diffusion references. |
| Flow Matching | FlowMol / PropMolFlow-style models | Mostly 3D or task-specific; cite as related work unless adapted. |

## Claim Rule

Use Tier 1 for quantitative DriftingMol-vs-generator comparisons. Use Tier 2
as related-work context or future external adaptation unless the exact
target-bin protocol is reproduced.
