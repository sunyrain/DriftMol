# Graph Archive Launchability Audit

Overall: PASS

## Required Artifacts

| Config | Field | Path | Exists |
|---|---|---|---|
| archive/graph_vae_line/configs/e36_dec_drift_cfg.yaml | vae.checkpoint | `outputs/vae_v3_valence/best.pt` | yes |
| archive/graph_vae_line/configs/e36_dec_drift_cfg.yaml | phi.checkpoint | `outputs/latent_mae_v3/best_latent_mae.pt` | yes |
| archive/graph_vae_line/configs/e36_dec_drift_cfg.yaml | data.latent_cache_path | `data/cache/qm9_latent_cache_v3.pt` | yes |
| archive/graph_vae_line/configs/e40_logp_bins_queue.yaml | vae.checkpoint | `outputs/vae_v3_valence/best.pt` | yes |
| archive/graph_vae_line/configs/e40_logp_bins_queue.yaml | phi.checkpoint | `outputs/latent_mae_v3/best_latent_mae.pt` | yes |
| archive/graph_vae_line/configs/e40_logp_bins_queue.yaml | data.latent_cache_path | `data/cache/qm9_latent_cache_v3.pt` | yes |

## Namespace

- archived `src` package exists: True
- archived `src.utils` exists: True
- current `src.utils` graph-ready: True
- namespace adapter plan exists: True

## Archived Diagnostic Metrics

| Run | Property | Metrics | Best alpha | Best rho | Validity | Uniqueness | Novelty |
|---|---|---|---:|---:|---:|---:|---:|
| e36_dec_drift_cfg | qed | yes | 4.0 | 0.159 | 1.000 | 0.285 | 0.775 |
| e40_logp_bins_queue | logp | yes | 1.5 | 0.145 | 1.000 | 0.249 | 0.708 |

## Recovery Candidates

- graph cache: `archive/data_qm9/qm9_graph_cache.pt` exists=True
- legacy latent caches: 2
- raw gdb9.sdf candidates: 0
- VAE v3 final metrics: `archive/graph_vae_line/outputs/vae_v3_valence/final_metrics.json` exists=True
- latent-MAE v3 train log: `archive/graph_vae_line/outputs/latent_mae_v3_train.log` exists=True, mentions checkpoint=True
