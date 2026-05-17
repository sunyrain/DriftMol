# Graph Route Completion Plan

Updated: 2026-05-15 UTC

This item is no longer just a diagnostic note. The graph-based route must be
completed as a reviewer-facing package covering graph VAE recovery, graph
latent drifting, raw/repaired validity, QED/LogP control, and a fair comparison
against the SELFIES route.

## Current State

- The archived graph namespace adapter exists under `archive/graph_vae_line/src`,
  so old `src.*` imports resolve to graph modules rather than the current
  SELFIES modules.
- Archived diagnostics remain useful but insufficient: E36 QED decoder-drift
  CFG reaches best archived rho `0.159`, and E40 LogP bin-queue drift reaches
  best archived rho `0.145`; both show a graph control-diversity bottleneck.
- The missing blockers are concrete artifacts: graph VAE `best.pt`, graph
  latent cache `qm9_latent_cache_v3.pt`, and graph Latent-MAE
  `best_latent_mae.pt`.
- GPU2 is currently idle, so graph recovery can start after cache validation.

## Recovery Sequence

1. Rebuild the archive-local graph dataset from the recovered QM9 SMILES list:

```bash
python scripts/rebuild_graph_qm9_cache_from_smiles.py
```

This writes `archive/graph_vae_line/data/qm9/qm9_from_cache.smi` and
`archive/graph_vae_line/data/cache/qm9_graph_cache.pt`.

2. Train or resume the graph VAE checkpoint:

```bash
cd archive/graph_vae_line
PYTHONPATH=. python -m src.train.train_vae --config configs/publication/vae_v3_valence_recover.yaml --resume
```

3. Build the graph latent cache:

```bash
cd archive/graph_vae_line
PYTHONPATH=. python scripts/build_latent_cache.py --vae_ckpt outputs/vae_v3_valence/best.pt --output data/cache/qm9_latent_cache_v3.pt
```

4. Train graph Latent-MAE:

```bash
cd archive/graph_vae_line
PYTHONPATH=. python -m src.train.train_latent_mae configs/publication/latent_mae_v3_recover.yaml
```

5. Run fresh graph latent drifting:

```bash
cd archive/graph_vae_line
PYTHONPATH=. python -m src.train.train_generator --config configs/publication/e36_dec_drift_cfg_fresh.yaml
PYTHONPATH=. python -m src.train.train_generator --config configs/publication/e40_logp_bins_queue_fresh.yaml
PYTHONPATH=. python -m src.train.train_generator --config configs/publication/e36_no_drift_fresh.yaml
```

6. Evaluate raw versus repaired decoding:

```bash
cd archive/graph_vae_line
PYTHONPATH=. python scripts/eval_raw_vs_repair.py --experiments ../../outputs/publication_ext/graph_stress/e36_dec_drift_cfg_fresh ../../outputs/publication_ext/graph_stress/e40_logp_bins_queue_fresh --temperatures 0.0 0.5 1.0 --num_samples 10000 --batch_size 256
```

7. Rebuild the graph-vs-SELFIES comparison:

```bash
python scripts/summarize_graph_stress.py
python scripts/plot_graph_stress.py
python scripts/audit_graph_archive_launchability.py
```

## Completion Gate

The graph package is complete only when all of the following exist:

- `archive/graph_vae_line/outputs/vae_v3_valence/best.pt`
- `archive/graph_vae_line/data/cache/qm9_latent_cache_v3.pt`
- `archive/graph_vae_line/outputs/latent_mae_v3/best_latent_mae.pt`
- fresh QED graph-control `final_metrics.json`
- fresh LogP graph-control `final_metrics.json`
- no-drift graph ablation `final_metrics.json`
- fresh raw-vs-repaired validity JSON/table
- updated graph-vs-SELFIES table using the same validity, uniqueness, novelty,
  and Spearman control metrics as the SELFIES route

The executable manifest is
`configs/publication_ext/graph_stress_manifest.json`.
