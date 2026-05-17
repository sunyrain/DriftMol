# DriftingMol

Decoder-coupled latent drifting for property-conditional molecular generation.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

DriftingMol is a two-stage molecular generation study. Stage 1 trains a
SELFIES beta-VAE on ZINC250K. Stage 2 freezes the VAE decoder and trains a
conditional latent generator with a drift loss measured in decoder feature
space. At inference time the model uses one generator forward pass plus one
frozen decoder pass.

SELFIES gives representation-level robustness, while all reported molecular
validity is still computed after SELFIES decoding and RDKit canonicalization.

## Repository Contents

This repository tracks the code, configs, compact result summaries, figures,
and manuscript sources needed to reproduce the current paper draft. It does
not track raw data, latent caches, checkpoints, generated PDFs, LaTeX auxiliary
files, or large experiment outputs.

- Reproducibility guide: [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)
- Current generated tables: [results/publication_summary.md](results/publication_summary.md)
- Legacy Static Snapshot: [docs/FULL_RESULTS.md](docs/FULL_RESULTS.md)
- Manuscript build notes: [docs/PAPER_BUILD.md](docs/PAPER_BUILD.md)

## Main Results

All values below are generated from JSON artifacts under `outputs/` and
summarized by `scripts/collect_results.py`.

| Protocol | Setting | Spearman rho | Uniqueness |
|---|---|---:|---:|
| QED, three seeds | Single temperature | 0.515 mean | 94.8% mean |
| QED, three seeds | Full decoder-coupled drift | 0.512 mean | 94.3% mean |
| QED, three seeds | No z-diversity | 0.513 mean | 77.5% mean |
| QED, seed 42 | Full decoder-coupled drift | 0.493 | 94.7% |
| Multi-4-property v2 | Single temperature | 0.598 mean across properties | 88.2% lowest bin |
| Multi-4-property v2 | Full decoder-coupled drift | 0.560 mean across properties | 91.5% lowest bin |

The main ablation pattern is that preserving the gradient path through frozen
decoder features produces stronger property control than z-space drift,
random-feature drift, detached decoder controls, or trained external feature
maps under the same protocol.

Same-backbone diagnostic baselines are tracked separately in
[results/generative_baselines_qed.json](results/generative_baselines_qed.json)
and audited by `scripts/audit_generative_baselines.py`.

## Minimal Workflow

Requirements: Python 3.10+, PyTorch with CUDA, RDKit, SELFIES, PyYAML, NumPy,
SciPy, pandas, and matplotlib.

```bash
git clone git@github.com:sunyrain/DriftMol.git
cd DriftMol
pip install -r requirements.txt
```

The full reproducibility path is documented in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). The short form is:

```bash
# 1. Put ZINC250K at data/raw/zinc250k.csv, then build local caches.
mkdir -p data/raw
wget -O data/raw/zinc250k.csv \
  "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
python scripts/build_zinc_cache.py

# 2. Train the SELFIES VAE and cache latents.
python -m src.train.train_selfies_vae \
  --config configs/foundation/zinc_selfies_vae_v2.yaml
python scripts/build_selfies_latent_cache.py \
  --vae_ckpt outputs/foundation/zinc_selfies_vae_v2/best.pt \
  --graph_cache data/cache/zinc250k_smiles_cache.pt \
  --output data/cache/zinc_latent_cache_v2.pt

# 3. Run a representative decoder-coupled drift experiment.
python -m src.train.train_selfies_cfg \
  --config configs/final/exp_F_qed.yaml
```

Manifest runners are available for full reproduction and extension queues:

```bash
python scripts/run_publication_experiments.py \
  --manifest configs/publication/manifest.json

python scripts/run_manifest_parallel.py \
  --manifest configs/publication_ext/manifest.json \
  --devices 0,1,2,3
```

## Audits

After experiments finish, refresh and audit tracked artifacts:

```bash
python scripts/collect_results.py
python scripts/export_latex_tables.py
python scripts/plot_result_figures.py
python scripts/benchmark_inference.py
python scripts/audit_publication_completion.py --run-tests
python scripts/audit_extension_completion.py --strict
python scripts/audit_generative_baselines.py
```

`python scripts/audit_publication_completion.py --run-tests` is the main
completion gate; the current repository passes that audit with 95 unit tests.

## Layout

```text
src/                         core models, drift losses, data and eval code
configs/foundation/          SELFIES VAE and feature-model configs
configs/final*/              main ablation configs
configs/publication*/        manifest-driven paper and extension configs
scripts/                     data prep, runners, collectors, audits, plotting
results/                     tracked summaries, generated tables, audit status
docs/                        paper sources, figures, plans, build notes
data/raw/, data/cache/       ignored local data and latent caches
outputs/                     ignored local checkpoints and generated metrics
archive/                     ignored local development history
```

## Citation

```bibtex
@misc{driftingmol2026,
  title  = {DriftingMol: Decoder-Coupled Latent Drifting for Molecular Generation},
  author = {Qiu, Jiangjie and Li, Yijun and Li, Wentao and Wang, Xiaonan},
  year   = {2026},
  note   = {Manuscript in preparation}
}

@misc{deng2026drifting,
  title  = {Generative Modeling via Drifting},
  author = {Deng, Mingyang and Li, He and Li, Tianhong and Du, Yilun and He, Kaiming},
  year   = {2026},
  eprint = {2602.04770},
  archivePrefix = {arXiv}
}
```
