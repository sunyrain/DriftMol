# Reproducibility Guide

This guide describes the tracked commands and artifacts used to reproduce the
current DriftingMol paper draft. Large local artifacts are intentionally ignored
by Git: raw ZINC files, caches, checkpoints, `outputs/`, generated PDFs, and
LaTeX build files.

## Environment

Recommended baseline:

- Python 3.10+
- CUDA-capable PyTorch
- RDKit
- SELFIES
- NumPy, SciPy, pandas, matplotlib, PyYAML, pytest

Install:

```bash
pip install -r requirements.txt
```

## Data

Download ZINC250K and build the SMILES cache:

```bash
mkdir -p data/raw
wget -O data/raw/zinc250k.csv \
  "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"

python scripts/build_zinc_cache.py
```

Expected local outputs:

- `data/raw/zinc250k.csv`
- `data/cache/zinc250k_smiles_cache.pt`

Both paths are local-only and ignored by Git.

## Stage 1: SELFIES VAE

Train the foundation model:

```bash
python -m src.train.train_selfies_vae \
  --config configs/foundation/zinc_selfies_vae_v2.yaml
```

Build the latent cache used by drift experiments:

```bash
python scripts/build_selfies_latent_cache.py \
  --vae_ckpt outputs/foundation/zinc_selfies_vae_v2/best.pt \
  --graph_cache data/cache/zinc250k_smiles_cache.pt \
  --output data/cache/zinc_latent_cache_v2.pt
```

Expected local outputs:

- `outputs/foundation/zinc_selfies_vae_v2/best.pt`
- `data/cache/zinc_latent_cache_v2.pt`

## Stage 2: Decoder-Coupled Drift

Run a single QED experiment:

```bash
python -m src.train.train_selfies_cfg \
  --config configs/final/exp_F_qed.yaml
```

Run the publication manifest:

```bash
python scripts/run_publication_experiments.py \
  --manifest configs/publication/manifest.json
```

Run extension queues across several GPUs:

```bash
python scripts/run_manifest_parallel.py \
  --manifest configs/publication_ext/manifest.json \
  --devices 0,1,2,3 \
  --poll-seconds 1800
```

Manifest runners skip entries that already have `final_metrics.json` unless
`--force` is passed.

## Result Collection

Refresh the tracked summaries and LaTeX table fragments:

```bash
python scripts/collect_results.py
python scripts/collect_extension_results.py
python scripts/collect_generalization_results.py
python scripts/collect_trained_baselines.py
python scripts/export_latex_tables.py
```

Refresh figures and benchmark summaries:

```bash
python scripts/plot_result_figures.py
python scripts/benchmark_inference.py
python scripts/update_manuscript_benchmark.py
```

Key tracked outputs:

- `results/publication_summary.md`
- `results/publication_results.csv`
- `results/tables/*.tex`
- `docs/figures/*.pdf`
- `docs/figures/*.png`

## Audits

Run the main paper audit:

```bash
python scripts/audit_publication_completion.py --run-tests
```

Run extension and reviewer-facing audits:

```bash
python scripts/audit_extension_completion.py --strict
python scripts/audit_reviewer_experiment_readiness.py
python scripts/audit_generative_baselines.py
python scripts/audit_graph_archive_launchability.py
```

Run unit tests directly:

```bash
python -m pytest tests
```

The main audit expects the README to link `results/publication_summary.md` and
to label the fair `Multi-4-property v2` protocol. Keep those strings stable
when editing public documentation.

## Baselines and Stress Tests

Same-backbone generative baselines are defined in:

- `configs/publication_ext/generative_baselines_manifest.json`
- `scripts/train_latent_generative_baseline.py`
- `scripts/audit_generative_baselines.py`

Graph-route stress tests and VAE sensitivity checks are extension evidence, not
replacements for the SELFIES route:

- `docs/GRAPH_STRESS_TEST.md`
- `docs/GRAPH_STRESS_EXECUTION_PLAN.md`
- `configs/publication_ext/graph_stress_manifest.json`
- `configs/publication_ext/vae_drift_manifest.json`

## Manuscript Build

Use [docs/PAPER_BUILD.md](PAPER_BUILD.md) for LaTeX build commands and template
notes. Generated PDFs and auxiliary files are ignored locally; commit source,
figures, bibliography, table fragments, and audit summaries instead.
