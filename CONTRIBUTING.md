# Contributing to DriftingMol

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

```bash
git clone https://github.com/your-org/DriftingMol.git
cd DriftingMol
pip install -r requirements.txt
```

## Code Structure

DriftingMol is a two-stage pipeline:

| Stage | Module | Description |
|-------|--------|-------------|
| VAE | `src/models/selfies_vae.py` | SELFIES GRU β-VAE: SELFIES ↔ z ∈ R^256 |
| Generator | `src/utils.py` | `LatentDiTGeneratorCFG`: noise → z via DiT with FiLM + CFG |
| Drift | `src/drifting/drift_latent_phi.py` | Multi-temperature drift field in decoder-φ space |
| Training | `src/train/train_selfies_cfg.py` | Main training loop with drift loss + z-div regularization |
| Evaluation | `src/eval/` | Validity, molecular metrics, quality gate |

### Key Design Decisions

- **φ = VAE decoder**: The frozen decoder's 512D intermediate representation serves as the drift feature space. Gradients flow through φ to update the generator. No separate feature extractor needed.
- **SELFIES**: Guarantees 100% chemical validity by construction. Unlike graph representations, SELFIES preserves continuous property signals through decoding.
- **1-NFE**: Single-step generation is a hard constraint — no iterative refinement.

## Adding New Experiments

1. Create a YAML config in `configs/final/` following the pattern of `exp_F_qed.yaml`
2. Run: `python -m src.train.train_selfies_cfg --config configs/final/your_config.yaml`
3. Results are saved to `outputs/` with metrics in `final_metrics.json`

## Code Style

- Follow existing patterns in the codebase
- Use type hints for function signatures
- Keep imports organized: stdlib → third-party → local

## Reporting Issues

When reporting bugs, please include:
- Python and PyTorch versions
- Full error traceback
- Config file used
- GPU model and CUDA version
