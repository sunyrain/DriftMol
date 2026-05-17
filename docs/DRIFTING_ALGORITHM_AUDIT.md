# Drifting Algorithm Audit

Updated: 2026-05-15 UTC

This note maps the original Drifting Models equations and pseudocode to the
DriftingMol implementation. It is meant for supplemental material and internal
review; it should prevent ambiguity between faithful Drifting reproduction and
the proposed decoder-coupled molecular extension.

## Core Mapping

| Original Drifting component | DriftingMol implementation | Evidence |
|---|---|---|
| Pushforward generator `x=f_theta(epsilon,c,alpha)` | Conditional latent DiT maps Gaussian noise, property/bin condition, and alpha to SELFIES VAE latent `z_gen`. | `src/train/train_selfies_cfg.py:1901-1924` |
| Feature-space drifting loss, Eq. 13 | `phi_gen_all = phi_model.extract_features(z_gen_n_phi)` and group-wise loss in phi-space. | `src/train/train_selfies_cfg.py:1926-1953`, `2034-2080` |
| Generated samples as negatives | `phi_gen` is passed to `compute_drift_field_paper` and internally used for `dist_neg` and negative targets. | `src/drifting/drift_latent_phi.py:647-710`, `799-808` |
| L2 kernel distance, Eq. 12 | `torch.cdist(..., p=2)` followed by logits `-distance / temperature`. | `src/drifting/drift_latent_phi.py:701-703`, `730-750` |
| Feature-distance normalization, Appendix A.6 | Strict configs keep `drift_normalize_dist: true`, which normalizes feature distances before applying the kernel temperature. | `src/drifting/drift_latent_phi.py:653-683`, `716-725`; `configs/reviewer_faithful/core/*.yaml` |
| Joint positive/negative normalization | Positive and negative logits are concatenated before softmax. | `src/drifting/drift_latent_phi.py:761-762` |
| Bidirectional softmax | Row softmax over y and column softmax over x. | `src/drifting/drift_latent_phi.py:767-775` |
| Cross-multiplication weights, Algorithm 2 | `W_pos = A_pos * A_neg.sum(...)`, `W_neg = A_neg * A_pos.sum(...)`. | `src/drifting/drift_latent_phi.py:789-797` |
| Drift vector `V = V+ - V-` | `V = attraction_scale * drift_pos - repulsion_scale * drift_neg`. | `src/drifting/drift_latent_phi.py:799-808` |
| Stop-gradient target, Eq. 6 / Eq. 13 | Target is `phi_gen.detach() + V`; loss is MSE to the frozen target. | `src/drifting/drift_latent_phi.py:857-876` |
| Multiple temperatures | `multi_temp_drift_loss` receives `drift_temperatures` and sums per-temperature drift. | `src/train/train_selfies_cfg.py:2066-2078`; `src/drifting/drift_latent_phi.py:527-604` |
| CFG training-time alpha | `alpha_val = sample_cfg_alpha(...)`; unconditional negatives receive `w=(alpha-1)(Nneg-1)/Nunc`. | `src/train/train_selfies_cfg.py:1901-1903`, `2051-2070` |
| Class-conditional positives | Molecular analogue uses QED quantile bins as class labels; positives are sampled from the same bin in strict faithful configs. | `src/train/train_selfies_cfg.py:1257-1302`, `1993-2002`; `configs/reviewer_faithful/core/*.yaml` |
| Positive/negative allocation | Strict configs use original-style `Nc`, `Npos`, `Nneg`, and fixed effective batch sweeps. | `configs/reviewer_faithful/manifest.json` |
| Anti-symmetry destructive tests | Attraction/repulsion scales and normalization modes implement the original destructive-ablation logic. | `src/train/train_selfies_cfg.py:1192-1213`; `configs/publication_ext/destructive/*.yaml` |

## What Is Faithful

The following components are direct reproductions or molecular analogues of the
original Drifting recipe:

- Algorithm-2 field computation.
- generated batch as negative distribution `q`.
- positive samples from the target distribution `p`; for molecules, QED bins
  act as class labels.
- latent-MAE phi feature extractor.
- multiple temperature aggregation.
- Appendix-A.6 feature-distance and drift normalization.
- training-time CFG with unconditional data negatives.
- destructive anti-symmetry ablations.
- positive/negative sample allocation sweeps.

The tensor-level audit in `scripts/audit_drifting_faithfulness.py` confirms
that `compute_drift_field_paper` exactly matches a direct transcription of
Algorithm 2 on synthetic inputs when optional extensions are disabled. The same
audit now separately verifies that the reviewer-faithful configs keep the
strict protocol settings, including Appendix-A.6 distance normalization,
generated-negative allocation, no decoder coupling, and no z-diversity:

- current `max_abs_diff`: `0.0`
- strict protocol config status: `PASS`
- evidence: `results/drifting_faithfulness_status.json`

## What Is A Molecular Extension

The following are DriftingMol-specific adaptations and should not be described
as part of the original Drifting recipe:

- SELFIES VAE latent space and SELFIES validity guarantee.
- decoder hidden states as phi.
- decoder-coupled gradients through the frozen decoder.
- hybrid bin plus decoder-nearest positive sampling.
- z-diversity regularization.
- fixed lambda calibration used by the strongest main-paper settings.
- graph/SELFIES representation stress tests.

These extensions are legitimate method contributions, but they must be
separated from the faithful-reproduction claim.

## Reviewer-Facing Interpretation

Recommended wording:

> We first audit the implementation against the original Drifting Models
> Algorithm 2 and run a strict latent-MAE phi reproduction in the SELFIES VAE
> latent space. This separates faithful reproduction of Drifting from the
> proposed decoder-coupled molecular adaptation.

The strict reproduction package is now complete because:

1. `faithful_core` final metrics exist,
2. `python scripts/audit_drifting_faithfulness.py` reports `PASS`,
3. `results/tables/tab_faithful_drifting_core.tex` contains completed rows,
4. `results/tables/tab_faithful_drifting_allocation.tex` contains completed
   allocation rows,
5. and the final paper/supplement states which components are faithful versus
   molecule-specific.
