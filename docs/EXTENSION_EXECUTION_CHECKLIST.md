# Extension Execution Checklist

Updated: 2026-05-15 UTC

This checklist tracks extension-stage experiments that are intentionally
separate from the audited 8-page AAAI submission package.

## Destructive Drift Ablations

| Group | Experiment | Change | Status | Command |
|---|---|---|---|---|
| destructive_drift | `ext_D_ATTR_qed_s42` | attraction only | complete_fail | `python -m src.train.train_selfies_cfg --config configs/publication_ext/destructive/ext_D_ATTR_qed_s42.yaml` |
| destructive_drift | `ext_D_BROKEN_ATTR_qed_s42` | 1.5x attraction | complete_fail | `python -m src.train.train_selfies_cfg --config configs/publication_ext/destructive/ext_D_BROKEN_ATTR_qed_s42.yaml` |
| destructive_drift | `ext_D_BROKEN_REPL_qed_s42` | 1.5x repulsion | complete_pass | `python -m src.train.train_selfies_cfg --config configs/publication_ext/destructive/ext_D_BROKEN_REPL_qed_s42.yaml` |
| destructive_drift | `ext_D_NOCROSS_qed_s42` | no cross-multiplication | complete_pass | `python -m src.train.train_selfies_cfg --config configs/publication_ext/destructive/ext_D_NOCROSS_qed_s42.yaml` |
| destructive_drift | `ext_D_NONORM_qed_s42` | no normalization | complete_pass | `python -m src.train.train_selfies_cfg --config configs/publication_ext/destructive/ext_D_NONORM_qed_s42.yaml` |
| destructive_drift | `ext_D_REPL_qed_s42` | repulsion only | complete_pass | `python -m src.train.train_selfies_cfg --config configs/publication_ext/destructive/ext_D_REPL_qed_s42.yaml` |
| destructive_drift | `ext_D_YONLY_qed_s42` | y-only normalization | complete_pass | `python -m src.train.train_selfies_cfg --config configs/publication_ext/destructive/ext_D_YONLY_qed_s42.yaml` |
| vae_sensitivity | `ext_V_BETA_HIGH_vae_s42` | higher beta tests whether stronger regularization weakens conditional control | complete_pass | `python -m src.train.train_selfies_vae --config configs/publication_ext/vae_sensitivity/ext_V_BETA_HIGH_vae_s42.yaml` |
| vae_sensitivity | `ext_V_BETA_LOW_vae_s42` | lower beta tests whether a more information-rich latent changes prior quality | complete_pass | `python -m src.train.train_selfies_vae --config configs/publication_ext/vae_sensitivity/ext_V_BETA_LOW_vae_s42.yaml` |
| vae_sensitivity | `ext_V_DEC6_vae_s42` | decoder-capacity sensitivity with a deeper decoder | complete_pass | `python -m src.train.train_selfies_vae --config configs/publication_ext/vae_sensitivity/ext_V_DEC6_vae_s42.yaml` |
| vae_sensitivity | `ext_V_LATENT128_vae_s42` | latent dimension sensitivity with a narrower latent bottleneck | complete_pass | `python -m src.train.train_selfies_vae --config configs/publication_ext/vae_sensitivity/ext_V_LATENT128_vae_s42.yaml` |

Completion gate: at least three destructive ablations should complete and
act as interpretable negative controls before this evidence is promoted into
the main paper.
