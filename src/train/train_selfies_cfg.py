"""
Classifier-Free Guidance (CFG) Drifting Generator for SELFIES latent space.

Key innovations over the base generator:
  1. **Conditional drift**: Drift targets are matched to the conditioning property.
     When conditioning on LogP=x, generated samples drift toward data with LogP ≈ x.
  2. **Property regression loss**: Lightweight head z → property prediction.
  3. **α-aware training**: Generator learns to modulate output by guidance strength.

Usage:
    python -m src.train.train_selfies_cfg --config configs/selfies_drifting_cfg_logp.yaml
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Suppress RDKit "please use MorganGenerator" deprecation warnings
from rdkit import RDLogger as _RDLogger
_RDLogger.logger().setLevel(_RDLogger.ERROR)

from src.drifting.drift_latent_phi import (
    phi_drift_loss_v4_bn,
    phi_drift_loss_paper,
    multi_temp_drift_loss,
    sample_cfg_alpha,
    z_space_repulsion_loss,
    phi_space_repulsion_loss,
    decoupled_phi_drift_loss,
    multiscale_decoupled_drift_loss,
    sinkhorn_alignment_loss,
    knn_barycentric_alignment_loss,
    correlation_structure_loss,
    covariance_matching_loss,
)
from src.eval.quality_gate import QualityGate
from src.models.latent_mae import LatentMAE
from src.models.selfies_vae import (
    SelfiesVAE, SelfiesVAEConfig,
    set_vocab, batch_token_ids_to_smiles, PAD_IDX,
)
from src.models.selfies_vae_spatial import (
    SelfiesSpatialVAE, SelfiesSpatialVAEConfig,
)
from src.utils import (
    LatentDiTGeneratorCFG,
    build_latent_generator,
    build_lr_scheduler,
    load_config,
    save_json,
    set_seed,
)

# ── Load frozen models (shared with base generator) ──────────────────

def load_selfies_vae(ckpt_path: str, device: torch.device):
    """Load a trained SELFIES VAE from checkpoint (frozen). Supports flat and spatial."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_type = ckpt.get("model_type", "flat")
    vocab = ckpt["vocab"]
    set_vocab(vocab)

    if model_type == "spatial":
        vae_cfg = SelfiesSpatialVAEConfig()
        for k, v in ckpt["vae_cfg"].items():
            if hasattr(vae_cfg, k):
                setattr(vae_cfg, k, v)
        vae_cfg.vocab_size = len(vocab)
        model = SelfiesSpatialVAE(vae_cfg).to(device)
    else:
        vae_cfg = SelfiesVAEConfig(**ckpt["vae_cfg"])
        model = SelfiesVAE(vae_cfg).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_frozen_phi(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    phi_cfg = ckpt["cfg"]
    model = LatentMAE(
        latent_dim=ckpt["latent_dim"],
        phi_dim=phi_cfg.get("phi_dim", 128),
        hidden_dim=phi_cfg.get("hidden_dim", 256),
        num_encoder_layers=phi_cfg.get("num_encoder_layers", 4),
        num_decoder_layers=phi_cfg.get("num_decoder_layers", 2),
        mask_ratio=phi_cfg.get("mask_ratio", 0.5),
        num_properties=phi_cfg.get("num_properties", 0),
        dropout=phi_cfg.get("dropout", 0.1),
        normalize_output=phi_cfg.get("normalize_output", True),
        multi_scale_dims=phi_cfg.get("multi_scale_dims", None),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    z_mean = ckpt["z_mean"].to(device)
    z_std = ckpt["z_std"].to(device)
    return model, z_mean, z_std


# ── Decoder-native feature extractor (Approach 1) ───────────────────

class RandomFeatureExtractor(nn.Module):
    """Random frozen MLP as φ — ablation to test if decoder structure matters.

    A 2-layer MLP with random weights, frozen forever. Same interface as
    DecoderFeatureExtractor but provides no learned molecular structure.
    If drift works well with this φ, then the decoder's learned structure
    is not important. If it fails, the decoder's semantic quality is essential.
    """

    def __init__(self, input_dim: int, output_dim: int = 512):
        super().__init__()
        self.feature_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )
        # Freeze all parameters
        for p in self.parameters():
            p.requires_grad_(False)

    @property
    def phi_dim(self):
        return self.feature_dim

    def extract_features(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    @property
    def num_layers(self) -> int:
        return 1


class DecoderFeatureExtractor(nn.Module):
    """Use VAE decoder hidden states as drifting observation space.

    Instead of a separately-trained MAE encoder φ, this uses the VAE's own
    decoder as the feature extractor.  The decoder already knows *how z maps
    to molecules*, so its hidden states encode richer molecular-structure
    information than the abstract z vector.

    Architecture:
        z → dec_z_proj(Linear 256→512) → expand [B,L,H] + pos_embed
          → TransformerEncoder(4 layers) → mean_pool → [B, 512]

    Key advantages over MAE φ:
      • Gradient ∂feat/∂z flows through the actual molecule-writing pathway.
      • 2.5× better noise robustness (R² drops 4% at σ=0.2 vs 20% for raw z).
      • No separate model to train/tune — just the frozen VAE decoder.

    The VAE parameters stay frozen; gradients flow *through* the decoder
    to z_gen exactly like a frozen perceptual-loss backbone.
    Uses gradient checkpointing to save activation memory.
    """

    def __init__(self, vae: SelfiesVAE):
        super().__init__()
        self.vae = vae
        self.feature_dim = vae.cfg.hidden_dim  # 512

    @property
    def phi_dim(self):
        """Compatibility with code that reads phi_model.phi_dim (e.g. logging)."""
        return self.feature_dim

    def _decoder_forward(self, h: torch.Tensor) -> torch.Tensor:
        """Wrapper for checkpoint compatibility."""
        return self.vae.decoder(h)

    def extract_features(self, z: torch.Tensor) -> torch.Tensor:
        """Extract mean-pooled decoder hidden states.

        Features are returned UN-normalized (raw norms ≈ 27).  The drift
        loss function should use ``normalize_distances=True`` to auto-
        calibrate the kernel temperature.  L2-normalization would destroy
        QED signal which resides partly in feature magnitude.

        Args:
            z: [B, latent_dim]  — raw latent vectors (pass-through when
               z_mean=0, z_std=1 so the ``(z - z_mean)/z_std`` normalisation
               in the training loop is an identity).
        Returns:
            [B, hidden_dim] mean-pooled decoder features.
        """
        z_proj = self.vae.dec_z_proj(z).unsqueeze(1)        # [B, 1, H]
        L = self.vae.cfg.max_len
        h = z_proj.expand(-1, L, -1) + self.vae.dec_pos     # [B, L, H]
        # Use gradient checkpointing when grad is active to save memory:
        # activations are recomputed during backward instead of stored.
        if torch.is_grad_enabled() and z.requires_grad:
            from torch.utils.checkpoint import checkpoint
            h_dec = checkpoint(self._decoder_forward, h, use_reentrant=False)
        else:
            h_dec = self.vae.decoder(h)                      # [B, L, H]
        return h_dec.mean(dim=1)                             # [B, H]

    def _decoder_layers_forward(self, h: torch.Tensor) -> list[torch.Tensor]:
        """Run decoder layers individually, collecting per-layer outputs."""
        outs = []
        for layer in self.vae.decoder.layers:
            h = layer(h)
            outs.append(h.mean(dim=1))   # [B, H] per layer
        return outs

    def extract_features_multi(self, z: torch.Tensor) -> list[torch.Tensor]:
        """Extract per-layer mean-pooled decoder features (Paper A.5).

        Returns a list of [B, hidden_dim] tensors, one per decoder layer.
        Each layer's features are used as an independent φ for drift loss,
        and the losses are summed (multi-scale drifting).
        """
        z_proj = self.vae.dec_z_proj(z).unsqueeze(1)        # [B, 1, H]
        L = self.vae.cfg.max_len
        h = z_proj.expand(-1, L, -1) + self.vae.dec_pos     # [B, L, H]
        if torch.is_grad_enabled() and z.requires_grad:
            from torch.utils.checkpoint import checkpoint
            # Checkpoint each layer individually for memory efficiency
            outs = []
            for layer in self.vae.decoder.layers:
                h = checkpoint(layer, h, use_reentrant=False)
                outs.append(h.mean(dim=1))  # [B, H]
            return outs
        else:
            return self._decoder_layers_forward(h)

    @property
    def num_layers(self) -> int:
        return len(self.vae.decoder.layers)

    def fit_property_head(self, z_data: torch.Tensor, props: torch.Tensor,
                          alpha: float = 1.0, batch_size: int = 512) -> None:
        """Fit a frozen linear property head on top of decoder features.

        Uses Ridge regression: prop ≈ W · φ(z) + b.
        After fitting, ``self.property_head`` is a frozen ``nn.Linear``.

        Args:
            z_data:  [N, latent_dim] z-normalized training data.
            props:   [N, num_props] normalized property targets.
            alpha:   Ridge regularisation strength.
            batch_size: for φ extraction (memory-limited decoder pass).
        """
        device = z_data.device
        # Extract φ features in batches (no grad needed)
        phi_parts = []
        with torch.no_grad():
            for i in range(0, z_data.shape[0], batch_size):
                phi_parts.append(self.extract_features(z_data[i:i + batch_size]))
        phi = torch.cat(phi_parts, dim=0)  # [N, feature_dim]

        # Ridge regression in float64 for stability
        X = phi.double()
        Y = props.double()
        X_mean = X.mean(0, keepdim=True)
        Y_mean = Y.mean(0, keepdim=True)
        Xc = X - X_mean
        Yc = Y - Y_mean
        XtX = Xc.T @ Xc
        reg = alpha * torch.eye(X.shape[1], dtype=torch.float64, device=device)
        W = torch.linalg.solve(XtX + reg, Xc.T @ Yc)  # [D, num_props]
        b = (Y_mean - X_mean @ W).squeeze(0)

        # Store as frozen nn.Linear so gradients flow through φ but not W
        head = nn.Linear(self.feature_dim, props.shape[1]).to(device)
        with torch.no_grad():
            head.weight.copy_(W.T.float())
            head.bias.copy_(b.float())
        for p in head.parameters():
            p.requires_grad_(False)
        self.property_head = head

        # Report quality
        with torch.no_grad():
            pred = head(phi)
        for c in range(props.shape[1]):
            from scipy.stats import pearsonr
            r, _ = pearsonr(pred[:, c].cpu().numpy(), props[:, c].cpu().numpy())
            mse = (pred[:, c] - props[:, c]).pow(2).mean().item()
            print(f"  [decoder prop_head] col {c}: Pearson r={r:.3f}, MSE={mse:.4f}")

    def fit_drift_projector(self, z_data: torch.Tensor, props: torch.Tensor,
                            proj_dim: int = 16, alpha: float = 1.0,
                            batch_size: int = 512, prop_col: int = 0) -> None:
        """Fit a QED-supervised low-dim projection for drift computation.

        The projection is constructed as:
          1. Ridge regression φ → QED gives weight vector w_qed (QED direction)
          2. Residual = φ - (φ @ w_qed) * w_qed
          3. PCA on residual → top (proj_dim - 1) components
          4. Projection matrix P = [w_qed, pca_1, ..., pca_{d-1}]

        Drift is then computed in P-projected space (proj_dim dimensions)
        where dim 0 is maximally QED-correlated.

        After fitting, ``self.drift_proj`` is a frozen parameter [D, proj_dim].

        Args:
            z_data:   [N, latent_dim] z-normalized training data.
            props:    [N, num_props] normalized property targets.
            proj_dim: target dimensionality for drift space (default 16).
            alpha:    Ridge regularization for QED direction.
            batch_size: for φ extraction.
            prop_col: which property column to use for supervision.
        """
        device = z_data.device
        # Extract φ features
        phi_parts = []
        with torch.no_grad():
            for i in range(0, z_data.shape[0], batch_size):
                phi_parts.append(self.extract_features(z_data[i:i + batch_size]))
        phi = torch.cat(phi_parts, dim=0).double()  # [N, D]
        qed = props[:, prop_col:prop_col + 1].double()  # [N, 1]

        D = phi.shape[1]
        phi_mean = phi.mean(0, keepdim=True)
        qed_mean = qed.mean(0, keepdim=True)
        Xc = phi - phi_mean
        Yc = qed - qed_mean

        # 1. Ridge regression → QED direction
        XtX = Xc.T @ Xc
        reg = alpha * torch.eye(D, dtype=torch.float64, device=device)
        w_qed = torch.linalg.solve(XtX + reg, Xc.T @ Yc)  # [D, 1]
        w_qed_unit = w_qed / w_qed.norm().clamp(min=1e-8)  # [D, 1]

        # Check QED R² in projected space
        qed_pred = Xc @ w_qed + qed_mean
        ss_res = (qed - qed_pred).pow(2).sum()
        ss_tot = Yc.pow(2).sum()
        r2_qed = (1.0 - ss_res / ss_tot.clamp(min=1e-8)).item()

        # 2. Project out QED direction from φ
        proj_on_qed = (Xc @ w_qed_unit) @ w_qed_unit.T  # [N, D]
        residual = Xc - proj_on_qed  # [N, D]

        # 3. PCA on residual → top (proj_dim - 1) components
        U, S, Vt = torch.linalg.svd(residual, full_matrices=False)
        pca_dirs = Vt[:proj_dim - 1].T  # [D, proj_dim-1]

        # Check variance explained
        total_var = residual.pow(2).sum()
        kept_var = S[:proj_dim - 1].pow(2).sum()
        var_pct = (kept_var / total_var.clamp(min=1e-8) * 100).item()

        # 4. Concatenate: P = [w_qed_unit, pca_dirs]  → [D, proj_dim]
        P = torch.cat([w_qed_unit, pca_dirs], dim=1)  # [D, proj_dim]

        # Verify orthogonality (should be ~0 since PCA residual ⊥ w_qed)
        orth_check = (P.T @ P - torch.eye(proj_dim, dtype=torch.float64, device=device)).abs().max().item()

        # Store as frozen parameter
        self.register_buffer("drift_proj", P.float())
        self.register_buffer("drift_proj_phi_mean", phi_mean.float())

        # Compute projected distances for temperature calibration
        phi_proj = ((phi - phi_mean) @ P).float()  # [N, proj_dim]
        idx = torch.randperm(len(phi_proj))[:2048]
        sample = phi_proj[idx]
        dists = torch.cdist(sample, sample, p=2)
        mask = ~torch.eye(len(sample), dtype=torch.bool, device=device)
        mean_dist = dists[mask].mean().item()

        print(f"  [drift_projector] dim={proj_dim}, QED R²={r2_qed:.3f}")
        print(f"  [drift_projector] residual PCA variance: {var_pct:.1f}% in {proj_dim-1} dims")
        print(f"  [drift_projector] orthogonality check: max|P^T P - I| = {orth_check:.2e}")
        print(f"  [drift_projector] projected mean pairwise dist = {mean_dist:.2f}")
        print(f"  [drift_projector] suggested τ: 1%={mean_dist*0.01:.3f}, "
              f"5%={mean_dist*0.05:.3f}, 10%={mean_dist*0.10:.3f}")

    def project_for_drift(self, phi: torch.Tensor) -> torch.Tensor:
        """Project φ features into drift space using fitted projector.

        Args:
            phi: [B, D] decoder features (raw, with grad if needed)
        Returns:
            [B, proj_dim] projected features for drift computation
        """
        return (phi - self.drift_proj_phi_mean) @ self.drift_proj


# ── Property prediction head ─────────────────────────────────────────

class PropertyHead(nn.Module):
    """MLP: z_latent → predicted property (for conditional loss)."""
    def __init__(self, latent_dim: int, cond_dim: int, hidden_dim: int = 256,
                 num_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(latent_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, cond_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class LinearPropertyHead(nn.Module):
    """
    Linear property head: z → w·z + b.
    
    Key advantage over MLP: a linear model has NO out-of-distribution regions.
    The generator cannot find "sweet spots" where the model is miscalibrated,
    because the error surface is convex and consistent everywhere.
    
    The gradient ∂loss/∂z = 2(pred - target) × w is always along a FIXED 
    direction w, preventing the generator from gaming the predictor.
    """
    def __init__(self, latent_dim: int, cond_dim: int):
        super().__init__()
        self.linear = nn.Linear(latent_dim, cond_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)

    @classmethod
    def from_ridge_regression(cls, z_data: torch.Tensor, props: torch.Tensor,
                               alpha: float = 1.0) -> "LinearPropertyHead":
        """
        Initialize weights from Ridge regression on real data.
        Gives the optimal linear predictor: prop ≈ w·z + b.
        """
        latent_dim = z_data.shape[1]
        cond_dim = props.shape[1]
        head = cls(latent_dim, cond_dim)
        
        # Ridge regression: w = (Z^T Z + αI)^{-1} Z^T y
        # Move to float64 for numerical stability
        Z = z_data.double()
        Y = props.double()
        Z_mean = Z.mean(dim=0, keepdim=True)
        Y_mean = Y.mean(dim=0, keepdim=True)
        Zc = Z - Z_mean
        Yc = Y - Y_mean
        
        # Closed-form solution
        ZtZ = Zc.T @ Zc  # [D, D]
        ZtY = Zc.T @ Yc  # [D, cond_dim]
        reg = alpha * torch.eye(latent_dim, dtype=torch.float64, device=z_data.device)
        W = torch.linalg.solve(ZtZ + reg, ZtY)  # [D, cond_dim]
        b = (Y_mean - Z_mean @ W).squeeze(0)     # [cond_dim]
        
        with torch.no_grad():
            head.linear.weight.copy_(W.T.float())
            head.linear.bias.copy_(b.float())
        
        return head# ── Conditional drift: match data targets to conditions ──────────────

# NOTE: The old conditional_drift_loss (top-K → random pick 1) is replaced
# by group-based training in the main loop, faithful to the paper's per-class
# Algorithm 1 with N_pos matched positives + N_unc unconditional negatives.


# ── Evaluation for conditional generation ────────────────────────────

@torch.no_grad()
def evaluate_conditional(
    generator: LatentDiTGeneratorCFG,
    vae: SelfiesVAE,
    train_smiles: set[str],
    prop_mean: torch.Tensor,
    prop_std: torch.Tensor,
    prop_name: str,
    num_samples: int,
    batch_size: int,
    device: torch.device,
    alpha: float = 1.0,
    target_values: list[float] | None = None,
    bin_edges: list[float] | None = None,       # bin edges for class→QED mapping
    bin_centers_raw: list[float] | None = None,  # raw (un-normalized) bin centers
    two_pass_cfg: bool = False,                  # use two-pass CFG: z = z_u + α(z_c - z_u) (violates 1-NFE!)
    ref_smiles_list: list[str] | None = None,    # reference SMILES for FCD / NN metrics
    cond_dim: int = 1,                           # full condition dimensionality
    prop_idx: int = 0,                           # which condition dimension to vary
) -> dict:
    """
    Evaluate conditional generation quality.
    
    Generates molecules at specific target property values and measures:
    - VUN (validity, uniqueness, novelty)
    - Spearman rank correlation between target and actual property
    - MAE between target and actual property
    - Property distribution statistics
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED as QED_mod

    generator.eval()

    # Default target values: sweep across data distribution
    if target_values is None:
        # 5 evenly spaced quantiles of the property
        target_values = np.linspace(-2.0, 2.0, 5).tolist()  # normalized

    use_class_cond = getattr(generator, 'num_classes', 0) > 0

    # If using class conditioning, build target→bin mapping
    def _target_to_bin_id(target_raw: float) -> int:
        """Map a raw property value to the nearest bin ID."""
        if bin_edges is not None and len(bin_edges) > 1:
            for b_idx in range(len(bin_edges) - 1):
                if target_raw < bin_edges[b_idx + 1]:
                    return b_idx
            return len(bin_edges) - 2  # last bin
        return 0

    all_results = {}
    all_targets = []
    all_actual = []
    all_smiles_list = []
    per_bin_canonical: dict[str, list[str]] = {}  # target label → list of canonical SMILES

    for target_norm in target_values:
        target_raw = target_norm * prop_std.item() + prop_mean.item()

        if use_class_cond:
            # Class embedding mode: convert target to bin ID
            bin_id = _target_to_bin_id(target_raw)
            cond = torch.full((batch_size,), bin_id, device=device, dtype=torch.long)
            # Use bin center as the official "target" for metric computation
            if bin_centers_raw is not None:
                target_raw = bin_centers_raw[bin_id]
        else:
            if cond_dim > 1:
                cond = torch.zeros((batch_size, cond_dim), device=device)
                cond[:, prop_idx] = target_norm
            else:
                cond = torch.full((batch_size, 1), target_norm, device=device)

        smiles_batch = []
        while len(smiles_batch) < num_samples // len(target_values):
            n = min(batch_size, num_samples // len(target_values) - len(smiles_batch))
            noise = torch.randn(n, generator.noise_dim, device=device)
            c = cond[:n]
            if two_pass_cfg and alpha > 1.0:
                # Two-pass CFG: z = z_uncond + α * (z_cond - z_uncond)
                z_cond = generator(noise, cond=c, alpha=alpha)
                # Unconditional pass: use null class (num_classes idx) or None
                if use_class_cond:
                    null_cond = torch.full((n,), generator.null_class_id,
                                           device=device, dtype=torch.long)
                    z_uncond = generator(noise, cond=null_cond, alpha=1.0)
                else:
                    z_uncond = generator(noise, cond=None, alpha=1.0)
                z = z_uncond + alpha * (z_cond - z_uncond)
            else:
                z = generator(noise, cond=c, alpha=alpha)
            smi = vae.sample_smiles(z, temperature=0.0)
            smiles_batch.extend(smi)

        smiles_batch = smiles_batch[:num_samples // len(target_values)]

        # Compute actual property value (match the conditioned property)
        bin_label = f"target={target_raw:.2f}"
        bin_canonical_list: list[str] = []
        for smi in smiles_batch:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                try:
                    pn = prop_name.lower()
                    if pn in ("qed",):
                        actual_val = QED_mod.qed(mol)
                    elif pn in ("logp", "mollogp"):
                        actual_val = Descriptors.MolLogP(mol)
                    elif pn in ("sa_score", "sa"):
                        from rdkit.Chem import RDConfig
                        import os, sys
                        sa_path = os.path.join(RDConfig.RDContribDir, 'SA_Score')
                        if sa_path not in sys.path:
                            sys.path.insert(0, sa_path)
                        import sascorer
                        actual_val = sascorer.calculateScore(mol)
                    elif pn in ("molwt", "mw"):
                        actual_val = Descriptors.MolWt(mol)
                    else:
                        actual_val = Descriptors.MolLogP(mol)  # fallback
                except Exception:
                    continue
                if actual_val is None:
                    continue
                can_smi = Chem.MolToSmiles(mol)
                bin_canonical_list.append(can_smi)
                all_targets.append(target_raw)
                all_actual.append(actual_val)
                all_smiles_list.append(smi)
        per_bin_canonical[bin_label] = bin_canonical_list

    # Compute metrics
    from rdkit import Chem
    valid_mols = [Chem.MolFromSmiles(s) for s in all_smiles_list]
    valid_mols = [m for m in valid_mols if m is not None]
    canonical = set(Chem.MolToSmiles(m) for m in valid_mols)

    v = len(valid_mols) / max(len(all_smiles_list), 1)
    u = len(canonical) / max(len(valid_mols), 1)
    novel = sum(1 for s in canonical if s not in train_smiles)
    n = novel / max(len(canonical), 1)

    results = {
        "alpha": alpha,
        "validity": v,
        "uniqueness": u,
        "novelty": n,
        "vun": v * u * n,
        "num_generated": len(all_smiles_list),
    }

    # Internal diversity: mean pairwise Tanimoto distance (1 - similarity)
    try:
        from rdkit.Chem import AllChem, DataStructs
        fps = []
        for m in valid_mols[:500]:  # cap for speed
            fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
            fps.append(fp)
        if len(fps) >= 2:
            sim_sum, sim_count = 0.0, 0
            # Sample pairs for speed
            n_pairs = min(5000, len(fps) * (len(fps) - 1) // 2)
            import random
            for _ in range(n_pairs):
                i, j = random.sample(range(len(fps)), 2)
                sim_sum += DataStructs.TanimotoSimilarity(fps[i], fps[j])
                sim_count += 1
            avg_sim = sim_sum / sim_count
            results["int_div"] = float(1.0 - avg_sim)  # internal diversity
            results["avg_tanimoto"] = float(avg_sim)
    except Exception:
        pass

    if len(all_targets) >= 10:
        targets = np.array(all_targets)
        actuals = np.array(all_actual)
        from scipy.stats import spearmanr, pearsonr
        rho, p_val = spearmanr(targets, actuals)
        r, _ = pearsonr(targets, actuals)
        mae = np.mean(np.abs(targets - actuals))
        results["spearman_rho"] = float(rho)
        results["spearman_pval"] = float(p_val)
        results["pearson_r"] = float(r)
        results["mae"] = float(mae)
        results["target_mean"] = float(targets.mean())
        results["actual_mean"] = float(actuals.mean())
        results["actual_std"] = float(actuals.std())

        # Per-target-bin stats
        per_bin = {}
        target_bins = sorted(set(np.round(targets, 2)))
        for t in target_bins:
            mask = np.abs(targets - t) < 0.01
            if mask.sum() > 0:
                per_bin[f"target={t:.2f}"] = {
                    "n": int(mask.sum()),
                    "actual_mean": float(actuals[mask].mean()),
                    "actual_std": float(actuals[mask].std()),
                    "mae": float(np.abs(actuals[mask] - t).mean()),
                }
        results["per_target"] = per_bin

        # Compact per-bin summary string for logging
        bin_strs = []
        for t in target_bins:
            key = f"target={t:.2f}"
            if key in per_bin:
                b = per_bin[key]
                bin_strs.append(f"{t:.2f}→{b['actual_mean']:.2f}")
        results["bin_summary"] = " ".join(bin_strs)

        # ── Slope: linear regression actual = slope * target + intercept ──
        if len(targets) >= 20:
            slope, intercept = np.polyfit(targets, actuals, 1)
            results["slope"] = float(slope)
            results["intercept"] = float(intercept)

        # ── Tail-MAE: MAE at the lowest and highest target bins ──
        if len(target_bins) >= 3:
            lo_bin, hi_bin = target_bins[0], target_bins[-1]
            for label, tb in [("tail_lo", lo_bin), ("tail_hi", hi_bin)]:
                mask = np.abs(targets - tb) < 0.01
                if mask.sum() > 0:
                    results[f"mae_{label}"] = float(np.abs(actuals[mask] - tb).mean())
            # average tail-MAE
            if f"mae_tail_lo" in results and f"mae_tail_hi" in results:
                results["mae_tail_avg"] = (results["mae_tail_lo"] + results["mae_tail_hi"]) / 2

    # ── Per-bin uniqueness (collapse detection) ──
    per_bin_uniqueness = {}
    for blabel, smis in per_bin_canonical.items():
        if len(smis) > 0:
            per_bin_uniqueness[blabel] = len(set(smis)) / len(smis)
    results["per_bin_uniqueness"] = per_bin_uniqueness
    if per_bin_uniqueness:
        results["min_bin_uniqueness"] = min(per_bin_uniqueness.values())

    # ── Scaffold diversity ──
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold
        scaffolds = set()
        for m in valid_mols:
            try:
                core = MurckoScaffold.GetScaffoldForMol(m)
                scaffolds.add(Chem.MolToSmiles(core))
            except Exception:
                pass
        if valid_mols:
            results["scaffold_diversity"] = len(scaffolds) / len(valid_mols)
            results["n_unique_scaffolds"] = len(scaffolds)
    except Exception:
        pass

    # ── NN similarity to training set (memorization check) ──
    try:
        train_list = list(train_smiles)
        if len(canonical) >= 10 and len(train_list) >= 10:
            nn_sim = _compute_snn(list(canonical)[:500], train_list[:5000])
            results["nn_sim_mean"] = nn_sim
    except Exception:
        pass

    # ── FCD (if reference SMILES provided) ──
    if ref_smiles_list and len(ref_smiles_list) > 0 and len(canonical) >= 50:
        try:
            from fcd_torch import FCD as FCD_cls
            fcd_calc = FCD_cls(device=str(device), n_jobs=1)
            fcd_val = fcd_calc(ref_smiles_list[:10000], list(canonical)[:5000])
            results["fcd"] = float(fcd_val)
        except Exception:
            pass

    return results


def _compute_frag_similarity(gen_smiles: list[str], ref_smiles: list[str]) -> float:
    """Cosine similarity of BRICS fragment frequency vectors."""
    from rdkit import Chem
    from rdkit.Chem import BRICS
    from collections import Counter

    def _frag_counts(smi_list):
        counts = Counter()
        for smi in smi_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                try:
                    frags = BRICS.BRICSDecompose(mol, returnMols=False)
                    for f in frags:
                        counts[f] += 1
                except Exception:
                    pass
        return counts

    gen_c = _frag_counts(gen_smiles)
    ref_c = _frag_counts(ref_smiles)
    all_keys = set(gen_c) | set(ref_c)
    if not all_keys:
        return 0.0
    g = np.array([gen_c.get(k, 0) for k in all_keys], dtype=float)
    r = np.array([ref_c.get(k, 0) for k in all_keys], dtype=float)
    denom = np.linalg.norm(g) * np.linalg.norm(r)
    return float(np.dot(g, r) / denom) if denom > 0 else 0.0


def _compute_scaf_similarity(gen_smiles: list[str], ref_smiles: list[str]) -> float:
    """Cosine similarity of Murcko scaffold frequency vectors."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from collections import Counter

    def _scaf_counts(smi_list):
        counts = Counter()
        for smi in smi_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                try:
                    core = MurckoScaffold.GetScaffoldForMol(mol)
                    counts[Chem.MolToSmiles(core)] += 1
                except Exception:
                    pass
        return counts

    gen_c = _scaf_counts(gen_smiles)
    ref_c = _scaf_counts(ref_smiles)
    all_keys = set(gen_c) | set(ref_c)
    if not all_keys:
        return 0.0
    g = np.array([gen_c.get(k, 0) for k in all_keys], dtype=float)
    r = np.array([ref_c.get(k, 0) for k in all_keys], dtype=float)
    denom = np.linalg.norm(g) * np.linalg.norm(r)
    return float(np.dot(g, r) / denom) if denom > 0 else 0.0


def _compute_snn(gen_smiles: list[str], ref_smiles: list[str],
                 n_ref_sample: int = 5000) -> float:
    """Average Tanimoto similarity of each generated mol to its nearest neighbor
    in the reference set."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    import random as _rnd

    ref_sample = _rnd.sample(ref_smiles, min(n_ref_sample, len(ref_smiles)))
    ref_fps = []
    for smi in ref_sample:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            ref_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    if not ref_fps:
        return 0.0

    nn_sims = []
    for smi in gen_smiles[:1000]:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
            nn_sims.append(max(sims))
    return float(np.mean(nn_sims)) if nn_sims else 0.0


def _compute_kl_div(gen_vals: list[float], ref_vals: list[float],
                    n_bins: int = 100) -> float:
    """Discretized KL divergence D_KL(gen || ref)."""
    lo = min(min(gen_vals), min(ref_vals))
    hi = max(max(gen_vals), max(ref_vals))
    bins = np.linspace(lo - 1e-6, hi + 1e-6, n_bins + 1)
    p = np.histogram(gen_vals, bins=bins)[0].astype(float) + 1e-8
    q = np.histogram(ref_vals, bins=bins)[0].astype(float) + 1e-8
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def _compute_filters_pass_rate(smiles_list: list[str]) -> float:
    """Fraction of molecules passing PAINS filters."""
    from rdkit import Chem
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog(params)
    n_pass, n_total = 0, 0
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            n_total += 1
            if not catalog.HasMatch(mol):
                n_pass += 1
    return n_pass / max(n_total, 1)


@torch.no_grad()
def evaluate_unconditional(
    generator: LatentDiTGeneratorCFG,
    vae: SelfiesVAE,
    train_smiles: set[str],
    num_samples: int,
    batch_size: int,
    device: torch.device,
    ref_smiles_list: list[str] | None = None,
) -> dict:
    """Comprehensive unconditional evaluation with distribution metrics.

    Metrics: V/U/N, IntDiv, QED/MW/LogP/SA (mean±std),
    FCD, SNN, Frag, Scaf, KL(QED), KL(MW), KL(LogP), Filters.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED as QED_mod

    generator.eval()
    all_smiles = []

    while len(all_smiles) < num_samples:
        n = min(batch_size, num_samples - len(all_smiles))
        noise = torch.randn(n, generator.noise_dim, device=device)
        z = generator(noise, cond=None, alpha=1.0)
        smi = vae.sample_smiles(z, temperature=0.0)
        all_smiles.extend(smi)
    all_smiles = all_smiles[:num_samples]

    valid_set = []
    canonical_set = set()
    for smi in all_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_set.append(Chem.MolToSmiles(mol))  # canonical
            canonical_set.add(valid_set[-1])

    v = len(valid_set) / num_samples
    u = len(canonical_set) / max(len(valid_set), 1)
    novel = sum(1 for s in canonical_set if s not in train_smiles)
    n = novel / max(len(canonical_set), 1)

    result = {
        "validity": v,
        "uniqueness": u,
        "novelty": n,
        "vun": v * u * n,
    }

    # ── Property distributions ──
    qeds, logps, mws, sas = [], [], [], []
    for smi in valid_set:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            try:
                qeds.append(QED_mod.qed(mol))
                logps.append(Descriptors.MolLogP(mol))
                mws.append(Descriptors.MolWt(mol))
            except Exception:
                pass
            try:
                from rdkit.Chem import RDConfig
                import os as _os, sys as _sys
                sa_path = _os.path.join(RDConfig.RDContribDir, 'SA_Score')
                if sa_path not in _sys.path:
                    _sys.path.insert(0, sa_path)
                import sascorer
                sc = sascorer.calculateScore(mol)
                if sc is not None:
                    sas.append(sc)
            except Exception:
                pass
    if qeds:
        result["qed_mean"] = float(np.mean(qeds))
        result["qed_std"] = float(np.std(qeds))
        result["logp_mean"] = float(np.mean(logps))
        result["logp_std"] = float(np.std(logps))
        result["mw_mean"] = float(np.mean(mws))
        result["mw_std"] = float(np.std(mws))
    if sas:
        sas = [x for x in sas if x is not None]
    if sas:
        result["sa_mean"] = float(np.mean(sas))
        result["sa_std"] = float(np.std(sas))

    # ── Internal diversity ──
    try:
        from rdkit.Chem import AllChem, DataStructs
        fps = []
        for smi in valid_set[:500]:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
        if len(fps) >= 2:
            import random
            sim_sum, sim_count = 0.0, 0
            n_pairs = min(5000, len(fps) * (len(fps) - 1) // 2)
            for _ in range(n_pairs):
                i, j = random.sample(range(len(fps)), 2)
                sim_sum += DataStructs.TanimotoSimilarity(fps[i], fps[j])
                sim_count += 1
            result["int_div"] = float(1.0 - sim_sum / sim_count)
    except Exception:
        pass

    # ── Reference-dependent distributional metrics ──
    if ref_smiles_list and len(ref_smiles_list) > 0 and len(valid_set) >= 50:
        # FCD (Fréchet ChemNet Distance)
        try:
            from fcd_torch import FCD as FCD_cls
            fcd_calc = FCD_cls(device=str(device), n_jobs=1)
            fcd_val = fcd_calc(ref_smiles_list[:10000], valid_set[:10000])
            result["fcd"] = float(fcd_val)
        except Exception:
            pass

        # SNN (Similarity to Nearest Neighbor)
        try:
            result["snn"] = _compute_snn(valid_set, ref_smiles_list)
        except Exception:
            pass

        # Fragment similarity
        try:
            result["frag"] = _compute_frag_similarity(valid_set[:2000], ref_smiles_list[:10000])
        except Exception:
            pass

        # Scaffold similarity
        try:
            result["scaf"] = _compute_scaf_similarity(valid_set[:2000], ref_smiles_list[:10000])
        except Exception:
            pass

        # KL divergences on properties
        ref_qeds, ref_logps, ref_mws = [], [], []
        for smi in ref_smiles_list[:10000]:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                try:
                    ref_qeds.append(QED_mod.qed(mol))
                    ref_logps.append(Descriptors.MolLogP(mol))
                    ref_mws.append(Descriptors.MolWt(mol))
                except Exception:
                    pass
        if qeds and ref_qeds:
            result["kl_qed"] = _compute_kl_div(qeds, ref_qeds)
            result["kl_logp"] = _compute_kl_div(logps, ref_logps)
            result["kl_mw"] = _compute_kl_div(mws, ref_mws)

        # Filters (PAINS)
        try:
            result["filters"] = _compute_filters_pass_rate(valid_set[:2000])
        except Exception:
            pass

    return result


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # ── DDP setup (auto-detect from torchrun env vars) ──
    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = (rank == 0)

    cfg = load_config(args.config)
    seed = int(cfg.get("experiment", {}).get("seed", 42)) + rank
    set_seed(seed)

    out_dir = Path(cfg["experiment"]["output_dir"])
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_json(cfg, out_dir / "resolved_config.json")
    if ddp:
        dist.barrier()

    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    if is_main:
        print(f"[device] {device}" + (f" (DDP world_size={world_size})" if ddp else ""))

    # ── Load frozen SELFIES VAE ──
    vae = load_selfies_vae(cfg["vae"]["checkpoint"], device)
    latent_dim = vae.cfg.latent_dim
    if is_main:
        print(f"[vae] latent_dim={latent_dim}, vocab={vae.cfg.vocab_size}")

    # ── Load frozen feature extractor (φ or VAE-decoder) ──
    loss_cfg_pre = cfg.get("loss", {})
    need_phi = (float(loss_cfg_pre.get("lambda_drift", 1.0)) > 0 or
                float(loss_cfg_pre.get("lambda_drift_struct", 0.0)) > 0 or
                float(loss_cfg_pre.get("lambda_decoupled_drift", 0.0)) > 0)
    feature_mode = cfg.get("feature_space", {}).get("mode", "phi")
    phi_model, z_mean, z_std = None, None, None
    decoder_phi_model = None  # optional auxiliary decoder feature extractor

    if feature_mode == "random":
        # ── Ablation B1: Random frozen MLP as φ ──
        phi_model = RandomFeatureExtractor(latent_dim, 512).to(device)
        z_mean = torch.zeros(latent_dim, device=device)
        z_std = torch.ones(latent_dim, device=device)
        need_phi = True
        if is_main:
            print(f"[feature] mode=random: frozen random MLP {latent_dim}→512d")
    elif feature_mode == "decoder":
        # ── Approach 1: VAE decoder hidden states as drifting features ──
        phi_model = DecoderFeatureExtractor(vae)
        z_mean = torch.zeros(latent_dim, device=device)
        z_std = torch.ones(latent_dim, device=device)
        need_phi = True  # ensure φ pre-computation path runs
        if is_main:
            print(f"[feature] mode=decoder: VAE decoder mean-pool → {vae.cfg.hidden_dim}d features")
    elif feature_mode == "phi_decoder":
        # ── Approach 3: φ v9 primary + decoder features auxiliary ──
        # φ: primary multi-scale drift backbone
        phi_model, z_mean, z_std = load_frozen_phi(cfg["phi"]["checkpoint"], device)
        # Decoder: auxiliary structural features for decoder-space drift
        decoder_phi_model = DecoderFeatureExtractor(vae)
        need_phi = True
        if is_main:
            print(f"[feature] mode=phi_decoder: φ primary (multiscale)"
                  f" + decoder auxiliary ({vae.cfg.hidden_dim}d)")
    elif need_phi and cfg.get("phi", {}).get("checkpoint"):
        phi_model, z_mean, z_std = load_frozen_phi(cfg["phi"]["checkpoint"], device)
        if is_main:
            print(f"[φ_prop] loaded")
    else:
        z_mean = torch.zeros(latent_dim, device=device)
        z_std = torch.ones(latent_dim, device=device)
        if is_main:
            print(f"[φ] skipped (pure z-drift mode)")

    # ── Optional: Load frozen φ_struct (manifold anchoring, replaces zmatch) ──
    phi_struct_model = None
    if "phi_struct" in cfg:
        phi_struct_model, _, _ = load_frozen_phi(cfg["phi_struct"]["checkpoint"], device)
        if is_main:
            print(f"[φ_struct] loaded (manifold anchoring)")

    # ── Load latent cache ──
    cache = torch.load(cfg["data"]["latent_cache_path"], map_location=device, weights_only=False)
    z_data_train = cache["train"].to(device)
    z_data_val = cache["val"].to(device)
    train_smiles_list = cache.get("train_smiles", [])
    prop_names_all = cache.get("prop_names", [])
    if is_main:
        print(f"[data] z_train={z_data_train.shape}")

    # Build canonical train SMILES set
    from rdkit import Chem
    train_smiles = set()
    for smi in train_smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            train_smiles.add(Chem.MolToSmiles(mol))
    if is_main:
        print(f"[data] train_smiles: {len(train_smiles)}")

    # ── Load & select properties ──
    raw_props = cache["train_props"].to(device)
    raw_props_val = cache["val_props"].to(device)
    prop_indices = cfg.get("data", {}).get("prop_indices", None)
    if prop_indices is not None:
        raw_props = raw_props[:, prop_indices]
        raw_props_val = raw_props_val[:, prop_indices]
        sel_prop_names = [prop_names_all[i] for i in prop_indices]
    else:
        sel_prop_names = list(prop_names_all)

    # Handle NaN
    for col in range(raw_props.shape[1]):
        valid = ~torch.isnan(raw_props[:, col])
        if valid.any():
            raw_props[~valid, col] = raw_props[valid, col].mean()
        valid_v = ~torch.isnan(raw_props_val[:, col])
        if valid_v.any():
            raw_props_val[~valid_v, col] = raw_props_val[valid_v, col].mean()

    prop_mean = raw_props.mean(dim=0, keepdim=True)
    prop_std = raw_props.std(dim=0, keepdim=True).clamp(min=1e-6)
    props_train = (raw_props - prop_mean) / prop_std  # [N, cond_dim] normalized
    props_val = (raw_props_val - prop_mean) / prop_std
    cond_dim = props_train.shape[1]
    if is_main:
        print(f"[cfg] {cond_dim} properties: {sel_prop_names}")
    if is_main:
        print(f"[cfg] raw range: mean={raw_props.mean(0).cpu().numpy()}, std={raw_props.std(0).cpu().numpy()}")

    # ── Parse loss config ──
    loss_cfg = cfg.get("loss", {})
    lambda_drift = float(loss_cfg.get("lambda_drift", 1.0))
    lambda_moment = float(loss_cfg.get("lambda_moment", 0.0))
    lambda_zdiv = float(loss_cfg.get("lambda_zdiv", 0.0))
    zdiv_margin = float(loss_cfg.get("zdiv_margin", 3.0))
    zdiv_topk = int(loss_cfg.get("zdiv_topk", 5))
    lambda_znorm = float(loss_cfg.get("lambda_znorm", 0.0))
    lambda_prop = float(loss_cfg.get("lambda_prop", 0.0))  # optional property head
    drift_temperatures = loss_cfg.get("temperatures", [0.02, 0.05, 0.2])
    zmatch_mode = str(loss_cfg.get("zmatch_mode", "centroid"))  # centroid or nn_soft
    zmatch_temp = float(loss_cfg.get("zmatch_temp", 1.0))  # temperature for nn_soft mode
    lambda_zcontrast = float(loss_cfg.get("lambda_zcontrast", 0.0))
    zcontrast_margin = float(loss_cfg.get("zcontrast_margin", 0.5))
    lambda_zdrift = float(loss_cfg.get("lambda_zdrift", 0.0))
    zdrift_temperatures = loss_cfg.get("zdrift_temperatures", None)  # None → use same as φ-drift
    lambda_phidiv = float(loss_cfg.get("lambda_phidiv", 0.0))  # φ-space repulsion
    phidiv_margin = float(loss_cfg.get("phidiv_margin", 5.0))  # data φ-pairwise~17.9
    phidiv_topk = int(loss_cfg.get("phidiv_topk", 5))
    lambda_drift_struct = float(loss_cfg.get("lambda_drift_struct", 0.0))  # multi-φ structural drift
    drift_struct_n_pos = int(loss_cfg.get("drift_struct_n_pos", 128))  # kNN positives for struct drift
    drift_struct_n_unc = int(loss_cfg.get("drift_struct_n_unc", 32))   # unconditional negatives for struct drift
    drift_normalize = bool(loss_cfg.get("drift_normalize", True))  # Per-temp drift normalization (Paper Eq.23-25)
    drift_normalize_dist = bool(loss_cfg.get("drift_normalize_dist", True))  # Global distance normalization (Paper A.6)
    drift_attraction_scale = float(loss_cfg.get("drift_attraction_scale", 1.0))
    drift_repulsion_scale = float(loss_cfg.get("drift_repulsion_scale", 1.0))
    lambda_decoupled_drift = float(loss_cfg.get("lambda_decoupled_drift", 0.0))  # φ-oracle + z-gradient drift
    decoupled_drift_temps = loss_cfg.get("decoupled_drift_temps", None)  # temperatures for decoupled drift
    drift_knn_k = loss_cfg.get("drift_knn_k", None)  # kNN-restricted V⁺: only attend to k-nearest positives
    if drift_knn_k is not None:
        drift_knn_k = int(drift_knn_k)
    stop_grad_drift = bool(loss_cfg.get("stop_grad_drift", False))  # B3 ablation: detach z before drift
    if is_main:
        print(f"[loss] λ_drift={lambda_drift}, λ_moment={lambda_moment}, "
          f"λ_zdiv={lambda_zdiv}, λ_znorm={lambda_znorm}, λ_prop={lambda_prop}")
    if stop_grad_drift and is_main:
        print(f"[loss] ⚠ stop_grad_drift=True: drift gradients will NOT flow to generator")
    if drift_knn_k is not None and is_main:
        print(f"[loss] drift kNN restriction: V⁺ limited to k={drift_knn_k} nearest positives")
    if (drift_attraction_scale, drift_repulsion_scale) != (1.0, 1.0) and is_main:
        print(
            "[loss] destructive drift scales: "
            f"attraction={drift_attraction_scale}, repulsion={drift_repulsion_scale}"
        )
    if lambda_phidiv > 0 and is_main:
        print(f"[loss] λ_phidiv={lambda_phidiv}, phidiv_margin={phidiv_margin}, phidiv_topk={phidiv_topk}")
    if not drift_normalize or not drift_normalize_dist:
        if is_main:
            print(f"[loss] drift_normalize={drift_normalize}, drift_normalize_dist={drift_normalize_dist}")
    if lambda_drift_struct > 0:
        if is_main:
            print(f"[loss] λ_drift_struct={lambda_drift_struct} (structural drift, n_pos={drift_struct_n_pos}, n_unc={drift_struct_n_unc})")
    if is_main:
        print(f"[loss] λ_zcontrast={lambda_zcontrast}, zcontrast_margin={zcontrast_margin}")
    if is_main:
        print(f"[loss] λ_zdrift={lambda_zdrift}, zdrift_temperatures={zdrift_temperatures}")
    if is_main:
        print(f"[loss] drift temperatures τ={drift_temperatures}")
    if is_main:
        print(f"[loss] zmatch_mode={zmatch_mode}, zmatch_temp={zmatch_temp}")

    # ── Parse CFG config (Paper Section 3.5, A.7, Table 8) ──
    cfg_section = cfg.get("cfg", {})
    n_groups = int(cfg_section.get("n_groups", 16))       # N_c: condition groups per step
    n_gen = int(cfg_section.get("n_gen", 32))              # N_neg per group
    n_pos = int(cfg_section.get("n_pos", 64))              # N_pos per group
    n_unc = int(cfg_section.get("n_unc", 16))              # N_unc unconditional negatives
    positive_mode = str(cfg_section.get("positive_mode", "prop"))  # "prop" (default) or "phi" (φ-space kNN)
    knn_pool_factor = int(cfg_section.get("knn_pool_factor", 4))   # stochastic kNN: sample n_pos from top-(n_pos*factor)
    alpha_power = float(cfg_section.get("alpha_power", 3)) # p(α) ∝ α^{-power}
    alpha_min = float(cfg_section.get("alpha_min", 1.0))
    alpha_max = float(cfg_section.get("alpha_max", 4.0))
    effective_batch = n_groups * n_gen
    # DDP: distribute groups across GPUs — each GPU processes its share,
    # allreduce averages gradients → equivalent to n_groups total per step.
    local_n_groups = max(1, n_groups // world_size) if ddp else n_groups
    local_batch = local_n_groups * n_gen
    if is_main:
        print(f"[cfg] N_c={n_groups}, N_gen={n_gen}, N_pos={n_pos}, N_unc={n_unc}, positive_mode={positive_mode}"
              + (f", knn_pool_factor={knn_pool_factor}" if positive_mode == "phi" else ""))
    if is_main and ddp:
        print(f"[DDP] distributing groups: {n_groups} total → {local_n_groups} per GPU × {world_size} GPUs")
    if is_main:
        print(f"[cfg] α ~ p(α)∝α^{{-{alpha_power}}} on [{alpha_min},{alpha_max}]")
    if is_main:
        print(f"[cfg] effective batch = {effective_batch}")

    # ── Condition binning (discretize continuous props into "classes") ──
    bin_cfg = cfg.get("cond_binning", {})
    use_bins = bool(bin_cfg.get("enabled", False))
    n_bins = int(bin_cfg.get("n_bins", 20))
    bin_method = str(bin_cfg.get("method", "quantile"))  # "quantile" or "equal"
    bin_indices = None   # list of LongTensor, one per bin
    bin_centers_t = None  # [n_bins, cond_dim]
    bin_edges_raw = None  # raw (un-normalized) bin edges for eval mapping
    bin_centers_raw = None  # raw (un-normalized) bin centers for eval
    if use_bins:
        # Build bins for 1D property (extend to multi-dim later if needed)
        prop_vals = props_train[:, 0]  # first property
        n_total = prop_vals.shape[0]
        if bin_method == "quantile":
            # Equal-count bins
            sorted_vals, _ = prop_vals.sort()
            edges = [sorted_vals[0].item()]
            for b in range(1, n_bins):
                edges.append(sorted_vals[b * n_total // n_bins].item())
            edges.append(sorted_vals[-1].item() + 1e-6)
        else:
            # Equal-width bins
            edges = torch.linspace(prop_vals.min().item(), prop_vals.max().item() + 1e-6, n_bins + 1).tolist()

        bin_indices = []
        centers = []
        for b in range(n_bins):
            mask = (prop_vals >= edges[b]) & (prop_vals < edges[b + 1])
            idx = mask.nonzero(as_tuple=True)[0]
            bin_indices.append(idx)
            # Bin center = mean of actual props in bin (for all cond dims)
            centers.append(props_train[idx].mean(dim=0))
        bin_centers_t = torch.stack(centers)  # [n_bins, cond_dim] (normalized)
        # Raw (un-normalized) edges and centers for eval mapping
        p_mean_0 = prop_mean.view(-1)[0].item()
        p_std_0 = prop_std.view(-1)[0].item()
        bin_edges_raw = [e * p_std_0 + p_mean_0 for e in edges]
        bin_centers_raw = [(bin_centers_t[b, 0].item() * p_std_0 + p_mean_0) for b in range(n_bins)]
        if is_main:
            sizes = [len(b) for b in bin_indices]
            print(f"[bins] {n_bins} {bin_method} bins, sizes: min={min(sizes)}, max={max(sizes)}, "
                  f"mean={sum(sizes)/len(sizes):.0f}")
            for b in range(min(n_bins, 5)):
                print(f"  bin {b}: n={sizes[b]:5d}, center={bin_centers_t[b].cpu().numpy()}")
            if n_bins > 5:
                print(f"  ... ({n_bins - 5} more bins)")

    # ── Pre-compute φ(y+) ──
    z_data_train_n = (z_data_train - z_mean) / z_std
    phi_data_all = None       # [N, D] single-layer (backward compat)
    phi_data_layers = None    # list of [N, D] per-layer (multi-layer mode)
    use_multi_layer = bool(cfg.get("feature_space", {}).get("multi_layer", False))
    if phi_model is not None:
        precomp_bs = 512 if feature_mode == "decoder" else 2048
        feat_label = "decoder" if feature_mode == "decoder" else "φ"
        if use_multi_layer and feature_mode == "decoder":
            n_dec_layers = phi_model.num_layers
            if is_main:
                print(f"[{feat_label}] pre-computing MULTI-LAYER embeddings "
                      f"({n_dec_layers} layers, bs={precomp_bs})...")
            phi_layer_lists = [[] for _ in range(n_dec_layers)]  # per-layer accumulators
            with torch.no_grad():
                for i in range(0, z_data_train_n.shape[0], precomp_bs):
                    layer_outs = phi_model.extract_features_multi(z_data_train_n[i:i + precomp_bs])
                    for li, feat in enumerate(layer_outs):
                        phi_layer_lists[li].append(feat)
            phi_data_layers = [torch.cat(ll, dim=0) for ll in phi_layer_lists]
            phi_data_all = phi_data_layers[-1]  # final layer = backward compat
            if is_main:
                for li, phl in enumerate(phi_data_layers):
                    print(f"  layer {li}: {phl.shape}, norm={phl.norm(dim=-1).mean():.3f}")
        else:
            if is_main:
                print(f"[{feat_label}] pre-computing embeddings (bs={precomp_bs})...")
            phi_data_list = []
            with torch.no_grad():
                for i in range(0, z_data_train_n.shape[0], precomp_bs):
                    phi_data_list.append(phi_model.extract_features(z_data_train_n[i:i + precomp_bs]))
            phi_data_all = torch.cat(phi_data_list, dim=0)
            if is_main:
                print(f"[{feat_label}] features: {phi_data_all.shape}, norm={phi_data_all.norm(dim=-1).mean():.3f}")
    else:
        if is_main:
            print("[φ] skipped φ pre-computation (pure z-drift mode)")

    # ── Multi-scale φ pre-computation ──
    phi_data_scales = None  # list of [N, d_s] per scale, or None
    use_multiscale_drift = False
    multiscale_temps = None  # list of τ matched to each scale
    if phi_model is not None and hasattr(phi_model, 'multi_scale_dims') and phi_model.multi_scale_dims is not None:
        ms_dims = phi_model.multi_scale_dims
        # Read per-scale temperatures from config, default: coarse→fine
        multiscale_temps = loss_cfg.get("multiscale_temperatures", None)
        if multiscale_temps is not None and len(multiscale_temps) == len(ms_dims):
            use_multiscale_drift = True
            if is_main:
                print(f"[φ multi-scale] dims={ms_dims}, temps={multiscale_temps}")
                print(f"[φ multi-scale] pre-computing per-scale embeddings...")
            phi_scale_lists = [[] for _ in range(len(ms_dims))]
            precomp_bs_ms = 2048
            with torch.no_grad():
                for i in range(0, z_data_train_n.shape[0], precomp_bs_ms):
                    phis_batch = phi_model.extract_features_multiscale(z_data_train_n[i:i + precomp_bs_ms])
                    for si, f in enumerate(phis_batch):
                        phi_scale_lists[si].append(f)
            phi_data_scales = [torch.cat(sl, dim=0) for sl in phi_scale_lists]
            if is_main:
                for si, (pds, d_s) in enumerate(zip(phi_data_scales, ms_dims)):
                    print(f"  scale {d_s}D: {pds.shape}, norm={pds.norm(dim=-1).mean():.3f}")
        else:
            if is_main and multiscale_temps is not None:
                print(f"[φ multi-scale] WARN: multiscale_temperatures length {len(multiscale_temps)} "
                      f"!= multi_scale_dims length {len(ms_dims)}, skipping multi-scale")

    # ── Fixed λ_τ mode config (actual computation deferred to after projector setup) ──
    drift_normalize_mode = str(loss_cfg.get("drift_normalize_mode", "batch"))  # "batch" (old) or "fixed"
    drift_norm_mode = str(loss_cfg.get("drift_norm_mode", "xy"))  # "xy","y","none","xy_nocross","y_nocross"
    lambda_tau_scale = float(loss_cfg.get("lambda_tau_scale", 1.0))  # scale factor for fixed λ_τ
    fixed_drift_lambdas = None  # dict[float, float] or None — computed below

    # ── Pre-compute φ_struct embeddings + kNN index (if multi-φ) ──
    phi_struct_data_all = None
    phi_struct_knn_idx = None  # [N_train, drift_struct_n_pos] kNN indices in φ_struct space
    if phi_struct_model is not None and lambda_drift_struct > 0:
        if is_main:
            print("[φ_struct] pre-computing φ_struct embeddings...")
        phi_struct_list = []
        with torch.no_grad():
            for i in range(0, z_data_train_n.shape[0], 2048):
                phi_struct_list.append(phi_struct_model.extract_features(z_data_train_n[i:i + 2048]))
        phi_struct_data_all = torch.cat(phi_struct_list, dim=0)
        if is_main:
            print(f"[φ_struct] φ_struct_data: {phi_struct_data_all.shape}, norm={phi_struct_data_all.norm(dim=-1).mean():.3f}")

        # Build kNN index: for each data point, find top-K nearest in φ_struct space
        # This allows struct-drift positives = structurally similar real molecules
        if is_main:
            print(f"[φ_struct] building kNN index (k={drift_struct_n_pos})...")
        knn_k = drift_struct_n_pos
        phi_struct_knn_idx = torch.empty(phi_struct_data_all.shape[0], knn_k, dtype=torch.long, device=device)
        batch_knn = 512
        for i in range(0, phi_struct_data_all.shape[0], batch_knn):
            chunk = phi_struct_data_all[i:i + batch_knn]  # [B, D]
            dists = torch.cdist(chunk, phi_struct_data_all, p=2)  # [B, N_train]
            # Exclude self by setting self-dist to inf
            for j in range(chunk.shape[0]):
                dists[j, i + j] = float('inf')
            _, topk_idx = dists.topk(knn_k, dim=1, largest=False)  # [B, k]
            phi_struct_knn_idx[i:i + chunk.shape[0]] = topk_idx
        if is_main:
            print(f"[φ_struct] kNN index built: {phi_struct_knn_idx.shape}")

    # ── Build generator ──
    generator = build_latent_generator(cfg, latent_dim).to(device)
    assert isinstance(generator, LatentDiTGeneratorCFG), \
        f"Expected LatentDiTGeneratorCFG, got {type(generator)}"
    gc = cfg.get("generator", {})
    noise_dim = int(gc.get("noise_dim", latent_dim))
    param_count = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    if is_main:
        print(f"[gen] {param_count:,} params, noise_dim={noise_dim}")

    # ── Wrap generator in DDP ──
    generator_raw = generator  # keep reference for state_dict
    if ddp:
        generator = DDP(generator, device_ids=[local_rank])
        generator_raw = generator.module

    # ── Optional Property prediction head ──
    prop_head = None
    lambda_zmatch = float(loss_cfg.get("lambda_zmatch", 0.0))
    zmatch_k = int(loss_cfg.get("zmatch_k", 256))
    if lambda_zmatch > 0:
        if is_main:
            print(f"[loss] z-space conditional matching: λ_zmatch={lambda_zmatch}, k={zmatch_k}")

    # ── Set-level alignment (replaces zmatch) ──
    lambda_sinkhorn = float(loss_cfg.get("lambda_sinkhorn", 0.0))
    sinkhorn_epsilon = float(loss_cfg.get("sinkhorn_epsilon", 1.0))
    sinkhorn_n_iter = int(loss_cfg.get("sinkhorn_n_iter", 20))
    lambda_knn_bary = float(loss_cfg.get("lambda_knn_bary", 0.0))
    knn_bary_k = int(loss_cfg.get("knn_bary_k", 8))
    knn_bary_temp = float(loss_cfg.get("knn_bary_temp", 1.0))
    if lambda_sinkhorn > 0 and is_main:
        print(f"[loss] Sinkhorn OT alignment: λ={lambda_sinkhorn}, ε={sinkhorn_epsilon}, n_iter={sinkhorn_n_iter}")
    if lambda_knn_bary > 0 and is_main:
        print(f"[loss] kNN barycentric alignment: λ={lambda_knn_bary}, k={knn_bary_k}, τ={knn_bary_temp}")

    # ── Correlation structure loss (MW recovery) ──
    lambda_corr = float(loss_cfg.get("lambda_corr", 0.0))
    corr_subspace_dims = loss_cfg.get("corr_subspace_dims", None)
    if corr_subspace_dims is not None:
        corr_subspace_dims = int(corr_subspace_dims)
    lambda_cov = float(loss_cfg.get("lambda_cov", 0.0))
    if lambda_corr > 0 and is_main:
        print(f"[loss] Correlation structure loss: λ={lambda_corr}, subspace_dims={corr_subspace_dims}")
    if lambda_cov > 0 and is_main:
        print(f"[loss] Covariance matching loss: λ={lambda_cov}")

    # ── λ ramp-in schedule (staged activation) ──
    # For each loss, optional ramp_start / ramp_end epoch.
    # Before ramp_start: λ_eff=0; linear ramp to full λ by ramp_end.
    _ramp_keys = ["knn_bary", "cov", "corr", "sinkhorn"]
    _lambda_ramps = {}  # key -> (ramp_start, ramp_end)
    for _rk in _ramp_keys:
        rs = loss_cfg.get(f"lambda_{_rk}_ramp_start", None)
        re = loss_cfg.get(f"lambda_{_rk}_ramp_end", None)
        if rs is not None and re is not None:
            rs, re = int(rs), int(re)
            _lambda_ramps[_rk] = (rs, re)
            if is_main:
                print(f"[loss] λ_{_rk} ramp: dormant until epoch {rs}, "
                      f"linear ramp {rs}→{re}")

    def _ramp_scale(key: str, epoch: int) -> float:
        """Return 0→1 multiplier for a ramped λ."""
        if key not in _lambda_ramps:
            return 1.0
        rs, re = _lambda_ramps[key]
        if epoch < rs:
            return 0.0
        if epoch >= re:
            return 1.0
        return (epoch - rs) / (re - rs)

    # ── Decoder-space auxiliary drift (Step 5: decoder hidden state control) ──
    lambda_dec_drift = float(loss_cfg.get("lambda_dec_drift", 0.0))
    dec_drift_temps = loss_cfg.get("dec_drift_temps", [0.05, 0.2])
    dec_drift_multiscale = bool(loss_cfg.get("dec_drift_multiscale", False))
    dec_drift_layer_temps = loss_cfg.get("dec_drift_layer_temps", None)  # per-layer temp, e.g. [0.2, 0.1, 0.05, 0.02]
    if lambda_dec_drift > 0 and is_main:
        print(f"[loss] Decoder drift: λ={lambda_dec_drift}, temps={dec_drift_temps}")
        if dec_drift_multiscale:
            print(f"[loss] Decoder multi-scale drift: layer_temps={dec_drift_layer_temps}")
    prop_head_cfg = cfg.get("prop_head", {})
    ph_hidden = int(prop_head_cfg.get("hidden_dim", 256))
    ph_layers = int(prop_head_cfg.get("num_layers", 3))
    ph_pretrain_steps = int(prop_head_cfg.get("pretrain_steps", 200))
    ph_freeze = bool(prop_head_cfg.get("freeze", False))
    ph_calib_every = int(prop_head_cfg.get("calib_every", 0))  # 0=every step, >0=every N steps
    ph_calib_weight = float(prop_head_cfg.get("calib_weight", 0.5))
    ph_type = str(prop_head_cfg.get("type", "mlp"))  # "mlp", "linear", or "phi_prop"
    phi_prop_col = int(prop_head_cfg.get("phi_prop_col", 0))  # which φ property column to use
    use_phi_prop = (ph_type == "phi_prop")
    if lambda_prop > 0:
        if ph_type == "phi_prop":
            # Use φ model's built-in property head — much stronger than z-space Ridge.
            # For decoder mode: auto-fit a Ridge head on decoder features.
            # Gradient flows: loss→property_head→φ→decoder→z_gen→generator
            # (property_head & decoder weights frozen — act as fixed differentiable function)
            if not hasattr(phi_model, 'property_head') or phi_model.property_head is None:
                # Decoder mode: fit Ridge on decoder features
                ph_ridge_alpha = float(prop_head_cfg.get("ridge_alpha", 1.0))
                if is_main:
                    print(f"[prop_head] PHI_PROP: fitting Ridge head on decoder features (α={ph_ridge_alpha})...")
                phi_model.fit_property_head(z_data_train_n, props_train, alpha=ph_ridge_alpha)
            # Validate on real data
            with torch.no_grad():
                z_val_n = (z_data_val - z_mean) / z_std
                phi_val_parts = []
                for _i in range(0, z_val_n.shape[0], 512):
                    phi_val_parts.append(phi_model.extract_features(z_val_n[_i:_i + 512]))
                phi_val = torch.cat(phi_val_parts, dim=0)
                phi_pred_val = phi_model.property_head(phi_val)[:, phi_prop_col:phi_prop_col+1]
                val_rho = float(torch.corrcoef(torch.stack([
                    phi_pred_val.squeeze(), props_val[:, 0]
                ]))[0, 1].item())
            if is_main:
                print(f"[prop_head] PHI_PROP (col={phi_prop_col}), val Pearson r={val_rho:.3f}")
                print(f"[prop_head] property_head is frozen, gradient flows through decoder→z_gen→generator")
            prop_head = None  # not used — we use phi_model.property_head directly
            ph_freeze = True
        elif ph_type == "linear":
            # Linear prop_head via Ridge regression — NO OOD exploitation possible
            ridge_alpha = float(prop_head_cfg.get("ridge_alpha", 10.0))
            prop_head = LinearPropertyHead.from_ridge_regression(
                z_data_train, props_train, alpha=ridge_alpha
            ).to(device)
            with torch.no_grad():
                pred_test = prop_head(z_data_val)
                val_mse = F.mse_loss(pred_test, props_val).item()
                val_r2 = 1.0 - val_mse  # props_val is normalized (mean=0, std=1)
            if is_main:
                print(f"[prop_head] LINEAR (Ridge α={ridge_alpha}), val MSE={val_mse:.4f}, R²={val_r2:.4f}")
            # Always freeze linear head
            for p in prop_head.parameters():
                p.requires_grad_(False)
            prop_head.eval()
            ph_freeze = True
        else:
            prop_head = PropertyHead(latent_dim, cond_dim, hidden_dim=ph_hidden,
                                     num_layers=ph_layers).to(device)
            prop_head_params = sum(p.numel() for p in prop_head.parameters())
            if is_main:
                print(f"[prop_head] MLP {prop_head_params:,} params (hidden={ph_hidden}, layers={ph_layers})")
                print(f"[prop_head] freeze={ph_freeze}, calib_every={ph_calib_every}, calib_weight={ph_calib_weight}")

            # Pre-train prop_head on real data
            if is_main:
                print(f"[prop_head] pre-training on real data ({ph_pretrain_steps} steps)...")
            ph_opt = torch.optim.Adam(prop_head.parameters(), lr=1e-3)
            ph_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(ph_opt, T_max=ph_pretrain_steps)
            for _pt in range(ph_pretrain_steps):
                idx = torch.randint(0, z_data_train.shape[0], (1024,), device=device)
                pred = prop_head(z_data_train[idx])
                loss_ph = F.mse_loss(pred, props_train[idx])
                ph_opt.zero_grad()
                loss_ph.backward()
                ph_opt.step()
                ph_scheduler.step()
            with torch.no_grad():
                idx_test = torch.randint(0, z_data_val.shape[0], (2000,), device=device)
                pred_test = prop_head(z_data_val[idx_test])
                val_mse = F.mse_loss(pred_test, props_val[idx_test]).item()
            if is_main:
                print(f"[prop_head] pre-train done, val MSE={val_mse:.4f}")

            if ph_freeze:
                for p in prop_head.parameters():
                    p.requires_grad_(False)
                prop_head.eval()
                if is_main:
                    print("[prop_head] FROZEN — gradients flow through but params don't update")
            else:
                prop_head_optimizer = torch.optim.Adam(prop_head.parameters(), lr=3e-4)
                if is_main:
                    print("[prop_head] separate optimizer for online calibration")

    # ── Optional drift projector (QED-supervised low-dim projection) ──
    drift_proj_cfg = cfg.get("drift_projector", {})
    use_drift_proj = bool(drift_proj_cfg.get("enabled", False))
    if use_drift_proj and isinstance(phi_model, DecoderFeatureExtractor):
        dp_dim = int(drift_proj_cfg.get("dim", 16))
        dp_alpha = float(drift_proj_cfg.get("ridge_alpha", 1.0))
        dp_prop_col = int(drift_proj_cfg.get("prop_col", 0))
        if not hasattr(phi_model, "drift_proj"):
            if is_main:
                print(f"[drift_projector] Fitting QED-supervised projection φ(512d)→{dp_dim}d...")
            phi_model.fit_drift_projector(
                z_data_train_n, props_train, proj_dim=dp_dim,
                alpha=dp_alpha, prop_col=dp_prop_col,
            )
        # Pre-project ALL data features → [N_train, proj_dim] (saves repeated projections)
        if phi_data_all is not None:
            with torch.no_grad():
                phi_data_proj = phi_model.project_for_drift(phi_data_all)  # [N_train, proj_dim]
            if is_main:
                print(f"[drift_projector] Pre-projected data features: {phi_data_proj.shape}")
        else:
            phi_data_proj = None
        if is_main:
            print(f"[drift_projector] Drift will be computed in {dp_dim}d projected space")
    elif use_drift_proj:
        if is_main:
            print("[drift_projector] WARNING: drift_projector only works with decoder feature mode, ignoring")
        use_drift_proj = False
    # Placeholder when projector not used
    if not use_drift_proj:
        phi_data_proj = None

    # ── Pre-compute fixed λ_τ from data (Paper Eq. 23-25, but FIXED) ──
    # Must happen AFTER projector setup so we use the right feature space.
    # When drift_normalize=True, per-batch λ_τ makes loss constant (~D per τ).
    # Fix: compute λ_τ once from data (where p≈q gives the baseline scale).
    if drift_normalize and drift_normalize_mode == "fixed":
        # Use PROJECTED features if projector is active, else raw φ features
        _phi_for_lambda = phi_data_proj if (use_drift_proj and phi_data_proj is not None) else phi_data_all
        if _phi_for_lambda is not None:
            from src.drifting.drift_latent_phi import compute_drift_field_paper
            if is_main:
                print(f"[drift] computing fixed λ_τ from data (p≈q baseline, dim={_phi_for_lambda.shape[-1]})...")
            _n_cal = min(n_gen, _phi_for_lambda.shape[0] // 2)  # match training N_gen
            _n_cal_pos = min(n_pos, _phi_for_lambda.shape[0] - _n_cal)  # match training N_pos (balanced)
            _phi_cal_gen = _phi_for_lambda[:_n_cal]
            _phi_cal_pos = _phi_for_lambda[_n_cal:_n_cal + _n_cal_pos]
            D_drift = _phi_for_lambda.shape[-1]
            fixed_drift_lambdas = {}
            with torch.no_grad():
                for tau in drift_temperatures:
                    V_tau_cal = compute_drift_field_paper(
                        _phi_cal_gen, _phi_cal_pos, temperature=tau,
                        normalize_distances=drift_normalize_dist,
                        norm_mode=drift_norm_mode,
                        attraction_scale=drift_attraction_scale,
                        repulsion_scale=drift_repulsion_scale,
                    )
                    lam = (V_tau_cal.pow(2).sum(dim=-1).mean() / D_drift).sqrt().clamp(min=1e-8).item()
                    fixed_drift_lambdas[tau] = lam
                    if is_main:
                        print(f"  τ={tau}: λ_τ = {lam:.6f} (||V_raw||² = {V_tau_cal.pow(2).sum(dim=-1).mean():.4f})")
            # Apply manual scale factor (useful to match different calibration conditions)
            if lambda_tau_scale != 1.0:
                for tau in fixed_drift_lambdas:
                    fixed_drift_lambdas[tau] *= lambda_tau_scale
                if is_main:
                    print(f"[drift] lambda_tau_scale={lambda_tau_scale} → scaled_lambdas = {fixed_drift_lambdas}")
            if is_main:
                print(f"[drift] fixed_lambdas = {fixed_drift_lambdas}")
    elif drift_normalize and drift_normalize_mode == "batch":
        if is_main:
            print("[drift] using per-batch λ_τ normalization (original mode)")
    if is_main:
        print(f"[drift] norm_mode = {drift_norm_mode}")

    # ── Pre-compute fixed lambda_tau for z-space drift ──
    fixed_zdrift_lambdas = None
    if lambda_zdrift > 0 and drift_normalize and drift_normalize_mode == "fixed":
        from src.drifting.drift_latent_phi import compute_drift_field_paper
        z_drift_temps_for_cal = zdrift_temperatures if zdrift_temperatures is not None else drift_temperatures
        _n_cal_z = min(n_gen, z_data_train.shape[0] // 2)  # match training N_gen
        _n_cal_pos_z = min(n_pos, z_data_train.shape[0] - _n_cal_z)  # match training N_pos (balanced)
        _z_cal_gen = z_data_train[:_n_cal_z]
        _z_cal_pos = z_data_train[_n_cal_z:_n_cal_z + _n_cal_pos_z]
        D_z = z_data_train.shape[-1]
        fixed_zdrift_lambdas = {}
        if is_main:
            print(f"[zdrift] computing fixed lambda_tau from z-space data (dim={D_z}, normalize_dist={drift_normalize_dist})...")
        with torch.no_grad():
            for tau in z_drift_temps_for_cal:
                V_z_cal = compute_drift_field_paper(
                    _z_cal_gen, _z_cal_pos, temperature=tau,
                    normalize_distances=drift_normalize_dist,
                    norm_mode=drift_norm_mode,
                    attraction_scale=drift_attraction_scale,
                    repulsion_scale=drift_repulsion_scale,
                )
                lam_z = (V_z_cal.pow(2).sum(dim=-1).mean() / D_z).sqrt().clamp(min=1e-8).item()
                fixed_zdrift_lambdas[tau] = lam_z
                if is_main:
                    print(f"  tau={tau}: lambda_tau = {lam_z:.6f} (||V_raw||^2 = {V_z_cal.pow(2).sum(dim=-1).mean():.4f})")
        if is_main:
            print(f"[zdrift] fixed_zdrift_lambdas = {fixed_zdrift_lambdas}")

    # ── Pre-compute single-scale decoder data features (for dec_drift positive lookup) ──
    dec_data_single = None  # [N, 512] pre-computed decoder features for single-scale dec drift
    fixed_dec_drift_single_lambdas = None  # dict[tau -> lambda]
    if lambda_dec_drift > 0 and not dec_drift_multiscale and decoder_phi_model is not None:
        if is_main:
            print(f"[dec-drift] pre-computing single-scale decoder features for positive lookup...")
        _dec_parts = []
        with torch.no_grad():
            for i in range(0, z_data_train.shape[0], 512):
                _dec_parts.append(decoder_phi_model.extract_features(z_data_train[i:i + 512]))
        dec_data_single = torch.cat(_dec_parts, dim=0)
        if is_main:
            print(f"[dec-drift] decoder features: {dec_data_single.shape}, norm={dec_data_single.norm(dim=-1).mean():.3f}")
        # Pre-compute fixed λ_τ for single-scale dec drift normalization
        if drift_normalize and drift_normalize_mode == "fixed":
            from src.drifting.drift_latent_phi import compute_drift_field_paper
            fixed_dec_drift_single_lambdas = {}
            _n_cal = min(n_gen, dec_data_single.shape[0] // 2)
            _n_cal_pos = min(n_pos, dec_data_single.shape[0] - _n_cal)
            D_dec = dec_data_single.shape[-1]
            for tau in dec_drift_temps:
                _dec_cal_gen = dec_data_single[:_n_cal]
                _dec_cal_pos = dec_data_single[_n_cal:_n_cal + _n_cal_pos]
                with torch.no_grad():
                    V_cal = compute_drift_field_paper(
                        _dec_cal_gen, _dec_cal_pos, temperature=tau,
                        normalize_distances=drift_normalize_dist,
                        norm_mode=drift_norm_mode,
                        attraction_scale=drift_attraction_scale,
                        repulsion_scale=drift_repulsion_scale,
                    )
                    lam = (V_cal.pow(2).sum(dim=-1).mean() / D_dec).sqrt().clamp(min=1e-8).item()
                fixed_dec_drift_single_lambdas[tau] = lam
                if is_main:
                    print(f"  τ={tau}: λ_τ={lam:.6f} (||V_raw||²={V_cal.pow(2).sum(dim=-1).mean():.4f})")

    # ── Pre-compute per-layer decoder data features (multi-scale dec drift) ──
    dec_data_layers = None   # list of [N, 512] per decoder layer, or None
    fixed_dec_drift_lambdas = None  # dict[layer_idx -> dict[tau -> lambda]]
    if lambda_dec_drift > 0 and dec_drift_multiscale and decoder_phi_model is not None:
        n_dec_layers = decoder_phi_model.num_layers
        if dec_drift_layer_temps is not None and len(dec_drift_layer_temps) == n_dec_layers:
            if is_main:
                print(f"[dec-drift] pre-computing {n_dec_layers}-layer decoder features...")
            dec_layer_lists = [[] for _ in range(n_dec_layers)]
            precomp_bs_dec = 512
            with torch.no_grad():
                for i in range(0, z_data_train.shape[0], precomp_bs_dec):
                    layer_outs = decoder_phi_model.extract_features_multi(z_data_train[i:i + precomp_bs_dec])
                    for li, feat in enumerate(layer_outs):
                        dec_layer_lists[li].append(feat)
            dec_data_layers = [torch.cat(ll, dim=0) for ll in dec_layer_lists]
            if is_main:
                for li, ddl in enumerate(dec_data_layers):
                    print(f"  dec layer {li}: {ddl.shape}, norm={ddl.norm(dim=-1).mean():.3f}")
            # Pre-compute fixed λ_τ per layer (like the main drift)
            if drift_normalize and drift_normalize_mode == "fixed":
                from src.drifting.drift_latent_phi import compute_drift_field_paper
                fixed_dec_drift_lambdas = {}
                _n_cal = min(n_gen, dec_data_layers[0].shape[0] // 2)
                _n_cal_pos = min(n_pos, dec_data_layers[0].shape[0] - _n_cal)
                D_dec = dec_data_layers[0].shape[-1]
                for li in range(n_dec_layers):
                    tau_li = dec_drift_layer_temps[li]
                    _dec_cal_gen = dec_data_layers[li][:_n_cal]
                    _dec_cal_pos = dec_data_layers[li][_n_cal:_n_cal + _n_cal_pos]
                    with torch.no_grad():
                        V_cal = compute_drift_field_paper(
                            _dec_cal_gen, _dec_cal_pos, temperature=tau_li,
                            normalize_distances=drift_normalize_dist,
                            norm_mode=drift_norm_mode,
                            attraction_scale=drift_attraction_scale,
                            repulsion_scale=drift_repulsion_scale,
                        )
                        lam = (V_cal.pow(2).sum(dim=-1).mean() / D_dec).sqrt().clamp(min=1e-8).item()
                    fixed_dec_drift_lambdas[li] = {tau_li: lam}
                    if is_main:
                        print(f"  dec layer {li}: τ={tau_li}, λ_τ={lam:.6f}")
        else:
            if is_main and dec_drift_layer_temps is not None:
                print(f"[dec-drift] WARN: dec_drift_layer_temps length {len(dec_drift_layer_temps)} "
                      f"!= decoder layers {n_dec_layers}, falling back to single-scale")
            dec_drift_multiscale = False

    # ── Pre-compute decoder last-layer data features for hybrid positive mode ──
    dec_data_last_hybrid = None  # [N, 512] or None
    _hybrid_use_phi_model = False  # True when decoder mode uses phi_model for hybrid kNN
    if positive_mode == "hybrid" and decoder_phi_model is not None:
        if dec_data_layers is not None:
            dec_data_last_hybrid = dec_data_layers[-1]  # reuse multi-scale pre-computation
            if is_main:
                print(f"[hybrid-pos] reusing dec_data_layers[-1]: {dec_data_last_hybrid.shape}")
        elif dec_data_single is not None:
            dec_data_last_hybrid = dec_data_single  # reuse single-scale pre-computation
            if is_main:
                print(f"[hybrid-pos] reusing dec_data_single: {dec_data_last_hybrid.shape}")
        else:
            if is_main:
                print(f"[hybrid-pos] pre-computing decoder last-layer features for kNN...")
            _parts = []
            with torch.no_grad():
                for i in range(0, z_data_train.shape[0], 512):
                    _parts.append(decoder_phi_model.extract_features(z_data_train[i:i + 512]))
            dec_data_last_hybrid = torch.cat(_parts, dim=0)
            if is_main:
                print(f"[hybrid-pos] decoder features: {dec_data_last_hybrid.shape}")
    elif positive_mode == "hybrid" and feature_mode == "decoder" and phi_data_all is not None:
        # In decoder mode, phi_model IS the decoder — reuse phi_data_all for kNN
        dec_data_last_hybrid = phi_data_all
        _hybrid_use_phi_model = True
        if is_main:
            print(f"[hybrid-pos] decoder mode: reusing phi_data_all: {dec_data_last_hybrid.shape}")
    elif positive_mode == "hybrid" and feature_mode == "phi" and phi_data_all is not None:
        # In phi mode, use φ features for hybrid kNN
        dec_data_last_hybrid = phi_data_all
        _hybrid_use_phi_model = True
        if is_main:
            print(f"[hybrid-pos] phi mode: reusing phi_data_all for kNN: {dec_data_last_hybrid.shape}")
    elif positive_mode == "hybrid" and is_main:
        print(f"[hybrid-pos] WARN: hybrid mode requires decoder features, falling back to bin mode")

    # EMA (on raw model, not DDP wrapper)
    ema_decay = float(cfg.get("training", {}).get("ema_decay", 0.999))
    ema_generator = copy.deepcopy(generator_raw).eval()
    for p in ema_generator.parameters():
        p.requires_grad_(False)

    # Optimizer — ONLY generator params (prop_head excluded to prevent adversarial)
    base_lr = float(cfg["training"].get("lr", 2e-4))
    opt_params = list(generator.parameters())
    optimizer = torch.optim.AdamW(
        opt_params,
        lr=base_lr,
        weight_decay=float(cfg["training"].get("weight_decay", 0.01)),
    )

    epochs = int(cfg["training"].get("epochs", 120))
    grad_clip = float(cfg["training"].get("grad_clip_norm", 1.0))
    warmup_epochs = int(cfg["training"].get("warmup_epochs", 5))

    scheduler = build_lr_scheduler(
        optimizer, total_epochs=epochs, warmup_epochs=warmup_epochs,
        schedule=str(cfg["training"].get("lr_schedule", "cosine")),
        min_lr_ratio=float(cfg["training"].get("min_lr_ratio", 0.01)),
    )

    sel_cfg = cfg.get("selection", {})
    eval_every = int(sel_cfg.get("eval_every_epochs", 5))
    eval_samples = int(sel_cfg.get("num_generated_samples", 2000))
    eval_batch = int(sel_cfg.get("sample_batch_size", 512))

    # ── Quality gate ──
    gate_cfg = cfg.get("quality_gate", {})
    is_unconditional_mode = (float(cfg.get("cfg", {}).get("alpha_min", 1.0))
                             == float(cfg.get("cfg", {}).get("alpha_max", 1.0)))
    gate_cfg.setdefault("is_conditional", not is_unconditional_mode)
    quality_gate = QualityGate.from_config(gate_cfg)
    if is_main:
        print(f"[gate] QualityGate active: conditional={quality_gate.is_conditional}"
              f" min_U={quality_gate.min_uniqueness}"
              f" min_per_bin_U={quality_gate.min_per_bin_uniqueness}"
              f" min_ρ={quality_gate.min_spearman_rho}")

    z_norms = z_data_train.norm(dim=-1)
    target_z_norm_mean = z_norms.mean().item()

    N_train = z_data_train.shape[0]
    steps_per_epoch = max(1, N_train // effective_batch)  # same as single-GPU
    best_score = -1.0
    best_epoch = 0
    start_epoch = 1
    history = []

    # ── Resume from last.pt if it exists ─────────────────────────────
    resume_path = out_dir / "last.pt"
    if resume_path.exists():
        if is_main:
            print(f"[resume] Loading checkpoint from {resume_path} ...")
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        generator_raw.load_state_dict(resume_ckpt["generator_state"])
        ema_generator.load_state_dict(resume_ckpt["ema_state"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        for _ in range(resume_ckpt["epoch"]):
            scheduler.step()
        best_score = resume_ckpt.get("best_score", -1.0)
        best_epoch = resume_ckpt.get("best_epoch", 0)
        start_epoch = resume_ckpt["epoch"] + 1
        if prop_head is not None and resume_ckpt.get("prop_head_state") is not None:
            prop_head.load_state_dict(resume_ckpt["prop_head_state"])
        if is_main:
            print(f"[resume] Resuming from epoch {start_epoch}, best_score={best_score:.4f}")
        del resume_ckpt

    if is_main:
        print(f"\n[train] {epochs} epochs (start={start_epoch}), steps/ep={steps_per_epoch}, "
              f"effective_batch={effective_batch}, eval_every={eval_every}\n")

    # ── Training loop (Paper-faithful: per-group drift + CFG) ────────
    for epoch in range(start_epoch, epochs + 1):
        generator.train()
        if prop_head is not None:
            prop_head.train()
        current_lr = optimizer.param_groups[0]["lr"]
        running = {"total": 0., "drift": 0., "drift_struct": 0., "moment": 0., "zdiv": 0.,
                   "znorm": 0., "prop": 0., "zmatch": 0., "zcontrast": 0.,
                   "zdrift": 0., "phidiv": 0., "dcdrift": 0.,
                   "sinkhorn": 0., "knn_bary": 0.,
                   "corr": 0., "cov": 0., "dec_drift": 0.}
        n_steps = 0
        t0 = time.time()

        for _step in range(steps_per_epoch):
            # ── 1. Sample α from paper distribution: p(α) ∝ α^{-power} ──
            alpha_val = sample_cfg_alpha(alpha_power, alpha_min, alpha_max)

            # ── 2. Sample N_c condition centers ──
            use_class_cond = getattr(generator_raw, 'num_classes', 0) > 0
            if use_bins:
                # Bin mode: sample bins
                sampled_bins = torch.randint(0, n_bins, (local_n_groups,))
                cond_centers = bin_centers_t[sampled_bins].to(device)
            else:
                center_idx = torch.randint(0, N_train, (local_n_groups,), device=device)
                cond_centers = props_train[center_idx]

            # ── 3. Generate: local_N_c groups × N_gen per group ──
            noise = torch.randn(local_n_groups * n_gen, noise_dim, device=device)
            # prop_targets: always normalized property centers (for prop_loss)
            prop_targets = cond_centers.repeat_interleave(n_gen, dim=0)  # [N_c*N_gen, cond_dim]
            if use_class_cond and use_bins:
                # Class embedding: pass integer bin IDs to generator
                cond_expanded = sampled_bins.repeat_interleave(n_gen).to(device)  # [N_c*N_gen] LongTensor
            else:
                cond_expanded = prop_targets  # [N_c*N_gen, cond_dim]
            z_gen = generator(noise, cond=cond_expanded, alpha=alpha_val)

            # ── 4. Compute φ(x_gen) for all ──
            z_gen_n = (z_gen - z_mean) / z_std
            # B3 ablation: stop-grad drift — detach z so drift loss can't
            # backprop to generator (tests if gradient signal is essential)
            z_gen_n_phi = z_gen_n.detach() if stop_grad_drift else z_gen_n
            phi_gen_all = None     # [B, D] single-layer (backward compat)
            phi_gen_layers = None  # list of [B, D] multi-layer
            _dec_mbs = int(cfg.get("feature_space", {}).get("micro_batch", 256))
            if phi_model is None:
                pass  # no φ model loaded — skip φ computation
            elif use_multi_layer and feature_mode == "decoder":
                # Multi-layer: extract per-layer features with micro-batching
                n_dec_layers = phi_model.num_layers
                phi_gen_layer_parts = [[] for _ in range(n_dec_layers)]
                for _i in range(0, z_gen_n_phi.shape[0], _dec_mbs):
                    layer_outs = phi_model.extract_features_multi(z_gen_n_phi[_i:_i + _dec_mbs])
                    for li, feat in enumerate(layer_outs):
                        phi_gen_layer_parts[li].append(feat)
                phi_gen_layers = [torch.cat(ll, dim=0) for ll in phi_gen_layer_parts]
                phi_gen_all = phi_gen_layers[-1]  # final layer for prop_head etc.
            elif feature_mode == "decoder":
                # Single-layer decoder: micro-batch
                phi_gen_parts = []
                for _i in range(0, z_gen_n_phi.shape[0], _dec_mbs):
                    phi_gen_parts.append(phi_model.extract_features(z_gen_n_phi[_i:_i + _dec_mbs]))
                phi_gen_all = torch.cat(phi_gen_parts, dim=0)
            else:
                phi_gen_all = phi_model.extract_features(z_gen_n_phi)  # [N_c*N_gen, phi_dim]

            # ── 4b. Decoder gen features for hybrid positive selection ──
            dec_gen_for_knn = None  # [B, 512] or None
            if positive_mode == "hybrid" and dec_data_last_hybrid is not None:
                if _hybrid_use_phi_model and phi_gen_all is not None:
                    # decoder mode: phi_gen_all already has decoder features
                    dec_gen_for_knn = phi_gen_all.detach()
                elif decoder_phi_model is not None:
                    with torch.no_grad():
                        _dec_mbs_knn = 128
                        _parts = []
                        for _i in range(0, z_gen.shape[0], _dec_mbs_knn):
                            _parts.append(decoder_phi_model.extract_features(z_gen[_i:_i + _dec_mbs_knn]))
                        dec_gen_for_knn = torch.cat(_parts, dim=0)  # [B, 512]

            # ── 5. Pre-compute positive indices per group ──
            _per_sample_pos = None  # [local_n_groups, n_gen, n_pos] or None
            if positive_mode == "hybrid" and use_bins and dec_gen_for_knn is not None:
                # Hybrid: half bin-random + half decoder-space kNN within bin
                n_pos_global = n_pos // 2
                n_pos_local = n_pos - n_pos_global
                all_pos_idx = torch.zeros(local_n_groups, n_pos, dtype=torch.long, device=device)
                for g in range(local_n_groups):
                    bin_id = sampled_bins[g].item()
                    bin_idx = bin_indices[bin_id]
                    n_avail = bin_idx.shape[0]
                    bin_idx_dev = bin_idx.to(device)
                    # Global half: random from same bin
                    sel_global = torch.randint(0, n_avail, (n_pos_global,), device=device)
                    # Local half: decoder-space kNN within the bin
                    dec_gen_g = dec_gen_for_knn[g * n_gen:(g + 1) * n_gen]  # [n_gen, 512]
                    dec_bin = dec_data_last_hybrid[bin_idx_dev]  # [n_avail, 512]
                    # Group-mean query
                    query = dec_gen_g.mean(dim=0, keepdim=True)  # [1, 512]
                    dists = torch.cdist(query, dec_bin, p=2).squeeze(0)  # [n_avail]
                    K_pool = min(n_pos_local * knn_pool_factor, n_avail)
                    _, topK = dists.topk(K_pool, largest=False)  # [K_pool]
                    sel_local = topK[torch.randperm(K_pool, device=device)[:n_pos_local]]
                    all_pos_idx[g] = torch.cat([bin_idx_dev[sel_global], bin_idx_dev[sel_local]])
            elif use_bins:
                # Bin mode: positives = random samples from same bin
                all_pos_idx = torch.zeros(local_n_groups, n_pos, dtype=torch.long, device=device)
                for g in range(local_n_groups):
                    bin_id = sampled_bins[g].item()
                    bin_idx = bin_indices[bin_id]  # indices of data in this bin
                    n_avail = bin_idx.shape[0]
                    # Random sample with replacement if bin smaller than n_pos
                    sel = torch.randint(0, n_avail, (n_pos,), device=device)
                    all_pos_idx[g] = bin_idx[sel].to(device)
            elif positive_mode == "phi" and phi_gen_all is not None and phi_data_all is not None:
                # φ-space stochastic kNN: per-sample kNN with random subsampling
                # Each z_gen sample finds its own top-K neighbors, then randomly picks n_pos.
                # This breaks the positive feedback loop of centroid-based kNN.
                with torch.no_grad():
                    K_pool = min(n_pos * knn_pool_factor, N_train)
                    # Per-sample distances: [B, N_train] where B = local_n_groups * n_gen
                    phi_dists_all = torch.cdist(phi_gen_all, phi_data_all, p=2)
                    _, topK_idx = phi_dists_all.topk(K_pool, dim=1, largest=False)  # [B, K_pool]
                    # Random subsample n_pos from K_pool for each sample
                    B_total = topK_idx.shape[0]
                    rand_sel = torch.argsort(torch.rand(B_total, K_pool, device=device), dim=1)[:, :n_pos]  # [B, n_pos]
                    per_sample_pos_idx = topK_idx.gather(1, rand_sel)  # [B, n_pos]
                    # Reshape to per-group: [local_n_groups, n_gen, n_pos]
                    per_sample_pos_idx = per_sample_pos_idx.view(local_n_groups, n_gen, n_pos)
                    _per_sample_pos = per_sample_pos_idx  # save for per-sample drift
                    # Per-group fallback: unique union of all samples' kNN, subsample n_pos
                    all_pos_idx = torch.zeros(local_n_groups, n_pos, dtype=torch.long, device=device)
                    for g in range(local_n_groups):
                        union = per_sample_pos_idx[g].reshape(-1).unique()
                        if union.shape[0] >= n_pos:
                            sel = torch.randperm(union.shape[0], device=device)[:n_pos]
                            all_pos_idx[g] = union[sel]
                        else:
                            # pad with random if union is small
                            pad = torch.randint(0, N_train, (n_pos - union.shape[0],), device=device)
                            all_pos_idx[g] = torch.cat([union, pad])
            else:
                prop_dists = torch.cdist(cond_centers, props_train, p=2)  # [N_c, N_train]
                _, all_pos_idx = prop_dists.topk(n_pos, dim=1, largest=False)  # [N_c, n_pos]

            # ── 6. Group-wise drift loss (Paper Algorithm 1 per "class") ──
            # Multi-layer mode (Paper A.5): independent drift loss per decoder layer, summed.
            # Single-layer mode: same as before (final layer only).
            loss_drift = torch.zeros((), device=device)
            if lambda_drift > 0 and phi_gen_all is not None:
                _drift_phi_layers = phi_gen_layers if phi_gen_layers is not None else [phi_gen_all]
                _drift_data_layers = phi_data_layers if phi_data_layers is not None else [phi_data_all]
                total_drift_loss = torch.tensor(0.0, device=device)
                for li in range(len(_drift_phi_layers)):
                    phi_gen_li = _drift_phi_layers[li]
                    phi_data_li = _drift_data_layers[li]
                    for g in range(local_n_groups):
                        phi_gen_g = phi_gen_li[g * n_gen:(g + 1) * n_gen]  # [N_gen, phi_dim]

                        # Positives: data matching this condition (like same-class in ImageNet)
                        phi_pos_g = phi_data_li[all_pos_idx[g]]  # [N_pos, phi_dim]

                        # Unconditional negatives: random from ALL data (Paper Eq. 15)
                        unc_idx = torch.randint(0, N_train, (n_unc,), device=device)
                        phi_unc_g = phi_data_li[unc_idx]  # [N_unc, phi_dim]

                        # Optional: project to low-dim QED-supervised drift space
                        if use_drift_proj and phi_data_proj is not None:
                            # phi_gen: project with grad (flows through decoder→z→generator)
                            phi_gen_g = phi_model.project_for_drift(phi_gen_g)
                            # Data: use pre-projected (no grad needed)
                            phi_pos_g = phi_data_proj[all_pos_idx[g]]
                            phi_unc_g = phi_data_proj[unc_idx]

                        # CFG weight. At α=1 the unconditional branch is
                        # omitted downstream, so no log(0) is evaluated.
                        w = max(
                            0.0,
                            (alpha_val - 1.0) * max(n_gen - 1, 1) / max(n_unc, 1),
                        )

                        # Multi-temperature drift loss
                        loss_g = multi_temp_drift_loss(
                            phi_gen_g, phi_pos_g,
                            temperatures=drift_temperatures,
                            phi_unc=phi_unc_g, cfg_w=w,
                            normalize_drift=drift_normalize,
                            normalize_distances=drift_normalize_dist,
                            fixed_lambdas=fixed_drift_lambdas,
                            norm_mode=drift_norm_mode,
                            knn_restrict_k=drift_knn_k,
                            attraction_scale=drift_attraction_scale,
                            repulsion_scale=drift_repulsion_scale,
                        )
                        total_drift_loss = total_drift_loss + loss_g
                loss_drift = total_drift_loss / (local_n_groups * len(_drift_phi_layers))

            # ── 7. Optional property regression loss ──
            loss_prop = torch.zeros((), device=device)
            if lambda_prop > 0 and (prop_head is not None or use_phi_prop):
                if use_phi_prop:
                    # Use φ model's property head — much stronger signal.
                    # phi_gen_all is already computed in step 4 (with grad through z_gen).
                    phi_pred = phi_model.property_head(phi_gen_all)  # [B, num_props]
                    pred_prop = phi_pred[:, phi_prop_col:phi_prop_col+1]  # [B, 1]
                    # prop_targets is in generator's normalized space.
                    # φ prop_head output is in φ's normalized space.
                    # Both use same QED normalization, so directly compare.
                    loss_prop = F.mse_loss(pred_prop, prop_targets)
                elif prop_head is not None:
                    # Online calibration FIRST (before building generator's computation graph)
                    if not ph_freeze and (ph_calib_every == 0 or _step % ph_calib_every == 0):
                        idx_real = torch.randint(0, N_train, (1024,), device=device)
                        pred_real = prop_head(z_data_train[idx_real])
                        calib_loss = F.mse_loss(pred_real, props_train[idx_real])
                        prop_head_optimizer.zero_grad()
                        calib_loss.backward()
                        prop_head_optimizer.step()

                    # Now compute property loss for generator
                    # (prop_head params are detached from generator optimizer)
                    pred_prop = prop_head(z_gen)
                    loss_prop = F.mse_loss(pred_prop, prop_targets)

            # ── 7a-2. Multi-φ structural drift (φ_struct) ──
            # Positives: kNN in φ_struct space (structurally similar real data)
            # Negatives: random from ALL data (same CFG pattern as φ_prop)
            loss_drift_struct = torch.zeros((), device=device)
            if lambda_drift_struct > 0 and phi_struct_data_all is not None:
                phi_struct_gen = phi_struct_model.extract_features(z_gen_n)  # [N_c*N_gen, phi_struct_dim]
                # Find nearest data point for each generated sample → use its kNN as positives
                with torch.no_grad():
                    struct_dists = torch.cdist(phi_struct_gen, phi_struct_data_all, p=2)  # [N_c*N_gen, N_train]
                    nn_data_idx = struct_dists.argmin(dim=1)  # [N_c*N_gen] nearest data index
                for g in range(local_n_groups):
                    phi_s_gen_g = phi_struct_gen[g * n_gen:(g + 1) * n_gen]
                    # Positives: kNN of the nearest data point (structurally matched)
                    nn_idx_g = nn_data_idx[g * n_gen:(g + 1) * n_gen]  # [n_gen]
                    # Use kNN of the first sample as shared positives for this group
                    # (could also merge kNN from all samples, but this is simpler)
                    anchor_idx = nn_idx_g[0].item()
                    struct_pos_idx = phi_struct_knn_idx[anchor_idx]  # [drift_struct_n_pos]
                    phi_s_pos_g = phi_struct_data_all[struct_pos_idx]
                    # Unconditional negatives: random from ALL data (provides contrast)
                    struct_unc_idx = torch.randint(0, N_train, (drift_struct_n_unc,), device=device)
                    phi_s_unc_g = phi_struct_data_all[struct_unc_idx]
                    # CFG weight same formula as φ_prop
                    w_struct = max(
                        0.0,
                        (alpha_val - 1.0) * max(n_gen - 1, 1) / max(drift_struct_n_unc, 1),
                    )
                    loss_drift_struct = loss_drift_struct + multi_temp_drift_loss(
                        phi_s_gen_g, phi_s_pos_g,
                        temperatures=drift_temperatures,
                        phi_unc=phi_s_unc_g, cfg_w=w_struct,
                        normalize_drift=drift_normalize,
                        normalize_distances=drift_normalize_dist,
                        attraction_scale=drift_attraction_scale,
                        repulsion_scale=drift_repulsion_scale,
                    )
                loss_drift_struct = loss_drift_struct / local_n_groups

            # ── 7b. Z-space conditional matching loss ──
            loss_zmatch = torch.zeros((), device=device)
            if lambda_zmatch > 0:
                for g in range(local_n_groups):
                    z_gen_g = z_gen[g * n_gen:(g + 1) * n_gen]   # [N_gen, latent_dim]
                    z_pos_g = z_data_train[all_pos_idx[g]]       # [n_pos, latent_dim]

                    if zmatch_mode == "nn_soft":
                        # Per-sample soft nearest neighbor: each z_gen targets
                        # a weighted combination of condition-matched real z,
                        # weighted by proximity (softmin of distances)
                        with torch.no_grad():
                            dists = torch.cdist(z_gen_g, z_pos_g)  # [n_gen, n_pos]
                            weights = F.softmax(-dists / zmatch_temp, dim=1)  # [n_gen, n_pos]
                        z_target = torch.bmm(
                            weights.unsqueeze(1),  # [n_gen, 1, n_pos]
                            z_pos_g.unsqueeze(0).expand(z_gen_g.size(0), -1, -1)  # [n_gen, n_pos, D]
                        ).squeeze(1)  # [n_gen, D]
                        loss_zmatch = loss_zmatch + F.mse_loss(z_gen_g, z_target)
                    else:
                        # Original centroid mode
                        z_target = z_pos_g.mean(dim=0, keepdim=True)
                        loss_zmatch = loss_zmatch + F.mse_loss(z_gen_g, z_target.expand_as(z_gen_g))
                loss_zmatch = loss_zmatch / local_n_groups

            # ── 7b-1a. Sinkhorn OT alignment (set-level, replaces zmatch) ──
            loss_sinkhorn = torch.zeros((), device=device)
            if lambda_sinkhorn > 0:
                for g in range(local_n_groups):
                    z_gen_g = z_gen[g * n_gen:(g + 1) * n_gen]
                    z_pos_g = z_data_train[all_pos_idx[g]]
                    loss_sinkhorn = loss_sinkhorn + sinkhorn_alignment_loss(
                        z_gen_g, z_pos_g,
                        epsilon=sinkhorn_epsilon,
                        n_iter=sinkhorn_n_iter,
                    )
                loss_sinkhorn = loss_sinkhorn / local_n_groups

            # ── 7b-1b. kNN barycentric alignment (set-level, replaces zmatch) ──
            loss_knn_bary = torch.zeros((), device=device)
            if lambda_knn_bary > 0:
                for g in range(local_n_groups):
                    z_gen_g = z_gen[g * n_gen:(g + 1) * n_gen]
                    z_pos_g = z_data_train[all_pos_idx[g]]
                    loss_knn_bary = loss_knn_bary + knn_barycentric_alignment_loss(
                        z_gen_g, z_pos_g,
                        k=min(knn_bary_k, z_pos_g.shape[0]),
                        temperature=knn_bary_temp,
                    )
                loss_knn_bary = loss_knn_bary / local_n_groups

            # ── 7b-2. Z-space drifting (full V⁺ - V⁻ in z-space) ──
            loss_zdrift = torch.zeros((), device=device)
            if lambda_zdrift > 0:
                z_drift_temps = zdrift_temperatures if zdrift_temperatures is not None else drift_temperatures
                for g in range(local_n_groups):
                    z_gen_g = z_gen[g * n_gen:(g + 1) * n_gen]   # [N_gen, latent_dim]
                    z_pos_g = z_data_train[all_pos_idx[g]]       # [n_pos, latent_dim]
                    # Unconditional negatives in z-space
                    unc_idx_z = torch.randint(0, N_train, (n_unc,), device=device)
                    z_unc_g = z_data_train[unc_idx_z]            # [n_unc, latent_dim]
                    w_z = max(
                        0.0,
                        (alpha_val - 1.0) * max(n_gen - 1, 1) / max(n_unc, 1),
                    )
                    loss_zdrift = loss_zdrift + multi_temp_drift_loss(
                        z_gen_g, z_pos_g,
                        temperatures=z_drift_temps,
                        phi_unc=z_unc_g, cfg_w=w_z,
                        normalize_drift=drift_normalize,
                        normalize_distances=drift_normalize_dist,
                        fixed_lambdas=fixed_zdrift_lambdas,
                        norm_mode=drift_norm_mode,
                        knn_restrict_k=drift_knn_k,
                        attraction_scale=drift_attraction_scale,
                        repulsion_scale=drift_repulsion_scale,
                    )
                loss_zdrift = loss_zdrift / local_n_groups

            # ── 7c. Z-space contrastive loss (pull toward pos, push from neg) ──
            loss_zcontrast = torch.zeros((), device=device)
            if lambda_zcontrast > 0:
                for g in range(local_n_groups):
                    z_gen_g = z_gen[g * n_gen:(g + 1) * n_gen]  # [n_gen, D]
                    z_pos_g = z_data_train[all_pos_idx[g]]       # [n_pos, D]
                    # Random negatives (from ALL data, most are condition-mismatched)
                    neg_idx = torch.randint(0, N_train, (n_pos,), device=device)
                    z_neg_g = z_data_train[neg_idx]              # [n_pos, D]
                    # Nearest positive distance per z_gen
                    d_pos = torch.cdist(z_gen_g, z_pos_g).min(dim=1).values  # [n_gen]
                    # Nearest negative distance per z_gen
                    d_neg = torch.cdist(z_gen_g, z_neg_g).min(dim=1).values  # [n_gen]
                    # Triplet-like loss: d_pos should be < d_neg by margin
                    loss_zcontrast = loss_zcontrast + F.relu(d_pos - d_neg + zcontrast_margin).mean()
                loss_zcontrast = loss_zcontrast / local_n_groups

            # ── 7d. Decoupled φ-oracle drift (φ weights, z gradient) ──
            loss_decoupled_drift = torch.zeros((), device=device)
            if lambda_decoupled_drift > 0 and phi_model is not None:
                dc_temps = decoupled_drift_temps if decoupled_drift_temps is not None else drift_temperatures
                for g in range(local_n_groups):
                    z_gen_g = z_gen[g * n_gen:(g + 1) * n_gen]
                    z_pos_g = z_data_train[all_pos_idx[g]]
                    if use_multiscale_drift and phi_data_scales is not None:
                        # Multi-scale: each scale uses its matched temperature
                        z_gen_g_n = (z_gen_g - z_mean) / z_std
                        with torch.no_grad():
                            phi_gen_ms = phi_model.extract_features_multiscale(z_gen_g_n)
                        phi_pos_ms = [pds[all_pos_idx[g]] for pds in phi_data_scales]
                        loss_decoupled_drift = loss_decoupled_drift + multiscale_decoupled_drift_loss(
                            z_gen_g, z_pos_g, phi_gen_ms, phi_pos_ms,
                            scale_temperatures=multiscale_temps,
                            normalize_distances=drift_normalize_dist,
                            norm_mode=drift_norm_mode,
                            knn_restrict_k=drift_knn_k,
                        )
                    else:
                        # Single-scale path (backward compat)
                        phi_gen_g = phi_gen_all[g * n_gen:(g + 1) * n_gen] if phi_gen_all is not None else phi_model.extract_features(z_gen_g)
                        phi_pos_g = phi_data_all[all_pos_idx[g]] if phi_data_all is not None else phi_model.extract_features(z_pos_g)
                        loss_decoupled_drift = loss_decoupled_drift + decoupled_phi_drift_loss(
                            z_gen_g, z_pos_g, phi_gen_g, phi_pos_g,
                            temperatures=dc_temps,
                            normalize_distances=drift_normalize_dist,
                            norm_mode=drift_norm_mode,
                            knn_restrict_k=drift_knn_k,
                        )
                loss_decoupled_drift = loss_decoupled_drift / local_n_groups

            # ── 7e. Decoder-space auxiliary drift ──
            loss_dec_drift = torch.zeros((), device=device)
            if lambda_dec_drift > 0 and decoder_phi_model is not None:
                # Pre-compute decoder features for ALL z_gen at once (batched)
                # instead of per-group to avoid 32× redundant small forwards
                _dec_mbs = int(cfg.get("feature_space", {}).get("micro_batch", 256))
                if dec_drift_multiscale and dec_data_layers is not None:
                    n_dec_layers = decoder_phi_model.num_layers
                    dec_gen_layer_parts = [[] for _ in range(n_dec_layers)]
                    for _i in range(0, z_gen.shape[0], _dec_mbs):
                        layer_outs = decoder_phi_model.extract_features_multi(z_gen[_i:_i + _dec_mbs])
                        for li, feat in enumerate(layer_outs):
                            dec_gen_layer_parts[li].append(feat)
                    dec_gen_all_layers = [torch.cat(ll, dim=0) for ll in dec_gen_layer_parts]
                    for g in range(local_n_groups):
                        for li in range(n_dec_layers):
                            tau_li = dec_drift_layer_temps[li]
                            dec_feat_gen_li = dec_gen_all_layers[li][g * n_gen:(g + 1) * n_gen]
                            dec_feat_pos_li = dec_data_layers[li][all_pos_idx[g]]
                            fixed_lam_li = fixed_dec_drift_lambdas.get(li) if fixed_dec_drift_lambdas else None
                            loss_dec_drift = loss_dec_drift + multi_temp_drift_loss(
                                dec_feat_gen_li, dec_feat_pos_li,
                                temperatures=[tau_li],
                                normalize_drift=drift_normalize,
                                normalize_distances=drift_normalize_dist,
                                fixed_lambdas=fixed_lam_li,
                                norm_mode=drift_norm_mode,
                                attraction_scale=drift_attraction_scale,
                                repulsion_scale=drift_repulsion_scale,
                            )
                    loss_dec_drift = loss_dec_drift / (local_n_groups * n_dec_layers)
                else:
                    # Single-scale: batch forward all z_gen, then slice per group
                    dec_gen_parts = []
                    for _i in range(0, z_gen.shape[0], _dec_mbs):
                        dec_gen_parts.append(decoder_phi_model.extract_features(z_gen[_i:_i + _dec_mbs]))
                    dec_feat_gen_all = torch.cat(dec_gen_parts, dim=0)
                    for g in range(local_n_groups):
                        dec_feat_gen_g = dec_feat_gen_all[g * n_gen:(g + 1) * n_gen]
                        if dec_data_single is not None:
                            dec_feat_pos = dec_data_single[all_pos_idx[g]]
                        else:
                            z_pos_g = z_data_train[all_pos_idx[g]]
                            with torch.no_grad():
                                dec_feat_pos = decoder_phi_model.extract_features(z_pos_g)
                        loss_dec_drift = loss_dec_drift + multi_temp_drift_loss(
                            dec_feat_gen_g, dec_feat_pos,
                            temperatures=dec_drift_temps,
                            normalize_drift=drift_normalize,
                            normalize_distances=drift_normalize_dist,
                            fixed_lambdas=fixed_dec_drift_single_lambdas,
                            norm_mode=drift_norm_mode,
                            attraction_scale=drift_attraction_scale,
                            repulsion_scale=drift_repulsion_scale,
                        )
                    loss_dec_drift = loss_dec_drift / local_n_groups

            # ── 8. Auxiliary losses ──
            loss_moment = torch.zeros((), device=device)
            if lambda_moment > 0:
                z_data_sub = z_data_train[torch.randint(0, N_train, (local_batch,), device=device)]
                loss_moment = (
                    (z_gen.mean(0) - z_data_sub.mean(0)).pow(2).sum() +
                    (z_gen.std(0) - z_data_sub.std(0)).pow(2).sum()
                )

            loss_zdiv = torch.zeros((), device=device)
            if lambda_zdiv > 0:
                loss_zdiv = z_space_repulsion_loss(z_gen, margin=zdiv_margin, top_k=zdiv_topk)

            loss_phidiv = torch.zeros((), device=device)
            if lambda_phidiv > 0:
                loss_phidiv = phi_space_repulsion_loss(phi_gen_all, margin=phidiv_margin, top_k=phidiv_topk)

            loss_znorm = torch.zeros((), device=device)
            if lambda_znorm > 0:
                gen_norms = z_gen.norm(dim=-1)
                loss_znorm = (gen_norms - target_z_norm_mean).pow(2).mean()

            # ── 8b. Correlation / covariance structure losses ──
            loss_corr = torch.zeros((), device=device)
            if lambda_corr > 0:
                z_data_sub = z_data_train[torch.randint(0, N_train, (min(local_batch * 4, N_train),), device=device)]
                loss_corr = correlation_structure_loss(
                    z_gen, z_data_sub, subspace_dims=corr_subspace_dims)

            loss_cov = torch.zeros((), device=device)
            if lambda_cov > 0:
                z_data_sub_cov = z_data_train[torch.randint(0, N_train, (min(local_batch * 4, N_train),), device=device)]
                loss_cov = covariance_matching_loss(z_gen, z_data_sub_cov)

            # ── 9. Total loss ──
            _r_sink = _ramp_scale("sinkhorn", epoch)
            _r_knn  = _ramp_scale("knn_bary", epoch)
            _r_corr = _ramp_scale("corr", epoch)
            _r_cov  = _ramp_scale("cov", epoch)
            loss = (lambda_drift * loss_drift
                    + lambda_drift_struct * loss_drift_struct
                    + lambda_moment * loss_moment
                    + lambda_zdiv * loss_zdiv
                    + lambda_phidiv * loss_phidiv
                    + lambda_znorm * loss_znorm
                    + lambda_prop * loss_prop
                    + lambda_zmatch * loss_zmatch
                    + lambda_zcontrast * loss_zcontrast
                    + lambda_zdrift * loss_zdrift
                    + lambda_decoupled_drift * loss_decoupled_drift
                    + lambda_sinkhorn * _r_sink * loss_sinkhorn
                    + lambda_knn_bary * _r_knn * loss_knn_bary
                    + lambda_corr * _r_corr * loss_corr
                    + lambda_cov * _r_cov * loss_cov
                    + lambda_dec_drift * loss_dec_drift)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
            optimizer.step()

            # EMA update (use raw model, not DDP wrapper)
            with torch.no_grad():
                for p_ema, p_gen in zip(ema_generator.parameters(), generator_raw.parameters()):
                    p_ema.data.mul_(ema_decay).add_(p_gen.data, alpha=1 - ema_decay)

            running["total"] += loss.item()
            running["drift"] += loss_drift.item()
            running["drift_struct"] += loss_drift_struct.item()
            running["moment"] += loss_moment.item()
            running["zdiv"] += loss_zdiv.item()
            running["znorm"] += loss_znorm.item()
            running["prop"] += loss_prop.item()
            running["zmatch"] += loss_zmatch.item()
            running["zcontrast"] += loss_zcontrast.item()
            running["zdrift"] += loss_zdrift.item()
            running["phidiv"] += loss_phidiv.item()
            running["dcdrift"] += loss_decoupled_drift.item()
            running["sinkhorn"] += loss_sinkhorn.item()
            running["knn_bary"] += loss_knn_bary.item()
            running["corr"] += loss_corr.item()
            running["cov"] += loss_cov.item()
            running["dec_drift"] += loss_dec_drift.item()
            n_steps += 1

        scheduler.step()
        dt = time.time() - t0
        avg = lambda x: running[x] / max(n_steps, 1)

        # Build compact loss line: only show active losses
        loss_parts = [f"loss={avg('total'):.4f}"]
        if lambda_drift > 0: loss_parts.append(f"drift={avg('drift'):.4f}")
        if lambda_drift_struct > 0: loss_parts.append(f"dstruct={avg('drift_struct'):.4f}")
        if lambda_prop > 0: loss_parts.append(f"prop={avg('prop'):.4f}")
        if lambda_zmatch > 0: loss_parts.append(f"zmatch={avg('zmatch'):.4f}")
        if lambda_zdrift > 0: loss_parts.append(f"zdrift={avg('zdrift'):.4f}")
        if lambda_decoupled_drift > 0: loss_parts.append(f"dcdrift={avg('dcdrift'):.4f}")
        if lambda_sinkhorn > 0: loss_parts.append(f"sink={avg('sinkhorn'):.4f}" + (f"×{_r_sink:.2f}" if _r_sink < 1 else ""))
        if lambda_knn_bary > 0: loss_parts.append(f"knnb={avg('knn_bary'):.4f}" + (f"×{_r_knn:.2f}" if _r_knn < 1 else ""))
        if lambda_corr > 0: loss_parts.append(f"corr={avg('corr'):.4f}" + (f"×{_r_corr:.2f}" if _r_corr < 1 else ""))
        if lambda_cov > 0: loss_parts.append(f"cov={avg('cov'):.4f}" + (f"×{_r_cov:.2f}" if _r_cov < 1 else ""))
        if lambda_dec_drift > 0: loss_parts.append(f"decdrift={avg('dec_drift'):.4f}")
        if lambda_zcontrast > 0: loss_parts.append(f"zcon={avg('zcontrast'):.4f}")
        if lambda_moment > 0: loss_parts.append(f"moment={avg('moment'):.4f}")
        if lambda_zdiv > 0: loss_parts.append(f"zdiv={avg('zdiv'):.4f}")
        if lambda_phidiv > 0: loss_parts.append(f"phidiv={avg('phidiv'):.4f}")
        if lambda_znorm > 0: loss_parts.append(f"znorm={avg('znorm'):.4f}")
        line = f"[epoch {epoch:3d}] lr={current_lr:.2e} {' '.join(loss_parts)} ({dt:.0f}s)"

        epoch_metrics = {
            "epoch": epoch,
            "loss": avg("total"),
            "drift": avg("drift"),
            "drift_struct": avg("drift_struct"),
            "prop": avg("prop"),
        }

        # ── Evaluation ──
        is_unconditional = (alpha_min == alpha_max)
        if ddp:
            dist.barrier()
        if is_main and (epoch % eval_every == 0 or epoch == 1):
          if is_unconditional:
            # ── Unconditional evaluation: distribution matching ──
            uncond_eval = evaluate_unconditional(
                ema_generator, vae, train_smiles,
                num_samples=eval_samples, batch_size=eval_batch,
                device=device,
                ref_smiles_list=list(train_smiles_list),
            )
            cond_v = uncond_eval.get('validity', 0)
            cond_u = uncond_eval.get('uniqueness', 0)
            cond_n = uncond_eval.get('novelty', 0)
            cond_intdiv = uncond_eval.get('int_div', 0)
            qed_m = uncond_eval.get('qed_mean', 0)
            qed_s = uncond_eval.get('qed_std', 0)
            mw_m = uncond_eval.get('mw_mean', 0)
            mw_s = uncond_eval.get('mw_std', 0)
            logp_m = uncond_eval.get('logp_mean', 0)
            logp_s = uncond_eval.get('logp_std', 0)
            sa_m = uncond_eval.get('sa_mean', 0)
            sa_s = uncond_eval.get('sa_std', 0)
            fcd_v = uncond_eval.get('fcd', -1)
            snn_v = uncond_eval.get('snn', 0)
            frag_v = uncond_eval.get('frag', 0)
            scaf_v = uncond_eval.get('scaf', 0)
            kl_qed = uncond_eval.get('kl_qed', -1)
            kl_mw = uncond_eval.get('kl_mw', -1)
            kl_logp = uncond_eval.get('kl_logp', -1)
            filt_v = uncond_eval.get('filters', 0)
            line += (f" | V={cond_v*100:.0f}% U={cond_u*100:.1f}%"
                     f" N={cond_n*100:.1f}%"
                     f" IntDiv={cond_intdiv:.3f}")
            line += (f"\n  QED={qed_m:.3f}±{qed_s:.3f}"
                     f" MW={mw_m:.1f}±{mw_s:.1f}"
                     f" LogP={logp_m:.2f}±{logp_s:.2f}"
                     f" SA={sa_m:.2f}±{sa_s:.2f}")
            line += (f"\n  FCD={fcd_v:.3f}"
                     f" SNN={snn_v:.3f}"
                     f" Frag={frag_v:.3f}"
                     f" Scaf={scaf_v:.3f}"
                     f" Filt={filt_v:.3f}")
            if kl_qed >= 0:
                line += (f"\n  KL: QED={kl_qed:.3f}"
                         f" MW={kl_mw:.3f}"
                         f" LogP={kl_logp:.3f}")
            cond_eval = uncond_eval
            cond_eval['spearman_rho'] = 0  # not applicable
            epoch_metrics.update({f"cond_{k}": v for k, v in cond_eval.items()
                                  if not isinstance(v, dict)})
            gate_result = quality_gate.evaluate(cond_eval)
            score = gate_result.gated_score
            # Fallback: when Gate never passes (score=0), use VUN so best.pt still improves
            if score == 0:
                vun = cond_eval.get('validity', 0) * cond_eval.get('uniqueness', 0) * cond_eval.get('novelty', 0)
                score = vun * 1e-6  # tiny but >0 so later epochs can beat epoch 1
            line += f"\n  {gate_result.summary()}"
            if score > best_score:
                best_score = score
                best_epoch = epoch
                ckpt = {
                    "model_state": ema_generator.state_dict(),
                    "prop_head_state": prop_head.state_dict() if prop_head is not None else None,
                    "cfg": cfg,
                    "epoch": epoch,
                    "best_score": best_score,
                    "prop_mean": prop_mean,
                    "prop_std": prop_std,
                    "prop_names": sel_prop_names,
                }
                torch.save(ckpt, out_dir / "best.pt")
                line += " ★"
          else:
            # ── Conditional evaluation ──
            eval_alpha = float(cfg.get("selection", {}).get("eval_alpha", 3.0))
            eval_two_pass = bool(cfg.get("selection", {}).get("eval_two_pass_cfg", False))  # 1-NFE default
            cond_eval = evaluate_conditional(
                ema_generator, vae, train_smiles,
                prop_mean=prop_mean[0, 0],
                prop_std=prop_std[0, 0],
                prop_name=sel_prop_names[0],
                num_samples=eval_samples, batch_size=eval_batch,
                device=device, alpha=eval_alpha,
                target_values=[-1.5, -0.5, 0.5, 1.5],  # normalized
                bin_edges=bin_edges_raw,
                bin_centers_raw=bin_centers_raw,
                two_pass_cfg=eval_two_pass,
                ref_smiles_list=list(train_smiles_list),
                cond_dim=cond_dim,
            )
            cond_v = cond_eval.get('validity', 0)
            cond_u = cond_eval.get('uniqueness', 0)
            cond_n = cond_eval.get('novelty', 0)
            cond_rho = cond_eval.get('spearman_rho', 0)
            cond_mae = cond_eval.get('mae', 0)
            cond_intdiv = cond_eval.get('int_div', 0)
            cond_slope = cond_eval.get('slope', 0)
            cond_tail = cond_eval.get('mae_tail_avg', -1)
            line += (f" | V={cond_v*100:.0f}% U={cond_u*100:.1f}%"
                     f" N={cond_n*100:.1f}%"
                     f" ρ={cond_rho:.3f} MAE={cond_mae:.3f}"
                     f" slope={cond_slope:.3f}"
                     + (f" tailMAE={cond_tail:.3f}" if cond_tail >= 0 else "")
                     + f" IntDiv={cond_intdiv:.3f}")
            # Per-bin property pull: target→actual
            bin_summary = cond_eval.get('bin_summary', '')
            if bin_summary:
                line += f"\n  [{sel_prop_names[0]}] {bin_summary}"
            epoch_metrics.update({f"cond_{k}": v for k, v in cond_eval.items()
                                  if not isinstance(v, dict)})
            # ── Multi-prop: evaluate remaining properties ──
            if cond_dim > 1 and len(sel_prop_names) > 1:
                for p_idx in range(1, len(sel_prop_names)):
                    extra_eval = evaluate_conditional(
                        ema_generator, vae, train_smiles,
                        prop_mean=prop_mean[0, p_idx],
                        prop_std=prop_std[0, p_idx],
                        prop_name=sel_prop_names[p_idx],
                        num_samples=eval_samples, batch_size=eval_batch,
                        device=device, alpha=eval_alpha,
                        target_values=[-1.5, -0.5, 0.5, 1.5],
                        two_pass_cfg=eval_two_pass,
                        cond_dim=cond_dim,
                        prop_idx=p_idx,
                    )
                    extra_rho = extra_eval.get('spearman_rho', 0)
                    extra_mae = extra_eval.get('mae', 0)
                    extra_bin = extra_eval.get('bin_summary', '')
                    line += f"\n  [{sel_prop_names[p_idx]}] ρ={extra_rho:.3f} MAE={extra_mae:.3f}"
                    if extra_bin:
                        line += f" {extra_bin}"
                    epoch_metrics.update({f"cond_{sel_prop_names[p_idx]}_{k}": v
                                         for k, v in extra_eval.items()
                                         if not isinstance(v, dict)})
            # ── Quality gate ──
            gate_result = quality_gate.evaluate(cond_eval)
            score = gate_result.gated_score
            # Fallback: when Gate never passes (score=0), use VUN so best.pt still improves
            if score == 0:
                vun = cond_eval.get('validity', 0) * cond_eval.get('uniqueness', 0) * cond_eval.get('novelty', 0)
                score = vun * 1e-6  # tiny but >0 so later epochs can beat epoch 1
            # Log per-bin uniqueness and gate status
            min_bin_u = cond_eval.get('min_bin_uniqueness', -1)
            nn_sim = cond_eval.get('nn_sim_mean', -1)
            scaf_div = cond_eval.get('scaffold_diversity', -1)
            fcd_v = cond_eval.get('fcd', -1)
            extra_parts = []
            if min_bin_u >= 0:
                extra_parts.append(f"minBinU={min_bin_u:.2f}")
            if nn_sim >= 0:
                extra_parts.append(f"NN={nn_sim:.3f}")
            if scaf_div >= 0:
                extra_parts.append(f"ScafDiv={scaf_div:.3f}")
            if fcd_v >= 0:
                extra_parts.append(f"FCD={fcd_v:.2f}")
            if extra_parts:
                line += f"\n  {' '.join(extra_parts)}"
            line += f"\n  {gate_result.summary()}"
            if score > best_score:
                best_score = score
                best_epoch = epoch
                ckpt = {
                    "model_state": ema_generator.state_dict(),
                    "prop_head_state": prop_head.state_dict() if prop_head is not None else None,
                    "cfg": cfg,
                    "epoch": epoch,
                    "best_score": best_score,
                    "prop_mean": prop_mean,
                    "prop_std": prop_std,
                    "prop_names": sel_prop_names,
                }
                torch.save(ckpt, out_dir / "best.pt")
                line += " ★"

        # Save last.pt for resume (rank 0 only)
        if is_main:
            last_ckpt = {
                "generator_state": generator_raw.state_dict(),
                "ema_state": ema_generator.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "prop_head_state": prop_head.state_dict() if prop_head is not None else None,
                "epoch": epoch,
                "best_score": best_score,
                "best_epoch": best_epoch,
            }
            torch.save(last_ckpt, out_dir / "last.pt")

        if is_main:
            print(line, flush=True)
        history.append(epoch_metrics)

        # Barrier AFTER eval to prevent DDP deadlock: ranks 1..N-1 must wait
        # for rank 0's evaluate_conditional() before entering next epoch's
        # backward() which triggers allreduce.
        if ddp:
            dist.barrier()

    # ── Final evaluation with α sweep (rank 0 only) ──────────────────
    if ddp:
        dist.barrier()
    if not is_main:
        if ddp:
            dist.destroy_process_group()
        return
    print("\n[final] Loading best checkpoint...")
    best_ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    ema_generator.load_state_dict(best_ckpt["model_state"])

    final_n = int(cfg.get("eval", {}).get("num_generated_samples", 5000))
    alpha_sweep = cfg.get("eval", {}).get("alpha_sweep", [1.0, 1.5, 2.0, 3.0, 5.0])
    final_two_pass = bool(cfg.get("selection", {}).get("eval_two_pass_cfg", False))

    # Reference stats for all conditioned properties
    final_eval_t0 = time.time()
    final_results = {
        "reference": {},
        "generation_protocol": {
            "nfe": 2 if final_two_pass else 1,
            "two_pass_cfg": final_two_pass,
            "alpha_sweep": alpha_sweep,
            "num_generated_samples_per_alpha": final_n,
            "eval_batch_size": eval_batch,
        },
    }
    for p_idx in range(len(sel_prop_names)):
        ref_prop_mean = raw_props[:, p_idx].mean().item()
        ref_prop_std = raw_props[:, p_idx].std().item()
        final_results["reference"][f"{sel_prop_names[p_idx]}_mean"] = ref_prop_mean
        final_results["reference"][f"{sel_prop_names[p_idx]}_std"] = ref_prop_std

    for alpha in alpha_sweep:
        alpha_t0 = time.time()
        print(f"\n  α={alpha:.1f}:")

        # Unconditional
        uncond = evaluate_unconditional(
            ema_generator, vae, train_smiles,
            num_samples=final_n, batch_size=eval_batch, device=device,
        )
        print(f"    uncond: VUN={uncond['vun']:.3f} LogP={uncond.get('logp_mean', 0):.3f}")

        # Conditional with property-specific targets — evaluate ALL properties
        alpha_results = {"unconditional": uncond}

        for p_idx in range(len(sel_prop_names)):
            pn = sel_prop_names[p_idx].lower()
            pm_i = prop_mean[0, p_idx]
            ps_i = prop_std[0, p_idx]

            if p_idx == 0 and bin_centers_raw is not None and len(bin_centers_raw) > 1:
                target_raw = list(bin_centers_raw)
            else:
                # Use data-driven percentile targets for all properties
                q = raw_props[:, p_idx].cpu().numpy()
                target_raw = [float(np.percentile(q, p)) for p in [5, 15, 25, 35, 50, 65, 75, 85, 95]]
            target_norm = [(t - pm_i.item()) / ps_i.item() for t in target_raw]

            cond = evaluate_conditional(
                ema_generator, vae, train_smiles,
                prop_mean=pm_i,
                prop_std=ps_i,
                prop_name=sel_prop_names[p_idx],
                num_samples=final_n, batch_size=eval_batch,
                device=device, alpha=alpha,
                target_values=target_norm,
                bin_edges=bin_edges_raw if p_idx == 0 else None,
                bin_centers_raw=bin_centers_raw if p_idx == 0 else None,
                two_pass_cfg=final_two_pass,
                cond_dim=cond_dim,
                prop_idx=p_idx,
            )
            print(f"    [{sel_prop_names[p_idx]}] ρ={cond.get('spearman_rho', 0):.3f} "
                  f"MAE={cond.get('mae', 0):.3f} "
                  f"r={cond.get('pearson_r', 0):.3f} "
                  f"U={cond.get('uniqueness', 0)*100:.1f}%")

            if "per_target" in cond:
                for k, v in cond["per_target"].items():
                    print(f"      {k}: actual={v['actual_mean']:.3f}±{v['actual_std']:.3f}")

            alpha_results[f"conditional_{sel_prop_names[p_idx]}"] = {
                k: v for k, v in cond.items() if not isinstance(v, dict)
            }
            alpha_results[f"per_target_{sel_prop_names[p_idx]}"] = cond.get("per_target", {})

        # Backward-compatible keys: copy first property as default 'conditional'
        first_key = f"conditional_{sel_prop_names[0]}"
        if first_key in alpha_results:
            alpha_results["conditional"] = alpha_results[first_key]
            alpha_results["per_target"] = alpha_results.get(f"per_target_{sel_prop_names[0]}", {})

        final_results[f"alpha={alpha}"] = alpha_results
        alpha_results["runtime"] = {
            "eval_seconds": time.time() - alpha_t0,
            "nfe": 2 if final_two_pass else 1,
            "two_pass_cfg": final_two_pass,
        }

    final_results["runtime"] = {
        "final_eval_seconds": time.time() - final_eval_t0,
        "nfe": 2 if final_two_pass else 1,
        "two_pass_cfg": final_two_pass,
    }
    save_json(final_results, out_dir / "final_metrics.json")
    save_json(history, out_dir / "train_history.json")
    print("\n[done]")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
