"""
Latent-MAE: 隐空间掩码自编码器 (Feature Extractor φ).

在 VAE 的连续隐空间上进行自监督预训练。
掩码掉隐向量的部分维度，由网络重建→学到隐空间的深层化学语义。

训练好后，仅保留 Encoder 部分作为 φ，冻结参数。

Architecture:
  Encoder: latent_dim → [mask] → MLP/Transformer → phi_dim (feature space)
  Decoder: phi_dim → MLP → latent_dim (reconstruction, 训练后丢弃)
  PropertyHead: phi_dim → n_props (可选属性微调, 训练后丢弃)

Contrastive Mode (optional):
  Soft Supervised Contrastive Loss — 用连续属性值定义正负样本软权重，
  使 φ-space 的度量结构直接反映属性相似性。
  不需要属性预测头; 属性标签仅用于定义样本间的正/负关系。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════
# Standalone loss function for Soft Supervised Contrastive
# ═══════════════════════════════════════════════════════

def soft_supcon_loss(
    phi: torch.Tensor,
    prop_values: torch.Tensor,
    temperature: float = 0.1,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Soft Supervised Contrastive Loss for continuous properties.

    使属性相近的样本在 φ-space 中靠近，属性差异大的样本远离。
    正负样本不是离散标签，而是基于属性距离的连续软权重。

    Args:
        phi: [B, D] L2-normalized features on unit sphere
        prop_values: [B] single property values (already normalized)
        temperature: contrastive temperature τ (lower → sharper)
        sigma: bandwidth for property similarity kernel
            (controls how quickly similarity decays with property distance)

    Returns:
        scalar loss value
    """
    B = phi.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=phi.device, requires_grad=True)

    # Pairwise cosine similarity (normalize internally to be safe even if
    # phi is not L2-normalized — gradient still flows through phi)
    phi_norm = F.normalize(phi, dim=-1)
    sim = phi_norm @ phi_norm.T / temperature  # [B, B]

    # Pairwise property distance → soft positive weights
    prop_diff = prop_values.unsqueeze(0) - prop_values.unsqueeze(1)  # [B, B]
    soft_weights = torch.exp(-prop_diff.pow(2) / (2 * sigma ** 2))  # [B, B]

    # Mask out self-similarity
    mask_self = ~torch.eye(B, dtype=torch.bool, device=phi.device)
    soft_weights = soft_weights * mask_self.float()

    # Normalize weights per row → probability distribution over positive/negative
    weight_sum = soft_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
    soft_weights = soft_weights / weight_sum  # [B, B], rows sum to 1

    # Log-softmax over similarities (exclude self)
    sim_masked = sim.masked_fill(~mask_self, float('-inf'))
    log_prob = F.log_softmax(sim_masked, dim=1)  # [B, B]

    # Weighted contrastive loss: -sum_j w_ij * log P(j|i)
    # Note: diagonal has soft_weights=0 and log_prob=-inf → 0*(-inf)=NaN
    # Fix: zero out diagonal in the product before summing
    product = soft_weights * log_prob
    product = product.masked_fill(~mask_self, 0.0)
    loss = -product.sum(dim=1).mean()

    return loss


def vicreg_variance_loss(phi: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Variance loss: keep std of each dimension above gamma (default 1)."""
    std = phi.std(dim=0)
    return F.relu(gamma - std).mean()


def vicreg_covariance_loss(phi: torch.Tensor) -> torch.Tensor:
    """Covariance loss: decorrelate dimensions (off-diagonal of cov → 0)."""
    B, D = phi.shape
    phi_c = phi - phi.mean(dim=0)
    cov = (phi_c.T @ phi_c) / max(B - 1, 1)  # [D, D]
    off_diag = cov.pow(2).sum() - cov.diag().pow(2).sum()
    return off_diag / D


def multi_prop_supcon_loss(
    phi: torch.Tensor,
    properties: torch.Tensor,
    temperature: float = 0.1,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Multi-property soft supervised contrastive loss.
    Uses ALL properties jointly (L2 distance in property space) instead of single property.

    Args:
        phi: [B, D] features
        properties: [B, K] multiple normalized property values
        temperature: contrastive temperature
        sigma: bandwidth for property similarity kernel
    """
    B = phi.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=phi.device, requires_grad=True)

    phi_norm = F.normalize(phi, dim=-1)
    sim = phi_norm @ phi_norm.T / temperature  # [B, B]

    # Multi-property L2 distance → soft positive weights
    prop_dist_sq = torch.cdist(properties, properties, p=2).pow(2)  # [B, B]
    soft_weights = torch.exp(-prop_dist_sq / (2 * sigma ** 2))  # [B, B]

    mask_self = ~torch.eye(B, dtype=torch.bool, device=phi.device)
    soft_weights = soft_weights * mask_self.float()
    weight_sum = soft_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
    soft_weights = soft_weights / weight_sum

    sim_masked = sim.masked_fill(~mask_self, float('-inf'))
    log_prob = F.log_softmax(sim_masked, dim=1)

    product = soft_weights * log_prob
    product = product.masked_fill(~mask_self, 0.0)
    return -product.sum(dim=1).mean()


class _ResidualEncoder(nn.Module):
    """Encoder with pre-norm residual blocks, wrapped as nn.Module so self.encoder(x) works."""
    def __init__(self, input_proj, blocks, norms, output_proj):
        super().__init__()
        self.input_proj = input_proj
        self.blocks = blocks
        self.norms = norms
        self.output_proj = output_proj

    def forward(self, x):
        h = self.input_proj(x)
        for block, norm in zip(self.blocks, self.norms):
            h = h + block(norm(h))  # pre-norm residual
        return self.output_proj(h)


class LatentMAE(nn.Module):
    """
    Masked Autoencoder for latent space feature extraction.

    Input:  z ∈ R^{latent_dim} (VAE latent vectors)
    Output: φ(z) ∈ R^{phi_dim} (semantic features on unit sphere)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        phi_dim: int = 128,           # feature space dimension
        hidden_dim: int = 256,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 2,
        mask_ratio: float = 0.5,      # fraction of dimensions to mask
        num_properties: int = 0,       # 0 = no property prediction head
        dropout: float = 0.1,
        normalize_output: bool = True, # L2-normalize φ output (unit sphere)
        prop_grad_through_encoder: bool = False,  # if True, property loss backprops through encoder
        # ── Contrastive learning config ──
        contrastive_mode: str = "none",  # "none" | "supcon"
        contrastive_property_idx: int = 0,  # which property to use for SupCon
        contrastive_temperature: float = 0.1,  # τ for contrastive loss
        contrastive_sigma: float = 1.0,  # bandwidth for soft positive weight kernel
        # ── Multi-scale config ──
        multi_scale_dims: list[int] | None = None,  # e.g. [8, 16, 32] for coarse→fine
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.phi_dim = phi_dim
        self.mask_ratio = mask_ratio
        self.normalize_output = normalize_output
        self.prop_grad_through_encoder = prop_grad_through_encoder
        self.multi_scale_dims = multi_scale_dims

        # Contrastive settings
        self.contrastive_mode = contrastive_mode
        self.contrastive_property_idx = contrastive_property_idx
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_sigma = contrastive_sigma

        # ── Encoder: masked z → φ ──
        # Use residual blocks for deeper encoders (>=6 layers)
        self._encoder_residual = (num_encoder_layers >= 6)
        if self._encoder_residual:
            # Input projection: latent_dim → hidden_dim
            self.enc_input = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
            # Residual blocks: each = LN → Linear → GELU → Dropout → Linear → Dropout + skip
            self.enc_blocks = nn.ModuleList()
            self.enc_norms = nn.ModuleList()
            for _ in range(num_encoder_layers - 1):
                self.enc_blocks.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout),
                ))
                self.enc_norms.append(nn.LayerNorm(hidden_dim))
            if multi_scale_dims is not None:
                # Multi-scale: backbone outputs hidden_dim, heads project to each scale
                self.enc_output = nn.LayerNorm(hidden_dim)
                self.encoder = _ResidualEncoder(
                    self.enc_input, self.enc_blocks, self.enc_norms, self.enc_output)
                self.scale_heads = nn.ModuleList([
                    nn.Linear(hidden_dim, d) for d in multi_scale_dims
                ])
            else:
                self.enc_output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, phi_dim))
                self.encoder = _ResidualEncoder(
                    self.enc_input, self.enc_blocks, self.enc_norms, self.enc_output)
        else:
            enc_layers = [nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            for _ in range(num_encoder_layers - 1):
                enc_layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            enc_layers.append(nn.Linear(hidden_dim, phi_dim))
            self.encoder = nn.Sequential(*enc_layers)

        # ── Decoder: φ → reconstructed z ──
        dec_input_dim = sum(multi_scale_dims) if multi_scale_dims is not None else phi_dim
        dec_layers = [nn.Linear(dec_input_dim, hidden_dim), nn.GELU()]
        for _ in range(num_decoder_layers - 1):
            dec_layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        dec_layers.append(nn.Linear(hidden_dim, latent_dim))
        self.decoder = nn.Sequential(*dec_layers)

        # ── Property prediction head (optional) ──
        self.property_head = None
        if num_properties > 0:
            self.property_head = nn.Sequential(
                nn.Linear(phi_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_properties),
            )

        # Learnable mask token (replaces masked dimensions)
        self.mask_token = nn.Parameter(torch.zeros(latent_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def _apply_mask(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random dimension-level masking to z.

        Returns:
            z_masked: [B, latent_dim] with masked dims replaced by mask_token
            mask: [B, latent_dim] bool, True where masked
        """
        B, D = z.shape
        num_mask = int(D * self.mask_ratio)

        # Random mask per sample
        noise = torch.rand(B, D, device=z.device)
        ids_sorted = noise.argsort(dim=1)
        mask = torch.zeros(B, D, dtype=torch.bool, device=z.device)
        mask.scatter_(1, ids_sorted[:, :num_mask], True)

        # Replace masked dims with learnable mask token
        z_masked = z.clone()
        z_masked[mask] = self.mask_token.expand(B, -1)[mask]

        return z_masked, mask

    def extract_features(self, z: torch.Tensor) -> torch.Tensor:
        """
        Extract features φ(z) WITHOUT masking (used at inference / for drift).
        Returns finest-scale φ [B, phi_dim] for backward compatibility.
        """
        if self.multi_scale_dims is not None:
            h = self.encoder(z)  # [B, hidden_dim]
            phi = self.scale_heads[-1](h)  # finest (largest dim) scale
        else:
            phi = self.encoder(z)
        if self.normalize_output:
            phi = F.normalize(phi, dim=-1)
        return phi

    def extract_features_multiscale(self, z: torch.Tensor) -> list[torch.Tensor]:
        """
        Extract multi-scale features [coarsest → finest].
        Returns list of [B, d_i] tensors for each scale in self.multi_scale_dims.
        """
        assert self.multi_scale_dims is not None, "multi_scale_dims not configured"
        h = self.encoder(z)  # [B, hidden_dim]
        features = []
        for head in self.scale_heads:
            f = head(h)
            if self.normalize_output:
                f = F.normalize(f, dim=-1)
            features.append(f)
        return features

    def forward(
        self,
        z: torch.Tensor,
        properties: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Training forward pass:
          1. Mask z → z_masked
          2. Encode z_masked → φ
          3. Decode φ → z_recon
          4. Reconstruction loss on masked dims only
          5. (Optional) property prediction loss
          6. (Optional) soft supervised contrastive loss

        Returns dict with losses and intermediate tensors.
        """
        z_masked, mask = self._apply_mask(z)

        # Encode (with masking)
        if self.multi_scale_dims is not None:
            h = self.encoder(z_masked)  # [B, hidden_dim]
            phis_raw = [head(h) for head in self.scale_heads]  # list of [B, d_i]
            if self.normalize_output:
                phis_norm = [F.normalize(p, dim=-1) for p in phis_raw]
            else:
                phis_norm = phis_raw
            phi_normalized = phis_norm[-1]  # finest scale for property head
            phi_concat = torch.cat(phis_raw, dim=-1)  # for decoder
            z_recon = self.decoder(phi_concat)
        else:
            phi = self.encoder(z_masked)
            if self.normalize_output:
                phi_normalized = F.normalize(phi, dim=-1)
            else:
                phi_normalized = phi
            z_recon = self.decoder(phi)  # use un-normalized phi for decoder
            phis_norm = None

        # Reconstruction loss: only on masked dimensions
        recon_loss = F.mse_loss(z_recon[mask], z[mask])

        result = {
            "phi": phi_normalized,
            "z_recon": z_recon,
            "mask": mask,
            "recon_loss": recon_loss,
        }
        if phis_norm is not None:
            result["phis"] = phis_norm  # multi-scale features for per-scale losses

        # Property prediction (optional)
        if self.property_head is not None and properties is not None:
            # If prop_grad_through_encoder: gradient flows through encoder → prop head
            # helps encoder learn property-aware features
            # If not: detach → prop head acts as a probe only
            phi_for_prop = phi_normalized if self.prop_grad_through_encoder else phi_normalized.detach()
            prop_pred = self.property_head(phi_for_prop)
            prop_loss = F.mse_loss(prop_pred, properties)
            result["prop_loss"] = prop_loss
            result["prop_pred"] = prop_pred

        # Soft Supervised Contrastive Loss (optional)
        if self.contrastive_mode == "supcon" and properties is not None:
            # Extract single property for contrastive learning
            if properties.dim() == 2:
                prop_single = properties[:, self.contrastive_property_idx]
            else:
                prop_single = properties  # already 1D
            contrastive = soft_supcon_loss(
                phi_normalized,
                prop_single,
                temperature=self.contrastive_temperature,
                sigma=self.contrastive_sigma,
            )
            result["contrastive_loss"] = contrastive

        return result

    @torch.no_grad()
    def predict_properties(self, z: torch.Tensor) -> torch.Tensor | None:
        """
        Predict properties from z (no masking, inference mode).
        Returns: [B, num_properties] or None if no property head.
        """
        if self.property_head is None:
            return None
        phi = self.extract_features(z)
        return self.property_head(phi)
